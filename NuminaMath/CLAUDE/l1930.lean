import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1930_193009

/-- A quadratic function f(x) = x² + bx - 5 with axis of symmetry at x = 2 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 5

/-- The axis of symmetry of f is at x = 2 -/
def axis_of_symmetry (b : ℝ) : Prop := -b/(2*1) = 2

/-- The equation f(x) = 2x - 13 -/
def equation (b : ℝ) (x : ℝ) : Prop := f b x = 2*x - 13

theorem quadratic_equation_solution (b : ℝ) :
  axis_of_symmetry b →
  (∃ x y : ℝ, x = 2 ∧ y = 4 ∧ equation b x ∧ equation b y ∧
    ∀ z : ℝ, equation b z → (z = x ∨ z = y)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1930_193009


namespace NUMINAMATH_CALUDE_remainder_sum_l1930_193051

theorem remainder_sum (n : ℤ) : n % 15 = 7 → (n % 3 + n % 5 = 3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1930_193051


namespace NUMINAMATH_CALUDE_hamburgers_served_equals_three_l1930_193059

/-- The number of hamburgers made by the restaurant -/
def total_hamburgers : ℕ := 9

/-- The number of hamburgers left over -/
def leftover_hamburgers : ℕ := 6

/-- The number of hamburgers served -/
def served_hamburgers : ℕ := total_hamburgers - leftover_hamburgers

theorem hamburgers_served_equals_three : served_hamburgers = 3 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_served_equals_three_l1930_193059


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1930_193026

/-- Calculates the length of a bridge given train and crossing parameters -/
theorem bridge_length_calculation
  (train_length : ℝ)
  (initial_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (wind_resistance_factor : ℝ)
  (h_train_length : train_length = 750)
  (h_initial_speed : initial_speed_kmh = 120)
  (h_crossing_time : crossing_time = 45)
  (h_wind_resistance : wind_resistance_factor = 0.9)
  : ∃ (bridge_length : ℝ), bridge_length = 600 :=
by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l1930_193026


namespace NUMINAMATH_CALUDE_bobs_family_adults_l1930_193013

theorem bobs_family_adults (total_apples : ℕ) (num_children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) 
  (h1 : total_apples = 1200)
  (h2 : num_children = 45)
  (h3 : apples_per_child = 15)
  (h4 : apples_per_adult = 5) :
  (total_apples - num_children * apples_per_child) / apples_per_adult = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_bobs_family_adults_l1930_193013


namespace NUMINAMATH_CALUDE_candy_sharing_l1930_193032

theorem candy_sharing (bags : ℕ) (candies_per_bag : ℕ) (people : ℕ) :
  bags = 25 →
  candies_per_bag = 16 →
  people = 2 →
  (bags * candies_per_bag) / people = 200 := by
  sorry

end NUMINAMATH_CALUDE_candy_sharing_l1930_193032


namespace NUMINAMATH_CALUDE_parallel_line_x_coordinate_l1930_193040

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines when two points form a line segment parallel to the y-axis. -/
def parallelToYAxis (p q : Point) : Prop :=
  p.x = q.x

/-- The problem statement -/
theorem parallel_line_x_coordinate 
  (M N : Point)
  (h_parallel : parallelToYAxis M N)
  (h_M : M = ⟨3, -5⟩)
  (h_N : N = ⟨N.x, 2⟩) :
  N.x = 3 := by
  sorry

#check parallel_line_x_coordinate

end NUMINAMATH_CALUDE_parallel_line_x_coordinate_l1930_193040


namespace NUMINAMATH_CALUDE_books_per_shelf_l1930_193004

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) 
  (h1 : total_books = 504) (h2 : num_shelves = 9) :
  total_books / num_shelves = 56 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l1930_193004


namespace NUMINAMATH_CALUDE_incorrect_comparison_l1930_193047

theorem incorrect_comparison : ¬((-5.2 : ℚ) > -5.1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_comparison_l1930_193047


namespace NUMINAMATH_CALUDE_triangle_side_length_l1930_193010

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (2 * b = a + c) →  -- Arithmetic sequence
  (B = π / 6) →      -- Angle B = 30°
  (1 / 2 * a * c * Real.sin B = 3 / 2) →  -- Area of triangle
  -- Conclusion
  b = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1930_193010


namespace NUMINAMATH_CALUDE_product_of_differences_l1930_193043

theorem product_of_differences (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) :
  (a - 1) * (b - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l1930_193043


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1930_193094

theorem consecutive_integers_average (c d : ℤ) : 
  (c > 0) →
  (d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1930_193094


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_l1930_193084

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + y = 2 * x * y) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + y' = 2 * x' * y' → x' + 4 * y' ≥ 9/2) ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 2 * x₀ * y₀ ∧ x₀ + 4 * y₀ = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_l1930_193084


namespace NUMINAMATH_CALUDE_nonagon_triangle_probability_l1930_193044

/-- The number of vertices in a regular nonagon -/
def nonagon_vertices : ℕ := 9

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of ways to choose 3 vertices from 9 vertices -/
def total_triangles : ℕ := Nat.choose nonagon_vertices triangle_vertices

/-- The number of triangles with at least one side being a side of the nonagon -/
def favorable_triangles : ℕ := 54

/-- The probability of forming a triangle with at least one side being a side of the nonagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem nonagon_triangle_probability : probability = 9 / 14 := by sorry

end NUMINAMATH_CALUDE_nonagon_triangle_probability_l1930_193044


namespace NUMINAMATH_CALUDE_all_offers_count_l1930_193086

def stadium_capacity : ℕ := 4500

def hot_dog_interval : ℕ := 90
def soda_interval : ℕ := 45
def popcorn_interval : ℕ := 60
def ice_cream_interval : ℕ := 45

def fans_with_all_offers : ℕ := stadium_capacity / (Nat.lcm hot_dog_interval (Nat.lcm soda_interval popcorn_interval))

theorem all_offers_count :
  fans_with_all_offers = 25 :=
sorry

end NUMINAMATH_CALUDE_all_offers_count_l1930_193086


namespace NUMINAMATH_CALUDE_polygon_sides_l1930_193015

theorem polygon_sides (n : ℕ) (h : n ≥ 3) : 
  (n - 2) * 180 + 360 = 1260 → n = 7 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l1930_193015


namespace NUMINAMATH_CALUDE_ashley_state_quarters_amount_l1930_193020

/-- The amount Ashley receives for her state quarters -/
def ashley_amount (num_quarters : ℕ) (face_value : ℚ) (percentage : ℕ) : ℚ :=
  (num_quarters : ℚ) * face_value * (percentage : ℚ) / 100

/-- Theorem stating the amount Ashley receives for her state quarters -/
theorem ashley_state_quarters_amount :
  ashley_amount 6 0.25 1500 = 22.50 := by
  sorry

end NUMINAMATH_CALUDE_ashley_state_quarters_amount_l1930_193020


namespace NUMINAMATH_CALUDE_min_difference_f_g_l1930_193006

noncomputable def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_difference_f_g :
  ∃ (min_val : ℝ), min_val = 1/2 + 1/2 * Real.log 2 ∧
  ∀ (x : ℝ), x > 0 → |f x - g x| ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_difference_f_g_l1930_193006


namespace NUMINAMATH_CALUDE_g_monotonic_intervals_exactly_two_tangent_points_l1930_193055

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else -x^2 + 2*x - 1/2

-- Define g(x) = x * f(x)
noncomputable def g (x : ℝ) : ℝ := x * f x

-- Theorem for monotonic intervals of g(x)
theorem g_monotonic_intervals :
  (∀ x y, x < y ∧ y < -1 → g y < g x) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y ≤ 0 → g x < g y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < (4 - Real.sqrt 10) / 6 → g y < g x) ∧
  (∀ x y, (4 - Real.sqrt 10) / 6 < x ∧ x < y ∧ y < (4 + Real.sqrt 10) / 6 → g x < g y) ∧
  (∀ x y, (4 + Real.sqrt 10) / 6 < x ∧ x < y → g y < g x) :=
sorry

-- Theorem for existence of exactly two tangent points
theorem exactly_two_tangent_points :
  ∃! (x₁ x₂ : ℝ), x₁ < x₂ ∧
    ∃ (m b : ℝ), 
      (∀ x, f x ≤ m * x + b) ∧
      f x₁ = m * x₁ + b ∧
      f x₂ = m * x₂ + b :=
sorry

end NUMINAMATH_CALUDE_g_monotonic_intervals_exactly_two_tangent_points_l1930_193055


namespace NUMINAMATH_CALUDE_sum_equals_210_l1930_193073

theorem sum_equals_210 : 145 + 35 + 25 + 5 = 210 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_210_l1930_193073


namespace NUMINAMATH_CALUDE_distance_ratio_theorem_l1930_193012

/-- Represents a square pyramid -/
structure SquarePyramid where
  -- Base side length
  a : ℝ
  -- Height
  h : ℝ
  -- Assume positive dimensions
  a_pos : 0 < a
  h_pos : 0 < h

/-- A point inside the base square -/
structure BasePoint where
  x : ℝ
  y : ℝ
  -- Assume the point is inside the base square
  x_bound : 0 < x ∧ x < 1
  y_bound : 0 < y ∧ y < 1

/-- Sum of distances from a point to the triangular faces -/
noncomputable def sumDistancesToFaces (p : SquarePyramid) (e : BasePoint) : ℝ := sorry

/-- Sum of distances from a point to the base edges -/
noncomputable def sumDistancesToEdges (p : SquarePyramid) (e : BasePoint) : ℝ := sorry

/-- The main theorem -/
theorem distance_ratio_theorem (p : SquarePyramid) (e : BasePoint) :
  sumDistancesToFaces p e / sumDistancesToEdges p e = p.h / p.a := by sorry

end NUMINAMATH_CALUDE_distance_ratio_theorem_l1930_193012


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1930_193002

theorem imaginary_part_of_z (z : ℂ) (h : z = Complex.I * (2 - z)) : z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1930_193002


namespace NUMINAMATH_CALUDE_daves_weight_l1930_193093

/-- Proves Dave's weight given the conditions from the problem -/
theorem daves_weight (dave_weight : ℝ) (dave_bench : ℝ) (craig_bench : ℝ) (mark_bench : ℝ) :
  dave_bench = 3 * dave_weight →
  craig_bench = 0.2 * dave_bench →
  mark_bench = craig_bench - 50 →
  mark_bench = 55 →
  dave_weight = 175 := by
sorry

end NUMINAMATH_CALUDE_daves_weight_l1930_193093


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l1930_193076

/-- The number of chocolate bars in a box, given the cost per bar and the sales amount when all but 3 bars are sold. -/
def number_of_bars (cost_per_bar : ℕ) (sales_amount : ℕ) : ℕ :=
  (sales_amount + 3 * cost_per_bar) / cost_per_bar

theorem chocolate_bar_count : number_of_bars 3 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l1930_193076


namespace NUMINAMATH_CALUDE_mr_grey_polo_shirts_l1930_193000

/-- Represents the purchase of gifts by Mr. Grey -/
structure GiftPurchase where
  polo_shirt_price : ℕ
  necklace_price : ℕ
  computer_game_price : ℕ
  necklace_count : ℕ
  rebate : ℕ
  total_cost : ℕ

/-- Calculates the number of polo shirts bought given the gift purchase details -/
def calculate_polo_shirts (purchase : GiftPurchase) : ℕ :=
  (purchase.total_cost + purchase.rebate - purchase.necklace_price * purchase.necklace_count - purchase.computer_game_price) / purchase.polo_shirt_price

/-- Theorem stating that Mr. Grey bought 3 polo shirts -/
theorem mr_grey_polo_shirts :
  let purchase : GiftPurchase := {
    polo_shirt_price := 26,
    necklace_price := 83,
    computer_game_price := 90,
    necklace_count := 2,
    rebate := 12,
    total_cost := 322
  }
  calculate_polo_shirts purchase = 3 := by
  sorry

end NUMINAMATH_CALUDE_mr_grey_polo_shirts_l1930_193000


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1930_193075

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 256 ∧ 
  (∀ m : ℕ, (1019 + m) % 25 = 0 ∧ (1019 + m) % 17 = 0 → m ≥ n) ∧
  (1019 + n) % 25 = 0 ∧ (1019 + n) % 17 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1930_193075


namespace NUMINAMATH_CALUDE_right_triangle_cos_b_l1930_193028

theorem right_triangle_cos_b (A B C : ℝ) (h1 : A = 90) (h2 : Real.sin B = 3/5) :
  Real.cos B = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_b_l1930_193028


namespace NUMINAMATH_CALUDE_certain_number_divisor_l1930_193099

theorem certain_number_divisor (n : Nat) (h1 : n = 1020) : 
  ∃ x : Nat, x > 0 ∧ 
  (n - 12) % x = 0 ∧
  (n - 12) % 12 = 0 ∧ 
  (n - 12) % 24 = 0 ∧ 
  (n - 12) % 36 = 0 ∧ 
  (n - 12) % 48 = 0 ∧
  x = 7 ∧
  x ∉ Nat.divisors (Nat.lcm 12 (Nat.lcm 24 (Nat.lcm 36 48))) ∧
  ∀ y : Nat, y > x → (n - 12) % y ≠ 0 ∨ 
    y ∈ Nat.divisors (Nat.lcm 12 (Nat.lcm 24 (Nat.lcm 36 48))) :=
by sorry

end NUMINAMATH_CALUDE_certain_number_divisor_l1930_193099


namespace NUMINAMATH_CALUDE_correct_number_probability_l1930_193025

def first_three_digits : ℕ := 3

def last_four_digits : List ℕ := [0, 1, 1, 7]

def permutations_of_last_four : ℕ := 12

theorem correct_number_probability :
  (1 : ℚ) / (first_three_digits * permutations_of_last_four) = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_probability_l1930_193025


namespace NUMINAMATH_CALUDE_fish_population_estimate_l1930_193074

/-- Estimate the number of fish in a pond using mark-recapture method -/
theorem fish_population_estimate 
  (initially_marked : ℕ) 
  (recaptured : ℕ) 
  (marked_in_recapture : ℕ) 
  (h1 : initially_marked = 2000)
  (h2 : recaptured = 500)
  (h3 : marked_in_recapture = 40) :
  (initially_marked * recaptured) / marked_in_recapture = 25000 := by
  sorry

#eval (2000 * 500) / 40

end NUMINAMATH_CALUDE_fish_population_estimate_l1930_193074


namespace NUMINAMATH_CALUDE_mindy_message_count_l1930_193070

/-- The number of emails and phone messages Mindy has in total -/
def total_messages (phone_messages : ℕ) (emails : ℕ) : ℕ :=
  phone_messages + emails

/-- The relationship between emails and phone messages -/
def email_phone_relation (phone_messages : ℕ) : ℕ :=
  9 * phone_messages - 7

theorem mindy_message_count :
  ∃ (phone_messages : ℕ),
    email_phone_relation phone_messages = 83 ∧
    total_messages phone_messages 83 = 93 := by
  sorry

end NUMINAMATH_CALUDE_mindy_message_count_l1930_193070


namespace NUMINAMATH_CALUDE_equation_solution_l1930_193017

theorem equation_solution : 
  ∃ x : ℝ, x ≠ 2 ∧ (3 / (x - 2) = 2 + x / (2 - x)) ↔ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1930_193017


namespace NUMINAMATH_CALUDE_tray_height_l1930_193030

theorem tray_height (side_length : ℝ) (corner_distance : ℝ) (cut_angle : ℝ) 
  (h1 : side_length = 150)
  (h2 : corner_distance = 5)
  (h3 : cut_angle = 45) : 
  let tray_height := corner_distance * Real.sqrt 2 * Real.sin (cut_angle * π / 180)
  tray_height = 5 := by sorry

end NUMINAMATH_CALUDE_tray_height_l1930_193030


namespace NUMINAMATH_CALUDE_problem_solution_l1930_193023

theorem problem_solution (a b c d e : ℝ) 
  (eq1 : a - b - c + d = 18)
  (eq2 : a + b - c - d = 6)
  (eq3 : c + d - e = 5) :
  (2 * b - d + e) ^ 3 = 13824 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1930_193023


namespace NUMINAMATH_CALUDE_no_real_solutions_l1930_193087

theorem no_real_solutions : ¬ ∃ (x : ℝ), (x^(1/4) : ℝ) = 20 / (9 - 2 * (x^(1/4) : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1930_193087


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1930_193034

theorem absolute_value_inequality (x : ℝ) : 
  |((x + 1) / x)| > ((x + 1) / x) ↔ -1 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1930_193034


namespace NUMINAMATH_CALUDE_figure_area_theorem_l1930_193011

theorem figure_area_theorem (y : ℝ) :
  (3 * y)^2 + (7 * y)^2 + (1/2 * 3 * y * 7 * y) = 1200 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_theorem_l1930_193011


namespace NUMINAMATH_CALUDE_unused_ribbon_length_l1930_193085

/-- Given a ribbon of length 30 meters cut into 6 equal parts, 
    if 4 parts are used, then 10 meters of ribbon are not used. -/
theorem unused_ribbon_length 
  (total_length : ℝ) 
  (num_parts : ℕ) 
  (used_parts : ℕ) 
  (h1 : total_length = 30) 
  (h2 : num_parts = 6) 
  (h3 : used_parts = 4) : 
  total_length - (total_length / num_parts) * used_parts = 10 := by
  sorry


end NUMINAMATH_CALUDE_unused_ribbon_length_l1930_193085


namespace NUMINAMATH_CALUDE_triangle_area_l1930_193053

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π →
  A + B + C = π →
  b^2 + c^2 = a^2 - Real.sqrt 3 * b * c →
  b * c * Real.cos A = -4 →
  (1/2) * b * c * Real.sin A = (2 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1930_193053


namespace NUMINAMATH_CALUDE_square_difference_48_3_l1930_193016

theorem square_difference_48_3 : 48^2 - 2*(48*3) + 3^2 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_48_3_l1930_193016


namespace NUMINAMATH_CALUDE_impossibleToDetectAllGenuine_l1930_193049

/-- Represents a diamond --/
inductive Diamond
| genuine
| fake

/-- Represents the expert's response --/
inductive ExpertResponse
| zero
| one
| two

/-- A strategy for the expert to choose which pair to reveal --/
def ExpertStrategy := (Diamond × Diamond × Diamond) → (Diamond × Diamond)

/-- The expert's response based on the chosen pair --/
def expertRespond (pair : Diamond × Diamond) : ExpertResponse :=
  match pair with
  | (Diamond.genuine, Diamond.genuine) => ExpertResponse.two
  | (Diamond.fake, Diamond.fake) => ExpertResponse.zero
  | _ => ExpertResponse.one

/-- Represents the state of our knowledge about the diamonds --/
def Knowledge := Fin 100 → Option Diamond

theorem impossibleToDetectAllGenuine :
  ∃ (strategy : ExpertStrategy),
    ∀ (initialState : Knowledge),
    ∃ (finalState : Knowledge),
      (∃ i j, i ≠ j ∧ finalState i = none ∧ finalState j = none) ∧
      (∀ k, finalState k ≠ some Diamond.genuine → initialState k ≠ some Diamond.genuine) :=
by sorry

end NUMINAMATH_CALUDE_impossibleToDetectAllGenuine_l1930_193049


namespace NUMINAMATH_CALUDE_chess_pool_theorem_l1930_193022

theorem chess_pool_theorem (U : Type) 
  (A : Set U) -- Set of people who play chess
  (B : Set U) -- Set of people who are not interested in mathematics
  (C : Set U) -- Set of people who bathe in the pool every day
  (h1 : (A ∩ B).Nonempty) -- Condition 1
  (h2 : (C ∩ B ∩ A) = ∅) -- Condition 2
  : ¬(A ⊆ C) := by
  sorry

end NUMINAMATH_CALUDE_chess_pool_theorem_l1930_193022


namespace NUMINAMATH_CALUDE_horner_operations_for_f_l1930_193019

/-- Represents a polynomial of degree 4 -/
structure Polynomial4 where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Counts the number of operations in Horner's method for a polynomial of degree 4 -/
def hornerOperations (p : Polynomial4) : ℕ × ℕ :=
  sorry

/-- The specific polynomial f(x) = 2x^4 + 3x^3 - 2x^2 + 4x - 6 -/
def f : Polynomial4 := {
  a := 2
  b := 3
  c := -2
  d := 4
  e := -6
}

theorem horner_operations_for_f :
  hornerOperations f = (4, 4) := by sorry

end NUMINAMATH_CALUDE_horner_operations_for_f_l1930_193019


namespace NUMINAMATH_CALUDE_jurassic_zoo_bill_l1930_193061

/-- The Jurassic Zoo billing problem -/
theorem jurassic_zoo_bill :
  let adult_price : ℕ := 8
  let child_price : ℕ := 4
  let total_people : ℕ := 201
  let total_children : ℕ := 161
  let total_adults : ℕ := total_people - total_children
  let adults_bill : ℕ := total_adults * adult_price
  let children_bill : ℕ := total_children * child_price
  let total_bill : ℕ := adults_bill + children_bill
  total_bill = 964 := by
  sorry

end NUMINAMATH_CALUDE_jurassic_zoo_bill_l1930_193061


namespace NUMINAMATH_CALUDE_poly_expansion_nonzero_terms_l1930_193078

/-- The polynomial expression -/
def poly (x : ℝ) : ℝ := (2*x+5)*(3*x^2 - x + 4) - 4*(2*x^3 - 3*x^2 + x - 1)

/-- The expanded form of the polynomial -/
def expanded_poly (x : ℝ) : ℝ := 14*x^3 + x^2 + 7*x + 16

/-- The number of nonzero terms in the expanded polynomial -/
def num_nonzero_terms : ℕ := 4

theorem poly_expansion_nonzero_terms :
  (∀ x : ℝ, poly x = expanded_poly x) →
  num_nonzero_terms = 4 :=
by sorry

end NUMINAMATH_CALUDE_poly_expansion_nonzero_terms_l1930_193078


namespace NUMINAMATH_CALUDE_circle_points_m_value_l1930_193082

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points lie on the same circle -/
def onSameCircle (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (D E F : ℝ),
    p1.x^2 + p1.y^2 + D*p1.x + E*p1.y + F = 0 ∧
    p2.x^2 + p2.y^2 + D*p2.x + E*p2.y + F = 0 ∧
    p3.x^2 + p3.y^2 + D*p3.x + E*p3.y + F = 0 ∧
    p4.x^2 + p4.y^2 + D*p4.x + E*p4.y + F = 0

/-- Theorem: If (2,1), (4,2), (3,4), and (1,m) lie on the same circle, then m = 2 or m = 3 -/
theorem circle_points_m_value :
  ∀ (m : ℝ),
    onSameCircle
      (Point.mk 2 1)
      (Point.mk 4 2)
      (Point.mk 3 4)
      (Point.mk 1 m) →
    m = 2 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_m_value_l1930_193082


namespace NUMINAMATH_CALUDE_BC_completion_time_l1930_193063

/-- The time it takes for a group of workers to complete a job -/
def completion_time (work_rate : ℚ) : ℚ := 1 / work_rate

/-- The work rate of a single worker A -/
def work_rate_A : ℚ := 1 / 10

/-- The combined work rate of workers A and B -/
def work_rate_AB : ℚ := 1 / 5

/-- The combined work rate of workers A, B, and C -/
def work_rate_ABC : ℚ := 1 / 3

/-- The combined work rate of workers B and C -/
def work_rate_BC : ℚ := work_rate_ABC - work_rate_A

theorem BC_completion_time :
  completion_time work_rate_BC = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_BC_completion_time_l1930_193063


namespace NUMINAMATH_CALUDE_cd_player_only_percentage_l1930_193058

theorem cd_player_only_percentage
  (power_windows : ℝ)
  (anti_lock_brakes : ℝ)
  (cd_player : ℝ)
  (gps_system : ℝ)
  (pw_abs : ℝ)
  (abs_cd : ℝ)
  (pw_cd : ℝ)
  (gps_abs : ℝ)
  (gps_cd : ℝ)
  (pw_gps : ℝ)
  (h1 : power_windows = 60)
  (h2 : anti_lock_brakes = 40)
  (h3 : cd_player = 75)
  (h4 : gps_system = 50)
  (h5 : pw_abs = 10)
  (h6 : abs_cd = 15)
  (h7 : pw_cd = 20)
  (h8 : gps_abs = 12)
  (h9 : gps_cd = 18)
  (h10 : pw_gps = 25)
  (h11 : ∀ x, x ≤ 100) -- Assuming percentages are ≤ 100%
  : cd_player - (abs_cd + pw_cd + gps_cd) = 22 :=
by sorry


end NUMINAMATH_CALUDE_cd_player_only_percentage_l1930_193058


namespace NUMINAMATH_CALUDE_dividend_calculation_l1930_193037

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (hq : quotient = 120) 
  (hd : divisor = 456) 
  (hr : remainder = 333) : 
  divisor * quotient + remainder = 55053 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1930_193037


namespace NUMINAMATH_CALUDE_three_x_intercepts_l1930_193092

/-- The function representing the curve x = y^3 - 4y^2 + 3y + 2 -/
def f (y : ℝ) : ℝ := y^3 - 4*y^2 + 3*y + 2

/-- Theorem stating that the equation f(y) = 0 has exactly 3 real solutions -/
theorem three_x_intercepts : ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ y : ℝ, y ∈ s ↔ f y = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_x_intercepts_l1930_193092


namespace NUMINAMATH_CALUDE_water_volume_ratio_l1930_193095

/-- Represents a location with rainfall and area data -/
structure Location where
  rainfall : ℝ  -- rainfall in cm
  area : ℝ      -- area in hectares

/-- Calculates the volume of water collected at a location -/
def waterVolume (loc : Location) : ℝ :=
  loc.rainfall * loc.area * 10

/-- Theorem stating the ratio of water volumes collected at locations A, B, and C -/
theorem water_volume_ratio 
  (locA locB locC : Location)
  (hA : locA = { rainfall := 7, area := 2 })
  (hB : locB = { rainfall := 11, area := 3.5 })
  (hC : locC = { rainfall := 15, area := 5 }) :
  ∃ (k : ℝ), k > 0 ∧ 
    waterVolume locA = 140 * k ∧ 
    waterVolume locB = 385 * k ∧ 
    waterVolume locC = 750 * k :=
sorry

end NUMINAMATH_CALUDE_water_volume_ratio_l1930_193095


namespace NUMINAMATH_CALUDE_greatest_t_value_l1930_193007

theorem greatest_t_value : ∃ (t : ℝ), 
  (∀ (s : ℝ), (s^2 - s - 40) / (s - 8) = 5 / (s + 5) → s ≤ t) ∧
  (t^2 - t - 40) / (t - 8) = 5 / (t + 5) ∧
  t = -2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_t_value_l1930_193007


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1930_193021

theorem fraction_sum_equality : (3 : ℚ) / 30 + 9 / 300 + 27 / 3000 = 0.139 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1930_193021


namespace NUMINAMATH_CALUDE_train_crossing_time_l1930_193065

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_pass_man : ℝ) 
  (h1 : train_length = 186)
  (h2 : platform_length = 279)
  (h3 : time_pass_man = 8) :
  (train_length + platform_length) / (train_length / time_pass_man) = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1930_193065


namespace NUMINAMATH_CALUDE_percentage_difference_in_earnings_l1930_193090

/-- Given Mike's and Phil's hourly earnings, calculate the percentage difference -/
theorem percentage_difference_in_earnings (mike_earnings phil_earnings : ℝ) 
  (h1 : mike_earnings = 14)
  (h2 : phil_earnings = 7) :
  (mike_earnings - phil_earnings) / mike_earnings * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_in_earnings_l1930_193090


namespace NUMINAMATH_CALUDE_nathan_tokens_used_l1930_193088

/-- The number of times Nathan played air hockey -/
def air_hockey_games : ℕ := 2

/-- The number of times Nathan played basketball -/
def basketball_games : ℕ := 4

/-- The cost in tokens for each game -/
def tokens_per_game : ℕ := 3

/-- The total number of tokens Nathan used -/
def total_tokens : ℕ := air_hockey_games * tokens_per_game + basketball_games * tokens_per_game

theorem nathan_tokens_used : total_tokens = 18 := by
  sorry

end NUMINAMATH_CALUDE_nathan_tokens_used_l1930_193088


namespace NUMINAMATH_CALUDE_unique_prime_arith_seq_l1930_193031

/-- An arithmetic sequence of three prime numbers with common difference 80. -/
structure PrimeArithSeq where
  p₁ : ℕ
  p₂ : ℕ
  p₃ : ℕ
  prime_p₁ : Nat.Prime p₁
  prime_p₂ : Nat.Prime p₂
  prime_p₃ : Nat.Prime p₃
  diff_p₂_p₁ : p₂ = p₁ + 80
  diff_p₃_p₂ : p₃ = p₂ + 80

/-- There exists exactly one arithmetic sequence of three prime numbers with common difference 80. -/
theorem unique_prime_arith_seq : ∃! seq : PrimeArithSeq, True :=
  sorry

end NUMINAMATH_CALUDE_unique_prime_arith_seq_l1930_193031


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1930_193077

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y z t : ℝ, (f x + f z) * (f y + f t) = f (x * y - z * t) + f (x * t + y * z)) →
  (∀ x : ℝ, f x = 0 ∨ f x = (1 : ℝ) / 2 ∨ f x = x ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1930_193077


namespace NUMINAMATH_CALUDE_triangle_ratios_l1930_193066

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B = 8 ∧ d t.A t.C = 6 ∧ d t.B t.C = 4

-- Define angle bisector
def isAngleBisector (t : Triangle) (D : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.B D / d t.C D = d t.A t.B / d t.A t.C

-- Define the intersection point P
def intersectionPoint (t : Triangle) (D E : ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- The actual calculation of the intersection point

-- Define the circumcenter
def circumcenter (t : Triangle) : ℝ × ℝ :=
  sorry  -- The actual calculation of the circumcenter

-- Main theorem
theorem triangle_ratios (t : Triangle) (D E : ℝ × ℝ) :
  isValidTriangle t →
  isAngleBisector t D →
  isAngleBisector t E →
  let P := intersectionPoint t D E
  let O := circumcenter t
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.B P / d P E = 2 ∧
  d O D / d D t.A = 1/3 :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_ratios_l1930_193066


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1930_193069

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1930_193069


namespace NUMINAMATH_CALUDE_odd_function_value_l1930_193027

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 9

-- State the theorem
theorem odd_function_value (hf_odd : ∀ x, f (-x) = -f x) (hg : g (-2) = 3) : f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l1930_193027


namespace NUMINAMATH_CALUDE_factorial_99_trailing_zeros_l1930_193001

/-- The number of trailing zeros in n factorial -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 99! has 22 trailing zeros -/
theorem factorial_99_trailing_zeros :
  trailingZeros 99 = 22 := by
  sorry

end NUMINAMATH_CALUDE_factorial_99_trailing_zeros_l1930_193001


namespace NUMINAMATH_CALUDE_test_score_problem_l1930_193050

theorem test_score_problem (total_questions : ℕ) (score : ℤ) 
  (correct_answers : ℕ) (incorrect_answers : ℕ) : 
  total_questions = 100 →
  score = correct_answers - 2 * incorrect_answers →
  correct_answers + incorrect_answers = total_questions →
  score = 73 →
  correct_answers = 91 := by
sorry

end NUMINAMATH_CALUDE_test_score_problem_l1930_193050


namespace NUMINAMATH_CALUDE_solution_value_l1930_193057

theorem solution_value (k : ℝ) : (2 * 3 - k + 1 = 0) → k = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1930_193057


namespace NUMINAMATH_CALUDE_quadratic_functions_coincidence_l1930_193014

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if two quadratic functions can coincide through parallel translation -/
def can_coincide (f g : QuadraticFunction) : Prop :=
  f.a = g.a ∧ f.a ≠ 0

/-- The three given quadratic functions -/
def A : QuadraticFunction := ⟨1, 0, -1⟩
def B : QuadraticFunction := ⟨-1, 0, 1⟩
def C : QuadraticFunction := ⟨1, 2, -1⟩

theorem quadratic_functions_coincidence :
  can_coincide A C ∧ ¬can_coincide A B ∧ ¬can_coincide B C := by sorry

end NUMINAMATH_CALUDE_quadratic_functions_coincidence_l1930_193014


namespace NUMINAMATH_CALUDE_million_to_scientific_notation_l1930_193083

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem million_to_scientific_notation :
  toScientificNotation (42.39 * 1000000) = ScientificNotation.mk 4.239 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_million_to_scientific_notation_l1930_193083


namespace NUMINAMATH_CALUDE_incorrect_statement_B_l1930_193072

/-- Definition of a "2 times root equation" -/
def is_two_times_root_equation (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ (a*x^2 + b*x + c = 0) ∧ (a*y^2 + b*y + c = 0) ∧ (x = 2*y ∨ y = 2*x)

/-- The statement to be proven false -/
theorem incorrect_statement_B :
  ¬(∀ (m n : ℝ), is_two_times_root_equation 1 (m-2) (-2*m) → m + n = 0) :=
sorry

end NUMINAMATH_CALUDE_incorrect_statement_B_l1930_193072


namespace NUMINAMATH_CALUDE_unfoldable_cylinder_volume_l1930_193064

/-- A cylinder with a lateral surface that unfolds into a rectangle -/
structure UnfoldableCylinder where
  rectangle_length : ℝ
  rectangle_width : ℝ

/-- The volume of an unfoldable cylinder -/
def cylinder_volume (c : UnfoldableCylinder) : Set ℝ :=
  { v | ∃ (r h : ℝ), 
    ((2 * Real.pi * r = c.rectangle_length ∧ h = c.rectangle_width) ∨
     (2 * Real.pi * r = c.rectangle_width ∧ h = c.rectangle_length)) ∧
    v = Real.pi * r^2 * h }

/-- Theorem: The volume of a cylinder with lateral surface unfolding to a 4π by 1 rectangle is either 4π or 1 -/
theorem unfoldable_cylinder_volume :
  let c := UnfoldableCylinder.mk (4 * Real.pi) 1
  cylinder_volume c = {4 * Real.pi, 1} := by
  sorry

end NUMINAMATH_CALUDE_unfoldable_cylinder_volume_l1930_193064


namespace NUMINAMATH_CALUDE_variation_relationship_l1930_193060

theorem variation_relationship (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_variation_relationship_l1930_193060


namespace NUMINAMATH_CALUDE_function_inequality_l1930_193056

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1930_193056


namespace NUMINAMATH_CALUDE_combined_average_age_l1930_193045

theorem combined_average_age (people_c people_d : ℕ) (avg_age_c avg_age_d : ℚ) :
  people_c = 8 →
  people_d = 6 →
  avg_age_c = 30 →
  avg_age_d = 35 →
  (people_c * avg_age_c + people_d * avg_age_d) / (people_c + people_d) = 32 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_age_l1930_193045


namespace NUMINAMATH_CALUDE_henri_reads_670_words_l1930_193046

def total_time : ℝ := 8
def movie_durations : List ℝ := [3.5, 1.5, 1.25, 0.75]
def reading_speeds : List (ℝ × ℝ) := [(30, 12), (20, 8)]

def calculate_words_read (total_time : ℝ) (movie_durations : List ℝ) (reading_speeds : List (ℝ × ℝ)) : ℕ :=
  sorry

theorem henri_reads_670_words :
  calculate_words_read total_time movie_durations reading_speeds = 670 := by
  sorry

end NUMINAMATH_CALUDE_henri_reads_670_words_l1930_193046


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1930_193039

theorem rectangle_dimensions (x : ℝ) : 
  (x + 3) * (3 * x - 2) = 9 * x + 1 → 
  x > 0 → 
  3 * x - 2 > 0 → 
  x = (11 + Real.sqrt 205) / 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1930_193039


namespace NUMINAMATH_CALUDE_subset_condition_l1930_193068

-- Define the sets A and B
def A : Set ℝ := {x | (x + 1) / (x - 2) < 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 > 0}

-- Define the complement of A in ℝ
def A_complement : Set ℝ := {x | ¬ (x ∈ A)}

-- State the theorem
theorem subset_condition (a : ℝ) :
  0 < a ∧ a ≤ 1/2 → B a ⊆ A_complement :=
by sorry

end NUMINAMATH_CALUDE_subset_condition_l1930_193068


namespace NUMINAMATH_CALUDE_abc_log_sum_l1930_193052

theorem abc_log_sum (A B C : ℕ+) (h_coprime : Nat.gcd A.val (Nat.gcd B.val C.val) = 1)
  (h_eq : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) :
  A + B + C = 5 := by
  sorry

end NUMINAMATH_CALUDE_abc_log_sum_l1930_193052


namespace NUMINAMATH_CALUDE_correct_factorization_l1930_193033

theorem correct_factorization (a : ℝ) : 2*a^2 - 4*a + 2 = 2*(a-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1930_193033


namespace NUMINAMATH_CALUDE_digestion_period_correct_l1930_193091

/-- The period (in days) for a python to completely digest an alligator -/
def digestion_period : ℕ := 7

/-- The number of days observed -/
def observation_days : ℕ := 616

/-- The maximum number of alligators eaten in the observation period -/
def max_alligators_eaten : ℕ := 88

/-- Theorem stating that the digestion period is correct given the observed data -/
theorem digestion_period_correct : 
  digestion_period * max_alligators_eaten = observation_days :=
by sorry

end NUMINAMATH_CALUDE_digestion_period_correct_l1930_193091


namespace NUMINAMATH_CALUDE_petr_ivanovich_insurance_contract_l1930_193096

/-- Represents an insurance tool --/
inductive InsuranceTool
| AggregateInsuranceAmount
| Deductible

/-- Represents an insurance document --/
inductive InsuranceDocument
| InsuranceRules

/-- Represents a person --/
structure Person where
  name : String

/-- Represents an insurance contract --/
structure InsuranceContract where
  owner : Person
  tools : List InsuranceTool
  appendix : InsuranceDocument

/-- Theorem stating the correct insurance tools and document for Petr Ivanovich's contract --/
theorem petr_ivanovich_insurance_contract :
  ∃ (contract : InsuranceContract),
    contract.owner = Person.mk "Petr Ivanovich" ∧
    contract.tools = [InsuranceTool.AggregateInsuranceAmount, InsuranceTool.Deductible] ∧
    contract.appendix = InsuranceDocument.InsuranceRules :=
by sorry

end NUMINAMATH_CALUDE_petr_ivanovich_insurance_contract_l1930_193096


namespace NUMINAMATH_CALUDE_paco_cookies_left_l1930_193089

/-- The number of cookies Paco has left -/
def cookies_left (initial : ℕ) (given_away : ℕ) (eaten : ℕ) : ℕ :=
  initial - given_away - eaten

/-- Theorem stating that Paco has 12 cookies left -/
theorem paco_cookies_left :
  cookies_left 36 14 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_left_l1930_193089


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1930_193018

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem absolute_value_inequality
  (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_f_1 : f 1 = 0)
  (h_functional_equation : ∀ x y, x > 0 → y > 0 → f x + f y = f (x * y)) :
  ∀ x y, 0 < x → x < y → y < 1 → |f x| > |f y| :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1930_193018


namespace NUMINAMATH_CALUDE_curve_is_line_segment_l1930_193003

-- Define the parametric equations
def x (t : ℝ) : ℝ := 3 * t^2 + 2
def y (t : ℝ) : ℝ := t^2 - 1

-- Define the range of t
def t_range : Set ℝ := {t | 0 ≤ t ∧ t ≤ 5}

-- Define the curve as a set of points
def curve : Set (ℝ × ℝ) := {(x t, y t) | t ∈ t_range}

-- Theorem statement
theorem curve_is_line_segment : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ (p : ℝ × ℝ), p ∈ curve → a * p.1 + b * p.2 + c = 0) ∧
  (∃ (p q : ℝ × ℝ), p ∈ curve ∧ q ∈ curve ∧ p ≠ q ∧
    ∀ (r : ℝ × ℝ), r ∈ curve → ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ r = (1 - t) • p + t • q) :=
sorry

end NUMINAMATH_CALUDE_curve_is_line_segment_l1930_193003


namespace NUMINAMATH_CALUDE_sheet_width_l1930_193005

/-- Given a rectangular sheet of paper with length 10 inches, a 1.5-inch margin all around,
    and a picture covering 38.5 square inches, prove that the width of the sheet is 8.5 inches. -/
theorem sheet_width (W : ℝ) (margin : ℝ) (picture_area : ℝ) : 
  margin = 1.5 →
  picture_area = 38.5 →
  (W - 2 * margin) * (10 - 2 * margin) = picture_area →
  W = 8.5 := by
sorry

end NUMINAMATH_CALUDE_sheet_width_l1930_193005


namespace NUMINAMATH_CALUDE_jogging_distance_l1930_193036

theorem jogging_distance (x t : ℝ) 
  (h1 : (x + 3/4) * (3*t/4) = x * t)
  (h2 : (x - 3/4) * (t + 3) = x * t) :
  x * t = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_jogging_distance_l1930_193036


namespace NUMINAMATH_CALUDE_spring_excursion_participants_l1930_193035

theorem spring_excursion_participants :
  let water_students : ℕ := 80
  let fruit_students : ℕ := 70
  let neither_students : ℕ := 6
  let both_students : ℕ := (water_students + fruit_students - neither_students) / 2
  let total_participants : ℕ := both_students * 2
  total_participants = 104 :=
by sorry

end NUMINAMATH_CALUDE_spring_excursion_participants_l1930_193035


namespace NUMINAMATH_CALUDE_no_rational_points_on_sqrt3_circle_l1930_193062

theorem no_rational_points_on_sqrt3_circle : 
  ¬∃ (x y : ℚ), x^2 + y^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_no_rational_points_on_sqrt3_circle_l1930_193062


namespace NUMINAMATH_CALUDE_marge_final_plant_count_l1930_193048

/-- Calculates the number of plants Marge ended up with in her garden -/
def marges_garden (total_seeds : ℕ) (non_growing_seeds : ℕ) (weed_kept : ℕ) : ℕ :=
  let growing_plants := total_seeds - non_growing_seeds
  let eaten_plants := growing_plants / 3
  let uneaten_plants := growing_plants - eaten_plants
  let strangled_plants := uneaten_plants / 3
  let surviving_plants := uneaten_plants - strangled_plants
  surviving_plants + weed_kept

/-- Theorem stating that Marge ended up with 9 plants in her garden -/
theorem marge_final_plant_count :
  marges_garden 23 5 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_marge_final_plant_count_l1930_193048


namespace NUMINAMATH_CALUDE_circle_radius_increase_l1930_193080

theorem circle_radius_increase (r₁ r₂ : ℝ) : 
  2 * Real.pi * r₁ = 30 → 2 * Real.pi * r₂ = 40 → r₂ - r₁ = 5 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increase_l1930_193080


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l1930_193097

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a| + 3

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x y, 1 < x ∧ x < y → f a x < f a y) → a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l1930_193097


namespace NUMINAMATH_CALUDE_polynomial_division_l1930_193038

def P (x : ℝ) : ℝ := x^6 - 6*x^4 - 4*x^3 + 9*x^2 + 12*x + 4

def Q (x : ℝ) : ℝ := x^4 + x^3 - 3*x^2 - 5*x - 2

def R (x : ℝ) : ℝ := x^2 - x - 2

theorem polynomial_division :
  ∀ x : ℝ, P x = Q x * R x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_l1930_193038


namespace NUMINAMATH_CALUDE_smallest_abs_value_rational_l1930_193071

theorem smallest_abs_value_rational : ∀ q : ℚ, |0| ≤ |q| := by
  sorry

end NUMINAMATH_CALUDE_smallest_abs_value_rational_l1930_193071


namespace NUMINAMATH_CALUDE_final_value_properties_l1930_193079

/-- Represents the transformation process on the blackboard -/
def blackboard_transform (n : ℕ) : Set ℕ → Set ℕ := sorry

/-- The final state of the blackboard after transformations -/
def final_state (n : ℕ) : Set ℕ := sorry

/-- Theorem stating the properties of the final value k -/
theorem final_value_properties (n : ℕ) (h : n ≥ 3) :
  ∃ (k t : ℕ), final_state n = {k} ∧ k = 2^t ∧ k ≥ n :=
sorry

end NUMINAMATH_CALUDE_final_value_properties_l1930_193079


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1930_193067

/-- A hyperbola with center at the origin, a focus at (√2, 0), and the distance
    from this focus to an asymptote being 1 has the equation x^2 - y^2 = 1 -/
theorem hyperbola_equation (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 - y^2 = 1) ↔
  (0, 0) ∈ C ∧ 
  F = (Real.sqrt 2, 0) ∧ 
  F ∈ C ∧
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ C → a * y = b * x ∨ a * y = -b * x) ∧
    (abs (b * Real.sqrt 2) / Real.sqrt (a^2 + b^2) = 1)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1930_193067


namespace NUMINAMATH_CALUDE_negation_equivalence_l1930_193008

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < 0 ∧ Real.log (x^2 - 2*x - 1) ≥ 0) ↔
  (∀ x : ℝ, x < 0 → Real.log (x^2 - 2*x - 1) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1930_193008


namespace NUMINAMATH_CALUDE_tom_chocolate_boxes_l1930_193042

/-- The number of pieces Tom gave away -/
def pieces_given_away : ℕ := 8

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 3

/-- The number of pieces Tom still has -/
def pieces_remaining : ℕ := 18

/-- The number of boxes Tom bought initially -/
def boxes_bought : ℕ := 8

theorem tom_chocolate_boxes :
  boxes_bought * pieces_per_box = pieces_given_away + pieces_remaining :=
by sorry

end NUMINAMATH_CALUDE_tom_chocolate_boxes_l1930_193042


namespace NUMINAMATH_CALUDE_teacher_arrangement_count_teacher_arrangement_proof_l1930_193081

theorem teacher_arrangement_count : Nat → Nat → Nat
  | n, k => Nat.choose n k

theorem teacher_arrangement_proof :
  teacher_arrangement_count 22 5 = 26334 := by
  sorry

end NUMINAMATH_CALUDE_teacher_arrangement_count_teacher_arrangement_proof_l1930_193081


namespace NUMINAMATH_CALUDE_phase_shift_cosine_l1930_193098

/-- The phase shift of y = 3 cos(4x - π/4) is π/16 -/
theorem phase_shift_cosine (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 3 * Real.cos (4 * x - π / 4)
  ∃ (shift : ℝ), shift = π / 16 ∧ 
    ∀ (t : ℝ), f (t + shift) = 3 * Real.cos (4 * t) := by
  sorry

end NUMINAMATH_CALUDE_phase_shift_cosine_l1930_193098


namespace NUMINAMATH_CALUDE_xiaohuas_stamp_buying_ways_l1930_193041

/-- Represents the number of ways to buy stamps given the total money and stamp prices -/
def waysToByStamps (totalMoney : ℕ) (stamp1Price : ℕ) (stamp2Price : ℕ) : ℕ := 
  let maxStamp1 := totalMoney / stamp1Price
  let maxStamp2 := totalMoney / stamp2Price
  (maxStamp1 + 1) * (maxStamp2 + 1) - 1

/-- The problem statement -/
theorem xiaohuas_stamp_buying_ways :
  waysToByStamps 7 2 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_xiaohuas_stamp_buying_ways_l1930_193041


namespace NUMINAMATH_CALUDE_programmer_is_odd_one_out_l1930_193024

/-- Represents a profession --/
inductive Profession
  | Dentist
  | ElementarySchoolTeacher
  | Programmer

/-- Represents whether a profession has special pension benefits --/
def has_special_pension_benefits (p : Profession) : Prop :=
  match p with
  | Profession.Dentist => true
  | Profession.ElementarySchoolTeacher => true
  | Profession.Programmer => false

/-- Theorem stating that the programmer is the odd one out --/
theorem programmer_is_odd_one_out :
  ∃! p : Profession, ¬(has_special_pension_benefits p) :=
sorry


end NUMINAMATH_CALUDE_programmer_is_odd_one_out_l1930_193024


namespace NUMINAMATH_CALUDE_triangle_properties_l1930_193029

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  median_CM : ℝ → ℝ → ℝ
  altitude_BH : ℝ → ℝ → ℝ

/-- The given triangle satisfies the problem conditions -/
def given_triangle : Triangle where
  A := (5, 1)
  B := sorry
  C := sorry
  median_CM := λ x y ↦ 2*x - y - 5
  altitude_BH := λ x y ↦ x - 2*y - 5

theorem triangle_properties (t : Triangle) (h : t = given_triangle) : 
  t.C = (4, 3) ∧ 
  (λ x y ↦ 6*x - 5*y - 9) = (λ x y ↦ 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1930_193029


namespace NUMINAMATH_CALUDE_tangent_circle_center_l1930_193054

/-- A circle tangent to two parallel lines with its center on a third line --/
structure TangentCircle where
  /-- The x-coordinate of the circle's center --/
  x : ℝ
  /-- The y-coordinate of the circle's center --/
  y : ℝ
  /-- The circle is tangent to the line 4x - 3y = 30 --/
  tangent_line1 : 4 * x - 3 * y = 30
  /-- The circle is tangent to the line 4x - 3y = -10 --/
  tangent_line2 : 4 * x - 3 * y = -10
  /-- The center of the circle lies on the line 2x + y = 0 --/
  center_line : 2 * x + y = 0

/-- The center of the circle satisfies all conditions and has coordinates (1, -2) --/
theorem tangent_circle_center : 
  ∃ (c : TangentCircle), c.x = 1 ∧ c.y = -2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_center_l1930_193054
