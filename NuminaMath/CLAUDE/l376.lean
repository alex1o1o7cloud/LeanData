import Mathlib

namespace NUMINAMATH_CALUDE_square_difference_l376_37680

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 8 / 15) 
  (h2 : x - y = 1 / 45) : 
  x^2 - y^2 = 8 / 675 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l376_37680


namespace NUMINAMATH_CALUDE_complex_sum_reciprocal_magnitude_l376_37656

theorem complex_sum_reciprocal_magnitude (z w : ℂ) :
  Complex.abs z = 2 →
  Complex.abs w = 4 →
  Complex.abs (z + w) = 3 →
  ∃ θ : ℝ, θ = Real.pi / 3 ∧ z * Complex.exp (Complex.I * θ) = w →
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_reciprocal_magnitude_l376_37656


namespace NUMINAMATH_CALUDE_circle_tangent_and_shortest_chord_l376_37640

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define points P and M
def P : ℝ × ℝ := (2, 5)
def M : ℝ × ℝ := (5, 0)

-- Define the line with shortest chord length
def shortest_chord_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the tangent lines
def tangent_line1 (x y : ℝ) : Prop := 3*x + 4*y - 15 = 0
def tangent_line2 (x : ℝ) : Prop := x = 5

theorem circle_tangent_and_shortest_chord :
  (∀ x y, C x y → shortest_chord_line x y → (x, y) = P ∨ (C x y ∧ shortest_chord_line x y)) ∧
  (∀ x y, C x y → tangent_line1 x y → (x, y) = M ∨ (C x y ∧ tangent_line1 x y)) ∧
  (∀ x y, C x y → tangent_line2 x → (x, y) = M ∨ (C x y ∧ tangent_line2 x)) := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_and_shortest_chord_l376_37640


namespace NUMINAMATH_CALUDE_no_rain_probability_l376_37679

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l376_37679


namespace NUMINAMATH_CALUDE_shaded_area_of_squares_l376_37602

theorem shaded_area_of_squares (s₁ s₂ : ℝ) (h₁ : s₁ = 2) (h₂ : s₂ = 6) :
  (1/2 * s₁ * s₁) + (1/2 * s₂ * s₂) = 20 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_squares_l376_37602


namespace NUMINAMATH_CALUDE_construction_time_for_330_meters_l376_37648

/-- Represents the daily progress of road construction in meters -/
def daily_progress : ℕ := 30

/-- Calculates the cumulative progress given the number of days -/
def cumulative_progress (days : ℕ) : ℕ :=
  daily_progress * days

/-- Theorem stating that 330 meters of cumulative progress corresponds to 11 days of construction -/
theorem construction_time_for_330_meters :
  cumulative_progress 11 = 330 ∧ cumulative_progress 10 ≠ 330 := by
  sorry

end NUMINAMATH_CALUDE_construction_time_for_330_meters_l376_37648


namespace NUMINAMATH_CALUDE_two_positive_real_roots_condition_no_real_roots_necessary_condition_l376_37622

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : Prop := x^2 + (m - 3) * x + m = 0

-- Define the condition for two positive real roots
def has_two_positive_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
  quadratic_equation m x₁ ∧ quadratic_equation m x₂

-- Define the condition for no real roots
def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, ¬(quadratic_equation m x)

-- Theorem for two positive real roots
theorem two_positive_real_roots_condition :
  ∀ m : ℝ, has_two_positive_real_roots m ↔ (0 < m ∧ m ≤ 1) :=
sorry

-- Theorem for necessary condition of no real roots
theorem no_real_roots_necessary_condition :
  ∀ m : ℝ, has_no_real_roots m → m > 1 :=
sorry

end NUMINAMATH_CALUDE_two_positive_real_roots_condition_no_real_roots_necessary_condition_l376_37622


namespace NUMINAMATH_CALUDE_equation_solution_l376_37659

theorem equation_solution :
  ∀ x : ℝ, (x + 4)^2 = 5*(x + 4) ↔ x = -4 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l376_37659


namespace NUMINAMATH_CALUDE_M_intersect_N_l376_37665

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}

theorem M_intersect_N : M ∩ N = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l376_37665


namespace NUMINAMATH_CALUDE_profit_percentage_is_twenty_percent_l376_37636

/-- Calculates the profit percentage given wholesale price, retail price, and discount percentage. -/
def profit_percentage (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that under the given conditions, the profit percentage is 20%. -/
theorem profit_percentage_is_twenty_percent :
  profit_percentage 90 120 (10/100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_twenty_percent_l376_37636


namespace NUMINAMATH_CALUDE_difference_in_cost_l376_37629

def joy_pencils : ℕ := 30
def colleen_pencils : ℕ := 50
def pencil_cost : ℕ := 4

theorem difference_in_cost : (colleen_pencils - joy_pencils) * pencil_cost = 80 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_cost_l376_37629


namespace NUMINAMATH_CALUDE_parabola_rotation_l376_37652

/-- A parabola in the xy-plane -/
structure Parabola where
  a : ℝ  -- coefficient of (x-h)^2
  h : ℝ  -- x-coordinate of vertex
  k : ℝ  -- y-coordinate of vertex

/-- Rotate a parabola by 180 degrees around its vertex -/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := p.h, k := p.k }

theorem parabola_rotation (p : Parabola) (hp : p = ⟨2, 3, -2⟩) :
  rotate180 p = ⟨-2, 3, -2⟩ := by
  sorry

#check parabola_rotation

end NUMINAMATH_CALUDE_parabola_rotation_l376_37652


namespace NUMINAMATH_CALUDE_probability_a_speaks_truth_l376_37683

theorem probability_a_speaks_truth 
  (prob_b : ℝ)
  (prob_a_and_b : ℝ)
  (h1 : prob_b = 0.60)
  (h2 : prob_a_and_b = 0.51)
  (h3 : ∃ (prob_a : ℝ), prob_a_and_b = prob_a * prob_b) :
  ∃ (prob_a : ℝ), prob_a = 0.85 := by
sorry

end NUMINAMATH_CALUDE_probability_a_speaks_truth_l376_37683


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l376_37645

theorem binomial_coefficient_equality (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l376_37645


namespace NUMINAMATH_CALUDE_freds_shopping_cost_l376_37678

/-- Calculates the total cost of Fred's shopping trip --/
def calculate_shopping_cost (orange_price : ℚ) (orange_quantity : ℕ)
                            (cereal_price : ℚ) (cereal_quantity : ℕ)
                            (bread_price : ℚ) (bread_quantity : ℕ)
                            (cookie_price : ℚ) (cookie_quantity : ℕ)
                            (bakery_discount_threshold : ℚ)
                            (bakery_discount_rate : ℚ)
                            (coupon_threshold : ℚ)
                            (coupon_value : ℚ) : ℚ :=
  let orange_total := orange_price * orange_quantity
  let cereal_total := cereal_price * cereal_quantity
  let bread_total := bread_price * bread_quantity
  let cookie_total := cookie_price * cookie_quantity
  let bakery_total := bread_total + cookie_total
  let total_before_discounts := orange_total + cereal_total + bakery_total
  let bakery_discount := if bakery_total > bakery_discount_threshold then bakery_total * bakery_discount_rate else 0
  let total_after_bakery_discount := total_before_discounts - bakery_discount
  let final_total := if total_before_discounts ≥ coupon_threshold then total_after_bakery_discount - coupon_value else total_after_bakery_discount
  final_total

/-- Theorem stating that Fred's shopping cost is $30.5 --/
theorem freds_shopping_cost :
  calculate_shopping_cost 2 4 4 3 3 3 6 1 15 (1/10) 30 3 = 30.5 := by
  sorry

end NUMINAMATH_CALUDE_freds_shopping_cost_l376_37678


namespace NUMINAMATH_CALUDE_xiaoming_average_is_92_l376_37621

/-- Calculates the weighted average of Xiao Ming's math scores -/
def xiaoming_weighted_average : ℚ :=
  let regular_score : ℚ := 89
  let midterm_score : ℚ := 91
  let final_score : ℚ := 95
  let regular_weight : ℚ := 3
  let midterm_weight : ℚ := 3
  let final_weight : ℚ := 4
  (regular_score * regular_weight + midterm_score * midterm_weight + final_score * final_weight) /
  (regular_weight + midterm_weight + final_weight)

/-- Theorem stating that Xiao Ming's weighted average math score is 92 -/
theorem xiaoming_average_is_92 : xiaoming_weighted_average = 92 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_average_is_92_l376_37621


namespace NUMINAMATH_CALUDE_students_walking_home_l376_37618

theorem students_walking_home (bus car bicycle skateboard : ℚ) 
  (h1 : bus = 3/8)
  (h2 : car = 2/5)
  (h3 : bicycle = 1/8)
  (h4 : skateboard = 5/100)
  : 1 - (bus + car + bicycle + skateboard) = 1/20 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_l376_37618


namespace NUMINAMATH_CALUDE_fish_tank_water_l376_37649

theorem fish_tank_water (initial_water : ℝ) (added_water : ℝ) :
  initial_water = 7.75 →
  added_water = 7 →
  initial_water + added_water = 14.75 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_tank_water_l376_37649


namespace NUMINAMATH_CALUDE_decreasing_function_l376_37672

-- Define the four functions
def f1 (x : ℝ) : ℝ := x^2 + 1
def f2 (x : ℝ) : ℝ := -x^2 + 1
def f3 (x : ℝ) : ℝ := 2*x + 1
def f4 (x : ℝ) : ℝ := -2*x + 1

-- Theorem statement
theorem decreasing_function : 
  (∀ x : ℝ, HasDerivAt f4 (-2) x) ∧ 
  (∀ x : ℝ, (HasDerivAt f1 (2*x) x) ∨ (HasDerivAt f2 (-2*x) x) ∨ (HasDerivAt f3 2 x)) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_l376_37672


namespace NUMINAMATH_CALUDE_sine_equality_implies_equal_arguments_l376_37675

theorem sine_equality_implies_equal_arguments
  (α β γ τ : ℝ)
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0 ∧ τ > 0)
  (h_eq : ∀ x : ℝ, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (τ * x)) :
  α = γ ∨ α = τ :=
sorry

end NUMINAMATH_CALUDE_sine_equality_implies_equal_arguments_l376_37675


namespace NUMINAMATH_CALUDE_banana_groups_indeterminate_l376_37646

theorem banana_groups_indeterminate 
  (total_bananas : ℕ) 
  (total_oranges : ℕ) 
  (orange_groups : ℕ) 
  (oranges_per_group : ℕ) 
  (h1 : total_bananas = 142) 
  (h2 : total_oranges = 356) 
  (h3 : orange_groups = 178) 
  (h4 : oranges_per_group = 2) 
  (h5 : total_oranges = orange_groups * oranges_per_group) : 
  ∀ (banana_groups : ℕ), ¬ (∃ (bananas_per_group : ℕ), total_bananas = banana_groups * bananas_per_group) :=
by sorry

end NUMINAMATH_CALUDE_banana_groups_indeterminate_l376_37646


namespace NUMINAMATH_CALUDE_sponge_city_philosophy_l376_37673

/-- Represents a sponge city -/
structure SpongeCity where
  resilience : Bool
  waterManagement : Bool
  pilotProject : Bool

/-- Philosophical perspectives on sponge cities -/
inductive PhilosophicalPerspective
  | overall_function_greater
  | integrated_thinking
  | new_connections
  | internal_structure_optimization

/-- Checks if a given philosophical perspective applies to sponge cities -/
def applies_to_sponge_cities (sc : SpongeCity) (pp : PhilosophicalPerspective) : Prop :=
  match pp with
  | PhilosophicalPerspective.overall_function_greater => true
  | PhilosophicalPerspective.integrated_thinking => true
  | _ => false

/-- Theorem: Sponge cities reflect specific philosophical perspectives -/
theorem sponge_city_philosophy (sc : SpongeCity) 
  (h1 : sc.resilience = true) 
  (h2 : sc.waterManagement = true) 
  (h3 : sc.pilotProject = true) :
  (applies_to_sponge_cities sc PhilosophicalPerspective.overall_function_greater) ∧
  (applies_to_sponge_cities sc PhilosophicalPerspective.integrated_thinking) :=
by
  sorry

end NUMINAMATH_CALUDE_sponge_city_philosophy_l376_37673


namespace NUMINAMATH_CALUDE_average_problem_l376_37601

theorem average_problem (numbers : List ℕ) (x : ℕ) : 
  numbers = [201, 202, 204, 205, 206, 209, 209, 210, 212] →
  (numbers.sum + x) / 10 = 207 →
  x = 212 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l376_37601


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l376_37657

theorem constant_ratio_problem (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  (∀ x y, (4 * x + 3) / (2 * y - 5) = k) →
  x₁ = 1 →
  y₁ = 5 →
  y₂ = 10 →
  x₂ = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l376_37657


namespace NUMINAMATH_CALUDE_effective_CAGR_l376_37641

/-- The effective Compound Annual Growth Rate (CAGR) for an investment with stepped interest rates, inflation, and currency exchange rate changes. -/
theorem effective_CAGR 
  (R1 R2 R3 R4 I C : ℝ) 
  (h_growth : (3/5 : ℝ) = (1 + R1/100)^(5/2) * (1 + R2/100)^(5/2) * (1 + R3/100)^(5/2) * (1 + R4/100)^(5/2)) :
  ∃ CAGR : ℝ, 
    CAGR = ((1 + R1/100)^(5/2) * (1 + R2/100)^(5/2) * (1 + R3/100)^(5/2) * (1 + R4/100)^(5/2) / (1 + I/100)^10 * (1 + C/100)^10)^(1/10) - 1 := by
  sorry

end NUMINAMATH_CALUDE_effective_CAGR_l376_37641


namespace NUMINAMATH_CALUDE_original_number_proof_l376_37626

theorem original_number_proof :
  ∀ (original_number : ℤ),
    original_number + 3377 = 13200 →
    original_number = 9823 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l376_37626


namespace NUMINAMATH_CALUDE_cubic_root_equation_solutions_l376_37669

theorem cubic_root_equation_solutions :
  let f (x : ℝ) := Real.rpow (18 * x - 2) (1/3) + Real.rpow (16 * x + 2) (1/3) - 6 * Real.rpow x (1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = -1/12 ∨ x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solutions_l376_37669


namespace NUMINAMATH_CALUDE_special_quadrilateral_is_square_special_quadrilateral_is_not_always_square_l376_37698

/-- A quadrilateral with perpendicular and equal diagonals --/
structure SpecialQuadrilateral where
  /-- The diagonals are perpendicular --/
  diagonals_perpendicular : Bool
  /-- The diagonals are equal in length --/
  diagonals_equal : Bool

/-- Definition of a square --/
def is_square (q : SpecialQuadrilateral) : Prop :=
  q.diagonals_perpendicular ∧ q.diagonals_equal

/-- The statement to be proven false --/
theorem special_quadrilateral_is_square (q : SpecialQuadrilateral) :
  q.diagonals_perpendicular ∧ q.diagonals_equal → is_square q :=
by
  sorry

/-- The theorem stating that the above statement is false --/
theorem special_quadrilateral_is_not_always_square :
  ¬ (∀ q : SpecialQuadrilateral, q.diagonals_perpendicular ∧ q.diagonals_equal → is_square q) :=
by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_is_square_special_quadrilateral_is_not_always_square_l376_37698


namespace NUMINAMATH_CALUDE_small_painting_price_is_80_l376_37627

/-- The price of a small painting given the conditions of Michael's art sale -/
def small_painting_price (large_price : ℕ) (large_sold small_sold : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings - large_price * large_sold) / small_sold

/-- Theorem stating that the price of a small painting is $80 under the given conditions -/
theorem small_painting_price_is_80 :
  small_painting_price 100 5 8 1140 = 80 := by
  sorry

end NUMINAMATH_CALUDE_small_painting_price_is_80_l376_37627


namespace NUMINAMATH_CALUDE_books_in_boxes_l376_37660

/-- The number of ways to place n different objects into k different boxes -/
def arrangements (n k : ℕ) : ℕ := k^n

/-- There are 6 different books -/
def num_books : ℕ := 6

/-- There are 5 different boxes -/
def num_boxes : ℕ := 5

/-- Theorem: The number of ways to place 6 different books into 5 different boxes is 5^6 -/
theorem books_in_boxes : arrangements num_books num_boxes = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_books_in_boxes_l376_37660


namespace NUMINAMATH_CALUDE_equation_solution_existence_l376_37623

theorem equation_solution_existence (a : ℝ) :
  (∃ x : ℝ, 3 * 4^(x - 2) + 27 = a + a * 4^(x - 2)) ↔ 3 < a ∧ a < 27 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_existence_l376_37623


namespace NUMINAMATH_CALUDE_no_solution_iff_a_equals_two_l376_37690

theorem no_solution_iff_a_equals_two (a : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (a * x) / (x - 1) + 3 / (1 - x) ≠ 2) ↔ a = 2 := by
sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_equals_two_l376_37690


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_a5_l376_37607

theorem geometric_sequence_minimum_a5 (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) (h_diff : a 2 - a 1 = 1) :
  (∃ q : ℝ, q > 0 ∧ ∀ n, a n = a 1 * q^(n - 1)) →
  (∃ min_a5 : ℝ, ∀ q : ℝ, q > 0 → a 1 * q^4 ≥ min_a5) →
  (∀ n, a n = 3 * (4/3)^(n - 1)) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_a5_l376_37607


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l376_37661

theorem quadratic_solution_sum (c d : ℝ) : 
  (5 * (c + d * I) ^ 2 + 4 * (c + d * I) + 20 = 0) ∧ 
  (5 * (c - d * I) ^ 2 + 4 * (c - d * I) + 20 = 0) →
  c + d ^ 2 = 86 / 25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l376_37661


namespace NUMINAMATH_CALUDE_range_of_a_l376_37681

theorem range_of_a (a : ℝ) : 
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) → a ∈ Set.Iic (-1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l376_37681


namespace NUMINAMATH_CALUDE_min_f_tetrahedron_l376_37674

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron ABCD -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Distance between two points -/
def distance (p q : Point3D) : ℝ := sorry

/-- Function f(P) for a given tetrahedron and point P -/
def f (t : Tetrahedron) (P : Point3D) : ℝ :=
  distance P t.A + distance P t.B + distance P t.C + distance P t.D

/-- Theorem: Minimum value of f(P) for a tetrahedron with given properties -/
theorem min_f_tetrahedron (t : Tetrahedron) (a b c : ℝ) :
  (distance t.A t.D = a) →
  (distance t.B t.C = a) →
  (distance t.A t.C = b) →
  (distance t.B t.D = b) →
  (distance t.A t.B * distance t.C t.D = c^2) →
  ∃ (min_val : ℝ), (∀ (P : Point3D), f t P ≥ min_val) ∧ (min_val = Real.sqrt ((a^2 + b^2 + c^2) / 2)) :=
sorry

end NUMINAMATH_CALUDE_min_f_tetrahedron_l376_37674


namespace NUMINAMATH_CALUDE_ball_count_in_jar_l376_37668

/-- Given a jar with white and red balls in the ratio of 3:2, 
    if there are 9 white balls, then there are 6 red balls. -/
theorem ball_count_in_jar (white_balls red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 3 / 2 →
  white_balls = 9 →
  red_balls = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_in_jar_l376_37668


namespace NUMINAMATH_CALUDE_power_sum_integer_l376_37671

theorem power_sum_integer (a : ℝ) (h : ∃ k : ℤ, a + 1 / a = k) :
  ∀ n : ℕ, ∃ m : ℤ, a^n + 1 / a^n = m :=
by sorry

end NUMINAMATH_CALUDE_power_sum_integer_l376_37671


namespace NUMINAMATH_CALUDE_speed_increase_from_weight_cut_l376_37695

/-- Proves that the speed increase from weight cut is 10 mph given the initial conditions --/
theorem speed_increase_from_weight_cut 
  (original_speed : ℝ) 
  (supercharge_increase_percent : ℝ)
  (final_speed : ℝ) :
  original_speed = 150 →
  supercharge_increase_percent = 30 →
  final_speed = 205 →
  final_speed - (original_speed * (1 + supercharge_increase_percent / 100)) = 10 := by
sorry

end NUMINAMATH_CALUDE_speed_increase_from_weight_cut_l376_37695


namespace NUMINAMATH_CALUDE_largest_n_for_unique_k_l376_37658

theorem largest_n_for_unique_k : 
  ∀ n : ℕ, n > 112 → 
  ¬(∃! k : ℤ, (7 : ℚ)/16 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/17) ∧ 
  (∃! k : ℤ, (7 : ℚ)/16 < (112 : ℚ)/(112 + k) ∧ (112 : ℚ)/(112 + k) < 8/17) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_unique_k_l376_37658


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l376_37655

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_third_term (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 2 + a 3 + a 4 + a 5 = 20 →
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l376_37655


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_tank_l376_37612

/-- The area of a stripe painted on a cylindrical tank -/
theorem stripe_area_on_cylindrical_tank 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 40)
  (h2 : stripe_width = 4)
  (h3 : revolutions = 3) : 
  stripe_width * (Real.pi * diameter * revolutions) = 480 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_tank_l376_37612


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_l376_37637

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents income levels -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Community structure -/
structure Community where
  totalFamilies : Nat
  highIncomeFamilies : Nat
  middleIncomeFamilies : Nat
  lowIncomeFamilies : Nat

/-- Sample size for family survey -/
def familySampleSize : Nat := 100

/-- Student selection parameters -/
structure StudentSelection where
  totalStudents : Nat
  studentsToSelect : Nat

/-- Function to determine the optimal sampling method for family survey -/
def optimalFamilySamplingMethod (c : Community) : SamplingMethod := sorry

/-- Function to determine the optimal sampling method for student selection -/
def optimalStudentSamplingMethod (s : StudentSelection) : SamplingMethod := sorry

/-- Theorem stating the optimal sampling methods for the given scenario -/
theorem optimal_sampling_methods 
  (community : Community)
  (studentSelection : StudentSelection)
  (h1 : community.totalFamilies = 800)
  (h2 : community.highIncomeFamilies = 200)
  (h3 : community.middleIncomeFamilies = 480)
  (h4 : community.lowIncomeFamilies = 120)
  (h5 : studentSelection.totalStudents = 10)
  (h6 : studentSelection.studentsToSelect = 3) :
  optimalFamilySamplingMethod community = SamplingMethod.Stratified ∧
  optimalStudentSamplingMethod studentSelection = SamplingMethod.SimpleRandom := by
  sorry


end NUMINAMATH_CALUDE_optimal_sampling_methods_l376_37637


namespace NUMINAMATH_CALUDE_consecutive_sum_18_l376_37604

def consecutive_sum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

theorem consecutive_sum_18 :
  ∃! (start length : ℕ), 2 ≤ length ∧ consecutive_sum start length = 18 :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_18_l376_37604


namespace NUMINAMATH_CALUDE_lattice_points_count_l376_37685

/-- The number of lattice points on a line segment --/
def countLatticePoints (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (4,19) to (39,239) is 6 --/
theorem lattice_points_count :
  countLatticePoints 4 19 39 239 = 6 := by sorry

end NUMINAMATH_CALUDE_lattice_points_count_l376_37685


namespace NUMINAMATH_CALUDE_triangle_area_l376_37651

theorem triangle_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_angle : Real.cos (30 * π / 180) = c / (2 * a)) (h_side : b = 8) : 
  (1/2) * a * b = 32 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l376_37651


namespace NUMINAMATH_CALUDE_product_nonnegative_proof_l376_37614

theorem product_nonnegative_proof :
  -- Original proposition
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0) ∧
  -- Contrapositive is true
  (∀ x y : ℝ, x * y < 0 → x < 0 ∨ y < 0) ∧
  -- Converse is false
  ¬(∀ x y : ℝ, x * y ≥ 0 → x ≥ 0 ∧ y ≥ 0) ∧
  -- Negation is false
  ¬(∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ x * y < 0) := by
sorry

end NUMINAMATH_CALUDE_product_nonnegative_proof_l376_37614


namespace NUMINAMATH_CALUDE_tan_fifteen_to_sqrt_three_l376_37689

theorem tan_fifteen_to_sqrt_three : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_to_sqrt_three_l376_37689


namespace NUMINAMATH_CALUDE_portias_school_size_l376_37662

-- Define variables for the number of students in each school
variable (L : ℕ) -- Lara's high school
variable (P : ℕ) -- Portia's high school
variable (M : ℕ) -- Mia's high school

-- Define the conditions
axiom portia_students : P = 4 * L
axiom mia_students : M = 2 * L
axiom total_students : P + L + M = 4200

-- Theorem to prove
theorem portias_school_size : P = 2400 := by
  sorry

end NUMINAMATH_CALUDE_portias_school_size_l376_37662


namespace NUMINAMATH_CALUDE_translator_selection_ways_l376_37654

-- Define the staff members
def total_staff : ℕ := 7
def english_only : ℕ := 3
def japanese_only : ℕ := 2
def bilingual : ℕ := 2

-- Define the required translators
def english_translators : ℕ := 3
def japanese_translators : ℕ := 2

-- Define the function to calculate the number of ways to select translators
def select_translators : ℕ := 27

-- Theorem statement
theorem translator_selection_ways :
  select_translators = 27 :=
sorry

end NUMINAMATH_CALUDE_translator_selection_ways_l376_37654


namespace NUMINAMATH_CALUDE_pretty_numbers_characterization_l376_37688

def is_pretty (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ k ℓ : ℕ, 0 < k ∧ k < n ∧ 0 < ℓ ∧ ℓ < n ∧ k ∣ n ∧ ℓ ∣ n →
    (2 * k - ℓ) ∣ n ∨ (2 * ℓ - k) ∣ n

theorem pretty_numbers_characterization (n : ℕ) :
  is_pretty n ↔ Nat.Prime n ∨ n = 6 ∨ n = 9 ∨ n = 15 :=
sorry

end NUMINAMATH_CALUDE_pretty_numbers_characterization_l376_37688


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l376_37620

theorem quadratic_equation_integer_roots (m : ℕ) (a : ℝ) :
  (1 ≤ m) →
  (m ≤ 50) →
  (∃ x₁ x₂ : ℕ, 
    x₁ ≠ x₂ ∧
    (x₁ - 2)^2 + (a - m)^2 = 2 * m * x₁ + a^2 - 2 * a * m ∧
    (x₂ - 2)^2 + (a - m)^2 = 2 * m * x₂ + a^2 - 2 * a * m) →
  ∃ k : ℕ, m = k^2 ∧ k^2 ≤ 49 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l376_37620


namespace NUMINAMATH_CALUDE_problem_l376_37696

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem problem (a : ℝ) : f a (f a 0) = 4*a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_l376_37696


namespace NUMINAMATH_CALUDE_boxes_in_marker_carton_l376_37633

def pencil_cartons : ℕ := 20
def pencil_boxes_per_carton : ℕ := 10
def pencil_box_cost : ℕ := 2
def marker_cartons : ℕ := 10
def marker_carton_cost : ℕ := 4
def total_spent : ℕ := 600

theorem boxes_in_marker_carton :
  ∃ (x : ℕ), 
    x * marker_carton_cost * marker_cartons + 
    pencil_cartons * pencil_boxes_per_carton * pencil_box_cost = 
    total_spent ∧ 
    x = 5 := by sorry

end NUMINAMATH_CALUDE_boxes_in_marker_carton_l376_37633


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l376_37686

def A : Set ℝ := {0, 1, 2, 3, 4}

def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = {0, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l376_37686


namespace NUMINAMATH_CALUDE_fourth_grade_classroom_count_l376_37647

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 22

/-- The number of pet hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 3

/-- The number of pet guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 1

/-- The difference between the total number of students and the total number of pets -/
def student_pet_difference : ℕ := 90

theorem fourth_grade_classroom_count :
  num_classrooms * students_per_classroom - 
  num_classrooms * (hamsters_per_classroom + guinea_pigs_per_classroom) = 
  student_pet_difference := by sorry

end NUMINAMATH_CALUDE_fourth_grade_classroom_count_l376_37647


namespace NUMINAMATH_CALUDE_symphony_orchestra_members_l376_37624

theorem symphony_orchestra_members : ∃! n : ℕ,
  200 < n ∧ n < 300 ∧
  n % 6 = 2 ∧
  n % 8 = 3 ∧
  n % 9 = 4 ∧
  n = 260 := by
sorry

end NUMINAMATH_CALUDE_symphony_orchestra_members_l376_37624


namespace NUMINAMATH_CALUDE_inequality_not_holding_l376_37666

theorem inequality_not_holding (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  (1/a)^(1/b) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_holding_l376_37666


namespace NUMINAMATH_CALUDE_negation_equivalence_l376_37682

variable (m : ℝ)

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - m*x - m < 0) ↔ (∀ x : ℝ, x^2 - m*x - m ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l376_37682


namespace NUMINAMATH_CALUDE_wood_measurement_problem_l376_37676

theorem wood_measurement_problem (x y : ℝ) :
  (x + 4.5 = y ∧ x + 1 = (1/2) * y) ↔
  (∃ (wood_length rope_length : ℝ),
    wood_length = x ∧
    rope_length = y ∧
    wood_length + 4.5 = rope_length ∧
    wood_length + 1 = (1/2) * rope_length) :=
by sorry

end NUMINAMATH_CALUDE_wood_measurement_problem_l376_37676


namespace NUMINAMATH_CALUDE_margo_age_in_three_years_l376_37663

/-- Margo's age in three years given Benjie's current age and their age difference -/
def margos_future_age (benjies_age : ℕ) (age_difference : ℕ) : ℕ :=
  (benjies_age - age_difference) + 3

theorem margo_age_in_three_years :
  margos_future_age 6 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_margo_age_in_three_years_l376_37663


namespace NUMINAMATH_CALUDE_officer_selection_l376_37694

theorem officer_selection (n m : ℕ) (hn : n = 5) (hm : m = 3) :
  (n.choose m) * m.factorial = 60 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_l376_37694


namespace NUMINAMATH_CALUDE_tripodasaurus_count_l376_37630

/-- A tripodasaurus is a creature with 3 legs and 1 head -/
structure Tripodasaurus where
  legs : Nat
  head : Nat

/-- A flock of tripodasauruses -/
structure Flock where
  count : Nat

/-- The total number of heads and legs in a flock -/
def totalHeadsAndLegs (f : Flock) : Nat :=
  f.count * (3 + 1)  -- 3 legs + 1 head per tripodasaurus

theorem tripodasaurus_count (f : Flock) :
  totalHeadsAndLegs f = 20 → f.count = 5 := by
  sorry

end NUMINAMATH_CALUDE_tripodasaurus_count_l376_37630


namespace NUMINAMATH_CALUDE_bears_in_stock_calculation_l376_37699

/-- Calculates the number of bears in stock before a new shipment arrived -/
def bears_in_stock_before_shipment (new_shipment : ℕ) (bears_per_shelf : ℕ) (num_shelves : ℕ) : ℕ :=
  num_shelves * bears_per_shelf - new_shipment

theorem bears_in_stock_calculation (new_shipment : ℕ) (bears_per_shelf : ℕ) (num_shelves : ℕ) :
  bears_in_stock_before_shipment new_shipment bears_per_shelf num_shelves =
  num_shelves * bears_per_shelf - new_shipment :=
by
  sorry

end NUMINAMATH_CALUDE_bears_in_stock_calculation_l376_37699


namespace NUMINAMATH_CALUDE_basketball_game_theorem_l376_37677

/-- Represents the score of a team for each quarter -/
structure GameScore :=
  (q1 q2 q3 q4 : ℚ)

/-- Checks if a GameScore forms a geometric sequence with ratio 3 -/
def isGeometricSequence (score : GameScore) : Prop :=
  score.q2 = 3 * score.q1 ∧ score.q3 = 3 * score.q2 ∧ score.q4 = 3 * score.q3

/-- Checks if a GameScore forms an arithmetic sequence with difference 12 -/
def isArithmeticSequence (score : GameScore) : Prop :=
  score.q2 = score.q1 + 12 ∧ score.q3 = score.q2 + 12 ∧ score.q4 = score.q3 + 12

/-- Calculates the total score for a GameScore -/
def totalScore (score : GameScore) : ℚ :=
  score.q1 + score.q2 + score.q3 + score.q4

/-- Calculates the first half score for a GameScore -/
def firstHalfScore (score : GameScore) : ℚ :=
  score.q1 + score.q2

theorem basketball_game_theorem (eagles lions : GameScore) : 
  eagles.q1 = lions.q1 →  -- Tied at the end of first quarter
  isGeometricSequence eagles →
  isArithmeticSequence lions →
  totalScore eagles = totalScore lions + 3 →  -- Eagles won by 3 points
  totalScore eagles ≤ 120 →
  totalScore lions ≤ 120 →
  firstHalfScore eagles + firstHalfScore lions = 15 :=
by sorry

end NUMINAMATH_CALUDE_basketball_game_theorem_l376_37677


namespace NUMINAMATH_CALUDE_triangle_quadratic_no_solution_l376_37693

theorem triangle_quadratic_no_solution (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b^2 + c^2 - a^2)^2 - 4*(b^2)*(c^2) < 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_quadratic_no_solution_l376_37693


namespace NUMINAMATH_CALUDE_hares_log_cutting_l376_37609

/-- The number of cuts required to create n pieces from a single log -/
def num_cuts (n : ℕ) : ℕ := n - 1

/-- Theorem: The number of cuts required to create 12 pieces from a single log is 11 -/
theorem hares_log_cutting :
  ∃ (n : ℕ), n = 12 ∧ num_cuts n = 11 :=
sorry

end NUMINAMATH_CALUDE_hares_log_cutting_l376_37609


namespace NUMINAMATH_CALUDE_noodle_purchase_problem_l376_37667

theorem noodle_purchase_problem :
  -- First scenario
  let total_cost : ℕ := 3000
  let total_portions : ℕ := 170
  let mixed_sauce_price : ℕ := 15
  let beef_price : ℕ := 20
  -- Second scenario
  let mixed_sauce_cost : ℕ := 1260
  let beef_cost : ℕ := 1200
  let mixed_sauce_ratio : ℚ := 3/2
  let price_difference : ℕ := 6

  -- First scenario solution
  ∃ (mixed_sauce_portions beef_portions : ℕ),
    mixed_sauce_portions + beef_portions = total_portions ∧
    mixed_sauce_portions * mixed_sauce_price + beef_portions * beef_price = total_cost ∧
    mixed_sauce_portions = 80 ∧
    beef_portions = 90 ∧

  -- Second scenario solution
  ∃ (beef_portions_2 : ℕ),
    (mixed_sauce_ratio * beef_portions_2 : ℚ) * (beef_cost / beef_portions_2 - price_difference : ℚ) = mixed_sauce_cost ∧
    beef_portions_2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_noodle_purchase_problem_l376_37667


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l376_37606

theorem adult_ticket_cost (child_ticket_cost : ℝ) (total_tickets : ℕ) (total_revenue : ℝ) (child_tickets : ℕ) : ℝ :=
  let adult_tickets := total_tickets - child_tickets
  let child_revenue := child_ticket_cost * child_tickets
  let adult_revenue := total_revenue - child_revenue
  let adult_ticket_cost := adult_revenue / adult_tickets
  by
    have h1 : child_ticket_cost = 4.5 := by sorry
    have h2 : total_tickets = 400 := by sorry
    have h3 : total_revenue = 2100 := by sorry
    have h4 : child_tickets = 200 := by sorry
    sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l376_37606


namespace NUMINAMATH_CALUDE_chi_square_greater_than_critical_l376_37697

/-- Represents the contingency table data --/
structure ContingencyTable where
  total_sample : ℕ
  disease_probability : ℚ
  blue_collar_with_disease : ℕ
  white_collar_without_disease : ℕ

/-- Calculates the chi-square value for the given contingency table --/
def calculate_chi_square (table : ContingencyTable) : ℚ :=
  let white_collar_with_disease := table.total_sample * table.disease_probability - table.blue_collar_with_disease
  let blue_collar_without_disease := table.total_sample * (1 - table.disease_probability) - table.white_collar_without_disease
  let n := table.total_sample
  let a := white_collar_with_disease
  let b := table.white_collar_without_disease
  let c := table.blue_collar_with_disease
  let d := blue_collar_without_disease
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for α = 0.005 --/
def critical_value : ℚ := 7879 / 1000

/-- Theorem stating that the calculated chi-square value is greater than the critical value --/
theorem chi_square_greater_than_critical (table : ContingencyTable) 
  (h1 : table.total_sample = 50)
  (h2 : table.disease_probability = 3/5)
  (h3 : table.blue_collar_with_disease = 10)
  (h4 : table.white_collar_without_disease = 5) :
  calculate_chi_square table > critical_value :=
sorry

end NUMINAMATH_CALUDE_chi_square_greater_than_critical_l376_37697


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l376_37600

theorem unique_four_digit_number : ∃! n : ℕ,
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n % 1000 = b^3) ∧
  (∃ c : ℕ, n % 100 = c^4) ∧
  n = 9216 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l376_37600


namespace NUMINAMATH_CALUDE_upper_bound_necessary_not_sufficient_l376_37615

variable {α : Type*} [PartialOrder α]
variable (I : Set α) (f : α → ℝ) (M : ℝ)

def is_upper_bound (f : α → ℝ) (M : ℝ) (I : Set α) : Prop :=
  ∀ x ∈ I, f x ≤ M

def is_maximum (f : α → ℝ) (M : ℝ) (I : Set α) : Prop :=
  (is_upper_bound f M I) ∧ (∃ x ∈ I, f x = M)

theorem upper_bound_necessary_not_sufficient :
  (is_upper_bound f M I → is_maximum f M I) ∧
  ¬(is_maximum f M I → is_upper_bound f M I) :=
sorry

end NUMINAMATH_CALUDE_upper_bound_necessary_not_sufficient_l376_37615


namespace NUMINAMATH_CALUDE_reflection_count_theorem_l376_37691

/-- Represents a semicircular room -/
structure SemicircularRoom where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a light beam -/
structure LightBeam where
  start : ℝ × ℝ
  angle : ℝ

/-- Counts the number of reflections before the light beam returns to its starting point -/
def count_reflections (room : SemicircularRoom) (beam : LightBeam) : ℕ :=
  sorry

/-- The main theorem stating the number of reflections -/
theorem reflection_count_theorem (room : SemicircularRoom) (beam : LightBeam) :
  room.center = (0, 0) →
  room.radius = 1 →
  beam.start = (-1, 0) →
  beam.angle = 46 * π / 180 →
  count_reflections room beam = 65 :=
sorry

end NUMINAMATH_CALUDE_reflection_count_theorem_l376_37691


namespace NUMINAMATH_CALUDE_concave_function_triangle_inequality_l376_37625

def f (x : ℝ) := x^2 - 2*x + 2

theorem concave_function_triangle_inequality (m : ℝ) : 
  (∀ a b c : ℝ, 1/3 ≤ a ∧ a < b ∧ b < c ∧ c ≤ m^2 - m + 2 → 
    f a + f b > f c ∧ f b + f c > f a ∧ f c + f a > f b) ↔ 
  0 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_concave_function_triangle_inequality_l376_37625


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l376_37616

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  let circumference := π * r
  let base_radius := circumference / (2 * π)
  let slant_height := r
  let cone_height := Real.sqrt (slant_height^2 - base_radius^2)
  let volume := (1/3) * π * base_radius^2 * cone_height
  volume = 9 * π * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l376_37616


namespace NUMINAMATH_CALUDE_b_rent_exceeds_total_cost_l376_37639

/-- Represents the rent rates for different animals -/
structure RentRates where
  horse : ℕ
  cow : ℕ
  sheep : ℕ
  goat : ℕ

/-- Represents the animals and duration for a renter -/
structure RenterAnimals where
  horses : ℕ
  horseDuration : ℕ
  sheep : ℕ
  sheepDuration : ℕ
  goats : ℕ
  goatDuration : ℕ

/-- Calculates the total rent for a renter given their animals and rent rates -/
def calculateRent (animals : RenterAnimals) (rates : RentRates) : ℕ :=
  animals.horses * animals.horseDuration * rates.horse +
  animals.sheep * animals.sheepDuration * rates.sheep +
  animals.goats * animals.goatDuration * rates.goat

/-- The total cost of the pasture -/
def totalPastureCost : ℕ := 5820

/-- The rent rates for different animals -/
def givenRates : RentRates :=
  { horse := 30
    cow := 40
    sheep := 20
    goat := 25 }

/-- B's animals and their durations -/
def bAnimals : RenterAnimals :=
  { horses := 16
    horseDuration := 9
    sheep := 18
    sheepDuration := 7
    goats := 4
    goatDuration := 6 }

theorem b_rent_exceeds_total_cost :
  calculateRent bAnimals givenRates > totalPastureCost := by
  sorry

end NUMINAMATH_CALUDE_b_rent_exceeds_total_cost_l376_37639


namespace NUMINAMATH_CALUDE_product_expansion_l376_37664

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l376_37664


namespace NUMINAMATH_CALUDE_alex_grocery_delivery_l376_37632

/-- Represents the problem of calculating the total value of groceries Alex delivered --/
theorem alex_grocery_delivery 
  (savings : ℝ) 
  (car_cost : ℝ) 
  (trip_charge : ℝ) 
  (grocery_charge_percent : ℝ) 
  (num_trips : ℕ) 
  (h1 : savings = 14500) 
  (h2 : car_cost = 14600) 
  (h3 : trip_charge = 1.5) 
  (h4 : grocery_charge_percent = 0.05) 
  (h5 : num_trips = 40) 
  (h6 : savings + num_trips * trip_charge + grocery_charge_percent * (car_cost - savings - num_trips * trip_charge) / grocery_charge_percent ≥ car_cost) : 
  (car_cost - savings - num_trips * trip_charge) / grocery_charge_percent = 800 := by
  sorry

end NUMINAMATH_CALUDE_alex_grocery_delivery_l376_37632


namespace NUMINAMATH_CALUDE_min_sum_of_two_digits_is_one_l376_37617

/-- A digit is a natural number from 0 to 9 -/
def Digit := { n : ℕ // n ≤ 9 }

/-- The theorem states that the minimum sum of two digits P and Q is 1,
    given that P, Q, R, and S are four different digits,
    and (P+Q)/(R+S) is an integer and as small as possible. -/
theorem min_sum_of_two_digits_is_one
  (P Q R S : Digit)
  (h_distinct : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S)
  (h_integer : ∃ k : ℕ, (P.val + Q.val : ℚ) / (R.val + S.val) = k)
  (h_min : ∀ (P' Q' R' S' : Digit),
           P' ≠ Q' ∧ P' ≠ R' ∧ P' ≠ S' ∧ Q' ≠ R' ∧ Q' ≠ S' ∧ R' ≠ S' →
           (∃ k : ℕ, (P'.val + Q'.val : ℚ) / (R'.val + S'.val) = k) →
           (P.val + Q.val : ℚ) / (R.val + S.val) ≤ (P'.val + Q'.val : ℚ) / (R'.val + S'.val)) :
  P.val + Q.val = 1 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_two_digits_is_one_l376_37617


namespace NUMINAMATH_CALUDE_sin_double_angle_on_line_l376_37611

theorem sin_double_angle_on_line (θ : Real) :
  (∃ (x y : Real), y = 3 * x ∧ x = Real.cos θ ∧ y = Real.sin θ) →
  Real.sin (2 * θ) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_on_line_l376_37611


namespace NUMINAMATH_CALUDE_part_i_part_ii_l376_37650

-- Define the function f
def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

-- Part I
theorem part_i : 
  let m : ℝ := -1
  let a : ℝ := -1
  {x : ℝ | f x a m ≥ x} = {x : ℝ | x ≤ -2 ∨ (0 ≤ x ∧ x ≤ 2)} := by sorry

-- Part II
theorem part_ii (m : ℝ) (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∀ x : ℝ, f x a m ≥ 2) 
  (h4 : a ≤ -3 ∨ a ≥ 3) : 
  m = 1/3 := by sorry

end NUMINAMATH_CALUDE_part_i_part_ii_l376_37650


namespace NUMINAMATH_CALUDE_platform_length_l376_37644

/-- Given a train of length 450 m, running at 108 kmph, crosses a platform in 25 seconds,
    prove that the length of the platform is 300 m. -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 450 ∧ 
  train_speed_kmph = 108 ∧ 
  crossing_time = 25 →
  (train_speed_kmph * 1000 / 3600 * crossing_time - train_length) = 300 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l376_37644


namespace NUMINAMATH_CALUDE_inequality_solution_range_l376_37687

/-- Given that the inequality |x+a|+|x-1|+a>2009 (where a is a constant) has a non-empty set of solutions, 
    the range of values for a is (-∞, 1004) -/
theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + a| + |x - 1| + a > 2009) → 
  a ∈ Set.Iio 1004 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l376_37687


namespace NUMINAMATH_CALUDE_greatest_four_digit_number_l376_37653

theorem greatest_four_digit_number : ∃ (n : ℕ), 
  (n = 9997) ∧ 
  (n < 10000) ∧ 
  (∃ (k : ℕ), n = 7 * k + 1) ∧ 
  (∃ (j : ℕ), n = 8 * j + 5) ∧ 
  (∀ (m : ℕ), m < 10000 → (∃ (k : ℕ), m = 7 * k + 1) → (∃ (j : ℕ), m = 8 * j + 5) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_number_l376_37653


namespace NUMINAMATH_CALUDE_y_divisibility_l376_37643

theorem y_divisibility : ∃ k : ℕ, 
  (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) = 4 * k ∧ 
  (∃ m : ℕ, (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) ≠ 8 * m ∨ 
            (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) ≠ 16 * m ∨ 
            (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) ≠ 32 * m) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l376_37643


namespace NUMINAMATH_CALUDE_tank_capacity_l376_37610

/-- Represents a cylindrical water tank --/
structure WaterTank where
  capacity : ℝ
  currentPercentage : ℝ
  currentVolume : ℝ

/-- Theorem: A cylindrical tank that is 25% full with 60 liters has a total capacity of 240 liters --/
theorem tank_capacity (tank : WaterTank) 
  (h1 : tank.currentPercentage = 0.25)
  (h2 : tank.currentVolume = 60) : 
  tank.capacity = 240 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l376_37610


namespace NUMINAMATH_CALUDE_olivers_score_l376_37634

theorem olivers_score (n : ℕ) (avg_24 : ℚ) (avg_25 : ℚ) (oliver_score : ℚ) :
  n = 25 →
  avg_24 = 76 →
  avg_25 = 78 →
  (n - 1) * avg_24 + oliver_score = n * avg_25 →
  oliver_score = 126 := by
  sorry

end NUMINAMATH_CALUDE_olivers_score_l376_37634


namespace NUMINAMATH_CALUDE_parabola_points_theorem_l376_37613

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions
def satisfies_conditions (A B : Point) : Prop :=
  A.y = parabola A.x ∧ 
  B.y = parabola B.x ∧ 
  A.x < 0 ∧ 
  B.x > 0 ∧
  A.y > B.y

-- Define the theorem
theorem parabola_points_theorem (A B : Point) (h : satisfies_conditions A B) :
  (A.x = -4 ∧ B.x = 2) ∨ (A.x = 4 ∧ B.x = -2) :=
sorry

end NUMINAMATH_CALUDE_parabola_points_theorem_l376_37613


namespace NUMINAMATH_CALUDE_last_date_theorem_l376_37638

/-- Represents a date in DD.MM.YYYY format -/
structure Date :=
  (day : Nat)
  (month : Nat)
  (year : Nat)

/-- Check if a date is valid -/
def is_valid_date (d : Date) : Bool :=
  d.day ≥ 1 && d.day ≤ 31 && d.month ≥ 1 && d.month ≤ 12 && d.year ≥ 1

/-- Get the set of digits used in a date -/
def date_digits (d : Date) : Finset Nat :=
  sorry

/-- Check if a date is before another date -/
def is_before (d1 d2 : Date) : Bool :=
  sorry

/-- Find the last date before a given date with the same set of digits -/
def last_date_with_same_digits (d : Date) : Date :=
  sorry

theorem last_date_theorem (current_date : Date) :
  let target_date := Date.mk 15 12 2012
  current_date = Date.mk 22 11 2015 →
  is_valid_date target_date ∧
  is_before target_date current_date ∧
  date_digits target_date = date_digits current_date ∧
  (∀ d : Date, is_valid_date d ∧ is_before d current_date ∧ date_digits d = date_digits current_date →
    is_before d target_date ∨ d = target_date) :=
by sorry

end NUMINAMATH_CALUDE_last_date_theorem_l376_37638


namespace NUMINAMATH_CALUDE_B_equals_l376_37684

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define sets A and B
variable (A B : Set Nat)

-- State the conditions
axiom union_eq : A ∪ B = U
axiom intersection_eq : A ∩ (U \ B) = {2, 4, 6}

-- Theorem to prove
theorem B_equals : B = {1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_B_equals_l376_37684


namespace NUMINAMATH_CALUDE_inequality_solution_set_l376_37603

theorem inequality_solution_set (x : ℝ) : 
  (x^2 * (x + 1)) / (-x^2 - 5*x + 6) ≤ 0 ∧ (-x^2 - 5*x + 6) ≠ 0 ↔ 
  (-6 < x ∧ x ≤ -1) ∨ x = 0 ∨ x > 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l376_37603


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l376_37619

/-- The equation of a circle passing through points A(2, 0), B(4, 0), and C(1, 2) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - (7/2)*y + 8 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (2, 0)

/-- Point B -/
def point_B : ℝ × ℝ := (4, 0)

/-- Point C -/
def point_C : ℝ × ℝ := (1, 2)

/-- Theorem stating that the circle equation passes through points A, B, and C -/
theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l376_37619


namespace NUMINAMATH_CALUDE_pierre_ice_cream_scoops_l376_37631

/-- 
Given:
- The cost of each scoop of ice cream
- The number of scoops Pierre's mom gets
- The total bill amount
Prove that Pierre gets 3 scoops of ice cream
-/
theorem pierre_ice_cream_scoops 
  (cost_per_scoop : ℕ) 
  (mom_scoops : ℕ) 
  (total_bill : ℕ) 
  (h1 : cost_per_scoop = 2)
  (h2 : mom_scoops = 4)
  (h3 : total_bill = 14) :
  ∃ (pierre_scoops : ℕ), 
    pierre_scoops = 3 ∧ 
    cost_per_scoop * (pierre_scoops + mom_scoops) = total_bill :=
by sorry

end NUMINAMATH_CALUDE_pierre_ice_cream_scoops_l376_37631


namespace NUMINAMATH_CALUDE_bad_carrots_count_l376_37608

theorem bad_carrots_count (faye_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) :
  faye_carrots = 23 →
  mom_carrots = 5 →
  good_carrots = 12 →
  faye_carrots + mom_carrots - good_carrots = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l376_37608


namespace NUMINAMATH_CALUDE_solution_set_x_abs_x_leq_one_l376_37605

theorem solution_set_x_abs_x_leq_one (x : ℝ) : x * |x| ≤ 1 ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_x_abs_x_leq_one_l376_37605


namespace NUMINAMATH_CALUDE_part_i_part_ii_l376_37692

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.log x + 1 + 2 / x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 1 / x - 2 / (x^2)

-- Theorem for part I
theorem part_i :
  (∃ a : ℝ, a > 0 ∧ 
    (∀ x : ℝ, x > 0 → (f x - f a = -1/a * (x - a)) → (x = 0 → f x = 4)) ∧
    (∀ x : ℝ, x ∈ Set.Ioo 0 2 → f' x < 0) ∧
    (∀ x : ℝ, x > 2 → f' x > 0)) :=
sorry

-- Theorem for part II
theorem part_ii :
  (∃ k : ℤ, k = 7 ∧
    (∀ x : ℝ, x > 1 → 2 * f x > k * (1 - 1/x)) ∧
    (∀ m : ℤ, m > k → ∃ x : ℝ, x > 1 ∧ 2 * f x ≤ m * (1 - 1/x))) :=
sorry

end NUMINAMATH_CALUDE_part_i_part_ii_l376_37692


namespace NUMINAMATH_CALUDE_davids_english_marks_l376_37642

def marks_mathematics : ℕ := 85
def marks_physics : ℕ := 92
def marks_chemistry : ℕ := 87
def marks_biology : ℕ := 95
def average_marks : ℕ := 89
def number_of_subjects : ℕ := 5

theorem davids_english_marks :
  ∃ (marks_english : ℕ),
    marks_english +
    marks_mathematics +
    marks_physics +
    marks_chemistry +
    marks_biology =
    average_marks * number_of_subjects ∧
    marks_english = 86 := by
  sorry

end NUMINAMATH_CALUDE_davids_english_marks_l376_37642


namespace NUMINAMATH_CALUDE_system_equations_properties_l376_37670

theorem system_equations_properties (a x y : ℝ) 
  (eq1 : x + y = 1 - a) 
  (eq2 : x - y = 3 * a + 5) 
  (x_pos : x > 0) 
  (y_nonneg : y ≥ 0) : 
  (a = -5/3 → x = y) ∧ 
  (a = -2 → x + y = 5 + a) ∧ 
  (0 < x ∧ x ≤ 1 → 2 ≤ y ∧ y < 4) := by
  sorry

end NUMINAMATH_CALUDE_system_equations_properties_l376_37670


namespace NUMINAMATH_CALUDE_parallel_vectors_result_symmetric_function_range_l376_37635

-- Part 1
theorem parallel_vectors_result (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, Real.cos x)
  let b : ℝ × ℝ := (3, -1)
  (∃ (k : ℝ), a = k • b) →
  2 * (Real.sin x)^2 - 3 * (Real.cos x)^2 = 3/2 := by sorry

-- Part 2
theorem symmetric_function_range (x m : ℝ) :
  let a : ℝ → ℝ × ℝ := λ t => (Real.sin t, m * Real.cos t)
  let b : ℝ × ℝ := (3, -1)
  let f : ℝ → ℝ := λ t => (a t).1 * b.1 + (a t).2 * b.2
  (∀ t, f (2*π/3 - t) = f (2*π/3 + t)) →
  ∃ y ∈ Set.Icc (-Real.sqrt 3) (2 * Real.sqrt 3),
    ∃ x ∈ Set.Icc (π/8) (2*π/3), f (2*x) = y := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_result_symmetric_function_range_l376_37635


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l376_37628

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + ((m^2 - 3*m) : ℝ)*I) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l376_37628
