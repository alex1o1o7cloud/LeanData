import Mathlib

namespace NUMINAMATH_CALUDE_total_unique_customers_l4057_405789

/-- Represents the number of customers who had meals with both ham and cheese -/
def ham_cheese : ℕ := 80

/-- Represents the number of customers who had meals with both ham and tomatoes -/
def ham_tomato : ℕ := 90

/-- Represents the number of customers who had meals with both tomatoes and cheese -/
def tomato_cheese : ℕ := 100

/-- Represents the number of customers who had meals with all three ingredients -/
def all_three : ℕ := 20

/-- Theorem stating that the total number of unique customers is 230 -/
theorem total_unique_customers : 
  ham_cheese + ham_tomato + tomato_cheese - 2 * all_three = 230 := by
  sorry

end NUMINAMATH_CALUDE_total_unique_customers_l4057_405789


namespace NUMINAMATH_CALUDE_sine_cosine_increasing_interval_l4057_405792

theorem sine_cosine_increasing_interval :
  ∀ (a b : ℝ), (a = -π / 2 ∧ b = 0) ↔ 
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y)) ∧
    (¬(∀ x y, -π ≤ x ∧ x < y ∧ y ≤ -π / 2 → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y))) ∧
    (¬(∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 2 → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y))) ∧
    (¬(∀ x y, π / 2 ≤ x ∧ x < y ∧ y ≤ π → 
      (Real.sin x < Real.sin y ∧ Real.cos x < Real.cos y))) :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_increasing_interval_l4057_405792


namespace NUMINAMATH_CALUDE_middle_manager_sample_size_l4057_405790

/-- Calculates the number of middle-level managers to be sampled in a stratified sampling scenario -/
theorem middle_manager_sample_size (total_employees : ℕ) (middle_managers : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 1000)
  (h2 : middle_managers = 150)
  (h3 : sample_size = 200) :
  (middle_managers : ℚ) / (total_employees : ℚ) * (sample_size : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_middle_manager_sample_size_l4057_405790


namespace NUMINAMATH_CALUDE_water_depth_is_208_l4057_405738

/-- The depth of water given Ron's height -/
def water_depth (ron_height : ℝ) : ℝ := 16 * ron_height

/-- Ron's height in feet -/
def ron_height : ℝ := 13

/-- Theorem stating that the water depth is 208 feet -/
theorem water_depth_is_208 : water_depth ron_height = 208 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_is_208_l4057_405738


namespace NUMINAMATH_CALUDE_smallest_number_l4057_405720

theorem smallest_number (π : ℝ) (h : π > 0) : min (-π) (min (-2) (min 0 (Real.sqrt 3))) = -π := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l4057_405720


namespace NUMINAMATH_CALUDE_arithmetic_sequence_minimum_l4057_405779

theorem arithmetic_sequence_minimum (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (a 2 * a 12).sqrt = 4 →  -- Geometric mean of a_2 and a_12 is 4
  (∃ r : ℝ, ∀ n, a (n + 1) = a n + r) →  -- Arithmetic sequence
  (∃ m : ℝ, ∀ r : ℝ, 2 * a 5 + 8 * a 9 ≥ m) →  -- Minimum exists
  a 3 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_minimum_l4057_405779


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l4057_405786

/-- The parabola is defined by the equation y = x^2 - 3x - 4 -/
def parabola (x y : ℝ) : Prop := y = x^2 - 3*x - 4

/-- The y-axis is defined by x = 0 -/
def y_axis (x : ℝ) : Prop := x = 0

/-- Theorem: The intersection point of the parabola y = x^2 - 3x - 4 with the y-axis has coordinates (0, -4) -/
theorem parabola_y_axis_intersection :
  ∃ (x y : ℝ), parabola x y ∧ y_axis x ∧ x = 0 ∧ y = -4 :=
sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l4057_405786


namespace NUMINAMATH_CALUDE_total_flour_used_l4057_405708

-- Define the ratios
def cake_ratio : Fin 3 → ℕ
  | 0 => 3  -- flour
  | 1 => 2  -- butter
  | 2 => 1  -- sugar
  | _ => 0

def cream_ratio : Fin 2 → ℕ
  | 0 => 2  -- butter
  | 1 => 3  -- sugar
  | _ => 0

def cookie_ratio : Fin 3 → ℕ
  | 0 => 5  -- flour
  | 1 => 3  -- butter
  | 2 => 2  -- sugar
  | _ => 0

-- Define the additional flour
def additional_flour : ℕ := 300

-- Theorem statement
theorem total_flour_used (x y : ℕ) :
  (3 * x + additional_flour) / (2 * x + 2 * y) = 5 / 3 →
  (2 * x + 2 * y) / (x + 3 * y) = 3 / 2 →
  3 * x + additional_flour = 1200 :=
by sorry

end NUMINAMATH_CALUDE_total_flour_used_l4057_405708


namespace NUMINAMATH_CALUDE_periodic_function_l4057_405748

/-- A function f is periodic if there exists a non-zero real number p such that
    f(x + p) = f(x) for all x in the domain of f. -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

/-- The given conditions on function f -/
structure FunctionConditions (f : ℝ → ℝ) (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  (sum_neq : a₁ + b₁ ≠ a₂ + b₂)
  (cond : ∀ x : ℝ, (f (a₁ + x) = f (b₁ - x) ∧ f (a₂ + x) = f (b₂ - x)) ∨
                   (f (a₁ + x) = -f (b₁ - x) ∧ f (a₂ + x) = -f (b₂ - x)))

/-- The main theorem stating that f is periodic with the given period -/
theorem periodic_function (f : ℝ → ℝ) (a₁ b₁ a₂ b₂ : ℝ)
    (h : FunctionConditions f a₁ b₁ a₂ b₂) :
    IsPeriodic f ∧ ∃ p : ℝ, p = |((a₂ + b₂) - (a₁ + b₁))| ∧
    ∀ x : ℝ, f (x + p) = f x :=
  sorry


end NUMINAMATH_CALUDE_periodic_function_l4057_405748


namespace NUMINAMATH_CALUDE_set_equality_condition_l4057_405711

-- Define set A
def A : Set ℝ := {x | (x + 1)^2 * (2 - x) / (4 + x) ≥ 0 ∧ x ≠ -4}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 2*a + 1) ≤ 0}

-- Theorem statement
theorem set_equality_condition (a : ℝ) : 
  A ∪ B a = A ↔ -3/2 < a ∧ a ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_set_equality_condition_l4057_405711


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_l4057_405774

theorem necessary_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_l4057_405774


namespace NUMINAMATH_CALUDE_meeting_chair_rows_l4057_405766

/-- Calculates the number of rows of meeting chairs given the initial water amount,
    cup capacity, chairs per row, and remaining water. -/
theorem meeting_chair_rows
  (initial_water : ℕ)  -- Initial water in gallons
  (cup_capacity : ℕ)   -- Cup capacity in ounces
  (chairs_per_row : ℕ) -- Number of chairs per row
  (water_left : ℕ)     -- Water left after filling cups in ounces
  (h1 : initial_water = 3)
  (h2 : cup_capacity = 6)
  (h3 : chairs_per_row = 10)
  (h4 : water_left = 84)
  : ℕ := by
  sorry

#check meeting_chair_rows

end NUMINAMATH_CALUDE_meeting_chair_rows_l4057_405766


namespace NUMINAMATH_CALUDE_discount_percentage_l4057_405788

theorem discount_percentage (original_price sale_price : ℝ) 
  (h1 : original_price = 600)
  (h2 : sale_price = 480) : 
  (original_price - sale_price) / original_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l4057_405788


namespace NUMINAMATH_CALUDE_john_remaining_money_l4057_405728

/-- The amount of money John has left after purchasing pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let initial_money : ℝ := 50
  let drink_cost : ℝ := q
  let small_pizza_cost : ℝ := q
  let large_pizza_cost : ℝ := 4 * q
  let num_drinks : ℕ := 4
  let num_small_pizzas : ℕ := 2
  let num_large_pizzas : ℕ := 1
  initial_money - (num_drinks * drink_cost + num_small_pizzas * small_pizza_cost + num_large_pizzas * large_pizza_cost)

/-- Theorem stating that John's remaining money is equal to 50 - 10q -/
theorem john_remaining_money (q : ℝ) : money_left q = 50 - 10 * q := by
  sorry

end NUMINAMATH_CALUDE_john_remaining_money_l4057_405728


namespace NUMINAMATH_CALUDE_percentage_difference_l4057_405740

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.35)) :
  y = x * (1 + 0.35) := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l4057_405740


namespace NUMINAMATH_CALUDE_all_functions_have_clever_value_point_l4057_405764

-- Define the concept of a "clever value point"
def has_clever_value_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = deriv f x₀

-- State the theorem
theorem all_functions_have_clever_value_point :
  (has_clever_value_point (λ x : ℝ => x^2)) ∧
  (has_clever_value_point (λ x : ℝ => Real.exp (-x))) ∧
  (has_clever_value_point (λ x : ℝ => Real.log x)) ∧
  (has_clever_value_point (λ x : ℝ => Real.tan x)) :=
sorry

end NUMINAMATH_CALUDE_all_functions_have_clever_value_point_l4057_405764


namespace NUMINAMATH_CALUDE_equation_system_solution_l4057_405749

theorem equation_system_solution (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l4057_405749


namespace NUMINAMATH_CALUDE_exponent_product_rule_l4057_405712

theorem exponent_product_rule (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_product_rule_l4057_405712


namespace NUMINAMATH_CALUDE_rabbit_exchange_l4057_405776

/-- The exchange problem between two rabbits --/
theorem rabbit_exchange (white_carrots gray_cabbages : ℕ) 
  (h1 : white_carrots = 180) 
  (h2 : gray_cabbages = 120) : 
  ∃ (x : ℕ), x > 0 ∧ x < gray_cabbages ∧ 
  (gray_cabbages - x + 3 * x = (white_carrots + gray_cabbages) / 2) ∧
  (white_carrots - 3 * x + x = (white_carrots + gray_cabbages) / 2) := by
sorry

#eval (180 + 120) / 2  -- Expected output: 150

end NUMINAMATH_CALUDE_rabbit_exchange_l4057_405776


namespace NUMINAMATH_CALUDE_uncle_li_parking_duration_l4057_405707

/-- Calculates the parking duration given the total amount paid and the fee structure -/
def parking_duration (total_paid : ℚ) (first_hour_fee : ℚ) (additional_half_hour_fee : ℚ) : ℚ :=
  (total_paid - first_hour_fee) / (additional_half_hour_fee / (1/2)) + 1

theorem uncle_li_parking_duration :
  let total_paid : ℚ := 25/2
  let first_hour_fee : ℚ := 5/2
  let additional_half_hour_fee : ℚ := 5/2
  parking_duration total_paid first_hour_fee additional_half_hour_fee = 3 := by
sorry

end NUMINAMATH_CALUDE_uncle_li_parking_duration_l4057_405707


namespace NUMINAMATH_CALUDE_one_not_identity_for_star_l4057_405777

/-- The set of all non-zero real numbers -/
def S : Set ℝ := {x : ℝ | x ≠ 0}

/-- The binary operation * on S -/
def star (a b : ℝ) : ℝ := a^2 + 2*a*b

/-- Theorem: 1 is not an identity element for * in S -/
theorem one_not_identity_for_star :
  ¬(∀ a : ℝ, a ∈ S → (star 1 a = a ∧ star a 1 = a)) :=
sorry

end NUMINAMATH_CALUDE_one_not_identity_for_star_l4057_405777


namespace NUMINAMATH_CALUDE_product_of_solutions_l4057_405709

theorem product_of_solutions (x₁ x₂ : ℝ) : 
  (|5 * x₁ - 1| + 4 = 54) → 
  (|5 * x₂ - 1| + 4 = 54) → 
  x₁ ≠ x₂ →
  x₁ * x₂ = -99.96 := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l4057_405709


namespace NUMINAMATH_CALUDE_mustard_total_l4057_405756

theorem mustard_total (table1 table2 table3 : ℚ) 
  (h1 : table1 = 0.25)
  (h2 : table2 = 0.25)
  (h3 : table3 = 0.38) :
  table1 + table2 + table3 = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_mustard_total_l4057_405756


namespace NUMINAMATH_CALUDE_james_muffins_l4057_405703

theorem james_muffins (arthur_muffins : ℕ) (james_multiplier : ℕ) 
  (h1 : arthur_muffins = 115)
  (h2 : james_multiplier = 12) :
  arthur_muffins * james_multiplier = 1380 :=
by sorry

end NUMINAMATH_CALUDE_james_muffins_l4057_405703


namespace NUMINAMATH_CALUDE_expression_simplification_l4057_405724

theorem expression_simplification (a : ℝ) (h : a = 2 + Real.sqrt 2) :
  (a / (a + 2) + 1 / (a^2 - 4)) / ((a - 1) / (a + 2)) + 1 / (a - 2) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4057_405724


namespace NUMINAMATH_CALUDE_x0_in_N_l4057_405770

def M : Set ℝ := {x | ∃ k : ℤ, x = k + 1/2}
def N : Set ℝ := {x | ∃ k : ℤ, x = k/2 + 1}

theorem x0_in_N (x0 : ℝ) (h : x0 ∈ M) : x0 ∈ N := by
  sorry

end NUMINAMATH_CALUDE_x0_in_N_l4057_405770


namespace NUMINAMATH_CALUDE_currency_and_length_conversion_l4057_405704

/-- Conversion rate from jiao to yuan -/
def jiao_to_yuan : ℚ := 1 / 10

/-- Conversion rate from meters to centimeters -/
def meters_to_cm : ℚ := 100

/-- Convert yuan and jiao to yuan -/
def convert_to_yuan (yuan : ℚ) (jiao : ℚ) : ℚ :=
  yuan + jiao * jiao_to_yuan

/-- Convert units of 0.1 meters to centimeters -/
def convert_to_cm (units : ℚ) : ℚ :=
  units * 0.1 * meters_to_cm

theorem currency_and_length_conversion :
  (convert_to_yuan 5 5 = 5.05) ∧
  (convert_to_cm 12 = 120) := by
  sorry

end NUMINAMATH_CALUDE_currency_and_length_conversion_l4057_405704


namespace NUMINAMATH_CALUDE_mia_sixth_game_shots_l4057_405768

-- Define the initial conditions
def initial_shots : ℕ := 50
def initial_made : ℕ := 20
def new_shots : ℕ := 15

-- Define the function to calculate the new shooting average
def new_average (x : ℕ) : ℚ :=
  (initial_made + x : ℚ) / (initial_shots + new_shots : ℚ)

-- Theorem statement
theorem mia_sixth_game_shots :
  ∃ x : ℕ, x ≤ new_shots ∧ new_average x = 45 / 100 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_mia_sixth_game_shots_l4057_405768


namespace NUMINAMATH_CALUDE_married_men_fraction_l4057_405794

-- Define the faculty
structure Faculty where
  total : ℕ
  women : ℕ
  married : ℕ
  men : ℕ

-- Define the conditions
def faculty_conditions (f : Faculty) : Prop :=
  f.women = (70 * f.total) / 100 ∧
  f.married = (40 * f.total) / 100 ∧
  f.men = f.total - f.women

-- Define the fraction of single men
def single_men_fraction (f : Faculty) : ℚ :=
  1 / 3

-- Theorem to prove
theorem married_men_fraction (f : Faculty) 
  (h : faculty_conditions f) : 
  (f.married - (f.women - (f.total - f.married))) / f.men = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l4057_405794


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l4057_405730

def coin_distribution (x : ℕ) : ℕ × ℕ := 
  (x * (x + 1) / 2, x)

theorem pirate_treasure_distribution :
  ∃ x : ℕ, 
    let (bob_coins, sam_coins) := coin_distribution x
    bob_coins = 3 * sam_coins ∧ 
    bob_coins + sam_coins = 20 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l4057_405730


namespace NUMINAMATH_CALUDE_number_problem_l4057_405714

theorem number_problem :
  ∃ (x : ℝ), ∃ (y : ℝ), 0.5 * x = y + 20 ∧ x - 2 * y = 40 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4057_405714


namespace NUMINAMATH_CALUDE_locus_is_hyperbola_l4057_405745

/-- The locus of points equidistant from two fixed points is a hyperbola -/
theorem locus_is_hyperbola (P : ℝ × ℝ) :
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (4, 0)
  let dist (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  dist P F - dist P O = 1 →
  ∃ (a b : ℝ), (P.1 / a)^2 - (P.2 / b)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_locus_is_hyperbola_l4057_405745


namespace NUMINAMATH_CALUDE_line_through_points_l4057_405706

/-- Given a line y = ax + b passing through points (3,7) and (7,19), prove that a - b = 5 -/
theorem line_through_points (a b : ℝ) : 
  (3 * a + b = 7) → (7 * a + b = 19) → a - b = 5 := by sorry

end NUMINAMATH_CALUDE_line_through_points_l4057_405706


namespace NUMINAMATH_CALUDE_polygon_perimeter_is_52_l4057_405705

/-- The perimeter of a polygon formed by removing six 2x2 squares from the sides of an 8x12 rectangle -/
def polygon_perimeter (rectangle_length : ℕ) (rectangle_width : ℕ) (square_side : ℕ) (num_squares : ℕ) : ℕ :=
  2 * (rectangle_length + rectangle_width) + 2 * num_squares * square_side

theorem polygon_perimeter_is_52 :
  polygon_perimeter 12 8 2 6 = 52 := by
  sorry

end NUMINAMATH_CALUDE_polygon_perimeter_is_52_l4057_405705


namespace NUMINAMATH_CALUDE_min_surface_area_large_solid_l4057_405763

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surface_area (d : Dimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- The dimensions of the smaller rectangular solids -/
def small_solid : Dimensions :=
  { length := 3, width := 4, height := 5 }

/-- The number of smaller rectangular solids -/
def num_small_solids : ℕ := 24

/-- Theorem stating the minimum surface area of the large rectangular solid -/
theorem min_surface_area_large_solid :
  ∃ (d : Dimensions), surface_area d = 788 ∧
  (∀ (d' : Dimensions), surface_area d' ≥ surface_area d) := by
  sorry


end NUMINAMATH_CALUDE_min_surface_area_large_solid_l4057_405763


namespace NUMINAMATH_CALUDE_trig_identity_l4057_405755

theorem trig_identity (α : Real) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l4057_405755


namespace NUMINAMATH_CALUDE_line_through_M_and_P_line_through_M_perp_to_line_l4057_405795

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3*x + 4*y - 11 = 0
def l₂ (x y : ℝ) : Prop := 2*x + 3*y - 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, 2)

-- Define point P
def P : ℝ × ℝ := (3, 1)

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 3*x + 2*y + 5 = 0

-- Part 1: Line equation through M and P
theorem line_through_M_and_P :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ (l₁ x y ∧ l₂ x y) ∨ (x = P.1 ∧ y = P.2)) →
    a = 1 ∧ b = 2 ∧ c = -5 :=
sorry

-- Part 2: Line equation through M and perpendicular to perp_line
theorem line_through_M_perp_to_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ (l₁ x y ∧ l₂ x y) ∨ 
      (∃ (k : ℝ), a*3 + b*2 = 0 ∧ x = M.1 + k*2 ∧ y = M.2 - k*3)) →
    a = 2 ∧ b = -3 ∧ c = 4 :=
sorry

end NUMINAMATH_CALUDE_line_through_M_and_P_line_through_M_perp_to_line_l4057_405795


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_sqrt_neg_seven_squared_l4057_405761

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

theorem sqrt_neg_seven_squared : Real.sqrt ((-7)^2) = 7 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_sqrt_neg_seven_squared_l4057_405761


namespace NUMINAMATH_CALUDE_collision_count_is_25_l4057_405760

/-- Represents a set of identical balls moving in one direction -/
structure BallSet :=
  (count : Nat)
  (direction : Bool)  -- True for left to right, False for right to left

/-- Calculates the total number of collisions between two sets of balls -/
def totalCollisions (set1 set2 : BallSet) : Nat :=
  set1.count * set2.count

/-- Theorem stating that the total number of collisions is 25 -/
theorem collision_count_is_25 :
  ∀ (left right : BallSet),
    left.count = 5 ∧ 
    right.count = 5 ∧ 
    left.direction ≠ right.direction →
    totalCollisions left right = 25 := by
  sorry

#eval totalCollisions ⟨5, true⟩ ⟨5, false⟩

end NUMINAMATH_CALUDE_collision_count_is_25_l4057_405760


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_equals_2sqrt3_quadratic_equation_solutions_l4057_405734

-- Problem 1
theorem sqrt_sum_difference_equals_2sqrt3 :
  Real.sqrt 12 + Real.sqrt 27 / 9 - Real.sqrt (1/3) = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem quadratic_equation_solutions (x : ℝ) :
  x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_equals_2sqrt3_quadratic_equation_solutions_l4057_405734


namespace NUMINAMATH_CALUDE_kevin_sells_50_crates_l4057_405757

/-- Kevin's weekly fruit sales --/
def weekly_fruit_sales (grapes mangoes passion_fruits : ℕ) : ℕ :=
  grapes + mangoes + passion_fruits

/-- Theorem: Kevin sells 50 crates of fruit per week --/
theorem kevin_sells_50_crates :
  weekly_fruit_sales 13 20 17 = 50 := by
  sorry

end NUMINAMATH_CALUDE_kevin_sells_50_crates_l4057_405757


namespace NUMINAMATH_CALUDE_probability_not_red_l4057_405782

def total_jelly_beans : ℕ := 7 + 8 + 9 + 10
def non_red_jelly_beans : ℕ := 8 + 9 + 10

theorem probability_not_red :
  (non_red_jelly_beans : ℚ) / total_jelly_beans = 27 / 34 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_red_l4057_405782


namespace NUMINAMATH_CALUDE_crocus_bulb_cost_l4057_405700

theorem crocus_bulb_cost (total_space : ℕ) (daffodil_cost : ℚ) (total_budget : ℚ) (crocus_count : ℕ) :
  total_space = 55 →
  daffodil_cost = 65/100 →
  total_budget = 2915/100 →
  crocus_count = 22 →
  ∃ (crocus_cost : ℚ), crocus_cost = 35/100 ∧
    crocus_count * crocus_cost + (total_space - crocus_count) * daffodil_cost = total_budget :=
by sorry

end NUMINAMATH_CALUDE_crocus_bulb_cost_l4057_405700


namespace NUMINAMATH_CALUDE_b_55_divisible_by_55_l4057_405765

/-- Function that generates b_n as described in the problem -/
def b (n : ℕ) : ℕ := sorry

/-- Theorem stating that b(55) is divisible by 55 -/
theorem b_55_divisible_by_55 : 55 ∣ b 55 := by sorry

end NUMINAMATH_CALUDE_b_55_divisible_by_55_l4057_405765


namespace NUMINAMATH_CALUDE_wheel_of_fortune_probability_l4057_405718

theorem wheel_of_fortune_probability (p_D p_E p_F p_G : ℚ) : 
  p_D = 3/8 → p_E = 1/4 → p_G = 1/8 → p_D + p_E + p_F + p_G = 1 → p_F = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_wheel_of_fortune_probability_l4057_405718


namespace NUMINAMATH_CALUDE_decimal_sum_as_fraction_l4057_405798

/-- The sum of 0.01, 0.002, 0.0003, 0.00004, and 0.000005 is equal to 2469/200000 -/
theorem decimal_sum_as_fraction : 
  (0.01 : ℚ) + 0.002 + 0.0003 + 0.00004 + 0.000005 = 2469 / 200000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_as_fraction_l4057_405798


namespace NUMINAMATH_CALUDE_mean_temperature_is_79_9_l4057_405701

def temperatures : List ℝ := [75, 74, 76, 77, 80, 81, 83, 85, 83, 85]

theorem mean_temperature_is_79_9 :
  (temperatures.sum / temperatures.length : ℝ) = 79.9 := by
sorry

end NUMINAMATH_CALUDE_mean_temperature_is_79_9_l4057_405701


namespace NUMINAMATH_CALUDE_headmaster_retirement_l4057_405787

/-- Represents the months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Returns the month that is n months after the given month -/
def monthsAfter (start : Month) (n : ℕ) : Month :=
  match n with
  | 0 => start
  | n + 1 => monthsAfter (match start with
    | Month.January => Month.February
    | Month.February => Month.March
    | Month.March => Month.April
    | Month.April => Month.May
    | Month.May => Month.June
    | Month.June => Month.July
    | Month.July => Month.August
    | Month.August => Month.September
    | Month.September => Month.October
    | Month.October => Month.November
    | Month.November => Month.December
    | Month.December => Month.January
  ) n

theorem headmaster_retirement (start_month : Month) (duration : ℕ) :
  start_month = Month.March → duration = 3 →
  monthsAfter start_month duration = Month.May :=
by
  sorry

end NUMINAMATH_CALUDE_headmaster_retirement_l4057_405787


namespace NUMINAMATH_CALUDE_vacuum_savings_theorem_l4057_405715

/-- The number of weeks needed to save for a vacuum cleaner. -/
def weeks_to_save (initial_savings : ℕ) (weekly_savings : ℕ) (vacuum_cost : ℕ) : ℕ :=
  ((vacuum_cost - initial_savings) + weekly_savings - 1) / weekly_savings

/-- Theorem stating that it takes 10 weeks to save for the vacuum cleaner. -/
theorem vacuum_savings_theorem :
  weeks_to_save 20 10 120 = 10 := by
  sorry

end NUMINAMATH_CALUDE_vacuum_savings_theorem_l4057_405715


namespace NUMINAMATH_CALUDE_class_average_score_l4057_405781

theorem class_average_score (total_students : Nat) (score1 score2 : Nat) (other_avg : Nat) : 
  total_students = 40 →
  score1 = 98 →
  score2 = 100 →
  other_avg = 79 →
  (other_avg * (total_students - 2) + score1 + score2) / total_students = 80 :=
by sorry

end NUMINAMATH_CALUDE_class_average_score_l4057_405781


namespace NUMINAMATH_CALUDE_quadratic_sine_interpolation_l4057_405741

theorem quadratic_sine_interpolation (f : ℝ → ℝ) (h : f = λ x => -4 / Real.pi ^ 2 * x ^ 2 + 4 / Real.pi * x) :
  f 0 = 0 ∧ f (Real.pi / 2) = 1 ∧ f Real.pi = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sine_interpolation_l4057_405741


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l4057_405793

theorem algebraic_expression_value (a b : ℝ) 
  (h1 : a + b = 3) 
  (h2 : a * b = 2) : 
  a^2 * b + 2 * a^2 * b^2 + a * b^3 = 18 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l4057_405793


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4057_405726

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 * x + 7) = 10 → x = 31 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4057_405726


namespace NUMINAMATH_CALUDE_largest_c_value_l4057_405750

theorem largest_c_value (c : ℝ) : (3 * c + 7) * (c - 2) = 9 * c → c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_c_value_l4057_405750


namespace NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l4057_405753

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_to_hemisphere_volume_ratio (q : ℝ) (q_pos : 0 < q) :
  (4 / 3 * Real.pi * q ^ 3) / (1 / 2 * 4 / 3 * Real.pi * (3 * q) ^ 3) = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l4057_405753


namespace NUMINAMATH_CALUDE_library_visitors_library_visitors_proof_l4057_405736

/-- Calculates the average number of visitors on non-Sunday days in a library -/
theorem library_visitors (sunday_avg : ℕ) (month_avg : ℕ) : ℕ :=
  let total_days : ℕ := 30
  let sunday_count : ℕ := 5
  let other_days : ℕ := total_days - sunday_count
  let total_visitors : ℕ := month_avg * total_days
  let sunday_visitors : ℕ := sunday_avg * sunday_count
  (total_visitors - sunday_visitors) / other_days

/-- Proves that the average number of visitors on non-Sunday days is 240 -/
theorem library_visitors_proof :
  library_visitors 660 310 = 240 := by
sorry

end NUMINAMATH_CALUDE_library_visitors_library_visitors_proof_l4057_405736


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l4057_405767

/-- The volume of a regular tetrahedron with given base side length and angle between lateral face and base. -/
theorem tetrahedron_volume 
  (base_side : ℝ) 
  (lateral_angle : ℝ) 
  (h : base_side = Real.sqrt 3) 
  (θ : lateral_angle = π / 3) : 
  (1 / 3 : ℝ) * base_side ^ 2 * (base_side / 2) / Real.tan lateral_angle = 1 / 2 := by
  sorry

#check tetrahedron_volume

end NUMINAMATH_CALUDE_tetrahedron_volume_l4057_405767


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4057_405729

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, x^2 - x - 20 > 0 → 1 - x^2 < 0) ∧
  (∃ x : ℝ, 1 - x^2 < 0 ∧ ¬(x^2 - x - 20 > 0)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4057_405729


namespace NUMINAMATH_CALUDE_pets_remaining_l4057_405784

theorem pets_remaining (initial_puppies initial_kittens puppies_sold kittens_sold : ℕ) :
  initial_puppies = 7 →
  initial_kittens = 6 →
  puppies_sold = 2 →
  kittens_sold = 3 →
  initial_puppies + initial_kittens - (puppies_sold + kittens_sold) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_pets_remaining_l4057_405784


namespace NUMINAMATH_CALUDE_b_power_sum_l4057_405732

theorem b_power_sum (b : ℝ) (h : 5 = b + b⁻¹) : b^6 + b⁻¹^6 = 12239 := by sorry

end NUMINAMATH_CALUDE_b_power_sum_l4057_405732


namespace NUMINAMATH_CALUDE_square_difference_l4057_405731

/-- A configuration of four squares with specific side length differences -/
structure SquareConfiguration where
  small : ℝ
  third : ℝ
  second : ℝ
  largest : ℝ
  third_diff : third = small + 13
  second_diff : second = third + 5
  largest_diff : largest = second + 11

/-- The theorem stating that the difference between the largest and smallest square's side lengths is 29 -/
theorem square_difference (config : SquareConfiguration) : config.largest - config.small = 29 :=
  sorry

end NUMINAMATH_CALUDE_square_difference_l4057_405731


namespace NUMINAMATH_CALUDE_escalator_least_time_l4057_405780

/-- The least time needed for people to go up an escalator with variable speed -/
theorem escalator_least_time (n l α : ℝ) (hn : n > 0) (hl : l > 0) (hα : α > 0) :
  let speed (m : ℝ) := m ^ (-α)
  let time_one_by_one := n * l
  let time_all_together := l * n ^ α
  min time_one_by_one time_all_together = l * n ^ min α 1 := by
  sorry

end NUMINAMATH_CALUDE_escalator_least_time_l4057_405780


namespace NUMINAMATH_CALUDE_one_fourth_of_six_times_eight_l4057_405722

theorem one_fourth_of_six_times_eight : (1 / 4 : ℚ) * (6 * 8) = 12 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_six_times_eight_l4057_405722


namespace NUMINAMATH_CALUDE_min_max_area_14_sided_lattice_polygon_l4057_405713

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A lattice polygon is a polygon with all vertices at lattice points -/
structure LatticePolygon where
  vertices : List LatticePoint
  is_convex : Bool
  is_closed : Bool

/-- A lattice parallelogram is a parallelogram with all vertices at lattice points -/
structure LatticeParallelogram where
  vertices : List LatticePoint
  is_parallelogram : Bool

def area (p : LatticeParallelogram) : ℚ :=
  sorry

def can_be_divided_into_parallelograms (poly : LatticePolygon) (parallelograms : List LatticeParallelogram) : Prop :=
  sorry

theorem min_max_area_14_sided_lattice_polygon :
  ∀ (poly : LatticePolygon) (parallelograms : List LatticeParallelogram),
    poly.vertices.length = 14 →
    poly.is_convex →
    can_be_divided_into_parallelograms poly parallelograms →
    (∃ (C : ℚ), ∀ (p : LatticeParallelogram), p ∈ parallelograms → area p ≤ C) →
    (∀ (C : ℚ), (∀ (p : LatticeParallelogram), p ∈ parallelograms → area p ≤ C) → C ≥ 5) :=
  sorry

end NUMINAMATH_CALUDE_min_max_area_14_sided_lattice_polygon_l4057_405713


namespace NUMINAMATH_CALUDE_mark_sold_one_less_l4057_405751

/-- Given:
  n: total number of boxes allocated
  M: number of boxes Mark sold
  A: number of boxes Ann sold
-/
theorem mark_sold_one_less (n M A : ℕ) : 
  n = 8 → 
  M < n → 
  M ≥ 1 → 
  A = n - 2 → 
  A ≥ 1 → 
  M + A < n → 
  M = 7 :=
by sorry

end NUMINAMATH_CALUDE_mark_sold_one_less_l4057_405751


namespace NUMINAMATH_CALUDE_tower_surface_area_l4057_405775

-- Define the volumes of the cubes
def cube_volumes : List ℝ := [1, 8, 27, 64, 125, 216, 343]

-- Function to calculate the side length of a cube given its volume
def side_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Function to calculate the surface area of a cube given its side length
def surface_area (side : ℝ) : ℝ := 6 * side^2

-- Function to calculate the exposed surface area of a cube in the tower
def exposed_surface_area (side : ℝ) (is_bottom : Bool) : ℝ :=
  if is_bottom then surface_area side else surface_area side - side^2

-- Theorem statement
theorem tower_surface_area :
  let sides := cube_volumes.map side_length
  let exposed_areas := List.zipWith exposed_surface_area sides [true, false, false, false, false, false, false]
  exposed_areas.sum = 701 := by sorry

end NUMINAMATH_CALUDE_tower_surface_area_l4057_405775


namespace NUMINAMATH_CALUDE_total_age_l4057_405771

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 6 years old
Prove that the total of their ages is 17 years. -/
theorem total_age (a b c : ℕ) : 
  b = 6 → 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 17 := by
sorry

end NUMINAMATH_CALUDE_total_age_l4057_405771


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l4057_405702

/-- A polynomial x^2 + bx + c has exactly one real root if and only if its discriminant is zero -/
def has_one_real_root (b c : ℝ) : Prop :=
  b^2 - 4*c = 0

/-- The theorem statement -/
theorem unique_root_quadratic (b c : ℝ) 
  (h1 : has_one_real_root b c)
  (h2 : b = c^2 + 1) : 
  c = 1 ∨ c = -1 :=
sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l4057_405702


namespace NUMINAMATH_CALUDE_point_A_in_second_quadrant_l4057_405735

/-- A point is in the second quadrant if its x-coordinate is negative and its y-coordinate is positive -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The coordinates of point A -/
def point_A : ℝ × ℝ := (-3, 4)

/-- Theorem: Point A is located in the second quadrant -/
theorem point_A_in_second_quadrant :
  is_in_second_quadrant point_A.1 point_A.2 := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_second_quadrant_l4057_405735


namespace NUMINAMATH_CALUDE_cylinder_max_volume_l4057_405762

/-- Given a cylinder with a constant cross-section perimeter of 4,
    prove that its maximum volume is 8π/27 -/
theorem cylinder_max_volume :
  ∀ (r h : ℝ), r > 0 → h > 0 →
  (4 * r + 2 * h = 4) →
  (π * r^2 * h ≤ 8 * π / 27) ∧
  (∃ (r₀ h₀ : ℝ), r₀ > 0 ∧ h₀ > 0 ∧ 4 * r₀ + 2 * h₀ = 4 ∧ π * r₀^2 * h₀ = 8 * π / 27) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_max_volume_l4057_405762


namespace NUMINAMATH_CALUDE_arccos_zero_l4057_405744

theorem arccos_zero (h : Set.Icc 0 π = Set.range acos) : acos 0 = π / 2 := by sorry

end NUMINAMATH_CALUDE_arccos_zero_l4057_405744


namespace NUMINAMATH_CALUDE_interesting_number_expected_value_l4057_405716

/-- A type representing a 6-digit number with specific properties -/
structure InterestingNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  d_positive : d > 0
  e_positive : e > 0
  f_positive : f > 0
  a_less_b : a < b
  b_less_c : b < c
  d_ge_e : d ≥ e
  e_ge_f : e ≥ f
  a_le_9 : a ≤ 9
  b_le_9 : b ≤ 9
  c_le_9 : c ≤ 9
  d_le_9 : d ≤ 9
  e_le_9 : e ≤ 9
  f_le_9 : f ≤ 9

/-- The expected value of an interesting number -/
def expectedValue (n : InterestingNumber) : ℝ :=
  100000 * n.a + 10000 * n.b + 1000 * n.c + 100 * n.d + 10 * n.e + n.f

/-- The theorem stating the expected value of all interesting numbers -/
theorem interesting_number_expected_value :
  ∃ (μ : ℝ), ∀ (n : InterestingNumber), μ = 308253 := by
  sorry

end NUMINAMATH_CALUDE_interesting_number_expected_value_l4057_405716


namespace NUMINAMATH_CALUDE_farmer_field_area_l4057_405783

/-- Represents the farmer's field ploughing problem -/
def FarmerField (initial_productivity : ℝ) (productivity_increase : ℝ) (days_saved : ℕ) : Prop :=
  ∃ (total_days : ℕ) (field_area : ℝ),
    field_area = initial_productivity * total_days ∧
    field_area = (2 * initial_productivity) + 
      ((total_days - days_saved - 2) * (initial_productivity * (1 + productivity_increase))) ∧
    field_area = 1440

/-- Theorem stating that the field area is 1440 hectares given the problem conditions -/
theorem farmer_field_area :
  FarmerField 120 0.25 2 :=
sorry

end NUMINAMATH_CALUDE_farmer_field_area_l4057_405783


namespace NUMINAMATH_CALUDE_jacket_price_calculation_l4057_405785

/-- Calculates the final price of an item after applying three sequential discounts -/
def final_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)

/-- Theorem stating that the final price of a $250 jacket after three specific discounts is $94.5 -/
theorem jacket_price_calculation : 
  final_price 250 0.4 0.3 0.1 = 94.5 := by
  sorry

#eval final_price 250 0.4 0.3 0.1

end NUMINAMATH_CALUDE_jacket_price_calculation_l4057_405785


namespace NUMINAMATH_CALUDE_multiplication_problem_l4057_405758

theorem multiplication_problem : ∃ x : ℕ, 72517 * x = 724807415 ∧ x = 9999 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l4057_405758


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_theorem_l4057_405797

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * (x - 1)

-- Define the complementary angles condition
def complementary_angles (x_A y_A x_B y_B m : ℝ) : Prop :=
  (y_A / (x_A - m)) + (y_B / (x_B - m)) = 0

theorem hyperbola_ellipse_theorem :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  (∃ (x_F y_F : ℝ), hyperbola x_F y_F ∧ 
    (∀ (x y : ℝ), ellipse_C x y a b ↔ 
      x^2/3 + y^2/2 = 1)) ∧
  (∃ (k x_A y_A x_B y_B : ℝ), k ≠ 0 ∧
    line_l x_A y_A k ∧ line_l x_B y_B k ∧
    ellipse_C x_A y_A 3 2 ∧ ellipse_C x_B y_B 3 2 ∧
    complementary_angles x_A y_A x_B y_B 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_theorem_l4057_405797


namespace NUMINAMATH_CALUDE_domain_of_function_l4057_405723

/-- The domain of the function f(x) = √(x - 1) + ∛(8 - x) is [1, 8] -/
theorem domain_of_function (f : ℝ → ℝ) (h : f = fun x ↦ Real.sqrt (x - 1) + (8 - x) ^ (1/3)) :
  Set.Icc 1 8 = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_domain_of_function_l4057_405723


namespace NUMINAMATH_CALUDE_circle_area_ratio_l4057_405796

theorem circle_area_ratio (R_A R_B R_C : ℝ) 
  (h1 : (60 / 360) * (2 * π * R_A) = (40 / 360) * (2 * π * R_B))
  (h2 : (30 / 360) * (2 * π * R_B) = (90 / 360) * (2 * π * R_C)) :
  (π * R_A^2) / (π * R_C^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l4057_405796


namespace NUMINAMATH_CALUDE_inequality_implies_sum_nonnegative_l4057_405791

theorem inequality_implies_sum_nonnegative (a b : ℝ) :
  Real.exp a + Real.pi ^ b ≥ Real.exp (-b) + Real.pi ^ (-a) → a + b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_sum_nonnegative_l4057_405791


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4057_405754

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    if its asymptote intersects the circle with its foci as diameter
    at the point (2, 1) in the first quadrant, then a = 2 and b = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (c : ℝ), c > 0 ∧ 
    (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → y = (b / a) * x) ∧
    (2^2 + 1^2 = c^2) ∧
    (a^2 + b^2 = c^2)) →
  a = 2 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4057_405754


namespace NUMINAMATH_CALUDE_cube_painting_l4057_405799

theorem cube_painting (m : ℚ) : 
  m > 0 → 
  let n : ℚ := 12 / m
  6 * (n - 2)^2 = 12 * (n - 2) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_painting_l4057_405799


namespace NUMINAMATH_CALUDE_expression_evaluation_l4057_405747

theorem expression_evaluation (a b : ℝ) : 
  (b - 1)^2 + |a + 3| = 0 → 
  -a^2*b + (3*a*b^2 - a^2*b) - 2*(2*a*b^2 - a^2*b) = 3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4057_405747


namespace NUMINAMATH_CALUDE_total_balls_l4057_405772

theorem total_balls (jungkook_red_balls : ℕ) (yoongi_blue_balls : ℕ) 
  (h1 : jungkook_red_balls = 3) (h2 : yoongi_blue_balls = 4) : 
  jungkook_red_balls + yoongi_blue_balls = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_l4057_405772


namespace NUMINAMATH_CALUDE_expression_equals_588_times_10_to_1007_l4057_405742

theorem expression_equals_588_times_10_to_1007 :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 588 * 10^1007 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_588_times_10_to_1007_l4057_405742


namespace NUMINAMATH_CALUDE_problem_solution_l4057_405778

theorem problem_solution (a b : ℚ) 
  (h1 : 7 * a + 3 * b = 0) 
  (h2 : b - 4 = a) : 
  9 * b = 126 / 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4057_405778


namespace NUMINAMATH_CALUDE_hexagon_coloring_count_l4057_405727

/-- A regular hexagon with 6 regions -/
inductive HexagonRegion
| A | B | C | D | E | F

/-- The available colors for planting -/
inductive PlantColor
| Color1 | Color2 | Color3 | Color4

/-- A coloring of the hexagon -/
def HexagonColoring := HexagonRegion → PlantColor

/-- Check if two regions are adjacent -/
def isAdjacent (r1 r2 : HexagonRegion) : Bool :=
  match r1, r2 with
  | HexagonRegion.A, HexagonRegion.B => true
  | HexagonRegion.A, HexagonRegion.F => true
  | HexagonRegion.B, HexagonRegion.C => true
  | HexagonRegion.C, HexagonRegion.D => true
  | HexagonRegion.D, HexagonRegion.E => true
  | HexagonRegion.E, HexagonRegion.F => true
  | _, _ => false

/-- Check if a coloring is valid (adjacent regions have different colors) -/
def isValidColoring (c : HexagonColoring) : Prop :=
  ∀ r1 r2 : HexagonRegion, isAdjacent r1 r2 → c r1 ≠ c r2

/-- The number of valid colorings -/
def numValidColorings : ℕ := 732

/-- The main theorem -/
theorem hexagon_coloring_count :
  (c : HexagonColoring) → (isValidColoring c) → numValidColorings = 732 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_coloring_count_l4057_405727


namespace NUMINAMATH_CALUDE_range_of_a_l4057_405721

-- Define the sets A and B
def A : Set ℝ := {x | 4 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ 2*a - 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = A → 3 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4057_405721


namespace NUMINAMATH_CALUDE_shooting_competition_probability_l4057_405773

theorem shooting_competition_probability 
  (p_single : ℝ) 
  (p_twice : ℝ) 
  (h1 : p_single = 4/5) 
  (h2 : p_twice = 1/2) : 
  p_twice / p_single = 5/8 := by
sorry

end NUMINAMATH_CALUDE_shooting_competition_probability_l4057_405773


namespace NUMINAMATH_CALUDE_ned_remaining_pieces_l4057_405743

/-- The number of boxes Ned originally bought -/
def total_boxes : ℝ := 14.0

/-- The number of boxes Ned gave to his little brother -/
def given_boxes : ℝ := 7.0

/-- The number of pieces in each box -/
def pieces_per_box : ℝ := 6.0

/-- The number of pieces Ned still had -/
def remaining_pieces : ℝ := (total_boxes - given_boxes) * pieces_per_box

theorem ned_remaining_pieces :
  remaining_pieces = 42.0 := by sorry

end NUMINAMATH_CALUDE_ned_remaining_pieces_l4057_405743


namespace NUMINAMATH_CALUDE_road_trip_distance_l4057_405746

theorem road_trip_distance (D : ℝ) : 
  (D / 3 + (D - D / 3) / 4 + 300 = D) → D = 600 := by sorry

end NUMINAMATH_CALUDE_road_trip_distance_l4057_405746


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_negative_one_l4057_405752

theorem intersection_nonempty_implies_a_greater_than_negative_one 
  (A : Set ℝ) (B : Set ℝ) (a : ℝ) :
  A = {x : ℝ | -1 ≤ x ∧ x < 2} →
  B = {x : ℝ | x < a} →
  (A ∩ B).Nonempty →
  a > -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_negative_one_l4057_405752


namespace NUMINAMATH_CALUDE_milk_water_ratio_l4057_405769

theorem milk_water_ratio (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) : 
  initial_volume = 45 ∧ 
  initial_milk_ratio = 4 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 18 → 
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let final_water := initial_water + added_water
  let final_milk_ratio := initial_milk / final_water
  let final_water_ratio := final_water / final_water
  final_milk_ratio / final_water_ratio = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l4057_405769


namespace NUMINAMATH_CALUDE_absolute_difference_of_powers_greater_than_half_l4057_405739

theorem absolute_difference_of_powers_greater_than_half :
  |2^3000 - 3^2006| > (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_absolute_difference_of_powers_greater_than_half_l4057_405739


namespace NUMINAMATH_CALUDE_imaginary_unit_problem_l4057_405737

theorem imaginary_unit_problem : Complex.I * (1 + Complex.I)^2 = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_problem_l4057_405737


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_at_2_0_l4057_405710

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- The x-axis -/
def x_axis : Line := { p1 := ⟨0, 0⟩, p2 := ⟨1, 0⟩ }

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

/-- Check if a point lies on the x-axis -/
def point_on_x_axis (p : Point) : Prop := p.y = 0

/-- The main theorem -/
theorem line_intersects_x_axis_at_2_0 :
  let l : Line := { p1 := ⟨4, -2⟩, p2 := ⟨0, 2⟩ }
  let intersection : Point := ⟨2, 0⟩
  point_on_line intersection l ∧ point_on_x_axis intersection := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_at_2_0_l4057_405710


namespace NUMINAMATH_CALUDE_cyclist_problem_l4057_405733

/-- Proves that given the conditions of the cyclist problem, the speed of cyclist A is 10 mph --/
theorem cyclist_problem (distance : ℝ) (speed_difference : ℝ) (meeting_distance : ℝ)
  (h1 : distance = 100)
  (h2 : speed_difference = 5)
  (h3 : meeting_distance = 20) :
  ∃ (speed_a : ℝ), speed_a = 10 ∧ 
    (distance - meeting_distance) / speed_a = 
    (distance + meeting_distance) / (speed_a + speed_difference) :=
by
  sorry


end NUMINAMATH_CALUDE_cyclist_problem_l4057_405733


namespace NUMINAMATH_CALUDE_x_cubed_y_squared_minus_y_squared_x_cubed_eq_zero_l4057_405759

theorem x_cubed_y_squared_minus_y_squared_x_cubed_eq_zero 
  (x y : ℝ) : x^3 * y^2 - y^2 * x^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_y_squared_minus_y_squared_x_cubed_eq_zero_l4057_405759


namespace NUMINAMATH_CALUDE_intersection_M_N_l4057_405717

def M : Set ℕ := {1, 2, 4, 8}

def N : Set ℕ := {x | ∃ k, x = 2 * k}

theorem intersection_M_N : M ∩ N = {2, 4, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4057_405717


namespace NUMINAMATH_CALUDE_point_M_satisfies_conditions_l4057_405719

-- Define the function f(x) = 2x^2 + 1
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x

theorem point_M_satisfies_conditions :
  let x₀ : ℝ := -2
  let y₀ : ℝ := 9
  f x₀ = y₀ ∧ f' x₀ = -8 := by
  sorry

end NUMINAMATH_CALUDE_point_M_satisfies_conditions_l4057_405719


namespace NUMINAMATH_CALUDE_range_of_a_l4057_405725

-- Define the set of real numbers x that satisfy 0 < x < 2
def P : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define the set of real numbers x that satisfy a-1 < x ≤ a
def Q (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x ≤ a}

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, (Q a ⊆ P) ∧ (Q a ≠ P)) → 
  {a : ℝ | 1 ≤ a ∧ a < 2} = {a : ℝ | ∃ x : ℝ, x ∈ Q a} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4057_405725
