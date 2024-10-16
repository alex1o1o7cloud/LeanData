import Mathlib

namespace NUMINAMATH_CALUDE_fold_line_equation_l2841_284131

/-- The perpendicular bisector of the line segment joining two points (x₁, y₁) and (x₂, y₂) -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - x₁)^2 + (p.2 - y₁)^2 = (p.1 - x₂)^2 + (p.2 - y₂)^2}

theorem fold_line_equation :
  perpendicular_bisector 5 3 1 (-1) = {p : ℝ × ℝ | p.2 = -p.1 + 4} := by
  sorry

end NUMINAMATH_CALUDE_fold_line_equation_l2841_284131


namespace NUMINAMATH_CALUDE_mean_median_difference_l2841_284146

theorem mean_median_difference (x : ℕ) : 
  let set := [x, x + 2, x + 4, x + 7, x + 27]
  let median := x + 4
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5
  mean = median + 4 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2841_284146


namespace NUMINAMATH_CALUDE_atomic_weight_sodium_l2841_284108

/-- The atomic weight of chlorine in atomic mass units (amu) -/
def atomic_weight_chlorine : ℝ := 35.45

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def atomic_weight_oxygen : ℝ := 16.00

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight_compound : ℝ := 74.00

/-- Theorem stating that the atomic weight of sodium is 22.55 amu -/
theorem atomic_weight_sodium :
  molecular_weight_compound = atomic_weight_chlorine + atomic_weight_oxygen + 22.55 := by
  sorry

end NUMINAMATH_CALUDE_atomic_weight_sodium_l2841_284108


namespace NUMINAMATH_CALUDE_crayons_per_day_l2841_284171

def boxes_per_day : ℕ := 45
def crayons_per_box : ℕ := 7

theorem crayons_per_day : boxes_per_day * crayons_per_box = 315 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_day_l2841_284171


namespace NUMINAMATH_CALUDE_circles_tangent_internally_l2841_284147

def circle_O₁_center : ℝ × ℝ := (2, 0)
def circle_O₁_radius : ℝ := 1
def circle_O₂_center : ℝ × ℝ := (-1, 0)
def circle_O₂_radius : ℝ := 3

theorem circles_tangent_internally :
  let d := Real.sqrt ((circle_O₂_center.1 - circle_O₁_center.1)^2 + (circle_O₂_center.2 - circle_O₁_center.2)^2)
  d = circle_O₂_radius ∧ d > circle_O₁_radius :=
by sorry

end NUMINAMATH_CALUDE_circles_tangent_internally_l2841_284147


namespace NUMINAMATH_CALUDE_shopping_money_l2841_284180

theorem shopping_money (remaining_amount : ℝ) (spent_percentage : ℝ) (initial_amount : ℝ) :
  remaining_amount = 217 →
  spent_percentage = 30 →
  remaining_amount = initial_amount * (1 - spent_percentage / 100) →
  initial_amount = 310 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_l2841_284180


namespace NUMINAMATH_CALUDE_work_completion_time_l2841_284155

/-- The time it takes for worker A to complete the work alone -/
def time_A : ℝ := 12

/-- The time it takes for workers A and B to complete the work together -/
def time_AB : ℝ := 7.2

/-- The time it takes for worker B to complete the work alone -/
def time_B : ℝ := 18

/-- Theorem stating that given the time for A and the time for A and B together,
    we can prove that B takes 18 days to complete the work alone -/
theorem work_completion_time :
  (1 / time_A + 1 / time_B = 1 / time_AB) → time_B = 18 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2841_284155


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l2841_284100

theorem cos_double_angle_special_case (θ : Real) 
  (h : Real.sin (Real.pi / 2 + θ) = 1 / 3) : 
  Real.cos (2 * θ) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l2841_284100


namespace NUMINAMATH_CALUDE_angle_trig_values_l2841_284174

theorem angle_trig_values (α : Real) (m : Real) 
  (h1 : m ≠ 0)
  (h2 : Real.sin α = (Real.sqrt 2 / 4) * m)
  (h3 : -Real.sqrt 3 = Real.cos α * Real.sqrt (3 + m^2))
  (h4 : m = Real.sin α * Real.sqrt (3 + m^2)) :
  Real.cos α = -Real.sqrt 6 / 4 ∧ 
  (Real.tan α = Real.sqrt 15 / 3 ∨ Real.tan α = -Real.sqrt 15 / 3) := by
sorry

end NUMINAMATH_CALUDE_angle_trig_values_l2841_284174


namespace NUMINAMATH_CALUDE_system_solution_l2841_284145

theorem system_solution : ∃! (x y : ℝ), (x / 3 - (y + 1) / 2 = 1) ∧ (4 * x - (2 * y - 5) = 11) ∧ x = 0 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2841_284145


namespace NUMINAMATH_CALUDE_average_increase_percentage_l2841_284157

def S : Finset Int := {6, 7, 10, 12, 15}
def N : Int := 34

theorem average_increase_percentage (S : Finset Int) (N : Int) :
  S = {6, 7, 10, 12, 15} →
  N = 34 →
  let original_sum := S.sum id
  let original_count := S.card
  let original_avg := original_sum / original_count
  let new_sum := original_sum + N
  let new_count := original_count + 1
  let new_avg := new_sum / new_count
  let increase := new_avg - original_avg
  let percentage_increase := (increase / original_avg) * 100
  percentage_increase = 40 := by
sorry

end NUMINAMATH_CALUDE_average_increase_percentage_l2841_284157


namespace NUMINAMATH_CALUDE_amusement_park_price_calculation_l2841_284162

/-- Calculates the total price for a family visiting an amusement park on a weekend -/
def amusement_park_price (adult_price : ℝ) (child_price : ℝ) (adult_count : ℕ) (child_count : ℕ) (adult_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let total_before_discount := adult_price * adult_count + child_price * child_count
  let adult_discount_amount := adult_price * adult_count * adult_discount
  let total_after_discount := total_before_discount - adult_discount_amount
  let tax_amount := total_after_discount * sales_tax
  total_after_discount + tax_amount

/-- Theorem: The total price for a family of 2 adults and 2 children visiting the amusement park on a weekend is $66 -/
theorem amusement_park_price_calculation :
  amusement_park_price 25 10 2 2 0.2 0.1 = 66 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_price_calculation_l2841_284162


namespace NUMINAMATH_CALUDE_triangle_line_equations_l2841_284186

/-- Triangle ABC with vertices A(-4,0), B(0,-3), and C(-2,1) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of the specific triangle in the problem -/
def triangle_ABC : Triangle :=
  { A := (-4, 0)
  , B := (0, -3)
  , C := (-2, 1) }

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of line BC and altitude from A to BC -/
theorem triangle_line_equations (t : Triangle) (h : t = triangle_ABC) :
  ∃ (line_BC altitude : LineEquation),
    line_BC = { a := 2, b := 1, c := 3 } ∧
    altitude = { a := 1, b := -2, c := 4 } := by
  sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l2841_284186


namespace NUMINAMATH_CALUDE_wendy_walking_distance_l2841_284187

theorem wendy_walking_distance
  (ran_distance : ℝ)
  (difference : ℝ)
  (h1 : ran_distance = 19.833333333333332)
  (h2 : difference = 10.666666666666666)
  (h3 : ran_distance = walked_distance + difference) :
  walked_distance = 9.166666666666666 :=
by
  sorry

end NUMINAMATH_CALUDE_wendy_walking_distance_l2841_284187


namespace NUMINAMATH_CALUDE_no_valid_codes_l2841_284140

/-- Represents a 5-digit identification code -/
structure IDCode where
  digits : Fin 5 → Fin 5
  unique : ∀ i j, i ≠ j → digits i ≠ digits j
  second_twice_first : digits 1 = 2 * digits 0
  fourth_half_third : 2 * digits 3 = digits 2

/-- The set of all valid ID codes -/
def ValidCodes : Set IDCode :=
  {code : IDCode | True}

/-- Theorem stating that there are no valid ID codes -/
theorem no_valid_codes : ValidCodes = ∅ := by
  sorry

end NUMINAMATH_CALUDE_no_valid_codes_l2841_284140


namespace NUMINAMATH_CALUDE_cos_product_sevenths_pi_l2841_284189

theorem cos_product_sevenths_pi : 
  Real.cos (π / 7) * Real.cos (2 * π / 7) * Real.cos (4 * π / 7) = -1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_sevenths_pi_l2841_284189


namespace NUMINAMATH_CALUDE_symmetrical_line_intersection_l2841_284101

/-- Given points A and B, if the line symmetrical to AB about y=a intersects
    the circle (x+3)^2 + (y+2)^2 = 1, then 1/3 ≤ a ≤ 3/2 -/
theorem symmetrical_line_intersection (a : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (0, a)
  let symmetrical_line (x y : ℝ) := (3 - a) * x - 2 * y + 2 * a = 0
  let circle (x y : ℝ) := (x + 3)^2 + (y + 2)^2 = 1
  (∃ x y, symmetrical_line x y ∧ circle x y) → 1/3 ≤ a ∧ a ≤ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_symmetrical_line_intersection_l2841_284101


namespace NUMINAMATH_CALUDE_x_to_y_equals_nine_l2841_284118

theorem x_to_y_equals_nine (x y : ℝ) : y = Real.sqrt (x - 3) + Real.sqrt (3 - x) + 2 → x^y = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_to_y_equals_nine_l2841_284118


namespace NUMINAMATH_CALUDE_problem_solution_l2841_284188

def smallest_positive_integer : ℕ := 1

def opposite_is_self (b : ℤ) : Prop := -b = b

def largest_negative_integer : ℤ := -1

theorem problem_solution (a b c : ℤ) 
  (ha : a = smallest_positive_integer)
  (hb : opposite_is_self b)
  (hc : c = largest_negative_integer + 3) :
  (2*a + 3*c) * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2841_284188


namespace NUMINAMATH_CALUDE_wind_speed_calculation_l2841_284154

/-- Given a jet's flight conditions, prove the wind speed is 50 mph -/
theorem wind_speed_calculation (j w : ℝ) 
  (h1 : 2000 = (j + w) * 4)   -- Equation for flight with tailwind
  (h2 : 2000 = (j - w) * 5)   -- Equation for return flight against wind
  : w = 50 := by
  sorry

end NUMINAMATH_CALUDE_wind_speed_calculation_l2841_284154


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2841_284149

theorem sphere_surface_area (triangle_side_length : ℝ) (center_to_plane_distance : ℝ) : 
  triangle_side_length = 3 →
  center_to_plane_distance = Real.sqrt 7 →
  ∃ (sphere_radius : ℝ),
    sphere_radius ^ 2 = triangle_side_length ^ 2 / 3 + center_to_plane_distance ^ 2 ∧
    4 * Real.pi * sphere_radius ^ 2 = 40 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2841_284149


namespace NUMINAMATH_CALUDE_total_profit_is_35000_l2841_284172

/-- Represents the subscription amounts and profit for a business venture -/
structure BusinessVenture where
  total_subscription : ℕ
  a_extra : ℕ
  b_extra : ℕ
  a_profit : ℕ

/-- Calculates the total profit given a BusinessVenture -/
def calculate_total_profit (bv : BusinessVenture) : ℕ :=
  sorry

/-- Theorem stating that for the given business venture, the total profit is 35000 -/
theorem total_profit_is_35000 : 
  let bv : BusinessVenture := {
    total_subscription := 50000,
    a_extra := 4000,
    b_extra := 5000,
    a_profit := 14700
  }
  calculate_total_profit bv = 35000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_35000_l2841_284172


namespace NUMINAMATH_CALUDE_equalize_payment_is_two_l2841_284126

/-- The amount B should give A to equalize payments when buying basketballs -/
def equalize_payment (n : ℕ+) : ℚ :=
  let total_cost := n.val * n.val
  let full_payments := total_cost / 10
  let remainder := total_cost % 10
  if remainder = 0 then 0
  else (10 - remainder) / 2

theorem equalize_payment_is_two (n : ℕ+) : equalize_payment n = 2 := by
  sorry


end NUMINAMATH_CALUDE_equalize_payment_is_two_l2841_284126


namespace NUMINAMATH_CALUDE_f_zero_f_odd_f_range_l2841_284182

noncomputable section

variable (f : ℝ → ℝ)

-- Define the properties of f
axiom add_hom : ∀ x y : ℝ, f (x + y) = f x + f y
axiom pos_map_pos : ∀ x : ℝ, x > 0 → f x > 0
axiom f_neg_one : f (-1) = -2

-- Theorem statements
theorem f_zero : f 0 = 0 := by sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem f_range : ∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f x ∈ Set.Icc (-4) 2 := by sorry

end

end NUMINAMATH_CALUDE_f_zero_f_odd_f_range_l2841_284182


namespace NUMINAMATH_CALUDE_consecutive_even_integers_cube_sum_l2841_284123

/-- Given three consecutive even integers whose squares sum to 2930, 
    prove that the sum of their cubes is 81720 -/
theorem consecutive_even_integers_cube_sum (n : ℤ) : 
  (∃ (n : ℤ), 
    (n^2 + (n+2)^2 + (n+4)^2 = 2930) ∧ 
    (∃ (k : ℤ), n = 2*k)) →
  n^3 + (n+2)^3 + (n+4)^3 = 81720 :=
sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_cube_sum_l2841_284123


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2841_284184

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 3}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2841_284184


namespace NUMINAMATH_CALUDE_prob_sum_less_than_7_is_5_12_l2841_284196

/-- The number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum less than 7) -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a sum less than 7 when throwing two dice -/
def prob_sum_less_than_7 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_less_than_7_is_5_12 : prob_sum_less_than_7 = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_7_is_5_12_l2841_284196


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l2841_284136

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x + a)(x - 4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

/-- If f(x) = (x + a)(x - 4) is an even function, then a = 4 -/
theorem even_function_implies_a_equals_four :
  ∀ a : ℝ, IsEven (f a) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l2841_284136


namespace NUMINAMATH_CALUDE_problem_statement_l2841_284199

theorem problem_statement : 
  (∀ x : ℝ, x^2 + x + 1 > 0) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2841_284199


namespace NUMINAMATH_CALUDE_rectangle_division_perimeter_l2841_284110

theorem rectangle_division_perimeter (a b x y : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < x) ∧ (x < a) ∧ (0 < y) ∧ (y < b) →
  (∃ (k₁ k₂ k₃ : ℤ),
    2 * (x + y) = k₁ ∧
    2 * (x + b - y) = k₂ ∧
    2 * (a - x + y) = k₃) →
  ∃ (k₄ : ℤ), 2 * (a - x + b - y) = k₄ :=
by sorry

end NUMINAMATH_CALUDE_rectangle_division_perimeter_l2841_284110


namespace NUMINAMATH_CALUDE_inequality_proof_l2841_284127

theorem inequality_proof (x y : ℝ) (n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) (h4 : n ≥ 2) :
  x^n / (x + y^3) + y^n / (x^3 + y) ≥ 2^(4-n) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2841_284127


namespace NUMINAMATH_CALUDE_segment_length_product_l2841_284115

theorem segment_length_product (a : ℝ) : 
  (∃ a₁ a₂ : ℝ, 
    (∀ a, ((3*a - 5)^2 + (2*a - 4)^2 = 34) ↔ (a = a₁ ∨ a = a₂)) ∧
    (a₁ * a₂ = -722/169)) :=
by sorry

end NUMINAMATH_CALUDE_segment_length_product_l2841_284115


namespace NUMINAMATH_CALUDE_randy_theorem_l2841_284160

def randy_problem (initial_amount : ℕ) (smith_contribution : ℕ) (sally_gift : ℕ) : Prop :=
  let total := initial_amount + smith_contribution
  let remaining := total - sally_gift
  remaining = 2000

theorem randy_theorem : randy_problem 3000 200 1200 := by
  sorry

end NUMINAMATH_CALUDE_randy_theorem_l2841_284160


namespace NUMINAMATH_CALUDE_acid_solution_concentration_l2841_284183

theorem acid_solution_concentration 
  (P : ℝ) -- Original acid concentration percentage
  (h1 : 0 ≤ P ∧ P ≤ 100) -- Ensure P is a valid percentage
  (h2 : 0.5 * P + 0.5 * 20 = 35) -- Equation representing the mixing process
  : P = 50 := by
  sorry

end NUMINAMATH_CALUDE_acid_solution_concentration_l2841_284183


namespace NUMINAMATH_CALUDE_equation_solution_l2841_284163

theorem equation_solution : 
  ∃! x : ℝ, (x / (x + 2) + 3 / (x + 2) + 2 * x / (x + 2) = 4) ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2841_284163


namespace NUMINAMATH_CALUDE_fruit_cost_calculation_l2841_284173

/-- The cost of a water bottle in dollars -/
def water_cost : ℚ := 0.5

/-- The cost of a snack in dollars -/
def snack_cost : ℚ := 1

/-- The number of water bottles in a bundle -/
def water_count : ℕ := 1

/-- The number of snacks in a bundle -/
def snack_count : ℕ := 3

/-- The number of fruits in a bundle -/
def fruit_count : ℕ := 2

/-- The selling price of a bundle in dollars -/
def bundle_price : ℚ := 4.6

/-- The cost of each fruit in dollars -/
def fruit_cost : ℚ := 0.55

theorem fruit_cost_calculation :
  water_cost * water_count + snack_cost * snack_count + fruit_cost * fruit_count = bundle_price :=
by sorry

end NUMINAMATH_CALUDE_fruit_cost_calculation_l2841_284173


namespace NUMINAMATH_CALUDE_wedding_fish_count_l2841_284194

/-- The number of tables at Glenda's wedding reception. -/
def num_tables : ℕ := 32

/-- The number of fish in each fishbowl, except for one special table. -/
def fish_per_table : ℕ := 2

/-- The number of fish in the special table's fishbowl. -/
def fish_in_special_table : ℕ := 3

/-- The total number of fish at Glenda's wedding reception. -/
def total_fish : ℕ := (num_tables - 1) * fish_per_table + fish_in_special_table

theorem wedding_fish_count : total_fish = 65 := by
  sorry

end NUMINAMATH_CALUDE_wedding_fish_count_l2841_284194


namespace NUMINAMATH_CALUDE_aardvark_path_length_l2841_284107

/- Define the radii of the circles -/
def small_radius : ℝ := 15
def large_radius : ℝ := 30

/- Define pi as a real number -/
noncomputable def π : ℝ := Real.pi

/- Theorem statement -/
theorem aardvark_path_length :
  let small_arc := π * small_radius
  let large_arc := π * large_radius
  let radial_segment := large_radius - small_radius
  small_arc + large_arc + 2 * radial_segment = 45 * π + 30 := by
  sorry

#check aardvark_path_length

end NUMINAMATH_CALUDE_aardvark_path_length_l2841_284107


namespace NUMINAMATH_CALUDE_xyz_value_l2841_284117

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + y * z + z * x) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 14/3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2841_284117


namespace NUMINAMATH_CALUDE_only_one_statement_correct_l2841_284178

-- Define the concept of opposite numbers
def are_opposites (a b : ℝ) : Prop := a + b = 0

-- Define the four statements
def statement1 : Prop := ∀ a b : ℝ, (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) → are_opposites a b
def statement2 : Prop := ∀ a : ℝ, ∃ b : ℝ, are_opposites a b ∧ b < 0
def statement3 : Prop := ∀ a b : ℝ, are_opposites a b → a + b = 0
def statement4 : Prop := ∀ a b : ℝ, are_opposites a b → (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

-- Theorem stating that only one of the statements is correct
theorem only_one_statement_correct :
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∧
  ¬(statement1 ∨ statement2 ∨ statement4) :=
sorry

end NUMINAMATH_CALUDE_only_one_statement_correct_l2841_284178


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2841_284143

theorem quadratic_equal_roots (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 4 * x - 1 = 0) ↔ (a = -4 ∧ ∃ x : ℝ, x = 1/2 ∧ a * x^2 + 4 * x - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2841_284143


namespace NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_l2841_284159

/-- Given that θ is an internal angle of a triangle and sin θ + cos θ = 1/2,
    prove that x²sin θ - y²cos θ = 1 represents an ellipse with foci on the y-axis -/
theorem ellipse_foci_on_y_axis (θ : Real) 
  (h1 : 0 < θ ∧ θ < π) -- θ is an internal angle of a triangle
  (h2 : Real.sin θ + Real.cos θ = 1/2) :
  ∃ (a b : Real), 
    a > 0 ∧ b > 0 ∧ 
    ∀ (x y : Real), 
      x^2 * Real.sin θ - y^2 * Real.cos θ = 1 ↔ 
      (x^2 / a^2) + (y^2 / b^2) = 1 ∧
      a < b :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_l2841_284159


namespace NUMINAMATH_CALUDE_simple_interest_rate_approx_l2841_284119

/-- The rate of simple interest given principal, amount, and time -/
def simple_interest_rate (principal amount : ℕ) (time : ℕ) : ℚ :=
  let simple_interest := amount - principal
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that the rate of simple interest is approximately 3.53% -/
theorem simple_interest_rate_approx :
  let rate := simple_interest_rate 12000 17500 13
  ∃ ε > 0, abs (rate - 353/100) < ε ∧ ε < 1/100 :=
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_approx_l2841_284119


namespace NUMINAMATH_CALUDE_min_value_theorem_l2841_284134

theorem min_value_theorem (x : ℝ) (h : x > 2) :
  x + 4 / (x - 2) ≥ 6 ∧ (x + 4 / (x - 2) = 6 ↔ x = 4) := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2841_284134


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_l2841_284151

theorem smallest_k_with_remainder (k : ℕ) : k = 534 ↔ 
  (k > 2) ∧ 
  (k % 19 = 2) ∧ 
  (k % 7 = 2) ∧ 
  (k % 4 = 2) ∧ 
  (∀ m : ℕ, m > 2 ∧ m % 19 = 2 ∧ m % 7 = 2 ∧ m % 4 = 2 → k ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_l2841_284151


namespace NUMINAMATH_CALUDE_train_overtake_time_l2841_284152

/-- The time (in seconds) for a faster train to overtake a slower train after they meet -/
def overtake_time (v1 v2 l : ℚ) : ℚ :=
  (2 * l) / ((v2 - v1) / 3600)

theorem train_overtake_time :
  let v1 : ℚ := 50  -- speed of slower train (mph)
  let v2 : ℚ := 70  -- speed of faster train (mph)
  let l : ℚ := 1/6  -- length of each train (miles)
  overtake_time v1 v2 l = 60 := by
sorry

end NUMINAMATH_CALUDE_train_overtake_time_l2841_284152


namespace NUMINAMATH_CALUDE_remainder_theorem_l2841_284116

theorem remainder_theorem (P D D' Q Q' R R' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + R')
  (h3 : R < D)
  (h4 : R' < D') :
  P % (2 * D * D') = D * R' + R :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2841_284116


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2841_284141

theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, a * x^2 + x - 1 ≤ 0) → a ≤ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2841_284141


namespace NUMINAMATH_CALUDE_absolute_value_of_w_l2841_284198

theorem absolute_value_of_w (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 2 / w = s) : 
  Complex.abs w = 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_of_w_l2841_284198


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_19_l2841_284197

theorem smallest_k_for_64_power_gt_4_power_19 : 
  ∃ k : ℕ, (∀ m : ℕ, 64^m > 4^19 → k ≤ m) ∧ 64^k > 4^19 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_19_l2841_284197


namespace NUMINAMATH_CALUDE_bob_total_candies_l2841_284133

/-- Calculates Bob's share of candies given the total amount and the ratio --/
def bobShare (total : ℕ) (samRatio : ℕ) (bobRatio : ℕ) : ℕ :=
  (total * bobRatio) / (samRatio + bobRatio)

/-- Theorem: Bob receives 64 candies in total --/
theorem bob_total_candies : 
  let chewingGums := bobShare 45 2 3
  let chocolateBars := bobShare 60 3 1
  let assortedCandies := 45 / 2
  chewingGums + chocolateBars + assortedCandies = 64 := by
  sorry

#eval bobShare 45 2 3 -- Should output 27
#eval bobShare 60 3 1 -- Should output 15
#eval 45 / 2          -- Should output 22

end NUMINAMATH_CALUDE_bob_total_candies_l2841_284133


namespace NUMINAMATH_CALUDE_jars_left_unpacked_eighty_jars_left_l2841_284158

/-- The number of jars left unpacked given the packing configuration and total jars --/
theorem jars_left_unpacked (jars_per_box1 : ℕ) (num_boxes1 : ℕ) 
  (jars_per_box2 : ℕ) (num_boxes2 : ℕ) (total_jars : ℕ) : ℕ :=
  by
  have h1 : jars_per_box1 = 12 := by sorry
  have h2 : num_boxes1 = 10 := by sorry
  have h3 : jars_per_box2 = 10 := by sorry
  have h4 : num_boxes2 = 30 := by sorry
  have h5 : total_jars = 500 := by sorry
  
  let packed_jars := jars_per_box1 * num_boxes1 + jars_per_box2 * num_boxes2
  
  have packed_eq : packed_jars = 420 := by sorry
  
  exact total_jars - packed_jars

/-- The main theorem stating that 80 jars will be left unpacked --/
theorem eighty_jars_left : jars_left_unpacked 12 10 10 30 500 = 80 := by sorry

end NUMINAMATH_CALUDE_jars_left_unpacked_eighty_jars_left_l2841_284158


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2841_284181

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y - 1 = 0 ↔ 2*x - 4*y + 5 = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2841_284181


namespace NUMINAMATH_CALUDE_tangent_implies_positive_derivative_l2841_284164

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that the tangent line at (2,3) passes through (-1,2)
def tangent_condition (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), k * (-1 - 2) + 3 = 2 ∧ HasDerivAt f k 2

-- State the theorem
theorem tangent_implies_positive_derivative (f : ℝ → ℝ) 
  (h : tangent_condition f) : 
  ∃ (d : ℝ), HasDerivAt f d 2 ∧ d > 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_implies_positive_derivative_l2841_284164


namespace NUMINAMATH_CALUDE_salary_increase_l2841_284125

theorem salary_increase (new_salary : ℝ) (increase_percentage : ℝ) (old_salary : ℝ) : 
  new_salary = 120 ∧ increase_percentage = 100 → old_salary = 60 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l2841_284125


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2841_284185

theorem quadratic_equation_result (x : ℝ) (h : x^2 - x + 3 = 0) : 
  (x - 3) * (x + 2) = -9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2841_284185


namespace NUMINAMATH_CALUDE_max_students_above_median_l2841_284168

theorem max_students_above_median (n : ℕ) (h : n = 101) :
  ∃ (scores : Fin n → ℝ),
    (∃ (median : ℝ), ∀ i : Fin n, scores i ≥ median → 
      (Fintype.card {i : Fin n | scores i > median} ≤ 50)) ∧
    (∃ (median : ℝ), Fintype.card {i : Fin n | scores i > median} = 50) :=
by sorry

end NUMINAMATH_CALUDE_max_students_above_median_l2841_284168


namespace NUMINAMATH_CALUDE_negative_three_cubed_equality_l2841_284138

theorem negative_three_cubed_equality : (-3)^3 = -3^3 := by sorry

end NUMINAMATH_CALUDE_negative_three_cubed_equality_l2841_284138


namespace NUMINAMATH_CALUDE_student_tickets_sold_l2841_284120

/-- Proves the number of student tickets sold given total tickets, total money, and ticket prices -/
theorem student_tickets_sold 
  (total_tickets : ℕ) 
  (total_money : ℕ) 
  (student_price : ℕ) 
  (nonstudent_price : ℕ) 
  (h1 : total_tickets = 821)
  (h2 : total_money = 1933)
  (h3 : student_price = 2)
  (h4 : nonstudent_price = 3) :
  ∃ (student_tickets : ℕ) (nonstudent_tickets : ℕ),
    student_tickets + nonstudent_tickets = total_tickets ∧
    student_tickets * student_price + nonstudent_tickets * nonstudent_price = total_money ∧
    student_tickets = 530 := by
  sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l2841_284120


namespace NUMINAMATH_CALUDE_probability_of_sum_15_l2841_284150

/-- Represents a standard playing card --/
inductive Card
| Number (n : Nat)
| Face

/-- A standard 52-card deck --/
def Deck : Finset Card :=
  sorry

/-- Predicate for a card being a number card (2 through 10) --/
def isNumberCard (c : Card) : Prop :=
  match c with
  | Card.Number n => 2 ≤ n ∧ n ≤ 10
  | Card.Face => False

/-- Predicate for two cards summing to 15 --/
def sumsTo15 (c1 c2 : Card) : Prop :=
  match c1, c2 with
  | Card.Number n1, Card.Number n2 => n1 + n2 = 15
  | _, _ => False

/-- The probability of selecting two number cards that sum to 15 --/
def probabilityOfSum15 : ℚ :=
  sorry

theorem probability_of_sum_15 :
  probabilityOfSum15 = 8 / 442 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_sum_15_l2841_284150


namespace NUMINAMATH_CALUDE_division_value_problem_l2841_284195

theorem division_value_problem (x : ℝ) : 
  ((7.5 / x) * 12 = 15) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l2841_284195


namespace NUMINAMATH_CALUDE_sequence_property_l2841_284176

def is_valid_sequence (s : List Nat) : Prop :=
  (∀ x ∈ s, x = 0 ∨ x = 1) ∧ 
  (∀ i j, i + 4 < s.length → j + 4 < s.length → 
    (List.take 5 (List.drop i s) ≠ List.take 5 (List.drop j s) ∨ i = j)) ∧
  (∀ x, x = 0 ∨ x = 1 → 
    ¬(∀ i j, i + 4 < (s ++ [x]).length → j + 4 < (s ++ [x]).length → 
      (List.take 5 (List.drop i (s ++ [x])) ≠ List.take 5 (List.drop j (s ++ [x])) ∨ i = j)))

theorem sequence_property (s : List Nat) (h : is_valid_sequence s) (h_length : s.length ≥ 8) :
  List.take 4 s = List.take 4 (List.reverse s) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l2841_284176


namespace NUMINAMATH_CALUDE_black_card_probability_l2841_284191

theorem black_card_probability (total_cards : ℕ) (black_cards : ℕ) 
  (h_total : total_cards = 52) 
  (h_black : black_cards = 17) : 
  (black_cards * (black_cards - 1) * (black_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 40 / 1301 := by
  sorry

end NUMINAMATH_CALUDE_black_card_probability_l2841_284191


namespace NUMINAMATH_CALUDE_student_arrangements_eq_60_l2841_284114

/-- The number of ways to arrange 6 students among three venues A, B, and C,
    where venue A receives 1 student, venue B receives 2 students,
    and venue C receives 3 students. -/
def student_arrangements : ℕ :=
  Nat.choose 6 1 * Nat.choose 5 2

theorem student_arrangements_eq_60 : student_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangements_eq_60_l2841_284114


namespace NUMINAMATH_CALUDE_quiz_score_theorem_l2841_284169

/-- Represents a quiz with scoring rules and results -/
structure Quiz where
  totalQuestions : ℕ
  correctPoints : ℕ
  incorrectPoints : ℕ
  totalScore : ℤ

/-- Calculates the score based on the number of correct answers -/
def calculateScore (q : Quiz) (correctAnswers : ℕ) : ℤ :=
  (correctAnswers : ℤ) * q.correctPoints - (q.totalQuestions - correctAnswers : ℤ) * q.incorrectPoints

/-- Theorem: Given the quiz parameters, 15 correct answers result in a score of 70 -/
theorem quiz_score_theorem (q : Quiz) 
    (h1 : q.totalQuestions = 20)
    (h2 : q.correctPoints = 5)
    (h3 : q.incorrectPoints = 1)
    (h4 : q.totalScore = 70) :
  calculateScore q 15 = q.totalScore := by
  sorry


end NUMINAMATH_CALUDE_quiz_score_theorem_l2841_284169


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2841_284167

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_complement_theorem :
  A ∩ (Set.univ \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2841_284167


namespace NUMINAMATH_CALUDE_distribute_problems_l2841_284165

theorem distribute_problems (num_problems : ℕ) (num_friends : ℕ) :
  num_problems = 6 → num_friends = 15 →
  (num_friends : ℕ) ^ (num_problems : ℕ) = 11390625 := by
  sorry

end NUMINAMATH_CALUDE_distribute_problems_l2841_284165


namespace NUMINAMATH_CALUDE_derivative_f_at_negative_two_l2841_284179

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem derivative_f_at_negative_two :
  deriv f (-2) = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_negative_two_l2841_284179


namespace NUMINAMATH_CALUDE_emily_weight_l2841_284121

def heather_weight : ℕ := 87
def weight_difference : ℕ := 78

theorem emily_weight : 
  ∃ (emily_weight : ℕ), 
    emily_weight = heather_weight - weight_difference ∧ 
    emily_weight = 9 := by
  sorry

end NUMINAMATH_CALUDE_emily_weight_l2841_284121


namespace NUMINAMATH_CALUDE_solve_for_x_l2841_284170

theorem solve_for_x (y : ℚ) (h1 : y = -3/2) (h2 : -2*x - y^2 = 1/4) : x = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2841_284170


namespace NUMINAMATH_CALUDE_square_root_of_a_minus_b_l2841_284192

theorem square_root_of_a_minus_b (a b : ℝ) (h1 : |a| = 3) (h2 : Real.sqrt (b^2) = 4) (h3 : a > b) :
  Real.sqrt (a - b) = Real.sqrt 7 ∨ Real.sqrt (a - b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_a_minus_b_l2841_284192


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l2841_284177

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l2841_284177


namespace NUMINAMATH_CALUDE_largest_non_representable_l2841_284129

def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

def is_representable (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_representable : 
  (∀ n : ℕ, n > 157 → is_representable n) ∧
  ¬is_representable 157 :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_l2841_284129


namespace NUMINAMATH_CALUDE_harry_travel_time_ratio_l2841_284105

/-- Given Harry's travel times, prove the ratio of walking time to bus journey time -/
theorem harry_travel_time_ratio :
  let total_time : ℕ := 60
  let bus_time_elapsed : ℕ := 15
  let bus_time_remaining : ℕ := 25
  let bus_time_total : ℕ := bus_time_elapsed + bus_time_remaining
  let walking_time : ℕ := total_time - bus_time_total
  walking_time / bus_time_total = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_harry_travel_time_ratio_l2841_284105


namespace NUMINAMATH_CALUDE_min_value_of_quartic_plus_constant_l2841_284103

theorem min_value_of_quartic_plus_constant :
  (∀ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2024 ≥ 2027) ∧
  (∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2024 = 2027) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_quartic_plus_constant_l2841_284103


namespace NUMINAMATH_CALUDE_spectators_count_l2841_284190

/-- The number of wristbands given to each spectator -/
def wristbands_per_person : ℕ := 2

/-- The total number of wristbands distributed -/
def total_wristbands : ℕ := 290

/-- The number of people who watched the game -/
def spectators : ℕ := total_wristbands / wristbands_per_person

theorem spectators_count : spectators = 145 := by
  sorry

end NUMINAMATH_CALUDE_spectators_count_l2841_284190


namespace NUMINAMATH_CALUDE_circle_radius_l2841_284148

theorem circle_radius (x y : Real) (h : x + y = 90 * Real.pi) :
  ∃ (r : Real), r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 9 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l2841_284148


namespace NUMINAMATH_CALUDE_residue_products_l2841_284137

theorem residue_products (m k : ℕ) (hm : m > 0) (hk : k > 0) :
  (Nat.gcd m k = 1 →
    ∃ (a : Fin m → ℤ) (b : Fin k → ℤ),
      ∀ (i j i' j' : ℕ) (hi : i < m) (hj : j < k) (hi' : i' < m) (hj' : j' < k),
        (i ≠ i' ∨ j ≠ j') →
        (a ⟨i, hi⟩ * b ⟨j, hj⟩) % (m * k) ≠ (a ⟨i', hi'⟩ * b ⟨j', hj'⟩) % (m * k)) ∧
  (Nat.gcd m k > 1 →
    ∀ (a : Fin m → ℤ) (b : Fin k → ℤ),
      ∃ (i j i' j' : ℕ) (hi : i < m) (hj : j < k) (hi' : i' < m) (hj' : j' < k),
        (i ≠ i' ∨ j ≠ j') ∧
        (a ⟨i, hi⟩ * b ⟨j, hj⟩) % (m * k) = (a ⟨i', hi'⟩ * b ⟨j', hj'⟩) % (m * k)) :=
by sorry

end NUMINAMATH_CALUDE_residue_products_l2841_284137


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2841_284122

def A : Set Int := {-1, 0, 1}
def B : Set Int := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2841_284122


namespace NUMINAMATH_CALUDE_direct_proportion_when_constant_quotient_l2841_284161

-- Define the variables
variable (a b c : ℝ)

-- Define the theorem
theorem direct_proportion_when_constant_quotient :
  (∀ x y : ℝ, x ≠ 0 → a = y / x → c ≠ 0 → a = b / c) →
  (∃ k : ℝ, ∀ x y : ℝ, x ≠ 0 → a = y / x → y = k * x) :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_when_constant_quotient_l2841_284161


namespace NUMINAMATH_CALUDE_fencing_cost_is_105_rupees_l2841_284109

/-- Represents a rectangular field -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  ratio : ℝ × ℝ

/-- Calculates the cost of fencing a rectangular field -/
def fencingCost (field : RectangularField) (costPerMeter : ℝ) : ℝ :=
  2 * (field.length + field.width) * costPerMeter

/-- Theorem: The cost of fencing a specific rectangular field is 105 rupees -/
theorem fencing_cost_is_105_rupees : 
  ∀ (field : RectangularField),
    field.ratio = (3, 4) →
    field.area = 10800 →
    fencingCost field 0.25 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_105_rupees_l2841_284109


namespace NUMINAMATH_CALUDE_equation_solution_l2841_284144

theorem equation_solution : ∀ x : ℚ, (2/3 : ℚ) - (1/4 : ℚ) = 1/x → x = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2841_284144


namespace NUMINAMATH_CALUDE_bus_average_speed_l2841_284132

/-- Proves that the average speed of a bus line is 60 km/h given specific conditions -/
theorem bus_average_speed
  (stop_interval : ℕ) -- Time interval between stops in minutes
  (num_stops : ℕ) -- Number of stops to the destination
  (distance : ℝ) -- Distance to the destination in kilometers
  (h1 : stop_interval = 5)
  (h2 : num_stops = 8)
  (h3 : distance = 40) :
  distance / (↑(stop_interval * num_stops) / 60) = 60 :=
by sorry

end NUMINAMATH_CALUDE_bus_average_speed_l2841_284132


namespace NUMINAMATH_CALUDE_coefficient_x_fourth_l2841_284135

def binomial_coeff (n k : ℕ) : ℕ := sorry

def binomial_expansion_term (n r : ℕ) (a b : ℚ) : ℚ := sorry

theorem coefficient_x_fourth (n : ℕ) (h : n = 5) :
  ∃ (k : ℕ), binomial_coeff n k * (2 * k - 5) = 10 ∧
             binomial_expansion_term n k 1 1 = binomial_coeff n k * (2 * k - 5) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_fourth_l2841_284135


namespace NUMINAMATH_CALUDE_max_value_inequality_l2841_284128

theorem max_value_inequality (x y : ℝ) :
  (x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2841_284128


namespace NUMINAMATH_CALUDE_initial_peaches_l2841_284112

theorem initial_peaches (initial : ℕ) : initial + 52 = 86 → initial = 34 := by
  sorry

end NUMINAMATH_CALUDE_initial_peaches_l2841_284112


namespace NUMINAMATH_CALUDE_log_equality_implies_y_value_l2841_284153

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := sorry

-- Define the variables
variable (a b c x : ℝ)
variable (p q r y : ℝ)

-- State the theorem
theorem log_equality_implies_y_value
  (h1 : log a / p = log b / q)
  (h2 : log b / q = log c / r)
  (h3 : log c / r = log x)
  (h4 : x ≠ 1)
  (h5 : b^3 / (a^2 * c) = x^y) :
  y = 3*q - 2*p - r := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_y_value_l2841_284153


namespace NUMINAMATH_CALUDE_max_abc_value_l2841_284124

theorem max_abc_value (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a * b + b * c + a * c = 1) : 
  a * b * c ≤ Real.sqrt 3 / 9 := by
sorry

end NUMINAMATH_CALUDE_max_abc_value_l2841_284124


namespace NUMINAMATH_CALUDE_solution_equals_expected_l2841_284130

-- Define the clubsuit operation
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points satisfying x ⋆ y = y ⋆ x
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | clubsuit p.1 p.2 = clubsuit p.2 p.1}

-- Define the union of x-axis, y-axis, and lines y = x and y = -x
def expected_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

-- Theorem statement
theorem solution_equals_expected : solution_set = expected_set := by
  sorry

end NUMINAMATH_CALUDE_solution_equals_expected_l2841_284130


namespace NUMINAMATH_CALUDE_probability_at_least_seven_three_times_l2841_284113

/-- The probability of rolling at least a seven on a single roll of an 8-sided die -/
def p : ℚ := 1/4

/-- The number of rolls -/
def n : ℕ := 4

/-- The probability of rolling at least a seven at least three times in four rolls of an 8-sided die -/
theorem probability_at_least_seven_three_times : 
  (Finset.sum (Finset.range 2) (λ k => (n.choose (n - k)) * p^(n - k) * (1 - p)^k)) = 13/256 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_seven_three_times_l2841_284113


namespace NUMINAMATH_CALUDE_age_difference_l2841_284104

theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 5 →
  sachin_age * 12 = rahul_age * 5 →
  rahul_age - sachin_age = 7 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2841_284104


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l2841_284106

-- Define the parabola function
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x - 7

-- Define the theorem
theorem parabola_point_relationship (a : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h_opens_down : a < 0)
  (h_y₁ : y₁ = parabola a (-4))
  (h_y₂ : y₂ = parabola a 2)
  (h_y₃ : y₃ = parabola a 3) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l2841_284106


namespace NUMINAMATH_CALUDE_sum_of_digits_11_pow_2010_l2841_284102

/-- The sum of the tens digit and the units digit in the decimal representation of 11^2010 is 1. -/
theorem sum_of_digits_11_pow_2010 : ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 11^2010 % 100 = 10 * a + b ∧ a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_11_pow_2010_l2841_284102


namespace NUMINAMATH_CALUDE_zoe_yogurt_consumption_l2841_284142

/-- Calculates the number of ounces of yogurt Zoe ate given the following conditions:
  * Zoe ate 12 strawberries and some ounces of yogurt
  * Strawberries have 4 calories each
  * Yogurt has 17 calories per ounce
  * Zoe ate a total of 150 calories
-/
theorem zoe_yogurt_consumption (
  strawberry_count : ℕ)
  (strawberry_calories : ℕ)
  (yogurt_calories_per_ounce : ℕ)
  (total_calories : ℕ)
  (h1 : strawberry_count = 12)
  (h2 : strawberry_calories = 4)
  (h3 : yogurt_calories_per_ounce = 17)
  (h4 : total_calories = 150)
  : (total_calories - strawberry_count * strawberry_calories) / yogurt_calories_per_ounce = 6 := by
  sorry

end NUMINAMATH_CALUDE_zoe_yogurt_consumption_l2841_284142


namespace NUMINAMATH_CALUDE_min_value_abc_l2841_284193

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_3 : a + b + c = 3) :
  (a + b) / (a * b * c) ≥ 16 / 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_abc_l2841_284193


namespace NUMINAMATH_CALUDE_irregular_shape_area_l2841_284156

/-- The area of an irregular shape formed by removing a smaller rectangle and a right triangle from a larger rectangle --/
theorem irregular_shape_area (large_length large_width small_length small_width triangle_base triangle_height : ℝ)
  (h1 : large_length = 10)
  (h2 : large_width = 6)
  (h3 : small_length = 4)
  (h4 : small_width = 3)
  (h5 : triangle_base = small_length)
  (h6 : triangle_height = 3) :
  large_length * large_width - (small_length * small_width + 1/2 * triangle_base * triangle_height) = 42 := by
  sorry

end NUMINAMATH_CALUDE_irregular_shape_area_l2841_284156


namespace NUMINAMATH_CALUDE_abs_inequality_abs_inequality_with_constraints_l2841_284166

-- Part I
theorem abs_inequality (x : ℝ) : 
  |x - 1| + |2*x + 1| > 3 ↔ x < -1 ∨ x > 1 := by sorry

-- Part II
theorem abs_inequality_with_constraints (a b : ℝ) 
  (ha : a ∈ Set.Icc (-1 : ℝ) 1) (hb : b ∈ Set.Icc (-1 : ℝ) 1) : 
  |1 + a*b/4| > |(a + b)/2| := by sorry

end NUMINAMATH_CALUDE_abs_inequality_abs_inequality_with_constraints_l2841_284166


namespace NUMINAMATH_CALUDE_set_intersection_equiv_a_range_l2841_284175

/-- Given real number a, define set A -/
def A (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}

/-- Define set B based on set A -/
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 2 * x + 3}

/-- Define set C based on set A -/
def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = x^2}

/-- Theorem stating the equivalence of the set intersection condition and the range of a -/
theorem set_intersection_equiv_a_range (a : ℝ) :
  (B a ∩ C a = C a) ↔ (a < -2 ∨ (1/2 ≤ a ∧ a ≤ 3)) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equiv_a_range_l2841_284175


namespace NUMINAMATH_CALUDE_problem_solution_l2841_284139

theorem problem_solution (x : ℝ) : 
  (1/4 : ℝ) + 4 * ((1/2013 : ℝ) + 1/x) = 7/4 → 
  1872 + 48 * ((2013 * x) / (x + 2013)) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2841_284139


namespace NUMINAMATH_CALUDE_two_digit_number_representation_l2841_284111

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

theorem two_digit_number_representation (n : TwoDigitNumber) :
  n.value = 10 * n.tens + n.units := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_representation_l2841_284111
