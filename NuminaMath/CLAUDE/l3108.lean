import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3108_310838

theorem complex_number_quadrant (a : ℝ) : 
  (((2*a + 2*Complex.I) / (1 + Complex.I)).im ≠ 0 ∧ 
   ((2*a + 2*Complex.I) / (1 + Complex.I)).re = 0) → 
  (2*a < 0 ∧ 2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3108_310838


namespace NUMINAMATH_CALUDE_slope_at_five_is_zero_l3108_310859

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem slope_at_five_is_zero
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_diff : Differentiable ℝ f)
  (h_period : has_period f 5) :
  deriv f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_slope_at_five_is_zero_l3108_310859


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3108_310815

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x + 1| > 3} = {x : ℝ | x < -2 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3108_310815


namespace NUMINAMATH_CALUDE_product_of_roots_l3108_310812

theorem product_of_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdistinct : x ≠ y)
  (h : x + 4 / x = y + 4 / y) : x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3108_310812


namespace NUMINAMATH_CALUDE_jaxon_toys_count_l3108_310850

/-- The number of toys Jaxon has -/
def jaxon_toys : ℕ := 15

/-- The number of toys Gabriel has -/
def gabriel_toys : ℕ := 2 * jaxon_toys

/-- The number of toys Jerry has -/
def jerry_toys : ℕ := gabriel_toys + 8

theorem jaxon_toys_count :
  jaxon_toys + gabriel_toys + jerry_toys = 83 ∧ jaxon_toys = 15 :=
by sorry

end NUMINAMATH_CALUDE_jaxon_toys_count_l3108_310850


namespace NUMINAMATH_CALUDE_figure_area_theorem_l3108_310821

theorem figure_area_theorem (y : ℝ) :
  (3 * y)^2 + (7 * y)^2 + (1/2 * 3 * y * 7 * y) = 1200 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_theorem_l3108_310821


namespace NUMINAMATH_CALUDE_fixed_point_and_bisecting_line_l3108_310898

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 + a = 0

-- Define lines l₁ and l₂
def line_l1 (x y : ℝ) : Prop := 4 * x + y + 3 = 0
def line_l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 5 = 0

-- Define the fixed point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define line m
def line_m (x y : ℝ) : Prop := 3 * x + y + 1 = 0

theorem fixed_point_and_bisecting_line :
  (∀ a : ℝ, line_l a (point_P.1) (point_P.2)) ∧
  (∀ x y : ℝ, line_m x y ↔ 
    ∃ t : ℝ, 
      line_l1 t (-4*t-3) ∧ 
      line_l2 (-t-2) (4*t+7) ∧
      point_P = ((t + (-t-2))/2, ((-4*t-3) + (4*t+7))/2)) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_and_bisecting_line_l3108_310898


namespace NUMINAMATH_CALUDE_sqrt_less_than_3y_minus_1_l3108_310825

theorem sqrt_less_than_3y_minus_1 (y : ℝ) :
  y > 0 → (Real.sqrt y < 3 * y - 1 ↔ y > 1) := by sorry

end NUMINAMATH_CALUDE_sqrt_less_than_3y_minus_1_l3108_310825


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3108_310813

/-- Given a hyperbola C with equation x²/m - y² = 1 and one focus at (2, 0),
    prove that its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (m : ℝ) (h1 : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / m - y^2 = 1}
  let focus : ℝ × ℝ := (2, 0)
  focus ∈ {f | ∃ (x y : ℝ), (x, y) ∈ C ∧ (x - f.1)^2 + (y - f.2)^2 = (x + f.1)^2 + (y - f.2)^2} →
  let e := Real.sqrt ((2 : ℝ)^2 / m)
  e = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3108_310813


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3108_310803

theorem quadratic_inequality_solution (y : ℝ) :
  -y^2 + 9*y - 20 < 0 ↔ y < 4 ∨ y > 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3108_310803


namespace NUMINAMATH_CALUDE_tangent_line_equations_l3108_310864

/-- Given a cubic curve and a point, prove the equations of tangent lines passing through the point. -/
theorem tangent_line_equations (x y : ℝ → ℝ) (P : ℝ × ℝ) : 
  (∀ t, y t = (1/3) * (x t)^3 + 4/3) →  -- Curve equation
  P = (2, 4) →  -- Point P
  ∃ (A B : ℝ), 
    ((4 * x A - y A - 4 = 0) ∨ (x B - y B + 2 = 0)) ∧ 
    (∀ t, (4 * t - y A - 4 = 0) → (x A, y A) = (2, 4)) ∧
    (∀ t, (t - y B + 2 = 0) → (x B, y B) = (2, 4)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l3108_310864


namespace NUMINAMATH_CALUDE_same_temperature_exists_l3108_310817

/-- Conversion function from Celsius to Fahrenheit -/
def celsius_to_fahrenheit (c : ℝ) : ℝ := 1.8 * c + 32

/-- Theorem stating that there exists a temperature that is the same in both Celsius and Fahrenheit scales -/
theorem same_temperature_exists : ∃ t : ℝ, t = celsius_to_fahrenheit t := by
  sorry

end NUMINAMATH_CALUDE_same_temperature_exists_l3108_310817


namespace NUMINAMATH_CALUDE_range_of_y_minus_2x_l3108_310848

theorem range_of_y_minus_2x (x y : ℝ) 
  (hx : -2 ≤ x ∧ x ≤ 1) 
  (hy : 2 ≤ y ∧ y ≤ 4) : 
  0 ≤ y - 2*x ∧ y - 2*x ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_y_minus_2x_l3108_310848


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_l3108_310872

/-- The area of a cross-section of a cylinder with given dimensions and cut angle -/
theorem cylinder_cross_section_area (h r : ℝ) (θ : ℝ) :
  h = 10 ∧ r = 7 ∧ θ = 150 * π / 180 →
  ∃ (A : ℝ), A = (73.5 : ℝ) * π + 70 * Real.sqrt 6 ∧
    A = r * (2 * r * Real.sin (θ / 2)) + π * (r * Real.sin (θ / 2))^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_l3108_310872


namespace NUMINAMATH_CALUDE_students_per_group_l3108_310832

theorem students_per_group 
  (total_students : ℕ) 
  (unpicked_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 65) 
  (h2 : unpicked_students = 17) 
  (h3 : num_groups = 8) : 
  (total_students - unpicked_students) / num_groups = 6 := by
  sorry

end NUMINAMATH_CALUDE_students_per_group_l3108_310832


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_144_l3108_310831

theorem greatest_prime_factor_of_144 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 144 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 144 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_144_l3108_310831


namespace NUMINAMATH_CALUDE_soup_weight_proof_l3108_310840

theorem soup_weight_proof (initial_weight : ℝ) : 
  (((initial_weight / 2) / 2) / 2 = 5) → initial_weight = 40 := by
  sorry

end NUMINAMATH_CALUDE_soup_weight_proof_l3108_310840


namespace NUMINAMATH_CALUDE_new_profit_percentage_l3108_310856

theorem new_profit_percentage
  (original_profit_rate : ℝ)
  (original_selling_price : ℝ)
  (price_reduction_rate : ℝ)
  (additional_profit : ℝ)
  (h1 : original_profit_rate = 0.1)
  (h2 : original_selling_price = 439.99999999999966)
  (h3 : price_reduction_rate = 0.1)
  (h4 : additional_profit = 28) :
  let original_cost_price := original_selling_price / (1 + original_profit_rate)
  let new_cost_price := original_cost_price * (1 - price_reduction_rate)
  let new_selling_price := original_selling_price + additional_profit
  let new_profit := new_selling_price - new_cost_price
  let new_profit_percentage := (new_profit / new_cost_price) * 100
  new_profit_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_new_profit_percentage_l3108_310856


namespace NUMINAMATH_CALUDE_manufacturer_cost_effectiveness_l3108_310861

/-- Represents the cost calculation for manufacturers A and B -/
def cost_calculation (x : ℝ) : Prop :=
  let desk_price : ℝ := 200
  let chair_price : ℝ := 50
  let desk_quantity : ℝ := 60
  let discount_rate : ℝ := 0.9
  let cost_A : ℝ := desk_price * desk_quantity + chair_price * (x - desk_quantity)
  let cost_B : ℝ := (desk_price * desk_quantity + chair_price * x) * discount_rate
  (x ≥ desk_quantity) ∧
  (x < 360 → cost_A < cost_B) ∧
  (x > 360 → cost_B < cost_A)

/-- Theorem stating the conditions for cost-effectiveness of manufacturers A and B -/
theorem manufacturer_cost_effectiveness :
  ∀ x : ℝ, cost_calculation x :=
sorry

end NUMINAMATH_CALUDE_manufacturer_cost_effectiveness_l3108_310861


namespace NUMINAMATH_CALUDE_sixDigitPermutations_eq_90_l3108_310877

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 1, 3, 3, 7, and 7 -/
def sixDigitPermutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of such permutations is 90 -/
theorem sixDigitPermutations_eq_90 : sixDigitPermutations = 90 := by
  sorry

end NUMINAMATH_CALUDE_sixDigitPermutations_eq_90_l3108_310877


namespace NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l3108_310814

/-- The ratio of the volume of an octahedron formed by the centers of the faces of a cube
    to the volume of the cube itself, given that the cube has side length 2. -/
theorem octahedron_cube_volume_ratio :
  let cube_side_length : ℝ := 2
  let cube_volume : ℝ := cube_side_length ^ 3
  let octahedron_volume : ℝ := 4 / 3
  octahedron_volume / cube_volume = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l3108_310814


namespace NUMINAMATH_CALUDE_kendras_family_size_l3108_310801

/-- Proves the number of people in Kendra's family given the cookie baking scenario --/
theorem kendras_family_size 
  (cookies_per_batch : ℕ) 
  (num_batches : ℕ) 
  (chips_per_cookie : ℕ) 
  (chips_per_person : ℕ) 
  (h1 : cookies_per_batch = 12)
  (h2 : num_batches = 3)
  (h3 : chips_per_cookie = 2)
  (h4 : chips_per_person = 18)
  : (cookies_per_batch * num_batches * chips_per_cookie) / chips_per_person = 4 := by
  sorry

#eval (12 * 3 * 2) / 18  -- Should output 4

end NUMINAMATH_CALUDE_kendras_family_size_l3108_310801


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3108_310869

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_sum : a 2 + a 3 + a 4 = 15)
  (h_geom : is_geometric_sequence (λ n => 
    match n with
    | 1 => a 1 + 2
    | 2 => a 3 + 4
    | 3 => a 6 + 16
    | _ => 0
  )) :
  a 10 = 19 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3108_310869


namespace NUMINAMATH_CALUDE_rubble_purchase_l3108_310890

/-- Calculates the remaining money after a purchase. -/
def remaining_money (initial_amount notebook_cost pen_cost notebook_count pen_count : ℚ) : ℚ :=
  initial_amount - (notebook_cost * notebook_count + pen_cost * pen_count)

/-- Proves that Rubble will have $4.00 left after his purchase. -/
theorem rubble_purchase : 
  let initial_amount : ℚ := 15
  let notebook_cost : ℚ := 4
  let pen_cost : ℚ := 1.5
  let notebook_count : ℚ := 2
  let pen_count : ℚ := 2
  remaining_money initial_amount notebook_cost pen_cost notebook_count pen_count = 4 := by
  sorry

#eval remaining_money 15 4 1.5 2 2

end NUMINAMATH_CALUDE_rubble_purchase_l3108_310890


namespace NUMINAMATH_CALUDE_recreational_space_perimeter_l3108_310807

-- Define the playground and sandbox dimensions
def playground_width : ℕ := 20
def playground_height : ℕ := 16
def sandbox_width : ℕ := 4
def sandbox_height : ℕ := 3

-- Define the sandbox position
def sandbox_top_distance : ℕ := 6
def sandbox_left_distance : ℕ := 8

-- Define the perimeter calculation function
def calculate_perimeter (playground_width playground_height sandbox_width sandbox_height sandbox_top_distance sandbox_left_distance : ℕ) : ℕ :=
  let right_width := playground_width - sandbox_left_distance - sandbox_width
  let bottom_height := playground_height - sandbox_top_distance - sandbox_height
  let right_perimeter := 2 * (playground_height + right_width)
  let bottom_perimeter := 2 * (bottom_height + sandbox_left_distance)
  let left_perimeter := 2 * (sandbox_top_distance + sandbox_left_distance)
  let overlap := 4 * sandbox_left_distance
  right_perimeter + bottom_perimeter + left_perimeter - overlap

-- Theorem statement
theorem recreational_space_perimeter :
  calculate_perimeter playground_width playground_height sandbox_width sandbox_height sandbox_top_distance sandbox_left_distance = 74 := by
  sorry

end NUMINAMATH_CALUDE_recreational_space_perimeter_l3108_310807


namespace NUMINAMATH_CALUDE_parrot_fraction_l3108_310842

theorem parrot_fraction (p t : ℝ) : 
  p + t = 1 →                     -- Total fraction of birds
  (2/3 : ℝ) * p + (1/4 : ℝ) * t = (1/2 : ℝ) →  -- Male birds equation
  p = (3/5 : ℝ) := by             -- Fraction of parrots
sorry

end NUMINAMATH_CALUDE_parrot_fraction_l3108_310842


namespace NUMINAMATH_CALUDE_det_submatrix_l3108_310868

theorem det_submatrix (a b c d : ℝ) :
  Matrix.det !![1, a, b; 2, c, d; 3, 0, 0] = 6 →
  Matrix.det !![a, b; c, d] = 2 := by
sorry

end NUMINAMATH_CALUDE_det_submatrix_l3108_310868


namespace NUMINAMATH_CALUDE_sin_75_cos_75_eq_half_l3108_310867

theorem sin_75_cos_75_eq_half : 2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_75_eq_half_l3108_310867


namespace NUMINAMATH_CALUDE_range_of_f_l3108_310839

def f (x : ℕ) : ℕ := 2 * x + 1

def domain : Set ℕ := {1, 2, 3, 4}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {3, 5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3108_310839


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3108_310858

/-- A quadratic function with coefficients a, b, and c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The discriminant of a quadratic function -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_one_root (k : ℝ) : 
  (∃! x, QuadraticFunction 1 (-2) k x = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3108_310858


namespace NUMINAMATH_CALUDE_loan_split_l3108_310888

/-- Given a total sum of 2691 Rs. split into two parts, if the interest on the first part
    for 8 years at 3% per annum is equal to the interest on the second part for 3 years
    at 5% per annum, then the second part of the sum is 1656 Rs. -/
theorem loan_split (x : ℚ) : 
  (x ≥ 0) →
  (2691 - x ≥ 0) →
  (x * 3 * 8 / 100 = (2691 - x) * 5 * 3 / 100) →
  (2691 - x = 1656) :=
by sorry

end NUMINAMATH_CALUDE_loan_split_l3108_310888


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3108_310870

/-- Represents the population groups in the organization -/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Represents a stratified sample -/
structure StratifiedSample where
  total : Nat
  young : Nat

/-- Theorem stating the relationship between the population, sample, and sample size -/
theorem stratified_sample_size 
  (pop : Population)
  (sample : StratifiedSample)
  (h1 : pop.elderly = 20)
  (h2 : pop.middleAged = 120)
  (h3 : pop.young = 100)
  (h4 : sample.young = 10) :
  sample.total = 24 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l3108_310870


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3108_310828

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 3 = (x - 1)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3108_310828


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3108_310884

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3108_310884


namespace NUMINAMATH_CALUDE_proposition_two_l3108_310835

theorem proposition_two (a b : ℝ) : a > b → ((1 / a < 1 / b) ↔ (a * b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_two_l3108_310835


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3108_310863

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The length of the base
  base : ℝ
  -- The length of the median drawn to one of the congruent sides
  median : ℝ
  -- The length of each congruent side
  side : ℝ
  -- Condition that the base is 4√2
  base_eq : base = 4 * Real.sqrt 2
  -- Condition that the median is 5
  median_eq : median = 5

/-- Theorem stating the length of the congruent sides in the specific isosceles triangle -/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) : t.side = Real.sqrt 34 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3108_310863


namespace NUMINAMATH_CALUDE_tshirt_sale_revenue_l3108_310891

/-- Calculates the money made per minute during a t-shirt sale -/
def money_per_minute (total_shirts : ℕ) (sale_duration : ℕ) (black_price white_price : ℚ) : ℚ :=
  let black_shirts := total_shirts / 2
  let white_shirts := total_shirts / 2
  let total_revenue := (black_shirts : ℚ) * black_price + (white_shirts : ℚ) * white_price
  total_revenue / (sale_duration : ℚ)

/-- Proves that the money made per minute during the specific t-shirt sale is $220 -/
theorem tshirt_sale_revenue : money_per_minute 200 25 30 25 = 220 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_sale_revenue_l3108_310891


namespace NUMINAMATH_CALUDE_sum_remainder_mod_16_l3108_310852

theorem sum_remainder_mod_16 : (List.sum [75, 76, 77, 78, 79, 80, 81, 82]) % 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_16_l3108_310852


namespace NUMINAMATH_CALUDE_divisibility_by_nineteen_l3108_310847

theorem divisibility_by_nineteen (k : ℕ) : 19 ∣ (2^(26*k + 2) + 3) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_nineteen_l3108_310847


namespace NUMINAMATH_CALUDE_counterexample_exists_l3108_310887

theorem counterexample_exists : ∃ n : ℕ+, 
  ¬(Nat.Prime n.val) ∧ Nat.Prime (n.val - 2) ∧ n.val = 33 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3108_310887


namespace NUMINAMATH_CALUDE_cos_30_minus_cos_60_l3108_310865

theorem cos_30_minus_cos_60 : Real.cos (30 * π / 180) - Real.cos (60 * π / 180) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_30_minus_cos_60_l3108_310865


namespace NUMINAMATH_CALUDE_inequality_proof_l3108_310836

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 * (y^2 + z^2)^2 + y^3 * (z^2 + x^2)^2 + z^3 * (x^2 + y^2)^2 ≥
  x * y * z * (x * y * (x + y)^2 + y * z * (y + z)^2 + z * x * (z + x)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3108_310836


namespace NUMINAMATH_CALUDE_apple_purchase_multiple_l3108_310899

theorem apple_purchase_multiple : ∀ x : ℕ,
  (15 : ℕ) + 15 * x + 60 * x = (240 : ℕ) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_multiple_l3108_310899


namespace NUMINAMATH_CALUDE_largest_equal_cost_under_500_equal_cost_242_largest_equal_cost_is_242_l3108_310862

/-- Convert a natural number to its base 3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 3) :: aux (m / 3)
  aux n |>.reverse

/-- Sum of digits in decimal representation -/
def sumDecimalDigits (n : ℕ) : ℕ :=
  let digits := n.repr.toList.map (λ c => c.toNat - '0'.toNat)
  digits.sum

/-- Sum of digits in base 3 representation -/
def sumBase3Digits (n : ℕ) : ℕ :=
  (toBase3 n).sum

/-- Predicate for numbers with equal cost in decimal and base 3 -/
def equalCost (n : ℕ) : Prop :=
  sumDecimalDigits n = sumBase3Digits n

theorem largest_equal_cost_under_500 :
  ∀ n : ℕ, n < 500 → n > 242 → ¬(equalCost n) :=
by sorry

theorem equal_cost_242 : equalCost 242 :=
by sorry

theorem largest_equal_cost_is_242 :
  ∃! n : ℕ, n < 500 ∧ equalCost n ∧ ∀ m : ℕ, m < 500 → equalCost m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_equal_cost_under_500_equal_cost_242_largest_equal_cost_is_242_l3108_310862


namespace NUMINAMATH_CALUDE_inequality_proof_l3108_310878

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) :
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) ≥ 
  2 / (1 + a) + 2 / (1 + b) + 2 / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3108_310878


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3108_310833

theorem inequality_solution_set (x : ℝ) : (x - 2) * (3 - x) > 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3108_310833


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_l3108_310830

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_remainder (n : ℕ) (h : n ≥ 100) :
  sum_factorials n % 30 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 30 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_l3108_310830


namespace NUMINAMATH_CALUDE_rectangle_perimeter_bound_l3108_310886

/-- Given a unit square covered by m^2 rectangles, there exists a rectangle with perimeter at least 4/m -/
theorem rectangle_perimeter_bound (m : ℝ) (h_m : m > 0) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b ≤ 1 / m^2 ∧ 2 * (a + b) ≥ 4 / m := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_bound_l3108_310886


namespace NUMINAMATH_CALUDE_max_supervisors_is_three_l3108_310896

/-- Represents the number of years in a supervisor's term -/
def termLength : ℕ := 4

/-- Represents the gap year between supervisors -/
def gapYear : ℕ := 1

/-- Represents the total period in years -/
def totalPeriod : ℕ := 15

/-- Calculates the maximum number of supervisors that can be hired -/
def maxSupervisors : ℕ := (totalPeriod + gapYear) / (termLength + gapYear)

theorem max_supervisors_is_three :
  maxSupervisors = 3 :=
sorry

end NUMINAMATH_CALUDE_max_supervisors_is_three_l3108_310896


namespace NUMINAMATH_CALUDE_unique_number_base_conversion_l3108_310882

def is_valid_base_8_digit (d : ℕ) : Prop := d < 8
def is_valid_base_6_digit (d : ℕ) : Prop := d < 6

def base_8_to_decimal (a b : ℕ) : ℕ := 8 * a + b
def base_6_to_decimal (b a : ℕ) : ℕ := 6 * b + a

theorem unique_number_base_conversion : ∃! n : ℕ, 
  ∃ (a b : ℕ), 
    is_valid_base_8_digit a ∧
    is_valid_base_6_digit b ∧
    n = base_8_to_decimal a b ∧
    n = base_6_to_decimal b a ∧
    n = 45 := by sorry

end NUMINAMATH_CALUDE_unique_number_base_conversion_l3108_310882


namespace NUMINAMATH_CALUDE_union_of_sets_l3108_310823

theorem union_of_sets (A B : Set ℝ) : 
  (A = {x : ℝ | x ≥ 0}) → 
  (B = {x : ℝ | x < 1}) → 
  A ∪ B = Set.univ := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3108_310823


namespace NUMINAMATH_CALUDE_evaluate_expression_l3108_310893

theorem evaluate_expression : (-1 : ℤ) ^ (6 ^ 2) + (1 : ℤ) ^ (3 ^ 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3108_310893


namespace NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_l3108_310855

theorem imaginary_part_of_pure_imaginary (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) / (a - Complex.I)
  (z.re = 0) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_l3108_310855


namespace NUMINAMATH_CALUDE_hyperbola_condition_exclusive_or_condition_l3108_310816

def P (a : ℝ) : Prop := ∀ x, x^2 - a*x + a + 5/4 > 0

def Q (a : ℝ) : Prop := ∃ x y, x^2 / (4*a + 7) + y^2 / (a - 3) = 1

theorem hyperbola_condition (a : ℝ) : Q a ↔ a ∈ Set.Ioo (-7/4 : ℝ) 3 := by sorry

theorem exclusive_or_condition (a : ℝ) : (¬(P a ∧ Q a) ∧ (P a ∨ Q a)) ↔ 
  a ∈ Set.Ioc (-7/4 : ℝ) (-1) ∪ Set.Ico 3 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_exclusive_or_condition_l3108_310816


namespace NUMINAMATH_CALUDE_b_work_days_l3108_310845

/-- The number of days it takes for worker A to complete the work alone -/
def a_days : ℝ := 15

/-- The number of days A and B work together -/
def together_days : ℝ := 8

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.06666666666666665

/-- The number of days it takes for worker B to complete the work alone -/
def b_days : ℝ := 20

/-- Theorem stating that given the conditions, B can complete the work alone in 20 days -/
theorem b_work_days : 
  together_days * (1 / a_days + 1 / b_days) = 1 - work_left :=
sorry

end NUMINAMATH_CALUDE_b_work_days_l3108_310845


namespace NUMINAMATH_CALUDE_gcd_of_256_180_720_l3108_310811

theorem gcd_of_256_180_720 : Nat.gcd 256 (Nat.gcd 180 720) = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_720_l3108_310811


namespace NUMINAMATH_CALUDE_favorite_toy_change_probability_l3108_310897

def toy_count : ℕ := 10
def min_price : ℚ := 1/2
def max_price : ℚ := 5
def price_increment : ℚ := 1/2
def initial_quarters : ℕ := 10
def favorite_toy_price : ℚ := 9/2

def toy_prices : List ℚ := 
  List.range toy_count |>.map (λ i => max_price - i * price_increment)

theorem favorite_toy_change_probability :
  let total_sequences := toy_count.factorial
  let favorable_sequences := (toy_count - 1).factorial + (toy_count - 2).factorial
  (1 : ℚ) - (favorable_sequences : ℚ) / total_sequences = 8/9 :=
sorry

end NUMINAMATH_CALUDE_favorite_toy_change_probability_l3108_310897


namespace NUMINAMATH_CALUDE_expression_simplification_l3108_310879

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 2) :
  (1 / (a - 2) - 2 / (a^2 - 4)) / ((a^2 - 2*a) / (a^2 - 4)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3108_310879


namespace NUMINAMATH_CALUDE_line_length_in_sphere_l3108_310827

/-- Given a sphere and three perpendicular lines intersecting inside it, 
    prove the length of one line given conditions on the others. -/
theorem line_length_in_sphere (r : ℝ) (d_m : ℝ) (bb₁ : ℝ) (c_ratio : ℝ) :
  r = 11 ∧ 
  d_m = Real.sqrt 59 ∧
  bb₁ = 18 ∧ 
  c_ratio = (8 + Real.sqrt 2) / (8 - Real.sqrt 2) →
  ∃ (aa₁ : ℝ), aa₁ = 20 := by
sorry

end NUMINAMATH_CALUDE_line_length_in_sphere_l3108_310827


namespace NUMINAMATH_CALUDE_cost_of_fifty_roses_l3108_310892

/-- Represents the cost of a bouquet of roses -/
def bouquetCost (roses : ℕ) : ℚ :=
  if roses ≤ 30 then
    (30 : ℚ) / 15 * roses
  else
    (30 : ℚ) / 15 * 30 + (30 : ℚ) / 15 / 2 * (roses - 30)

/-- The theorem stating the cost of a bouquet with 50 roses -/
theorem cost_of_fifty_roses : bouquetCost 50 = 80 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_fifty_roses_l3108_310892


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3108_310824

theorem algebraic_expression_value (p q : ℤ) :
  p * 3^3 + 3 * q + 1 = 2015 →
  p * (-3)^3 - 3 * q + 1 = -2013 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3108_310824


namespace NUMINAMATH_CALUDE_k_domain_l3108_310881

noncomputable def k (x : ℝ) : ℝ := 1 / (x + 3) + 1 / (x^2 + 3) + 1 / (x^3 + 3)

def domain_k : Set ℝ := {x | x ≠ -3 ∧ x ≠ -Real.rpow 3 (1/3)}

theorem k_domain :
  {x : ℝ | ∃ y, k x = y} = domain_k :=
sorry

end NUMINAMATH_CALUDE_k_domain_l3108_310881


namespace NUMINAMATH_CALUDE_trees_in_yard_l3108_310857

/-- The number of trees in a yard with given conditions -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem stating the number of trees in the yard under given conditions -/
theorem trees_in_yard :
  let yard_length : ℕ := 150
  let tree_distance : ℕ := 15
  number_of_trees yard_length tree_distance = 11 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l3108_310857


namespace NUMINAMATH_CALUDE_custom_op_M_T_l3108_310880

def custom_op (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def T : Set ℝ := {x | x < 2}

theorem custom_op_M_T :
  custom_op M T = {x | x < -1 ∨ (2 ≤ x ∧ x ≤ 4)} :=
by sorry

end NUMINAMATH_CALUDE_custom_op_M_T_l3108_310880


namespace NUMINAMATH_CALUDE_arman_age_to_40_l3108_310895

/-- Given that Arman is six times older than his sister and his sister was 2 years old four years ago,
    prove that Arman will be 40 years old in 4 years. -/
theorem arman_age_to_40 (arman_age sister_age : ℕ) : 
  sister_age = 2 + 4 →  -- Sister's current age
  arman_age = 6 * sister_age →  -- Arman's current age
  40 - arman_age = 4 :=
by sorry

end NUMINAMATH_CALUDE_arman_age_to_40_l3108_310895


namespace NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l3108_310885

/-- The polar to Cartesian conversion theorem for a specific curve -/
theorem polar_to_cartesian_conversion (ρ θ x y : ℝ) :
  (ρ * (Real.cos θ)^2 = 2 * Real.sin θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  x^2 = 2*y := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l3108_310885


namespace NUMINAMATH_CALUDE_base5_132_to_base10_l3108_310874

/-- Converts a base-5 digit to its base-10 equivalent --/
def base5ToBase10Digit (d : Nat) : Nat :=
  if d < 5 then d else 0

/-- Converts a 3-digit base-5 number to its base-10 equivalent --/
def base5ToBase10 (d2 d1 d0 : Nat) : Nat :=
  (base5ToBase10Digit d2) * 25 + (base5ToBase10Digit d1) * 5 + (base5ToBase10Digit d0)

/-- Theorem stating that the base-10 representation of the base-5 number 132 is 42 --/
theorem base5_132_to_base10 : base5ToBase10 1 3 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_base5_132_to_base10_l3108_310874


namespace NUMINAMATH_CALUDE_olivia_atm_withdrawal_l3108_310829

theorem olivia_atm_withdrawal (initial_amount spent_amount final_amount : ℕ) 
  (h1 : initial_amount = 100)
  (h2 : spent_amount = 89)
  (h3 : final_amount = 159) : 
  initial_amount + (spent_amount + final_amount) - initial_amount = 148 := by
  sorry

end NUMINAMATH_CALUDE_olivia_atm_withdrawal_l3108_310829


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3108_310802

theorem lcm_from_product_and_hcf (a b : ℕ+) 
  (h_product : a * b = 145862784)
  (h_hcf : Nat.gcd a b = 792) :
  Nat.lcm a b = 184256 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3108_310802


namespace NUMINAMATH_CALUDE_mouse_jump_distance_l3108_310851

/-- The jump distances of animals in a contest -/
def JumpContest (grasshopper frog mouse : ℕ) : Prop :=
  (grasshopper = 39) ∧ 
  (grasshopper = frog + 19) ∧
  (frog = mouse + 12)

/-- Theorem: Given the conditions of the jump contest, the mouse jumped 8 inches -/
theorem mouse_jump_distance (grasshopper frog mouse : ℕ) 
  (h : JumpContest grasshopper frog mouse) : mouse = 8 := by
  sorry

end NUMINAMATH_CALUDE_mouse_jump_distance_l3108_310851


namespace NUMINAMATH_CALUDE_is_quadratic_x_squared_minus_3x_plus_2_l3108_310810

/-- Definition of a quadratic equation -/
def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ (x : ℝ), a * x^2 + b * x + c = 0

/-- The equation x² - 3x + 2 = 0 is a quadratic equation -/
theorem is_quadratic_x_squared_minus_3x_plus_2 :
  is_quadratic_equation 1 (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_is_quadratic_x_squared_minus_3x_plus_2_l3108_310810


namespace NUMINAMATH_CALUDE_face_washing_unit_is_liters_l3108_310800

/-- Represents units of volume measurement -/
inductive VolumeUnit
  | Liters
  | Milliliters
  | Grams

/-- Represents the amount of water used for face washing -/
def face_washing_amount : ℝ := 2

/-- Determines if a given volume unit is appropriate for face washing -/
def is_appropriate_unit (unit : VolumeUnit) : Prop :=
  match unit with
  | VolumeUnit.Liters => true
  | _ => false

theorem face_washing_unit_is_liters :
  is_appropriate_unit VolumeUnit.Liters = true :=
sorry

end NUMINAMATH_CALUDE_face_washing_unit_is_liters_l3108_310800


namespace NUMINAMATH_CALUDE_max_radius_circle_through_points_l3108_310820

/-- A circle in a rectangular coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def lieOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The maximum possible radius of a circle passing through (16, 0) and (-16, 0) is 16 -/
theorem max_radius_circle_through_points :
  ∃ (c : Circle), lieOnCircle c (16, 0) ∧ lieOnCircle c (-16, 0) →
  ∀ (c' : Circle), lieOnCircle c' (16, 0) ∧ lieOnCircle c' (-16, 0) →
  c'.radius ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_radius_circle_through_points_l3108_310820


namespace NUMINAMATH_CALUDE_rectangle_area_l3108_310837

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width * rectangle_width = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3108_310837


namespace NUMINAMATH_CALUDE_relationship_abc_l3108_310846

-- Define the constants
noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.8 / Real.log 1.2
noncomputable def c : ℝ := (1.5 : ℝ) ^ (1/2)

-- State the theorem
theorem relationship_abc : c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3108_310846


namespace NUMINAMATH_CALUDE_problem_2009_2007_2008_l3108_310876

theorem problem_2009_2007_2008 : 2009 * (2007 / 2008) + 1 / 2008 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_problem_2009_2007_2008_l3108_310876


namespace NUMINAMATH_CALUDE_certain_number_exists_and_is_one_l3108_310822

theorem certain_number_exists_and_is_one : 
  ∃ (x : ℕ), x > 0 ∧ (57 * x) % 8 = 7 ∧ ∀ (y : ℕ), y > 0 ∧ (57 * y) % 8 = 7 → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_and_is_one_l3108_310822


namespace NUMINAMATH_CALUDE_prob_sum_le_10_prob_sum_le_10_is_11_12_l3108_310805

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of outcomes where the sum is greater than 10 -/
def outcomesGreaterThan10 : ℕ := 3

/-- The probability that the sum of two fair six-sided dice is less than or equal to 10 -/
theorem prob_sum_le_10 : ℚ :=
  1 - (outcomesGreaterThan10 : ℚ) / totalOutcomes

/-- Proof that the probability of the sum of two fair six-sided dice being less than or equal to 10 is 11/12 -/
theorem prob_sum_le_10_is_11_12 : prob_sum_le_10 = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_le_10_prob_sum_le_10_is_11_12_l3108_310805


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3108_310843

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/5
  let r : ℚ := 1/5
  let n : ℕ := 7
  geometric_sum a r n = 78124/312500 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3108_310843


namespace NUMINAMATH_CALUDE_inequality_and_function_property_l3108_310894

def f (x : ℝ) := |x - 1|

theorem inequality_and_function_property :
  (∀ x : ℝ, f x + f (x + 4) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → a ≠ 0 → f (a * b) > |a| * f (b / a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_function_property_l3108_310894


namespace NUMINAMATH_CALUDE_deck_size_proof_l3108_310806

theorem deck_size_proof (r b : ℕ) : 
  r / (r + b : ℚ) = 1/4 →
  r / (r + (b + 6) : ℚ) = 1/5 →
  r + b = 24 :=
by sorry

end NUMINAMATH_CALUDE_deck_size_proof_l3108_310806


namespace NUMINAMATH_CALUDE_compute_expression_l3108_310844

theorem compute_expression : 25 * (216 / 3 + 36 / 6 + 16 / 25 + 2) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3108_310844


namespace NUMINAMATH_CALUDE_computer_cost_l3108_310808

theorem computer_cost (initial_amount printer_cost amount_left : ℕ) 
  (h1 : initial_amount = 450)
  (h2 : printer_cost = 40)
  (h3 : amount_left = 10) :
  initial_amount - printer_cost - amount_left = 400 :=
by sorry

end NUMINAMATH_CALUDE_computer_cost_l3108_310808


namespace NUMINAMATH_CALUDE_magnitude_of_linear_combination_l3108_310853

/-- Given two planar unit vectors with a right angle between them, 
    prove that the magnitude of 3 times the first vector plus 4 times the second vector is 5. -/
theorem magnitude_of_linear_combination (m n : ℝ × ℝ) : 
  ‖m‖ = 1 → ‖n‖ = 1 → m • n = 0 → ‖3 • m + 4 • n‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_linear_combination_l3108_310853


namespace NUMINAMATH_CALUDE_y_coordinate_relationship_l3108_310866

/-- A parabola defined by y = 2(x+1)² + c -/
structure Parabola where
  c : ℝ

/-- A point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = 2 * (x + 1)^2 + p.c

/-- Theorem stating the relationship between y-coordinates of three points on the parabola -/
theorem y_coordinate_relationship (p : Parabola) 
  (A : PointOnParabola p) (B : PointOnParabola p) (C : PointOnParabola p)
  (hA : A.x = -2) (hB : B.x = 1) (hC : C.x = 2) :
  C.y > B.y ∧ B.y > A.y := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_relationship_l3108_310866


namespace NUMINAMATH_CALUDE_loss_per_metre_is_12_l3108_310871

/-- Calculates the loss per metre of cloth given the total metres sold, total selling price, and cost price per metre. -/
def loss_per_metre (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) : ℕ :=
  let total_cost_price := total_metres * cost_price_per_metre
  let total_loss := total_cost_price - total_selling_price
  total_loss / total_metres

/-- Theorem stating that the loss per metre of cloth is 12 given the specified conditions. -/
theorem loss_per_metre_is_12 :
  loss_per_metre 200 12000 72 = 12 := by
  sorry

end NUMINAMATH_CALUDE_loss_per_metre_is_12_l3108_310871


namespace NUMINAMATH_CALUDE_smallest_four_digit_number_l3108_310873

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def T (n : ℕ) : ℕ := (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def S (n : ℕ) : ℕ := (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_number (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) :
  ∃ (x : ℕ), is_four_digit x ∧ T x = p^k ∧ S x = p^p - 5 ∧
  ∀ (y : ℕ), is_four_digit y ∧ T y = p^k ∧ S y = p^p - 5 → x ≤ y :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_number_l3108_310873


namespace NUMINAMATH_CALUDE_eagles_per_section_l3108_310849

theorem eagles_per_section 
  (total_eagles : ℕ) 
  (total_sections : ℕ) 
  (h1 : total_eagles = 18) 
  (h2 : total_sections = 3) 
  (h3 : total_eagles % total_sections = 0) : 
  total_eagles / total_sections = 6 := by
  sorry

end NUMINAMATH_CALUDE_eagles_per_section_l3108_310849


namespace NUMINAMATH_CALUDE_wheel_speed_calculation_l3108_310834

/-- Prove that given a wheel with a circumference of 8 feet, if reducing the time
    for a complete rotation by 0.5 seconds increases the speed by 6 miles per hour,
    then the original speed of the wheel is 9 miles per hour. -/
theorem wheel_speed_calculation (r : ℝ) : 
  let circumference : ℝ := 8 / 5280  -- circumference in miles
  let t : ℝ := circumference * 3600 / r  -- time for one rotation in seconds
  let new_t : ℝ := t - 0.5  -- new time after reduction
  let new_r : ℝ := r + 6  -- new speed after increase
  (new_r * new_t / 3600 = circumference) →  -- equation for new speed and time
  r = 9 := by
sorry

end NUMINAMATH_CALUDE_wheel_speed_calculation_l3108_310834


namespace NUMINAMATH_CALUDE_min_value_product_quotient_min_value_achieved_l3108_310889

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 4*x + 2) * (y^2 + 5*y + 3) * (z^2 + 6*z + 4) / (x*y*z) ≥ 336 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 4*a + 2) * (b^2 + 5*b + 3) * (c^2 + 6*c + 4) / (a*b*c) = 336 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_min_value_achieved_l3108_310889


namespace NUMINAMATH_CALUDE_elmwood_population_l3108_310854

/-- The number of cities in the County of Elmwood -/
def num_cities : ℕ := 25

/-- The lower bound of the average population per city -/
def avg_pop_lower : ℕ := 3200

/-- The upper bound of the average population per city -/
def avg_pop_upper : ℕ := 3700

/-- The total population of the County of Elmwood -/
def total_population : ℕ := 86250

theorem elmwood_population :
  ∃ (avg_pop : ℚ),
    avg_pop > avg_pop_lower ∧
    avg_pop < avg_pop_upper ∧
    (num_cities : ℚ) * avg_pop = total_population :=
sorry

end NUMINAMATH_CALUDE_elmwood_population_l3108_310854


namespace NUMINAMATH_CALUDE_triangle_area_three_lines_l3108_310883

/-- The area of the triangle formed by the intersection of three lines -/
theorem triangle_area_three_lines : 
  let line1 : ℝ → ℝ := λ x => 3 * x - 4
  let line2 : ℝ → ℝ := λ x => -2 * x + 16
  let y_axis : ℝ → ℝ := λ x => 0
  let intersection_x : ℝ := (16 + 4) / (3 + 2)
  let intersection_y : ℝ := line1 intersection_x
  let y_intercept1 : ℝ := line1 0
  let y_intercept2 : ℝ := line2 0
  let base : ℝ := y_intercept2 - y_intercept1
  let height : ℝ := intersection_x
  let area : ℝ := (1/2) * base * height
  area = 40 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_three_lines_l3108_310883


namespace NUMINAMATH_CALUDE_integer_ratio_condition_l3108_310804

theorem integer_ratio_condition (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ a ∈ S, ∃ k : ℤ, (a^m + a - 1 : ℤ) = k * (a^n + a^2 - 1)) →
  (m = n + 2 ∧ m = 5 ∧ n = 3) :=
by sorry

end NUMINAMATH_CALUDE_integer_ratio_condition_l3108_310804


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3108_310841

-- First equation: 3x = 2x + 12
theorem solve_equation_one : ∃ x : ℝ, 3 * x = 2 * x + 12 ∧ x = 12 := by sorry

-- Second equation: x/2 - 3 = 5
theorem solve_equation_two : ∃ x : ℝ, x / 2 - 3 = 5 ∧ x = 16 := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3108_310841


namespace NUMINAMATH_CALUDE_parallel_vector_component_l3108_310826

/-- Given two vectors a and b in R², if a is parallel to a + 2b, then the second component of a is -4. -/
theorem parallel_vector_component (l m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, m]
  let b : Fin 2 → ℝ := ![l, -2]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • (a + 2 • b)) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vector_component_l3108_310826


namespace NUMINAMATH_CALUDE_quadratic_monotone_decreasing_condition_l3108_310875

/-- A quadratic function f(x) = x^2 + ax + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

/-- The function is monotonically decreasing in the interval (-∞, 3) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 3 → y < 3 → f a x > f a y

/-- The theorem states that if f is monotonically decreasing in (-∞, 3), then a ∈ (-∞, -6] -/
theorem quadratic_monotone_decreasing_condition (a : ℝ) :
  is_monotone_decreasing a → a ≤ -6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotone_decreasing_condition_l3108_310875


namespace NUMINAMATH_CALUDE_power_two_2014_mod_7_l3108_310818

theorem power_two_2014_mod_7 :
  ∃ (k : ℤ), 2^2014 = 7 * k + 9 := by sorry

end NUMINAMATH_CALUDE_power_two_2014_mod_7_l3108_310818


namespace NUMINAMATH_CALUDE_impossible_to_guarantee_same_state_l3108_310860

/-- Represents the state of a usamon (has an electron or not) -/
inductive UsamonState
| HasElectron
| NoElectron

/-- Represents a usamon with its current state -/
structure Usamon :=
  (state : UsamonState)

/-- Represents the action of connecting a diode between two usamons -/
def connectDiode (a b : Usamon) : Usamon × Usamon :=
  match a.state, b.state with
  | UsamonState.HasElectron, UsamonState.NoElectron => 
      ({ state := UsamonState.NoElectron }, { state := UsamonState.HasElectron })
  | _, _ => (a, b)

/-- The main theorem stating that it's impossible to guarantee two usamons are in the same state -/
theorem impossible_to_guarantee_same_state (usamons : Fin 2015 → Usamon) :
  ∀ (sequence : List (Fin 2015 × Fin 2015)),
  ¬∃ (i j : Fin 2015), i ≠ j ∧ (usamons i).state = (usamons j).state := by
  sorry


end NUMINAMATH_CALUDE_impossible_to_guarantee_same_state_l3108_310860


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3108_310819

/-- A parallelogram with vertices A, B, C, and D in 2D space -/
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

/-- The property that the diagonals of a parallelogram bisect each other -/
def diagonals_bisect (p : Parallelogram) : Prop :=
  let midpoint_AD := ((p.A.1 + p.D.1) / 2, (p.A.2 + p.D.2) / 2)
  let midpoint_BC := ((p.B.1 + p.C.1) / 2, (p.B.2 + p.C.2) / 2)
  midpoint_AD = midpoint_BC

/-- The sum of coordinates of a point -/
def sum_coordinates (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

/-- The main theorem -/
theorem parallelogram_vertex_sum :
  ∀ (p : Parallelogram),
    p.A = (2, 3) →
    p.B = (5, 7) →
    p.D = (11, -1) →
    diagonals_bisect p →
    sum_coordinates p.C = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3108_310819


namespace NUMINAMATH_CALUDE_log_sum_simplification_l3108_310809

theorem log_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l3108_310809
