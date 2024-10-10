import Mathlib

namespace dolly_initial_tickets_l3171_317199

/- Define the number of rides for each attraction -/
def ferris_wheel_rides : Nat := 2
def roller_coaster_rides : Nat := 3
def log_ride_rides : Nat := 7

/- Define the ticket cost for each attraction -/
def ferris_wheel_cost : Nat := 2
def roller_coaster_cost : Nat := 5
def log_ride_cost : Nat := 1

/- Define the additional tickets needed -/
def additional_tickets : Nat := 6

/- Theorem to prove -/
theorem dolly_initial_tickets : 
  (ferris_wheel_rides * ferris_wheel_cost + 
   roller_coaster_rides * roller_coaster_cost + 
   log_ride_rides * log_ride_cost) - 
  additional_tickets = 20 := by
  sorry

end dolly_initial_tickets_l3171_317199


namespace population_growth_l3171_317101

theorem population_growth (initial_population : ℝ) : 
  (initial_population * (1 + 0.1)^2 = 16940) → initial_population = 14000 := by
  sorry

end population_growth_l3171_317101


namespace expression_equality_l3171_317168

theorem expression_equality : 
  (Real.sqrt (4/3) + Real.sqrt 3) * Real.sqrt 6 - (Real.sqrt 20 - Real.sqrt 5) / Real.sqrt 5 = 5 * Real.sqrt 2 - 1 := by
  sorry

end expression_equality_l3171_317168


namespace factorial_2007_properties_l3171_317120

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def trailingZeros (n : ℕ) : ℕ :=
  (List.range 4).foldl (λ acc i => acc + n / (5 ^ (i + 1))) 0

def lastNonZeroDigit (n : ℕ) : ℕ := n % 10

theorem factorial_2007_properties :
  trailingZeros (factorial 2007) = 500 ∧
  lastNonZeroDigit (factorial 2007 / (10 ^ trailingZeros (factorial 2007))) = 2 := by
  sorry

end factorial_2007_properties_l3171_317120


namespace ellipse_x_intersection_l3171_317152

/-- Definition of the ellipse based on the given conditions -/
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (0, 1)
  let F₂ : ℝ × ℝ := (4, 0)
  let d : ℝ := Real.sqrt 2 + 3
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = d

/-- The theorem stating the other intersection point of the ellipse with the x-axis -/
theorem ellipse_x_intersection :
  ellipse (1, 0) →
  ∃ x : ℝ, x ≠ 1 ∧ ellipse (x, 0) ∧ x = 3 * Real.sqrt 2 / 4 + 1 := by
  sorry

end ellipse_x_intersection_l3171_317152


namespace altered_detergent_amount_is_180_l3171_317176

/-- Represents the ratio of components in a cleaning solution -/
structure CleaningSolution :=
  (bleach : ℚ)
  (detergent : ℚ)
  (fabricSoftener : ℚ)
  (water : ℚ)

/-- Calculates the amount of detergent in the altered solution -/
def alteredDetergentAmount (original : CleaningSolution) (alteredWaterAmount : ℚ) : ℚ :=
  let bleachToDetergentRatio := original.bleach / original.detergent * 3
  let fabricSoftenerToDetergentRatio := (original.fabricSoftener / original.detergent) / 2
  let detergentToWaterRatio := (original.detergent / original.water) * (2/3)
  
  let newDetergentToWaterRatio := detergentToWaterRatio * alteredWaterAmount
  
  newDetergentToWaterRatio

/-- Theorem stating that the altered solution contains 180 liters of detergent -/
theorem altered_detergent_amount_is_180 :
  let original := CleaningSolution.mk 4 40 60 100
  let alteredWaterAmount := 300
  alteredDetergentAmount original alteredWaterAmount = 180 := by
  sorry

end altered_detergent_amount_is_180_l3171_317176


namespace factorization_proof_l3171_317189

theorem factorization_proof (x y : ℝ) : x^2 - 2*x^2*y + x*y^2 = x*(x - 2*x*y + y^2) := by
  sorry

end factorization_proof_l3171_317189


namespace garden_plant_count_l3171_317103

/-- The number of plants in a garden with given rows and columns -/
def garden_plants (rows : ℕ) (columns : ℕ) : ℕ := rows * columns

/-- Theorem: A garden with 52 rows and 15 columns has 780 plants -/
theorem garden_plant_count : garden_plants 52 15 = 780 := by
  sorry

end garden_plant_count_l3171_317103


namespace max_product_ab_l3171_317143

theorem max_product_ab (a b : ℝ) 
  (h : ∀ x : ℝ, Real.exp x ≥ a * (x - 1) + b) : 
  a * b ≤ (1/2) * Real.exp 3 := by
sorry

end max_product_ab_l3171_317143


namespace monomial_properties_l3171_317140

-- Define the structure of a monomial
structure Monomial (α : Type*) [Field α] where
  coeff : α
  x_exp : ℕ
  y_exp : ℕ

-- Define the given monomial
def given_monomial : Monomial ℚ := {
  coeff := -1/7,
  x_exp := 2,
  y_exp := 1
}

-- Define the coefficient of a monomial
def coefficient (m : Monomial ℚ) : ℚ := m.coeff

-- Define the degree of a monomial
def degree (m : Monomial ℚ) : ℕ := m.x_exp + m.y_exp

-- Theorem statement
theorem monomial_properties :
  coefficient given_monomial = -1/7 ∧ degree given_monomial = 3 := by
  sorry

end monomial_properties_l3171_317140


namespace phantom_needs_126_more_l3171_317100

/-- The amount of additional money Phantom needs to buy printer inks -/
def additional_money_needed (initial_money : ℕ) 
  (black_price red_price yellow_price blue_price : ℕ) 
  (black_quantity red_quantity yellow_quantity blue_quantity : ℕ) : ℕ :=
  let total_cost := black_price * black_quantity + 
                    red_price * red_quantity + 
                    yellow_price * yellow_quantity + 
                    blue_price * blue_quantity
  total_cost - initial_money

/-- Theorem stating that Phantom needs $126 more to buy the printer inks -/
theorem phantom_needs_126_more : 
  additional_money_needed 50 12 16 14 17 3 4 3 2 = 126 := by
  sorry

end phantom_needs_126_more_l3171_317100


namespace total_dolls_count_l3171_317121

theorem total_dolls_count (big_box_capacity : ℕ) (small_box_capacity : ℕ) 
                          (big_box_count : ℕ) (small_box_count : ℕ) 
                          (h1 : big_box_capacity = 7)
                          (h2 : small_box_capacity = 4)
                          (h3 : big_box_count = 5)
                          (h4 : small_box_count = 9) :
  big_box_capacity * big_box_count + small_box_capacity * small_box_count = 71 :=
by sorry

end total_dolls_count_l3171_317121


namespace hyperbola_equation_l3171_317128

theorem hyperbola_equation (a c : ℝ) (h1 : a = 5) (h2 : c = 7) :
  ∃ (x y : ℝ), (x^2 / 25 - y^2 / 24 = 1) ∨ (y^2 / 25 - x^2 / 24 = 1) := by
  sorry

end hyperbola_equation_l3171_317128


namespace polygon_perimeter_l3171_317124

/-- The perimeter of a polygon formed by cutting a right triangle from a rectangle --/
theorem polygon_perimeter (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c < 10) : 
  2 * (10 + (10 - b)) - a = 29 :=
by sorry

end polygon_perimeter_l3171_317124


namespace sqrt_simplification_complex_expression_simplification_square_difference_simplification_l3171_317109

-- Problem 1
theorem sqrt_simplification :
  Real.sqrt 27 + Real.sqrt 3 - Real.sqrt 12 = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem complex_expression_simplification :
  1 / Real.sqrt 24 + |Real.sqrt 6 - 3| + (1 / 2)⁻¹ - 2016^0 = 4 - 13 * Real.sqrt 6 / 12 := by sorry

-- Problem 3
theorem square_difference_simplification :
  (Real.sqrt 3 + Real.sqrt 2)^2 - (Real.sqrt 3 - Real.sqrt 2)^2 = 4 * Real.sqrt 6 := by sorry

end sqrt_simplification_complex_expression_simplification_square_difference_simplification_l3171_317109


namespace distance_A_P_main_theorem_l3171_317111

/-- A rectangle with two equilateral triangles positioned on its sides -/
structure TrianglesOnRectangle where
  /-- The length of side YC of rectangle YQZC -/
  yc : ℝ
  /-- The length of side CZ of rectangle YQZC -/
  cz : ℝ
  /-- The side length of equilateral triangles ABC and PQR -/
  triangle_side : ℝ
  /-- Assumption that YC = 8 -/
  yc_eq : yc = 8
  /-- Assumption that CZ = 15 -/
  cz_eq : cz = 15
  /-- Assumption that the side length of triangles is 9 -/
  triangle_side_eq : triangle_side = 9

/-- The distance between points A and P is 10 -/
theorem distance_A_P (t : TrianglesOnRectangle) : ℝ :=
  10

#check distance_A_P

/-- The main theorem stating that the distance between A and P is 10 -/
theorem main_theorem (t : TrianglesOnRectangle) : distance_A_P t = 10 := by
  sorry

end distance_A_P_main_theorem_l3171_317111


namespace derivative_parity_l3171_317138

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem derivative_parity (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (IsEven f → IsOdd f') ∧ (IsOdd f → IsEven f') := by sorry

end derivative_parity_l3171_317138


namespace sulfuric_acid_mixture_l3171_317181

/-- Proves that mixing 42 liters of 2% sulfuric acid solution with 18 liters of 12% sulfuric acid solution results in a 60-liter solution containing 5% sulfuric acid. -/
theorem sulfuric_acid_mixture :
  let solution1_volume : ℝ := 42
  let solution1_concentration : ℝ := 0.02
  let solution2_volume : ℝ := 18
  let solution2_concentration : ℝ := 0.12
  let total_volume : ℝ := solution1_volume + solution2_volume
  let total_acid : ℝ := solution1_volume * solution1_concentration + solution2_volume * solution2_concentration
  let final_concentration : ℝ := total_acid / total_volume
  total_volume = 60 ∧ final_concentration = 0.05 := by
  sorry

#check sulfuric_acid_mixture

end sulfuric_acid_mixture_l3171_317181


namespace questionnaire_C_count_l3171_317160

/-- Represents the total population size -/
def population_size : ℕ := 1000

/-- Represents the sample size -/
def sample_size : ℕ := 50

/-- Represents the first number drawn in the systematic sample -/
def first_number : ℕ := 8

/-- Represents the lower bound of the interval for questionnaire C -/
def lower_bound : ℕ := 751

/-- Represents the upper bound of the interval for questionnaire C -/
def upper_bound : ℕ := 1000

/-- Theorem stating that the number of people taking questionnaire C is 12 -/
theorem questionnaire_C_count :
  (Finset.filter (fun n => lower_bound ≤ (first_number + (n - 1) * (population_size / sample_size)) ∧
                           (first_number + (n - 1) * (population_size / sample_size)) ≤ upper_bound)
                 (Finset.range sample_size)).card = 12 :=
by sorry

end questionnaire_C_count_l3171_317160


namespace prime_divisibility_l3171_317148

theorem prime_divisibility (p a b : ℕ) : 
  Prime p → 
  p ≠ 3 → 
  a > 0 → 
  b > 0 → 
  p ∣ (a + b) → 
  p^2 ∣ (a^3 + b^3) → 
  p^2 ∣ (a + b) ∨ p^3 ∣ (a^3 + b^3) := by
sorry

end prime_divisibility_l3171_317148


namespace total_employee_purchase_price_l3171_317167

/-- Represents an item in the store -/
structure Item where
  name : String
  wholesale_cost : ℝ
  markup : ℝ
  employee_discount : ℝ

/-- Calculates the final price for an employee -/
def employee_price (item : Item) : ℝ :=
  item.wholesale_cost * (1 + item.markup) * (1 - item.employee_discount)

/-- The three items in the store -/
def video_recorder : Item :=
  { name := "Video Recorder", wholesale_cost := 200, markup := 0.20, employee_discount := 0.30 }

def digital_camera : Item :=
  { name := "Digital Camera", wholesale_cost := 150, markup := 0.25, employee_discount := 0.20 }

def smart_tv : Item :=
  { name := "Smart TV", wholesale_cost := 800, markup := 0.15, employee_discount := 0.25 }

/-- Theorem: The total amount paid by an employee for all three items is $1008 -/
theorem total_employee_purchase_price :
  employee_price video_recorder + employee_price digital_camera + employee_price smart_tv = 1008 := by
  sorry

end total_employee_purchase_price_l3171_317167


namespace four_digit_numbers_divisible_by_13_l3171_317119

theorem four_digit_numbers_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card + 1 = 693 := by
  sorry

end four_digit_numbers_divisible_by_13_l3171_317119


namespace min_value_of_complex_sum_l3171_317190

theorem min_value_of_complex_sum (z : ℂ) (h : Complex.abs (z + Complex.I) + Complex.abs (z - Complex.I) = 2) :
  ∃ (min_val : ℝ), min_val = 1 ∧ ∀ w : ℂ, Complex.abs (w + Complex.I) + Complex.abs (w - Complex.I) = 2 →
    Complex.abs (z + Complex.I + 1) ≤ Complex.abs (w + Complex.I + 1) :=
by sorry

end min_value_of_complex_sum_l3171_317190


namespace fraction_inequality_solution_l3171_317106

theorem fraction_inequality_solution (y : ℝ) : 
  1 / (y * (y + 2)) - 1 / ((y + 2) * (y + 4)) < 1 / 4 ↔ 
  y < -4 ∨ (-2 < y ∧ y < 0) ∨ y > 2 :=
sorry

end fraction_inequality_solution_l3171_317106


namespace pedal_triangle_largest_angle_l3171_317162

/-- Represents an acute triangle with vertices A, B, C and corresponding angles α, β, γ. -/
structure AcuteTriangle where
  α : Real
  β : Real
  γ : Real
  acute_angles : α ≤ β ∧ β ≤ γ ∧ γ < Real.pi / 2
  angle_sum : α + β + γ = Real.pi

/-- Represents the pedal triangle of an acute triangle. -/
def PedalTriangle (t : AcuteTriangle) : Prop :=
  ∃ (largest_pedal_angle : Real),
    largest_pedal_angle = Real.pi - 2 * t.α ∧
    largest_pedal_angle ≥ t.γ

/-- 
Theorem: The largest angle in the pedal triangle of an acute triangle is at least 
as large as the largest angle in the original triangle. Equality holds when the 
original triangle is isosceles with the equal angles at least 60°.
-/
theorem pedal_triangle_largest_angle (t : AcuteTriangle) : 
  PedalTriangle t ∧ 
  (Real.pi - 2 * t.α = t.γ ↔ t.α = t.β ∧ t.γ ≥ Real.pi / 3) := by
  sorry


end pedal_triangle_largest_angle_l3171_317162


namespace x_plus_y_value_l3171_317178

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.sin y = 2010)
  (h2 : x + 2010 * Real.cos y = 2009)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 + Real.pi / 2 := by
sorry

end x_plus_y_value_l3171_317178


namespace turnip_bag_weights_l3171_317131

def bag_weights : List Nat := [13, 15, 16, 17, 21, 24]

def total_weight : Nat := bag_weights.sum

structure BagDistribution where
  turnip_weight : Nat
  onion_weights : List Nat
  carrot_weights : List Nat

def is_valid_distribution (d : BagDistribution) : Prop :=
  d.turnip_weight ∈ bag_weights ∧
  (d.onion_weights ++ d.carrot_weights).sum = total_weight - d.turnip_weight ∧
  d.carrot_weights.sum = 2 * d.onion_weights.sum ∧
  (d.onion_weights ++ d.carrot_weights).toFinset ⊆ bag_weights.toFinset.erase d.turnip_weight

theorem turnip_bag_weights :
  ∀ d : BagDistribution, is_valid_distribution d → d.turnip_weight = 13 ∨ d.turnip_weight = 16 := by
  sorry

end turnip_bag_weights_l3171_317131


namespace value_of_p_l3171_317196

theorem value_of_p (p q r : ℝ) 
  (sum_eq : p + q + r = 70)
  (p_eq : p = 2 * q)
  (q_eq : q = 3 * r) : 
  p = 42 := by
sorry

end value_of_p_l3171_317196


namespace min_value_xyz_product_min_value_achieved_l3171_317115

theorem min_value_xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x + 2 * y) * (y + 2 * z) * (x * z + 1) ≥ 16 :=
sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 2 * y₀) * (y₀ + 2 * z₀) * (x₀ * z₀ + 1) = 16 :=
sorry

end min_value_xyz_product_min_value_achieved_l3171_317115


namespace line_counting_theorem_l3171_317108

theorem line_counting_theorem (n : ℕ) : 
  n > 0 → 
  n % 4 = 3 → 
  (∀ k : ℕ, k ≤ n → k % 4 = (if k % 4 = 0 then 4 else k % 4)) → 
  n = 47 := by
sorry

end line_counting_theorem_l3171_317108


namespace intersection_value_l3171_317193

/-- The value of k for which the lines -3x + y = k and 2x + y = 8 intersect when x = -6 -/
theorem intersection_value : ∃ k : ℝ, 
  (∀ x y : ℝ, -3*x + y = k ∧ 2*x + y = 8 → x = -6) → k = 38 := by
  sorry

end intersection_value_l3171_317193


namespace scientific_notation_170000_l3171_317166

theorem scientific_notation_170000 :
  170000 = 1.7 * (10 : ℝ)^5 := by sorry

end scientific_notation_170000_l3171_317166


namespace factorial_ratio_l3171_317145

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end factorial_ratio_l3171_317145


namespace can_mark_any_rational_ratio_l3171_317146

/-- Represents the ability to mark points on a segment -/
structure SegmentMarker where
  /-- Mark a point that divides a segment in half -/
  mark_half : ∀ (a b : ℝ), ∃ (c : ℝ), c = (a + b) / 2
  /-- Mark a point that divides a segment in the ratio n:(n+1) -/
  mark_ratio : ∀ (a b : ℝ) (n : ℕ), ∃ (c : ℝ), (c - a) / (b - c) = n / (n + 1)

/-- Theorem stating that with given marking abilities, any rational ratio can be achieved -/
theorem can_mark_any_rational_ratio (marker : SegmentMarker) :
  ∀ (a b : ℝ) (p q : ℕ), ∃ (c : ℝ), (c - a) / (b - c) = p / q :=
sorry

end can_mark_any_rational_ratio_l3171_317146


namespace beths_crayon_packs_l3171_317164

/-- The number of crayon packs Beth has after distribution and finding more -/
def beths_total_packs (initial_packs : ℚ) (total_friends : ℕ) (new_packs : ℚ) : ℚ :=
  (initial_packs / total_friends) + new_packs

/-- Theorem stating Beth's total packs under the given conditions -/
theorem beths_crayon_packs : 
  beths_total_packs 4 10 6 = 6.4 := by sorry

end beths_crayon_packs_l3171_317164


namespace multiplication_problem_l3171_317185

theorem multiplication_problem : 8 * (1 / 15) * 30 * 3 = 48 := by
  sorry

end multiplication_problem_l3171_317185


namespace no_basic_operation_satisfies_equation_l3171_317194

def basic_operations := [Int.add, Int.sub, Int.mul, Int.div]

theorem no_basic_operation_satisfies_equation :
  ∀ op ∈ basic_operations, (op 8 2) - 5 + 7 - (3^2 - 4) ≠ 6 := by
  sorry

end no_basic_operation_satisfies_equation_l3171_317194


namespace triangle_third_side_l3171_317133

theorem triangle_third_side (a b m : ℝ) (ha : a = 11) (hb : b = 23) (hm : m = 10) :
  ∃ c : ℝ, c = 30 ∧ m^2 = (2*a^2 + 2*b^2 - c^2) / 4 :=
sorry

end triangle_third_side_l3171_317133


namespace polynomial_evaluation_l3171_317132

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 10 = 0 ∧ x^3 - 3*x^2 - 9*x + 7 = 12 := by
  sorry

end polynomial_evaluation_l3171_317132


namespace profit_function_satisfies_conditions_max_profit_at_45_profit_function_is_quadratic_l3171_317161

/-- The profit function for a toy store -/
def profit_function (x : ℝ) : ℝ := -2 * (x - 30) * (x - 60)

/-- The theorem stating that the profit function satisfies all required conditions -/
theorem profit_function_satisfies_conditions :
  (profit_function 30 = 0) ∧ 
  (∃ (max_profit : ℝ), max_profit = profit_function 45 ∧ 
    ∀ (x : ℝ), profit_function x ≤ max_profit) ∧
  (profit_function 45 = 450) ∧
  (profit_function 60 = 0) := by
  sorry

/-- The maximum profit occurs at x = 45 -/
theorem max_profit_at_45 :
  ∀ (x : ℝ), profit_function x ≤ profit_function 45 := by
  sorry

/-- The profit function is a quadratic function -/
theorem profit_function_is_quadratic :
  ∃ (a b c : ℝ), ∀ (x : ℝ), profit_function x = a * x^2 + b * x + c := by
  sorry

end profit_function_satisfies_conditions_max_profit_at_45_profit_function_is_quadratic_l3171_317161


namespace tangent_line_at_2_monotonicity_intervals_l3171_317116

-- Define the function f(x) = 3x - x^3
def f (x : ℝ) : ℝ := 3*x - x^3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 - 3*x^2

-- Theorem for the tangent line at x = 2
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), m = -9 ∧ b = 18 ∧
  ∀ x, f x = m * (x - 2) + f 2 := by sorry

-- Theorem for monotonicity intervals
theorem monotonicity_intervals :
  (∀ x < -1, (f' x < 0)) ∧
  (∀ x ∈ Set.Ioo (-1) 1, (f' x > 0)) ∧
  (∀ x > 1, (f' x < 0)) := by sorry

end tangent_line_at_2_monotonicity_intervals_l3171_317116


namespace stephanies_age_to_jobs_age_ratio_l3171_317136

/-- Given the ages of Freddy, Stephanie, and Job, prove the ratio of Stephanie's age to Job's age -/
theorem stephanies_age_to_jobs_age_ratio :
  ∀ (freddy_age stephanie_age job_age : ℕ),
  freddy_age = 18 →
  stephanie_age = freddy_age + 2 →
  job_age = 5 →
  (stephanie_age : ℚ) / (job_age : ℚ) = 4 := by
  sorry

end stephanies_age_to_jobs_age_ratio_l3171_317136


namespace cranberry_juice_unit_cost_l3171_317123

/-- The unit cost of cranberry juice in cents per ounce -/
def unit_cost (total_cost : ℚ) (volume : ℚ) : ℚ :=
  total_cost / volume

/-- Theorem stating that the unit cost of cranberry juice is 7 cents per ounce -/
theorem cranberry_juice_unit_cost :
  let total_cost : ℚ := 84
  let volume : ℚ := 12
  unit_cost total_cost volume = 7 := by sorry

end cranberry_juice_unit_cost_l3171_317123


namespace inequality_proof_l3171_317197

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a * b ≤ 1/4 ∧ Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2 ∧ a^2 + b^2 ≥ 1/2 := by
  sorry

end inequality_proof_l3171_317197


namespace second_rewind_time_l3171_317169

theorem second_rewind_time (total_time first_segment first_rewind second_segment third_segment : ℕ) : 
  total_time = 120 ∧ 
  first_segment = 35 ∧ 
  first_rewind = 5 ∧ 
  second_segment = 45 ∧ 
  third_segment = 20 → 
  total_time - (first_segment + first_rewind + second_segment + third_segment) = 15 := by
  sorry

end second_rewind_time_l3171_317169


namespace ages_sum_l3171_317191

theorem ages_sum (a b c : ℕ+) (h1 : a = b) (h2 : a > c) (h3 : a * b * c = 72) : a + b + c = 14 := by
  sorry

end ages_sum_l3171_317191


namespace square_plot_area_l3171_317159

/-- Given a square plot with a perimeter that costs a certain amount to fence at a given price per foot, 
    this theorem proves that the area of the plot is as calculated. -/
theorem square_plot_area 
  (perimeter_cost : ℝ) 
  (price_per_foot : ℝ) 
  (perimeter_cost_positive : perimeter_cost > 0)
  (price_per_foot_positive : price_per_foot > 0)
  (h_cost : perimeter_cost = 3944)
  (h_price : price_per_foot = 58) : 
  (perimeter_cost / (4 * price_per_foot))^2 = 289 := by
  sorry

#eval (3944 / (4 * 58))^2  -- Should evaluate to 289.0

end square_plot_area_l3171_317159


namespace movie_theater_problem_l3171_317142

theorem movie_theater_problem (adult_price child_price : ℚ) 
  (total_people : ℕ) (total_paid : ℚ) : 
  adult_price = 9.5 → 
  child_price = 6.5 → 
  total_people = 7 → 
  total_paid = 54.5 → 
  ∃ (adults : ℕ), 
    adults ≤ total_people ∧ 
    (adult_price * adults + child_price * (total_people - adults) = total_paid) ∧
    adults = 3 :=
by sorry

end movie_theater_problem_l3171_317142


namespace only_q_is_true_l3171_317147

theorem only_q_is_true (p q m : Prop) 
  (h1 : (p ∨ q ∨ m) ∧ (¬(p ∧ q) ∧ ¬(p ∧ m) ∧ ¬(q ∧ m)))  -- Only one of p, q, and m is true
  (h2 : (p ∨ ¬(p ∨ q) ∨ m) ∧ (¬(p ∧ ¬(p ∨ q)) ∧ ¬(p ∧ m) ∧ ¬(¬(p ∨ q) ∧ m)))  -- Only one judgment is incorrect
  : q := by
sorry


end only_q_is_true_l3171_317147


namespace two_digit_divisor_with_remainder_l3171_317114

theorem two_digit_divisor_with_remainder (x y : ℕ) : ∃! n : ℕ, 
  (0 < x ∧ x ≤ 9) ∧ 
  (0 ≤ y ∧ y ≤ 9) ∧
  (n = 10 * x + y) ∧
  (∃ q : ℕ, 491 = n * q + 59) ∧
  (n = 72) := by
sorry

end two_digit_divisor_with_remainder_l3171_317114


namespace paper_cutting_equations_l3171_317183

/-- Represents the paper cutting scenario in a seventh-grade class. -/
theorem paper_cutting_equations (x y : ℕ) : 
  (x + y = 12 ∧ 6 * x = 3 * (4 * y)) ↔ 
  (x = number_of_sheets_for_stars ∧ 
   y = number_of_sheets_for_flowers ∧ 
   total_sheets_used = 12 ∧ 
   stars_per_sheet = 6 ∧ 
   flowers_per_sheet = 4 ∧ 
   total_stars = 3 * total_flowers) :=
sorry

end paper_cutting_equations_l3171_317183


namespace common_remainder_difference_l3171_317130

theorem common_remainder_difference (d r : ℕ) : 
  d.Prime → 
  d > 1 → 
  r < d → 
  1274 % d = r → 
  1841 % d = r → 
  2866 % d = r → 
  d - r = 6 := by sorry

end common_remainder_difference_l3171_317130


namespace complex_number_location_l3171_317137

theorem complex_number_location (z : ℂ) (h : (2 + 3*I)*z = 1 + I) :
  (z.re > 0) ∧ (z.im < 0) :=
sorry

end complex_number_location_l3171_317137


namespace constant_remainder_iff_b_eq_neg_four_thirds_l3171_317144

-- Define the polynomials
def f (b : ℚ) (x : ℚ) : ℚ := 12 * x^3 - 9 * x^2 + b * x + 8
def g (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

-- Define the remainder function
def remainder (b : ℚ) (x : ℚ) : ℚ := f b x - g x * ((4 * x) + (b + 7) / 3)

-- Theorem statement
theorem constant_remainder_iff_b_eq_neg_four_thirds :
  (∃ (c : ℚ), ∀ (x : ℚ), remainder b x = c) ↔ b = -4/3 :=
sorry

end constant_remainder_iff_b_eq_neg_four_thirds_l3171_317144


namespace existence_of_subset_l3171_317165

theorem existence_of_subset (n : ℕ+) (t : ℝ) (a : Fin (2*n.val-1) → ℝ) (ht : t ≠ 0) :
  ∃ (s : Finset (Fin (2*n.val-1))), s.card = n.val ∧
    ∀ (i j : Fin (2*n.val-1)), i ∈ s → j ∈ s → i ≠ j → a i - a j ≠ t :=
sorry

end existence_of_subset_l3171_317165


namespace sequence_periodicity_l3171_317198

theorem sequence_periodicity 
  (a b : ℕ → ℤ) 
  (h : ∀ n ≥ 3, (a n - a (n-1)) * (a n - a (n-2)) + (b n - b (n-1)) * (b n - b (n-2)) = 0) :
  ∃ k : ℕ+, a k = a (k + 2008) :=
sorry

end sequence_periodicity_l3171_317198


namespace short_students_fraction_l3171_317156

/-- Given a class with the following properties:
  * There are 400 total students
  * There are 90 tall students
  * There are 150 students with average height
  Prove that the fraction of short students to the total number of students is 2/5 -/
theorem short_students_fraction (total : ℕ) (tall : ℕ) (average : ℕ) 
  (h_total : total = 400)
  (h_tall : tall = 90)
  (h_average : average = 150) :
  (total - tall - average : ℚ) / total = 2 / 5 := by
  sorry

end short_students_fraction_l3171_317156


namespace office_clerks_count_l3171_317163

/-- Calculates the number of clerks in an office given specific salary information. -/
theorem office_clerks_count (total_avg : ℚ) (officer_avg : ℚ) (clerk_avg : ℚ) (officer_count : ℕ) :
  total_avg = 90 →
  officer_avg = 600 →
  clerk_avg = 84 →
  officer_count = 2 →
  ∃ (clerk_count : ℕ), 
    (officer_count * officer_avg + clerk_count * clerk_avg) / (officer_count + clerk_count) = total_avg ∧
    clerk_count = 170 :=
by sorry

end office_clerks_count_l3171_317163


namespace tangerine_cost_theorem_l3171_317141

/-- The cost of tangerines bought by Dong-jin -/
def tangerine_cost (original_money : ℚ) : ℚ :=
  original_money / 2

/-- The amount of money Dong-jin has after buying tangerines and giving some to his brother -/
def remaining_money (original_money : ℚ) : ℚ :=
  original_money / 2 * (1 - 3/8)

/-- Theorem stating the cost of tangerines given the conditions -/
theorem tangerine_cost_theorem (original_money : ℚ) :
  remaining_money original_money = 2500 →
  tangerine_cost original_money = 4000 :=
by
  sorry

end tangerine_cost_theorem_l3171_317141


namespace bear_cubs_count_l3171_317118

/-- Represents the bear's hunting scenario -/
structure BearHunt where
  totalMeat : ℕ  -- Total meat needed per week
  cubMeat : ℕ    -- Meat needed per cub per week
  rabbitWeight : ℕ -- Weight of each rabbit
  dailyCatch : ℕ  -- Number of rabbits caught daily

/-- Calculates the number of cubs based on the hunting scenario -/
def numCubs (hunt : BearHunt) : ℕ :=
  let weeklyHunt := hunt.dailyCatch * hunt.rabbitWeight * 7
  (weeklyHunt - hunt.totalMeat) / hunt.cubMeat

/-- Theorem stating that the number of cubs is 4 given the specific hunting scenario -/
theorem bear_cubs_count (hunt : BearHunt) 
  (h1 : hunt.totalMeat = 210)
  (h2 : hunt.cubMeat = 35)
  (h3 : hunt.rabbitWeight = 5)
  (h4 : hunt.dailyCatch = 10) :
  numCubs hunt = 4 := by
  sorry

end bear_cubs_count_l3171_317118


namespace xy_and_x3y_plus_x2_l3171_317139

theorem xy_and_x3y_plus_x2 (x y : ℝ) 
  (hx : x = 2 + Real.sqrt 3) 
  (hy : y = 2 - Real.sqrt 3) : 
  x * y = 1 ∧ x^3 * y + x^2 = 14 + 8 * Real.sqrt 3 := by
  sorry

end xy_and_x3y_plus_x2_l3171_317139


namespace right_triangle_area_l3171_317192

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 50) (h_sum_legs : a + b = 70) : 
  (1/2) * a * b = 300 := by
sorry

end right_triangle_area_l3171_317192


namespace complement_A_in_S_l3171_317175

def S : Set ℕ := {x | 0 ≤ x ∧ x ≤ 5}
def A : Set ℕ := {x | 1 < x ∧ x < 5}

theorem complement_A_in_S : 
  (S \ A) = {0, 1, 5} := by sorry

end complement_A_in_S_l3171_317175


namespace unique_solution_l3171_317127

theorem unique_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * Real.sqrt b - c = a) ∧ 
  (b * Real.sqrt c - a = b) ∧ 
  (c * Real.sqrt a - b = c) →
  a = 4 ∧ b = 4 ∧ c = 4 := by
sorry

end unique_solution_l3171_317127


namespace pizzeria_sales_l3171_317134

theorem pizzeria_sales (small_price large_price total_sales small_count : ℕ)
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : total_sales = 40)
  (h4 : small_count = 8) :
  ∃ large_count : ℕ,
    large_count * large_price + small_count * small_price = total_sales ∧
    large_count = 3 :=
by
  sorry

end pizzeria_sales_l3171_317134


namespace seventh_observation_value_l3171_317125

/-- Given 6 initial observations with an average of 15, prove that adding a 7th observation
    that decreases the overall average by 1 results in the 7th observation having a value of 8. -/
theorem seventh_observation_value (n : ℕ) (initial_average new_average : ℚ) :
  n = 6 →
  initial_average = 15 →
  new_average = initial_average - 1 →
  ∃ x : ℚ, x = 8 ∧ (n : ℚ) * initial_average + x = (n + 1 : ℚ) * new_average :=
by sorry

end seventh_observation_value_l3171_317125


namespace no_solution_exists_l3171_317184

theorem no_solution_exists : ¬∃ (a b : ℕ+), a^2 - 23 = b^11 := by
  sorry

end no_solution_exists_l3171_317184


namespace polar_equation_circle_and_ray_l3171_317126

/-- The polar equation (ρ - 1)(θ - π) = 0 with ρ ≥ 0 represents the union of a circle and a ray -/
theorem polar_equation_circle_and_ray (ρ θ : ℝ) :
  ρ ≥ 0 → (ρ - 1) * (θ - Real.pi) = 0 → 
  (∃ (x y : ℝ), x^2 + y^2 = 1) ∨ 
  (∃ (t : ℝ), t ≥ 0 → ∃ (x y : ℝ), x = -t ∧ y = 0) :=
by sorry

end polar_equation_circle_and_ray_l3171_317126


namespace quadratic_equation_solution_l3171_317154

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 2 + Real.sqrt 7 ∧ x₁^2 - 4*x₁ - 3 = 0) ∧
  (x₂ = 2 - Real.sqrt 7 ∧ x₂^2 - 4*x₂ - 3 = 0) := by
  sorry

end quadratic_equation_solution_l3171_317154


namespace right_triangle_hypotenuse_l3171_317170

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 8 →                  -- One leg measures 8 meters
  (1/2) * a * b = 48 →     -- Area is 48 square meters
  a^2 + b^2 = c^2 →        -- Pythagorean theorem for right triangle
  c = 4 * Real.sqrt 13 :=  -- Hypotenuse length is 4√13 meters
by
  sorry

end right_triangle_hypotenuse_l3171_317170


namespace ellipse_standard_equation_l3171_317153

/-- Represents an ellipse -/
structure Ellipse where
  center : ℝ × ℝ
  passes_through : ℝ × ℝ
  a_b_ratio : ℝ

/-- Checks if the given equation represents the standard form of the ellipse -/
def is_standard_equation (e : Ellipse) (eq : ℝ → ℝ → Bool) : Prop :=
  (eq 3 0 = true) ∧ 
  (∀ x y, eq x y ↔ (x^2 / 9 + y^2 = 1 ∨ y^2 / 81 + x^2 / 9 = 1))

/-- Theorem: Given the conditions, the ellipse has one of the two standard equations -/
theorem ellipse_standard_equation (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.passes_through = (3, 0))
  (h3 : e.a_b_ratio = 3) :
  ∃ eq : ℝ → ℝ → Bool, is_standard_equation e eq := by
  sorry

end ellipse_standard_equation_l3171_317153


namespace luke_trivia_score_l3171_317157

/-- Luke's trivia game score calculation -/
theorem luke_trivia_score (points_per_round : ℕ) (num_rounds : ℕ) :
  points_per_round = 146 →
  num_rounds = 157 →
  points_per_round * num_rounds = 22822 := by
  sorry

end luke_trivia_score_l3171_317157


namespace first_year_after_2020_with_digit_sum_18_l3171_317149

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isValidYear (year : ℕ) : Prop :=
  year > 2020 ∧ sumOfDigits year = 18

theorem first_year_after_2020_with_digit_sum_18 :
  ∀ year : ℕ, isValidYear year → year ≥ 2799 :=
sorry

end first_year_after_2020_with_digit_sum_18_l3171_317149


namespace banana_cost_18lbs_l3171_317104

/-- The cost of bananas given a rate, weight, and discount condition -/
def banana_cost (rate : ℚ) (rate_weight : ℚ) (weight : ℚ) (discount_threshold : ℚ) (discount_rate : ℚ) : ℚ :=
  let base_cost := (weight / rate_weight) * rate
  if weight ≥ discount_threshold then
    base_cost * (1 - discount_rate)
  else
    base_cost

/-- Theorem stating the cost of 18 pounds of bananas given the specified conditions -/
theorem banana_cost_18lbs : 
  banana_cost 3 3 18 15 (1/10) = 162/10 := by
  sorry

end banana_cost_18lbs_l3171_317104


namespace remaining_bulbs_correct_l3171_317177

def calculate_remaining_bulbs (initial_led : ℕ) (initial_incandescent : ℕ)
  (used_led : ℕ) (used_incandescent : ℕ)
  (alex_percent : ℚ) (bob_percent : ℚ) (charlie_led_percent : ℚ) (charlie_incandescent_percent : ℚ)
  : (ℕ × ℕ) :=
  sorry

theorem remaining_bulbs_correct :
  let initial_led := 24
  let initial_incandescent := 16
  let used_led := 10
  let used_incandescent := 6
  let alex_percent := 1/2
  let bob_percent := 1/4
  let charlie_led_percent := 1/5
  let charlie_incandescent_percent := 3/10
  calculate_remaining_bulbs initial_led initial_incandescent used_led used_incandescent
    alex_percent bob_percent charlie_led_percent charlie_incandescent_percent = (6, 6) :=
by
  sorry

end remaining_bulbs_correct_l3171_317177


namespace at_least_one_greater_than_one_l3171_317173

theorem at_least_one_greater_than_one (a b : ℝ) : a + b > 2 → max a b > 1 := by
  sorry

end at_least_one_greater_than_one_l3171_317173


namespace correct_division_l3171_317135

theorem correct_division (x : ℤ) : x + 4 = 40 → x / 4 = 9 := by
  sorry

end correct_division_l3171_317135


namespace ceiling_floor_difference_l3171_317113

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 := by
  sorry

end ceiling_floor_difference_l3171_317113


namespace dot_product_AO_AB_l3171_317107

/-- The circle O with equation x^2 + y^2 = 4 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- The theorem statement -/
theorem dot_product_AO_AB (O A B : ℝ × ℝ) :
  A ∈ Circle → B ∈ Circle →
  ‖(A - O) + (B - O)‖ = ‖(A - O) - (B - O)‖ →
  (A - O) • (A - B) = 4 := by
sorry

end dot_product_AO_AB_l3171_317107


namespace no_integer_solution_l3171_317186

theorem no_integer_solution : ¬ ∃ (a b c : ℤ), a^2 + b^2 - 8*c = 6 := by sorry

end no_integer_solution_l3171_317186


namespace range_of_m_l3171_317129

-- Define the sets M and N
def M (m : ℝ) : Set ℝ := {x : ℝ | x + m ≥ 0}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 8 < 0}

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Theorem statement
theorem range_of_m (m : ℝ) : (Set.compl (M m) ∩ N = ∅) → m ≥ 2 := by
  sorry

end range_of_m_l3171_317129


namespace inequality_proof_l3171_317195

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b + b^2 / c + c^2 / a) ≥ 3 * (a^3 + b^3 + c^3) / (a^2 + b^2 + c^2) := by
  sorry


end inequality_proof_l3171_317195


namespace complex_equation_solutions_l3171_317188

theorem complex_equation_solutions : 
  ∃ (S : Finset ℂ), (∀ z ∈ S, Complex.abs z < 24 ∧ Complex.exp z = (z - 2) / (z + 2)) ∧ 
                    Finset.card S = 8 ∧
                    ∀ z, Complex.abs z < 24 → Complex.exp z = (z - 2) / (z + 2) → z ∈ S := by
  sorry

end complex_equation_solutions_l3171_317188


namespace min_distance_squared_l3171_317151

def is_geometric_progression (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem min_distance_squared (x y z : ℝ) :
  is_geometric_progression x y z →
  is_arithmetic_progression (x * y) (y * z) (x * z) →
  z ≥ 1 →
  x ≠ y →
  y ≠ z →
  x ≠ z →
  (∀ x' y' z' : ℝ, 
    is_geometric_progression x' y' z' →
    is_arithmetic_progression (x' * y') (y' * z') (x' * z') →
    z' ≥ 1 →
    x' ≠ y' →
    y' ≠ z' →
    x' ≠ z' →
    (x - 1)^2 + (y - 1)^2 + (z - 1)^2 ≤ (x' - 1)^2 + (y' - 1)^2 + (z' - 1)^2) →
  (x - 1)^2 + (y - 1)^2 + (z - 1)^2 = 18 :=
by sorry

end min_distance_squared_l3171_317151


namespace parabola_focus_hyperbola_equation_l3171_317102

-- Part 1: Parabola
theorem parabola_focus (p : ℝ) (h1 : p > 0) :
  (∃ x y : ℝ, y^2 = 2*p*x ∧ 2*x - y - 4 = 0 ∧ x = p/2 ∧ y = 0) →
  p = 4 := by sorry

-- Part 2: Hyperbola
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (b/a = 3/4 ∧ a^2/(a^2 + b^2)^(1/2) = 16/5) →
  (∀ x y : ℝ, x^2/16 - y^2/9 = 1 ↔ x^2/a^2 - y^2/b^2 = 1) := by sorry

end parabola_focus_hyperbola_equation_l3171_317102


namespace sin_negative_45_degrees_l3171_317174

theorem sin_negative_45_degrees : Real.sin (-(π / 4)) = -(Real.sqrt 2 / 2) := by
  sorry

end sin_negative_45_degrees_l3171_317174


namespace geometric_sequence_common_ratio_l3171_317117

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) 
  (h2 : a 1 - a 2 = 3) 
  (h3 : a 1 - a 3 = 2) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = -1/3 := by
sorry

end geometric_sequence_common_ratio_l3171_317117


namespace arithmetic_sequence_ratio_l3171_317172

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : arithmetic_sequence a d)
  (h3 : a 3 ^ 2 = a 1 * a 9) :
  a 3 / a 6 = 1 / 2 := by
sorry

end arithmetic_sequence_ratio_l3171_317172


namespace switch_circuit_probability_l3171_317150

theorem switch_circuit_probability (P_A P_AB : ℝ) 
  (h1 : P_A = 1/2) 
  (h2 : P_AB = 1/5) : 
  P_AB / P_A = 2/5 := by
  sorry

end switch_circuit_probability_l3171_317150


namespace polygon_sides_l3171_317182

theorem polygon_sides (S : ℝ) (h : S = 1080) :
  ∃ n : ℕ, n > 2 ∧ (n - 2) * 180 = S ∧ n = 8 := by
  sorry

end polygon_sides_l3171_317182


namespace hash_difference_six_four_l3171_317155

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x

-- State the theorem
theorem hash_difference_six_four : hash 6 4 - hash 4 6 = -6 := by
  sorry

end hash_difference_six_four_l3171_317155


namespace rectangle_breadth_unchanged_l3171_317122

theorem rectangle_breadth_unchanged 
  (L B : ℝ) 
  (h1 : L > 0) 
  (h2 : B > 0) 
  (new_L : ℝ) 
  (h3 : new_L = L / 2) 
  (new_A : ℝ) 
  (h4 : new_A = L * B / 2) :
  ∃ (new_B : ℝ), new_A = new_L * new_B ∧ new_B = B := by
sorry

end rectangle_breadth_unchanged_l3171_317122


namespace arithmetic_series_sum_40_60_1_7_l3171_317105

/-- Sum of an arithmetic series -/
def arithmetic_series_sum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_40_60_1_7 :
  arithmetic_series_sum 40 60 (1/7) = 7050 := by
  sorry

end arithmetic_series_sum_40_60_1_7_l3171_317105


namespace expression_evaluation_l3171_317187

theorem expression_evaluation : 2197 + 180 / 60 * 3 - 197 = 2009 := by
  sorry

end expression_evaluation_l3171_317187


namespace solve_slurpee_problem_l3171_317110

def slurpee_problem (initial_amount : ℕ) (slurpee_cost : ℕ) (change : ℕ) : Prop :=
  let amount_spent : ℕ := initial_amount - change
  let num_slurpees : ℕ := amount_spent / slurpee_cost
  num_slurpees = 6

theorem solve_slurpee_problem :
  slurpee_problem 20 2 8 := by
  sorry

end solve_slurpee_problem_l3171_317110


namespace max_min_values_of_f_l3171_317179

def f (x : ℝ) : ℝ := -x^2 + 4*x + 5

theorem max_min_values_of_f :
  let a : ℝ := 1
  let b : ℝ := 4
  (∀ x ∈ Set.Icc a b, f x ≤ 9) ∧
  (∃ x ∈ Set.Icc a b, f x = 9) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ 5) ∧
  (∃ x ∈ Set.Icc a b, f x = 5) :=
by sorry

end max_min_values_of_f_l3171_317179


namespace seedling_difference_seedling_difference_proof_l3171_317112

theorem seedling_difference : ℕ → ℕ → ℕ → Prop :=
  fun pine_seedlings poplar_multiplier difference =>
    pine_seedlings = 180 →
    poplar_multiplier = 4 →
    difference = poplar_multiplier * pine_seedlings - pine_seedlings →
    difference = 540

-- Proof
theorem seedling_difference_proof : seedling_difference 180 4 540 := by
  sorry

end seedling_difference_seedling_difference_proof_l3171_317112


namespace right_triangle_hypotenuse_l3171_317180

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 24 →
  b = 18 →
  c^2 = a^2 + b^2 →
  c = 30 :=
by
  sorry

end right_triangle_hypotenuse_l3171_317180


namespace total_cars_l3171_317158

/-- The number of cars owned by five people given specific relationships between their car counts -/
theorem total_cars (tommy : ℕ) (jessie : ℕ) : 
  tommy = 7 →
  jessie = 9 →
  (tommy + jessie + (jessie + 2) + (tommy - 3) + 2 * (jessie + 2)) = 53 := by
  sorry

end total_cars_l3171_317158


namespace quadratic_inequality_properties_l3171_317171

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 - b * x + c

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h_solution_set : ∀ x : ℝ, f a b c x > 0 ↔ -1 < x ∧ x < 2) :
  (a + b + c = 0) ∧ (a < 0) := by
  sorry


end quadratic_inequality_properties_l3171_317171
