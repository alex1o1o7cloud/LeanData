import Mathlib

namespace NUMINAMATH_CALUDE_box_length_l1976_197679

/-- The length of a rectangular box given its filling rate, width, depth, and filling time. -/
theorem box_length (fill_rate : ℝ) (width depth : ℝ) (fill_time : ℝ) :
  fill_rate = 3 →
  width = 4 →
  depth = 3 →
  fill_time = 20 →
  fill_rate * fill_time / (width * depth) = 5 := by
sorry

end NUMINAMATH_CALUDE_box_length_l1976_197679


namespace NUMINAMATH_CALUDE_johns_donation_l1976_197654

theorem johns_donation (
  initial_contributions : ℕ) 
  (new_average : ℚ)
  (increase_percentage : ℚ) :
  initial_contributions = 3 →
  new_average = 75 →
  increase_percentage = 50 / 100 →
  ∃ (johns_donation : ℚ),
    johns_donation = 150 ∧
    new_average = (initial_contributions * (new_average / (1 + increase_percentage)) + johns_donation) / (initial_contributions + 1) :=
by sorry

end NUMINAMATH_CALUDE_johns_donation_l1976_197654


namespace NUMINAMATH_CALUDE_chord_line_equation_l1976_197607

/-- The equation of a line containing a chord of an ellipse, given the ellipse equation and the midpoint of the chord. -/
theorem chord_line_equation (a b c : ℝ) (x₀ y₀ : ℝ) :
  (∀ x y, x^2 + a*y^2 = b) →  -- Ellipse equation
  (∃ x₁ y₁ x₂ y₂ : ℝ,  -- Existence of chord endpoints
    x₁^2 + a*y₁^2 = b ∧
    x₂^2 + a*y₂^2 = b ∧
    x₀ = (x₁ + x₂) / 2 ∧
    y₀ = (y₁ + y₂) / 2) →
  (∃ k m : ℝ, ∀ x y, (x - x₀) + k*(y - y₀) = 0 ↔ x + k*y = m) →
  (a = 4 ∧ b = 36 ∧ x₀ = 4 ∧ y₀ = 2 ∧ c = 8) →
  (∀ x y, x + 2*y - c = 0 ↔ (x - x₀) + 2*(y - y₀) = 0) :=
by sorry

#check chord_line_equation

end NUMINAMATH_CALUDE_chord_line_equation_l1976_197607


namespace NUMINAMATH_CALUDE_inverse_function_point_l1976_197697

-- Define a function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the property of f_inv being the inverse of f
def is_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- State the theorem
theorem inverse_function_point 
  (h1 : is_inverse f f_inv) 
  (h2 : f 3 = -1) : 
  f_inv (-3) = 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_point_l1976_197697


namespace NUMINAMATH_CALUDE_drive_time_proof_l1976_197641

/-- Proves the time driven at 60 mph given the conditions of the problem -/
theorem drive_time_proof (total_distance : ℝ) (initial_speed : ℝ) (final_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 120)
  (h2 : initial_speed = 60)
  (h3 : final_speed = 90)
  (h4 : total_time = 1.5) :
  ∃ t : ℝ, t + (total_distance - initial_speed * t) / final_speed = total_time ∧ t = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_drive_time_proof_l1976_197641


namespace NUMINAMATH_CALUDE_two_a_minus_two_d_is_zero_l1976_197656

/-- Given a function g and constants a, b, c, d, prove that 2a - 2d = 0 -/
theorem two_a_minus_two_d_is_zero
  (a b c d : ℝ)
  (h_abcd : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (g : ℝ → ℝ)
  (h_g : ∀ x, g x = (2*a*x - b) / (c*x - 2*d))
  (h_inv : ∀ x, g (g x) = x) :
  2*a - 2*d = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_a_minus_two_d_is_zero_l1976_197656


namespace NUMINAMATH_CALUDE_complex_number_location_l1976_197620

theorem complex_number_location (z : ℂ) (h : z = Complex.I * (1 + Complex.I)) :
  Complex.re z < 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1976_197620


namespace NUMINAMATH_CALUDE_remainder_after_adding_150_l1976_197619

theorem remainder_after_adding_150 (n : ℤ) :
  n % 6 = 1 → (n + 150) % 6 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_after_adding_150_l1976_197619


namespace NUMINAMATH_CALUDE_cube_difference_formula_l1976_197610

theorem cube_difference_formula (n : ℕ) : 
  (n + 1)^3 - n^3 = 3*n^2 + 3*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_formula_l1976_197610


namespace NUMINAMATH_CALUDE_kiley_crayons_l1976_197665

theorem kiley_crayons (initial_crayons : ℕ) (remaining_crayons : ℕ) 
  (h1 : initial_crayons = 48)
  (h2 : remaining_crayons = 18)
  (f : ℚ) -- fraction of crayons Kiley took
  (h3 : 0 ≤ f ∧ f < 1)
  (h4 : remaining_crayons = (initial_crayons : ℚ) * (1 - f) / 2) :
  f = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_kiley_crayons_l1976_197665


namespace NUMINAMATH_CALUDE_payback_time_l1976_197621

def initial_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def monthly_expenses : ℝ := 1500

theorem payback_time :
  let monthly_profit := monthly_revenue - monthly_expenses
  (initial_cost / monthly_profit : ℝ) = 10 := by sorry

end NUMINAMATH_CALUDE_payback_time_l1976_197621


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1976_197618

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 12) = 13 - x →
  x = (31 + Real.sqrt 333) / 2 ∨ x = (31 - Real.sqrt 333) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1976_197618


namespace NUMINAMATH_CALUDE_inequality_solution_l1976_197643

/-- The solution set of the inequality |ax-2|+|ax-a| ≥ 2 when a = 1 -/
def solution_set_a1 : Set ℝ := {x | x ≥ 2.5 ∨ x ≤ 0.5}

/-- The inequality |ax-2|+|ax-a| ≥ 2 -/
def inequality (a x : ℝ) : Prop := |a*x - 2| + |a*x - a| ≥ 2

theorem inequality_solution :
  (∀ x, inequality 1 x ↔ x ∈ solution_set_a1) ∧
  (∀ a, a > 0 → (∀ x, inequality a x) ↔ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1976_197643


namespace NUMINAMATH_CALUDE_min_box_value_l1976_197693

theorem min_box_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a*x + b) * (b*x + 2*a) = 36*x^2 + box*x + 72) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  (∃ a' b' box', 
    (∀ x, (a'*x + b') * (b'*x + 2*a') = 36*x^2 + box'*x + 72) ∧
    a' ≠ b' ∧ b' ≠ box' ∧ a' ≠ box' ∧
    box' < box) →
  box ≥ 332 :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l1976_197693


namespace NUMINAMATH_CALUDE_autumn_sales_l1976_197672

/-- Ice cream sales data for a city --/
structure IceCreamSales where
  spring : ℝ
  summer : ℝ
  autumn : ℝ
  winter : ℝ

/-- Theorem: Autumn ice cream sales calculation --/
theorem autumn_sales (sales : IceCreamSales) 
  (h1 : sales.spring = 3)
  (h2 : sales.summer = 6)
  (h3 : sales.winter = 5)
  (h4 : sales.spring = 0.2 * (sales.spring + sales.summer + sales.autumn + sales.winter)) :
  sales.autumn = 1 := by
  sorry

#check autumn_sales

end NUMINAMATH_CALUDE_autumn_sales_l1976_197672


namespace NUMINAMATH_CALUDE_function_composition_equality_l1976_197662

theorem function_composition_equality (c : ℝ) : 
  let p : ℝ → ℝ := λ x => 4 * x - 9
  let q : ℝ → ℝ := λ x => 5 * x - c
  p (q 3) = 14 → c = 9.25 := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1976_197662


namespace NUMINAMATH_CALUDE_remaining_money_l1976_197678

def initial_amount : ℕ := 100
def roast_cost : ℕ := 17
def vegetable_cost : ℕ := 11

theorem remaining_money :
  initial_amount - (roast_cost + vegetable_cost) = 72 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l1976_197678


namespace NUMINAMATH_CALUDE_glasses_in_smaller_box_l1976_197626

theorem glasses_in_smaller_box :
  ∀ x : ℕ,
  (x + 16) / 2 = 15 →
  x = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_glasses_in_smaller_box_l1976_197626


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1976_197625

theorem inequality_solution_set (x : ℝ) : 
  (x^2 * (x + 1)) / (-x^2 - 5*x + 6) ≤ 0 ∧ (-x^2 - 5*x + 6) ≠ 0 ↔ 
  (-6 < x ∧ x ≤ -1) ∨ x = 0 ∨ x > 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1976_197625


namespace NUMINAMATH_CALUDE_sqrt_6000_approx_l1976_197658

/-- Approximate value of the square root of 6 -/
def sqrt_6_approx : ℝ := 2.45

/-- Approximate value of the square root of 60 -/
def sqrt_60_approx : ℝ := 7.75

/-- Theorem stating that the square root of 6000 is approximately 77.5 -/
theorem sqrt_6000_approx : ∃ (ε : ℝ), ε > 0 ∧ |Real.sqrt 6000 - 77.5| < ε :=
sorry

end NUMINAMATH_CALUDE_sqrt_6000_approx_l1976_197658


namespace NUMINAMATH_CALUDE_square_difference_l1976_197664

theorem square_difference (a b : ℝ) 
  (h1 : a^2 - a*b = 10) 
  (h2 : a*b - b^2 = -15) : 
  a^2 - b^2 = -5 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1976_197664


namespace NUMINAMATH_CALUDE_total_cost_for_20_products_l1976_197667

/-- The total cost function for producing products -/
def total_cost (fixed_cost marginal_cost : ℝ) (n : ℕ) : ℝ :=
  fixed_cost + marginal_cost * n

/-- Theorem: The total cost for producing 20 products is $16000 -/
theorem total_cost_for_20_products
  (fixed_cost : ℝ)
  (marginal_cost : ℝ)
  (h1 : fixed_cost = 12000)
  (h2 : marginal_cost = 200) :
  total_cost fixed_cost marginal_cost 20 = 16000 := by
  sorry

#eval total_cost 12000 200 20

end NUMINAMATH_CALUDE_total_cost_for_20_products_l1976_197667


namespace NUMINAMATH_CALUDE_cart_distance_theorem_l1976_197628

/-- Represents a cart with two wheels -/
structure Cart where
  front_wheel_circumference : ℝ
  back_wheel_circumference : ℝ

/-- Calculates the distance traveled by the cart -/
def distance_traveled (c : Cart) (back_revolutions : ℝ) : ℝ :=
  back_revolutions * c.back_wheel_circumference

/-- Theorem stating the distance traveled by the cart -/
theorem cart_distance_theorem (c : Cart) 
    (h1 : c.front_wheel_circumference = 30)
    (h2 : c.back_wheel_circumference = 32)
    (h3 : ∃ (r : ℝ), r * c.back_wheel_circumference = (r + 5) * c.front_wheel_circumference) :
  ∃ (r : ℝ), distance_traveled c r = 2400 := by
  sorry

#check cart_distance_theorem

end NUMINAMATH_CALUDE_cart_distance_theorem_l1976_197628


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l1976_197659

theorem triangle_angle_inequality (y : ℝ) (p q : ℝ) : 
  y > 0 →
  y + 10 > 0 →
  y + 5 > 0 →
  4*y > 0 →
  y + 10 + y + 5 > 4*y →
  y + 10 + 4*y > y + 5 →
  y + 5 + 4*y > y + 10 →
  4*y > y + 10 →
  4*y > y + 5 →
  p < y →
  y < q →
  q - p ≥ 5/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l1976_197659


namespace NUMINAMATH_CALUDE_specific_shape_perimeter_l1976_197633

/-- A shape consisting of a regular hexagon, six triangles, and six squares -/
structure Shape where
  hexagon_side : ℝ
  num_triangles : ℕ
  num_squares : ℕ

/-- The outer perimeter of the shape -/
def outer_perimeter (s : Shape) : ℝ :=
  12 * s.hexagon_side

/-- Theorem stating that the outer perimeter of the specific shape is 216 cm -/
theorem specific_shape_perimeter :
  ∃ (s : Shape), s.hexagon_side = 18 ∧ s.num_triangles = 6 ∧ s.num_squares = 6 ∧ outer_perimeter s = 216 :=
by sorry

end NUMINAMATH_CALUDE_specific_shape_perimeter_l1976_197633


namespace NUMINAMATH_CALUDE_f_max_value_l1976_197689

/-- The quadratic function f(x) = 10x - 2x^2 -/
def f (x : ℝ) : ℝ := 10 * x - 2 * x^2

/-- The maximum value of f(x) is 12.5 -/
theorem f_max_value : ∃ (M : ℝ), M = 12.5 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l1976_197689


namespace NUMINAMATH_CALUDE_jeff_average_skips_l1976_197639

def jeff_skips (sam_skips : ℕ) (rounds : ℕ) : List ℕ :=
  let round1 := sam_skips - 1
  let round2 := sam_skips - 3
  let round3 := sam_skips + 4
  let round4 := sam_skips / 2
  let round5 := round4 + (sam_skips - round4 + 2)
  [round1, round2, round3, round4, round5]

def average_skips (skips : List ℕ) : ℚ :=
  (skips.sum : ℚ) / skips.length

theorem jeff_average_skips (sam_skips : ℕ) (rounds : ℕ) :
  sam_skips = 16 ∧ rounds = 5 →
  average_skips (jeff_skips sam_skips rounds) = 74/5 :=
by sorry

end NUMINAMATH_CALUDE_jeff_average_skips_l1976_197639


namespace NUMINAMATH_CALUDE_lines_parabolas_intersection_empty_l1976_197609

-- Define the set of all lines
def Lines := {f : ℝ → ℝ | ∃ (m b : ℝ), ∀ x, f x = m * x + b}

-- Define the set of all parabolas
def Parabolas := {f : ℝ → ℝ | ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c}

-- Theorem statement
theorem lines_parabolas_intersection_empty : Lines ∩ Parabolas = ∅ := by
  sorry

end NUMINAMATH_CALUDE_lines_parabolas_intersection_empty_l1976_197609


namespace NUMINAMATH_CALUDE_initial_water_percentage_l1976_197695

/-- 
Given a mixture of 180 liters, if adding 12 liters of water results in a new mixture 
where water is 25% of the total, then the initial percentage of water in the mixture was 20%.
-/
theorem initial_water_percentage (initial_volume : ℝ) (added_water : ℝ) (final_water_percentage : ℝ) :
  initial_volume = 180 →
  added_water = 12 →
  final_water_percentage = 25 →
  (initial_volume * (20 / 100) + added_water) / (initial_volume + added_water) = final_water_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l1976_197695


namespace NUMINAMATH_CALUDE_cloth_cost_price_l1976_197681

/-- Given a trader selling cloth, calculates the cost price per meter. -/
def cost_price_per_meter (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_meters) / total_meters

/-- Proves that the cost price of one metre of cloth is 128 rupees. -/
theorem cloth_cost_price :
  cost_price_per_meter 60 8400 12 = 128 := by
  sorry

#eval cost_price_per_meter 60 8400 12

end NUMINAMATH_CALUDE_cloth_cost_price_l1976_197681


namespace NUMINAMATH_CALUDE_cubic_function_extremum_l1976_197688

theorem cubic_function_extremum (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - a*x^2 - b*x + a^2
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 - 2*a*x - b
  (f 1 = 10 ∧ f' 1 = 0) → ((a = -4 ∧ b = 11) ∨ (a = 3 ∧ b = -3)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extremum_l1976_197688


namespace NUMINAMATH_CALUDE_ellen_legos_l1976_197605

/-- The number of legos Ellen lost -/
def lost_legos : ℕ := 57

/-- The number of legos Ellen currently has -/
def current_legos : ℕ := 323

/-- The initial number of legos Ellen had -/
def initial_legos : ℕ := lost_legos + current_legos

theorem ellen_legos : initial_legos = 380 := by
  sorry

end NUMINAMATH_CALUDE_ellen_legos_l1976_197605


namespace NUMINAMATH_CALUDE_representable_integers_l1976_197698

theorem representable_integers (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2004) : 
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ k = (m * n + 1) / (m + n) := by
  sorry

end NUMINAMATH_CALUDE_representable_integers_l1976_197698


namespace NUMINAMATH_CALUDE_find_number_l1976_197652

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 69 ∧ x = 7 := by sorry

end NUMINAMATH_CALUDE_find_number_l1976_197652


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1976_197653

/-- Represents the probability of hitting a specific ring -/
structure RingProbability where
  ring : Nat
  probability : Real

/-- Calculates the probability of hitting either the 10-ring or 9-ring -/
def prob_10_or_9 (probs : List RingProbability) : Real :=
  (probs.filter (fun p => p.ring == 10 || p.ring == 9)).map (fun p => p.probability) |>.sum

/-- Calculates the probability of hitting below the 7-ring -/
def prob_below_7 (probs : List RingProbability) : Real :=
  1 - (probs.map (fun p => p.probability) |>.sum)

/-- Theorem stating the probabilities for the given shooting scenario -/
theorem shooting_probabilities (probs : List RingProbability) 
  (h10 : RingProbability.mk 10 0.21 ∈ probs)
  (h9 : RingProbability.mk 9 0.23 ∈ probs)
  (h8 : RingProbability.mk 8 0.25 ∈ probs)
  (h7 : RingProbability.mk 7 0.28 ∈ probs)
  (h_no_other : ∀ p ∈ probs, p.ring ∈ [7, 8, 9, 10]) :
  prob_10_or_9 probs = 0.44 ∧ prob_below_7 probs = 0.03 := by
  sorry


end NUMINAMATH_CALUDE_shooting_probabilities_l1976_197653


namespace NUMINAMATH_CALUDE_directional_vector_for_line_l1976_197676

/-- A directional vector for a line ax + by + c = 0 is a vector (u, v) such that
    for any point (x, y) on the line, (x + u, y + v) is also on the line. -/
def IsDirectionalVector (a b c : ℝ) (u v : ℝ) : Prop :=
  ∀ x y : ℝ, a * x + b * y + c = 0 → a * (x + u) + b * (y + v) + c = 0

/-- The line 2x + 3y - 1 = 0 -/
def Line (x y : ℝ) : Prop := 2 * x + 3 * y - 1 = 0

/-- Theorem: (1, -2/3) is a directional vector for the line 2x + 3y - 1 = 0 -/
theorem directional_vector_for_line :
  IsDirectionalVector 2 3 (-1) 1 (-2/3) :=
sorry

end NUMINAMATH_CALUDE_directional_vector_for_line_l1976_197676


namespace NUMINAMATH_CALUDE_expression_equality_l1976_197634

theorem expression_equality : 
  abs (-3) - Real.sqrt 8 - (1/2)⁻¹ + 2 * Real.cos (π/4) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1976_197634


namespace NUMINAMATH_CALUDE_one_slice_remains_l1976_197694

/-- Calculates the number of slices of bread remaining after eating and making toast. -/
def remaining_slices (initial_slices : ℕ) (eaten_twice : ℕ) (slices_per_toast : ℕ) (toast_made : ℕ) : ℕ :=
  initial_slices - (2 * eaten_twice) - (slices_per_toast * toast_made)

/-- Theorem stating that given the specific conditions, 1 slice of bread remains. -/
theorem one_slice_remains : remaining_slices 27 3 2 10 = 1 := by
  sorry

#eval remaining_slices 27 3 2 10

end NUMINAMATH_CALUDE_one_slice_remains_l1976_197694


namespace NUMINAMATH_CALUDE_max_min_equation_characterization_l1976_197692

theorem max_min_equation_characterization (x y : ℝ) : 
  max x (x^2) + min y (y^2) = 1 ↔ 
    (y = 1 - x^2 ∧ y ≤ 0) ∨
    (x^2 + y^2 = 1 ∧ ((x ≤ -1 ∨ x > 0) ∧ 0 < y ∧ y < 1)) ∨
    (y^2 = 1 - x ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1) :=
by sorry

end NUMINAMATH_CALUDE_max_min_equation_characterization_l1976_197692


namespace NUMINAMATH_CALUDE_lino_shell_collection_l1976_197638

/-- The number of shells Lino picked up in the morning -/
def morning_shells : ℕ := 292

/-- The number of shells Lino picked up in the afternoon -/
def afternoon_shells : ℕ := 324

/-- The total number of shells Lino picked up -/
def total_shells : ℕ := morning_shells + afternoon_shells

/-- Theorem stating that the total number of shells Lino picked up is 616 -/
theorem lino_shell_collection : total_shells = 616 := by
  sorry

end NUMINAMATH_CALUDE_lino_shell_collection_l1976_197638


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1976_197649

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1976_197649


namespace NUMINAMATH_CALUDE_max_x_value_l1976_197673

theorem max_x_value (x : ℝ) : 
  ((4*x - 16)/(3*x - 4))^2 + (4*x - 16)/(3*x - 4) = 12 → x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l1976_197673


namespace NUMINAMATH_CALUDE_f_continuous_at_2_delta_epsilon_relation_l1976_197644

def f (x : ℝ) : ℝ := -3 * x^2 - 5

theorem f_continuous_at_2 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by sorry

theorem delta_epsilon_relation :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 3 ∧
    ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_f_continuous_at_2_delta_epsilon_relation_l1976_197644


namespace NUMINAMATH_CALUDE_salt_concentration_proof_l1976_197670

/-- Proves that adding 66.67 gallons of 25% salt solution to 100 gallons of pure water results in a 10% salt solution -/
theorem salt_concentration_proof (initial_water : ℝ) (saline_volume : ℝ) (salt_percentage : ℝ) :
  initial_water = 100 →
  saline_volume = 66.67 →
  salt_percentage = 0.25 →
  (salt_percentage * saline_volume) / (initial_water + saline_volume) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_salt_concentration_proof_l1976_197670


namespace NUMINAMATH_CALUDE_problem_statement_l1976_197657

theorem problem_statement (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) : 
  (1/3) * x^8 * y^9 = 2/5 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1976_197657


namespace NUMINAMATH_CALUDE_prob_at_least_one_red_l1976_197642

/-- The probability of selecting at least one red ball when randomly choosing 2 balls out of 5 balls (2 red and 3 white) is 7/10. -/
theorem prob_at_least_one_red (total : ℕ) (red : ℕ) (white : ℕ) (select : ℕ) :
  total = 5 →
  red = 2 →
  white = 3 →
  select = 2 →
  (Nat.choose total select - Nat.choose white select : ℚ) / Nat.choose total select = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_red_l1976_197642


namespace NUMINAMATH_CALUDE_student_comprehensive_score_l1976_197604

/-- Calculates the comprehensive score of a student in a competition --/
def comprehensiveScore (theoreticalWeight : ℝ) (innovativeWeight : ℝ) (presentationWeight : ℝ)
                       (theoreticalScore : ℝ) (innovativeScore : ℝ) (presentationScore : ℝ) : ℝ :=
  theoreticalWeight * theoreticalScore + innovativeWeight * innovativeScore + presentationWeight * presentationScore

/-- Theorem stating that the student's comprehensive score is 89.5 --/
theorem student_comprehensive_score :
  let theoreticalWeight : ℝ := 0.20
  let innovativeWeight : ℝ := 0.50
  let presentationWeight : ℝ := 0.30
  let theoreticalScore : ℝ := 80
  let innovativeScore : ℝ := 90
  let presentationScore : ℝ := 95
  comprehensiveScore theoreticalWeight innovativeWeight presentationWeight
                     theoreticalScore innovativeScore presentationScore = 89.5 := by
  sorry

end NUMINAMATH_CALUDE_student_comprehensive_score_l1976_197604


namespace NUMINAMATH_CALUDE_trigonometric_product_equality_l1976_197635

theorem trigonometric_product_equality : 
  (1 - 1 / Real.cos (30 * π / 180)) * 
  (1 + 2 / Real.sin (60 * π / 180)) * 
  (1 - 1 / Real.sin (30 * π / 180)) * 
  (1 + 2 / Real.cos (60 * π / 180)) = 
  (25 - 10 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_product_equality_l1976_197635


namespace NUMINAMATH_CALUDE_cloud9_diving_company_revenue_l1976_197671

/-- Cloud 9 Diving Company's financial calculation -/
theorem cloud9_diving_company_revenue 
  (individual_bookings : ℕ) 
  (group_bookings : ℕ) 
  (cancellations : ℕ) 
  (h1 : individual_bookings = 12000)
  (h2 : group_bookings = 16000)
  (h3 : cancellations = 1600) :
  individual_bookings + group_bookings - cancellations = 26400 :=
by sorry

end NUMINAMATH_CALUDE_cloud9_diving_company_revenue_l1976_197671


namespace NUMINAMATH_CALUDE_system_solution_l1976_197630

theorem system_solution (x y a : ℝ) : 
  (3 * x + y = 1 + 3 * a) → 
  (x + 3 * y = 1 - a) → 
  (x + y = 0) → 
  (a = -1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1976_197630


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l1976_197650

/-- The line equation in R³ --/
def line (x y z : ℝ) : Prop :=
  (x + 2) / 1 = (y - 2) / 0 ∧ (x + 2) / 1 = (z + 3) / 0

/-- The plane equation in R³ --/
def plane (x y z : ℝ) : Prop :=
  2 * x - 3 * y - 5 * z - 7 = 0

/-- The intersection point of the line and the plane --/
def intersection_point : ℝ × ℝ × ℝ := (-1, 2, -3)

theorem intersection_point_is_unique :
  ∃! p : ℝ × ℝ × ℝ, 
    line p.1 p.2.1 p.2.2 ∧ 
    plane p.1 p.2.1 p.2.2 ∧ 
    p = intersection_point :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l1976_197650


namespace NUMINAMATH_CALUDE_empty_can_weight_l1976_197622

theorem empty_can_weight (full_can : ℝ) (half_can : ℝ) (h1 : full_can = 35) (h2 : half_can = 18) :
  full_can - 2 * (full_can - half_can) = 1 :=
by sorry

end NUMINAMATH_CALUDE_empty_can_weight_l1976_197622


namespace NUMINAMATH_CALUDE_max_distance_from_circle_to_point_l1976_197606

theorem max_distance_from_circle_to_point (z : ℂ) :
  Complex.abs z = 2 → (⨆ z, Complex.abs (z - Complex.I)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_from_circle_to_point_l1976_197606


namespace NUMINAMATH_CALUDE_even_function_four_roots_sum_zero_l1976_197668

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f has exactly four real roots if there exist exactly four distinct real numbers that make f(x) = 0 -/
def HasFourRealRoots (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧
    ∀ x, f x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d

/-- The theorem stating that for an even function with exactly four real roots, the sum of its roots is zero -/
theorem even_function_four_roots_sum_zero (f : ℝ → ℝ) 
    (h_even : IsEven f) (h_four_roots : HasFourRealRoots f) :
    ∃ (a b c d : ℝ), HasFourRealRoots f ∧ a + b + c + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_four_roots_sum_zero_l1976_197668


namespace NUMINAMATH_CALUDE_student_contribution_l1976_197669

/-- Proves that if 30 students contribute equally every Friday for 2 months (8 Fridays)
    and collect a total of $480, then each student's contribution per Friday is $2. -/
theorem student_contribution
  (num_students : ℕ)
  (num_fridays : ℕ)
  (total_amount : ℕ)
  (h1 : num_students = 30)
  (h2 : num_fridays = 8)
  (h3 : total_amount = 480) :
  total_amount / (num_students * num_fridays) = 2 :=
by sorry

end NUMINAMATH_CALUDE_student_contribution_l1976_197669


namespace NUMINAMATH_CALUDE_shaded_area_of_squares_l1976_197624

theorem shaded_area_of_squares (s₁ s₂ : ℝ) (h₁ : s₁ = 2) (h₂ : s₂ = 6) :
  (1/2 * s₁ * s₁) + (1/2 * s₂ * s₂) = 20 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_squares_l1976_197624


namespace NUMINAMATH_CALUDE_exchange_cookies_to_bagels_l1976_197613

/-- Represents the exchange rate between gingerbread cookies and drying rings -/
def cookie_to_ring : ℚ := 6

/-- Represents the exchange rate between drying rings and bagels -/
def ring_to_bagel : ℚ := 4/9

/-- Represents the number of gingerbread cookies we want to exchange -/
def cookies : ℚ := 3

/-- Theorem stating that 3 gingerbread cookies can be exchanged for 8 bagels -/
theorem exchange_cookies_to_bagels :
  cookies * cookie_to_ring * ring_to_bagel = 8 := by
  sorry

end NUMINAMATH_CALUDE_exchange_cookies_to_bagels_l1976_197613


namespace NUMINAMATH_CALUDE_favorite_number_ratio_l1976_197614

theorem favorite_number_ratio :
  ∀ (misty_number glory_number : ℕ),
    glory_number = 450 →
    misty_number + glory_number = 600 →
    glory_number / misty_number = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_favorite_number_ratio_l1976_197614


namespace NUMINAMATH_CALUDE_constant_term_of_product_l1976_197696

/-- The constant term in the expansion of (x^6 + x^2 + 3)(x^4 + x^3 + 20) is 60 -/
theorem constant_term_of_product (x : ℝ) : 
  (x^6 + x^2 + 3) * (x^4 + x^3 + 20) = x^10 + x^9 + 20*x^6 + x^7 + x^6 + 20*x^2 + 3*x^4 + 3*x^3 + 60 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_of_product_l1976_197696


namespace NUMINAMATH_CALUDE_x_value_equality_l1976_197603

theorem x_value_equality : (2023^2 - 2023 - 10000) / 2023 = (2022 * 2023 - 10000) / 2023 := by
  sorry

end NUMINAMATH_CALUDE_x_value_equality_l1976_197603


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l1976_197685

theorem quadratic_equation_k_value (x₁ x₂ k : ℝ) : 
  x₁^2 - 3*x₁ + k = 0 →
  x₂^2 - 3*x₂ + k = 0 →
  x₁*x₂ + 2*x₁ + 2*x₂ = 1 →
  k = -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l1976_197685


namespace NUMINAMATH_CALUDE_pizza_slices_l1976_197632

theorem pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_pizzas = 17) 
  (h2 : slices_per_pizza = 4) : 
  num_pizzas * slices_per_pizza = 68 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l1976_197632


namespace NUMINAMATH_CALUDE_substitution_remainder_l1976_197660

/-- Represents the number of ways to make substitutions in a basketball game -/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the remainder when the number of substitution ways is divided by 1000 -/
theorem substitution_remainder :
  substitution_ways 15 5 4 % 1000 = 301 := by
  sorry

end NUMINAMATH_CALUDE_substitution_remainder_l1976_197660


namespace NUMINAMATH_CALUDE_f_2023_equals_1_l1976_197640

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def period_4 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

theorem f_2023_equals_1 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 2) = f (2 - x))
  (h3 : ∀ x ∈ Set.Icc 0 2, f x = x^2) : 
  f 2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2023_equals_1_l1976_197640


namespace NUMINAMATH_CALUDE_sheela_net_monthly_income_l1976_197646

/-- Calculates the total net monthly income for Sheela given her various income sources and tax rates. -/
theorem sheela_net_monthly_income 
  (savings_deposit : ℝ)
  (savings_deposit_percentage : ℝ)
  (freelance_income : ℝ)
  (annual_interest : ℝ)
  (freelance_tax_rate : ℝ)
  (interest_tax_rate : ℝ)
  (h1 : savings_deposit = 5000)
  (h2 : savings_deposit_percentage = 0.20)
  (h3 : freelance_income = 3000)
  (h4 : annual_interest = 2400)
  (h5 : freelance_tax_rate = 0.10)
  (h6 : interest_tax_rate = 0.05) :
  ∃ (total_net_monthly_income : ℝ), 
    total_net_monthly_income = 27890 :=
by sorry

end NUMINAMATH_CALUDE_sheela_net_monthly_income_l1976_197646


namespace NUMINAMATH_CALUDE_sock_selection_problem_l1976_197600

theorem sock_selection_problem :
  let total_socks : ℕ := 7
  let socks_to_choose : ℕ := 4
  let number_of_ways : ℕ := Nat.choose total_socks socks_to_choose
  number_of_ways = 35 := by
sorry

end NUMINAMATH_CALUDE_sock_selection_problem_l1976_197600


namespace NUMINAMATH_CALUDE_sine_function_period_l1976_197675

/-- Given a function f(x) = √3 * sin(ωx + φ) where ω > 0, 
    if the distance between adjacent symmetry axes of the graph is 2π, 
    then ω = 1/2 -/
theorem sine_function_period (ω φ : ℝ) (h_ω_pos : ω > 0) :
  (∀ x : ℝ, ∃ k : ℤ, (x + 2 * π) = x + 2 * k * π / ω) →
  ω = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_function_period_l1976_197675


namespace NUMINAMATH_CALUDE_number_difference_l1976_197615

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 21800)
  (b_div_100 : 100 ∣ b)
  (a_eq_b_div_100 : a = b / 100) :
  b - a = 21384 := by sorry

end NUMINAMATH_CALUDE_number_difference_l1976_197615


namespace NUMINAMATH_CALUDE_problem_solution_l1976_197602

noncomputable def problem (e₁ e₂ OA OB : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  let A := OA
  let B := OB
  -- e₁ and e₂ are unit vectors in the direction of x-axis and y-axis
  e₁ = (1, 0) ∧ e₂ = (0, 1) ∧
  -- OA = e₁ + e₂
  OA = (e₁.1 + e₂.1, e₁.2 + e₂.2) ∧
  -- OB = 5e₁ + 3e₂
  OB = (5 * e₁.1 + 3 * e₂.1, 5 * e₁.2 + 3 * e₂.2) ∧
  -- AB ⟂ AC
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- |AB| = |AC|
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →
  OB.1 * C.1 + OB.2 * C.2 = 6 ∨ OB.1 * C.1 + OB.2 * C.2 = 10

theorem problem_solution :
  ∀ (e₁ e₂ OA OB : ℝ × ℝ) (C : ℝ × ℝ), problem e₁ e₂ OA OB C :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1976_197602


namespace NUMINAMATH_CALUDE_root_equivalence_l1976_197631

theorem root_equivalence (α : ℂ) : 
  α^2 - 2*α - 2 = 0 → α^5 - 44*α^3 - 32*α^2 - 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_root_equivalence_l1976_197631


namespace NUMINAMATH_CALUDE_resort_flat_fee_is_40_l1976_197661

/-- Represents the pricing scheme of a resort -/
structure ResortPricing where
  flatFee : ℕ  -- Flat fee for the first night
  additionalNightFee : ℕ  -- Fee for each additional night

/-- Calculates the total cost for a stay -/
def totalCost (pricing : ResortPricing) (nights : ℕ) : ℕ :=
  pricing.flatFee + (nights - 1) * pricing.additionalNightFee

/-- Theorem stating the flat fee given the conditions -/
theorem resort_flat_fee_is_40 :
  ∀ (pricing : ResortPricing),
    totalCost pricing 5 = 320 →
    totalCost pricing 8 = 530 →
    pricing.flatFee = 40 := by
  sorry


end NUMINAMATH_CALUDE_resort_flat_fee_is_40_l1976_197661


namespace NUMINAMATH_CALUDE_freds_weekend_earnings_l1976_197616

/-- Fred's earnings from delivering newspapers -/
def newspaper_earnings : ℕ := 16

/-- Fred's earnings from washing cars -/
def car_washing_earnings : ℕ := 74

/-- Fred's total weekend earnings -/
def weekend_earnings : ℕ := 90

/-- Theorem stating that Fred's weekend earnings equal the sum of his newspaper delivery and car washing earnings -/
theorem freds_weekend_earnings : 
  newspaper_earnings + car_washing_earnings = weekend_earnings := by
  sorry

end NUMINAMATH_CALUDE_freds_weekend_earnings_l1976_197616


namespace NUMINAMATH_CALUDE_ladder_problem_l1976_197636

theorem ladder_problem (ladder_length height base : ℝ) :
  ladder_length = 13 ∧ height = 12 ∧ ladder_length^2 = height^2 + base^2 → base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l1976_197636


namespace NUMINAMATH_CALUDE_toys_problem_toys_problem_unique_l1976_197645

/-- Given the number of toys for Kamari, calculates the total number of toys for all three children. -/
def total_toys (kamari_toys : ℕ) : ℕ :=
  kamari_toys + (kamari_toys + 30) + (2 * kamari_toys)

/-- Theorem stating that given the conditions, the total number of toys is 290. -/
theorem toys_problem : ∃ (k : ℕ), k + (k + 30) = 160 ∧ total_toys k = 290 :=
  sorry

/-- Corollary: The solution to the problem exists and is unique. -/
theorem toys_problem_unique : ∃! (k : ℕ), k + (k + 30) = 160 ∧ total_toys k = 290 :=
  sorry

end NUMINAMATH_CALUDE_toys_problem_toys_problem_unique_l1976_197645


namespace NUMINAMATH_CALUDE_marbles_sum_theorem_l1976_197655

/-- The number of yellow marbles Mary and Joan have in total -/
def total_marbles (mary_marbles joan_marbles : ℕ) : ℕ :=
  mary_marbles + joan_marbles

/-- Theorem stating that Mary and Joan have 12 yellow marbles in total -/
theorem marbles_sum_theorem :
  total_marbles 9 3 = 12 := by sorry

end NUMINAMATH_CALUDE_marbles_sum_theorem_l1976_197655


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1976_197608

/-- The equations of the asymptotes for the hyperbola x²/16 - y²/9 = 1 are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := fun x y ↦ x^2/16 - y^2/9 = 1
  ∃ (f g : ℝ → ℝ), (∀ x, f x = (3/4) * x) ∧ (∀ x, g x = -(3/4) * x) ∧
    (∀ ε > 0, ∃ M > 0, ∀ x y, h x y → (|x| > M → |y - f x| < ε ∨ |y - g x| < ε)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1976_197608


namespace NUMINAMATH_CALUDE_odd_sum_difference_l1976_197611

def sum_odd_range (a b : ℕ) : ℕ :=
  let first := if a % 2 = 1 then a else a + 1
  let last := if b % 2 = 1 then b else b - 1
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

theorem odd_sum_difference : 
  sum_odd_range 101 300 - sum_odd_range 3 70 = 18776 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_difference_l1976_197611


namespace NUMINAMATH_CALUDE_bicycle_spokes_theorem_l1976_197612

/-- The number of spokes on each bicycle wheel given the total number of bicycles and spokes -/
def spokes_per_wheel (num_bicycles : ℕ) (total_spokes : ℕ) : ℕ :=
  total_spokes / (num_bicycles * 2)

/-- Theorem stating that 4 bicycles with a total of 80 spokes have 10 spokes per wheel -/
theorem bicycle_spokes_theorem :
  spokes_per_wheel 4 80 = 10 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_spokes_theorem_l1976_197612


namespace NUMINAMATH_CALUDE_alice_ice_cream_l1976_197691

/-- The number of pints of ice cream Alice bought on Sunday -/
def sunday_pints : ℕ := sorry

/-- The number of pints Alice had on Wednesday after returning expired ones -/
def wednesday_pints : ℕ := 18

theorem alice_ice_cream :
  sunday_pints = 4 ∧
  3 * sunday_pints + sunday_pints + sunday_pints - sunday_pints / 2 = wednesday_pints :=
by sorry

end NUMINAMATH_CALUDE_alice_ice_cream_l1976_197691


namespace NUMINAMATH_CALUDE_wall_width_l1976_197601

/-- The width of a wall given its dimensions, brick dimensions, and number of bricks required. -/
theorem wall_width
  (wall_length : ℝ)
  (wall_height : ℝ)
  (brick_length : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (num_bricks : ℕ)
  (h1 : wall_length = 7)
  (h2 : wall_height = 6)
  (h3 : brick_length = 0.25)
  (h4 : brick_width = 0.1125)
  (h5 : brick_height = 0.06)
  (h6 : num_bricks = 5600) :
  ∃ (wall_width : ℝ), wall_width = 0.225 ∧
    wall_length * wall_height * wall_width = ↑num_bricks * brick_length * brick_width * brick_height :=
by sorry

end NUMINAMATH_CALUDE_wall_width_l1976_197601


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1976_197648

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | x*(x-2)*(x-5) < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | 1 < x ∧ x < 5} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1976_197648


namespace NUMINAMATH_CALUDE_three_distinct_solutions_l1976_197617

theorem three_distinct_solutions : ∃ (x₁ x₂ x₃ : ℝ), 
  (356 * x₁ = 2492) ∧ 
  (x₂ / 39 = 235) ∧ 
  (1908 - x₃ = 529) ∧ 
  (x₁ ≠ x₂) ∧ (x₁ ≠ x₃) ∧ (x₂ ≠ x₃) :=
by sorry

end NUMINAMATH_CALUDE_three_distinct_solutions_l1976_197617


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1976_197690

theorem min_value_sum_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 8) 
  (h2 : e * f * g * h = 16) : 
  ∀ x y z w : ℝ, x * y * z * w = 8 → 
  ∀ p q r s : ℝ, p * q * r * s = 16 →
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 
  (x * p)^2 + (y * q)^2 + (z * r)^2 + (w * s)^2 ∧
  (∃ x y z w p q r s : ℝ, x * y * z * w = 8 ∧ p * q * r * s = 16 ∧
  (x * p)^2 + (y * q)^2 + (z * r)^2 + (w * s)^2 = 32) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1976_197690


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1976_197629

theorem divisibility_theorem (a b c : ℕ) 
  (h1 : b ∣ a^3) 
  (h2 : c ∣ b^3) 
  (h3 : a ∣ c^3) : 
  a * b * c ∣ (a + b + c)^13 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1976_197629


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1976_197686

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, -9)
  parallel a b → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1976_197686


namespace NUMINAMATH_CALUDE_point_n_coordinates_l1976_197687

/-- Given point M(5, -6) and vector a = (1, -2), if MN = -3a, then N has coordinates (2, 0) -/
theorem point_n_coordinates (M N : ℝ × ℝ) (a : ℝ × ℝ) :
  M = (5, -6) →
  a = (1, -2) →
  N.1 - M.1 = -3 * a.1 ∧ N.2 - M.2 = -3 * a.2 →
  N = (2, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_n_coordinates_l1976_197687


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l1976_197651

theorem triangle_area_inequality (a b c S_triangle : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hS : S_triangle > 0)
  (h_triangle : S_triangle = Real.sqrt ((a + b + c) * (a + b - c) * (b + c - a) * (c + a - b) / 16)) :
  1 - (8 * ((a - b)^2 + (b - c)^2 + (c - a)^2)) / (a + b + c)^2 
  ≤ (432 * S_triangle^2) / (a + b + c)^4 
  ∧ (432 * S_triangle^2) / (a + b + c)^4 
  ≤ 1 - (2 * ((a - b)^2 + (b - c)^2 + (c - a)^2)) / (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l1976_197651


namespace NUMINAMATH_CALUDE_three_number_sum_l1976_197699

theorem three_number_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  b = 7 →
  (a + b + c) / 3 = a + 15 →
  (a + b + c) / 3 = c - 20 →
  a + b + c = 36 := by
sorry

end NUMINAMATH_CALUDE_three_number_sum_l1976_197699


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1976_197680

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≤ 4
def q (x : ℝ) : Prop := x^2 < 5*x - 6

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1976_197680


namespace NUMINAMATH_CALUDE_ant_climb_floors_l1976_197682

-- Define the problem parameters
def time_per_floor : ℕ := 15
def total_time : ℕ := 105
def start_floor : ℕ := 1

-- State the theorem
theorem ant_climb_floors :
  ∃ (final_floor : ℕ),
    final_floor = (total_time / time_per_floor) + start_floor ∧
    final_floor = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ant_climb_floors_l1976_197682


namespace NUMINAMATH_CALUDE_librarian_took_books_oliver_book_problem_l1976_197627

theorem librarian_took_books (total_books : ℕ) (books_per_shelf : ℕ) (shelves_needed : ℕ) : ℕ :=
  let remaining_books := shelves_needed * books_per_shelf
  total_books - remaining_books

theorem oliver_book_problem :
  librarian_took_books 46 4 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_librarian_took_books_oliver_book_problem_l1976_197627


namespace NUMINAMATH_CALUDE_carrie_tshirt_purchase_l1976_197674

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℝ := 9.15

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 22

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := tshirt_cost * num_tshirts

theorem carrie_tshirt_purchase : total_cost = 201.30 := by
  sorry

end NUMINAMATH_CALUDE_carrie_tshirt_purchase_l1976_197674


namespace NUMINAMATH_CALUDE_inequality_proof_l1976_197647

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum : a + b + c = 1) :
  (1/a + 1/(b*c)) * (1/b + 1/(c*a)) * (1/c + 1/(a*b)) ≥ 1728 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1976_197647


namespace NUMINAMATH_CALUDE_tan_periodic_angle_l1976_197663

theorem tan_periodic_angle (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (1720 * π / 180) → n = -80 :=
by sorry

end NUMINAMATH_CALUDE_tan_periodic_angle_l1976_197663


namespace NUMINAMATH_CALUDE_intercepted_triangle_area_l1976_197677

/-- The region defined by the inequality |x - 1| + |y - 2| ≤ 2 -/
def diamond_region (x y : ℝ) : Prop :=
  abs (x - 1) + abs (y - 2) ≤ 2

/-- The line y = 3x + 1 -/
def intercepting_line (x y : ℝ) : Prop :=
  y = 3 * x + 1

/-- The triangle intercepted by the line from the diamond region -/
def intercepted_triangle (x y : ℝ) : Prop :=
  diamond_region x y ∧ intercepting_line x y

/-- The area of the intercepted triangle -/
noncomputable def triangle_area : ℝ := 2

theorem intercepted_triangle_area :
  triangle_area = 2 :=
sorry

end NUMINAMATH_CALUDE_intercepted_triangle_area_l1976_197677


namespace NUMINAMATH_CALUDE_range_of_a_l1976_197637

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else 2^(x - 1)

theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ (0 ≤ a ∧ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1976_197637


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1976_197666

theorem quadratic_inequality (x : ℝ) : x^2 - 4*x - 21 < 0 ↔ -3 < x ∧ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1976_197666


namespace NUMINAMATH_CALUDE_exists_valid_relation_l1976_197623

-- Define the type for positive integers
def PositiveInt := {n : ℕ // n > 0}

-- Define the properties of the relation
def IsValidRelation (r : PositiveInt → PositiveInt → Prop) : Prop :=
  -- For any pair, exactly one of the three conditions holds
  (∀ a b : PositiveInt, (r a b ∨ r b a ∨ a = b) ∧ 
    (r a b → ¬r b a ∧ a ≠ b) ∧
    (r b a → ¬r a b ∧ a ≠ b) ∧
    (a = b → ¬r a b ∧ ¬r b a)) ∧
  -- Transitivity
  (∀ a b c : PositiveInt, r a b → r b c → r a c) ∧
  -- The special property
  (∀ a b c : PositiveInt, r a b → r b c → 2 * b.val ≠ a.val + c.val)

-- Theorem statement
theorem exists_valid_relation : ∃ r : PositiveInt → PositiveInt → Prop, IsValidRelation r :=
sorry

end NUMINAMATH_CALUDE_exists_valid_relation_l1976_197623


namespace NUMINAMATH_CALUDE_part_one_part_two_l1976_197683

/-- Given c > 0 and c ≠ 1, define p and q as follows:
    p: The function y = c^x is monotonically decreasing
    q: The function f(x) = x^2 - 2cx + 1 is increasing on the interval (1/2, +∞) -/
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y
def q (c : ℝ) : Prop := ∀ x y : ℝ, 1/2 < x ∧ x < y → x^2 - 2*c*x + 1 < y^2 - 2*c*y + 1

/-- Part 1: If p is true and ¬q is false, then 0 < c ≤ 1/2 -/
theorem part_one (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) (h3 : p c) (h4 : ¬¬(q c)) :
  0 < c ∧ c ≤ 1/2 := by sorry

/-- Part 2: If "p AND q" is false and "p OR q" is true, then 1/2 < c < 1 -/
theorem part_two (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) (h3 : ¬(p c ∧ q c)) (h4 : p c ∨ q c) :
  1/2 < c ∧ c < 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1976_197683


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_minus_b_l1976_197684

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- State the theorem
theorem extreme_value_implies_a_minus_b (a b : ℝ) :
  (f a b (-1) = 0) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b x ≥ f a b (-1)) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b x ≤ f a b (-1)) →
  a - b = -7 := by
  sorry


end NUMINAMATH_CALUDE_extreme_value_implies_a_minus_b_l1976_197684
