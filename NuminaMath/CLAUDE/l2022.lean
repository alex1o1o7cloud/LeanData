import Mathlib

namespace NUMINAMATH_CALUDE_line_intersects_circle_l2022_202289

/-- The line y - 1 = k(x - 1) always intersects the circle x² + y² - 2y = 0 for any real number k. -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  (y - 1 = k * (x - 1)) ∧ (x^2 + y^2 - 2*y = 0) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2022_202289


namespace NUMINAMATH_CALUDE_equation_solution_l2022_202274

theorem equation_solution :
  ∃ y : ℚ, (3 / y - 3 / y * y / 5 = 1.2) ∧ (y = 5 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2022_202274


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l2022_202264

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_geometric_mean 
  (d : ℝ) (k : ℕ) 
  (h_d : d ≠ 0) 
  (h_k : k > 0) :
  let a := arithmetic_sequence (9 * d) d
  (a k) ^ 2 = a 1 * a (2 * k) → k = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l2022_202264


namespace NUMINAMATH_CALUDE_triangle_theorem_l2022_202269

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Represents a 2D vector --/
structure Vector2D where
  x : ℝ
  y : ℝ

theorem triangle_theorem (t : Triangle) 
  (h_acute : 0 < t.A ∧ t.A < π/2 ∧ 0 < t.B ∧ t.B < π/2 ∧ 0 < t.C ∧ t.C < π/2)
  (h_sum : t.A + t.B + t.C = π)
  (μ : Vector2D)
  (v : Vector2D)
  (h_μ : μ = ⟨t.a^2 + t.c^2 - t.b^2, Real.sqrt 3 * t.a * t.c⟩)
  (h_v : v = ⟨Real.cos t.B, Real.sin t.B⟩)
  (h_parallel : ∃ (k : ℝ), μ = Vector2D.mk (k * v.x) (k * v.y)) :
  t.B = π/3 ∧ 3 * Real.sqrt 3 / 2 < Real.sin t.A + Real.sin t.C ∧ 
  Real.sin t.A + Real.sin t.C ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2022_202269


namespace NUMINAMATH_CALUDE_sum_of_root_products_l2022_202236

theorem sum_of_root_products (a b c d : ℂ) : 
  (2 * a^4 - 6 * a^3 + 14 * a^2 - 13 * a + 8 = 0) →
  (2 * b^4 - 6 * b^3 + 14 * b^2 - 13 * b + 8 = 0) →
  (2 * c^4 - 6 * c^3 + 14 * c^2 - 13 * c + 8 = 0) →
  (2 * d^4 - 6 * d^3 + 14 * d^2 - 13 * d + 8 = 0) →
  a * b + a * c + a * d + b * c + b * d + c * d = -7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_root_products_l2022_202236


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2022_202248

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property (a b c : ℝ) :
  QuadraticFunction a b c 0 = -1 →
  QuadraticFunction a b c 4 = QuadraticFunction a b c 5 →
  ∃ (n : ℤ), QuadraticFunction a b c 11 = n →
  QuadraticFunction a b c 11 = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2022_202248


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l2022_202281

theorem mayoral_election_votes (candidate_A_percentage : Real)
                               (candidate_B_percentage : Real)
                               (candidate_C_percentage : Real)
                               (candidate_D_percentage : Real)
                               (vote_difference : ℕ) :
  candidate_A_percentage = 0.35 →
  candidate_B_percentage = 0.40 →
  candidate_C_percentage = 0.15 →
  candidate_D_percentage = 0.10 →
  candidate_A_percentage + candidate_B_percentage + candidate_C_percentage + candidate_D_percentage = 1 →
  vote_difference = 2340 →
  ∃ total_votes : ℕ, 
    (candidate_B_percentage - candidate_A_percentage) * total_votes = vote_difference ∧
    total_votes = 46800 :=
by sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l2022_202281


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2022_202202

theorem x_plus_y_value (x y : ℝ) (h : (x - 1)^2 + |2*y + 1| = 0) : x + y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2022_202202


namespace NUMINAMATH_CALUDE_final_water_fraction_l2022_202216

def container_size : ℚ := 25

def initial_water : ℚ := 25

def replacement_volume : ℚ := 5

def third_replacement_water : ℚ := 2

def calculate_final_water_fraction (initial_water : ℚ) (container_size : ℚ) 
  (replacement_volume : ℚ) (third_replacement_water : ℚ) : ℚ :=
  sorry

theorem final_water_fraction :
  calculate_final_water_fraction initial_water container_size replacement_volume third_replacement_water
  = 14.8 / 25 :=
sorry

end NUMINAMATH_CALUDE_final_water_fraction_l2022_202216


namespace NUMINAMATH_CALUDE_sara_spent_calculation_l2022_202207

/-- Calculates the total amount Sara spent on movies and snacks -/
def sara_total_spent (ticket_price : ℝ) (num_tickets : ℕ) (student_discount : ℝ) 
  (rented_movie_price : ℝ) (purchased_movie_price : ℝ) (snacks_price : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let discounted_tickets := ticket_price * num_tickets * (1 - student_discount)
  let taxable_items := discounted_tickets + rented_movie_price + purchased_movie_price
  let sales_tax := taxable_items * sales_tax_rate
  discounted_tickets + rented_movie_price + purchased_movie_price + sales_tax + snacks_price

/-- Theorem stating that Sara's total spent is $43.89 -/
theorem sara_spent_calculation : 
  sara_total_spent 10.62 2 0.1 1.59 13.95 7.50 0.05 = 43.89 := by
  sorry

#eval sara_total_spent 10.62 2 0.1 1.59 13.95 7.50 0.05

end NUMINAMATH_CALUDE_sara_spent_calculation_l2022_202207


namespace NUMINAMATH_CALUDE_divisibility_by_fifteen_l2022_202261

theorem divisibility_by_fifteen (n : ℕ) : n < 10 →
  (∃ k : ℕ, 80000 + 10000 * n + 945 = 15 * k) ↔ n % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_fifteen_l2022_202261


namespace NUMINAMATH_CALUDE_nut_raisin_mixture_l2022_202208

/-- The number of pounds of nuts mixed with raisins -/
def pounds_of_nuts : ℝ := 4

/-- The number of pounds of raisins -/
def pounds_of_raisins : ℝ := 3

/-- The ratio of the cost of nuts to the cost of raisins -/
def cost_ratio : ℝ := 4

/-- The ratio of the cost of raisins to the total cost of the mixture -/
def raisin_cost_ratio : ℝ := 0.15789473684210525

theorem nut_raisin_mixture :
  let total_cost := pounds_of_raisins + cost_ratio * pounds_of_nuts
  raisin_cost_ratio * total_cost = pounds_of_raisins :=
by sorry

end NUMINAMATH_CALUDE_nut_raisin_mixture_l2022_202208


namespace NUMINAMATH_CALUDE_book_cost_proof_l2022_202224

-- Define the cost of one book
def p : ℝ := 1.76

-- State the theorem
theorem book_cost_proof :
  14 * p < 25 ∧ 16 * p > 28 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_proof_l2022_202224


namespace NUMINAMATH_CALUDE_extreme_values_and_zero_condition_l2022_202209

/-- The cubic function f(x) = x^3 - x^2 - x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - x - a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

theorem extreme_values_and_zero_condition (a : ℝ) :
  (∃ x_max : ℝ, f a x_max = 5/27 - a ∧ ∀ x, f a x ≤ f a x_max) ∧
  (∃ x_min : ℝ, f a x_min = -1 - a ∧ ∀ x, f a x ≥ f a x_min) ∧
  (∃! x, f a x = 0) ↔ (a < -1 ∨ a > 5/27) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_zero_condition_l2022_202209


namespace NUMINAMATH_CALUDE_clusters_per_box_l2022_202240

/-- Given the following conditions:
    1. There are 4 clusters of oats in each spoonful.
    2. There are 25 spoonfuls of cereal in each bowl.
    3. There are 5 bowlfuls of cereal in each box.
    Prove that the number of clusters of oats in each box is equal to 500. -/
theorem clusters_per_box (clusters_per_spoon : ℕ) (spoons_per_bowl : ℕ) (bowls_per_box : ℕ)
  (h1 : clusters_per_spoon = 4)
  (h2 : spoons_per_bowl = 25)
  (h3 : bowls_per_box = 5) :
  clusters_per_spoon * spoons_per_bowl * bowls_per_box = 500 := by
  sorry

end NUMINAMATH_CALUDE_clusters_per_box_l2022_202240


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2022_202201

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13
  right_angle : a^2 + b^2 = c^2

/-- A square inscribed in the triangle with one side on the leg of length 5 -/
def inscribed_square_on_leg (t : RightTriangle) (x : ℝ) : Prop :=
  x = t.a

/-- A square inscribed in the triangle with one side on the hypotenuse -/
def inscribed_square_on_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y / t.c = (t.b - 2*y) / t.b ∧ y / t.c = (t.a - y) / t.a

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ) 
  (hx : inscribed_square_on_leg t x) (hy : inscribed_square_on_hypotenuse t y) : 
  x / y = 18 / 13 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2022_202201


namespace NUMINAMATH_CALUDE_seating_theorem_l2022_202292

/-- The number of different seating arrangements for n people in m seats,
    where exactly two empty seats are adjacent -/
def seating_arrangements (m n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for 6 seats and 3 people,
    the number of seating arrangements with exactly two adjacent empty seats is 72 -/
theorem seating_theorem : seating_arrangements 6 3 = 72 :=
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l2022_202292


namespace NUMINAMATH_CALUDE_division_remainder_3500_74_l2022_202288

theorem division_remainder_3500_74 : ∃ q : ℕ, 3500 = 74 * q + 22 ∧ 22 < 74 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_3500_74_l2022_202288


namespace NUMINAMATH_CALUDE_radiator_fluid_calculation_l2022_202206

theorem radiator_fluid_calculation (initial_antifreeze_percentage : Real)
                                   (drain_amount : Real)
                                   (replacement_antifreeze_percentage : Real)
                                   (final_antifreeze_percentage : Real) :
  initial_antifreeze_percentage = 0.10 →
  drain_amount = 2.2857 →
  replacement_antifreeze_percentage = 0.80 →
  final_antifreeze_percentage = 0.50 →
  ∃ x : Real, x = 4 ∧
    initial_antifreeze_percentage * x - 
    initial_antifreeze_percentage * drain_amount + 
    replacement_antifreeze_percentage * drain_amount = 
    final_antifreeze_percentage * x :=
by
  sorry

end NUMINAMATH_CALUDE_radiator_fluid_calculation_l2022_202206


namespace NUMINAMATH_CALUDE_equation_solutions_l2022_202259

theorem equation_solutions :
  let f (x : ℝ) := (8*x^2 - 20*x + 3)/(2*x - 1) + 7*x
  ∀ x : ℝ, f x = 9*x - 3 ↔ x = 1/2 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2022_202259


namespace NUMINAMATH_CALUDE_gumball_draw_theorem_l2022_202276

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)

/-- Represents the minimum number of gumballs to draw to guarantee 4 of the same color -/
def minDrawToGuaranteeFour (machine : GumballMachine) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem gumball_draw_theorem (machine : GumballMachine) 
  (h1 : machine.red = 9)
  (h2 : machine.white = 7)
  (h3 : machine.blue = 12) :
  minDrawToGuaranteeFour machine = 12 :=
sorry

end NUMINAMATH_CALUDE_gumball_draw_theorem_l2022_202276


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2022_202299

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ),
  (4 * π * r^2 : ℝ) = 256 * π →
  (4 / 3 * π * r^3 : ℝ) = 2048 / 3 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2022_202299


namespace NUMINAMATH_CALUDE_fraction_increase_l2022_202214

theorem fraction_increase (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2*x * 2*y) / (2*x + 2*y) = 2 * (x*y / (x + y)) :=
sorry

end NUMINAMATH_CALUDE_fraction_increase_l2022_202214


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l2022_202291

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_root_existence 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hf1 : f a b c 1.4 = -0.24) 
  (hf2 : f a b c 1.5 = 0.25) :
  ∃ x₁ : ℝ, f a b c x₁ = 0 ∧ 1.4 < x₁ ∧ x₁ < 1.5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l2022_202291


namespace NUMINAMATH_CALUDE_grade_assignment_count_l2022_202244

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of possible grades -/
def num_grades : ℕ := 4

/-- The number of ways to assign grades to all students -/
def ways_to_assign_grades : ℕ := num_grades ^ num_students

theorem grade_assignment_count :
  ways_to_assign_grades = 16777216 :=
sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l2022_202244


namespace NUMINAMATH_CALUDE_coat_price_reduction_percentage_l2022_202252

/-- The percentage reduction when a coat's price is reduced from $500 to $150 is 70% -/
theorem coat_price_reduction_percentage : 
  let original_price : ℚ := 500
  let reduced_price : ℚ := 150
  let reduction : ℚ := original_price - reduced_price
  let percentage_reduction : ℚ := (reduction / original_price) * 100
  percentage_reduction = 70 := by sorry

end NUMINAMATH_CALUDE_coat_price_reduction_percentage_l2022_202252


namespace NUMINAMATH_CALUDE_new_savings_amount_l2022_202277

def monthly_salary : ℝ := 5750
def initial_savings_rate : ℝ := 0.20
def expense_increase_rate : ℝ := 0.20

theorem new_savings_amount :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let new_expenses := initial_expenses * (1 + expense_increase_rate)
  let new_savings := monthly_salary - new_expenses
  new_savings = 230 := by sorry

end NUMINAMATH_CALUDE_new_savings_amount_l2022_202277


namespace NUMINAMATH_CALUDE_milk_selection_l2022_202237

theorem milk_selection (total : ℕ) (soda_count : ℕ) (milk_percent : ℚ) (soda_percent : ℚ) :
  soda_percent = 60 / 100 →
  milk_percent = 20 / 100 →
  soda_count = 72 →
  (milk_percent / soda_percent) * soda_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_milk_selection_l2022_202237


namespace NUMINAMATH_CALUDE_number_value_l2022_202228

theorem number_value (number y : ℝ) 
  (h1 : (number + 5) * (y - 5) = 0)
  (h2 : ∀ a b : ℝ, (a + 5) * (b - 5) = 0 → a^2 + b^2 ≥ number^2 + y^2)
  (h3 : number^2 + y^2 = 25) : 
  number = -5 := by
sorry

end NUMINAMATH_CALUDE_number_value_l2022_202228


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l2022_202247

/-- Given a square with perimeter 80 and a right triangle with height 72,
    if their areas are equal, then the base of the triangle is 100/9 -/
theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) 
  (triangle_base : ℝ) : 
  square_perimeter = 80 →
  triangle_height = 72 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * triangle_base →
  triangle_base = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l2022_202247


namespace NUMINAMATH_CALUDE_subset_condition_intersection_empty_condition_l2022_202270

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
def B : Set ℝ := {x | x < -2 ∨ x > 5}

-- Theorem for part (1)
theorem subset_condition (m : ℝ) : A m ⊆ B ↔ m < 2 ∨ m > 4 := by sorry

-- Theorem for part (2)
theorem intersection_empty_condition (m : ℝ) : A m ∩ B = ∅ ↔ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_intersection_empty_condition_l2022_202270


namespace NUMINAMATH_CALUDE_employee_payments_correct_l2022_202293

def video_recorder_price (wholesale : ℝ) (markup : ℝ) : ℝ :=
  wholesale * (1 + markup)

def employee_payment (retail : ℝ) (discount : ℝ) : ℝ :=
  retail * (1 - discount)

theorem employee_payments_correct :
  let wholesale_A := 200
  let wholesale_B := 250
  let wholesale_C := 300
  let markup_A := 0.20
  let markup_B := 0.25
  let markup_C := 0.30
  let discount_X := 0.15
  let discount_Y := 0.18
  let discount_Z := 0.20
  
  let retail_A := video_recorder_price wholesale_A markup_A
  let retail_B := video_recorder_price wholesale_B markup_B
  let retail_C := video_recorder_price wholesale_C markup_C
  
  let payment_X := employee_payment retail_A discount_X
  let payment_Y := employee_payment retail_B discount_Y
  let payment_Z := employee_payment retail_C discount_Z
  
  payment_X = 204 ∧ payment_Y = 256.25 ∧ payment_Z = 312 :=
by sorry

end NUMINAMATH_CALUDE_employee_payments_correct_l2022_202293


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2022_202235

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y => y = -2 * x + 1
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y ↔ y = (1/2) * x - 3/2) →  -- Slope-intercept form of L1
  (L2 P.1 P.2) →                          -- L2 passes through P
  ((-2) * (1/2) = -1) →                   -- Slopes are negative reciprocals
  ∀ x y, L1 x y → L2 x y → (x - P.1) * (x - P.1) + (y - P.2) * (y - P.2) ≠ 0 →
    (x - P.1) * (x - 2) + (y - P.2) * (y - (-3)) = 0 -- Perpendicular condition
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2022_202235


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l2022_202272

/-- 
If in a triangle ABC, where a, b, c are the lengths of sides opposite to angles A, B, C respectively, 
and a * cos(B) = b * cos(A), then the triangle ABC is isosceles.
-/
theorem isosceles_triangle_condition (A B C a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : a * Real.cos B = b * Real.cos A) :
  a = b ∨ b = c ∨ a = c := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l2022_202272


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l2022_202223

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- Represents a total distance in miles and yards -/
structure TotalDistance :=
  (miles : ℕ)
  (yards : ℕ)

def marathon_distance : MarathonDistance :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 15

theorem marathon_remainder_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yards_per_mile ∧
    TotalDistance.yards (
      {miles := m, 
       yards := y} : TotalDistance
    ) = 495 ∧
    m * yards_per_mile + y = 
      num_marathons * (marathon_distance.miles * yards_per_mile + marathon_distance.yards) :=
by sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l2022_202223


namespace NUMINAMATH_CALUDE_line_l_theorem_l2022_202222

/-- Definition of line l -/
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

/-- Intercepts are equal -/
def equal_intercepts (a : ℝ) : Prop :=
  ∃ k, k = a - 2 ∧ k = (a - 2) / (a + 1)

/-- Line does not pass through second quadrant -/
def not_in_second_quadrant (a : ℝ) : Prop :=
  -(a + 1) ≥ 0 ∧ a - 2 ≤ 0

theorem line_l_theorem :
  (∀ a : ℝ, equal_intercepts a → (a = 2 ∨ a = 0)) ∧
  (∀ a : ℝ, not_in_second_quadrant a ↔ a ≤ -1) :=
sorry

end NUMINAMATH_CALUDE_line_l_theorem_l2022_202222


namespace NUMINAMATH_CALUDE_min_sum_p_q_l2022_202266

theorem min_sum_p_q (p q : ℕ) (hp : p > 1) (hq : q > 1) (h_eq : 17 * (p + 1) = 20 * (q + 1)) :
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 20 * (q' + 1) → p + q ≤ p' + q' ∧ p + q = 37 := by
sorry

end NUMINAMATH_CALUDE_min_sum_p_q_l2022_202266


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2022_202273

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * crossing_time

/-- Proof that a train with speed 180 km/h crossing a pole in 1.9998400127989762 seconds is approximately 99.992 meters long -/
theorem train_length_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ |train_length 180 1.9998400127989762 - 99.992| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2022_202273


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l2022_202219

theorem polynomial_uniqueness (Q : ℝ → ℝ) :
  (∀ x, Q x = Q 0 + Q 1 * x + Q 2 * x^2 + Q 3 * x^3) →
  Q (-1) = 2 →
  ∀ x, Q x = x^2 + 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l2022_202219


namespace NUMINAMATH_CALUDE_circle_area_outside_square_is_zero_l2022_202294

/-- A square with an inscribed circle -/
structure SquareWithCircle where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- The circle is inscribed in the square -/
  circle_inscribed : radius = side_length / 2

/-- The area of the portion of the circle outside the square is zero -/
theorem circle_area_outside_square_is_zero (s : SquareWithCircle) (h : s.side_length = 10) :
  Real.pi * s.radius ^ 2 - s.side_length ^ 2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_circle_area_outside_square_is_zero_l2022_202294


namespace NUMINAMATH_CALUDE_twenty_eight_billion_scientific_notation_l2022_202249

/-- Represents 28 billion -/
def twenty_eight_billion : ℕ := 28000000000

/-- The scientific notation representation of 28 billion -/
def scientific_notation : ℝ := 2.8 * (10 ^ 9)

theorem twenty_eight_billion_scientific_notation : 
  (twenty_eight_billion : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_twenty_eight_billion_scientific_notation_l2022_202249


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_l2022_202231

theorem consecutive_numbers_product (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 48) : 
  n * (n + 2) = 255 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_l2022_202231


namespace NUMINAMATH_CALUDE_smallest_number_l2022_202257

def yoongi_number : ℕ := 4
def jungkook_number : ℕ := 6 + 3
def yuna_number : ℕ := 5

theorem smallest_number : 
  yoongi_number ≤ jungkook_number ∧ yoongi_number ≤ yuna_number :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2022_202257


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_union_equal_B_iff_m_in_range_l2022_202256

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

-- Theorem 1
theorem intersection_when_m_is_one :
  A 1 ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2
theorem union_equal_B_iff_m_in_range (m : ℝ) :
  A m ∪ B = B ↔ m ≥ 3 ∨ m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_union_equal_B_iff_m_in_range_l2022_202256


namespace NUMINAMATH_CALUDE_min_value_x_l2022_202287

theorem min_value_x (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log x / Real.log 3 ≥ Real.log 5 / Real.log 3 + (1/3) * (Real.log y / Real.log 3)) :
  x ≥ 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_l2022_202287


namespace NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l2022_202200

/-- The minimum distance from any point on the ellipse x² + y²/3 = 1 to the line x + y = 4 is √2. -/
theorem min_distance_ellipse_to_line :
  ∀ (x y : ℝ), x^2 + y^2/3 = 1 →
  (∃ (x' y' : ℝ), x' + y' = 4 ∧ (x - x')^2 + (y - y')^2 ≥ 2) ∧
  (∃ (x₀ y₀ : ℝ), x₀ + y₀ = 4 ∧ (x - x₀)^2 + (y - y₀)^2 = 2) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l2022_202200


namespace NUMINAMATH_CALUDE_car_trip_speed_l2022_202295

/-- Given a car trip with the following properties:
  1. The car averages a certain speed for the first 6 hours.
  2. The car averages 46 miles per hour for each additional hour.
  3. The average speed for the entire trip is 34 miles per hour.
  4. The trip is 8 hours long.
  Prove that the average speed for the first 6 hours of the trip is 30 miles per hour. -/
theorem car_trip_speed (initial_speed : ℝ) : initial_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_speed_l2022_202295


namespace NUMINAMATH_CALUDE_yoga_studio_average_weight_l2022_202239

theorem yoga_studio_average_weight
  (num_men : ℕ)
  (num_women : ℕ)
  (avg_weight_men : ℝ)
  (avg_weight_women : ℝ)
  (h1 : num_men = 8)
  (h2 : num_women = 6)
  (h3 : avg_weight_men = 190)
  (h4 : avg_weight_women = 120)
  : (num_men * avg_weight_men + num_women * avg_weight_women) / (num_men + num_women) = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_yoga_studio_average_weight_l2022_202239


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l2022_202245

theorem arthur_walk_distance :
  let blocks_east : ℕ := 8
  let blocks_north : ℕ := 15
  let miles_per_block : ℝ := 0.25
  let miles_east : ℝ := blocks_east * miles_per_block
  let miles_north : ℝ := blocks_north * miles_per_block
  let diagonal_miles : ℝ := Real.sqrt (miles_east^2 + miles_north^2)
  let total_miles : ℝ := miles_east + miles_north + diagonal_miles
  total_miles = 10 := by
sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l2022_202245


namespace NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_l2022_202215

def number : ℕ := 264

-- Define a function to get the prime factors of a number
def prime_factors (n : ℕ) : List ℕ := sorry

-- Define a function to get the two smallest elements from a list
def two_smallest (l : List ℕ) : List ℕ := sorry

theorem sum_two_smallest_prime_factors :
  (two_smallest (prime_factors number)).sum = 5 := by sorry

end NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_l2022_202215


namespace NUMINAMATH_CALUDE_dihedral_angle_range_in_regular_prism_l2022_202233

theorem dihedral_angle_range_in_regular_prism (n : ℕ) (h : n > 2) :
  ∃ θ : ℝ, ((n - 2 : ℝ) / n) * π < θ ∧ θ < π :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_range_in_regular_prism_l2022_202233


namespace NUMINAMATH_CALUDE_polygon_sides_l2022_202218

/-- A convex polygon with n sides where the sum of all angles except one is 2970 degrees -/
structure ConvexPolygon where
  n : ℕ
  sum_except_one : ℝ
  convex : sum_except_one = 2970

theorem polygon_sides (p : ConvexPolygon) : p.n = 19 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2022_202218


namespace NUMINAMATH_CALUDE_square_field_with_pond_area_l2022_202250

/-- The area of a square field with a circular pond in its center -/
theorem square_field_with_pond_area (side : Real) (radius : Real) 
  (h1 : side = 14) (h2 : radius = 3) : 
  side^2 - π * radius^2 = 196 - 9 * π := by
  sorry

end NUMINAMATH_CALUDE_square_field_with_pond_area_l2022_202250


namespace NUMINAMATH_CALUDE_billy_is_48_l2022_202255

-- Define Billy's age and Joe's age
def billy_age : ℕ := sorry
def joe_age : ℕ := sorry

-- State the conditions
axiom age_relation : billy_age = 3 * joe_age
axiom age_sum : billy_age + joe_age = 64

-- Theorem to prove
theorem billy_is_48 : billy_age = 48 := by
  sorry

end NUMINAMATH_CALUDE_billy_is_48_l2022_202255


namespace NUMINAMATH_CALUDE_wire_length_l2022_202204

/-- Given two vertical poles on a flat surface, where:
    - The distance between the pole bottoms is 8 feet
    - The height of the first pole is 10 feet
    - The height of the second pole is 4 feet
    This theorem proves that the length of a wire stretched from the top of the taller pole
    to the top of the shorter pole is 10 feet. -/
theorem wire_length (pole1_height pole2_height pole_distance : ℝ) 
  (h1 : pole1_height = 10)
  (h2 : pole2_height = 4)
  (h3 : pole_distance = 8) :
  Real.sqrt ((pole1_height - pole2_height)^2 + pole_distance^2) = 10 := by
  sorry

#check wire_length

end NUMINAMATH_CALUDE_wire_length_l2022_202204


namespace NUMINAMATH_CALUDE_inequality_solution_l2022_202254

theorem inequality_solution (x : ℝ) : 
  (10 * x^3 + 20 * x^2 - 75 * x - 105) / ((3 * x - 4) * (x + 5)) < 5 ↔ 
  (x > -5 ∧ x < -1) ∨ (x > 4/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2022_202254


namespace NUMINAMATH_CALUDE_apple_stack_theorem_l2022_202212

/-- Calculates the number of apples in a pyramid-like stack --/
def appleStack (baseWidth : Nat) (baseLength : Nat) : Nat :=
  let layers := min baseWidth baseLength
  List.range layers
  |>.map (fun i => (baseWidth - i) * (baseLength - i))
  |>.sum

/-- Theorem: A pyramid-like stack of apples with a 4x6 base contains 50 apples --/
theorem apple_stack_theorem :
  appleStack 4 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_stack_theorem_l2022_202212


namespace NUMINAMATH_CALUDE_public_area_diameter_l2022_202298

/-- The diameter of the outer boundary of a circular public area -/
def outer_boundary_diameter (play_area_diameter : ℝ) (garden_width : ℝ) (track_width : ℝ) : ℝ :=
  play_area_diameter + 2 * (garden_width + track_width)

/-- Theorem: The diameter of the outer boundary of the running track is 34 feet -/
theorem public_area_diameter : outer_boundary_diameter 14 6 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_public_area_diameter_l2022_202298


namespace NUMINAMATH_CALUDE_certain_number_l2022_202213

theorem certain_number : ∃ x : ℝ, x + 0.675 = 0.8 ∧ x = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_l2022_202213


namespace NUMINAMATH_CALUDE_missing_files_l2022_202211

theorem missing_files (total : ℕ) (organized_afternoon : ℕ) : total = 60 → organized_afternoon = 15 → total - (total / 2 + organized_afternoon) = 15 := by
  sorry

end NUMINAMATH_CALUDE_missing_files_l2022_202211


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2022_202278

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 7/12) 
  (h2 : x - y = 1/12) : 
  x^2 - y^2 = 7/144 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2022_202278


namespace NUMINAMATH_CALUDE_population_ratio_l2022_202203

-- Define the populations of cities X, Y, and Z
variable (X Y Z : ℝ)

-- Condition 1: City X has a population 3 times as great as the population of City Y
def condition1 : Prop := X = 3 * Y

-- Condition 2: The ratio of the population of City X to the population of City Z is 6
def condition2 : Prop := X / Z = 6

-- Theorem: The ratio of the population of City Y to the population of City Z is 2
theorem population_ratio (h1 : condition1 X Y) (h2 : condition2 X Z) : Y / Z = 2 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_l2022_202203


namespace NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l2022_202221

theorem xy_zero_necessary_not_sufficient :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x * y = 0) ∧
  (∃ x y : ℝ, x * y = 0 ∧ x^2 + y^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l2022_202221


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2022_202220

theorem polynomial_factorization (a : ℝ) : 
  49 * a^3 + 245 * a^2 + 588 * a = 49 * a * (a^2 + 5 * a + 12) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2022_202220


namespace NUMINAMATH_CALUDE_sin_product_10_30_50_70_l2022_202229

theorem sin_product_10_30_50_70 : 
  Real.sin (10 * π / 180) * Real.sin (30 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_sin_product_10_30_50_70_l2022_202229


namespace NUMINAMATH_CALUDE_high_school_ten_games_l2022_202230

/-- The number of teams in the conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team -/
def games_per_pair : ℕ := 2

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := (num_teams.choose 2) * games_per_pair + num_teams * non_conference_games

theorem high_school_ten_games : total_games = 150 := by
  sorry

end NUMINAMATH_CALUDE_high_school_ten_games_l2022_202230


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l2022_202205

/-- The hyperbola equation x^2 - y^2 = 4 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

/-- The line equation y = k(x-1) -/
def line (k x y : ℝ) : Prop := y = k * (x - 1)

/-- The number of intersection points between the line and the hyperbola -/
def intersection_count (k : ℝ) : ℕ := sorry

theorem hyperbola_line_intersection (k : ℝ) :
  (intersection_count k = 2 ↔ k ∈ Set.Ioo (-2 * Real.sqrt 3 / 3) (-1) ∪ 
                            Set.Ioo (-1) 1 ∪ 
                            Set.Ioo 1 (2 * Real.sqrt 3 / 3)) ∧
  (intersection_count k = 1 ↔ k = 1 ∨ k = -1 ∨ k = 2 * Real.sqrt 3 / 3 ∨ k = -2 * Real.sqrt 3 / 3) ∧
  (intersection_count k = 0 ↔ k ∈ Set.Iic (-2 * Real.sqrt 3 / 3) ∪ 
                            Set.Ici (2 * Real.sqrt 3 / 3)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l2022_202205


namespace NUMINAMATH_CALUDE_insurance_payment_calculation_l2022_202286

/-- The amount of a quarterly insurance payment in dollars. -/
def quarterly_payment : ℕ := 378

/-- The number of quarters in a year. -/
def quarters_per_year : ℕ := 4

/-- The annual insurance payment in dollars. -/
def annual_payment : ℕ := quarterly_payment * quarters_per_year

theorem insurance_payment_calculation :
  annual_payment = 1512 :=
by sorry

end NUMINAMATH_CALUDE_insurance_payment_calculation_l2022_202286


namespace NUMINAMATH_CALUDE_triangle_circle_area_relation_l2022_202280

theorem triangle_circle_area_relation (a b c : ℝ) (A B C : ℝ) : 
  a = 13 ∧ b = 14 ∧ c = 15 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  C ≥ A ∧ C ≥ B →
  (a + b + c) / 2 = 21 →
  Real.sqrt (21 * (21 - a) * (21 - b) * (21 - c)) = 84 →
  A + B + 84 = C := by
sorry

end NUMINAMATH_CALUDE_triangle_circle_area_relation_l2022_202280


namespace NUMINAMATH_CALUDE_cost_of_graveling_roads_l2022_202285

/-- The cost of graveling two intersecting roads on a rectangular lawn. -/
theorem cost_of_graveling_roads
  (lawn_length lawn_width road_width : ℕ)
  (cost_per_sq_m : ℚ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 60)
  (h3 : road_width = 10)
  (h4 : cost_per_sq_m = 2) :
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * cost_per_sq_m = 2600 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_graveling_roads_l2022_202285


namespace NUMINAMATH_CALUDE_elsa_marbles_proof_l2022_202242

/-- The number of marbles in Elsa's new bag -/
def new_bag_marbles : ℕ := by sorry

theorem elsa_marbles_proof :
  let initial_marbles : ℕ := 40
  let lost_at_breakfast : ℕ := 3
  let given_to_susie : ℕ := 5
  let final_marbles : ℕ := 54
  
  new_bag_marbles = 
    final_marbles - 
    (initial_marbles - lost_at_breakfast - given_to_susie + 2 * given_to_susie) := by sorry

end NUMINAMATH_CALUDE_elsa_marbles_proof_l2022_202242


namespace NUMINAMATH_CALUDE_min_chips_to_capture_all_l2022_202283

/-- Represents a rhombus-shaped game board -/
structure GameBoard :=
  (angle : ℝ)
  (side_divisions : ℕ)

/-- Represents a chip on the game board -/
structure Chip :=
  (position : ℕ × ℕ)

/-- Calculates the number of cells captured by a single chip -/
def cells_captured_by_chip (board : GameBoard) (chip : Chip) : ℕ :=
  sorry

/-- Calculates the total number of cells on the game board -/
def total_cells (board : GameBoard) : ℕ :=
  sorry

/-- Checks if a set of chips captures all cells on the board -/
def captures_all_cells (board : GameBoard) (chips : Finset Chip) : Prop :=
  sorry

/-- The main theorem stating the minimum number of chips required -/
theorem min_chips_to_capture_all (board : GameBoard) :
  board.angle = 60 ∧ board.side_divisions = 9 →
  ∃ (chips : Finset Chip), chips.card = 6 ∧ captures_all_cells board chips ∧
  ∀ (other_chips : Finset Chip), captures_all_cells board other_chips → other_chips.card ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_chips_to_capture_all_l2022_202283


namespace NUMINAMATH_CALUDE_friends_walking_problem_l2022_202226

/-- Two friends walking on a trail problem -/
theorem friends_walking_problem (v : ℝ) (h : v > 0) :
  let trail_length : ℝ := 22
  let speed_ratio : ℝ := 1.2
  let d : ℝ := trail_length / (1 + speed_ratio)
  trail_length - d = 12 := by sorry

end NUMINAMATH_CALUDE_friends_walking_problem_l2022_202226


namespace NUMINAMATH_CALUDE_courtyard_width_l2022_202251

/-- Proves that the width of a courtyard is 25 feet given specific conditions --/
theorem courtyard_width : ∀ (width : ℝ),
  (width > 0) →  -- Ensure width is positive
  (4 * 10 * width * (0.4 * 3 + 0.6 * 1.5) = 2100) →
  width = 25 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_width_l2022_202251


namespace NUMINAMATH_CALUDE_insufficient_payment_l2022_202279

def egg_price : ℝ := 3
def pancake_price : ℝ := 2
def cocoa_price : ℝ := 2
def croissant_price : ℝ := 1
def tax_rate : ℝ := 0.07

def initial_order_cost : ℝ := 4 * egg_price + 3 * pancake_price + 5 * cocoa_price + 2 * croissant_price

def additional_order_cost : ℝ := 2 * 3 * pancake_price + 3 * cocoa_price

def total_cost_before_tax : ℝ := initial_order_cost + additional_order_cost

def total_cost_with_tax : ℝ := total_cost_before_tax * (1 + tax_rate)

def payment : ℝ := 50

theorem insufficient_payment : total_cost_with_tax > payment ∧ 
  total_cost_with_tax - payment = 1.36 := by sorry

end NUMINAMATH_CALUDE_insufficient_payment_l2022_202279


namespace NUMINAMATH_CALUDE_toms_profit_is_21988_l2022_202234

/-- Calculates Tom's profit from making the world's largest dough ball -/
def toms_profit (flour_needed : ℕ) (flour_bag_size : ℕ) (flour_bag_cost : ℕ)
                (salt_needed : ℕ) (salt_cost : ℚ)
                (sugar_needed : ℕ) (sugar_cost : ℚ)
                (butter_needed : ℕ) (butter_cost : ℕ)
                (chef_cost : ℕ) (promotion_cost : ℕ)
                (ticket_price : ℕ) (tickets_sold : ℕ) : ℤ :=
  let flour_cost := (flour_needed / flour_bag_size) * flour_bag_cost
  let salt_cost_total := (salt_needed : ℚ) * salt_cost
  let sugar_cost_total := (sugar_needed : ℚ) * sugar_cost
  let butter_cost_total := butter_needed * butter_cost
  let total_cost := flour_cost + salt_cost_total.ceil + sugar_cost_total.ceil + 
                    butter_cost_total + chef_cost + promotion_cost
  let revenue := ticket_price * tickets_sold
  revenue - total_cost

/-- Tom's profit from making the world's largest dough ball is $21988 -/
theorem toms_profit_is_21988 : 
  toms_profit 500 50 20 10 (2/10) 20 (1/2) 50 2 700 1000 20 1200 = 21988 := by
  sorry

end NUMINAMATH_CALUDE_toms_profit_is_21988_l2022_202234


namespace NUMINAMATH_CALUDE_product_mod_25_l2022_202275

theorem product_mod_25 : ∃ n : ℕ, 0 ≤ n ∧ n < 25 ∧ (123 * 456 * 789) % 25 = n ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_25_l2022_202275


namespace NUMINAMATH_CALUDE_triangle_side_b_l2022_202267

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_side_b (t : Triangle) : 
  t.C = 4 * t.A →  -- ∠C = 4∠A
  t.a = 15 →       -- side a = 15
  t.c = 60 →       -- side c = 60
  t.b = 15 * Real.sqrt (2 + Real.sqrt 2) := by
    sorry


end NUMINAMATH_CALUDE_triangle_side_b_l2022_202267


namespace NUMINAMATH_CALUDE_spider_cylinder_ratio_l2022_202296

/-- In a cylindrical room, a spider can reach the opposite point on the floor
    by two paths of equal length. This theorem proves the ratio of the cylinder's
    height to its diameter given these conditions. -/
theorem spider_cylinder_ratio (m r : ℝ) (h_positive : m > 0 ∧ r > 0) :
  (m + 2*r = Real.sqrt (m^2 + (r*Real.pi)^2)) →
  m / (2*r) = (Real.pi^2 - 4) / 8 := by
  sorry

#check spider_cylinder_ratio

end NUMINAMATH_CALUDE_spider_cylinder_ratio_l2022_202296


namespace NUMINAMATH_CALUDE_tutors_next_meeting_l2022_202284

theorem tutors_next_meeting (elise_schedule fiona_schedule george_schedule harry_schedule : ℕ) 
  (h_elise : elise_schedule = 5)
  (h_fiona : fiona_schedule = 6)
  (h_george : george_schedule = 8)
  (h_harry : harry_schedule = 9) :
  Nat.lcm elise_schedule (Nat.lcm fiona_schedule (Nat.lcm george_schedule harry_schedule)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutors_next_meeting_l2022_202284


namespace NUMINAMATH_CALUDE_deck_size_proof_l2022_202232

theorem deck_size_proof (spades : ℕ) (prob_not_spade : ℚ) (total : ℕ) : 
  spades = 13 → 
  prob_not_spade = 3/4 → 
  (total - spades : ℚ) / total = prob_not_spade → 
  total = 52 := by
sorry

end NUMINAMATH_CALUDE_deck_size_proof_l2022_202232


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l2022_202297

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l2022_202297


namespace NUMINAMATH_CALUDE_initial_distance_calculation_l2022_202282

/-- Calculates the initial distance between a criminal and a policeman given their speeds and the distance after a certain time. -/
theorem initial_distance_calculation 
  (criminal_speed : ℝ) 
  (policeman_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : criminal_speed = 8) 
  (h2 : policeman_speed = 9) 
  (h3 : time = 3 / 60) 
  (h4 : final_distance = 190) : 
  ∃ (initial_distance : ℝ), 
    initial_distance = final_distance + (policeman_speed - criminal_speed) * time ∧ 
    initial_distance = 190.05 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_calculation_l2022_202282


namespace NUMINAMATH_CALUDE_counterexample_exists_l2022_202271

theorem counterexample_exists : ∃ (a b c : ℝ), 
  (a^2 + b^2) / (b^2 + c^2) = a/c ∧ a/b ≠ b/c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2022_202271


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l2022_202253

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B : Set ℝ := {x | 3 - x > 0}

-- Define the open interval (1, 2)
def open_interval : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = open_interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l2022_202253


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2022_202268

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2022_202268


namespace NUMINAMATH_CALUDE_michael_tom_flying_robots_ratio_l2022_202225

theorem michael_tom_flying_robots_ratio : 
  ∀ (michael_robots tom_robots : ℕ), 
    michael_robots = 12 → 
    tom_robots = 3 → 
    (michael_robots : ℚ) / (tom_robots : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_michael_tom_flying_robots_ratio_l2022_202225


namespace NUMINAMATH_CALUDE_inequality_solution_set_a_range_l2022_202262

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2|

-- Part 1
theorem inequality_solution_set (x : ℝ) :
  (2 * f x < 4 - |x - 1|) ↔ (-7/3 < x ∧ x < -1) :=
sorry

-- Part 2
theorem a_range (a m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) ↔ (-6 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_a_range_l2022_202262


namespace NUMINAMATH_CALUDE_same_last_four_digits_l2022_202260

theorem same_last_four_digits (N : ℕ) : 
  N > 0 ∧ 
  ∃ (a b c d : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
  N % 10000 = a * 1000 + b * 100 + c * 10 + d ∧
  (N * N) % 10000 = a * 1000 + b * 100 + c * 10 + d →
  N / 1000 = 937 :=
by sorry

end NUMINAMATH_CALUDE_same_last_four_digits_l2022_202260


namespace NUMINAMATH_CALUDE_launderette_machines_l2022_202210

/-- Represents the number of quarters in each machine -/
def quarters_per_machine : ℕ := 80

/-- Represents the number of dimes in each machine -/
def dimes_per_machine : ℕ := 100

/-- Represents the total amount of money after emptying all machines (in cents) -/
def total_amount : ℕ := 9000  -- $90 in cents

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Theorem stating that the number of machines in the launderette is 3 -/
theorem launderette_machines : 
  ∃ (n : ℕ), n * (quarters_per_machine * quarter_value + dimes_per_machine * dime_value) = total_amount ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_launderette_machines_l2022_202210


namespace NUMINAMATH_CALUDE_unique_intersection_point_l2022_202241

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-3, -3) := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l2022_202241


namespace NUMINAMATH_CALUDE_range_of_a_l2022_202243

-- Define the propositions p and q
def p (x : ℝ) : Prop := 2 * x^2 - 5 * x + 3 < 0
def q (x a : ℝ) : Prop := (x - (2 * a + 1)) * (x - 2 * a) ≤ 0

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ (∃ x, p x ∧ ¬q x a)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, necessary_not_sufficient a ↔ (1/4 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2022_202243


namespace NUMINAMATH_CALUDE_horner_method_v4_l2022_202238

def f (x : ℝ) : ℝ := 3*x^5 + 5*x^4 + 6*x^3 - 8*x^2 + 35*x + 12

def horner_v4 (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₅ * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀

theorem horner_method_v4 :
  horner_v4 3 5 6 (-8) 35 12 (-2) = 83 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v4_l2022_202238


namespace NUMINAMATH_CALUDE_problem_solution_l2022_202290

-- Define the absolute value function
def f (x : ℝ) : ℝ := abs x

-- Define the function g
def g (t : ℝ) (x : ℝ) : ℝ := 3 * f x - f (x - t)

theorem problem_solution :
  -- Part I
  (∀ a : ℝ, a > 0 → (∀ x : ℝ, 2 * f (x - 1) + f (2 * x - a) ≥ 1) →
    (a ∈ Set.Ioo 0 1 ∪ Set.Ici 3)) ∧
  -- Part II
  (∀ t : ℝ, t ≠ 0 →
    (∫ x, abs (g t x)) = 3 →
    t = 2 * Real.sqrt 2 ∨ t = -2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2022_202290


namespace NUMINAMATH_CALUDE_snow_probability_l2022_202246

theorem snow_probability (p : ℚ) (n : ℕ) (hp : p = 3/4) (hn : n = 5) :
  1 - (1 - p)^n = 1023/1024 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l2022_202246


namespace NUMINAMATH_CALUDE_same_hair_count_l2022_202258

theorem same_hair_count (population : ℕ) (hair_count : Fin population → ℕ) 
  (h1 : population > 500001) 
  (h2 : ∀ p, hair_count p ≤ 500000) : 
  ∃ p1 p2, p1 ≠ p2 ∧ hair_count p1 = hair_count p2 := by
  sorry

end NUMINAMATH_CALUDE_same_hair_count_l2022_202258


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l2022_202227

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (45/23, -64/23)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 8 * x - 3 * y = 24

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 10 * x + 2 * y = 14

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l2022_202227


namespace NUMINAMATH_CALUDE_ellipse_equation_l2022_202217

/-- Definition of an ellipse with given focal distance and major axis length -/
structure Ellipse :=
  (focal_distance : ℝ)
  (major_axis_length : ℝ)

/-- Standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), 
    a = e.major_axis_length / 2 ∧
    b^2 = a^2 - (e.focal_distance / 2)^2 ∧
    x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating the standard equation of the given ellipse -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.focal_distance = 8)
  (h2 : e.major_axis_length = 10) :
  standard_equation e x y ↔ x^2 / 25 + y^2 / 9 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2022_202217


namespace NUMINAMATH_CALUDE_product_of_ratios_l2022_202263

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2030) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2029)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2030) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2029)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2030) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2029)
  (h₇ : y₁ ≠ 0) (h₈ : y₂ ≠ 0) (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/1015 := by
sorry

end NUMINAMATH_CALUDE_product_of_ratios_l2022_202263


namespace NUMINAMATH_CALUDE_carnation_percentage_l2022_202265

/-- Represents a flower bouquet with various types of flowers -/
structure Bouquet where
  total : ℕ
  pink_roses : ℕ
  red_roses : ℕ
  pink_carnations : ℕ
  red_carnations : ℕ
  yellow_tulips : ℕ

/-- Conditions for the flower bouquet -/
def validBouquet (b : Bouquet) : Prop :=
  b.pink_roses + b.red_roses + b.pink_carnations + b.red_carnations + b.yellow_tulips = b.total ∧
  b.pink_roses + b.pink_carnations = b.total / 2 ∧
  b.pink_roses = (b.pink_roses + b.pink_carnations) * 2 / 5 ∧
  b.red_carnations = (b.red_roses + b.red_carnations) * 6 / 7 ∧
  b.yellow_tulips = b.total / 5

/-- Theorem stating that for a valid bouquet, 55% of the flowers are carnations -/
theorem carnation_percentage (b : Bouquet) (h : validBouquet b) :
  (b.pink_carnations + b.red_carnations) * 100 / b.total = 55 := by
  sorry

end NUMINAMATH_CALUDE_carnation_percentage_l2022_202265
