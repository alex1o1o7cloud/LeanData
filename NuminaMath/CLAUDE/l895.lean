import Mathlib

namespace total_triangle_area_is_36_l895_89547

/-- Represents a square in the grid -/
structure Square where
  x : Nat
  y : Nat
  deriving Repr

/-- Represents a triangle in a square -/
structure Triangle where
  square : Square
  deriving Repr

/-- The size of the grid -/
def gridSize : Nat := 6

/-- Calculate the area of a single triangle -/
def triangleArea : ℝ := 0.5

/-- Calculate the number of triangles in a square -/
def trianglesPerSquare : Nat := 2

/-- Calculate the total number of squares in the grid -/
def totalSquares : Nat := gridSize * gridSize

/-- Calculate the total area of all triangles in the grid -/
def totalTriangleArea : ℝ :=
  (totalSquares : ℝ) * (trianglesPerSquare : ℝ) * triangleArea

/-- Theorem stating that the total area of triangles in the grid is 36 -/
theorem total_triangle_area_is_36 : totalTriangleArea = 36 := by
  sorry

#eval totalTriangleArea

end total_triangle_area_is_36_l895_89547


namespace special_ellipse_intersecting_line_l895_89577

/-- An ellipse with its upper vertex and left focus on a given line --/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 + y^2 / b^2 = 1
  vertex_focus_line : ℝ → ℝ → Prop := fun x y => x - y + 2 = 0

/-- A line intersecting the ellipse --/
structure IntersectingLine (E : SpecialEllipse) where
  l : ℝ → ℝ → Prop
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_P : E.equation P.1 P.2 ∧ l P.1 P.2
  h_Q : E.equation Q.1 Q.2 ∧ l Q.1 Q.2
  h_midpoint : (P.1 + Q.1) / 2 = -1 ∧ (P.2 + Q.2) / 2 = 1

/-- The main theorem --/
theorem special_ellipse_intersecting_line 
  (E : SpecialEllipse) 
  (h_E : E.a^2 = 8 ∧ E.b^2 = 4) 
  (l : IntersectingLine E) : 
  l.l = fun x y => x - 2*y + 3 = 0 := by sorry

end special_ellipse_intersecting_line_l895_89577


namespace equation_solution_l895_89511

theorem equation_solution (x : ℝ) : 
  (∀ z : ℝ, 10 * x * z - 15 * z + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := by
sorry

end equation_solution_l895_89511


namespace sum_of_squares_l895_89542

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_eq_seventh : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/7 := by
  sorry

end sum_of_squares_l895_89542


namespace max_triples_value_l895_89597

/-- The size of the square table -/
def n : ℕ := 999

/-- Represents the color of a cell in the table -/
inductive CellColor
| White
| Red

/-- Represents a cell in the table -/
structure Cell where
  row : Fin n
  col : Fin n

/-- Represents the coloring of the table -/
def TableColoring := Fin n → Fin n → CellColor

/-- Counts the number of valid triples for a given table coloring -/
def countTriples (coloring : TableColoring) : ℕ := sorry

/-- The maximum number of valid triples possible -/
def maxTriples : ℕ := (4 * n^4) / 27

/-- Theorem stating that the maximum number of valid triples is (4 * 999⁴) / 27 -/
theorem max_triples_value :
  ∀ (coloring : TableColoring), countTriples coloring ≤ maxTriples :=
by sorry

end max_triples_value_l895_89597


namespace polynomial_identity_l895_89509

-- Define a polynomial function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem polynomial_identity 
  (h : ∀ x : ℝ, f (x^2 + 2) = x^4 + 5*x^2 + 1) :
  ∀ x : ℝ, f (x^2 - 2) = x^4 - 3*x^2 - 3 :=
by
  sorry

end polynomial_identity_l895_89509


namespace max_stamps_purchasable_l895_89564

def stamp_price : ℕ := 25
def available_amount : ℕ := 4000

theorem max_stamps_purchasable :
  ∀ n : ℕ, n * stamp_price ≤ available_amount ↔ n ≤ 160 :=
by sorry

end max_stamps_purchasable_l895_89564


namespace rebecca_gave_two_caps_l895_89531

def initial_caps : ℕ := 7
def final_caps : ℕ := 9

theorem rebecca_gave_two_caps : final_caps - initial_caps = 2 := by
  sorry

end rebecca_gave_two_caps_l895_89531


namespace valid_outfit_choices_l895_89522

/-- Represents the number of valid outfit choices given specific clothing items and constraints. -/
theorem valid_outfit_choices : 
  -- Define the number of shirts, pants, and their colors
  let num_shirts : ℕ := 6
  let num_pants : ℕ := 6
  let num_colors : ℕ := 6
  
  -- Define the number of hats
  let num_patterned_hats : ℕ := 6
  let num_solid_hats : ℕ := 6
  let total_hats : ℕ := num_patterned_hats + num_solid_hats
  
  -- Calculate total combinations
  let total_combinations : ℕ := num_shirts * num_pants * total_hats
  
  -- Calculate invalid combinations
  let same_color_combinations : ℕ := num_colors
  let pattern_mismatch_combinations : ℕ := num_patterned_hats * num_shirts * (num_pants - 1)
  
  -- Calculate valid combinations
  let valid_combinations : ℕ := total_combinations - same_color_combinations - pattern_mismatch_combinations
  
  -- Prove that the number of valid outfit choices is 246
  valid_combinations = 246 := by
    sorry

end valid_outfit_choices_l895_89522


namespace trip_duration_l895_89560

theorem trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) :
  initial_speed = 70 ∧
  initial_time = 4 ∧
  additional_speed = 60 ∧
  average_speed = 65 →
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed ∧
    total_time = 8 :=
by sorry

end trip_duration_l895_89560


namespace total_apples_in_basket_l895_89538

def initial_apples : Nat := 8
def added_apples : Nat := 7

theorem total_apples_in_basket : initial_apples + added_apples = 15 := by
  sorry

end total_apples_in_basket_l895_89538


namespace quadratic_bound_l895_89553

/-- Given a quadratic function f(x) = a x^2 + b x + c, if for any |u| ≤ 10/11 there exists a v 
    such that |u-v| ≤ 1/11 and |f(v)| ≤ 1, then for all x in [-1, 1], |f(x)| ≤ 2. -/
theorem quadratic_bound (a b c : ℝ) : 
  (∀ u : ℝ, |u| ≤ 10/11 → ∃ v : ℝ, |u - v| ≤ 1/11 ∧ |a * v^2 + b * v + c| ≤ 1) →
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 2) :=
by sorry

end quadratic_bound_l895_89553


namespace two_face_painted_count_l895_89543

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  side_length : Nat
  painted_faces : Nat

/-- Counts the number of smaller cubes painted on exactly two faces -/
def count_two_face_painted (c : CutCube) : Nat :=
  if c.side_length = 3 ∧ c.painted_faces = 6 then
    24
  else
    0

theorem two_face_painted_count (c : CutCube) :
  c.side_length = 3 ∧ c.painted_faces = 6 → count_two_face_painted c = 24 := by
  sorry

end two_face_painted_count_l895_89543


namespace aluminum_mass_calculation_l895_89523

/-- Given two parts with equal volume but different densities, 
    calculate the mass of one part given the mass difference. -/
theorem aluminum_mass_calculation 
  (ρA ρM : ℝ) -- densities of aluminum and copper
  (Δm : ℝ) -- mass difference
  (h1 : ρA = 2700) -- density of aluminum
  (h2 : ρM = 8900) -- density of copper
  (h3 : Δm = 0.06) -- mass difference in kg
  (h4 : ρM > ρA) -- copper is denser than aluminum
  : ∃ (mA : ℝ), mA = (ρA * Δm) / (ρM - ρA) :=
by sorry

end aluminum_mass_calculation_l895_89523


namespace conjecture_proof_l895_89567

theorem conjecture_proof (n : ℕ) (h : n ≥ 1) :
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end conjecture_proof_l895_89567


namespace f_minimum_value_l895_89552

def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ 1) ∧ (∃ x : ℝ, f x = 1) := by sorry

end f_minimum_value_l895_89552


namespace complement_intersection_theorem_l895_89550

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_theorem : (U \ A) ∩ B = {3, 4} := by
  sorry

end complement_intersection_theorem_l895_89550


namespace f_composition_value_l895_89508

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else Real.sin x

theorem f_composition_value : f (f (7 * Real.pi / 6)) = Real.sqrt 2 / 2 := by
  sorry

end f_composition_value_l895_89508


namespace min_product_of_tangents_l895_89569

theorem min_product_of_tangents (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := by
sorry

end min_product_of_tangents_l895_89569


namespace parallel_line_equation_l895_89561

-- Define a line by its slope and y-intercept
def Line (m b : ℝ) := {(x, y) : ℝ × ℝ | y = m * x + b}

-- Define parallel lines
def Parallel (l₁ l₂ : ℝ × ℝ → Prop) :=
  ∃ m b₁ b₂, l₁ = Line m b₁ ∧ l₂ = Line m b₂

theorem parallel_line_equation :
  let l₁ := Line (-4) 1  -- y = -4x + 1
  let l₂ := {(x, y) : ℝ × ℝ | 4 * x + y - 3 = 0}  -- 4x + y - 3 = 0
  Parallel l₁ l₂ ∧ (0, 3) ∈ l₂ := by sorry

end parallel_line_equation_l895_89561


namespace restaurant_problem_l895_89500

theorem restaurant_problem (people : ℕ) 
  (h1 : 7 * 10 + (88 / people + 7) = 88) : people = 8 := by
  sorry

end restaurant_problem_l895_89500


namespace complex_cube_sum_ratio_l895_89532

theorem complex_cube_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_squared_diff : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 13 :=
by sorry

end complex_cube_sum_ratio_l895_89532


namespace zoo_ticket_price_l895_89574

def ticket_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (service_fee : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  price_after_discount2 + service_fee

theorem zoo_ticket_price :
  ticket_price 15 0.4 0.1 2 = 10.1 := by
  sorry

end zoo_ticket_price_l895_89574


namespace money_left_after_taxes_l895_89593

def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18

theorem money_left_after_taxes : 
  annual_income * (1 - tax_rate) = 49200 := by
  sorry

end money_left_after_taxes_l895_89593


namespace complex_quadratic_equation_solution_l895_89503

theorem complex_quadratic_equation_solution :
  ∃ (z₁ z₂ : ℂ), 
    (z₁ = 1 + Real.sqrt 3 - (Real.sqrt 3 / 2) * Complex.I) ∧
    (z₂ = 1 - Real.sqrt 3 + (Real.sqrt 3 / 2) * Complex.I) ∧
    (∀ z : ℂ, 3 * z^2 - 2 * z = 7 - 3 * Complex.I ↔ z = z₁ ∨ z = z₂) :=
by sorry

end complex_quadratic_equation_solution_l895_89503


namespace modular_sum_equivalence_l895_89592

theorem modular_sum_equivalence : ∃ (x y z : ℤ), 
  (5 * x) % 29 = 1 ∧ 
  (5 * y) % 29 = 1 ∧ 
  (7 * z) % 29 = 1 ∧ 
  (x + y + z) % 29 = 13 := by
  sorry

end modular_sum_equivalence_l895_89592


namespace restaurant_bill_total_l895_89582

theorem restaurant_bill_total (number_of_people : ℕ) (individual_payment : ℕ) : 
  number_of_people = 3 → individual_payment = 45 → number_of_people * individual_payment = 135 := by
  sorry

end restaurant_bill_total_l895_89582


namespace correct_delivery_probability_l895_89556

def number_of_packages : ℕ := 5
def number_of_houses : ℕ := 5

theorem correct_delivery_probability :
  let total_arrangements := number_of_packages.factorial
  let correct_three_arrangements := (number_of_packages.choose 3) * 1 * 1
  (correct_three_arrangements : ℚ) / total_arrangements = 1 / 12 := by
  sorry

end correct_delivery_probability_l895_89556


namespace recess_time_calculation_l895_89554

/-- Calculates the total recess time based on the number of each grade received -/
def total_recess_time (normal_recess : ℕ) (extra_a : ℕ) (extra_b : ℕ) (extra_c : ℕ) (minus_d : ℕ) 
  (num_a : ℕ) (num_b : ℕ) (num_c : ℕ) (num_d : ℕ) : ℕ :=
  normal_recess + extra_a * num_a + extra_b * num_b + extra_c * num_c - minus_d * num_d

theorem recess_time_calculation : 
  total_recess_time 20 3 2 1 1 10 12 14 5 = 83 := by
  sorry

end recess_time_calculation_l895_89554


namespace problem_statement_l895_89515

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 7)
  (h_b_a : b + 1 / a = 11) :
  c + 1 / b = 5 / 19 := by
sorry

end problem_statement_l895_89515


namespace min_value_expression_l895_89588

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 3) (h_rel : x = 2 * y) :
  ∃ (min : ℝ), min = 4/3 ∧ ∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 3 → x = 2 * y →
    (x + y) / (x * y * z) ≥ min :=
by sorry

end min_value_expression_l895_89588


namespace ginger_cakes_l895_89521

/-- The number of cakes Ginger bakes in 10 years --/
def cakes_in_ten_years : ℕ :=
  let children := 2
  let children_holidays := 4
  let husband_holidays := 6
  let parents := 2
  let years := 10
  let cakes_per_year := children * children_holidays + husband_holidays + parents
  cakes_per_year * years

theorem ginger_cakes : cakes_in_ten_years = 160 := by
  sorry

end ginger_cakes_l895_89521


namespace football_count_proof_l895_89505

/-- The cost of one football -/
def football_cost : ℝ := 35

/-- The cost of one soccer ball -/
def soccer_cost : ℝ := 50

/-- The number of footballs in the first set -/
def num_footballs : ℕ := 3

/-- The total cost of the first set -/
def first_set_cost : ℝ := 155

/-- The total cost of the second set -/
def second_set_cost : ℝ := 220

theorem football_count_proof :
  (football_cost * num_footballs + soccer_cost = first_set_cost) ∧
  (2 * football_cost + 3 * soccer_cost = second_set_cost) :=
by sorry

end football_count_proof_l895_89505


namespace gangster_undetected_speed_l895_89506

/-- Represents the speed of a moving object -/
structure Speed :=
  (value : ℝ)

/-- Represents the distance between two points -/
structure Distance :=
  (value : ℝ)

/-- Represents a moving police officer -/
structure PoliceOfficer :=
  (speed : Speed)
  (spacing : Distance)

/-- Represents a moving gangster -/
structure Gangster :=
  (speed : Speed)

/-- Determines if a gangster is undetected by police officers -/
def is_undetected (g : Gangster) (p : PoliceOfficer) : Prop :=
  (g.speed.value = 2 * p.speed.value) ∨ (g.speed.value = p.speed.value / 2)

/-- Theorem stating the conditions for a gangster to remain undetected -/
theorem gangster_undetected_speed (v : ℝ) (a : ℝ) :
  ∀ (g : Gangster) (p : PoliceOfficer),
  p.speed.value = v →
  p.spacing.value = 9 * a →
  is_undetected g p :=
sorry

end gangster_undetected_speed_l895_89506


namespace inequality_proof_l895_89584

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 0) :
  (4 * (x^2 * y^2 + x * y^3 + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y + y :=
by sorry

#check inequality_proof

end inequality_proof_l895_89584


namespace pizza_slices_l895_89589

theorem pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) (h1 : num_pizzas = 7) (h2 : slices_per_pizza = 2) :
  num_pizzas * slices_per_pizza = 14 := by
  sorry

end pizza_slices_l895_89589


namespace quadratic_solution_sum_l895_89576

theorem quadratic_solution_sum (m n : ℝ) (h1 : m ≠ 0) 
  (h2 : m * 1^2 + n * 1 - 2022 = 0) : m + n + 1 = 2023 := by
  sorry

end quadratic_solution_sum_l895_89576


namespace straight_line_properties_l895_89540

-- Define a straight line in a Cartesian coordinate system
structure StraightLine where
  -- We don't define the line using a specific equation to keep it general
  slope_angle : Real
  has_defined_slope : Bool

-- Theorem statement
theorem straight_line_properties (l : StraightLine) : 
  (0 ≤ l.slope_angle ∧ l.slope_angle < π) ∧ 
  (l.has_defined_slope = false → l.slope_angle = π/2) :=
by sorry

end straight_line_properties_l895_89540


namespace square_sum_equals_two_l895_89525

theorem square_sum_equals_two (a b : ℝ) 
  (h1 : (a + b)^2 = 4) 
  (h2 : a * b = 1) : 
  a^2 + b^2 = 2 := by
sorry

end square_sum_equals_two_l895_89525


namespace cat_care_cost_is_40_l895_89585

/-- The cost to care for a cat at Mr. Sean's veterinary clinic -/
def cat_care_cost : ℕ → Prop
| cost => ∃ (dog_cost : ℕ),
  dog_cost = 60 ∧
  20 * dog_cost + 60 * cost = 3600

/-- Theorem: The cost to care for a cat at Mr. Sean's clinic is $40 -/
theorem cat_care_cost_is_40 : cat_care_cost 40 := by
  sorry

end cat_care_cost_is_40_l895_89585


namespace necessary_not_sufficient_condition_l895_89580

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, |x - 1| < 2 → -3 < x ∧ x < 3) ∧
  (∃ x : ℝ, -3 < x ∧ x < 3 ∧ ¬(|x - 1| < 2)) := by
  sorry

end necessary_not_sufficient_condition_l895_89580


namespace current_average_age_l895_89533

-- Define the number of people in the initial group
def initial_group : ℕ := 6

-- Define the average age of the initial group after two years
def future_average_age : ℕ := 43

-- Define the age of the new person joining the group
def new_person_age : ℕ := 69

-- Define the total number of people after the new person joins
def total_people : ℕ := initial_group + 1

-- Theorem to prove
theorem current_average_age :
  (initial_group * future_average_age - initial_group * 2 + new_person_age) / total_people = 45 :=
by sorry

end current_average_age_l895_89533


namespace apple_cost_price_l895_89587

theorem apple_cost_price (selling_price : ℝ) (loss_fraction : ℝ) (cost_price : ℝ) : 
  selling_price = 19 →
  loss_fraction = 1/6 →
  selling_price = cost_price - loss_fraction * cost_price →
  cost_price = 22.8 := by
sorry

end apple_cost_price_l895_89587


namespace pet_store_black_cats_l895_89545

/-- Given a pet store with white, black, and gray cats, prove the number of black cats. -/
theorem pet_store_black_cats 
  (total_cats : ℕ) 
  (white_cats : ℕ) 
  (gray_cats : ℕ) 
  (h_total : total_cats = 15) 
  (h_white : white_cats = 2) 
  (h_gray : gray_cats = 3) :
  total_cats - white_cats - gray_cats = 10 :=
by
  sorry

#check pet_store_black_cats

end pet_store_black_cats_l895_89545


namespace largest_multiple_of_15_under_500_l895_89539

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by
  sorry

end largest_multiple_of_15_under_500_l895_89539


namespace cristine_lemons_left_l895_89502

def dozen : ℕ := 12

def lemons_given_to_neighbor (total : ℕ) : ℕ := total / 4

def lemons_exchanged_for_oranges : ℕ := 2

theorem cristine_lemons_left (initial_lemons : ℕ) 
  (h1 : initial_lemons = dozen) 
  (h2 : lemons_given_to_neighbor initial_lemons = initial_lemons / 4) 
  (h3 : lemons_exchanged_for_oranges = 2) : 
  initial_lemons - lemons_given_to_neighbor initial_lemons - lemons_exchanged_for_oranges = 7 := by
  sorry

end cristine_lemons_left_l895_89502


namespace alcohol_concentration_l895_89595

/-- Prove that the concentration of alcohol in the final mixture is 30% --/
theorem alcohol_concentration (vessel1_capacity : ℝ) (vessel1_alcohol_percent : ℝ)
  (vessel2_capacity : ℝ) (vessel2_alcohol_percent : ℝ)
  (total_liquid : ℝ) (final_vessel_capacity : ℝ) :
  vessel1_capacity = 2 →
  vessel1_alcohol_percent = 30 →
  vessel2_capacity = 6 →
  vessel2_alcohol_percent = 40 →
  total_liquid = 8 →
  final_vessel_capacity = 10 →
  let total_alcohol := (vessel1_capacity * vessel1_alcohol_percent / 100) +
                       (vessel2_capacity * vessel2_alcohol_percent / 100)
  (total_alcohol / final_vessel_capacity) * 100 = 30 := by
  sorry

#check alcohol_concentration

end alcohol_concentration_l895_89595


namespace least_three_digit_multiple_of_eight_l895_89516

theorem least_three_digit_multiple_of_eight : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  n % 8 = 0 ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ m % 8 = 0 → n ≤ m) ∧ 
  n = 104 :=
sorry

end least_three_digit_multiple_of_eight_l895_89516


namespace sixth_term_of_geometric_sequence_l895_89570

theorem sixth_term_of_geometric_sequence (a₁ a₉ : ℝ) (h₁ : a₁ = 12) (h₂ : a₉ = 31104) :
  let r := (a₉ / a₁) ^ (1 / 8)
  let a₆ := a₁ * r ^ 5
  a₆ = 93312 := by
sorry

end sixth_term_of_geometric_sequence_l895_89570


namespace cos_equality_angle_l895_89546

theorem cos_equality_angle (n : ℤ) : 0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end cos_equality_angle_l895_89546


namespace range_of_m_l895_89527

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 4

-- Define proposition P
def P (m : ℝ) : Prop := ∀ x₁ x₂, 2 ≤ x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂

-- Define proposition Q
def Q (m : ℝ) : Prop := ∀ x, 4*x^2 + 4*(m-2)*x + 1 > 0

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → m ≤ 1 ∨ (2 < m ∧ m < 3) := by
  sorry

end range_of_m_l895_89527


namespace probability_two_teachers_in_A_proof_l895_89544

/-- The probability of exactly two out of three teachers being assigned to place A -/
def probability_two_teachers_in_A : ℚ := 3/8

/-- The number of teachers -/
def num_teachers : ℕ := 3

/-- The number of places -/
def num_places : ℕ := 2

theorem probability_two_teachers_in_A_proof :
  probability_two_teachers_in_A = 
    (Nat.choose num_teachers 2 : ℚ) / (num_places ^ num_teachers : ℚ) :=
by sorry

end probability_two_teachers_in_A_proof_l895_89544


namespace afternoon_evening_difference_is_24_l895_89596

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 33

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 34

/-- The number of campers who went rowing in the evening -/
def evening_campers : ℕ := 10

/-- The difference between the number of campers rowing in the afternoon and evening -/
def afternoon_evening_difference : ℕ := afternoon_campers - evening_campers

theorem afternoon_evening_difference_is_24 : 
  afternoon_evening_difference = 24 := by sorry

end afternoon_evening_difference_is_24_l895_89596


namespace lines_parallel_l895_89571

/-- The value of k that makes the given lines parallel -/
def k : ℚ := 16/5

/-- The first line's direction vector -/
def v1 : Fin 2 → ℚ := ![5, -8]

/-- The second line's direction vector -/
def v2 : Fin 2 → ℚ := ![-2, k]

/-- Theorem stating that k makes the lines parallel -/
theorem lines_parallel : ∃ (c : ℚ), v1 = c • v2 := by sorry

end lines_parallel_l895_89571


namespace binomial_coefficient_identity_a_binomial_coefficient_identity_b_binomial_coefficient_identity_c_binomial_coefficient_identity_d_binomial_coefficient_identity_e_l895_89598

-- Part (a)
theorem binomial_coefficient_identity_a (r m k : ℕ) (h1 : k ≤ m) (h2 : m ≤ r) :
  (r.choose m) * (m.choose k) = (r.choose k) * ((r - k).choose (m - k)) := by
  sorry

-- Part (b)
theorem binomial_coefficient_identity_b (n m : ℕ) :
  (n + 1).choose (m + 1) = n.choose m + n.choose (m + 1) := by
  sorry

-- Part (c)
theorem binomial_coefficient_identity_c (n : ℕ) :
  (2 * n).choose n = (Finset.range (n + 1)).sum (λ k => (n.choose k) ^ 2) := by
  sorry

-- Part (d)
theorem binomial_coefficient_identity_d (m n k : ℕ) (h : k ≤ n) :
  (m + n).choose k = (Finset.range (k + 1)).sum (λ p => (n.choose p) * (m.choose (k - p))) := by
  sorry

-- Part (e)
theorem binomial_coefficient_identity_e (n k : ℕ) (h : k ≤ n) :
  n.choose k = (Finset.range (n - k + 1)).sum (λ i => (k + i - 1).choose (k - 1)) := by
  sorry

end binomial_coefficient_identity_a_binomial_coefficient_identity_b_binomial_coefficient_identity_c_binomial_coefficient_identity_d_binomial_coefficient_identity_e_l895_89598


namespace domain_of_g_l895_89572

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-12) 3

-- Define the function g in terms of f
def g (x : ℝ) : ℝ := f (3 * x)

-- State the theorem
theorem domain_of_g : 
  {x : ℝ | g x ∈ Set.range f} = Set.Icc (-4) 1 := by sorry

end domain_of_g_l895_89572


namespace treehouse_planks_l895_89583

theorem treehouse_planks :
  ∀ (T : ℕ),
  (T / 4 : ℚ) + (T / 2 : ℚ) + 20 + 30 = T →
  T = 200 := by
sorry

end treehouse_planks_l895_89583


namespace fib_120_mod_5_l895_89563

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define the property that the Fibonacci sequence modulo 5 repeats every 20 terms
axiom fib_mod_5_period_20 : ∀ n : ℕ, fib n % 5 = fib (n % 20) % 5

-- Theorem statement
theorem fib_120_mod_5 : fib 120 % 5 = 0 := by
  sorry

end fib_120_mod_5_l895_89563


namespace erased_number_theorem_l895_89536

theorem erased_number_theorem (n : ℕ) (h1 : n = 20) :
  ∀ x ∈ Finset.range n,
    (∃ y ∈ Finset.range n \ {x}, (n * (n + 1) / 2 - x : ℚ) / (n - 1) = y) ↔ 
    x = 1 ∨ x = n :=
by sorry

end erased_number_theorem_l895_89536


namespace mikes_games_l895_89559

theorem mikes_games (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 101 →
  spent_amount = 47 →
  game_cost = 6 →
  (initial_amount - spent_amount) / game_cost = 9 := by
sorry

end mikes_games_l895_89559


namespace emily_purchase_cost_l895_89537

/-- Calculate the total cost of Emily's purchase including discount, tax, and installation fee -/
theorem emily_purchase_cost :
  let curtain_price : ℚ := 30
  let curtain_quantity : ℕ := 2
  let print_price : ℚ := 15
  let print_quantity : ℕ := 9
  let discount_rate : ℚ := 0.1
  let tax_rate : ℚ := 0.08
  let installation_fee : ℚ := 50

  let subtotal : ℚ := curtain_price * curtain_quantity + print_price * print_quantity
  let discounted_total : ℚ := subtotal * (1 - discount_rate)
  let taxed_total : ℚ := discounted_total * (1 + tax_rate)
  let total_cost : ℚ := taxed_total + installation_fee

  total_cost = 239.54 := by sorry

end emily_purchase_cost_l895_89537


namespace polygon_with_1800_degree_sum_is_dodecagon_l895_89562

theorem polygon_with_1800_degree_sum_is_dodecagon :
  ∀ n : ℕ, 
  n ≥ 3 →
  (n - 2) * 180 = 1800 →
  n = 12 :=
by
  sorry

end polygon_with_1800_degree_sum_is_dodecagon_l895_89562


namespace thursday_is_only_valid_start_day_l895_89504

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def DayOfWeek.next (d : DayOfWeek) : DayOfWeek :=
  match d with
  | .Sunday => .Monday
  | .Monday => .Tuesday
  | .Tuesday => .Wednesday
  | .Wednesday => .Thursday
  | .Thursday => .Friday
  | .Friday => .Saturday
  | .Saturday => .Sunday

def DayOfWeek.addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => (d.addDays n).next

def isOpen (d : DayOfWeek) : Bool :=
  match d with
  | .Sunday => false
  | .Monday => false
  | _ => true

def validRedemptionSchedule (startDay : DayOfWeek) : Bool :=
  let schedule := List.range 8 |>.map (fun i => startDay.addDays (i * 7))
  schedule.all isOpen

theorem thursday_is_only_valid_start_day :
  ∀ (d : DayOfWeek), validRedemptionSchedule d ↔ d = DayOfWeek.Thursday :=
sorry

#check thursday_is_only_valid_start_day

end thursday_is_only_valid_start_day_l895_89504


namespace total_pupils_across_schools_l895_89573

/-- The total number of pupils across three schools -/
def total_pupils (school_a_girls school_a_boys school_b_girls school_b_boys school_c_girls school_c_boys : ℕ) : ℕ :=
  school_a_girls + school_a_boys + school_b_girls + school_b_boys + school_c_girls + school_c_boys

/-- Theorem stating that the total number of pupils across the three schools is 3120 -/
theorem total_pupils_across_schools :
  total_pupils 542 387 713 489 628 361 = 3120 := by
  sorry

end total_pupils_across_schools_l895_89573


namespace modified_cube_surface_area_l895_89512

/-- Calculate the entire surface area of a modified cube -/
theorem modified_cube_surface_area :
  let cube_edge : ℝ := 5
  let large_hole_side : ℝ := 2
  let small_hole_side : ℝ := 0.5
  let original_surface_area : ℝ := 6 * cube_edge^2
  let large_holes_area : ℝ := 6 * large_hole_side^2
  let exposed_inner_area : ℝ := 6 * 4 * large_hole_side^2
  let small_holes_area : ℝ := 6 * 4 * small_hole_side^2
  original_surface_area - large_holes_area + exposed_inner_area - small_holes_area = 228 := by
  sorry

end modified_cube_surface_area_l895_89512


namespace sqrt_fourteen_times_sqrt_seven_minus_sqrt_two_l895_89549

theorem sqrt_fourteen_times_sqrt_seven_minus_sqrt_two : 
  Real.sqrt 14 * Real.sqrt 7 - Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_fourteen_times_sqrt_seven_minus_sqrt_two_l895_89549


namespace curve_intersects_median_l895_89507

/-- Given non-collinear points A, B, C in the complex plane corresponding to 
    z₀ = ai, z₁ = 1/2 + bi, z₂ = 1 + ci respectively, where a, b, c are real numbers,
    prove that the curve z = z₀cos⁴t + 2z₁cos²tsin²t + z₂sin⁴t intersects the median 
    of triangle ABC parallel to AC at exactly one point (1/2, (a+c+2b)/4). -/
theorem curve_intersects_median (a b c : ℝ) 
  (h_non_collinear : a + c - 2*b ≠ 0) : 
  ∃! p : ℂ, 
    (∃ t : ℝ, p = Complex.I * a * (Real.cos t)^4 + 
      2 * (1/2 + Complex.I * b) * (Real.cos t)^2 * (Real.sin t)^2 + 
      (1 + Complex.I * c) * (Real.sin t)^4) ∧ 
    p.im = (c - a) * p.re + (3*a + 2*b - c)/4 ∧ 
    p = Complex.mk (1/2) ((a + c + 2*b)/4) := by 
  sorry

end curve_intersects_median_l895_89507


namespace chess_group_players_l895_89541

theorem chess_group_players (n : ℕ) : n * (n - 1) / 2 = 1225 → n = 50 := by
  sorry

end chess_group_players_l895_89541


namespace unique_congruence_solution_l895_89575

theorem unique_congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ 10389 [ZMOD 12] ∧ n = 9 := by
  sorry

end unique_congruence_solution_l895_89575


namespace sum_is_non_horizontal_line_l895_89551

/-- A parabola is defined by its coefficients a, b, and c. -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Function f is the original parabola translated 3 units to the right. -/
def f (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - 3)^2 + p.b * (x - 3) + p.c

/-- Function g is the reflected parabola translated 3 units to the left. -/
def g (p : Parabola) (x : ℝ) : ℝ :=
  -p.a * (x + 3)^2 - p.b * (x + 3) - p.c

/-- The sum of f and g is a non-horizontal line. -/
theorem sum_is_non_horizontal_line (p : Parabola) :
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x, f p x + g p x = m * x + k := by
  sorry

end sum_is_non_horizontal_line_l895_89551


namespace complement_M_equals_closed_interval_l895_89579

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

theorem complement_M_equals_closed_interval :
  (Set.univ \ M) = Set.Icc 0 2 := by sorry

end complement_M_equals_closed_interval_l895_89579


namespace mushroom_collectors_problem_l895_89526

theorem mushroom_collectors_problem :
  ∃ (n m : ℕ),
    n > 0 ∧ m > 0 ∧
    6 + 13 * (n - 1) = 5 + 10 * (m - 1) ∧
    100 < 6 + 13 * (n - 1) ∧
    6 + 13 * (n - 1) < 200 ∧
    n = 14 ∧ m = 18 :=
by sorry

end mushroom_collectors_problem_l895_89526


namespace unread_books_l895_89565

theorem unread_books (total_books read_books : ℕ) : 
  total_books = 20 → read_books = 15 → total_books - read_books = 5 := by
  sorry

end unread_books_l895_89565


namespace complex_magnitude_l895_89586

theorem complex_magnitude (z : ℂ) (h : z * (1 + 2*Complex.I) + Complex.I = 0) : 
  Complex.abs z = Real.sqrt 5 / 5 := by
sorry

end complex_magnitude_l895_89586


namespace complex_sum_reciprocals_l895_89520

theorem complex_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
  sorry

end complex_sum_reciprocals_l895_89520


namespace not_always_parallel_lines_l895_89514

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem not_always_parallel_lines 
  (l m : Line) (α : Plane) 
  (h1 : parallel_plane_line α l) 
  (h2 : subset m α) : 
  ¬ (∀ l m α, parallel_plane_line α l → subset m α → parallel l m) :=
sorry

end not_always_parallel_lines_l895_89514


namespace probability_theorem_l895_89534

/-- Represents the total number of products -/
def total_products : ℕ := 7

/-- Represents the number of genuine products -/
def genuine_products : ℕ := 4

/-- Represents the number of defective products -/
def defective_products : ℕ := 3

/-- The probability of selecting a genuine product on the second draw,
    given that a defective product was selected on the first draw -/
def probability_genuine_second_given_defective_first : ℚ := 2/3

/-- Theorem stating the probability of selecting a genuine product on the second draw,
    given that a defective product was selected on the first draw -/
theorem probability_theorem :
  probability_genuine_second_given_defective_first = 2/3 :=
by sorry

end probability_theorem_l895_89534


namespace c_months_equals_six_l895_89548

def total_cost : ℚ := 435
def a_horses : ℕ := 12
def a_months : ℕ := 8
def b_horses : ℕ := 16
def b_months : ℕ := 9
def c_horses : ℕ := 18
def b_payment : ℚ := 180

theorem c_months_equals_six :
  ∃ (x : ℕ), 
    (b_payment / total_cost) * (a_horses * a_months + b_horses * b_months + c_horses * x) = 
    b_horses * b_months ∧ x = 6 := by
  sorry

end c_months_equals_six_l895_89548


namespace decagon_triangle_probability_l895_89568

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of possible triangles formed from the decagon -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with at least one side being a side of the decagon -/
def favorable_triangles : ℕ := 70

/-- The probability of forming a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem decagon_triangle_probability :
  probability = 7 / 12 := by sorry

end decagon_triangle_probability_l895_89568


namespace candle_ratio_l895_89530

/-- Proves the ratio of candles Alyssa used to total candles -/
theorem candle_ratio :
  ∀ (total candles_used_by_alyssa : ℕ) (chelsea_usage_percent : ℚ),
  total = 40 →
  chelsea_usage_percent = 70 / 100 →
  candles_used_by_alyssa + 
    (chelsea_usage_percent * (total - candles_used_by_alyssa)).floor + 6 = total →
  candles_used_by_alyssa * 2 = total := by
  sorry

#check candle_ratio

end candle_ratio_l895_89530


namespace scientific_notation_of_0_0813_l895_89518

theorem scientific_notation_of_0_0813 :
  ∃ (a : ℝ) (n : ℤ), 0.0813 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -2 :=
by sorry

end scientific_notation_of_0_0813_l895_89518


namespace f_decreasing_on_interval_l895_89566

def f (x : ℝ) := x^2 - 4*x + 3

theorem f_decreasing_on_interval :
  ∀ x y : ℝ, x < y → y ≤ 2 → f x > f y :=
by sorry

end f_decreasing_on_interval_l895_89566


namespace parabola_fixed_point_l895_89558

theorem parabola_fixed_point :
  ∀ t : ℝ, 3 * (2 : ℝ)^2 + t * 2 - 2 * t = 12 := by
  sorry

end parabola_fixed_point_l895_89558


namespace trig_simplification_l895_89591

theorem trig_simplification (x y : ℝ) :
  (Real.cos x)^2 + (Real.sin x)^2 + (Real.cos (x + y))^2 - 
  2 * (Real.cos x) * (Real.cos y) * (Real.cos (x + y)) - 
  (Real.sin x) * (Real.sin y) = (Real.sin (x - y))^2 := by
  sorry

end trig_simplification_l895_89591


namespace trefoils_per_case_l895_89528

theorem trefoils_per_case (total_boxes : ℕ) (total_cases : ℕ) (boxes_per_case : ℕ) : 
  total_boxes = 54 → total_cases = 9 → boxes_per_case = total_boxes / total_cases → boxes_per_case = 6 := by
  sorry

end trefoils_per_case_l895_89528


namespace adjacent_vertices_probability_l895_89557

/-- A decagon is a polygon with 10 vertices -/
def Decagon : ℕ := 10

/-- The number of vertices adjacent to any vertex in a decagon -/
def AdjacentVertices : ℕ := 2

/-- The probability of selecting two adjacent vertices in a decagon -/
def ProbAdjacentVertices : ℚ := 2 / 9

theorem adjacent_vertices_probability (d : ℕ) (av : ℕ) (p : ℚ) 
  (h1 : d = Decagon) 
  (h2 : av = AdjacentVertices) 
  (h3 : p = ProbAdjacentVertices) : 
  p = av / (d - 1) := by
  sorry

#check adjacent_vertices_probability

end adjacent_vertices_probability_l895_89557


namespace additional_days_to_double_earnings_l895_89513

/-- Represents the number of days John has worked so far -/
def days_worked : ℕ := 10

/-- Represents the amount of money John has earned so far in dollars -/
def current_earnings : ℕ := 250

/-- Calculates John's daily rate in dollars -/
def daily_rate : ℚ := current_earnings / days_worked

/-- Calculates the total amount John needs to earn to double his current earnings -/
def target_earnings : ℕ := 2 * current_earnings

/-- Calculates the additional amount John needs to earn -/
def additional_earnings : ℕ := target_earnings - current_earnings

/-- Theorem stating the number of additional days John needs to work -/
theorem additional_days_to_double_earnings : 
  (additional_earnings : ℚ) / daily_rate = 10 := by sorry

end additional_days_to_double_earnings_l895_89513


namespace chess_match_probability_l895_89510

theorem chess_match_probability (p_win p_draw : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_draw = 0.2) : 
  p_win + p_draw = 0.6 := by
  sorry

end chess_match_probability_l895_89510


namespace quadratic_sum_l895_89529

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (3 * x^2 + 9 * x - 81 = a * (x + b)^2 + c) ∧ (a + b + c = -83.25) := by
  sorry

end quadratic_sum_l895_89529


namespace quadratic_always_nonnegative_l895_89578

theorem quadratic_always_nonnegative (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) := by
  sorry

end quadratic_always_nonnegative_l895_89578


namespace police_officers_on_duty_l895_89519

theorem police_officers_on_duty 
  (total_female_officers : ℕ) 
  (female_duty_percentage : ℚ)
  (female_duty_ratio : ℚ) :
  total_female_officers = 400 →
  female_duty_percentage = 19 / 100 →
  female_duty_ratio = 1 / 2 →
  ∃ (officers_on_duty : ℕ), 
    officers_on_duty = 152 ∧ 
    (female_duty_percentage * total_female_officers : ℚ) = (female_duty_ratio * officers_on_duty : ℚ) := by
  sorry

end police_officers_on_duty_l895_89519


namespace balls_remaining_l895_89599

def initial_balls : ℕ := 10
def removed_balls : ℕ := 3

theorem balls_remaining : initial_balls - removed_balls = 7 := by
  sorry

end balls_remaining_l895_89599


namespace equation_solution_l895_89594

theorem equation_solution : 
  ∀ x : ℝ, (x - 1) * (x + 1) = x - 1 ↔ x = 0 ∨ x = 1 :=
by sorry

end equation_solution_l895_89594


namespace right_square_pyramid_base_neq_lateral_l895_89535

/-- A right square pyramid -/
structure RightSquarePyramid where
  baseEdge : ℝ
  lateralEdge : ℝ
  height : ℝ

/-- Theorem: In a right square pyramid, the base edge length cannot be equal to the lateral edge length -/
theorem right_square_pyramid_base_neq_lateral (p : RightSquarePyramid) : 
  p.baseEdge ≠ p.lateralEdge :=
sorry

end right_square_pyramid_base_neq_lateral_l895_89535


namespace equation_solution_l895_89524

theorem equation_solution : ∃! x : ℝ, 2 * x - 3 = 5 ∧ x = 4 := by
  sorry

end equation_solution_l895_89524


namespace sqrt_18_times_sqrt_32_l895_89590

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end sqrt_18_times_sqrt_32_l895_89590


namespace bottle_cap_cost_l895_89555

theorem bottle_cap_cost (cost_per_cap : ℝ) (num_caps : ℕ) : 
  cost_per_cap = 5 → num_caps = 5 → cost_per_cap * (num_caps : ℝ) = 25 := by
  sorry

end bottle_cap_cost_l895_89555


namespace detergent_needed_l895_89501

/-- The amount of detergent used per pound of clothes -/
def detergent_per_pound : ℝ := 2

/-- The amount of clothes to be washed, in pounds -/
def clothes_amount : ℝ := 9

/-- Theorem stating the amount of detergent needed for a given amount of clothes -/
theorem detergent_needed (detergent_per_pound : ℝ) (clothes_amount : ℝ) :
  detergent_per_pound * clothes_amount = 18 :=
by
  sorry

end detergent_needed_l895_89501


namespace interest_rate_is_one_percent_l895_89581

/-- Calculates the interest rate given principal, time, and total simple interest -/
def calculate_interest_rate (principal : ℚ) (time : ℚ) (total_interest : ℚ) : ℚ :=
  (total_interest * 100) / (principal * time)

/-- Theorem stating that given the problem conditions, the interest rate is 1% -/
theorem interest_rate_is_one_percent 
  (principal : ℚ) 
  (time : ℚ) 
  (total_interest : ℚ) 
  (h1 : principal = 44625)
  (h2 : time = 9)
  (h3 : total_interest = 4016.25) :
  calculate_interest_rate principal time total_interest = 1 := by
  sorry

#eval calculate_interest_rate 44625 9 4016.25

end interest_rate_is_one_percent_l895_89581


namespace nonagon_diagonals_l895_89517

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def is_nonagon (n : ℕ) : Prop := n = 9

theorem nonagon_diagonals :
  ∀ n : ℕ, is_nonagon n → num_diagonals n = 27 := by
  sorry

end nonagon_diagonals_l895_89517
