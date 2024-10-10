import Mathlib

namespace squared_plus_greater_than_self_l2972_297224

-- Define a monotonically increasing function on R
def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem squared_plus_greater_than_self
  (f : ℝ → ℝ) (h_monotone : monotone_increasing f) (t : ℝ) (h_t : t ≠ 0) :
  f (t^2 + t) > f t :=
sorry

end squared_plus_greater_than_self_l2972_297224


namespace turtleneck_discount_l2972_297274

theorem turtleneck_discount (C : ℝ) (C_pos : C > 0) : 
  let initial_price := 1.2 * C
  let marked_up_price := 1.25 * initial_price
  let final_price := (1 - 0.08) * marked_up_price
  final_price = 1.38 * C := by sorry

end turtleneck_discount_l2972_297274


namespace max_value_expression_l2972_297219

theorem max_value_expression (x a : ℝ) (hx : x > 0) (ha : a > 0) :
  (x^2 + a - Real.sqrt (x^4 + a^2)) / x ≤ 2 * a / (2 * Real.sqrt a + Real.sqrt (2 * a)) ∧
  (x^2 + a - Real.sqrt (x^4 + a^2)) / x = 2 * a / (2 * Real.sqrt a + Real.sqrt (2 * a)) ↔ x = Real.sqrt a :=
sorry

end max_value_expression_l2972_297219


namespace seven_pow_2015_ends_with_43_l2972_297276

/-- The last two digits of a natural number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- 7^2015 ends with 43 -/
theorem seven_pow_2015_ends_with_43 : lastTwoDigits (7^2015) = 43 := by
  sorry

#check seven_pow_2015_ends_with_43

end seven_pow_2015_ends_with_43_l2972_297276


namespace units_digit_G_100_l2972_297255

-- Define G_n
def G (n : ℕ) : ℕ := 2^(5^n) + 1

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_G_100 : units_digit (G 100) = 3 := by
  sorry

end units_digit_G_100_l2972_297255


namespace max_value_abc_l2972_297227

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c + a * b)) / ((a + b)^3 * (b + c)^3) ≤ 1/16 := by
  sorry

end max_value_abc_l2972_297227


namespace polar_to_cartesian_intersecting_lines_l2972_297239

/-- The polar coordinate equation ρ(cos²θ - sin²θ) = 0 represents two intersecting lines -/
theorem polar_to_cartesian_intersecting_lines :
  ∃ (x y : ℝ → ℝ), 
    (∀ θ : ℝ, x θ^2 = y θ^2) ∧ 
    (∀ θ : ℝ, x θ = y θ ∨ x θ = -y θ) ∧
    (∀ ρ θ : ℝ, ρ * (Real.cos θ^2 - Real.sin θ^2) = 0 → 
      x θ = ρ * Real.cos θ ∧ y θ = ρ * Real.sin θ) :=
sorry

end polar_to_cartesian_intersecting_lines_l2972_297239


namespace towel_area_decrease_l2972_297205

theorem towel_area_decrease : 
  ∀ (original_length original_width : ℝ),
  original_length > 0 → original_width > 0 →
  let new_length := original_length * 0.8
  let new_width := original_width * 0.9
  let original_area := original_length * original_width
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.28 := by
sorry

end towel_area_decrease_l2972_297205


namespace midpoint_theorem_ap_twice_pb_theorem_l2972_297250

-- Define the line and points
def Line := ℝ → ℝ → Prop
def Point := ℝ × ℝ

-- Define the given point P
def P : Point := (-3, 1)

-- Define the properties of points A and B
def on_x_axis (A : Point) : Prop := A.2 = 0
def on_y_axis (B : Point) : Prop := B.1 = 0

-- Define the property of P being the midpoint of AB
def is_midpoint (P A B : Point) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define the property of AP = 2PB
def ap_twice_pb (P A B : Point) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (2 * (P.1 - B.1), 2 * (P.2 - B.2))

-- Define the equations of the lines
def line_eq1 (x y : ℝ) : Prop := x - 3*y + 6 = 0
def line_eq2 (x y : ℝ) : Prop := x - 6*y + 9 = 0

-- Theorem 1
theorem midpoint_theorem (l : Line) (A B : Point) :
  on_x_axis A → on_y_axis B → is_midpoint P A B →
  (∀ x y, l x y ↔ line_eq1 x y) :=
sorry

-- Theorem 2
theorem ap_twice_pb_theorem (l : Line) (A B : Point) :
  on_x_axis A → on_y_axis B → ap_twice_pb P A B →
  (∀ x y, l x y ↔ line_eq2 x y) :=
sorry

end midpoint_theorem_ap_twice_pb_theorem_l2972_297250


namespace wendy_pastries_left_l2972_297208

/-- The number of pastries Wendy had left after the bake sale -/
def pastries_left (cupcakes cookies sold : ℕ) : ℕ :=
  cupcakes + cookies - sold

/-- Theorem stating that Wendy had 24 pastries left after the bake sale -/
theorem wendy_pastries_left : pastries_left 4 29 9 = 24 := by
  sorry

end wendy_pastries_left_l2972_297208


namespace geometric_sequence_sum_l2972_297214

/-- A geometric sequence with first term 3 and sum of first, third, and fifth terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ 
  (∃ q : ℝ, ∀ n : ℕ, a n = 3 * q ^ (n - 1)) ∧
  a 1 + a 3 + a 5 = 21

/-- The sum of the third, fifth, and seventh terms of the sequence is 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 3 + a 5 + a 7 = 42 := by
sorry

end geometric_sequence_sum_l2972_297214


namespace not_p_equiv_p_and_q_equiv_l2972_297203

-- Define propositions p and q
def p (x : ℝ) := x * (x - 2) ≥ 0
def q (x : ℝ) := |x - 2| < 1

-- Theorem 1: Negation of p is equivalent to 0 < x < 2
theorem not_p_equiv (x : ℝ) : ¬(p x) ↔ 0 < x ∧ x < 2 := by sorry

-- Theorem 2: p and q together are equivalent to 2 ≤ x < 3
theorem p_and_q_equiv (x : ℝ) : p x ∧ q x ↔ 2 ≤ x ∧ x < 3 := by sorry

end not_p_equiv_p_and_q_equiv_l2972_297203


namespace maze_side_length_l2972_297228

/-- Represents a maze on a square grid -/
structure Maze where
  sideLength : ℕ
  wallLength : ℕ

/-- Checks if the maze satisfies the unique path property -/
def hasUniquePaths (m : Maze) : Prop :=
  m.sideLength ^ 2 = 2 * m.sideLength * (m.sideLength - 1) - m.wallLength + 1

theorem maze_side_length (m : Maze) :
  m.wallLength = 400 → hasUniquePaths m → m.sideLength = 21 := by
  sorry

end maze_side_length_l2972_297228


namespace distance_center_to_point_l2972_297204

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y + 9

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 3)

-- Define the point
def point : ℝ × ℝ := (8, 3)

-- Theorem statement
theorem distance_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = 6 :=
sorry

end distance_center_to_point_l2972_297204


namespace effective_average_reduction_l2972_297290

theorem effective_average_reduction (initial_price : ℝ) (reduction_percent : ℝ) : 
  reduction_percent = 36 → 
  ∃ (effective_reduction : ℝ), 
    (1 - effective_reduction / 100)^2 * initial_price = 
    (1 - reduction_percent / 100)^2 * initial_price ∧
    effective_reduction = 20 := by
  sorry

end effective_average_reduction_l2972_297290


namespace output_is_six_l2972_297201

def program_output (a : ℕ) : ℕ :=
  if a < 10 then 2 * a else a * a

theorem output_is_six : program_output 3 = 6 := by
  sorry

end output_is_six_l2972_297201


namespace equation_solution_l2972_297218

theorem equation_solution : ∀ x : ℝ, (9 / x^2 = x / 25) → x = 5 := by
  sorry

end equation_solution_l2972_297218


namespace greatest_number_with_odd_factors_l2972_297296

theorem greatest_number_with_odd_factors : ∃ n : ℕ, 
  n < 200 ∧ 
  (∃ k : ℕ, n = k^2) ∧
  (∀ m : ℕ, m < 200 → (∃ j : ℕ, m = j^2) → m ≤ n) :=
by sorry

end greatest_number_with_odd_factors_l2972_297296


namespace no_divisible_lilac_flowers_l2972_297242

theorem no_divisible_lilac_flowers : ¬∃ (q c : ℕ), 
  (∃ (p₁ p₂ : ℕ), q + c = p₂^2 ∧ 4*q + 5*c = p₁^2) ∧ 
  (∃ (x : ℕ), q = c * x) := by
sorry

end no_divisible_lilac_flowers_l2972_297242


namespace cycle_selling_price_l2972_297245

/-- Calculates the selling price of a cycle given its cost price and gain percent -/
def calculate_selling_price (cost_price : ℚ) (gain_percent : ℚ) : ℚ :=
  cost_price * (1 + gain_percent / 100)

/-- Theorem: The selling price of a cycle bought for 840 with 45.23809523809524% gain is 1220 -/
theorem cycle_selling_price : 
  calculate_selling_price 840 45.23809523809524 = 1220 := by
  sorry

end cycle_selling_price_l2972_297245


namespace min_production_cost_l2972_297269

/-- Raw material requirements for products A and B --/
structure RawMaterial where
  a : ℕ  -- kg of material A required
  b : ℕ  -- kg of material B required

/-- Available raw materials and production constraints --/
structure ProductionConstraints where
  total_units : ℕ        -- Total units to be produced
  available_a : ℕ        -- Available kg of material A
  available_b : ℕ        -- Available kg of material B
  product_a : RawMaterial  -- Raw material requirements for product A
  product_b : RawMaterial  -- Raw material requirements for product B

/-- Cost information for products --/
structure CostInfo where
  cost_a : ℕ  -- Cost per unit of product A
  cost_b : ℕ  -- Cost per unit of product B

/-- Main theorem stating the minimum production cost --/
theorem min_production_cost 
  (constraints : ProductionConstraints)
  (costs : CostInfo)
  (h_constraints : constraints.total_units = 50 ∧ 
                   constraints.available_a = 360 ∧ 
                   constraints.available_b = 290 ∧
                   constraints.product_a = ⟨9, 4⟩ ∧
                   constraints.product_b = ⟨3, 10⟩)
  (h_costs : costs.cost_a = 70 ∧ costs.cost_b = 90) :
  ∃ (x : ℕ), x = 32 ∧ 
    (constraints.total_units - x) = 18 ∧
    costs.cost_a * x + costs.cost_b * (constraints.total_units - x) = 3860 :=
sorry

end min_production_cost_l2972_297269


namespace dimas_age_l2972_297275

theorem dimas_age (dima_age brother_age sister_age : ℕ) : 
  dima_age = 2 * brother_age →
  dima_age = 3 * sister_age →
  (dima_age + brother_age + sister_age) / 3 = 11 →
  dima_age = 18 := by
sorry

end dimas_age_l2972_297275


namespace cards_thrown_away_l2972_297253

theorem cards_thrown_away (cards_per_deck : ℕ) (half_full_decks : ℕ) (full_decks : ℕ) (remaining_cards : ℕ) : 
  cards_per_deck = 52 →
  half_full_decks = 3 →
  full_decks = 3 →
  remaining_cards = 200 →
  (cards_per_deck * full_decks + (cards_per_deck / 2) * half_full_decks) - remaining_cards = 34 :=
by sorry

end cards_thrown_away_l2972_297253


namespace internal_diagonal_intersects_576_cubes_l2972_297235

def rectangular_solid_dimensions : ℕ × ℕ × ℕ := (120, 210, 336)

-- Function to calculate the number of cubes intersected by the diagonal
def intersected_cubes (dims : ℕ × ℕ × ℕ) : ℕ :=
  let (x, y, z) := dims
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

theorem internal_diagonal_intersects_576_cubes :
  intersected_cubes rectangular_solid_dimensions = 576 := by
  sorry

end internal_diagonal_intersects_576_cubes_l2972_297235


namespace line_equation_l2972_297221

/-- Given a line with inclination angle π/3 and y-intercept 2, its equation is √3x - y + 2 = 0 -/
theorem line_equation (x y : ℝ) :
  let angle : ℝ := π / 3
  let y_intercept : ℝ := 2
  (Real.sqrt 3 * x - y + y_intercept = 0) ↔ 
    (y = Real.tan angle * x + y_intercept) :=
by sorry

end line_equation_l2972_297221


namespace largest_n_divisible_by_seven_l2972_297222

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 80000 → 
  (9 * (n - 3)^6 - 3 * n^3 + 21 * n - 33) % 7 = 0 → 
  n ≤ 79993 :=
by sorry

end largest_n_divisible_by_seven_l2972_297222


namespace apps_files_difference_l2972_297226

/-- Given Dave's initial and final numbers of apps and files on his phone, prove that he has 7 more apps than files left. -/
theorem apps_files_difference (initial_apps initial_files final_apps final_files : ℕ) :
  initial_apps = 24 →
  initial_files = 9 →
  final_apps = 12 →
  final_files = 5 →
  final_apps - final_files = 7 := by
  sorry

end apps_files_difference_l2972_297226


namespace range_of_positive_integers_in_list_l2972_297280

def consecutive_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List ℤ) : List ℤ :=
  l.filter (λ x => x > 0)

def range_of_list (l : List ℤ) : ℤ :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_list :
  let d := consecutive_integers (-4) 12
  let positives := positive_integers d
  range_of_list positives = 6 := by
sorry

end range_of_positive_integers_in_list_l2972_297280


namespace rectangle_area_18_l2972_297225

def rectangle_pairs : Set (ℕ × ℕ) :=
  {p | p.1 * p.2 = 18 ∧ p.1 < p.2 ∧ p.1 > 0 ∧ p.2 > 0}

theorem rectangle_area_18 :
  rectangle_pairs = {(1, 18), (2, 9), (3, 6)} := by
  sorry

end rectangle_area_18_l2972_297225


namespace george_oranges_l2972_297273

theorem george_oranges (george_oranges : ℕ) (george_apples : ℕ) (amelia_oranges : ℕ) (amelia_apples : ℕ) : 
  george_apples = amelia_apples + 5 →
  amelia_oranges = george_oranges - 18 →
  amelia_apples = 15 →
  george_oranges + george_apples + amelia_oranges + amelia_apples = 107 →
  george_oranges = 45 := by
sorry

end george_oranges_l2972_297273


namespace min_sum_cotangents_l2972_297288

theorem min_sum_cotangents (A B C : ℝ) (hAcute : 0 < A ∧ A < π/2) 
  (hBcute : 0 < B ∧ B < π/2) (hCcute : 0 < C ∧ C < π/2) 
  (hSum : A + B + C = π) (hSin : 2 * Real.sin A ^ 2 + Real.sin B ^ 2 = 2 * Real.sin C ^ 2) : 
  (∀ A' B' C', A' + B' + C' = π → 2 * Real.sin A' ^ 2 + Real.sin B' ^ 2 = 2 * Real.sin C' ^ 2 →
    1 / Real.tan A + 1 / Real.tan B + 1 / Real.tan C ≤ 
    1 / Real.tan A' + 1 / Real.tan B' + 1 / Real.tan C') ∧
  1 / Real.tan A + 1 / Real.tan B + 1 / Real.tan C = Real.sqrt 13 / 2 := by
  sorry

end min_sum_cotangents_l2972_297288


namespace express_c_in_terms_of_a_and_b_l2972_297212

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 3)
def c : ℝ × ℝ := (4, 1)

theorem express_c_in_terms_of_a_and_b :
  c = 2 • a - b := by sorry

end express_c_in_terms_of_a_and_b_l2972_297212


namespace equation_with_geometric_progression_roots_l2972_297207

theorem equation_with_geometric_progression_roots : ∃ (x₁ x₂ x₃ x₄ : ℝ) (q : ℝ),
  (x₁ ≠ x₂) ∧ (x₁ ≠ x₃) ∧ (x₁ ≠ x₄) ∧ (x₂ ≠ x₃) ∧ (x₂ ≠ x₄) ∧ (x₃ ≠ x₄) ∧
  (q ≠ 1) ∧ (q > 0) ∧
  (x₂ = q * x₁) ∧ (x₃ = q * x₂) ∧ (x₄ = q * x₃) ∧
  (16 * x₁^4 - 170 * x₁^3 + 357 * x₁^2 - 170 * x₁ + 16 = 0) ∧
  (16 * x₂^4 - 170 * x₂^3 + 357 * x₂^2 - 170 * x₂ + 16 = 0) ∧
  (16 * x₃^4 - 170 * x₃^3 + 357 * x₃^2 - 170 * x₃ + 16 = 0) ∧
  (16 * x₄^4 - 170 * x₄^3 + 357 * x₄^2 - 170 * x₄ + 16 = 0) := by
sorry

end equation_with_geometric_progression_roots_l2972_297207


namespace prime_solution_equation_l2972_297249

theorem prime_solution_equation : ∃! (p q : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 :=
by
  -- The proof would go here
  sorry

end prime_solution_equation_l2972_297249


namespace tan_range_problem_l2972_297211

open Real Set

theorem tan_range_problem (m : ℝ) : 
  (∃ x ∈ Icc 0 (π/4), ¬(tan x < m)) ↔ m ∈ Iic 1 :=
sorry

end tan_range_problem_l2972_297211


namespace kiddie_scoop_cost_l2972_297267

/-- The cost of ice cream scoops for the Martin family --/
def ice_cream_cost (kiddie_scoop : ℕ) : Prop :=
  let regular_scoop : ℕ := 4
  let double_scoop : ℕ := 6
  let total_cost : ℕ := 32
  let num_regular : ℕ := 2  -- Mr. and Mrs. Martin
  let num_kiddie : ℕ := 2   -- Their two children
  let num_double : ℕ := 3   -- Their three teenage children
  
  total_cost = num_regular * regular_scoop + num_kiddie * kiddie_scoop + num_double * double_scoop

theorem kiddie_scoop_cost : ice_cream_cost 3 := by
  sorry

end kiddie_scoop_cost_l2972_297267


namespace cube_immersion_theorem_l2972_297278

/-- The edge length of a cube that, when immersed in a rectangular vessel,
    causes a specific rise in water level. -/
def cube_edge_length (vessel_length vessel_width water_rise : ℝ) : ℝ :=
  (vessel_length * vessel_width * water_rise) ^ (1/3)

/-- Theorem stating that a cube with edge length 16 cm, when immersed in a
    rectangular vessel with base 20 cm × 15 cm, causes a water level rise
    of 13.653333333333334 cm. -/
theorem cube_immersion_theorem :
  cube_edge_length 20 15 13.653333333333334 = 16 := by
  sorry

end cube_immersion_theorem_l2972_297278


namespace system_solution_l2972_297247

theorem system_solution :
  ∃ (x y : ℝ), 
    y * (x + y)^2 = 9 ∧
    y * (x^3 - y^3) = 7 ∧
    x = 2 ∧ y = 1 := by
  sorry

end system_solution_l2972_297247


namespace square_ratio_side_length_l2972_297270

theorem square_ratio_side_length (area_ratio : ℚ) :
  area_ratio = 75 / 128 →
  ∃ (a b c : ℕ), 
    (a = 5 ∧ b = 6 ∧ c = 16) ∧
    (Real.sqrt area_ratio = a * Real.sqrt b / c) :=
by sorry

end square_ratio_side_length_l2972_297270


namespace zachary_crunch_pushup_difference_l2972_297297

/-- Given information about Zachary's and David's exercises, prove that Zachary did 12 more crunches than push-ups. -/
theorem zachary_crunch_pushup_difference :
  ∀ (zachary_pushups zachary_crunches david_pushups david_crunches : ℕ),
    zachary_pushups = 46 →
    zachary_crunches = 58 →
    david_pushups = zachary_pushups + 38 →
    david_crunches = zachary_crunches - 62 →
    zachary_crunches - zachary_pushups = 12 :=
by sorry

end zachary_crunch_pushup_difference_l2972_297297


namespace fencing_cost_is_225_rupees_l2972_297217

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculate the total fencing cost for a rectangular park -/
def calculate_fencing_cost (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width) * park.fencing_cost_per_meter

/-- Theorem: The fencing cost for the given rectangular park is 225 rupees -/
theorem fencing_cost_is_225_rupees :
  ∀ (park : RectangularPark),
    park.length / park.width = 3 / 2 →
    park.area = 3750 →
    park.fencing_cost_per_meter = 0.9 →
    calculate_fencing_cost park = 225 := by
  sorry


end fencing_cost_is_225_rupees_l2972_297217


namespace necessary_is_necessary_necessary_not_sufficient_l2972_297282

-- Define the proposition p
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + 2*m ≥ 0

-- Define the necessary condition
def necessary_condition (m : ℝ) : Prop := m ≥ 1

-- Theorem: The necessary condition is indeed necessary
theorem necessary_is_necessary : 
  ∀ m : ℝ, p m → necessary_condition m := by sorry

-- Theorem: The necessary condition is not sufficient
theorem necessary_not_sufficient :
  ∃ m : ℝ, necessary_condition m ∧ ¬(p m) := by sorry

end necessary_is_necessary_necessary_not_sufficient_l2972_297282


namespace ellipse_foci_l2972_297252

/-- The equation of an ellipse in the form (x²/a² + y²/b² = 1) -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The coordinates of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given ellipse equation (x²/2 + y² = 1), prove that its foci are at (±1, 0) -/
theorem ellipse_foci (e : Ellipse) (h : e.a^2 = 2 ∧ e.b^2 = 1) :
  ∃ (p₁ p₂ : Point), p₁.x = 1 ∧ p₁.y = 0 ∧ p₂.x = -1 ∧ p₂.y = 0 ∧
  (∀ (p : Point), (p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1) →
    (p = p₁ ∨ p = p₂ → 
      (p.x - 0)^2 + (p.y - 0)^2 = (e.a^2 - e.b^2))) :=
sorry

end ellipse_foci_l2972_297252


namespace complex_sum_power_l2972_297238

theorem complex_sum_power (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^100 + z^101 + z^102 + z^103 = 0 := by sorry

end complex_sum_power_l2972_297238


namespace conference_handshakes_l2972_297230

/-- Calculates the maximum number of handshakes in a conference with given constraints -/
def max_handshakes (total : ℕ) (committee : ℕ) (red_badges : ℕ) : ℕ :=
  let participants := total - committee - red_badges
  participants * (participants - 1) / 2

/-- Theorem stating the maximum number of handshakes for the given conference -/
theorem conference_handshakes :
  max_handshakes 50 10 5 = 595 := by
  sorry

end conference_handshakes_l2972_297230


namespace expected_value_ten_sided_die_l2972_297266

/-- A fair 10-sided die with faces numbered from 1 to 10 -/
def TenSidedDie : Finset ℕ := Finset.range 10

/-- The expected value of rolling the die -/
def ExpectedValue : ℚ := (Finset.sum TenSidedDie (λ i => i + 1)) / 10

/-- Theorem: The expected value of rolling a fair 10-sided die with faces numbered from 1 to 10 is 5.5 -/
theorem expected_value_ten_sided_die : ExpectedValue = 11/2 := by
  sorry

end expected_value_ten_sided_die_l2972_297266


namespace converse_not_always_true_l2972_297272

theorem converse_not_always_true : ∃ (a b : ℝ), a < b ∧ ¬(∀ (m : ℝ), a * m^2 < b * m^2) :=
sorry

end converse_not_always_true_l2972_297272


namespace hockey_league_games_l2972_297237

theorem hockey_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 15) (h2 : total_games = 1050) :
  ∃ (games_per_pair : ℕ), 
    games_per_pair * (n * (n - 1) / 2) = total_games ∧ 
    games_per_pair = 10 := by
sorry

end hockey_league_games_l2972_297237


namespace rectangle_triangle_perimeter_l2972_297209

theorem rectangle_triangle_perimeter (d : ℕ) : 
  let triangle_side := 3 * w - d
  let rectangle_width := w
  let rectangle_length := 3 * w
  let triangle_perimeter := 3 * triangle_side
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)
  (∀ w : ℕ, 
    triangle_perimeter > 0 ∧ 
    rectangle_perimeter = triangle_perimeter + 1950 ∧
    rectangle_length - triangle_side = d ∧
    rectangle_length = 3 * rectangle_width) →
  d > 650 :=
by sorry

end rectangle_triangle_perimeter_l2972_297209


namespace min_distinct_values_l2972_297271

/-- Represents a list of integers with a unique mode -/
structure IntegerList where
  elements : List Nat
  mode_count : Nat
  distinct_count : Nat
  mode_is_unique : Bool

/-- Properties of the integer list -/
def valid_integer_list (L : IntegerList) : Prop :=
  L.elements.length = 2018 ∧
  L.mode_count = 10 ∧
  L.mode_is_unique = true

/-- Theorem stating the minimum number of distinct values -/
theorem min_distinct_values (L : IntegerList) :
  valid_integer_list L → L.distinct_count ≥ 225 := by
  sorry

#check min_distinct_values

end min_distinct_values_l2972_297271


namespace function_growth_l2972_297263

open Real

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem function_growth (hf : ∀ x, f x < f' x) :
  (f 1 > Real.exp 1 * f 0) ∧ (f 2023 > Real.exp 2023 * f 0) := by
  sorry

end function_growth_l2972_297263


namespace greatest_multiple_of_5_and_6_less_than_700_l2972_297240

theorem greatest_multiple_of_5_and_6_less_than_700 : 
  ∃ n : ℕ, n = 690 ∧ 
  (∀ m : ℕ, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 700 → m ≤ n) ∧
  n % 5 = 0 ∧ n % 6 = 0 ∧ n < 700 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_700_l2972_297240


namespace smallest_special_number_is_correct_l2972_297281

/-- The smallest positive integer that is not prime, not a square, and has no prime factor less than 100 -/
def smallest_special_number : ℕ := 10403

/-- A number is special if it is not prime, not a square, and has no prime factor less than 100 -/
def is_special (n : ℕ) : Prop :=
  ¬ Nat.Prime n ∧ ¬ ∃ m : ℕ, n = m * m ∧ ∀ p : ℕ, Nat.Prime p → p < 100 → ¬ p ∣ n

theorem smallest_special_number_is_correct :
  is_special smallest_special_number ∧
  ∀ n : ℕ, 0 < n → n < smallest_special_number → ¬ is_special n :=
sorry

end smallest_special_number_is_correct_l2972_297281


namespace domain_of_f_l2972_297244

def f (x : ℝ) : ℝ := (x + 1) ^ 0

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -1} :=
sorry

end domain_of_f_l2972_297244


namespace age_difference_proof_l2972_297223

theorem age_difference_proof :
  ∀ (a b : ℕ),
    a + b = 2 →
    (10 * a + b) + (10 * b + a) = 22 →
    (10 * a + b + 7) = 3 * (10 * b + a + 7) →
    (10 * a + b) - (10 * b + a) = 18 :=
by
  sorry

end age_difference_proof_l2972_297223


namespace max_value_fraction_l2972_297248

theorem max_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 2 / 2 ∧
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    (x' * y' + y' * z') / (x'^2 + y'^2 + z'^2) = Real.sqrt 2 / 2 := by
  sorry

end max_value_fraction_l2972_297248


namespace gcd_lcm_product_l2972_297279

theorem gcd_lcm_product (a b : Nat) (h1 : a = 180) (h2 : b = 250) :
  (Nat.gcd a b) * (Nat.lcm a b) = 45000 := by
  sorry

end gcd_lcm_product_l2972_297279


namespace reflection_across_x_axis_l2972_297291

-- Define a function f over the real numbers
variable (f : ℝ → ℝ)

-- Define the reflection operation
def reflect (f : ℝ → ℝ) : ℝ → ℝ := λ x => -f x

-- Theorem statement
theorem reflection_across_x_axis (x y : ℝ) :
  (y = f x) ↔ (-y = (reflect f) x) :=
sorry

end reflection_across_x_axis_l2972_297291


namespace selection_probabilities_l2972_297200

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 4

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 3

/-- Represents the total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- Represents the number of people to be selected -/
def num_selected : ℕ := 2

/-- Calculates the probability of selecting two boys -/
def prob_two_boys : ℚ := (num_boys.choose 2) / (total_people.choose 2)

/-- Calculates the probability of selecting exactly one girl -/
def prob_one_girl : ℚ := (num_boys.choose 1 * num_girls.choose 1) / (total_people.choose 2)

/-- Calculates the probability of selecting at least one girl -/
def prob_at_least_one_girl : ℚ := 1 - prob_two_boys

theorem selection_probabilities :
  prob_two_boys = 2/7 ∧
  prob_one_girl = 4/7 ∧
  prob_at_least_one_girl = 5/7 := by
  sorry

end selection_probabilities_l2972_297200


namespace northwest_molded_handle_cost_l2972_297213

/-- Northwest Molded's handle production problem -/
theorem northwest_molded_handle_cost 
  (fixed_cost : ℝ) 
  (selling_price : ℝ) 
  (break_even_quantity : ℕ) 
  (h1 : fixed_cost = 7640)
  (h2 : selling_price = 4.60)
  (h3 : break_even_quantity = 1910) :
  ∃ (cost_per_handle : ℝ), 
    cost_per_handle = 0.60 ∧ 
    (selling_price * break_even_quantity : ℝ) = fixed_cost + (break_even_quantity : ℝ) * cost_per_handle :=
by sorry

end northwest_molded_handle_cost_l2972_297213


namespace rental_cost_equality_l2972_297261

/-- Represents the rental cost scenario for two computers -/
structure RentalCost where
  B : ℝ  -- Hourly rate for computer B
  T : ℝ  -- Time taken by computer A to complete the job

/-- The total cost is the same for both computers and equals 70 times the hourly rate of computer B -/
theorem rental_cost_equality (rc : RentalCost) : 
  1.4 * rc.B * rc.T = rc.B * (rc.T + 20) ∧ 
  1.4 * rc.B * rc.T = 70 * rc.B :=
by sorry

end rental_cost_equality_l2972_297261


namespace min_value_of_exponential_sum_l2972_297298

theorem min_value_of_exponential_sum (a b : ℝ) (h : 2 * a + 3 * b = 4) :
  ∃ (m : ℝ), m = 8 ∧ ∀ x y, 2 * x + 3 * y = 4 → 4^x + 8^y ≥ m :=
sorry

end min_value_of_exponential_sum_l2972_297298


namespace f_simplification_f_value_in_second_quadrant_l2972_297277

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 5 * Real.pi / 2) * Real.cos (3 * Real.pi / 2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_value_in_second_quadrant (α : Real) 
  (h1 : Real.cos (α + 3 * Real.pi / 2) = 1/5) 
  (h2 : 0 < α ∧ α < Real.pi) : 
  f α = 2 * Real.sqrt 6 / 5 := by sorry

end f_simplification_f_value_in_second_quadrant_l2972_297277


namespace tank_capacity_l2972_297262

/-- Represents a cylindrical water tank -/
structure WaterTank where
  capacity : ℝ
  initialWater : ℝ

/-- Proves that the tank's capacity is 30 liters given the conditions -/
theorem tank_capacity (tank : WaterTank)
  (h1 : tank.initialWater / tank.capacity = 1 / 6)
  (h2 : (tank.initialWater + 5) / tank.capacity = 1 / 3) :
  tank.capacity = 30 := by
  sorry

#check tank_capacity

end tank_capacity_l2972_297262


namespace kindergarten_tissues_l2972_297229

/-- The number of tissues in each mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The number of students in each kindergartner group -/
def group_sizes : List ℕ := [9, 10, 11]

/-- The total number of tissues brought by all kindergartner groups -/
def total_tissues : ℕ := (group_sizes.sum) * tissues_per_box

theorem kindergarten_tissues :
  total_tissues = 1200 :=
by sorry

end kindergarten_tissues_l2972_297229


namespace train_problem_solution_l2972_297220

/-- Represents the train problem scenario -/
structure TrainProblem where
  total_distance : ℝ
  train_a_speed : ℝ
  train_b_speed : ℝ
  separation_distance : ℝ

/-- The time when trains are at the given separation distance -/
def separation_time (p : TrainProblem) : Set ℝ :=
  { t : ℝ | t = (p.total_distance - p.separation_distance) / (p.train_a_speed + p.train_b_speed) ∨
             t = (p.total_distance + p.separation_distance) / (p.train_a_speed + p.train_b_speed) }

/-- Theorem stating the solution to the train problem -/
theorem train_problem_solution (p : TrainProblem) 
    (h1 : p.total_distance = 840)
    (h2 : p.train_a_speed = 68.5)
    (h3 : p.train_b_speed = 71.5)
    (h4 : p.separation_distance = 210) :
    separation_time p = {4.5, 7.5} := by
  sorry

end train_problem_solution_l2972_297220


namespace angle_sum_is_pi_over_two_l2972_297265

theorem angle_sum_is_pi_over_two (a b : ℝ) 
  (h_acute_a : 0 < a ∧ a < π / 2) 
  (h_acute_b : 0 < b ∧ b < π / 2)
  (h1 : 4 * Real.sin a ^ 2 + 3 * Real.sin b ^ 2 = 1)
  (h2 : 4 * Real.sin (2 * a) - 3 * Real.sin (2 * b) = 0) : 
  2 * a + b = π / 2 := by
sorry

end angle_sum_is_pi_over_two_l2972_297265


namespace vessel_combination_theorem_l2972_297216

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℚ
  denominator : ℚ

/-- Represents a vessel containing a mixture of milk and water -/
structure Vessel where
  volume : ℚ
  milkWaterRatio : Ratio

/-- Combines the contents of two vessels -/
def combineVessels (v1 v2 : Vessel) : Ratio :=
  let totalMilk := v1.volume * v1.milkWaterRatio.numerator / (v1.milkWaterRatio.numerator + v1.milkWaterRatio.denominator) +
                   v2.volume * v2.milkWaterRatio.numerator / (v2.milkWaterRatio.numerator + v2.milkWaterRatio.denominator)
  let totalWater := v1.volume * v1.milkWaterRatio.denominator / (v1.milkWaterRatio.numerator + v1.milkWaterRatio.denominator) +
                    v2.volume * v2.milkWaterRatio.denominator / (v2.milkWaterRatio.numerator + v2.milkWaterRatio.denominator)
  { numerator := totalMilk, denominator := totalWater }

theorem vessel_combination_theorem :
  let v1 : Vessel := { volume := 3, milkWaterRatio := { numerator := 1, denominator := 2 } }
  let v2 : Vessel := { volume := 5, milkWaterRatio := { numerator := 3, denominator := 2 } }
  let combinedRatio := combineVessels v1 v2
  combinedRatio.numerator = combinedRatio.denominator :=
by
  sorry

end vessel_combination_theorem_l2972_297216


namespace rahul_ppf_savings_l2972_297251

/-- Represents Rahul's savings in rupees -/
structure RahulSavings where
  nsc : ℕ  -- National Savings Certificate
  ppf : ℕ  -- Public Provident Fund

/-- The conditions of Rahul's savings -/
def savingsConditions (s : RahulSavings) : Prop :=
  s.nsc + s.ppf = 180000 ∧ s.nsc / 3 = s.ppf / 2

/-- Theorem stating Rahul's Public Provident Fund savings -/
theorem rahul_ppf_savings (s : RahulSavings) (h : savingsConditions s) : s.ppf = 72000 := by
  sorry

#check rahul_ppf_savings

end rahul_ppf_savings_l2972_297251


namespace quadratic_equation_solution_l2972_297293

theorem quadratic_equation_solution (k : ℝ) : 
  (8 * ((-15 - Real.sqrt 145) / 8)^2 + 15 * ((-15 - Real.sqrt 145) / 8) + k = 0) → 
  (k = 5/2) := by
  sorry

end quadratic_equation_solution_l2972_297293


namespace rectangular_field_length_l2972_297264

theorem rectangular_field_length (width : ℝ) (length : ℝ) : 
  width = 13.5 → length = 2 * width - 3 → length = 24 := by
  sorry

end rectangular_field_length_l2972_297264


namespace sum_of_50th_terms_l2972_297233

theorem sum_of_50th_terms (a₁ a₅₀ : ℝ) (d : ℝ) (g₁ g₅₀ : ℝ) (r : ℝ) : 
  a₁ = 3 → d = 6 → g₁ = 2 → r = 3 →
  a₅₀ = a₁ + 49 * d →
  g₅₀ = g₁ * r^49 →
  a₅₀ + g₅₀ = 297 + 2 * 3^49 :=
by sorry

end sum_of_50th_terms_l2972_297233


namespace sufficient_condition_ranges_not_sufficient_condition_ranges_l2972_297257

/-- Condition p: (x+1)(2-x) ≥ 0 -/
def p (x : ℝ) : Prop := (x + 1) * (2 - x) ≥ 0

/-- Condition q: x^2+mx-2m^2-3m-1 < 0, where m > -2/3 -/
def q (x m : ℝ) : Prop := x^2 + m*x - 2*m^2 - 3*m - 1 < 0 ∧ m > -2/3

theorem sufficient_condition_ranges (m : ℝ) :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) → m > 1 :=
sorry

theorem not_sufficient_condition_ranges (m : ℝ) :
  (∀ x, ¬p x → ¬q x m) ∧ (∃ x, ¬q x m ∧ p x) → -2/3 < m ∧ m ≤ 0 :=
sorry

end sufficient_condition_ranges_not_sufficient_condition_ranges_l2972_297257


namespace internet_cost_decrease_l2972_297283

theorem internet_cost_decrease (initial_cost final_cost : ℝ) 
  (h1 : initial_cost = 120)
  (h2 : final_cost = 45) : 
  (initial_cost - final_cost) / initial_cost * 100 = 62.5 := by
sorry

end internet_cost_decrease_l2972_297283


namespace optimal_landing_point_l2972_297299

/-- The optimal landing point for a messenger traveling from a boat to a camp on shore -/
theorem optimal_landing_point (boat_distance : ℝ) (camp_distance : ℝ) 
  (row_speed : ℝ) (walk_speed : ℝ) : ℝ :=
let landing_point := 12
let total_time (x : ℝ) := 
  (Real.sqrt (boat_distance^2 + x^2)) / row_speed + (camp_distance - x) / walk_speed
have h1 : boat_distance = 9 := by sorry
have h2 : camp_distance = 15 := by sorry
have h3 : row_speed = 4 := by sorry
have h4 : walk_speed = 5 := by sorry
have h5 : ∀ x, total_time landing_point ≤ total_time x := by sorry
landing_point

#check optimal_landing_point

end optimal_landing_point_l2972_297299


namespace min_Q_value_l2972_297289

def is_special_number (m : ℕ) : Prop :=
  m ≥ 10 ∧ m < 100 ∧ (m / 10) ≠ (m % 10) ∧ (m / 10) ≠ 0 ∧ (m % 10) ≠ 0

def F (m : ℕ) : ℤ :=
  let m₁ := (m % 10) * 10 + (m / 10)
  (m * 100 + m₁ - (m₁ * 100 + m)) / 99

def Q (s t : ℕ) : ℚ :=
  (t - s : ℚ) / s

theorem min_Q_value (s t : ℕ) (a b x y : ℕ) :
  is_special_number s →
  is_special_number t →
  s = 10 * a + b →
  t = 10 * x + y →
  1 ≤ b →
  b < a →
  a ≤ 7 →
  1 ≤ x →
  x ≤ 8 →
  1 ≤ y →
  y ≤ 8 →
  F s % 5 = 1 →
  F t - F s + 18 * x = 36 →
  ∀ (s' t' : ℕ), is_special_number s' → is_special_number t' → Q s' t' ≥ Q s t →
  Q s t = -42 / 73 :=
sorry

end min_Q_value_l2972_297289


namespace sqrt_sum_fractions_l2972_297254

theorem sqrt_sum_fractions : Real.sqrt ((1 : ℝ) / 8 + (1 : ℝ) / 18) = (Real.sqrt 26) / 12 := by
  sorry

end sqrt_sum_fractions_l2972_297254


namespace intersection_point_coordinates_l2972_297259

/-- Given a triangle ABC with points D and E as described, prove that the intersection P of BE and AD
    has the vector representation P = (8/14)A + (1/14)B + (4/14)C -/
theorem intersection_point_coordinates (A B C D E P : ℝ × ℝ) : 
  (∃ (k : ℝ), D = k • C + (1 - k) • B ∧ k = 5/4) →  -- BD:DC = 4:1
  (∃ (m : ℝ), E = m • A + (1 - m) • C ∧ m = 2/3) →  -- AE:EC = 2:1
  (∃ (t : ℝ), P = t • A + (1 - t) • D) →            -- P is on AD
  (∃ (s : ℝ), P = s • B + (1 - s) • E) →            -- P is on BE
  (∃ (x y z : ℝ), P = x • A + y • B + z • C ∧ x + y + z = 1) →
  P = (8/14) • A + (1/14) • B + (4/14) • C :=
by sorry


end intersection_point_coordinates_l2972_297259


namespace ratio_sum_squares_to_sum_l2972_297243

theorem ratio_sum_squares_to_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : b = 2 * a) (h5 : c = 4 * a) (h6 : a^2 + b^2 + c^2 = 1701) : 
  a + b + c = 63 := by
  sorry

end ratio_sum_squares_to_sum_l2972_297243


namespace product_and_reciprocal_sum_l2972_297292

theorem product_and_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 16) (h2 : 1 / x = 3 / y) : x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end product_and_reciprocal_sum_l2972_297292


namespace quadratic_equation_satisfaction_l2972_297285

theorem quadratic_equation_satisfaction (p q : ℝ) : 
  p^2 + 9*q^2 + 3*p - p*q = 30 ∧ p - 5*q - 8 = 0 → p^2 - p - 6 = 0 := by
  sorry

end quadratic_equation_satisfaction_l2972_297285


namespace cirrus_count_l2972_297241

/-- The number of cumulonimbus clouds -/
def cumulonimbus : ℕ := 3

/-- The number of cumulus clouds -/
def cumulus : ℕ := 12 * cumulonimbus

/-- The number of cirrus clouds -/
def cirrus : ℕ := 4 * cumulus

/-- The number of altostratus clouds -/
def altostratus : ℕ := 6 * (cirrus + cumulus)

/-- Theorem stating that the number of cirrus clouds is 144 -/
theorem cirrus_count : cirrus = 144 := by sorry

end cirrus_count_l2972_297241


namespace julians_boy_friends_percentage_l2972_297260

theorem julians_boy_friends_percentage 
  (julian_total_friends : ℕ)
  (julian_girls_percentage : ℚ)
  (boyd_total_friends : ℕ)
  (boyd_boys_percentage : ℚ)
  (h1 : julian_total_friends = 80)
  (h2 : julian_girls_percentage = 40/100)
  (h3 : boyd_total_friends = 100)
  (h4 : boyd_boys_percentage = 36/100)
  (h5 : (boyd_total_friends : ℚ) * (1 - boyd_boys_percentage) = 2 * (julian_total_friends : ℚ) * julian_girls_percentage) :
  (julian_total_friends : ℚ) * (1 - julian_girls_percentage) / julian_total_friends = 60/100 := by
  sorry

end julians_boy_friends_percentage_l2972_297260


namespace periodic_function_value_l2972_297287

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) + 4,
    where a, b, α, β are non-zero real numbers, and f(2007) = 5,
    prove that f(2008) = 3 -/
theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  f 2007 = 5 → f 2008 = 3 := by
  sorry

end periodic_function_value_l2972_297287


namespace eden_bears_count_eden_final_bears_count_l2972_297206

theorem eden_bears_count (initial_bears : ℕ) (favorite_bears : ℕ) (sisters : ℕ) (eden_initial_bears : ℕ) : ℕ :=
  let remaining_bears := initial_bears - favorite_bears
  let bears_per_sister := remaining_bears / sisters
  eden_initial_bears + bears_per_sister

theorem eden_final_bears_count :
  eden_bears_count 20 8 3 10 = 14 := by
  sorry

end eden_bears_count_eden_final_bears_count_l2972_297206


namespace quadratic_equation_solution_l2972_297202

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 8 * x - 18 = 0 ↔ x = 3 ∨ x = -4/3) → k = 9/2 :=
by sorry

end quadratic_equation_solution_l2972_297202


namespace parallel_lines_m_value_l2972_297256

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ (l1.a ≠ 0 ∨ l1.b ≠ 0) ∧ (l2.a ≠ 0 ∨ l2.b ≠ 0)

theorem parallel_lines_m_value :
  let l1 : Line := { a := 3, b := 4, c := -3 }
  let l2 : Line := { a := 6, b := m, c := 14 }
  parallel l1 l2 → m = 8 := by
  sorry

end parallel_lines_m_value_l2972_297256


namespace smallest_positive_a_for_parabola_l2972_297234

theorem smallest_positive_a_for_parabola :
  ∀ (a b c : ℚ),
  (∀ x y : ℚ, y = a * x^2 + b * x + c ↔ y + 5/4 = a * (x - 1/2)^2) →
  a > 0 →
  ∃ n : ℤ, a + b + c = n →
  (∀ a' : ℚ, a' > 0 → (∃ b' c' : ℚ, (∀ x y : ℚ, y = a' * x^2 + b' * x + c' ↔ y + 5/4 = a' * (x - 1/2)^2) ∧ 
                      (∃ n' : ℤ, a' + b' + c' = n')) → a' ≥ a) →
  a = 1 :=
by sorry

end smallest_positive_a_for_parabola_l2972_297234


namespace complex_number_properties_l2972_297286

theorem complex_number_properties (z : ℂ) (h : z * (2 + Complex.I) = Complex.I ^ 10) :
  (Complex.abs z = Real.sqrt 5 / 5) ∧
  (Complex.re z < 0 ∧ Complex.im z > 0) :=
sorry

end complex_number_properties_l2972_297286


namespace inequality_solution_set_l2972_297284

/-- Given an inequality tx^2 - 6x + t^2 < 0 with solution set (-∞,a)∪(1,+∞), prove that a = -3 -/
theorem inequality_solution_set (t : ℝ) (a : ℝ) :
  (∀ x : ℝ, (t * x^2 - 6 * x + t^2 < 0) ↔ (x < a ∨ x > 1)) →
  a = -3 :=
by sorry

end inequality_solution_set_l2972_297284


namespace root_equation_c_value_l2972_297231

theorem root_equation_c_value :
  ∀ (c d e : ℚ),
  (∃ (x : ℝ), x = -2 + 3 * Real.sqrt 5 ∧ x^4 + c*x^3 + d*x^2 + e*x - 48 = 0) →
  c = 5 := by
sorry

end root_equation_c_value_l2972_297231


namespace older_sibling_age_l2972_297295

def mother_charge : ℚ := 495 / 100
def child_charge_per_year : ℚ := 35 / 100
def total_bill : ℚ := 985 / 100

def is_valid_age_combination (twin_age older_age : ℕ) : Prop :=
  twin_age ≤ older_age ∧
  mother_charge + child_charge_per_year * (2 * twin_age + older_age) = total_bill

theorem older_sibling_age :
  ∃ (twin_age older_age : ℕ), is_valid_age_combination twin_age older_age ∧
  (older_age = 4 ∨ older_age = 6) :=
by sorry

end older_sibling_age_l2972_297295


namespace max_three_digit_with_remainders_l2972_297294

theorem max_three_digit_with_remainders :
  ∀ N : ℕ,
  (100 ≤ N ∧ N ≤ 999) →
  (N % 3 = 1) →
  (N % 7 = 3) →
  (N % 11 = 8) →
  (∀ M : ℕ, (100 ≤ M ∧ M ≤ 999) → (M % 3 = 1) → (M % 7 = 3) → (M % 11 = 8) → M ≤ N) →
  N = 976 := by
sorry

end max_three_digit_with_remainders_l2972_297294


namespace eighth_number_in_set_l2972_297232

theorem eighth_number_in_set (known_numbers : List ℕ) (average : ℚ) : 
  known_numbers = [1, 2, 4, 5, 6, 9, 9, 12] ∧ 
  average = 7 ∧
  (List.sum known_numbers + 12) / 9 = average →
  ∃ x : ℕ, x = 3 ∧ x ∈ (known_numbers ++ [12]) :=
by sorry

end eighth_number_in_set_l2972_297232


namespace k_value_proof_l2972_297215

theorem k_value_proof (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) →
  k = 6 := by
sorry

end k_value_proof_l2972_297215


namespace least_positive_integer_multiple_l2972_297268

theorem least_positive_integer_multiple (x : ℕ) : x = 42 ↔ 
  (x > 0 ∧ ∀ y : ℕ, y > 0 → y < x → ¬(∃ k : ℤ, (2 * y + 45)^2 = 43 * k)) ∧
  (∃ k : ℤ, (2 * x + 45)^2 = 43 * k) :=
by sorry

end least_positive_integer_multiple_l2972_297268


namespace range_of_a_l2972_297258

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 + 1 ≥ a

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 1 = 0

-- Define the main theorem
theorem range_of_a (a : ℝ) : 
  (¬(¬(p a) ∨ ¬(q a))) → (a ≤ -1 ∨ (1 ≤ a ∧ a ≤ 2)) :=
by sorry

end range_of_a_l2972_297258


namespace sheila_cinnamon_balls_l2972_297210

/-- The number of family members -/
def family_members : ℕ := 5

/-- The number of days Sheila can place cinnamon balls in socks -/
def days : ℕ := 10

/-- The number of cinnamon balls Sheila bought -/
def cinnamon_balls : ℕ := family_members * days

theorem sheila_cinnamon_balls : cinnamon_balls = 50 := by
  sorry

end sheila_cinnamon_balls_l2972_297210


namespace compound_interest_rate_l2972_297236

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r / 100)^2 = 2420)
  (h2 : P * (1 + r / 100)^3 = 3025) :
  r = 25 := by
  sorry

end compound_interest_rate_l2972_297236


namespace max_value_z_l2972_297246

theorem max_value_z (x y : ℝ) (h1 : 6 ≤ x + y) (h2 : x + y ≤ 8) (h3 : -2 ≤ x - y) (h4 : x - y ≤ 0) :
  ∃ (z : ℝ), z = 2 * x + 5 * y ∧ z ≤ 8 ∧ ∀ (w : ℝ), w = 2 * x + 5 * y → w ≤ z :=
by sorry

end max_value_z_l2972_297246
