import Mathlib

namespace percentage_sum_theorem_l3912_391285

theorem percentage_sum_theorem : (0.15 * 25) + (0.12 * 45) = 9.15 := by sorry

end percentage_sum_theorem_l3912_391285


namespace tensor_identity_l3912_391259

/-- Define a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Define the ⊗ operation -/
def tensor (m n : Vector2D) : Vector2D :=
  ⟨m.x * n.x + m.y * n.y, m.x * n.y + m.y * n.x⟩

theorem tensor_identity (p : Vector2D) : 
  (∀ m : Vector2D, tensor m p = m) → p = ⟨1, 0⟩ := by
  sorry

end tensor_identity_l3912_391259


namespace total_loaves_served_l3912_391209

theorem total_loaves_served (wheat_bread : Real) (white_bread : Real)
  (h1 : wheat_bread = 0.5)
  (h2 : white_bread = 0.4) :
  wheat_bread + white_bread = 0.9 := by
  sorry

end total_loaves_served_l3912_391209


namespace fly_distance_from_ceiling_l3912_391225

theorem fly_distance_from_ceiling :
  ∀ (x y z : ℝ),
  x = 3 →
  y = 4 →
  Real.sqrt (x^2 + y^2 + z^2) = 7 →
  z = 2 * Real.sqrt 6 := by sorry

end fly_distance_from_ceiling_l3912_391225


namespace arithmetic_calculation_l3912_391280

theorem arithmetic_calculation : (8 * 4) + 3 = 35 := by
  sorry

end arithmetic_calculation_l3912_391280


namespace tuesday_rejects_l3912_391227

/-- The percentage of meters rejected as defective -/
def reject_rate : ℝ := 0.0007

/-- The number of meters rejected on Monday -/
def monday_rejects : ℕ := 7

/-- The increase in meters examined on Tuesday compared to Monday -/
def tuesday_increase : ℝ := 0.25

theorem tuesday_rejects : ℕ := by
  sorry

end tuesday_rejects_l3912_391227


namespace g_of_six_l3912_391204

/-- A function satisfying the given properties -/
def FunctionG (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x + y) = g x + g y) ∧ g 5 = 6

/-- The main theorem -/
theorem g_of_six (g : ℝ → ℝ) (h : FunctionG g) : g 6 = 36/5 := by
  sorry

end g_of_six_l3912_391204


namespace quadratic_vertex_not_minus_one_minus_three_a_l3912_391271

/-- Given a quadratic function y = ax^2 + 2ax - 3a where a > 0,
    prove that its vertex coordinates are not (-1, -3a) -/
theorem quadratic_vertex_not_minus_one_minus_three_a (a : ℝ) (h : a > 0) :
  ∃ (x y : ℝ), (y = a*x^2 + 2*a*x - 3*a) ∧ 
  (∀ x' : ℝ, a*x'^2 + 2*a*x' - 3*a ≥ y) ∧
  (x ≠ -1 ∨ y ≠ -3*a) :=
by sorry

end quadratic_vertex_not_minus_one_minus_three_a_l3912_391271


namespace point_in_first_quadrant_l3912_391220

/-- A proportional function where y increases as x increases -/
structure IncreasingProportionalFunction where
  k : ℝ
  increasing : ∀ x₁ x₂, x₁ < x₂ → k * x₁ < k * x₂

/-- The point P with coordinates (3, k) -/
def P (f : IncreasingProportionalFunction) : ℝ × ℝ := (3, f.k)

/-- Theorem: P(3, k) lies in the first quadrant for an increasing proportional function -/
theorem point_in_first_quadrant (f : IncreasingProportionalFunction) :
  P f ∈ {p : ℝ × ℝ | 0 < p.1 ∧ 0 < p.2} := by
  sorry

end point_in_first_quadrant_l3912_391220


namespace f_2015_equals_one_l3912_391298

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2015_equals_one (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : ∀ x, f (x + 2) * f x = 1)
  (h3 : ∀ x, f x > 0) : 
  f 2015 = 1 := by sorry

end f_2015_equals_one_l3912_391298


namespace sequence_bounded_l3912_391226

/-- Given a sequence of nonnegative real numbers satisfying certain conditions, prove that it is bounded -/
theorem sequence_bounded (c : ℝ) (a : ℕ → ℝ) (hc : c > 2)
  (h1 : ∀ m n : ℕ, m ≥ 1 → n ≥ 1 → a (m + n) ≤ 2 * a m + 2 * a n)
  (h2 : ∀ k : ℕ, a (2^k) ≤ 1 / ((k : ℝ) + 1)^c)
  (h3 : ∀ n : ℕ, a n ≥ 0) :
  ∃ M : ℝ, ∀ n : ℕ, n ≥ 1 → a n ≤ M :=
sorry

end sequence_bounded_l3912_391226


namespace solution_set_f_positive_max_m_inequality_l3912_391228

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x > 1 ∨ x < -5} :=
sorry

-- Theorem for part II
theorem max_m_inequality (m : ℝ) :
  (∀ x : ℝ, f x + 3*|x - 4| > m) ↔ m < 9 :=
sorry

end solution_set_f_positive_max_m_inequality_l3912_391228


namespace probability_of_valid_assignment_l3912_391286

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

def valid_assignment (al bill cal : ℕ) : Prop :=
  1 ≤ al ∧ al ≤ 12 ∧
  1 ≤ bill ∧ bill ≤ 12 ∧
  1 ≤ cal ∧ cal ≤ 12 ∧
  is_multiple al bill ∧
  is_multiple bill cal

def total_assignments : ℕ := 12 * 12 * 12

def count_valid_assignments : ℕ := sorry

theorem probability_of_valid_assignment :
  (count_valid_assignments : ℚ) / total_assignments = 1 / 12 := by sorry

end probability_of_valid_assignment_l3912_391286


namespace scientific_notation_equality_l3912_391237

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  3230000 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.23 ∧ n = 6 := by
  sorry

end scientific_notation_equality_l3912_391237


namespace weight_range_proof_l3912_391263

theorem weight_range_proof (tracy_weight john_weight jake_weight : ℕ) : 
  tracy_weight = 52 →
  jake_weight = tracy_weight + 8 →
  tracy_weight + john_weight + jake_weight = 158 →
  (max tracy_weight (max john_weight jake_weight)) - 
  (min tracy_weight (min john_weight jake_weight)) = 14 := by
sorry

end weight_range_proof_l3912_391263


namespace bicycle_fog_problem_l3912_391233

/-- Bicycle and fog bank problem -/
theorem bicycle_fog_problem (v_bicycle : ℝ) (v_fog : ℝ) (r_fog : ℝ) (initial_distance : ℝ) :
  v_bicycle = 1/2 →
  v_fog = 1/3 * Real.sqrt 2 →
  r_fog = 40 →
  initial_distance = 100 →
  ∃ t₁ t₂ : ℝ,
    t₁ < t₂ ∧
    (∀ t, t₁ ≤ t ∧ t ≤ t₂ →
      (initial_distance - v_fog * t)^2 + (v_bicycle * t - v_fog * t)^2 ≤ r_fog^2) ∧
    (t₁ + t₂) / 2 = 240 :=
by sorry

end bicycle_fog_problem_l3912_391233


namespace unique_positive_solution_l3912_391236

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 6) / 16 = 6 / (x - 16) := by
  sorry

end unique_positive_solution_l3912_391236


namespace polynomial_has_negative_root_l3912_391203

-- Define the polynomial
def P (x : ℝ) : ℝ := x^7 - 2*x^6 - 7*x^4 - x^2 + 10

-- Theorem statement
theorem polynomial_has_negative_root : ∃ x : ℝ, x < 0 ∧ P x = 0 := by
  sorry

end polynomial_has_negative_root_l3912_391203


namespace sum_of_bounds_l3912_391247

def U : Type := ℝ

def A (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

def complement_A : Set ℝ := {x | x > 4 ∨ x < 3}

theorem sum_of_bounds (a b : ℝ) :
  A a b = (Set.univ \ complement_A) → a + b = 7 := by sorry

end sum_of_bounds_l3912_391247


namespace min_cost_square_base_l3912_391200

/-- Represents the dimensions and cost parameters of a rectangular open-top tank. -/
structure Tank where
  volume : ℝ
  depth : ℝ
  base_cost : ℝ
  wall_cost : ℝ

/-- Calculates the total cost of constructing the tank given its length and width. -/
def total_cost (t : Tank) (length width : ℝ) : ℝ :=
  t.base_cost * length * width + t.wall_cost * 2 * t.depth * (length + width)

/-- Theorem stating that the minimum cost for the specified tank is achieved with a square base of side length 3m. -/
theorem min_cost_square_base (t : Tank) 
    (h_volume : t.volume = 18)
    (h_depth : t.depth = 2)
    (h_base_cost : t.base_cost = 200)
    (h_wall_cost : t.wall_cost = 150) :
    ∃ (cost : ℝ), cost = 5400 ∧ 
    ∀ (l w : ℝ), l * w * t.depth = t.volume → total_cost t l w ≥ cost ∧
    total_cost t 3 3 = cost :=
  sorry

#check min_cost_square_base

end min_cost_square_base_l3912_391200


namespace min_value_fraction_l3912_391257

theorem min_value_fraction (x y : ℝ) (hx : 1/2 ≤ x ∧ x ≤ 2) (hy : 4/3 ≤ y ∧ y ≤ 3/2) :
  (x^3 * y^3) / (x^6 + 3*x^4*y^2 + 3*x^3*y^3 + 3*x^2*y^4 + y^6) ≥ 27/1081 :=
sorry

end min_value_fraction_l3912_391257


namespace balloon_arrangements_count_l3912_391238

/-- The number of distinct arrangements of letters in a word with 7 letters,
    where two letters are each repeated twice. -/
def balloonArrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of distinct arrangements of letters
    in a word with the given conditions is 1260. -/
theorem balloon_arrangements_count :
  balloonArrangements = 1260 := by
  sorry

end balloon_arrangements_count_l3912_391238


namespace new_girl_weight_l3912_391221

theorem new_girl_weight (n : ℕ) (initial_weight replaced_weight : ℝ) 
  (h1 : n = 25)
  (h2 : replaced_weight = 55)
  (h3 : (initial_weight - replaced_weight + new_weight) / n = initial_weight / n + 1) :
  new_weight = 80 :=
sorry

end new_girl_weight_l3912_391221


namespace largest_positive_integer_satisfying_condition_l3912_391245

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem largest_positive_integer_satisfying_condition :
  ∀ n : ℤ, n > 0 → binary_op n < -15 → n ≤ 4 ∧
  binary_op 4 < -15 ∧
  ∀ m : ℤ, m > 4 → binary_op m ≥ -15 := by
sorry

end largest_positive_integer_satisfying_condition_l3912_391245


namespace range_of_expression_l3912_391275

theorem range_of_expression (α β : ℝ) 
  (h_α : 1 < α ∧ α < 3) 
  (h_β : -4 < β ∧ β < 2) : 
  ∀ x : ℝ, (∃ α' β', 1 < α' ∧ α' < 3 ∧ -4 < β' ∧ β' < 2 ∧ x = 1/2 * α' - β') ↔ 
  (-3/2 < x ∧ x < 11/2) :=
sorry

end range_of_expression_l3912_391275


namespace workforce_from_company_a_l3912_391229

/-- Represents the workforce composition of a company -/
structure WorkforceComposition where
  managers : Real
  software_engineers : Real
  marketing : Real
  human_resources : Real
  support_staff : Real

/-- The workforce composition of Company A -/
def company_a : WorkforceComposition := {
  managers := 0.10,
  software_engineers := 0.70,
  marketing := 0.15,
  human_resources := 0.05,
  support_staff := 0
}

/-- The workforce composition of Company B -/
def company_b : WorkforceComposition := {
  managers := 0.25,
  software_engineers := 0.10,
  marketing := 0.15,
  human_resources := 0.05,
  support_staff := 0.45
}

/-- The workforce composition of the merged company -/
def merged_company : WorkforceComposition := {
  managers := 0.18,
  software_engineers := 0,
  marketing := 0,
  human_resources := 0.10,
  support_staff := 0.50
}

/-- The theorem stating the percentage of workforce from Company A in the merged company -/
theorem workforce_from_company_a : 
  ∃ (total_a total_b : Real), 
    total_a > 0 ∧ total_b > 0 ∧
    company_a.managers * total_a + company_b.managers * total_b = merged_company.managers * (total_a + total_b) ∧
    total_a / (total_a + total_b) = 7 / 15 := by
  sorry

#check workforce_from_company_a

end workforce_from_company_a_l3912_391229


namespace work_ratio_theorem_l3912_391276

theorem work_ratio_theorem (p1 p2 : ℕ) (h1 : p1 > 0) (h2 : p2 > 0) : 
  (p1 * 20 : ℚ) * (1 : ℚ) = (p2 * 5 : ℚ) * (1/2 : ℚ) → p2 / p1 = 2 := by
  sorry

end work_ratio_theorem_l3912_391276


namespace infinite_sum_of_digits_not_exceeding_two_l3912_391251

theorem infinite_sum_of_digits_not_exceeding_two (n : ℕ) :
  ∃ (x y z : ℤ), 4 * x^4 + y^4 - z^2 + 4 * x * y * z = 2 * (10 : ℤ)^(2 * n + 2) := by
  sorry

end infinite_sum_of_digits_not_exceeding_two_l3912_391251


namespace OL_length_OL_angle_tangent_intersection_product_l3912_391243

/-- Ellipse Γ: x²/4 + y² = 1 -/
def Γ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Point L in the third quadrant -/
def L : ℝ × ℝ := (-3, -3)

/-- OL = 3√2 -/
theorem OL_length : Real.sqrt (L.1^2 + L.2^2) = 3 * Real.sqrt 2 := by sorry

/-- Angle between negative x-axis and OL is π/4 -/
theorem OL_angle : Real.arctan (-L.2 / (-L.1)) = π / 4 := by sorry

/-- Function to represent a line passing through L with slope k -/
def line_through_L (k : ℝ) (x : ℝ) : ℝ := k * (x - L.1) + L.2

/-- Tangent line touches the ellipse at exactly one point -/
def is_tangent (k : ℝ) : Prop := 
  ∃! x, Γ x (line_through_L k x)

/-- The y-coordinates of the intersection points of the tangent lines with the y-axis -/
def y_intersections (k₁ k₂ : ℝ) : ℝ × ℝ := (line_through_L k₁ 0, line_through_L k₂ 0)

/-- Main theorem: The product of y-coordinates of intersection points is 9 -/
theorem tangent_intersection_product :
  ∃ k₁ k₂, is_tangent k₁ ∧ is_tangent k₂ ∧ k₁ ≠ k₂ ∧ 
    (y_intersections k₁ k₂).1 * (y_intersections k₁ k₂).2 = 9 := by sorry

end OL_length_OL_angle_tangent_intersection_product_l3912_391243


namespace min_value_f_inequality_abc_l3912_391201

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∀ x : ℝ, f x ≥ 3 := by sorry

-- Theorem for the inequality
theorem inequality_abc (a b c m : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hm : a + b + c = m) :
  b^2 / a + c^2 / b + a^2 / c ≥ 3 := by sorry

end min_value_f_inequality_abc_l3912_391201


namespace constant_sum_property_l3912_391281

/-- Represents a triangle with numbers assigned to its vertices -/
structure NumberedTriangle where
  x : ℝ  -- Number assigned to vertex A
  y : ℝ  -- Number assigned to vertex B
  z : ℝ  -- Number assigned to vertex C

/-- The sum of a vertex number and the opposite side sum is constant -/
theorem constant_sum_property (t : NumberedTriangle) :
  t.x + (t.y + t.z) = t.y + (t.z + t.x) ∧
  t.y + (t.z + t.x) = t.z + (t.x + t.y) ∧
  t.z + (t.x + t.y) = t.x + t.y + t.z :=
sorry

end constant_sum_property_l3912_391281


namespace sin_2alpha_value_l3912_391246

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the theorem
theorem sin_2alpha_value (α : ℝ) :
  (Real.cos α * P.1 = Real.sin α * P.2) →  -- Terminal side passes through P
  Real.sin (2 * α) = 4 / 5 := by
  sorry

end sin_2alpha_value_l3912_391246


namespace geometric_sequence_first_term_l3912_391219

/-- Given a geometric sequence where the third term is 27 and the fourth term is 36,
    prove that the first term of the sequence is 243/16. -/
theorem geometric_sequence_first_term (a : ℚ) (r : ℚ) :
  a * r^2 = 27 ∧ a * r^3 = 36 → a = 243/16 := by
  sorry

end geometric_sequence_first_term_l3912_391219


namespace salt_concentration_dilution_l3912_391282

/-- Proves that adding 70 kg of fresh water to 30 kg of sea water with 5% salt concentration
    results in a solution with 1.5% salt concentration. -/
theorem salt_concentration_dilution
  (initial_mass : ℝ)
  (initial_concentration : ℝ)
  (target_concentration : ℝ)
  (added_water : ℝ)
  (h1 : initial_mass = 30)
  (h2 : initial_concentration = 0.05)
  (h3 : target_concentration = 0.015)
  (h4 : added_water = 70) :
  let final_mass := initial_mass + added_water
  let salt_mass := initial_mass * initial_concentration
  (salt_mass / final_mass) = target_concentration :=
by sorry

end salt_concentration_dilution_l3912_391282


namespace dart_points_ratio_l3912_391218

/-- Prove that the ratio of the points of the third dart to the points of the bullseye is 1:2 -/
theorem dart_points_ratio :
  let bullseye_points : ℕ := 50
  let missed_points : ℕ := 0
  let total_score : ℕ := 75
  let third_dart_points : ℕ := total_score - bullseye_points - missed_points
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ a * bullseye_points = b * third_dart_points ∧ a = 1 ∧ b = 2 :=
by sorry

end dart_points_ratio_l3912_391218


namespace final_amount_calculation_l3912_391256

/-- Calculate the final amount after two years of compound interest --/
def final_amount (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount_after_first_year := initial_amount * (1 + rate1)
  amount_after_first_year * (1 + rate2)

/-- Theorem stating the final amount after two years of compound interest --/
theorem final_amount_calculation :
  final_amount 7644 0.04 0.05 = 8347.248 := by
  sorry

#eval final_amount 7644 0.04 0.05

end final_amount_calculation_l3912_391256


namespace courtyard_width_l3912_391272

theorem courtyard_width (length : ℝ) (num_bricks : ℕ) (brick_length brick_width : ℝ) :
  length = 25 ∧ 
  num_bricks = 20000 ∧ 
  brick_length = 0.2 ∧ 
  brick_width = 0.1 → 
  (num_bricks : ℝ) * brick_length * brick_width / length = 16 := by
  sorry

end courtyard_width_l3912_391272


namespace smallest_four_digit_divisible_by_43_l3912_391293

theorem smallest_four_digit_divisible_by_43 : 
  ∃ n : ℕ, 
    (n ≥ 1000 ∧ n < 10000) ∧ 
    n % 43 = 0 ∧
    (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) → m % 43 = 0 → m ≥ n) ∧
    n = 1032 := by
  sorry

end smallest_four_digit_divisible_by_43_l3912_391293


namespace mascot_prices_and_reduction_l3912_391266

/-- The price of a small mascot in yuan -/
def small_price : ℝ := 80

/-- The price of a large mascot in yuan -/
def large_price : ℝ := 120

/-- The price reduction in yuan -/
def price_reduction : ℝ := 10

theorem mascot_prices_and_reduction :
  /- Price of large mascot is 1.5 times the price of small mascot -/
  (large_price = 1.5 * small_price) ∧
  /- Number of small mascots purchased with 1200 yuan is 5 more than large mascots -/
  ((1200 / small_price) - (1200 / large_price) = 5) ∧
  /- Total sales revenue in February equals 75000 yuan -/
  ((small_price - price_reduction) * (500 + 10 * price_reduction) +
   (large_price - price_reduction) * 300 = 75000) := by
  sorry

end mascot_prices_and_reduction_l3912_391266


namespace exam_time_ratio_l3912_391268

theorem exam_time_ratio :
  let total_questions : ℕ := 200
  let type_a_questions : ℕ := 50
  let type_b_questions : ℕ := total_questions - type_a_questions
  let exam_duration_hours : ℕ := 3
  let exam_duration_minutes : ℕ := exam_duration_hours * 60
  let time_for_type_a : ℕ := 72
  let time_for_type_b : ℕ := exam_duration_minutes - time_for_type_a
  (time_for_type_a : ℚ) / time_for_type_b = 2 / 3 :=
by sorry

end exam_time_ratio_l3912_391268


namespace complex_equation_solution_l3912_391249

theorem complex_equation_solution (a : ℝ) : (a + Complex.I) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end complex_equation_solution_l3912_391249


namespace average_weight_b_c_l3912_391250

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 47 kg. -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 39 →
  (b + c) / 2 = 47 := by
sorry

end average_weight_b_c_l3912_391250


namespace not_divisible_by_two_2013_l3912_391202

-- Define a property for odd numbers
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Define what it means for a number to be not divisible by 2
def NotDivisibleByTwo (n : ℤ) : Prop := ¬ (∃ k : ℤ, n = 2 * k)

-- State the theorem
theorem not_divisible_by_two_2013 :
  (∀ n : ℤ, IsOdd n → NotDivisibleByTwo n) →
  IsOdd 2013 →
  NotDivisibleByTwo 2013 := by sorry

end not_divisible_by_two_2013_l3912_391202


namespace max_sum_rational_l3912_391265

theorem max_sum_rational (x y : ℚ) : 
  x > 0 ∧ y > 0 ∧ 
  (∃ a b c d : ℕ, x = a / c ∧ y = b / d ∧ 
    a + b = 9 ∧ c + d = 10 ∧
    ∀ m n : ℕ, m * c = n * a → m = c ∧ n = a ∧
    ∀ m n : ℕ, m * d = n * b → m = d ∧ n = b) →
  x + y ≤ 73 / 9 :=
by sorry

end max_sum_rational_l3912_391265


namespace lisa_photos_contradiction_l3912_391260

theorem lisa_photos_contradiction (animal_photos : ℕ) (flower_photos : ℕ) 
  (scenery_photos : ℕ) (abstract_photos : ℕ) :
  animal_photos = 20 ∧
  flower_photos = (3/2 : ℚ) * animal_photos ∧
  scenery_photos + abstract_photos = (2/5 : ℚ) * (animal_photos + flower_photos) ∧
  3 * abstract_photos = 2 * scenery_photos →
  ¬(80 ≤ animal_photos + flower_photos + scenery_photos + abstract_photos ∧
    animal_photos + flower_photos + scenery_photos + abstract_photos ≤ 100) :=
by sorry

end lisa_photos_contradiction_l3912_391260


namespace right_triangle_sides_l3912_391208

-- Define the triangle
structure RightTriangle where
  a : ℝ  -- first leg
  b : ℝ  -- second leg
  c : ℝ  -- hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Define the circumscribed and inscribed circle radii
def circumradius : ℝ := 15
def inradius : ℝ := 6

-- Theorem statement
theorem right_triangle_sides : ∃ (t : RightTriangle),
  t.c = 2 * circumradius ∧
  inradius = (t.a + t.b - t.c) / 2 ∧
  ((t.a = 18 ∧ t.b = 24) ∨ (t.a = 24 ∧ t.b = 18)) ∧
  t.c = 30 :=
sorry

end right_triangle_sides_l3912_391208


namespace small_square_side_length_wire_cut_lengths_l3912_391205

/-- The total length of the wire in centimeters -/
def total_wire_length : ℝ := 64

/-- Theorem for the first part of the problem -/
theorem small_square_side_length
  (small_side : ℝ)
  (large_side : ℝ)
  (h1 : small_side > 0)
  (h2 : large_side > 0)
  (h3 : 4 * small_side + 4 * large_side = total_wire_length)
  (h4 : large_side^2 = 2.25 * small_side^2) :
  small_side = 6.4 := by sorry

/-- Theorem for the second part of the problem -/
theorem wire_cut_lengths
  (small_side : ℝ)
  (large_side : ℝ)
  (h1 : small_side > 0)
  (h2 : large_side > 0)
  (h3 : 4 * small_side + 4 * large_side = total_wire_length)
  (h4 : small_side^2 + large_side^2 = 160) :
  (4 * small_side = 16 ∧ 4 * large_side = 48) ∨
  (4 * small_side = 48 ∧ 4 * large_side = 16) := by sorry

end small_square_side_length_wire_cut_lengths_l3912_391205


namespace cars_meeting_time_l3912_391289

/-- Two cars meeting on a highway -/
theorem cars_meeting_time 
  (highway_length : ℝ) 
  (car1_speed : ℝ) 
  (car2_speed : ℝ) 
  (h1 : highway_length = 45) 
  (h2 : car1_speed = 14) 
  (h3 : car2_speed = 16) : 
  (highway_length / (car1_speed + car2_speed)) = 1.5 := by
  sorry

end cars_meeting_time_l3912_391289


namespace sum_has_five_digits_l3912_391232

theorem sum_has_five_digits (A B : ℕ) (hA : A ≠ 0 ∧ A < 10) (hB : B ≠ 0 ∧ B < 10) :
  ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n = 9876 + (100 * A + 32) + (10 * B + 1) := by
  sorry

end sum_has_five_digits_l3912_391232


namespace inequality_proof_l3912_391291

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) ≤ 1 ∧
  ((a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) = 1 ↔ a = b ∧ b = c) :=
by sorry

end inequality_proof_l3912_391291


namespace remaining_work_for_x_l3912_391258

/-- The number of days x needs to finish the remaining work after y worked for 5 days --/
def remaining_days_for_x (x_days y_days : ℚ) : ℚ :=
  (1 - 5 / y_days) * x_days

theorem remaining_work_for_x :
  remaining_days_for_x 21 15 = 14 := by
  sorry

end remaining_work_for_x_l3912_391258


namespace four_solutions_l3912_391230

/-- The number of solutions to the equation 4/m + 2/n = 1 where m and n are positive integers -/
def num_solutions : ℕ := 4

/-- A function that checks if a pair of positive integers satisfies the equation 4/m + 2/n = 1 -/
def satisfies_equation (m n : ℕ+) : Prop :=
  (4 : ℚ) / m.val + (2 : ℚ) / n.val = 1

/-- The theorem stating that there are exactly 4 solutions to the equation -/
theorem four_solutions :
  ∃! (solutions : Finset (ℕ+ × ℕ+)),
    solutions.card = num_solutions ∧
    ∀ (pair : ℕ+ × ℕ+), pair ∈ solutions ↔ satisfies_equation pair.1 pair.2 :=
sorry

end four_solutions_l3912_391230


namespace right_triangle_hypotenuse_l3912_391210

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 15 → 
    b = 36 → 
    c^2 = a^2 + b^2 → 
    c = 39 :=
by
  sorry

end right_triangle_hypotenuse_l3912_391210


namespace sum_of_reciprocals_l3912_391284

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 4 * x * y) :
  1 / x + 1 / y = 4 := by
sorry

end sum_of_reciprocals_l3912_391284


namespace sum_of_median_scores_l3912_391207

def median_score (scores : List ℕ) : ℕ := sorry

theorem sum_of_median_scores (scores_A scores_B : List ℕ) 
  (h1 : scores_A.length = 9)
  (h2 : scores_B.length = 9)
  (h3 : median_score scores_A = 28)
  (h4 : median_score scores_B = 36) :
  median_score scores_A + median_score scores_B = 64 := by sorry

end sum_of_median_scores_l3912_391207


namespace shirts_washed_l3912_391279

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (unwashed : ℕ) 
  (h1 : short_sleeve = 39)
  (h2 : long_sleeve = 47)
  (h3 : unwashed = 66) :
  short_sleeve + long_sleeve - unwashed = 20 := by
  sorry

end shirts_washed_l3912_391279


namespace probability_purple_ten_sided_die_l3912_391231

/-- A die with a specific number of sides and purple faces -/
structure Die :=
  (sides : ℕ)
  (purple_faces : ℕ)

/-- The probability of rolling a purple face on a given die -/
def probability_purple (d : Die) : ℚ :=
  d.purple_faces / d.sides

/-- Theorem: The probability of rolling a purple face on a 10-sided die with 3 purple faces is 3/10 -/
theorem probability_purple_ten_sided_die :
  let d : Die := ⟨10, 3⟩
  probability_purple d = 3 / 10 := by
  sorry

end probability_purple_ten_sided_die_l3912_391231


namespace price_after_discount_l3912_391283

/-- 
Theorem: If an article's price after a 50% decrease is 1200 (in some currency unit), 
then its original price was 2400 (in the same currency unit).
-/
theorem price_after_discount (price_after : ℝ) (discount_percent : ℝ) (original_price : ℝ) : 
  price_after = 1200 ∧ discount_percent = 50 → original_price = 2400 :=
by sorry

end price_after_discount_l3912_391283


namespace curve_transformation_l3912_391239

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y = (1/3) * Real.cos (2 * x)

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop := x' = 2 * x ∧ y' = 3 * y

-- State the theorem
theorem curve_transformation (x y x' y' : ℝ) :
  original_curve x y → transformation x y x' y' → y' = Real.cos x' := by
  sorry

end curve_transformation_l3912_391239


namespace largest_constant_inequality_l3912_391217

theorem largest_constant_inequality (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) ∧
  ∀ C > 3, ∃ y₁ y₂ y₃ y₄ y₅ y₆ : ℝ, (y₁ + y₂ + y₃ + y₄ + y₅ + y₆)^2 < C * (y₁*(y₂ + y₃) + y₂*(y₃ + y₄) + y₃*(y₄ + y₅) + y₄*(y₅ + y₆) + y₅*(y₆ + y₁) + y₆*(y₁ + y₂)) :=
by sorry

end largest_constant_inequality_l3912_391217


namespace door_open_probability_l3912_391241

def num_keys : ℕ := 5

def probability_open_on_third_attempt : ℚ := 1 / 5

theorem door_open_probability :
  probability_open_on_third_attempt = 0.2 := by sorry

end door_open_probability_l3912_391241


namespace simplify_T_l3912_391222

theorem simplify_T (x : ℝ) : 
  (x + 2)^6 + 6*(x + 2)^5 + 15*(x + 2)^4 + 20*(x + 2)^3 + 15*(x + 2)^2 + 6*(x + 2) + 1 = (x + 3)^6 := by
  sorry

end simplify_T_l3912_391222


namespace profit_percent_calculation_l3912_391294

/-- Calculate the profit percent when buying 120 pens at the price of 95 pens and selling with a 2.5% discount -/
theorem profit_percent_calculation (marked_price : ℝ) (h_pos : marked_price > 0) : 
  let cost_price := 95 * marked_price
  let selling_price_per_pen := marked_price * (1 - 0.025)
  let total_selling_price := 120 * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  ∃ ε > 0, abs (profit_percent - 23.16) < ε :=
by sorry


end profit_percent_calculation_l3912_391294


namespace value_of_b_l3912_391297

theorem value_of_b (m a b d : ℝ) (h : m = (d * a * b) / (a + b)) :
  b = (m * a) / (d * a - m) := by sorry

end value_of_b_l3912_391297


namespace coordinates_of_point_A_l3912_391299

def point_A (a : ℝ) : ℝ × ℝ := (a - 1, 3 * a - 2)

theorem coordinates_of_point_A :
  ∀ a : ℝ, (point_A a).1 = (point_A a).2 + 3 → point_A a = (-2, -5) := by
  sorry

end coordinates_of_point_A_l3912_391299


namespace factorization_am2_minus_an2_l3912_391213

theorem factorization_am2_minus_an2 (a m n : ℝ) : a * m^2 - a * n^2 = a * (m + n) * (m - n) := by
  sorry

end factorization_am2_minus_an2_l3912_391213


namespace arithmetic_mean_of_integers_arithmetic_mean_of_52_integers_from_2_l3912_391264

theorem arithmetic_mean_of_integers (n : ℕ) (start : ℕ) :
  let seq := fun i => start + i - 1
  let sum := (n * (2 * start + n - 1)) / 2
  n ≠ 0 → sum / n = (2 * start + n - 1) / 2 := by
  sorry

theorem arithmetic_mean_of_52_integers_from_2 :
  let n := 52
  let start := 2
  let seq := fun i => start + i - 1
  let sum := (n * (2 * start + n - 1)) / 2
  sum / n = 27.5 := by
  sorry

end arithmetic_mean_of_integers_arithmetic_mean_of_52_integers_from_2_l3912_391264


namespace min_value_of_f_on_interval_l3912_391234

def f (x : ℝ) := -x^2 + 4*x + 5

theorem min_value_of_f_on_interval : 
  ∃ (x : ℝ), x ∈ Set.Icc 1 4 ∧ f x = 5 ∧ ∀ (y : ℝ), y ∈ Set.Icc 1 4 → f y ≥ f x := by
  sorry

end min_value_of_f_on_interval_l3912_391234


namespace largest_three_digit_sum_22_l3912_391242

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.Nodup digits

theorem largest_three_digit_sum_22 :
  ∃ (n : ℕ), is_three_digit n ∧ 
             has_distinct_digits n ∧ 
             sum_of_digits n = 22 ∧
             ∀ (m : ℕ), is_three_digit m → 
                        has_distinct_digits m → 
                        sum_of_digits m = 22 → 
                        m ≤ n :=
by sorry

end largest_three_digit_sum_22_l3912_391242


namespace remainder_problem_l3912_391288

theorem remainder_problem (x : ℤ) :
  x % 3 = 2 → x % 4 = 1 → x % 12 = 5 := by
  sorry

end remainder_problem_l3912_391288


namespace quadratic_equation_solution_l3912_391274

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end quadratic_equation_solution_l3912_391274


namespace angle_equality_l3912_391212

theorem angle_equality (α β : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2)
  (h3 : 0 < β) (h4 : β < Real.pi / 2)
  (h5 : Real.cos α + Real.cos β - Real.cos (α + β) = 3/2) :
  α = Real.pi / 3 ∧ β = Real.pi / 3 := by
sorry

end angle_equality_l3912_391212


namespace ninth_root_unity_product_l3912_391273

theorem ninth_root_unity_product : 
  let x : ℂ := Complex.exp (2 * π * I / 9)
  (3 * x + x^2) * (3 * x^3 + x^6) * (3 * x^4 + x^8) = 19 := by
  sorry

end ninth_root_unity_product_l3912_391273


namespace root_sum_fraction_l3912_391206

theorem root_sum_fraction (r₁ r₂ r₃ r₄ : ℂ) : 
  (r₁ * r₁ + r₂ * r₂ + r₃ * r₃ + r₄ * r₄ = 0) →
  (r₁ + r₂ + r₃ + r₄ = 4) →
  (r₁ * r₂ + r₁ * r₃ + r₁ * r₄ + r₂ * r₃ + r₂ * r₄ + r₃ * r₄ = 8) →
  (r₁^4 - 4*r₁^3 + 8*r₁^2 - 7*r₁ + 3 = 0) →
  (r₂^4 - 4*r₂^3 + 8*r₂^2 - 7*r₂ + 3 = 0) →
  (r₃^4 - 4*r₃^3 + 8*r₃^2 - 7*r₃ + 3 = 0) →
  (r₄^4 - 4*r₄^3 + 8*r₄^2 - 7*r₄ + 3 = 0) →
  (r₁^2 / (r₂^2 + r₃^2 + r₄^2) + r₂^2 / (r₁^2 + r₃^2 + r₄^2) + 
   r₃^2 / (r₁^2 + r₂^2 + r₄^2) + r₄^2 / (r₁^2 + r₂^2 + r₃^2) = -4) := by
sorry

end root_sum_fraction_l3912_391206


namespace grasshopper_jumps_l3912_391215

theorem grasshopper_jumps : ∃ (x y : ℕ), 80 * x - 50 * y = 170 ∧ x + y ≤ 7 := by
  sorry

end grasshopper_jumps_l3912_391215


namespace largest_valid_code_l3912_391295

def is_power_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5^k

def is_power_of_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def digits_to_nat (a b c d e : ℕ) : ℕ := a * 10000 + b * 1000 + c * 100 + d * 10 + e

def is_valid_code (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ,
    n = digits_to_nat a b c d e ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    is_power_of_5 (a * 10 + b) ∧
    is_power_of_2 (d * 10 + e) ∧
    ∃ k : ℕ, c = 3 * k ∧
    (a + b + c + d + e) % 2 = 1

theorem largest_valid_code :
  ∀ n : ℕ, is_valid_code n → n ≤ 25916 :=
sorry

end largest_valid_code_l3912_391295


namespace trouser_price_decrease_l3912_391252

theorem trouser_price_decrease (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 30) : 
  (original_price - sale_price) / original_price * 100 = 70 := by
  sorry

end trouser_price_decrease_l3912_391252


namespace number_of_juniors_l3912_391216

/-- Represents the number of students in a school program -/
def total_students : ℕ := 40

/-- Represents the ratio of juniors on the debate team -/
def junior_debate_ratio : ℚ := 3/10

/-- Represents the ratio of seniors on the debate team -/
def senior_debate_ratio : ℚ := 1/5

/-- Represents the ratio of juniors in the science club -/
def junior_science_ratio : ℚ := 2/5

/-- Represents the ratio of seniors in the science club -/
def senior_science_ratio : ℚ := 1/4

/-- Theorem stating that the number of juniors in the program is 16 -/
theorem number_of_juniors :
  ∃ (juniors seniors : ℕ),
    juniors + seniors = total_students ∧
    (junior_debate_ratio * juniors : ℚ) = (senior_debate_ratio * seniors : ℚ) ∧
    juniors = 16 :=
by sorry

end number_of_juniors_l3912_391216


namespace soda_difference_l3912_391240

def regular_soda : ℕ := 67
def diet_soda : ℕ := 9

theorem soda_difference : regular_soda - diet_soda = 58 := by
  sorry

end soda_difference_l3912_391240


namespace pitcher_problem_l3912_391292

theorem pitcher_problem (C : ℝ) (h : C > 0) :
  let juice_volume := (2 / 3) * C
  let num_cups := 6
  let cup_volume := juice_volume / num_cups
  cup_volume / C = 1 / 9 := by
  sorry

end pitcher_problem_l3912_391292


namespace initial_solution_volume_l3912_391254

/-- Given an initial solution with 42% alcohol, prove that its volume is 11 litres
    when 3 litres of water is added, resulting in a new mixture with 33% alcohol. -/
theorem initial_solution_volume (initial_percentage : Real) (added_water : Real) (final_percentage : Real) :
  initial_percentage = 0.42 →
  added_water = 3 →
  final_percentage = 0.33 →
  ∃ (initial_volume : Real),
    initial_volume * initial_percentage = (initial_volume + added_water) * final_percentage ∧
    initial_volume = 11 := by
  sorry

end initial_solution_volume_l3912_391254


namespace lucas_sixth_test_score_l3912_391278

def lucas_scores : List ℕ := [85, 90, 78, 88, 96]
def desired_mean : ℕ := 88
def num_tests : ℕ := 6

theorem lucas_sixth_test_score :
  ∃ (sixth_score : ℕ),
    (lucas_scores.sum + sixth_score) / num_tests = desired_mean ∧
    sixth_score = 91 := by
  sorry

end lucas_sixth_test_score_l3912_391278


namespace shifted_function_passes_through_origin_l3912_391223

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Represents a vertical shift of a function -/
structure VerticalShift where
  shift : ℝ

/-- Checks if a linear function passes through the origin -/
def passes_through_origin (f : LinearFunction) : Prop :=
  f.slope * 0 + f.intercept = 0

/-- Applies a vertical shift to a linear function -/
def apply_shift (f : LinearFunction) (s : VerticalShift) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept - s.shift }

/-- The original linear function y = 3x + 5 -/
def original_function : LinearFunction :=
  { slope := 3, intercept := 5 }

/-- The vertical shift of 5 units down -/
def shift_down : VerticalShift :=
  { shift := 5 }

theorem shifted_function_passes_through_origin :
  passes_through_origin (apply_shift original_function shift_down) := by
  sorry

end shifted_function_passes_through_origin_l3912_391223


namespace geometric_arithmetic_sequence_sum_l3912_391248

theorem geometric_arithmetic_sequence_sum (x y z : ℝ) 
  (h1 : (4*y)^2 = (3*x)*(5*z))  -- Geometric sequence condition
  (h2 : 2/y = 1/x + 1/z)        -- Arithmetic sequence condition
  : x/z + z/x = 34/15 := by
  sorry

end geometric_arithmetic_sequence_sum_l3912_391248


namespace problem_statement_l3912_391287

theorem problem_statement (a b : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x + a) * (x + b) * (x + 10) = 0 ∧
    (y + a) * (y + b) * (y + 10) = 0 ∧
    (z + a) * (z + b) * (z + 10) = 0 ∧
    x ≠ -4 ∧ y ≠ -4 ∧ z ≠ -4) →
  (∃! w : ℝ, (w + 2*a) * (w + 5) * (w + 8) = 0 ∧ 
    w ≠ -b ∧ w ≠ -10) →
  100 * a + b = 258 := by
sorry

end problem_statement_l3912_391287


namespace sqrt_fraction_sum_equals_sqrt_481_over_12_l3912_391267

theorem sqrt_fraction_sum_equals_sqrt_481_over_12 :
  Real.sqrt (9 / 16 + 25 / 9) = Real.sqrt 481 / 12 := by
  sorry

end sqrt_fraction_sum_equals_sqrt_481_over_12_l3912_391267


namespace train_length_l3912_391253

/-- Given a train that can cross an electric pole in 10 seconds at a speed of 180 km/h,
    prove that its length is 500 meters. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 180 →
  time_s = 10 →
  length_m = speed_kmh * (1000 / 3600) * time_s →
  length_m = 500 := by
  sorry

end train_length_l3912_391253


namespace quadratic_inequality_solution_l3912_391235

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 9 * x + 6 < 0 ↔ -2/3 < x ∧ x < 3 := by sorry

end quadratic_inequality_solution_l3912_391235


namespace andrews_eggs_l3912_391255

theorem andrews_eggs (total_needed : ℕ) (still_to_buy : ℕ) 
  (h1 : total_needed = 222) 
  (h2 : still_to_buy = 67) : 
  total_needed - still_to_buy = 155 := by
sorry

end andrews_eggs_l3912_391255


namespace negative_three_less_than_negative_two_l3912_391290

theorem negative_three_less_than_negative_two : -3 < -2 := by
  sorry

end negative_three_less_than_negative_two_l3912_391290


namespace yoo_seung_marbles_yoo_seung_marbles_proof_l3912_391270

/-- Proves that Yoo Seung has 108 marbles given the conditions in the problem -/
theorem yoo_seung_marbles : ℕ → ℕ → ℕ → Prop :=
  fun young_soo han_sol yoo_seung =>
    han_sol = young_soo + 15 ∧
    yoo_seung = 3 * han_sol ∧
    young_soo + han_sol + yoo_seung = 165 →
    yoo_seung = 108

/-- Proof of the theorem -/
theorem yoo_seung_marbles_proof : ∃ (young_soo han_sol yoo_seung : ℕ),
  yoo_seung_marbles young_soo han_sol yoo_seung :=
by
  sorry

end yoo_seung_marbles_yoo_seung_marbles_proof_l3912_391270


namespace tourist_cookie_problem_l3912_391262

theorem tourist_cookie_problem :
  ∃ (n : ℕ) (k : ℕ+), 
    (2 * n ≡ 1 [MOD k]) ∧ 
    (3 * n ≡ 13 [MOD k]) → 
    k = 23 := by
  sorry

end tourist_cookie_problem_l3912_391262


namespace complex_imaginary_part_l3912_391244

theorem complex_imaginary_part (a : ℝ) :
  let z : ℂ := (1 - a * Complex.I) / (1 + Complex.I)
  (z.re = -1) → (z.im = -2) := by
  sorry

end complex_imaginary_part_l3912_391244


namespace vector_subtraction_proof_l3912_391211

def a : ℝ × ℝ × ℝ := (5, -3, 2)
def b : ℝ × ℝ × ℝ := (-1, 4, -2)

theorem vector_subtraction_proof :
  a - 4 • b = (9, -19, 10) := by sorry

end vector_subtraction_proof_l3912_391211


namespace rita_jackets_l3912_391214

def problem (num_dresses num_pants jacket_cost dress_cost pants_cost transport_cost initial_amount remaining_amount : ℕ) : Prop :=
  let total_spent := initial_amount - remaining_amount
  let dress_pants_cost := num_dresses * dress_cost + num_pants * pants_cost
  let jacket_total_cost := total_spent - dress_pants_cost - transport_cost
  jacket_total_cost / jacket_cost = 4

theorem rita_jackets : 
  problem 5 3 30 20 12 5 400 139 := by sorry

end rita_jackets_l3912_391214


namespace multiplication_puzzle_l3912_391261

theorem multiplication_puzzle (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → (30 + a) * (10 * b + 4) = 142 → a + b = 4 := by
  sorry

end multiplication_puzzle_l3912_391261


namespace sequence_matches_first_10_terms_l3912_391296

/-- The sequence defined by a(n) = n(n-1) -/
def a (n : ℕ) : ℕ := n * (n - 1)

/-- The first 10 terms of the sequence -/
def first_10_terms : List ℕ := [0, 2, 6, 12, 20, 30, 42, 56, 72, 90]

theorem sequence_matches_first_10_terms :
  (List.range 10).map (fun i => a (i + 1)) = first_10_terms := by sorry

end sequence_matches_first_10_terms_l3912_391296


namespace min_value_quadratic_l3912_391224

/-- The function f(x) = 3x^2 - 18x + 7 attains its minimum value when x = 3 -/
theorem min_value_quadratic (x : ℝ) : 
  ∃ (min : ℝ), (∀ y : ℝ, 3 * x^2 - 18 * x + 7 ≥ 3 * y^2 - 18 * y + 7) ↔ x = 3 :=
by sorry

end min_value_quadratic_l3912_391224


namespace eleven_sided_polygon_equilateral_triangles_l3912_391277

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Counts the number of distinct equilateral triangles for a given regular polygon -/
def countDistinctEquilateralTriangles (n : ℕ) (polygon : RegularPolygon n) : ℕ :=
  sorry

theorem eleven_sided_polygon_equilateral_triangles :
  ∀ (polygon : RegularPolygon 11),
  countDistinctEquilateralTriangles 11 polygon = 88 :=
by sorry

end eleven_sided_polygon_equilateral_triangles_l3912_391277


namespace max_value_problem_1_max_value_problem_2_min_value_problem_3_l3912_391269

-- Problem 1
theorem max_value_problem_1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  (1/2) * x * (1 - 2*x) ≤ 1/16 :=
sorry

-- Problem 2
theorem max_value_problem_2 (x : ℝ) (h : x < 3) :
  4 / (x - 3) + x ≤ -1 :=
sorry

-- Problem 3
theorem min_value_problem_3 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 4) :
  1/x + 3/y ≥ 1 + Real.sqrt 3 / 2 :=
sorry

end max_value_problem_1_max_value_problem_2_min_value_problem_3_l3912_391269
