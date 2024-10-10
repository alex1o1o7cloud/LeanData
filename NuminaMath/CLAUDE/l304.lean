import Mathlib

namespace opposite_of_2023_l304_30410

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ → Prop :=
  λ b => a + b = 0

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 (-2023) := by
  sorry

end opposite_of_2023_l304_30410


namespace arithmetic_geometric_sequence_l304_30459

/-- An arithmetic sequence with common difference 3 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 3

/-- The first, third, and fourth terms form a geometric sequence -/
def geometric_subsequence (a : ℕ → ℝ) : Prop :=
  (a 3)^2 = a 1 * a 4

/-- Main theorem: If a is an arithmetic sequence with common difference 3
    and its first, third, and fourth terms form a geometric sequence,
    then the second term equals -9 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
    (h1 : arithmetic_sequence a) (h2 : geometric_subsequence a) : 
  a 2 = -9 := by
  sorry

end arithmetic_geometric_sequence_l304_30459


namespace complex_power_difference_l304_30408

theorem complex_power_difference (i : ℂ) : i^2 = -1 → i^123 - i^45 = -2*i := by
  sorry

end complex_power_difference_l304_30408


namespace cost_price_percentage_l304_30431

/-- Proves that given a discount of 12% and a gain percent of 37.5%, 
    the cost price is approximately 64% of the marked price. -/
theorem cost_price_percentage (marked_price : ℝ) (cost_price : ℝ) 
  (h1 : marked_price > 0) 
  (h2 : cost_price > 0) 
  (h3 : (marked_price - cost_price) / cost_price = 0.375) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |cost_price / marked_price - 0.64| < ε :=
sorry

end cost_price_percentage_l304_30431


namespace circle_contains_origin_l304_30412

theorem circle_contains_origin
  (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ a b c : ℝ)
  (h₁ : x₁ > 0) (h₂ : y₁ > 0)
  (h₃ : x₂ < 0) (h₄ : y₂ > 0)
  (h₅ : x₃ < 0) (h₆ : y₃ < 0)
  (h₇ : x₄ > 0) (h₈ : y₄ < 0)
  (h₉ : (x₁ - a)^2 + (y₁ - b)^2 ≤ c^2)
  (h₁₀ : (x₂ - a)^2 + (y₂ - b)^2 ≤ c^2)
  (h₁₁ : (x₃ - a)^2 + (y₃ - b)^2 ≤ c^2)
  (h₁₂ : (x₄ - a)^2 + (y₄ - b)^2 ≤ c^2)
  (h₁₃ : c > 0) :
  a^2 + b^2 < c^2 :=
by sorry

end circle_contains_origin_l304_30412


namespace function_analysis_l304_30498

/-- Given a real number a and a function f(x) = x²(x-a), this theorem proves:
    (I) If f'(1) = 3, then a = 0 and the equation of the tangent line at (1, f(1)) is 3x - y - 2 = 0
    (II) The maximum value of f(x) in the interval [0, 2] is max{8 - 4a, 0} for a < 3 and 0 for a ≥ 3 -/
theorem function_analysis (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 * (x - a)) :
  (deriv f 1 = 3 → a = 0 ∧ ∀ x y, 3*x - y - 2 = 0 ↔ y = f x ∧ x = 1) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≤ max (8 - 4*a) 0 ∧ (a ≥ 3 → f x ≤ 0)) :=
sorry

end function_analysis_l304_30498


namespace quadratic_inequality_solution_l304_30494

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop := x^2 - 2*x + a < 0

-- Define the solution set
def solution_set (t : ℝ) (x : ℝ) : Prop := -2 < x ∧ x < t

-- Define the second inequality
def second_inequality (c a : ℝ) (x : ℝ) : Prop := (c+a)*x^2 + 2*(c+a)*x - 1 < 0

theorem quadratic_inequality_solution :
  ∃ (a t : ℝ),
    (∀ x, quadratic_inequality a x ↔ solution_set t x) ∧
    a = -8 ∧
    t = 4 ∧
    ∀ c, (∀ x, second_inequality c a x) ↔ (7 < c ∧ c ≤ 8) :=
sorry

end quadratic_inequality_solution_l304_30494


namespace intersection_and_subset_l304_30481

def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | 6*x^2 - 5*x + 1 ≥ 0}
def C (m : ℝ) : Set ℝ := {x | (x - m)*(x - m - 9) < 0}

theorem intersection_and_subset :
  (A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1/3 ∨ 1/2 ≤ x ∧ x < 6}) ∧
  (∀ m : ℝ, A ⊆ C m ↔ -3 ≤ m ∧ m ≤ -1) := by
  sorry

end intersection_and_subset_l304_30481


namespace crayons_per_box_l304_30462

theorem crayons_per_box (total_boxes : ℕ) (crayons_to_mae : ℕ) (extra_crayons_to_lea : ℕ) (crayons_left : ℕ) :
  total_boxes = 4 →
  crayons_to_mae = 5 →
  extra_crayons_to_lea = 7 →
  crayons_left = 15 →
  ∃ (crayons_per_box : ℕ),
    crayons_per_box * total_boxes = crayons_to_mae + (crayons_to_mae + extra_crayons_to_lea) + crayons_left ∧
    crayons_per_box = 8 :=
by sorry

end crayons_per_box_l304_30462


namespace one_fourth_in_five_eighths_l304_30475

theorem one_fourth_in_five_eighths : (5 / 8 : ℚ) / (1 / 4 : ℚ) = 5 / 2 := by
  sorry

end one_fourth_in_five_eighths_l304_30475


namespace meaningful_expression_l304_30491

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 3)) ↔ x > 3 := by
sorry

end meaningful_expression_l304_30491


namespace all_expressions_zero_l304_30419

-- Define the vector space
variable {V : Type*} [AddCommGroup V]

-- Define points in the vector space
variable (A B C D O N M P Q : V)

-- Define the expressions
def expr1 (A B C : V) : V := (B - A) + (C - B) + (A - C)
def expr2 (A B C D : V) : V := (B - A) - (C - A) + (D - B) - (D - C)
def expr3 (O A D : V) : V := (A - O) - (D - O) + (D - A)
def expr4 (N Q P M : V) : V := (Q - N) + (P - Q) + (N - M) - (P - M)

-- Theorem stating that all expressions result in the zero vector
theorem all_expressions_zero (A B C D O N M P Q : V) : 
  expr1 A B C = 0 ∧ 
  expr2 A B C D = 0 ∧ 
  expr3 O A D = 0 ∧ 
  expr4 N Q P M = 0 :=
sorry

end all_expressions_zero_l304_30419


namespace basement_pump_time_l304_30453

-- Define the constants
def basement_length : ℝ := 30
def basement_width : ℝ := 40
def water_depth : ℝ := 2
def initial_pumps : ℕ := 4
def pump_capacity : ℝ := 10
def breakdown_time : ℝ := 120
def cubic_foot_to_gallon : ℝ := 7.5

-- Define the theorem
theorem basement_pump_time :
  let initial_volume : ℝ := basement_length * basement_width * water_depth * cubic_foot_to_gallon
  let initial_pump_rate : ℝ := initial_pumps * pump_capacity
  let volume_pumped_before_breakdown : ℝ := initial_pump_rate * breakdown_time
  let remaining_volume : ℝ := initial_volume - volume_pumped_before_breakdown
  let remaining_pumps : ℕ := initial_pumps - 1
  let remaining_pump_rate : ℝ := remaining_pumps * pump_capacity
  let remaining_time : ℝ := remaining_volume / remaining_pump_rate
  breakdown_time + remaining_time = 560 := by
  sorry

end basement_pump_time_l304_30453


namespace hyperbola_eccentricity_l304_30485

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 (a > 0) and distance between foci equal to 10,
    its eccentricity is 5/4 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ c : ℝ, 2 * c = 10) →
  (∃ e : ℝ, e = 5/4 ∧ e = c/a) :=
by sorry

end hyperbola_eccentricity_l304_30485


namespace complex_equation_solution_l304_30473

theorem complex_equation_solution :
  ∃ (x : ℂ), (5 : ℂ) + 2 * Complex.I * x = (3 : ℂ) - 4 * Complex.I * x ∧ x = Complex.I / 3 := by
  sorry

end complex_equation_solution_l304_30473


namespace quadratic_expression_value_l304_30418

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 3*x = 2) : 3*x^2 - 9*x - 7 = -1 := by
  sorry

end quadratic_expression_value_l304_30418


namespace partition_theorem_l304_30401

def is_valid_partition (n : ℕ) : Prop :=
  ∃ (partition : List (Fin n × Fin n × Fin n)),
    (∀ (i j : Fin n), i ≠ j → (∃ (t : Fin n × Fin n × Fin n), t ∈ partition ∧ (i = t.1 ∨ i = t.2.1 ∨ i = t.2.2)) →
                              (∃ (t : Fin n × Fin n × Fin n), t ∈ partition ∧ (j = t.1 ∨ j = t.2.1 ∨ j = t.2.2)) →
                              (∀ (t : Fin n × Fin n × Fin n), t ∈ partition → (i = t.1 ∨ i = t.2.1 ∨ i = t.2.2) →
                                                                              (j ≠ t.1 ∧ j ≠ t.2.1 ∧ j ≠ t.2.2))) ∧
    (∀ (t : Fin n × Fin n × Fin n), t ∈ partition → t.1.val + t.2.1.val = t.2.2.val ∨
                                                    t.1.val + t.2.2.val = t.2.1.val ∨
                                                    t.2.1.val + t.2.2.val = t.1.val)

theorem partition_theorem (n : ℕ) (h : n ∈ Finset.range 10 ∪ {3900}) : 
  is_valid_partition n ↔ n = 3900 ∨ n = 3903 :=
sorry

end partition_theorem_l304_30401


namespace sum_of_cubes_of_roots_l304_30452

theorem sum_of_cubes_of_roots (x₁ x₂ x₃ : ℝ) : 
  (2 * x₁^3 + 3 * x₁^2 - 11 * x₁ + 6 = 0) →
  (2 * x₂^3 + 3 * x₂^2 - 11 * x₂ + 6 = 0) →
  (2 * x₃^3 + 3 * x₃^2 - 11 * x₃ + 6 = 0) →
  (x₁ + x₂ + x₃ = -3/2) →
  (x₁*x₂ + x₂*x₃ + x₃*x₁ = -11/2) →
  (x₁*x₂*x₃ = -3) →
  x₁^3 + x₂^3 + x₃^3 = -99/8 :=
by sorry

end sum_of_cubes_of_roots_l304_30452


namespace number_of_teams_is_twelve_l304_30479

/-- The number of teams in the baseball league --/
def n : ℕ := sorry

/-- The number of games each team plays against every other team --/
def games_per_pair : ℕ := 6

/-- The total number of games played in the league --/
def total_games : ℕ := 396

/-- Theorem stating that the number of teams in the league is 12 --/
theorem number_of_teams_is_twelve :
  (n * (n - 1) / 2) * games_per_pair = total_games ∧ n = 12 := by sorry

end number_of_teams_is_twelve_l304_30479


namespace fourth_graders_pizza_problem_l304_30477

theorem fourth_graders_pizza_problem :
  ∀ (n : ℕ),
  (∀ (student : ℕ), student ≤ n → 20 * 6 * student = 1200) →
  n = 10 :=
by
  sorry

end fourth_graders_pizza_problem_l304_30477


namespace triangle_longest_side_l304_30470

theorem triangle_longest_side (x y : ℝ) :
  10 + (2 * y + 3) + (3 * x + 2) = 45 →
  (10 > 0) ∧ (2 * y + 3 > 0) ∧ (3 * x + 2 > 0) →
  max 10 (max (2 * y + 3) (3 * x + 2)) ≤ 32 :=
by sorry

end triangle_longest_side_l304_30470


namespace sum_of_cubes_l304_30488

theorem sum_of_cubes (p q r : ℝ) 
  (sum_eq : p + q + r = 4)
  (sum_prod_eq : p * q + p * r + q * r = 3)
  (prod_eq : p * q * r = -6) :
  p^3 + q^3 + r^3 = 34 := by
  sorry

end sum_of_cubes_l304_30488


namespace cube_construction_count_l304_30415

/-- The number of distinguishable ways to construct a cube from colored squares -/
def distinguishable_cube_constructions : ℕ := 1260

/-- The number of faces on a cube -/
def cube_faces : ℕ := 6

/-- The number of colored squares available -/
def colored_squares : ℕ := 8

/-- The number of rotational symmetries when one face is fixed -/
def rotational_symmetries : ℕ := 4

theorem cube_construction_count :
  distinguishable_cube_constructions = (colored_squares - 1).factorial / rotational_symmetries :=
sorry

end cube_construction_count_l304_30415


namespace expand_expression_l304_30406

theorem expand_expression (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4*x + 4) = x^4 + 4*x^3 - 16*x - 16 := by
  sorry

end expand_expression_l304_30406


namespace range_of_3x_plus_y_l304_30455

theorem range_of_3x_plus_y (x y : ℝ) :
  3 * x^2 + y^2 ≤ 1 →
  ∃ (max min : ℝ), max = 2 ∧ min = -2 ∧
    (3 * x + y ≤ max ∧ 3 * x + y ≥ min) :=
by sorry

end range_of_3x_plus_y_l304_30455


namespace difference_of_squares_l304_30458

theorem difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 18) : a^2 - b^2 = 1080 := by
  sorry

end difference_of_squares_l304_30458


namespace alpha_value_l304_30439

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 2*β)).re > 0)
  (h3 : β = 3 + 2*Complex.I) :
  α = 6 - 2*Complex.I := by
sorry

end alpha_value_l304_30439


namespace simplify_expression_1_simplify_expression_2_l304_30478

-- Define variables
variable (a b x y : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : 2*a - (4*a + 5*b) + 2*(3*a - 4*b) = 4*a - 13*b := by
  sorry

-- Theorem for the second expression
theorem simplify_expression_2 : 5*x^2 - 2*(3*y^2 - 5*x^2) + (-4*y^2 + 7*x*y) = 15*x^2 - 10*y^2 + 7*x*y := by
  sorry

end simplify_expression_1_simplify_expression_2_l304_30478


namespace log_expression_equality_l304_30403

-- Define lg as base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_expression_equality :
  (Real.log (Real.sqrt 27) / Real.log 3) + lg 25 + lg 4 + 7^(Real.log 2 / Real.log 7) + (-9.8)^0 = 13/2 := by
  sorry

end log_expression_equality_l304_30403


namespace problem_solution_l304_30465

theorem problem_solution (x : ℝ) : (0.50 * x = 0.05 * 500 - 20) → x = 10 := by
  sorry

end problem_solution_l304_30465


namespace joe_paint_usage_l304_30456

/-- The amount of paint Joe used given the initial amount and usage fractions -/
def paint_used (initial : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) : ℚ :=
  let first_week := initial * first_week_fraction
  let remaining := initial - first_week
  let second_week := remaining * second_week_fraction
  first_week + second_week

/-- Theorem stating that Joe used 168 gallons of paint -/
theorem joe_paint_usage :
  paint_used 360 (1/3) (1/5) = 168 := by
  sorry

end joe_paint_usage_l304_30456


namespace min_sin6_2cos6_l304_30435

theorem min_sin6_2cos6 :
  ∀ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ 2/3 := by
  sorry

end min_sin6_2cos6_l304_30435


namespace vehicles_with_high_speed_l304_30434

theorem vehicles_with_high_speed (vehicles_80_to_89 vehicles_90_to_99 vehicles_100_to_109 : ℕ) :
  vehicles_80_to_89 = 15 →
  vehicles_90_to_99 = 30 →
  vehicles_100_to_109 = 5 →
  vehicles_80_to_89 + vehicles_90_to_99 + vehicles_100_to_109 = 50 :=
by sorry

end vehicles_with_high_speed_l304_30434


namespace green_chips_count_l304_30428

theorem green_chips_count (total : ℕ) (blue white green : ℕ) : 
  blue = 3 →
  blue = total / 10 →
  white = total / 2 →
  green = total - blue - white →
  green = 12 := by
  sorry

end green_chips_count_l304_30428


namespace decreasing_interval_of_f_l304_30432

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem decreasing_interval_of_f :
  ∀ x y : ℝ, x < y ∧ x < 1 ∧ y < 1 → f x > f y :=
by
  sorry

end decreasing_interval_of_f_l304_30432


namespace winnie_fell_behind_l304_30468

/-- The number of repetitions Winnie fell behind --/
def repetitions_fell_behind (yesterday_reps today_reps : ℕ) : ℕ :=
  yesterday_reps - today_reps

/-- Proof that Winnie fell behind by 13 repetitions --/
theorem winnie_fell_behind :
  repetitions_fell_behind 86 73 = 13 := by
  sorry

end winnie_fell_behind_l304_30468


namespace b_share_is_3315_l304_30444

/-- Calculates the share of a partner in a partnership based on investments and known share. -/
def calculate_share (investment_a investment_b investment_c share_a : ℚ) : ℚ :=
  (share_a * investment_b) / investment_a

/-- Theorem stating that given the investments and a's share, b's share is 3315. -/
theorem b_share_is_3315 (investment_a investment_b investment_c share_a : ℚ) 
  (h1 : investment_a = 11000)
  (h2 : investment_b = 15000)
  (h3 : investment_c = 23000)
  (h4 : share_a = 2431) :
  calculate_share investment_a investment_b investment_c share_a = 3315 := by
sorry

#eval calculate_share 11000 15000 23000 2431

end b_share_is_3315_l304_30444


namespace equilateral_pyramid_cross_section_l304_30446

/-- Represents a pyramid with an equilateral triangular base -/
structure EquilateralPyramid where
  /-- Side length of the base triangle -/
  base_side : ℝ
  /-- Height of the pyramid -/
  height : ℝ

/-- Represents a plane that intersects the pyramid -/
structure IntersectingPlane where
  /-- Angle between the plane and the base of the pyramid -/
  angle_with_base : ℝ

/-- Calculates the area of the cross-section of the pyramid -/
noncomputable def cross_section_area (p : EquilateralPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

theorem equilateral_pyramid_cross_section
  (p : EquilateralPyramid)
  (plane : IntersectingPlane) :
  p.base_side = 3 ∧
  p.height = Real.sqrt 3 ∧
  plane.angle_with_base = π / 3 →
  cross_section_area p plane = 11 * Real.sqrt 3 / 10 := by
    sorry

end equilateral_pyramid_cross_section_l304_30446


namespace smallest_three_digit_multiple_of_17_l304_30480

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end smallest_three_digit_multiple_of_17_l304_30480


namespace triangle_formation_conditions_l304_30438

theorem triangle_formation_conditions 
  (E F G H : ℝ × ℝ)  -- Points in 2D plane
  (a b c : ℝ)        -- Lengths
  (θ φ : ℝ)          -- Angles
  (h_distinct : E ≠ F ∧ F ≠ G ∧ G ≠ H)  -- Distinct points
  (h_collinear : ∃ (m k : ℝ), F.2 = m * F.1 + k ∧ G.2 = m * G.1 + k ∧ H.2 = m * H.1 + k)  -- Collinearity
  (h_order : E.1 < F.1 ∧ F.1 < G.1 ∧ G.1 < H.1)  -- Order on line
  (h_lengths : dist E F = a ∧ dist E G = b ∧ dist E H = c)  -- Segment lengths
  (h_rotation : ∃ (E' : ℝ × ℝ), 
    dist F E' = a ∧ 
    dist G H = c - b ∧
    E' = H)  -- Rotation result
  (h_triangle : ∃ (F' G' : ℝ × ℝ), 
    dist E' F' = a ∧ 
    dist F' G' > 0 ∧ 
    dist G' E' = c - b ∧
    (F'.1 - E'.1) * (G'.2 - E'.2) ≠ (G'.1 - E'.1) * (F'.2 - E'.2))  -- Non-degenerate triangle formed
  : a < c / 2 ∧ b < a + c * Real.cos φ ∧ b * Real.cos θ < c / 2 :=
by sorry

end triangle_formation_conditions_l304_30438


namespace contributions_before_johns_l304_30442

def average_before_johns (n : ℕ) : ℚ := 50

def johns_contribution : ℚ := 150

def new_average (n : ℕ) : ℚ := 75

def total_before_johns (n : ℕ) : ℚ := n * average_before_johns n

def total_after_johns (n : ℕ) : ℚ := total_before_johns n + johns_contribution

theorem contributions_before_johns :
  ∃ n : ℕ, 
    (new_average n = (3/2) * average_before_johns n) ∧
    (new_average n = 75) ∧
    (johns_contribution = 150) ∧
    (new_average n = total_after_johns n / (n + 1)) ∧
    (n = 3) :=
by sorry

end contributions_before_johns_l304_30442


namespace purely_imaginary_implies_m_equals_one_l304_30416

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number in question -/
def complex_number (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m + 2, m^2 - 2*m⟩

/-- Theorem stating that if the complex number is purely imaginary, then m = 1 -/
theorem purely_imaginary_implies_m_equals_one :
  ∀ m : ℝ, is_purely_imaginary (complex_number m) → m = 1 := by
  sorry

#check purely_imaginary_implies_m_equals_one

end purely_imaginary_implies_m_equals_one_l304_30416


namespace inequality_condition_l304_30474

theorem inequality_condition (a b : ℝ) : 
  (a > b → ((a + b) / 2)^2 > a * b) ∧ 
  (∃ a b : ℝ, ((a + b) / 2)^2 > a * b ∧ a ≤ b) :=
by sorry

end inequality_condition_l304_30474


namespace starters_count_l304_30492

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 5 starters from a team of 12 players,
    including a set of twins, with at most one of the twins in the starting lineup -/
def chooseStarters : ℕ :=
  choose 10 5 + 2 * choose 10 4

theorem starters_count : chooseStarters = 672 := by sorry

end starters_count_l304_30492


namespace evaluate_expression_l304_30426

theorem evaluate_expression : (3 : ℝ)^4 - 4 * (3 : ℝ)^2 = 45 := by
  sorry

end evaluate_expression_l304_30426


namespace range_of_a_l304_30454

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ y : ℝ, y ≥ a ∧ |y - 1| ≥ 1) → 
  a ∈ Set.Iic 0 :=
by sorry

end range_of_a_l304_30454


namespace equal_bills_at_120_minutes_l304_30460

/-- The base rate for United Telephone service in dollars -/
def united_base_rate : ℝ := 6

/-- The per-minute charge for United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The base rate for Atlantic Call service in dollars -/
def atlantic_base_rate : ℝ := 12

/-- The per-minute charge for Atlantic Call in dollars -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℝ := 120

theorem equal_bills_at_120_minutes :
  united_base_rate + united_per_minute * equal_minutes =
  atlantic_base_rate + atlantic_per_minute * equal_minutes :=
by sorry

end equal_bills_at_120_minutes_l304_30460


namespace total_floor_area_l304_30445

/-- The total floor area covered by square stone slabs -/
theorem total_floor_area (num_slabs : ℕ) (slab_length : ℝ) : 
  num_slabs = 30 → slab_length = 150 → 
  (num_slabs * (slab_length / 100)^2 : ℝ) = 67.5 := by
  sorry

end total_floor_area_l304_30445


namespace circle_equation_l304_30424

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x + 3)^2 + (y + 3)^2 = 18

-- Define the point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the circle we want to prove
def target_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Theorem statement
theorem circle_equation :
  (∀ x y : ℝ, target_circle x y ↔ 
    (((x, y) = point_A ∨ (x, y) = origin) ∧ 
     (∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
       ((x' - x)^2 + (y' - y)^2 < δ^2 → 
        (target_circle x' y' ↔ ¬given_circle x' y'))))) := 
by sorry

end circle_equation_l304_30424


namespace birthday_check_value_l304_30469

theorem birthday_check_value (initial_balance : ℝ) (check_value : ℝ) : 
  initial_balance = 150 →
  check_value = (1/4) * (initial_balance + check_value) →
  check_value = 50 := by
sorry

end birthday_check_value_l304_30469


namespace complement_A_intersect_B_A_subset_C_implies_a_geq_7_l304_30430

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Statement 1
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Statement 2
theorem A_subset_C_implies_a_geq_7 (a : ℝ) (h : A ⊆ C a) : a ≥ 7 := by sorry

end complement_A_intersect_B_A_subset_C_implies_a_geq_7_l304_30430


namespace complex_product_sum_l304_30400

theorem complex_product_sum (i : ℂ) : i * i = -1 →
  let z := (1 + i) * (1 - i)
  let p := z.re
  let q := z.im
  p + q = 2 := by
sorry

end complex_product_sum_l304_30400


namespace base_12_remainder_div_7_l304_30440

-- Define the base-12 number
def base_12_num : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3

-- Theorem statement
theorem base_12_remainder_div_7 : base_12_num % 7 = 6 := by
  sorry

end base_12_remainder_div_7_l304_30440


namespace f_properties_l304_30482

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + log x

theorem f_properties :
  (∀ m : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (exp 1) ∧ f 1 x₀ ≤ m) ↔ m ∈ Set.Ici (1/2)) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 1 → f a x < 2 * a * x) ↔ a ∈ Set.Icc (-1/2) (1/2)) :=
by sorry

end f_properties_l304_30482


namespace solve_for_y_l304_30417

theorem solve_for_y (x y : ℝ) (h1 : x + 2*y = 10) (h2 : x = 2) : y = 4 := by
  sorry

end solve_for_y_l304_30417


namespace minimum_ladder_rungs_l304_30433

theorem minimum_ladder_rungs (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  let n := a + b - Nat.gcd a b
  ∀ m : ℕ, m < n → ¬ (∃ (x y : ℤ), x ≥ 0 ∧ y ≥ 0 ∧ a * x - b * y = m) ∧
  ∃ (x y : ℤ), x ≥ 0 ∧ y ≥ 0 ∧ a * x - b * y = n :=
by sorry

end minimum_ladder_rungs_l304_30433


namespace yolandas_walking_rate_l304_30436

/-- Proves that Yolanda's walking rate is 3 miles per hour given the problem conditions -/
theorem yolandas_walking_rate 
  (total_distance : ℝ) 
  (bobs_delay : ℝ) 
  (bobs_rate : ℝ) 
  (bobs_distance : ℝ) 
  (h1 : total_distance = 24)
  (h2 : bobs_delay = 1)
  (h3 : bobs_rate = 4)
  (h4 : bobs_distance = 12) : 
  (total_distance - bobs_distance) / (bobs_distance / bobs_rate + bobs_delay) = 3 := by
  sorry

#check yolandas_walking_rate

end yolandas_walking_rate_l304_30436


namespace distance_origin_to_point_l304_30493

theorem distance_origin_to_point :
  let x : ℝ := 20
  let y : ℝ := 21
  Real.sqrt (x^2 + y^2) = 29 :=
by sorry

end distance_origin_to_point_l304_30493


namespace oplus_example_l304_30464

-- Define the ⊕ operation
def oplus (a b : ℕ) : ℕ := a + b + a * b

-- Statement to prove
theorem oplus_example : oplus (oplus 2 3) 4 = 59 := by
  sorry

end oplus_example_l304_30464


namespace center_on_line_common_chord_condition_external_tangent_length_l304_30425

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C_k (k x y : ℝ) : Prop := (x - k)^2 + (y - Real.sqrt 3 * k)^2 = 4

-- Define the line y = √3x
def line_sqrt3 (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Theorem 1: The center of circle C_k always lies on the line y = √3x
theorem center_on_line (k : ℝ) : line_sqrt3 k (Real.sqrt 3 * k) := by sorry

-- Theorem 2: If the common chord length is √15/2, then k = ±1 or k = ±3/4
theorem common_chord_condition (k : ℝ) : 
  (∃ x y : ℝ, circle_O x y ∧ circle_C_k k x y ∧ 
   (x^2 + y^2 = 1 - (15/16))) → 
  (k = 1 ∨ k = -1 ∨ k = 3/4 ∨ k = -3/4) := by sorry

-- Theorem 3: When k = ±3/2, the length of the external common tangent is 2√2
theorem external_tangent_length : 
  ∀ k : ℝ, (k = 3/2 ∨ k = -3/2) → 
  (∃ x1 y1 x2 y2 : ℝ, 
    circle_O x1 y1 ∧ circle_C_k k x2 y2 ∧
    ((x2 - x1)^2 + (y2 - y1)^2 = 8)) := by sorry

end center_on_line_common_chord_condition_external_tangent_length_l304_30425


namespace correlation_coefficient_relationship_l304_30437

/-- The correlation coefficient type -/
def CorrelationCoefficient := { r : ℝ // -1 ≤ r ∧ r ≤ 1 }

/-- The degree of correlation between two variables -/
noncomputable def degreeOfCorrelation (r : CorrelationCoefficient) : ℝ := sorry

/-- Theorem stating the relationship between |r| and the degree of correlation -/
theorem correlation_coefficient_relationship (r1 r2 : CorrelationCoefficient) :
  (|r1.val| < |r2.val| ∧ |r2.val| ≤ 1) → degreeOfCorrelation r1 < degreeOfCorrelation r2 := by sorry

end correlation_coefficient_relationship_l304_30437


namespace red_jellybeans_count_l304_30449

/-- The probability of drawing 3 blue jellybeans in a row without replacement -/
def probability : ℚ := 10526315789473684 / 100000000000000000

/-- The number of blue jellybeans in the bag -/
def blue_jellybeans : ℕ := 10

/-- Calculates the probability of drawing 3 blue jellybeans in a row without replacement -/
def calculate_probability (red : ℕ) : ℚ :=
  (blue_jellybeans : ℚ) / (blue_jellybeans + red) *
  ((blue_jellybeans - 1) : ℚ) / (blue_jellybeans + red - 1) *
  ((blue_jellybeans - 2) : ℚ) / (blue_jellybeans + red - 2)

/-- Theorem stating that the number of red jellybeans is 10 -/
theorem red_jellybeans_count : ∃ (red : ℕ), calculate_probability red = probability ∧ red = 10 := by
  sorry

end red_jellybeans_count_l304_30449


namespace parabola_shift_l304_30483

def original_function (x : ℝ) : ℝ := (x + 1)^2 + 3

def shift_right (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f (x - shift)

def shift_down (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f x - shift

def final_function (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem parabola_shift :
  (shift_down (shift_right original_function 2) 1) = final_function := by
  sorry

end parabola_shift_l304_30483


namespace xyz_value_l304_30405

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x*y + x*z + y*z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19)
  (h3 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = 11) :
  x * y * z = 26 / 3 := by
sorry

end xyz_value_l304_30405


namespace wheel_revolutions_l304_30466

/-- The number of revolutions of a wheel with diameter 10 feet to travel half a mile -/
theorem wheel_revolutions (π : ℝ) (h : π > 0) : 
  let diameter : ℝ := 10
  let circumference : ℝ := π * diameter
  let half_mile_in_feet : ℝ := 5280 / 2
  half_mile_in_feet / circumference = 264 / π := by
  sorry

end wheel_revolutions_l304_30466


namespace unique_valid_n_l304_30490

def is_valid_number (n : ℕ) (x : ℕ) : Prop :=
  (x.digits 10).length = n ∧
  (x.digits 10).count 7 = 1 ∧
  (x.digits 10).count 1 = n - 1

def all_numbers_prime (n : ℕ) : Prop :=
  ∀ x : ℕ, is_valid_number n x → Nat.Prime x

theorem unique_valid_n : 
  ∀ n : ℕ, (n > 0 ∧ all_numbers_prime n) ↔ (n = 1 ∨ n = 2) :=
sorry

end unique_valid_n_l304_30490


namespace complex_modulus_sqrt10_implies_a_plusminus2_l304_30414

theorem complex_modulus_sqrt10_implies_a_plusminus2 (a : ℝ) : 
  Complex.abs ((a + Complex.I) * (1 - Complex.I)) = Real.sqrt 10 → 
  a = 2 ∨ a = -2 := by
sorry

end complex_modulus_sqrt10_implies_a_plusminus2_l304_30414


namespace journey_speed_calculation_l304_30422

/-- Given a journey with the following parameters:
  * total_distance: The total distance of the journey in km
  * total_time: The total time of the journey in hours
  * first_half_speed: The speed for the first half of the journey in km/hr
  
  This theorem proves that the speed for the second half of the journey is equal to the second_half_speed. -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ)
  (h1 : total_distance = 672)
  (h2 : total_time = 30)
  (h3 : first_half_speed = 21)
  : ∃ second_half_speed : ℝ, second_half_speed = 24 := by
  sorry

end journey_speed_calculation_l304_30422


namespace work_completed_together_l304_30448

/-- The amount of work that can be completed by two workers in one day, given their individual work rates. -/
theorem work_completed_together 
  (days_A : ℝ) -- Number of days A takes to complete the work
  (days_B : ℝ) -- Number of days B takes to complete the work
  (h1 : days_A = 10) -- A can finish the work in 10 days
  (h2 : days_B = days_A / 2) -- B can do the same work in half the time taken by A
  : (1 / days_A + 1 / days_B) = 3 / 10 := by
  sorry

end work_completed_together_l304_30448


namespace rhombus_region_area_l304_30463

/-- Represents a rhombus ABCD -/
structure Rhombus where
  side_length : ℝ
  angle_B : ℝ

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem rhombus_region_area (r : Rhombus) 
  (h1 : r.side_length = 3)
  (h2 : r.angle_B = π / 2) :
  area (region_R r) = 9 * π / 16 :=
sorry

end rhombus_region_area_l304_30463


namespace fold_square_diagonal_l304_30443

/-- Given a square ABCD with side length 8 cm, where corner C is folded to point E
    (located 1/3 of the way from A to D on AD), prove that the length of FD is 32/9 cm,
    where F is the point where the fold intersects CD. -/
theorem fold_square_diagonal (A B C D E F G : ℝ × ℝ) : 
  let side_length : ℝ := 8
  -- ABCD is a square
  (A.1 = 0 ∧ A.2 = 0) →
  (B.1 = side_length ∧ B.2 = 0) →
  (C.1 = side_length ∧ C.2 = side_length) →
  (D.1 = 0 ∧ D.2 = side_length) →
  -- E is one-third of the way along AD
  (E.1 = 0 ∧ E.2 = side_length / 3) →
  -- F is on CD
  (F.1 = side_length ∧ F.2 ≥ 0 ∧ F.2 ≤ side_length) →
  -- C coincides with E after folding
  (dist C E = dist C F) →
  -- FD length
  dist F D = 32 / 9 := by
sorry

end fold_square_diagonal_l304_30443


namespace range_of_f_on_interval_solution_sets_f_positive_l304_30441

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1) * x + a

-- Part 1: Range of f(x) when a = 3 on [-1, 3]
theorem range_of_f_on_interval :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3, -1 ≤ f 3 x ∧ f 3 x ≤ 8 :=
sorry

-- Part 2: Solution sets for f(x) > 0
theorem solution_sets_f_positive (a : ℝ) :
  (∀ x, f a x > 0 ↔ 
    (a > 1 ∧ (x < 1 ∨ x > a)) ∨
    (a < 1 ∧ (x < a ∨ x > 1)) ∨
    (a = 1 ∧ x ≠ 1)) :=
sorry

end range_of_f_on_interval_solution_sets_f_positive_l304_30441


namespace digit_3000_is_1_l304_30489

/-- Represents the decimal expansion of integers from 1 to 1001 concatenated -/
def x : ℝ :=
  sorry

/-- Returns the nth digit after the decimal point in the given real number -/
def nthDigit (n : ℕ) (r : ℝ) : ℕ :=
  sorry

/-- The 3000th digit after the decimal point in x is 1 -/
theorem digit_3000_is_1 : nthDigit 3000 x = 1 := by
  sorry

end digit_3000_is_1_l304_30489


namespace solution_set_of_inequality_l304_30497

theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : ∀ x ∈ Set.Icc (-1) 1, f (-x) + f x = 0)
  (h2 : ∀ m n, m ∈ Set.Icc 0 1 → n ∈ Set.Icc 0 1 → m ≠ n → (f m - f n) / (m - n) < 0) :
  {x : ℝ | f (1 - 3*x) ≤ f (x - 1)} = Set.Icc 0 (1/2) := by
  sorry

end solution_set_of_inequality_l304_30497


namespace arithmetic_sequence_sum_ratio_l304_30413

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n
  (∀ n m, a n - a m = (n - m) * (a 2 - a 1)) →  -- Definition of arithmetic sequence
  a 8 / a 7 = 13 / 5 →  -- Given condition
  S 15 / S 13 = 3 := by
sorry

end arithmetic_sequence_sum_ratio_l304_30413


namespace complex_equation_solution_l304_30461

theorem complex_equation_solution (z : ℂ) (h : z + Complex.abs z = 2 + 8 * I) : z = -15 + 8 * I := by
  sorry

end complex_equation_solution_l304_30461


namespace triangle_perimeter_l304_30409

theorem triangle_perimeter (a b c : ℕ) (ha : a = 3) (hb : b = 8) (hc : Odd c) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c = 18 ∨ a + b + c = 20 := by
sorry

end triangle_perimeter_l304_30409


namespace f_plus_one_is_odd_l304_30476

-- Define the property of the function f
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ + 1

-- Define what it means for a function to be odd
def is_odd (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

-- Theorem statement
theorem f_plus_one_is_odd (f : ℝ → ℝ) (h : satisfies_property f) :
  is_odd (fun x => f x + 1) :=
sorry

end f_plus_one_is_odd_l304_30476


namespace monkey_climb_theorem_l304_30429

/-- The time taken for a monkey to climb a tree -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) : ℕ :=
  let effective_climb := hop_distance - slip_distance
  let full_climbs := (tree_height - 1) / effective_climb
  let remaining_distance := (tree_height - 1) % effective_climb
  full_climbs + if remaining_distance > 0 then 1 else 0

/-- Theorem: A monkey climbing a 17 ft tree, hopping 3 ft and slipping 2 ft each hour, takes 17 hours to reach the top -/
theorem monkey_climb_theorem :
  monkey_climb_time 17 3 2 = 17 := by
  sorry

end monkey_climb_theorem_l304_30429


namespace distance_to_point_l304_30486

/-- The distance from the origin to the point (12, 5) on the line y = 5/12 x is 13 -/
theorem distance_to_point : 
  let point : ℝ × ℝ := (12, 5)
  let line (x : ℝ) : ℝ := (5/12) * x
  (point.2 = line point.1) →
  Real.sqrt ((point.1 - 0)^2 + (point.2 - 0)^2) = 13 :=
by sorry

end distance_to_point_l304_30486


namespace ace_ten_king_probability_l304_30427

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of Tens in a standard deck -/
def num_tens : ℕ := 4

/-- The number of Kings in a standard deck -/
def num_kings : ℕ := 4

/-- The probability of drawing an Ace, then a 10, and then a King from a standard deck -/
def prob_ace_ten_king : ℚ := 8 / 16575

theorem ace_ten_king_probability :
  (num_aces : ℚ) / deck_size *
  num_tens / (deck_size - 1) *
  num_kings / (deck_size - 2) = prob_ace_ten_king := by
  sorry

end ace_ten_king_probability_l304_30427


namespace right_triangle_third_side_l304_30411

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  ((a = 4 ∧ b = 5) ∨ (a = 4 ∧ c = 5) ∨ (b = 4 ∧ c = 5)) →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c = 3 ∨ c = Real.sqrt 41 :=
by sorry

end right_triangle_third_side_l304_30411


namespace min_phi_value_l304_30407

noncomputable def g (x φ : ℝ) : ℝ := Real.sin (2 * (x + φ))

theorem min_phi_value (φ : ℝ) : 
  (φ > 0) →
  (∀ x, g x φ = g ((2 * π / 3) - x) φ) →
  (∀ ψ, ψ > 0 → (∀ x, g x ψ = g ((2 * π / 3) - x) ψ) → φ ≤ ψ) →
  φ = 5 * π / 12 := by
sorry

end min_phi_value_l304_30407


namespace certain_number_exists_l304_30404

theorem certain_number_exists : ∃ x : ℝ, 5 * 1.25 * x^(1/4) * 60^(3/4) = 300 := by
  sorry

end certain_number_exists_l304_30404


namespace a_range_l304_30447

/-- Given a > 0, if the function y = a^x is not monotonically increasing on ℝ
    or the inequality ax^2 - ax + 1 > 0 does not hold for ∀x ∈ ℝ,
    and at least one of these conditions is true,
    then a ∈ (0,1] ∪ [4,+∞) -/
theorem a_range (a : ℝ) (h_a_pos : a > 0) : 
  (¬∀ x y : ℝ, x < y → a^x < a^y) ∨ 
  (¬∀ x : ℝ, a*x^2 - a*x + 1 > 0) ∧ 
  ((∀ x y : ℝ, x < y → a^x < a^y) ∨ 
   (∀ x : ℝ, a*x^2 - a*x + 1 > 0)) → 
  a ∈ Set.Ioc 0 1 ∪ Set.Ici 4 :=
sorry

end a_range_l304_30447


namespace seashells_count_l304_30499

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The total number of seashells found by Tom and Fred -/
def total_seashells : ℕ := tom_seashells + fred_seashells

theorem seashells_count : total_seashells = 58 := by
  sorry

end seashells_count_l304_30499


namespace ln_101_100_gt_2_201_l304_30472

theorem ln_101_100_gt_2_201 : Real.log (101/100) > 2/201 := by
  sorry

end ln_101_100_gt_2_201_l304_30472


namespace max_gcd_13n_plus_4_8n_plus_3_l304_30484

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ (max : ℕ), 
    (∀ (n : ℕ), n > 0 → Nat.gcd (13*n + 4) (8*n + 3) ≤ max) ∧ 
    (∃ (n : ℕ), n > 0 ∧ Nat.gcd (13*n + 4) (8*n + 3) = max)) ∧
  (∀ (m : ℕ), 
    (∀ (n : ℕ), n > 0 → Nat.gcd (13*n + 4) (8*n + 3) ≤ m) →
    (∃ (n : ℕ), n > 0 ∧ Nat.gcd (13*n + 4) (8*n + 3) = m) →
    m ≤ 9) :=
by sorry

end max_gcd_13n_plus_4_8n_plus_3_l304_30484


namespace fraction_equality_implies_numerator_equality_l304_30487

theorem fraction_equality_implies_numerator_equality 
  {x y m : ℝ} (h1 : m ≠ 0) (h2 : x / m = y / m) : x = y :=
by sorry

end fraction_equality_implies_numerator_equality_l304_30487


namespace ninas_school_students_l304_30496

theorem ninas_school_students : ∀ (n m : ℕ),
  n = 5 * m →
  n + m = 4800 →
  (n - 200) + (m + 200) = 2 * (m + 200) →
  n = 4000 := by
sorry

end ninas_school_students_l304_30496


namespace product_sum_of_three_numbers_l304_30402

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a + b + c = 22) : 
  a*b + b*c + a*c = 131 := by
sorry

end product_sum_of_three_numbers_l304_30402


namespace mike_afternoon_seeds_l304_30467

/-- Represents the number of tomato seeds planted by Mike and Ted -/
structure TomatoSeeds where
  mike_morning : ℕ
  ted_morning : ℕ
  mike_afternoon : ℕ
  ted_afternoon : ℕ

/-- The conditions of the tomato planting problem -/
def tomato_planting_conditions (s : TomatoSeeds) : Prop :=
  s.mike_morning = 50 ∧
  s.ted_morning = 2 * s.mike_morning ∧
  s.ted_afternoon = s.mike_afternoon - 20 ∧
  s.mike_morning + s.ted_morning + s.mike_afternoon + s.ted_afternoon = 250

/-- Theorem stating that under the given conditions, Mike planted 60 tomato seeds in the afternoon -/
theorem mike_afternoon_seeds (s : TomatoSeeds) 
  (h : tomato_planting_conditions s) : s.mike_afternoon = 60 := by
  sorry

end mike_afternoon_seeds_l304_30467


namespace circle_equation_part1_circle_equation_part2_l304_30457

-- Part 1
theorem circle_equation_part1 (A B : ℝ × ℝ) (center_line : ℝ → ℝ) :
  A = (5, 2) →
  B = (3, 2) →
  (∀ x y, center_line x = 2*x - y - 3) →
  ∃ h k r, (∀ x y, (x - h)^2 + (y - k)^2 = r^2 ↔ 
    ((x = 5 ∧ y = 2) ∨ (x = 3 ∧ y = 2)) ∧
    center_line h = k) ∧
  h = 4 ∧ k = 5 ∧ r^2 = 10 :=
sorry

-- Part 2
theorem circle_equation_part2 (A : ℝ × ℝ) (sym_line chord_line : ℝ → ℝ) :
  A = (2, 3) →
  (∀ x y, sym_line x = -x - 2*y) →
  (∀ x y, chord_line x = x - y + 1) →
  ∃ h k r, (∀ x y, (x - h)^2 + (y - k)^2 = r^2 ↔ 
    ((x = 2 ∧ y = 3) ∨ 
     (∃ x' y', sym_line x' = y' ∧ (x' - h)^2 + (y' - k)^2 = r^2)) ∧
    (∃ x1 y1 x2 y2, chord_line x1 = y1 ∧ chord_line x2 = y2 ∧
      (x1 - x2)^2 + (y1 - y2)^2 = 8)) ∧
  ((h = 6 ∧ k = -3 ∧ r^2 = 52) ∨ (h = 14 ∧ k = -7 ∧ r^2 = 244)) :=
sorry

end circle_equation_part1_circle_equation_part2_l304_30457


namespace max_value_of_expression_l304_30420

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 2) : 
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x^2 + y^2 + z^2 = 2 → 2*a*b + 3*b*c ≤ 3 :=
by sorry

end max_value_of_expression_l304_30420


namespace max_arrangements_at_six_min_winning_probability_at_six_l304_30421

/-- The number of cities and days in the championship --/
def n : ℕ := 8

/-- Calculate the number of possible arrangements for k rounds --/
def arrangements (k : ℕ) : ℕ :=
  (Nat.factorial n * Nat.factorial n) / (Nat.factorial (n - k) * Nat.factorial (n - k) * Nat.factorial k)

/-- Theorem stating that 6 rounds maximizes the number of arrangements --/
theorem max_arrangements_at_six :
  ∀ k : ℕ, k ≤ n → arrangements 6 ≥ arrangements k :=
by sorry

/-- Corollary: The probability of winning the grand prize is minimized when there are 6 rounds --/
theorem min_winning_probability_at_six :
  ∀ k : ℕ, k ≤ n → (1 : ℚ) / arrangements 6 ≤ (1 : ℚ) / arrangements k :=
by sorry

end max_arrangements_at_six_min_winning_probability_at_six_l304_30421


namespace value_of_d_l304_30451

theorem value_of_d (a b c d e : ℝ) 
  (h : 3 * (a^2 + b^2 + c^2) + 4 = 2*d + Real.sqrt (a + b + c - d + e)) 
  (he : e = 1) : 
  d = 7/4 := by
sorry

end value_of_d_l304_30451


namespace inequality_proof_l304_30471

/-- The function f(x) defined as |x-a| + |x-3| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 3|

/-- Theorem: Given the conditions, prove that m + 2n ≥ 2 -/
theorem inequality_proof (a m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_solution_set : Set.Icc 1 3 = {x | f a x ≤ 1 + |x - 3|})
  (h_a : 1/m + 1/(2*n) = a) : m + 2*n ≥ 2 := by
  sorry

end inequality_proof_l304_30471


namespace not_special_2013_l304_30423

/-- A year is special if there exists a month and day such that their product
    equals the last two digits of the year. -/
def is_special_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

/-- The last two digits of 2013. -/
def last_two_digits_2013 : ℕ := 13

/-- Theorem stating that 2013 is not a special year. -/
theorem not_special_2013 : ¬(is_special_year 2013) := by
  sorry

end not_special_2013_l304_30423


namespace max_sum_2_by_1009_l304_30450

/-- Represents a grid of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the maximum sum of numbers in white squares for a given grid -/
def maxSumWhiteSquares (g : Grid) : ℕ :=
  if g.rows ≠ 2 ∨ g.cols ≠ 1009 then 0
  else
    let interiorContribution := (g.cols - 2) * 3
    let endpointContribution := 2 * 2
    interiorContribution + endpointContribution

/-- The theorem stating the maximum sum for a 2 by 1009 grid -/
theorem max_sum_2_by_1009 :
  ∀ g : Grid, g.rows = 2 ∧ g.cols = 1009 → maxSumWhiteSquares g = 3025 :=
by
  sorry

#eval maxSumWhiteSquares ⟨2, 1009⟩

end max_sum_2_by_1009_l304_30450


namespace equation_solution_l304_30495

theorem equation_solution : 
  ∀ x : ℝ, (x + 4)^2 = 5*(x + 4) ↔ x = -4 ∨ x = 1 :=
by sorry

end equation_solution_l304_30495
