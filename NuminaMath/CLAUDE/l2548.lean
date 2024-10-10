import Mathlib

namespace function_range_and_inequality_l2548_254822

theorem function_range_and_inequality (a b c m : ℝ) : 
  (∀ x, -x^2 + a*x + b ≤ 0) →
  (∀ x, -x^2 + a*x + b > c - 1 ↔ m - 4 < x ∧ x < m + 1) →
  c = 29/4 := by sorry

end function_range_and_inequality_l2548_254822


namespace rotation_270_degrees_l2548_254818

theorem rotation_270_degrees (z : ℂ) : z = -8 - 4*I → z * (-I) = -4 + 8*I := by
  sorry

end rotation_270_degrees_l2548_254818


namespace investment_proof_l2548_254887

def total_investment : ℝ := 3000
def part_one_investment : ℝ := 800
def part_one_interest_rate : ℝ := 0.10
def total_yearly_interest : ℝ := 256

theorem investment_proof :
  ∃ (part_two_interest_rate : ℝ),
    part_one_investment * part_one_interest_rate +
    (total_investment - part_one_investment) * part_two_interest_rate =
    total_yearly_interest :=
by sorry

end investment_proof_l2548_254887


namespace one_plus_x_geq_two_sqrt_x_l2548_254856

theorem one_plus_x_geq_two_sqrt_x (x : ℝ) (h : x ≥ 0) : 1 + x ≥ 2 * Real.sqrt x := by
  sorry

end one_plus_x_geq_two_sqrt_x_l2548_254856


namespace root_transformation_l2548_254845

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 8 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 8 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 8 = 0) → 
  ((2*r₁)^3 - 6*(2*r₁)^2 + 64 = 0) ∧
  ((2*r₂)^3 - 6*(2*r₂)^2 + 64 = 0) ∧
  ((2*r₃)^3 - 6*(2*r₃)^2 + 64 = 0) := by
sorry

end root_transformation_l2548_254845


namespace quadrilaterals_equal_area_l2548_254861

/-- Represents a quadrilateral on a geoboard -/
structure Quadrilateral where
  area : ℝ

/-- Quadrilateral I can be rearranged to form a 3x1 rectangle -/
def quadrilateral_I : Quadrilateral :=
  { area := 3 * 1 }

/-- Quadrilateral II can be rearranged to form two 1x1.5 rectangles -/
def quadrilateral_II : Quadrilateral :=
  { area := 2 * (1 * 1.5) }

/-- Theorem: Quadrilateral I and Quadrilateral II have the same area -/
theorem quadrilaterals_equal_area : quadrilateral_I.area = quadrilateral_II.area := by
  sorry

#check quadrilaterals_equal_area

end quadrilaterals_equal_area_l2548_254861


namespace frederick_tyson_age_ratio_l2548_254850

/-- Represents the ages and relationships between Kyle, Julian, Frederick, and Tyson -/
structure AgeRelationships where
  kyle_age : ℕ
  tyson_age : ℕ
  kyle_julian_diff : ℕ
  julian_frederick_diff : ℕ
  kyle_age_is_25 : kyle_age = 25
  tyson_age_is_20 : tyson_age = 20
  kyle_older_than_julian : kyle_age = kyle_julian_diff + (kyle_age - kyle_julian_diff)
  julian_younger_than_frederick : kyle_age - kyle_julian_diff = (kyle_age - kyle_julian_diff + julian_frederick_diff) - julian_frederick_diff

/-- The ratio of Frederick's age to Tyson's age is 2:1 -/
theorem frederick_tyson_age_ratio (ar : AgeRelationships) :
  (ar.kyle_age - ar.kyle_julian_diff + ar.julian_frederick_diff) / ar.tyson_age = 2 := by
  sorry

end frederick_tyson_age_ratio_l2548_254850


namespace divisibility_by_48_l2548_254854

theorem divisibility_by_48 :
  (∀ (n : ℕ), n > 0 → ¬(48 ∣ (7^n + 1))) ∧
  (∀ (n : ℕ), n > 0 → (48 ∣ (7^n - 1) ↔ Even n)) := by
  sorry

end divisibility_by_48_l2548_254854


namespace lines_coplanar_iff_h_eq_two_l2548_254807

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are coplanar -/
def are_coplanar (l1 l2 : Line3D) : Prop :=
  ∃ (c : ℝ), l1.direction = c • l2.direction

/-- The first line parameterized by s -/
def line1 (h : ℝ) : Line3D :=
  { point := (1, 0, 4),
    direction := (2, -1, h) }

/-- The second line parameterized by t -/
def line2 : Line3D :=
  { point := (0, 0, -6),
    direction := (3, 1, -2) }

/-- The main theorem stating the condition for coplanarity -/
theorem lines_coplanar_iff_h_eq_two :
  ∀ h : ℝ, are_coplanar (line1 h) line2 ↔ h = 2 := by
  sorry


end lines_coplanar_iff_h_eq_two_l2548_254807


namespace shoe_alteration_cost_l2548_254824

theorem shoe_alteration_cost (pairs : ℕ) (total_cost : ℕ) (cost_per_shoe : ℕ) :
  pairs = 17 →
  total_cost = 986 →
  cost_per_shoe = total_cost / (pairs * 2) →
  cost_per_shoe = 29 := by
  sorry

end shoe_alteration_cost_l2548_254824


namespace grape_ratio_theorem_l2548_254852

/-- Represents the contents and cost of a fruit basket -/
structure FruitBasket where
  banana_count : ℕ
  apple_count : ℕ
  strawberry_count : ℕ
  avocado_count : ℕ
  banana_price : ℚ
  apple_price : ℚ
  strawberry_price : ℚ
  avocado_price : ℚ
  grape_portion_price : ℚ
  total_cost : ℚ

/-- Calculates the cost of fruits excluding grapes -/
def cost_excluding_grapes (fb : FruitBasket) : ℚ :=
  fb.banana_count * fb.banana_price +
  fb.apple_count * fb.apple_price +
  fb.strawberry_count / 12 * fb.strawberry_price +
  fb.avocado_count * fb.avocado_price

/-- Calculates the cost of grapes in the basket -/
def grape_cost (fb : FruitBasket) : ℚ :=
  fb.total_cost - cost_excluding_grapes fb

/-- Represents the ratio of grapes in the basket to a whole bunch -/
structure GrapeRatio where
  numerator : ℚ
  denominator : ℚ

/-- Theorem stating the ratio of grapes in the basket to a whole bunch -/
theorem grape_ratio_theorem (fb : FruitBasket) (x : ℚ) :
  fb.banana_count = 4 →
  fb.apple_count = 3 →
  fb.strawberry_count = 24 →
  fb.avocado_count = 2 →
  fb.banana_price = 1 →
  fb.apple_price = 2 →
  fb.strawberry_price = 4 →
  fb.avocado_price = 3 →
  fb.grape_portion_price = 2 →
  fb.total_cost = 28 →
  x > 2 →
  ∃ (gr : GrapeRatio), gr.numerator = 2 ∧ gr.denominator = x :=
by sorry

end grape_ratio_theorem_l2548_254852


namespace distance_to_work_l2548_254832

/-- Proves that the distance from home to work is 10 km given the conditions --/
theorem distance_to_work (outbound_speed return_speed : ℝ) (distance : ℝ) : 
  return_speed = 2 * outbound_speed →
  distance / outbound_speed + distance / return_speed = 6 →
  return_speed = 5 →
  distance = 10 := by
sorry

end distance_to_work_l2548_254832


namespace fraction_value_l2548_254837

theorem fraction_value (x : ℝ) (h : x + 1/x = 3) : 
  x^2 / (x^4 + x^2 + 1) = 1/8 := by sorry

end fraction_value_l2548_254837


namespace arithmetic_sequence_a6_l2548_254880

/-- Given an arithmetic sequence {aₙ} with a₁ = 2 and S₃ = 12, prove that a₆ = 12 -/
theorem arithmetic_sequence_a6 (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                             -- a₁ = 2
  S 3 = 12 →                            -- S₃ = 12
  a 6 = 12 := by
sorry


end arithmetic_sequence_a6_l2548_254880


namespace three_distinct_roots_l2548_254848

/-- The equation has exactly three distinct roots if and only if a is in the set {-1.5, -0.75, 0, 1/4} -/
theorem three_distinct_roots (a : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ w : ℝ, (w^2 + (2*a - 1)*w - 4*a - 2) * (w^2 + w + a) = 0 ↔ w = x ∨ w = y ∨ w = z)) ↔
  a = -1.5 ∨ a = -0.75 ∨ a = 0 ∨ a = 1/4 := by
sorry

end three_distinct_roots_l2548_254848


namespace cannot_tile_removed_square_board_l2548_254895

/-- Represents a chessboard with one square removed -/
def RemovedSquareBoard : Nat := 63

/-- Represents the size of a domino -/
def DominoSize : Nat := 2

theorem cannot_tile_removed_square_board :
  ¬ ∃ (n : Nat), n * DominoSize = RemovedSquareBoard :=
sorry

end cannot_tile_removed_square_board_l2548_254895


namespace right_triangle_side_length_l2548_254877

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) -- Points in 2D plane
  (h_right : (R.1 - S.1) * (Q.1 - S.1) + (R.2 - S.2) * (Q.2 - S.2) = 0) -- Right angle at S
  (h_cos : (R.1 - Q.1) / Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 3/5) -- cos R = 3/5
  (h_rs : Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 10) -- RS = 10
  : Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = 8 := by -- QS = 8
  sorry

end right_triangle_side_length_l2548_254877


namespace hot_sauce_duration_l2548_254864

-- Define the size of a quart in ounces
def quart_size : ℝ := 32

-- Define the size of the hot sauce jar
def jar_size : ℝ := quart_size - 2

-- Define the size of each serving
def serving_size : ℝ := 0.5

-- Define the number of servings used daily
def daily_servings : ℕ := 3

-- Define the daily consumption
def daily_consumption : ℝ := serving_size * daily_servings

-- Theorem to prove
theorem hot_sauce_duration : 
  (jar_size / daily_consumption : ℝ) = 20 := by sorry

end hot_sauce_duration_l2548_254864


namespace egyptian_fraction_sum_l2548_254862

theorem egyptian_fraction_sum : ∃ (b₂ b₃ b₄ b₅ : ℕ),
  (3 : ℚ) / 5 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧
  b₂ + b₃ + b₄ + b₅ = 4 := by
  sorry

end egyptian_fraction_sum_l2548_254862


namespace line_through_point_parallel_to_line_l2548_254839

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line
  (p : Point)
  (l1 l2 : Line)
  (h1 : p.liesOn l2)
  (h2 : l2.isParallelTo l1)
  (h3 : l1.a = 1)
  (h4 : l1.b = -2)
  (h5 : l1.c = 3)
  (h6 : p.x = 1)
  (h7 : p.y = -3)
  (h8 : l2.a = 1)
  (h9 : l2.b = -2)
  (h10 : l2.c = -7) :
  True := by sorry

end line_through_point_parallel_to_line_l2548_254839


namespace project_completion_time_l2548_254849

/-- The number of days A takes to complete the project alone -/
def A_days : ℝ := 20

/-- The number of days it takes A and B together to complete the project -/
def total_days : ℝ := 15

/-- The number of days before completion that A quits -/
def A_quit_days : ℝ := 5

/-- The number of days B takes to complete the project alone -/
def B_days : ℝ := 30

/-- Theorem stating that given the conditions, B can complete the project alone in 30 days -/
theorem project_completion_time :
  A_days = 20 ∧ total_days = 15 ∧ A_quit_days = 5 →
  (total_days - A_quit_days) * (1 / A_days + 1 / B_days) + A_quit_days * (1 / B_days) = 1 :=
by sorry

end project_completion_time_l2548_254849


namespace sum_of_fractions_l2548_254891

theorem sum_of_fractions : (2 : ℚ) / 5 + (3 : ℚ) / 11 = (37 : ℚ) / 55 := by
  sorry

end sum_of_fractions_l2548_254891


namespace min_value_theorem_l2548_254893

/-- Given f(x) = a^x - b, where a > 0, a ≠ 1, and b is real,
    and g(x) = x + 1, if f(x) * g(x) ≤ 0 for all real x,
    then the minimum value of 1/a + 4/b is 4. -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (ha' : a ≠ 1) 
  (hf : ∀ x : ℝ, a^x - b ≤ 0 ∨ x + 1 ≤ 0) :
  ∀ ε > 0, ∃ a₀ b₀ : ℝ, 1/a₀ + 4/b₀ < 4 + ε :=
sorry

end min_value_theorem_l2548_254893


namespace vector_eq_quadratic_eq_l2548_254894

/-- The vector representing k(3, -4, 1) - (6, 9, -2) --/
def v (k : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 3*k - 6
  | 1 => -4*k - 9
  | 2 => k + 2

/-- The squared norm of the vector --/
def squared_norm (k : ℝ) : ℝ := (v k 0)^2 + (v k 1)^2 + (v k 2)^2

/-- The theorem stating the equivalence between the vector equation and the quadratic equation --/
theorem vector_eq_quadratic_eq (k : ℝ) :
  squared_norm k = (3 * Real.sqrt 26)^2 ↔ 26 * k^2 + 40 * k - 113 = 0 := by sorry

end vector_eq_quadratic_eq_l2548_254894


namespace original_number_l2548_254833

theorem original_number (x : ℝ) : x * 1.5 = 150 → x = 100 := by
  sorry

end original_number_l2548_254833


namespace b_85_mod_50_l2548_254898

/-- The sequence b_n is defined as 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The 85th term of the sequence b_n is congruent to 36 modulo 50 -/
theorem b_85_mod_50 : b 85 ≡ 36 [ZMOD 50] := by sorry

end b_85_mod_50_l2548_254898


namespace cafeteria_extra_apples_l2548_254881

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 33

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 23

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 21

/-- Each student takes one apple -/
axiom one_apple_per_student : ℕ

/-- The number of extra apples the cafeteria ended up with -/
def extra_apples : ℕ := (red_apples + green_apples) - students_wanting_fruit

theorem cafeteria_extra_apples : extra_apples = 35 := by
  sorry

end cafeteria_extra_apples_l2548_254881


namespace orlans_rope_problem_l2548_254843

theorem orlans_rope_problem (total_length : ℝ) (allan_portion : ℝ) (jack_portion : ℝ) (remaining : ℝ) :
  total_length = 20 →
  jack_portion = (2/3) * (total_length - allan_portion) →
  remaining = 5 →
  total_length = allan_portion + jack_portion + remaining →
  allan_portion / total_length = 1/4 :=
by sorry

end orlans_rope_problem_l2548_254843


namespace gcd_102_238_l2548_254831

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by sorry

end gcd_102_238_l2548_254831


namespace fraction_problem_l2548_254825

theorem fraction_problem : ∃ x : ℚ, 
  x * (3/4 : ℚ) = (1/6 : ℚ) ∧ x - (1/12 : ℚ) = (5/36 : ℚ) := by
  sorry

end fraction_problem_l2548_254825


namespace complex_equation_solution_l2548_254816

theorem complex_equation_solution (z : ℂ) : (1 + z * Complex.I = z + Complex.I) → z = -1 := by
  sorry

end complex_equation_solution_l2548_254816


namespace alvin_marbles_l2548_254872

theorem alvin_marbles (initial_marbles : ℕ) : 
  initial_marbles - 18 + 25 = 64 → initial_marbles = 57 := by
  sorry

end alvin_marbles_l2548_254872


namespace last_number_proof_l2548_254821

theorem last_number_proof (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  (b + c + d) / 3 = 3 →
  a + d = 13 →
  d = 2 := by
sorry

end last_number_proof_l2548_254821


namespace factor_divisor_statements_l2548_254830

theorem factor_divisor_statements :
  (∃ k : ℕ, 45 = 5 * k) ∧
  (∃ m : ℕ, 42 = 14 * m) ∧
  (∀ n : ℕ, 63 ≠ 14 * n) ∧
  (∃ p : ℕ, 180 = 9 * p) := by
  sorry

end factor_divisor_statements_l2548_254830


namespace compound_molecular_weight_l2548_254858

/-- Atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- Number of Carbon atoms in the compound -/
def carbon_count : ℕ := 4

/-- Number of Hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 8

/-- Number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- Calculation of molecular weight -/
def molecular_weight : ℝ :=
  (carbon_count : ℝ) * carbon_weight +
  (hydrogen_count : ℝ) * hydrogen_weight +
  (oxygen_count : ℝ) * oxygen_weight

/-- Theorem stating that the molecular weight of the compound is 88.104 g/mol -/
theorem compound_molecular_weight :
  molecular_weight = 88.104 := by sorry

end compound_molecular_weight_l2548_254858


namespace cow_chicken_problem_l2548_254853

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 20) → cows = 10 := by
  sorry

end cow_chicken_problem_l2548_254853


namespace sum_equation_solution_l2548_254836

/-- Given a real number k > 1 satisfying the infinite sum equation,
    prove that k equals the given expression. -/
theorem sum_equation_solution (k : ℝ) 
  (h1 : k > 1)
  (h2 : ∑' n, (7 * n - 2) / k^n = 3) :
  k = (21 + Real.sqrt 477) / 6 := by
  sorry

end sum_equation_solution_l2548_254836


namespace employee_pay_percentage_l2548_254802

theorem employee_pay_percentage (total_pay y_pay : ℝ) (h1 : total_pay = 570) (h2 : y_pay = 259.09) :
  let x_pay := total_pay - y_pay
  (x_pay / y_pay) * 100 = 120.03 := by
  sorry

end employee_pay_percentage_l2548_254802


namespace same_side_of_line_l2548_254815

/-- Define a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the line equation --/
def line_equation (p : Point) : ℝ := p.x + p.y - 1

/-- Check if a point is on the positive side of the line --/
def is_positive_side (p : Point) : Prop := line_equation p > 0

/-- The reference point (1,2) --/
def reference_point : Point := ⟨1, 2⟩

/-- The point to be checked (-1,3) --/
def check_point : Point := ⟨-1, 3⟩

/-- Theorem statement --/
theorem same_side_of_line : 
  is_positive_side reference_point → is_positive_side check_point :=
by sorry

end same_side_of_line_l2548_254815


namespace eight_sum_product_theorem_l2548_254842

theorem eight_sum_product_theorem : 
  ∃ (a b c d e f g h : ℤ), 
    (a + b + c + d + e + f + g + h = 8) ∧ 
    (a * b * c * d * e * f * g * h = 8) :=
sorry

end eight_sum_product_theorem_l2548_254842


namespace smallest_y_for_perfect_cube_l2548_254851

def x : ℕ := 9 * 36 * 54

theorem smallest_y_for_perfect_cube (y : ℕ) : 
  (∀ z < y, ∃ (a b : ℕ), x * z = a^3 → False) ∧
  (∃ (a : ℕ), x * y = a^3) →
  y = 9 := by
sorry

end smallest_y_for_perfect_cube_l2548_254851


namespace equation_solution_l2548_254871

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -4 ∧ 
  (∀ x : ℝ, (x - 1) * (x + 3) = 5 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end equation_solution_l2548_254871


namespace train_overtake_l2548_254819

/-- Proves that Train B overtakes Train A in 120 minutes given the specified conditions -/
theorem train_overtake (speed_a speed_b : ℝ) (head_start : ℝ) (overtake_time : ℝ) : 
  speed_a = 60 →
  speed_b = 80 →
  head_start = 40 / 60 →
  overtake_time = 120 / 60 →
  speed_a * (head_start + overtake_time) = speed_b * overtake_time :=
by sorry

end train_overtake_l2548_254819


namespace system_solutions_correct_l2548_254855

theorem system_solutions_correct :
  -- System (1)
  (∃ x y : ℚ, x - y = 2 ∧ x + 1 = 2 * (y - 1) ∧ x = 7 ∧ y = 5) ∧
  -- System (2)
  (∃ x y : ℚ, 2 * x + 3 * y = 1 ∧ (y - 1) / 4 = (x - 2) / 3 ∧ x = 1 ∧ y = -1/3) :=
by sorry

end system_solutions_correct_l2548_254855


namespace rectangle_length_calculation_l2548_254812

theorem rectangle_length_calculation (w : ℝ) (l_increase : ℝ) (w_decrease : ℝ) :
  w = 40 →
  l_increase = 0.30 →
  w_decrease = 0.17692307692307693 →
  (1 + l_increase) * (1 - w_decrease) * w = w →
  ∃ l : ℝ, l = 40 / 1.3 :=
by sorry

end rectangle_length_calculation_l2548_254812


namespace teal_survey_result_l2548_254826

/-- Represents the survey results about the color teal --/
structure TealSurvey where
  total : ℕ
  blue : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of people who believe teal is a shade of green --/
def green_believers (survey : TealSurvey) : ℕ :=
  survey.total - survey.blue + survey.both - survey.neither

/-- Theorem stating the result of the survey --/
theorem teal_survey_result (survey : TealSurvey) 
  (h_total : survey.total = 200)
  (h_blue : survey.blue = 130)
  (h_both : survey.both = 45)
  (h_neither : survey.neither = 35) :
  green_believers survey = 80 := by
  sorry

#eval green_believers { total := 200, blue := 130, both := 45, neither := 35 }

end teal_survey_result_l2548_254826


namespace initial_charge_value_l2548_254805

/-- The charge for the first 1/5 of a minute in cents -/
def initial_charge : ℝ := sorry

/-- The charge for each additional 1/5 of a minute in cents -/
def additional_charge : ℝ := 0.40

/-- The total charge for an 8-minute call in cents -/
def total_charge : ℝ := 18.70

/-- The number of 1/5 minute intervals in 8 minutes -/
def total_intervals : ℕ := 8 * 5

/-- The number of additional 1/5 minute intervals after the first one -/
def additional_intervals : ℕ := total_intervals - 1

theorem initial_charge_value :
  initial_charge = 3.10 :=
by
  sorry

end initial_charge_value_l2548_254805


namespace tea_containers_theorem_l2548_254889

/-- Given a total amount of tea in gallons, the number of containers Geraldo drank,
    and the amount of tea Geraldo consumed in pints, calculate the total number of
    containers filled with tea. -/
def totalContainers (totalTea : ℚ) (containersDrunk : ℚ) (teaDrunk : ℚ) : ℚ :=
  (totalTea * 8) / (teaDrunk / containersDrunk)

/-- Prove that given 20 gallons of tea, where 3.5 containers contain 7 pints,
    the total number of containers filled is 80. -/
theorem tea_containers_theorem :
  totalContainers 20 (7/2) 7 = 80 := by
  sorry

end tea_containers_theorem_l2548_254889


namespace min_value_of_function_l2548_254874

theorem min_value_of_function (x a b : ℝ) 
  (hx : 0 < x ∧ x < 1) (ha : a > 0) (hb : b > 0) : 
  a^2 / x + b^2 / (1 - x) ≥ (a + b)^2 := by
  sorry

end min_value_of_function_l2548_254874


namespace no_real_solutions_l2548_254810

theorem no_real_solutions : ¬∃ (x y : ℝ), x^2 + 3*y^2 - 8*x - 12*y + 36 = 0 := by
  sorry

end no_real_solutions_l2548_254810


namespace circle_area_ratio_l2548_254873

/-- Given two circles C and D, if an arc of 60° on C has the same length as an arc of 40° on D,
    then the ratio of the area of C to the area of D is 9/4. -/
theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0) 
  (h : C * (60 / 360) = D * (40 / 360)) : 
  (C^2 / D^2 : Real) = 9/4 := by
  sorry

end circle_area_ratio_l2548_254873


namespace circle_radius_l2548_254846

theorem circle_radius (x y : ℝ) (h : x + y = 72 * Real.pi) :
  ∃ r : ℝ, r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 6 := by
  sorry

end circle_radius_l2548_254846


namespace homework_pages_l2548_254838

theorem homework_pages (math_pages reading_pages total_pages : ℕ) : 
  math_pages = 10 ∧ 
  reading_pages = math_pages + 3 →
  total_pages = math_pages + reading_pages →
  total_pages = 23 := by
sorry

end homework_pages_l2548_254838


namespace tan_product_pi_ninths_l2548_254884

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end tan_product_pi_ninths_l2548_254884


namespace omega_sequence_monotone_l2548_254828

def is_omega_sequence (d : ℕ → ℕ) : Prop :=
  (∀ n, (d n + d (n + 2)) / 2 ≤ d (n + 1)) ∧
  (∃ M : ℝ, ∀ n, (d n : ℝ) ≤ M)

theorem omega_sequence_monotone (d : ℕ → ℕ) 
  (h_omega : is_omega_sequence d) :
  ∀ n, d n ≤ d (n + 1) := by
sorry

end omega_sequence_monotone_l2548_254828


namespace imaginary_part_of_z_l2548_254876

theorem imaginary_part_of_z (z : ℂ) : (3 - 4*I)*z = Complex.abs (4 + 3*I) → Complex.im z = 4/5 := by
  sorry

end imaginary_part_of_z_l2548_254876


namespace inequalities_always_true_l2548_254885

theorem inequalities_always_true (a b : ℝ) (h : a * b > 0) :
  (a^2 + b^2 ≥ 2*a*b) ∧ (b/a + a/b ≥ 2) := by sorry

end inequalities_always_true_l2548_254885


namespace souvenir_shop_theorem_l2548_254899

/-- Represents the purchase and profit scenario of a souvenir shop. -/
structure SouvenirShop where
  price_A : ℚ  -- Purchase price of souvenir A
  price_B : ℚ  -- Purchase price of souvenir B
  profit_A : ℚ -- Profit per piece of souvenir A
  profit_B : ℚ -- Profit per piece of souvenir B

/-- Theorem stating the correct purchase prices and total profit -/
theorem souvenir_shop_theorem (shop : SouvenirShop) : 
  (7 * shop.price_A + 8 * shop.price_B = 380) →
  (10 * shop.price_A + 6 * shop.price_B = 380) →
  shop.profit_A = 5 →
  shop.profit_B = 7 →
  (∃ (m n : ℚ), m + n = 40 ∧ shop.price_A * m + shop.price_B * n = 900) →
  (shop.price_A = 20 ∧ shop.price_B = 30 ∧ 
   ∃ (m n : ℚ), m + n = 40 ∧ shop.price_A * m + shop.price_B * n = 900 ∧
                m * shop.profit_A + n * shop.profit_B = 220) := by
  sorry


end souvenir_shop_theorem_l2548_254899


namespace sum_a7_a9_eq_zero_l2548_254869

theorem sum_a7_a9_eq_zero (a : ℕ+ → ℤ) 
  (h : ∀ n : ℕ+, a n = 3 * n.val - 24) : 
  a 7 + a 9 = 0 := by
  sorry

end sum_a7_a9_eq_zero_l2548_254869


namespace sum_of_quadratic_and_linear_l2548_254878

/-- Given a quadratic function q and a linear function p satisfying certain conditions,
    prove that their sum has a specific form. -/
theorem sum_of_quadratic_and_linear 
  (q : ℝ → ℝ) 
  (p : ℝ → ℝ) 
  (hq_quad : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c)
  (hq_zeros : q 1 = 0 ∧ q 3 = 0)
  (hq_value : q 4 = 8)
  (hp_linear : ∃ m b : ℝ, ∀ x, p x = m * x + b)
  (hp_value : p 5 = 15) :
  ∀ x, p x + q x = (8/3) * x^2 - (29/3) * x + 8 := by
  sorry

end sum_of_quadratic_and_linear_l2548_254878


namespace sqrt_2450_minus_2_theorem_l2548_254800

theorem sqrt_2450_minus_2_theorem (a b : ℕ+) :
  (Real.sqrt 2450 - 2 : ℝ) = ((Real.sqrt a.val : ℝ) - b.val)^2 →
  a.val + b.val = 2451 := by
sorry

end sqrt_2450_minus_2_theorem_l2548_254800


namespace smallest_number_l2548_254860

theorem smallest_number (a b c d : ℝ) 
  (ha : a = 1) 
  (hb : b = -3) 
  (hc : c = -Real.sqrt 2) 
  (hd : d = -Real.pi) : 
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
sorry

end smallest_number_l2548_254860


namespace unique_cube_difference_61_l2548_254863

theorem unique_cube_difference_61 :
  ∃! (n k : ℕ), n^3 - k^3 = 61 :=
by sorry

end unique_cube_difference_61_l2548_254863


namespace rhombus_area_l2548_254835

/-- The area of a rhombus with side length 20 cm and an angle of 60 degrees between two adjacent sides is 200√3 cm². -/
theorem rhombus_area (side : ℝ) (angle : ℝ) (h1 : side = 20) (h2 : angle = π / 3) :
  side * side * Real.sin angle = 200 * Real.sqrt 3 := by
  sorry

end rhombus_area_l2548_254835


namespace permutation_equation_solution_l2548_254896

def A (k m : ℕ) : ℕ := (k.factorial) / (k - m).factorial

theorem permutation_equation_solution :
  ∃! n : ℕ, n > 0 ∧ A (2*n) 3 = 10 * A n 3 :=
sorry

end permutation_equation_solution_l2548_254896


namespace assistant_end_time_l2548_254804

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a bracelet producer -/
structure Producer where
  startTime : Time
  endTime : Time
  rate : Nat
  interval : Nat
  deriving Repr

def craftsman : Producer := {
  startTime := { hours := 8, minutes := 0 }
  endTime := { hours := 12, minutes := 0 }
  rate := 6
  interval := 20
}

def assistant : Producer := {
  startTime := { hours := 9, minutes := 0 }
  endTime := { hours := 0, minutes := 0 }  -- To be determined
  rate := 8
  interval := 30
}

def calculateProduction (p : Producer) : Nat :=
  sorry

def calculateEndTime (p : Producer) (targetProduction : Nat) : Time :=
  sorry

theorem assistant_end_time :
  calculateEndTime assistant (calculateProduction craftsman) = { hours := 13, minutes := 30 } :=
sorry

end assistant_end_time_l2548_254804


namespace sector_max_area_l2548_254820

/-- Given a sector with fixed perimeter P, prove that the maximum area is P^2/16
    and this maximum is achieved when the radius is P/4. -/
theorem sector_max_area (P : ℝ) (h : P > 0) :
  let max_area := P^2 / 16
  let max_radius := P / 4
  ∀ R l, R > 0 → l > 0 → 2 * R + l = P →
    (1/2 * R * l ≤ max_area) ∧
    (1/2 * max_radius * (P - 2 * max_radius) = max_area) :=
by sorry


end sector_max_area_l2548_254820


namespace equation_solution_l2548_254814

theorem equation_solution : 
  ∃! x : ℝ, x ≠ -1 ∧ x ≠ -(3/2) ∧ x ≠ 1/2 ∧ x ≠ -(1/2) ∧
  (((((2*x+1)/(2*x-1))-1)/(1-((2*x-1)/(2*x+1)))) + 
   ((((2*x+1)/(2*x-1))-2)/(2-((2*x-1)/(2*x+1)))) +
   ((((2*x+1)/(2*x-1))-3)/(3-((2*x-1)/(2*x+1))))) = 0 ∧
  x = -3 :=
by sorry

end equation_solution_l2548_254814


namespace f_1989_value_l2548_254879

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) * (1 - f x) = 1 + f x

theorem f_1989_value (f : ℝ → ℝ) 
    (h_eq : SatisfiesEquation f) 
    (h_f1 : f 1 = 2 + Real.sqrt 3) : 
    f 1989 = -2 + Real.sqrt 3 := by
  sorry

end f_1989_value_l2548_254879


namespace hypotenuse_length_l2548_254809

/-- A right triangle with specific properties -/
structure RightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  longer_leg_property : longer_leg = 3 * shorter_leg - 1
  area_property : (1 / 2) * shorter_leg * longer_leg = 24
  pythagorean_theorem : shorter_leg ^ 2 + longer_leg ^ 2 = hypotenuse ^ 2

/-- The length of the hypotenuse in the specific right triangle is √137 -/
theorem hypotenuse_length (t : RightTriangle) : t.hypotenuse = Real.sqrt 137 := by
  sorry

end hypotenuse_length_l2548_254809


namespace boat_speed_l2548_254875

/-- Given a boat that travels 11 km/h along a stream and 5 km/h against the same stream,
    the speed of the boat in still water is 8 km/h. -/
theorem boat_speed (b s : ℝ) 
    (h1 : b + s = 11)  -- Speed along the stream
    (h2 : b - s = 5)   -- Speed against the stream
    : b = 8 := by
  sorry

end boat_speed_l2548_254875


namespace base_conversion_and_division_l2548_254870

/-- Given that 746 in base 8 is equal to 4cd in base 10, where c and d are base-10 digits,
    prove that (c * d) / 12 = 4 -/
theorem base_conversion_and_division (c d : ℕ) : 
  c < 10 → d < 10 → 746 = 4 * c * 10 + d → (c * d) / 12 = 4 := by
  sorry

end base_conversion_and_division_l2548_254870


namespace quarter_to_fourth_power_decimal_l2548_254840

theorem quarter_to_fourth_power_decimal : (1 / 4 : ℝ) ^ 4 = 0.00390625 := by sorry

end quarter_to_fourth_power_decimal_l2548_254840


namespace cos_derivative_at_pi_sixth_l2548_254844

theorem cos_derivative_at_pi_sixth (f : ℝ → ℝ) :
  (∀ x, f x = Real.cos x) → HasDerivAt f (-1/2) (π/6) := by
  sorry

end cos_derivative_at_pi_sixth_l2548_254844


namespace two_congruent_rectangles_l2548_254886

/-- A point on a circle --/
structure CirclePoint where
  angle : ℝ
  angleInRange : 0 ≤ angle ∧ angle < 2 * Real.pi

/-- A rectangle inscribed in a circle --/
structure InscribedRectangle where
  vertices : Fin 4 → CirclePoint
  isRectangle : ∀ i : Fin 4, (vertices i).angle - (vertices ((i + 1) % 4)).angle = Real.pi / 2 ∨
                             (vertices i).angle - (vertices ((i + 1) % 4)).angle = -3 * Real.pi / 2

/-- The main theorem --/
theorem two_congruent_rectangles 
  (points : Fin 40 → CirclePoint)
  (equallySpaced : ∀ i : Fin 39, (points (i + 1)).angle - (points i).angle = Real.pi / 20)
  (rectangles : Fin 10 → InscribedRectangle)
  (verticesOnPoints : ∀ r : Fin 10, ∀ v : Fin 4, ∃ p : Fin 40, (rectangles r).vertices v = points p) :
  ∃ r1 r2 : Fin 10, r1 ≠ r2 ∧ rectangles r1 = rectangles r2 :=
sorry

end two_congruent_rectangles_l2548_254886


namespace min_value_sum_reciprocals_l2548_254817

theorem min_value_sum_reciprocals (n : ℕ) (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
  (1 / (1 + a^n)) + (1 / (1 + b^n)) ≥ 1 ∧ 
  ((1 / (1 + a^n)) + (1 / (1 + b^n)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

#check min_value_sum_reciprocals

end min_value_sum_reciprocals_l2548_254817


namespace distance_on_line_l2548_254811

/-- Given two points (a, b) and (c, d) on a line y = mx + k, 
    the distance between them is |a - c|√(1 + m²) -/
theorem distance_on_line (m k a b c d : ℝ) :
  b = m * a + k →
  d = m * c + k →
  Real.sqrt ((c - a)^2 + (d - b)^2) = |a - c| * Real.sqrt (1 + m^2) :=
by sorry

end distance_on_line_l2548_254811


namespace some_number_value_l2548_254806

theorem some_number_value (t k some_number : ℝ) 
  (h1 : t = 5 / 9 * (k - some_number))
  (h2 : t = 35)
  (h3 : k = 95) : 
  some_number = 32 := by
sorry

end some_number_value_l2548_254806


namespace sequence_with_2018_distinct_elements_l2548_254808

theorem sequence_with_2018_distinct_elements :
  ∃ a : ℝ, ∃ (x : ℕ → ℝ), 
    (x 1 = a) ∧ 
    (∀ n : ℕ, x (n + 1) = (1 / 2) * (x n - 1 / x n)) ∧
    (∃ m : ℕ, m ≤ 2018 ∧ x m = 0) ∧
    (∀ i j : ℕ, i < j ∧ j ≤ 2018 → x i ≠ x j) ∧
    (∀ k : ℕ, k > 2018 → x k = 0) :=
by sorry

end sequence_with_2018_distinct_elements_l2548_254808


namespace boys_in_class_l2548_254866

theorem boys_in_class (total : ℕ) (girl_ratio boy_ratio : ℕ) (h1 : total = 56) (h2 : girl_ratio = 4) (h3 : boy_ratio = 3) :
  (total * boy_ratio) / (girl_ratio + boy_ratio) = 24 :=
by sorry

end boys_in_class_l2548_254866


namespace circle_radius_isosceles_right_triangle_l2548_254865

/-- The radius of a circle tangent to both axes and the hypotenuse of an isosceles right triangle -/
theorem circle_radius_isosceles_right_triangle (O : ℝ × ℝ) (P Q R S T U : ℝ × ℝ) (r : ℝ) :
  -- PQR is an isosceles right triangle
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4 →
  (P.1 - R.1)^2 + (P.2 - R.2)^2 = (Q.1 - R.1)^2 + (Q.2 - R.2)^2 →
  (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0 →
  -- S is on PQ
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2) →
  -- Circle with center O is tangent to coordinate axes
  O.1 = r ∧ O.2 = r →
  -- Circle is tangent to PQ at T
  (T.1 - O.1)^2 + (T.2 - O.2)^2 = r^2 →
  ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ T = (s * P.1 + (1 - s) * Q.1, s * P.2 + (1 - s) * Q.2) →
  -- U is on x-axis and circle is tangent at U
  U.2 = 0 ∧ (U.1 - O.1)^2 + (U.2 - O.2)^2 = r^2 →
  -- The radius of the circle is 2 + √2
  r = 2 + Real.sqrt 2 := by
sorry

end circle_radius_isosceles_right_triangle_l2548_254865


namespace investment_proportional_to_profit_share_q_investment_l2548_254823

/-- Represents the investment and profit share of an investor -/
structure Investor where
  investment : ℕ
  profitShare : ℕ

/-- Given two investors with their investments and profit shares, 
    proves that their investments are proportional to their profit shares -/
theorem investment_proportional_to_profit_share 
  (p q : Investor) 
  (h1 : p.investment = 500000) 
  (h2 : p.profitShare = 2) 
  (h3 : q.profitShare = 4) : 
  q.investment = 1000000 := by
sorry

/-- Main theorem that proves Q's investment given P's investment and their profit sharing ratio -/
theorem q_investment 
  (p q : Investor) 
  (h1 : p.investment = 500000) 
  (h2 : p.profitShare = 2) 
  (h3 : q.profitShare = 4) : 
  q.investment = 1000000 := by
sorry

end investment_proportional_to_profit_share_q_investment_l2548_254823


namespace shoes_cost_calculation_l2548_254803

def budget : ℚ := 200
def shirt_cost : ℚ := 30
def pants_cost : ℚ := 46
def coat_cost : ℚ := 38
def socks_cost : ℚ := 11
def belt_cost : ℚ := 18
def necktie_cost : ℚ := 22
def remaining_money : ℚ := 16

def other_items_cost : ℚ := shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost + necktie_cost

theorem shoes_cost_calculation :
  ∃ (shoes_cost : ℚ), 
    shoes_cost = budget - remaining_money - other_items_cost ∧
    shoes_cost = 19 :=
by sorry

end shoes_cost_calculation_l2548_254803


namespace silver_solution_percentage_second_solution_percentage_l2548_254897

/-- Given two silver solutions mixed to form a new solution, 
    calculate the silver percentage in the second solution. -/
theorem silver_solution_percentage 
  (volume1 : ℝ) (percent1 : ℝ) 
  (volume2 : ℝ) (final_percent : ℝ) : ℝ :=
  let total_volume := volume1 + volume2
  let silver_volume1 := volume1 * (percent1 / 100)
  let total_silver := total_volume * (final_percent / 100)
  let silver_volume2 := total_silver - silver_volume1
  (silver_volume2 / volume2) * 100

/-- Prove that the percentage of silver in the second solution is 10% -/
theorem second_solution_percentage : 
  silver_solution_percentage 5 4 2.5 6 = 10 := by
  sorry

end silver_solution_percentage_second_solution_percentage_l2548_254897


namespace biased_coin_probability_l2548_254867

/-- The probability of getting exactly k successes in n trials of a binomial experiment -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 9 heads in 12 flips of a biased coin with 1/3 probability of landing heads -/
theorem biased_coin_probability : 
  binomialProbability 12 9 (1/3) = 1760/531441 := by
  sorry

end biased_coin_probability_l2548_254867


namespace parallel_line_through_point_l2548_254892

/-- Given a line L1 with equation x - 2y - 2 = 0, prove that the line L2 with equation x - 2y - 1 = 0
    passes through the point (1,0) and is parallel to L1. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (x - 2*y - 1 = 0) ∧ 
  (1 - 2*0 - 1 = 0) ∧ 
  (∃ k : ℝ, k ≠ 0 ∧ (1 : ℝ) = k * 1 ∧ (-2 : ℝ) = k * (-2)) := by
  sorry

end parallel_line_through_point_l2548_254892


namespace quadratic_inequality_range_l2548_254883

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ -4 < k ∧ k ≤ 0 := by
  sorry

end quadratic_inequality_range_l2548_254883


namespace cloth_sale_problem_l2548_254834

/-- Represents the problem of determining the number of metres of cloth sold --/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) :
  total_selling_price = 18000 →
  loss_per_metre = 5 →
  cost_price_per_metre = 95 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 200 := by
  sorry

#check cloth_sale_problem

end cloth_sale_problem_l2548_254834


namespace distributive_analogy_l2548_254829

theorem distributive_analogy (a b c : ℝ) (h : c ≠ 0) :
  (a + b) * c = a * c + b * c ↔ (a + b) / c = a / c + b / c :=
sorry

end distributive_analogy_l2548_254829


namespace complex_number_in_first_quadrant_l2548_254801

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (Complex.I / (1 + Complex.I) = ↑a + ↑b * Complex.I) := by
  sorry

end complex_number_in_first_quadrant_l2548_254801


namespace multiple_with_binary_digits_l2548_254890

theorem multiple_with_binary_digits (n : ℤ) : ∃ m : ℤ, 
  (n ∣ m) ∧ 
  (∃ k : ℕ, k ≤ n ∧ m < 10^k) ∧
  (∀ d : ℕ, d < 10 → (m / 10^d % 10 = 0 ∨ m / 10^d % 10 = 1)) :=
by sorry

end multiple_with_binary_digits_l2548_254890


namespace M_eq_302_l2548_254857

/-- The number of ways to write 3010 as a sum of powers of 10 with restricted coefficients -/
def M : ℕ :=
  (Finset.filter (fun (b₃ : ℕ) =>
    (Finset.filter (fun (b₂ : ℕ) =>
      (Finset.filter (fun (b₁ : ℕ) =>
        (Finset.filter (fun (b₀ : ℕ) =>
          b₃ * 1000 + b₂ * 100 + b₁ * 10 + b₀ = 3010
        ) (Finset.range 100)).card > 0
      ) (Finset.range 100)).card > 0
    ) (Finset.range 100)).card > 0
  ) (Finset.range 100)).card

/-- The theorem stating that M equals 302 -/
theorem M_eq_302 : M = 302 := by
  sorry

end M_eq_302_l2548_254857


namespace inequality_proof_l2548_254813

open Real BigOperators

theorem inequality_proof (n : ℕ) (r s t u v : Fin n → ℝ) 
  (hr : ∀ i, r i > 1) (hs : ∀ i, s i > 1) (ht : ∀ i, t i > 1) (hu : ∀ i, u i > 1) (hv : ∀ i, v i > 1) :
  let R := (∑ i, r i) / n
  let S := (∑ i, s i) / n
  let T := (∑ i, t i) / n
  let U := (∑ i, u i) / n
  let V := (∑ i, v i) / n
  ∑ i, ((r i * s i * t i * u i * v i + 1) / (r i * s i * t i * u i * v i - 1)) ≥ 
    ((R * S * T * U * V + 1) / (R * S * T * U * V - 1)) ^ n :=
by
  sorry

end inequality_proof_l2548_254813


namespace project_delay_without_additional_workers_l2548_254841

/-- Represents the construction project parameters and outcome -/
structure ConstructionProject where
  plannedDays : ℕ
  initialWorkers : ℕ
  additionalWorkers : ℕ
  additionalWorkersStartDay : ℕ
  actualCompletionDays : ℕ

/-- Calculates the total man-days of work for the project -/
def totalManDays (project : ConstructionProject) : ℕ :=
  project.initialWorkers * project.additionalWorkersStartDay +
  (project.initialWorkers + project.additionalWorkers) *
  (project.actualCompletionDays - project.additionalWorkersStartDay)

/-- Theorem: If a project is completed on time with additional workers,
    it would take longer without them -/
theorem project_delay_without_additional_workers
  (project : ConstructionProject)
  (h1 : project.plannedDays = 100)
  (h2 : project.initialWorkers = 100)
  (h3 : project.additionalWorkers = 100)
  (h4 : project.additionalWorkersStartDay = 50)
  (h5 : project.actualCompletionDays = 100) :
  (totalManDays project) / project.initialWorkers = 150 :=
sorry

end project_delay_without_additional_workers_l2548_254841


namespace inequality_and_equality_condition_l2548_254859

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (a + 2) * (b + 2) ≥ c * d ∧ 
  (∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ 
    (a₀ + 2) * (b₀ + 2) = c₀ * d₀ ∧ 
    a₀ = -1 ∧ b₀ = -1 ∧ c₀ = -1 ∧ d₀ = -1) := by
  sorry

end inequality_and_equality_condition_l2548_254859


namespace square_difference_l2548_254847

theorem square_difference (a b : ℝ) 
  (h1 : a^2 + a*b = 8) 
  (h2 : a*b + b^2 = 9) : 
  a^2 - b^2 = -1 := by
sorry

end square_difference_l2548_254847


namespace sum_of_squares_lower_bound_l2548_254888

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by sorry

end sum_of_squares_lower_bound_l2548_254888


namespace tv_price_reduction_l2548_254868

theorem tv_price_reduction (x : ℝ) : 
  (1 - x/100)^2 = 1 - 19/100 → x = 10 := by
  sorry

end tv_price_reduction_l2548_254868


namespace fixed_point_on_line_l2548_254882

theorem fixed_point_on_line (t : ℝ) : 
  (t + 1) * (-4) - (2 * t + 5) * (-2) - 6 = 0 := by
  sorry

#check fixed_point_on_line

end fixed_point_on_line_l2548_254882


namespace pencil_cost_l2548_254827

/-- Given that 120 pencils cost $36, prove that 3000 pencils cost $900 -/
theorem pencil_cost (pencils_per_box : ℕ) (cost_per_box : ℚ) (total_pencils : ℕ) :
  pencils_per_box = 120 →
  cost_per_box = 36 →
  total_pencils = 3000 →
  (cost_per_box / pencils_per_box) * total_pencils = 900 :=
by sorry

end pencil_cost_l2548_254827
