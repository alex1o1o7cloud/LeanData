import Mathlib

namespace emily_trivia_score_l1182_118214

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round : ℤ) (second_round : ℤ) (last_round : ℤ) 
  (h1 : first_round = 16)
  (h2 : second_round = 33)
  (h3 : last_round = -48) :
  first_round + second_round + last_round = 1 := by
  sorry

end emily_trivia_score_l1182_118214


namespace fractional_to_polynomial_equivalence_l1182_118281

theorem fractional_to_polynomial_equivalence (x : ℝ) (h : x ≠ 2) :
  (x / (x - 2) + 2 = 1 / (2 - x)) ↔ (x + 2 * (x - 2) = -1) :=
sorry

end fractional_to_polynomial_equivalence_l1182_118281


namespace isosceles_triangle_area_l1182_118225

/-- An isosceles triangle with two sides of length 5 and a median to the base of length 4 has an area of 12 -/
theorem isosceles_triangle_area (a b c : ℝ) (m : ℝ) (h_isosceles : a = b) (h_side : a = 5) (h_median : m = 4) : 
  (1/2 : ℝ) * c * m = 12 := by
  sorry

end isosceles_triangle_area_l1182_118225


namespace quadratic_equation_solution_l1182_118299

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end quadratic_equation_solution_l1182_118299


namespace point_C_range_l1182_118205

def parabola (x y : ℝ) : Prop := y^2 = x + 4

def perpendicular (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) * (y3 - y2) = -(x3 - x2) * (x2 - x1)

theorem point_C_range :
  ∀ y : ℝ,
  (∃ y1 : ℝ,
    parabola (y1^2 - 4) y1 ∧
    parabola (y^2 - 4) y ∧
    perpendicular 0 2 (y1^2 - 4) y1 (y^2 - 4) y) →
  y ≤ 0 ∨ y ≥ 4 := by
sorry

end point_C_range_l1182_118205


namespace calculate_e_l1182_118265

/-- Given the relationships between variables j, p, t, b, a, and e, prove that e = 21.5 -/
theorem calculate_e (j p t b a e : ℝ) 
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (e / 100) * p)
  (h4 : b = 1.40 * j)
  (h5 : a = 0.85 * b)
  (h6 : e = 2 * ((p - a) / p) * 100) :
  e = 21.5 := by
  sorry

end calculate_e_l1182_118265


namespace probability_calm_in_mathematics_l1182_118276

def letters_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}
def letters_calm : Finset Char := {'C', 'A', 'L', 'M'}

def count_occurrences (c : Char) : ℕ :=
  if c = 'M' ∨ c = 'A' then 2
  else if c ∈ letters_mathematics then 1
  else 0

def favorable_outcomes : ℕ := (letters_calm ∩ letters_mathematics).sum count_occurrences

theorem probability_calm_in_mathematics :
  (favorable_outcomes : ℚ) / 12 = 5 / 12 := by sorry

end probability_calm_in_mathematics_l1182_118276


namespace complex_magnitude_example_l1182_118261

theorem complex_magnitude_example : Complex.abs (-5 + (8/3) * Complex.I) = 17/3 := by
  sorry

end complex_magnitude_example_l1182_118261


namespace boxes_in_smallest_cube_l1182_118258

def box_width : ℕ := 8
def box_length : ℕ := 12
def box_height : ℕ := 30

def smallest_cube_side : ℕ := lcm (lcm box_width box_length) box_height

def box_volume : ℕ := box_width * box_length * box_height
def cube_volume : ℕ := smallest_cube_side ^ 3

theorem boxes_in_smallest_cube :
  cube_volume / box_volume = 600 := by sorry

end boxes_in_smallest_cube_l1182_118258


namespace parallelepipeds_crossed_diagonal_count_l1182_118298

/-- The edge length of the cube -/
def cube_edge : ℕ := 90

/-- The dimensions of the rectangular parallelepiped -/
def parallelepiped_dims : Fin 3 → ℕ
| 0 => 2
| 1 => 3
| 2 => 5

/-- The number of parallelepipeds that fit along each dimension of the cube -/
def parallelepipeds_per_dim (i : Fin 3) : ℕ := cube_edge / parallelepiped_dims i

/-- The total number of parallelepipeds that fit in the cube -/
def total_parallelepipeds : ℕ := (parallelepipeds_per_dim 0) * (parallelepipeds_per_dim 1) * (parallelepipeds_per_dim 2)

/-- The number of parallelepipeds crossed by a space diagonal of the cube -/
def parallelepipeds_crossed_by_diagonal : ℕ := 65

theorem parallelepipeds_crossed_diagonal_count :
  parallelepipeds_crossed_by_diagonal = 65 :=
sorry

end parallelepipeds_crossed_diagonal_count_l1182_118298


namespace projection_length_l1182_118235

def vector_a : ℝ × ℝ := (3, 4)
def vector_b : ℝ × ℝ := (0, 1)

theorem projection_length :
  let a := vector_a
  let b := vector_b
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = 4 := by sorry

end projection_length_l1182_118235


namespace inscribed_squares_ratio_l1182_118295

/-- Given a 3-4-5 right triangle, let x be the side length of a square inscribed
    with one vertex at the right angle, and y be the side length of a square
    inscribed with one side on the hypotenuse. -/
theorem inscribed_squares_ratio (x y : ℝ) 
  (hx : x * (7 / 3) = 4) -- Derived from the condition for x
  (hy : y * (37 / 12) = 5) -- Derived from the condition for y
  : x / y = 37 / 35 := by
  sorry

end inscribed_squares_ratio_l1182_118295


namespace unique_award_implies_all_defeated_l1182_118229

def Tournament (α : Type) := α → α → Prop

structure Award (α : Type) (t : Tournament α) (winner : α) : Prop :=
  (defeated_or_indirect : ∀ b : α, b ≠ winner → t winner b ∨ ∃ c, t winner c ∧ t c b)

theorem unique_award_implies_all_defeated 
  {α : Type} (t : Tournament α) (winner : α) :
  (∀ a b : α, a ≠ b → t a b ∨ t b a) →
  (∀ x : α, Award α t x ↔ x = winner) →
  (∀ b : α, b ≠ winner → t winner b) :=
sorry

end unique_award_implies_all_defeated_l1182_118229


namespace arithmetic_sequence_common_difference_l1182_118246

/-- Given an arithmetic sequence {a_n} where a₂ + 1 is the arithmetic mean of a₁ and a₄,
    the common difference of the sequence is 2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  (h_mean : a 1 + a 4 = 2 * (a 2 + 1))  -- a₂ + 1 is the arithmetic mean of a₁ and a₄
  : a 2 - a 1 = 2 :=  -- The common difference is 2
by sorry

end arithmetic_sequence_common_difference_l1182_118246


namespace quadratic_integral_inequality_l1182_118260

/-- For real numbers a, b, c, let f(x) = ax^2 + bx + c. 
    Prove that ∫_{-1}^1 (1 - x^2){f'(x)}^2 dx ≤ 6∫_{-1}^1 {f(x)}^2 dx -/
theorem quadratic_integral_inequality (a b c : ℝ) : 
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + c
  ∫ x in (-1)..1, (1 - x^2) * (deriv f x)^2 ≤ 6 * ∫ x in (-1)..1, (f x)^2 := by
  sorry

end quadratic_integral_inequality_l1182_118260


namespace blue_given_not_red_probability_l1182_118215

-- Define the total number of balls
def total_balls : ℕ := 20

-- Define the number of red balls
def red_balls : ℕ := 5

-- Define the number of yellow balls
def yellow_balls : ℕ := 5

-- Define the number of blue balls
def blue_balls : ℕ := 10

-- Define the number of non-red balls
def non_red_balls : ℕ := yellow_balls + blue_balls

-- Theorem: The probability of drawing a blue ball given that it's not red is 2/3
theorem blue_given_not_red_probability : 
  (blue_balls : ℚ) / (non_red_balls : ℚ) = 2 / 3 :=
sorry

end blue_given_not_red_probability_l1182_118215


namespace min_value_sum_of_squares_l1182_118237

theorem min_value_sum_of_squares (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_9 : a + b + c = 9) : 
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) ≥ 9 := by
sorry

end min_value_sum_of_squares_l1182_118237


namespace sue_shoe_probability_l1182_118250

/-- Represents the number of pairs for each shoe color --/
structure ShoeInventory where
  black : Nat
  brown : Nat
  gray : Nat
  red : Nat

/-- Calculates the probability of picking two shoes of the same color and opposite types --/
def probabilitySameColorOppositeTypes (inventory : ShoeInventory) : Rat :=
  let totalShoes := 2 * (inventory.black + inventory.brown + inventory.gray + inventory.red)
  let prob_black := (2 * inventory.black) / totalShoes * inventory.black / (totalShoes - 1)
  let prob_brown := (2 * inventory.brown) / totalShoes * inventory.brown / (totalShoes - 1)
  let prob_gray := (2 * inventory.gray) / totalShoes * inventory.gray / (totalShoes - 1)
  let prob_red := (2 * inventory.red) / totalShoes * inventory.red / (totalShoes - 1)
  prob_black + prob_brown + prob_gray + prob_red

/-- Sue's shoe inventory --/
def sueInventory : ShoeInventory := ⟨7, 4, 2, 2⟩

theorem sue_shoe_probability :
  probabilitySameColorOppositeTypes sueInventory = 73 / 435 := by
  sorry

end sue_shoe_probability_l1182_118250


namespace fraction_decomposition_l1182_118269

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ -4 ∧ x ≠ 2/3 →
    (7 * x - 15) / (3 * x^2 + 2 * x - 8) = A / (x + 4) + B / (3 * x - 2)) →
  A = 43 / 14 ∧ B = -31 / 14 := by
sorry

end fraction_decomposition_l1182_118269


namespace opposite_of_one_third_l1182_118240

theorem opposite_of_one_third : 
  -(1/3 : ℚ) = -1/3 := by sorry

end opposite_of_one_third_l1182_118240


namespace product_of_square_roots_l1182_118208

theorem product_of_square_roots (q : ℝ) (h : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (8 * q^3) * Real.sqrt (12 * q^5) = 12 * q^4 * Real.sqrt (10 * q) :=
by sorry

end product_of_square_roots_l1182_118208


namespace work_completion_theorem_l1182_118209

/-- Calculates the number of additional workers needed to complete a job earlier -/
def additional_workers (initial_workers : ℕ) (initial_days : ℕ) (actual_days : ℕ) : ℕ :=
  (initial_workers * initial_days / actual_days) - initial_workers

theorem work_completion_theorem (initial_workers : ℕ) (initial_days : ℕ) (actual_days : ℕ) 
  (h1 : initial_workers = 30)
  (h2 : initial_days = 8)
  (h3 : actual_days = 5) :
  additional_workers initial_workers initial_days actual_days = 18 := by
  sorry

#eval additional_workers 30 8 5

end work_completion_theorem_l1182_118209


namespace birthday_attendees_l1182_118272

theorem birthday_attendees (n : ℕ) : 
  (12 * (n + 2) = 16 * n) → n = 6 := by
  sorry

end birthday_attendees_l1182_118272


namespace sarah_walking_speed_l1182_118286

theorem sarah_walking_speed (v : ℝ) : 
  v > 0 → -- v is positive (walking speed)
  (6 / v + 6 / 4 = 3.5) → -- total time equation
  v = 3 := by
sorry

end sarah_walking_speed_l1182_118286


namespace parallel_vectors_imply_y_value_l1182_118275

/-- Given two vectors a and b in ℝ², prove that if a = (2,5) and b = (1,y) are parallel, then y = 5/2 -/
theorem parallel_vectors_imply_y_value (a b : ℝ × ℝ) (y : ℝ) :
  a = (2, 5) →
  b = (1, y) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  y = 5/2 := by
sorry

end parallel_vectors_imply_y_value_l1182_118275


namespace adjacent_probability_in_two_by_three_l1182_118245

/-- Represents a 2x3 seating arrangement -/
def SeatingArrangement := Fin 2 → Fin 3 → Fin 6

/-- Two positions are adjacent if they are next to each other in the same row or column -/
def adjacent (pos1 pos2 : Fin 2 × Fin 3) : Prop :=
  (pos1.1 = pos2.1 ∧ (pos1.2.val + 1 = pos2.2.val ∨ pos2.2.val + 1 = pos1.2.val)) ∨
  (pos1.2 = pos2.2 ∧ pos1.1 ≠ pos2.1)

/-- The probability of two specific students being adjacent in a random seating arrangement -/
def probability_adjacent : ℚ :=
  7 / 15

theorem adjacent_probability_in_two_by_three :
  probability_adjacent = 7 / 15 := by sorry

end adjacent_probability_in_two_by_three_l1182_118245


namespace range_of_a_l1182_118201

/-- The range of real number a when "x=1" is a sufficient but not necessary condition for "(x-a)[x-(a+2)]≤0" -/
theorem range_of_a : ∃ (a_min a_max : ℝ), a_min = -1 ∧ a_max = 1 ∧
  ∀ (a : ℝ), (∀ (x : ℝ), x = 1 → (x - a) * (x - (a + 2)) ≤ 0) ∧
             (∃ (x : ℝ), x ≠ 1 ∧ (x - a) * (x - (a + 2)) ≤ 0) ↔
             a_min ≤ a ∧ a ≤ a_max := by
  sorry

end range_of_a_l1182_118201


namespace min_triangle_area_l1182_118239

/-- The minimum area of the triangle formed by a line passing through (1, 2) 
    and intersecting the positive x and y axes is 4. -/
theorem min_triangle_area (k : ℝ) (h : k < 0) : 
  let f (x : ℝ) := k * (x - 1) + 2
  let x_intercept := 1 - 2 / k
  let y_intercept := f 0
  let area := (1/2) * x_intercept * y_intercept
  ∀ k, k < 0 → area ≥ 4 ∧ (area = 4 ↔ k = -2) :=
sorry

end min_triangle_area_l1182_118239


namespace vector_add_scale_l1182_118227

/-- Given two 3D vectors, prove that adding them and scaling the result by 2 yields the expected vector -/
theorem vector_add_scale (v1 v2 : Fin 3 → ℝ) (h1 : v1 = ![- 3, 2, 5]) (h2 : v2 = ![4, 7, - 3]) :
  (2 • (v1 + v2)) = ![2, 18, 4] := by
  sorry

end vector_add_scale_l1182_118227


namespace range_of_a_l1182_118288

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + a > 0

-- Define the theorem
theorem range_of_a (a : ℝ) : (¬(p a) ∧ q a) → (1 < a ∧ a < 4) := by
  sorry

end range_of_a_l1182_118288


namespace triangle_area_with_cosine_root_l1182_118231

theorem triangle_area_with_cosine_root (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → b = 5 → 
  5 * cos_theta^2 - 7 * cos_theta - 6 = 0 →
  cos_theta ≤ 1 →
  (1/2) * a * b * Real.sqrt (1 - cos_theta^2) = 6 := by
  sorry

end triangle_area_with_cosine_root_l1182_118231


namespace subtraction_result_l1182_118248

-- Define the two numbers
def a : ℚ := 888.88
def b : ℚ := 555.55

-- Define the result
def result : ℚ := a - b

-- Theorem to prove
theorem subtraction_result : result = 333.33 := by
  sorry

end subtraction_result_l1182_118248


namespace peaches_picked_proof_l1182_118216

/-- The number of peaches Mike had initially -/
def initial_peaches : ℕ := 34

/-- The total number of peaches Mike has now -/
def total_peaches : ℕ := 86

/-- The number of peaches Mike picked -/
def picked_peaches : ℕ := total_peaches - initial_peaches

theorem peaches_picked_proof :
  picked_peaches = total_peaches - initial_peaches :=
by sorry

end peaches_picked_proof_l1182_118216


namespace wire_length_ratio_l1182_118241

/-- Represents the construction of cube frames by Bonnie and Roark -/
structure CubeFrames where
  bonnie_wire_length : ℕ := 8
  bonnie_wire_pieces : ℕ := 12
  roark_wire_length : ℕ := 2

/-- Theorem stating the ratio of wire lengths used by Bonnie and Roark -/
theorem wire_length_ratio (cf : CubeFrames) : 
  (cf.bonnie_wire_length * cf.bonnie_wire_pieces : ℚ) / 
  (cf.roark_wire_length * 12 * (cf.bonnie_wire_length ^ 3 / cf.roark_wire_length ^ 3)) = 1 / 16 := by
  sorry

end wire_length_ratio_l1182_118241


namespace n_value_proof_l1182_118211

theorem n_value_proof (n : ℕ) 
  (h1 : ∃ k : ℕ, 31 * 13 * n = k)
  (h2 : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 2*x + 2*y + z = n)
  (h3 : (Finset.filter (λ (t : ℕ × ℕ × ℕ) => 
         t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧ 2*t.1 + 2*t.2.1 + t.2.2 = n) 
         (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card = 28) :
  n = 17 ∨ n = 18 := by
sorry

end n_value_proof_l1182_118211


namespace circle_area_from_circumference_l1182_118285

theorem circle_area_from_circumference (c : ℝ) (h : c = 36) : 
  (c^2 / (4 * π)) = 324 / π := by sorry

end circle_area_from_circumference_l1182_118285


namespace alpha_plus_beta_value_l1182_118289

theorem alpha_plus_beta_value (α β : ℝ) :
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 54*x + 621) / (x^2 + 42*x - 1764)) →
  α + β = 86 := by
sorry

end alpha_plus_beta_value_l1182_118289


namespace share_ratio_l1182_118270

/-- Given a total amount of $500 divided among three people a, b, and c,
    where a's share is $200, a gets a fraction of b and c's combined share,
    and b gets 6/9 of a and c's combined share, prove that the ratio of
    a's share to the combined share of b and c is 2:3. -/
theorem share_ratio (total : ℚ) (a b c : ℚ) :
  total = 500 →
  a = 200 →
  ∃ x : ℚ, a = x * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a / (b + c) = 2/3 :=
by sorry

end share_ratio_l1182_118270


namespace complement_A_inter_B_range_of_a_l1182_118200

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}
def C (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Theorem for the complement of A ∩ B
theorem complement_A_inter_B :
  (A ∩ B)ᶜ = {x | x < 3 ∨ x > 6} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) (h : A ∪ C a = C a) :
  a ≥ 6 := by sorry

end complement_A_inter_B_range_of_a_l1182_118200


namespace arithmetic_sequence_problem_l1182_118296

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₇ = 3 and a₁₉ = 2011, prove that a₁₃ = 1007 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_7 : a 7 = 3) 
  (h_19 : a 19 = 2011) : 
  a 13 = 1007 := by
  sorry

end arithmetic_sequence_problem_l1182_118296


namespace sin_960_degrees_l1182_118207

theorem sin_960_degrees : Real.sin (960 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_960_degrees_l1182_118207


namespace vat_percentage_calculation_l1182_118290

theorem vat_percentage_calculation (original_price final_price : ℝ) : 
  original_price = 1700 → 
  final_price = 1955 → 
  (final_price - original_price) / original_price * 100 = 15 := by
sorry

end vat_percentage_calculation_l1182_118290


namespace casper_candy_problem_l1182_118259

def candy_distribution (initial : ℕ) : ℕ :=
  let day1 := initial * 3 / 4 - 3
  let day2 := day1 * 4 / 5 - 5
  let day3 := day2 * 5 / 6 - 6
  day3

theorem casper_candy_problem :
  ∃ (initial : ℕ), candy_distribution initial = 10 ∧ initial = 678 :=
sorry

end casper_candy_problem_l1182_118259


namespace x_less_than_y_l1182_118283

theorem x_less_than_y (x y : ℝ) (h : (2023 : ℝ)^x + (2024 : ℝ)^(-y) < (2023 : ℝ)^y + (2024 : ℝ)^(-x)) : x < y := by
  sorry

end x_less_than_y_l1182_118283


namespace complex_square_root_of_negative_two_l1182_118202

theorem complex_square_root_of_negative_two (z : ℂ) : z^2 + 2 = 0 → z = Complex.I * Real.sqrt 2 ∨ z = -Complex.I * Real.sqrt 2 := by
  sorry

end complex_square_root_of_negative_two_l1182_118202


namespace expression_equality_l1182_118264

theorem expression_equality : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end expression_equality_l1182_118264


namespace louis_age_l1182_118249

/-- Given the ages of Matilda, Jerica, and Louis, prove Louis' age -/
theorem louis_age (matilda_age jerica_age louis_age : ℕ) : 
  matilda_age = 35 →
  matilda_age = jerica_age + 7 →
  jerica_age = 2 * louis_age →
  louis_age = 14 := by
  sorry

#check louis_age

end louis_age_l1182_118249


namespace sin_2alpha_value_l1182_118287

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π/4 - α) = 3/5) : 
  Real.sin (2 * α) = -7/25 := by
  sorry

end sin_2alpha_value_l1182_118287


namespace integer_sum_and_square_is_ten_l1182_118206

theorem integer_sum_and_square_is_ten (N : ℤ) : N^2 + N = 10 → N = 2 ∨ N = -5 := by
  sorry

end integer_sum_and_square_is_ten_l1182_118206


namespace even_function_inequality_range_l1182_118221

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_function_inequality_range
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f (2*x - 1) < f (-1)} = Set.Ioo 0 1 := by
sorry

end even_function_inequality_range_l1182_118221


namespace total_weight_lifted_l1182_118292

/-- Represents the weight a weightlifter can lift in one hand -/
def weight_per_hand : ℕ := 10

/-- Represents the number of hands a weightlifter has -/
def number_of_hands : ℕ := 2

/-- Theorem stating the total weight a weightlifter can lift -/
theorem total_weight_lifted : weight_per_hand * number_of_hands = 20 := by
  sorry

end total_weight_lifted_l1182_118292


namespace carries_shopping_money_l1182_118236

theorem carries_shopping_money (initial_amount sweater_cost tshirt_cost shoes_cost : ℕ) 
  (h1 : initial_amount = 91)
  (h2 : sweater_cost = 24)
  (h3 : tshirt_cost = 6)
  (h4 : shoes_cost = 11) :
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost) = 50 := by
  sorry

end carries_shopping_money_l1182_118236


namespace complex_subtraction_simplification_l1182_118271

theorem complex_subtraction_simplification :
  (4 - 3*Complex.I) - (7 - 5*Complex.I) = -3 + 2*Complex.I := by
  sorry

end complex_subtraction_simplification_l1182_118271


namespace rowan_rowing_distance_l1182_118252

-- Define the given constants
def downstream_time : ℝ := 2
def upstream_time : ℝ := 4
def still_water_speed : ℝ := 9.75

-- Define the variables
def current_speed : ℝ := sorry
def distance : ℝ := sorry

-- State the theorem
theorem rowan_rowing_distance :
  downstream_time = distance / (still_water_speed + current_speed) ∧
  upstream_time = distance / (still_water_speed - current_speed) →
  distance = 26 := by sorry

end rowan_rowing_distance_l1182_118252


namespace ellipse_equation_l1182_118223

-- Define the ellipse
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  distance_sum : ℝ

-- Define the standard form of an ellipse equation
def standard_ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem ellipse_equation (e : Ellipse) (x y : ℝ) : 
  e.foci = ((-4, 0), (4, 0)) →
  e.distance_sum = 10 →
  standard_ellipse_equation 25 9 x y :=
sorry

end ellipse_equation_l1182_118223


namespace greatest_prime_factor_of_99_l1182_118219

theorem greatest_prime_factor_of_99 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 99 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 99 → q ≤ p :=
  sorry

end greatest_prime_factor_of_99_l1182_118219


namespace constant_product_equals_one_fourth_l1182_118280

/-- Given a function f(x) = (bx + 1) / (2x + a), where a and b are constants,
    and ab ≠ 2, prove that if f(x) * f(1/x) is constant for all x ≠ 0,
    then this constant equals 1/4. -/
theorem constant_product_equals_one_fourth
  (a b : ℝ) (h : a * b ≠ 2)
  (f : ℝ → ℝ)
  (hf : ∀ x, x ≠ 0 → f x = (b * x + 1) / (2 * x + a))
  (h_constant : ∃ k, ∀ x, x ≠ 0 → f x * f (1/x) = k) :
  ∃ k, (∀ x, x ≠ 0 → f x * f (1/x) = k) ∧ k = 1/4 :=
by sorry

end constant_product_equals_one_fourth_l1182_118280


namespace interest_rate_proof_l1182_118279

/-- Proves that given specific conditions, the interest rate is 18% --/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (interest_difference : ℝ) : 
  principal = 4000 →
  time = 2 →
  interest_difference = 480 →
  (principal * time * (18 / 100)) = (principal * time * (12 / 100) + interest_difference) :=
by sorry

end interest_rate_proof_l1182_118279


namespace dime_probability_l1182_118262

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℕ
  | Coin.Quarter => 1250
  | Coin.Dime => 500
  | Coin.Penny => 250

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Quarter + coinCount Coin.Dime + coinCount Coin.Penny

/-- The probability of randomly choosing a dime from the jar -/
def probDime : ℚ := coinCount Coin.Dime / totalCoins

theorem dime_probability : probDime = 1 / 7 := by
  sorry

#eval probDime

end dime_probability_l1182_118262


namespace amy_hair_length_l1182_118220

/-- Amy's hair length before the haircut -/
def hair_length_before : ℕ := 11

/-- Amy's hair length after the haircut -/
def hair_length_after : ℕ := 7

/-- Length of hair cut off -/
def hair_cut_off : ℕ := 4

/-- Theorem: Amy's hair length before the haircut was 11 inches -/
theorem amy_hair_length : hair_length_before = hair_length_after + hair_cut_off := by
  sorry

end amy_hair_length_l1182_118220


namespace a_will_eat_hat_l1182_118263

-- Define the types of people
inductive Person : Type
| Knight : Person
| Liar : Person

-- Define the statement made by A about B
def statement_about_B (a b : Person) : Prop :=
  match a with
  | Person.Knight => b = Person.Knight
  | Person.Liar => True

-- Define A's statement about eating the hat
def statement_about_hat (a : Person) : Prop :=
  match a with
  | Person.Knight => True  -- Will eat the hat
  | Person.Liar => False   -- Won't eat the hat

-- Theorem statement
theorem a_will_eat_hat (a b : Person) :
  (statement_about_B a b = True) →
  (statement_about_hat a = True) := by
  sorry


end a_will_eat_hat_l1182_118263


namespace drums_per_day_l1182_118254

/-- Given that 2916 drums of grapes are filled in 9 days, 
    prove that 324 drums are filled per day. -/
theorem drums_per_day : 
  ∀ (total_drums : ℕ) (total_days : ℕ) (drums_per_day : ℕ),
    total_drums = 2916 →
    total_days = 9 →
    drums_per_day = total_drums / total_days →
    drums_per_day = 324 := by
  sorry

end drums_per_day_l1182_118254


namespace backpack_price_change_l1182_118226

theorem backpack_price_change (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.64 * P →
  x = 60 := by
  sorry

end backpack_price_change_l1182_118226


namespace beads_taken_out_l1182_118256

/-- Represents the number of beads in a container -/
structure BeadContainer where
  green : Nat
  brown : Nat
  red : Nat

/-- Calculates the total number of beads in a container -/
def totalBeads (container : BeadContainer) : Nat :=
  container.green + container.brown + container.red

theorem beads_taken_out (initial : BeadContainer) (left : Nat) :
  totalBeads initial = 6 → left = 4 → totalBeads initial - left = 2 := by
  sorry

end beads_taken_out_l1182_118256


namespace triangle_sides_product_square_l1182_118238

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

theorem triangle_sides_product_square (a b c : ℤ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Positive integers
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  (∀ d : ℤ, d > 1 → (d ∣ a ∧ d ∣ b ∧ d ∣ c) → False) →  -- No common divisor > 1
  (∃ k : ℤ, (a^2 + b^2 - c^2) = k * (a + b - c)) →  -- First fraction is integer
  (∃ l : ℤ, (b^2 + c^2 - a^2) = l * (b + c - a)) →  -- Second fraction is integer
  (∃ m : ℤ, (c^2 + a^2 - b^2) = m * (c + a - b)) →  -- Third fraction is integer
  (is_perfect_square ((a + b - c) * (b + c - a) * (c + a - b)) ∨ 
   is_perfect_square (2 * (a + b - c) * (b + c - a) * (c + a - b))) :=
by sorry

end triangle_sides_product_square_l1182_118238


namespace vacation_rental_families_l1182_118274

/-- The number of people in each family -/
def family_size : ℕ := 4

/-- The number of days of the vacation -/
def vacation_days : ℕ := 7

/-- The number of towels each person uses per day -/
def towels_per_person_per_day : ℕ := 1

/-- The capacity of the washing machine in towels -/
def washing_machine_capacity : ℕ := 14

/-- The number of loads needed to wash all towels -/
def total_loads : ℕ := 6

/-- The number of families sharing the vacation rental -/
def num_families : ℕ := 3

theorem vacation_rental_families :
  num_families * family_size * vacation_days * towels_per_person_per_day =
  total_loads * washing_machine_capacity := by sorry

end vacation_rental_families_l1182_118274


namespace triangle_has_obtuse_angle_l1182_118217

/-- A triangle with vertices A(1, 2), B(-3, 4), and C(0, -2) has an obtuse angle. -/
theorem triangle_has_obtuse_angle :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (-3, 4)
  let C : ℝ × ℝ := (0, -2)
  let AB : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC : ℝ := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  BC^2 > AB^2 + AC^2 := by sorry

end triangle_has_obtuse_angle_l1182_118217


namespace problem_proof_l1182_118244

theorem problem_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ a*b ≤ x*y) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → a*b ≤ 1/8) ∧
  ((1/b) + (b/a) ≥ 4) ∧
  (a^2 + b^2 ≥ 1/5) :=
sorry

end problem_proof_l1182_118244


namespace prob_adjacent_vertices_decagon_l1182_118251

/-- A decagon is a polygon with 10 vertices -/
def Decagon := { n : ℕ // n = 10 }

/-- The number of vertices in a decagon -/
def num_vertices (d : Decagon) : ℕ := d.val

/-- The number of adjacent vertices for any vertex in a decagon -/
def num_adjacent_vertices (d : Decagon) : ℕ := 2

/-- The probability of choosing two distinct adjacent vertices in a decagon -/
def prob_adjacent_vertices (d : Decagon) : ℚ :=
  (num_adjacent_vertices d : ℚ) / ((num_vertices d - 1) : ℚ)

/-- Theorem: The probability of choosing two distinct adjacent vertices in a decagon is 2/9 -/
theorem prob_adjacent_vertices_decagon (d : Decagon) :
  prob_adjacent_vertices d = 2 / 9 := by
  sorry

end prob_adjacent_vertices_decagon_l1182_118251


namespace sqrt_x_fifth_power_eq_1024_l1182_118204

theorem sqrt_x_fifth_power_eq_1024 (x : ℝ) : (Real.sqrt x) ^ 5 = 1024 → x = 16 := by
  sorry

end sqrt_x_fifth_power_eq_1024_l1182_118204


namespace distance_foci_to_asymptotes_for_given_hyperbola_l1182_118255

/-- The distance from the foci to the asymptotes of a hyperbola -/
def distance_foci_to_asymptotes (a b : ℝ) : ℝ := b

/-- The equation of a hyperbola in standard form -/
def is_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem distance_foci_to_asymptotes_for_given_hyperbola :
  ∀ x y : ℝ, is_hyperbola x y 1 3 → distance_foci_to_asymptotes 1 3 = 3 := by
  sorry

end distance_foci_to_asymptotes_for_given_hyperbola_l1182_118255


namespace parallel_segments_y_coordinate_l1182_118297

/-- Given four points A, B, X, Y on a Cartesian plane where AB is parallel to XY,
    prove that the y-coordinate of Y is 5. -/
theorem parallel_segments_y_coordinate (A B X Y : ℝ × ℝ) : 
  A = (-6, 2) →
  B = (2, -2) →
  X = (-2, 10) →
  Y.1 = 8 →
  (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1) →
  Y.2 = 5 := by
sorry

end parallel_segments_y_coordinate_l1182_118297


namespace negation_of_existential_quantifier_negation_of_inequality_l1182_118293

theorem negation_of_existential_quantifier (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 2 > 0) :=
by sorry

end negation_of_existential_quantifier_negation_of_inequality_l1182_118293


namespace max_d_value_l1182_118282

def a (n : ℕ+) : ℕ := 150 + 3 * n.val ^ 2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ (n : ℕ+), d n = 147) ∧ (∀ (n : ℕ+), d n ≤ 147) :=
sorry

end max_d_value_l1182_118282


namespace food_price_calculation_l1182_118267

/-- The original food price before tax and tip -/
def original_price : ℝ := 160

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.1

/-- The tip rate -/
def tip_rate : ℝ := 0.2

/-- The total bill amount -/
def total_bill : ℝ := 211.20

theorem food_price_calculation :
  original_price * (1 + sales_tax_rate) * (1 + tip_rate) = total_bill := by
  sorry


end food_price_calculation_l1182_118267


namespace sqrt_sum_problem_l1182_118233

theorem sqrt_sum_problem (a b : ℝ) 
  (h1 : Real.sqrt 44 = 2 * Real.sqrt a) 
  (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : 
  a + b = 17 := by
sorry

end sqrt_sum_problem_l1182_118233


namespace travel_distance_ratio_l1182_118268

theorem travel_distance_ratio (total_distance train_distance : ℝ)
  (h1 : total_distance = 500)
  (h2 : train_distance = 300)
  (h3 : ∃ bus_distance cab_distance : ℝ,
    total_distance = train_distance + bus_distance + cab_distance ∧
    cab_distance = (1/3) * bus_distance) :
  ∃ bus_distance : ℝ, bus_distance / train_distance = 1/2 :=
sorry

end travel_distance_ratio_l1182_118268


namespace x_squared_plus_inverse_squared_l1182_118224

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end x_squared_plus_inverse_squared_l1182_118224


namespace min_value_reciprocal_sum_l1182_118291

/-- Given a function y = x^α where α < 0, and a point A that lies on both y = x^α and y = mx + n 
    where m > 0 and n > 0, the minimum value of 1/m + 1/n is 4. -/
theorem min_value_reciprocal_sum (α m n : ℝ) (hα : α < 0) (hm : m > 0) (hn : n > 0) :
  (∃ x y : ℝ, y = x^α ∧ y = m*x + n) → 
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → (∃ x' y' : ℝ, y' = x'^α ∧ y' = m'*x' + n') → 1/m + 1/n ≤ 1/m' + 1/n') →
  1/m + 1/n = 4 :=
by sorry

end min_value_reciprocal_sum_l1182_118291


namespace intersection_A_B_when_m_zero_range_of_m_for_necessary_condition_l1182_118210

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

-- Theorem for part I
theorem intersection_A_B_when_m_zero :
  A ∩ B 0 = {x | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem for part II
theorem range_of_m_for_necessary_condition :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) ↔ m ≥ 4 ∨ m ≤ -2 := by sorry

end intersection_A_B_when_m_zero_range_of_m_for_necessary_condition_l1182_118210


namespace min_value_constraint_l1182_118278

theorem min_value_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  x^2 + 8*x*y + 25*y^2 + 16*y*z + 9*z^2 ≥ 403/9 := by
  sorry

end min_value_constraint_l1182_118278


namespace john_burritos_per_day_l1182_118232

theorem john_burritos_per_day 
  (boxes : ℕ) 
  (burritos_per_box : ℕ) 
  (days : ℕ) 
  (remaining : ℕ) 
  (h1 : boxes = 3) 
  (h2 : burritos_per_box = 20) 
  (h3 : days = 10) 
  (h4 : remaining = 10) : 
  (boxes * burritos_per_box - boxes * burritos_per_box / 3 - remaining) / days = 3 := by
  sorry

end john_burritos_per_day_l1182_118232


namespace discount_equation_l1182_118213

/-- Represents the discount scenario for a clothing item -/
structure DiscountScenario where
  original_price : ℝ
  final_price : ℝ
  discount_percentage : ℝ

/-- Theorem stating the relationship between original price, discount, and final price -/
theorem discount_equation (scenario : DiscountScenario) 
  (h1 : scenario.original_price = 280)
  (h2 : scenario.final_price = 177)
  (h3 : scenario.discount_percentage ≥ 0)
  (h4 : scenario.discount_percentage < 1) :
  scenario.original_price * (1 - scenario.discount_percentage)^2 = scenario.final_price := by
  sorry

#check discount_equation

end discount_equation_l1182_118213


namespace dragon_jewels_l1182_118230

theorem dragon_jewels (x : ℕ) (h1 : x / 3 = 6) : x + 6 = 24 := by
  sorry

#check dragon_jewels

end dragon_jewels_l1182_118230


namespace simplify_expression_l1182_118284

theorem simplify_expression (r : ℝ) : 90 * r - 44 * r = 46 * r := by
  sorry

end simplify_expression_l1182_118284


namespace min_value_theorem_l1182_118243

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 :=
by sorry

end min_value_theorem_l1182_118243


namespace hook_all_of_one_color_l1182_118218

/-- Represents a square sheet on the table -/
structure Sheet where
  color : Nat
  deriving Repr

/-- Represents the rectangular table with sheets -/
structure Table where
  sheets : List Sheet
  num_colors : Nat
  deriving Repr

/-- Two sheets can be hooked together -/
def can_hook (s1 s2 : Sheet) : Prop := sorry

/-- All sheets of the same color can be hooked together using the given number of hooks -/
def can_hook_color (t : Table) (c : Nat) (hooks : Nat) : Prop := sorry

/-- For any k different colored sheets, two can be hooked together -/
axiom hook_property (t : Table) :
  ∀ (diff_colored_sheets : List Sheet),
    diff_colored_sheets.length = t.num_colors →
    (∀ (s1 s2 : Sheet), s1 ∈ diff_colored_sheets → s2 ∈ diff_colored_sheets → s1 ≠ s2 → s1.color ≠ s2.color) →
    ∃ (s1 s2 : Sheet), s1 ∈ diff_colored_sheets ∧ s2 ∈ diff_colored_sheets ∧ s1 ≠ s2 ∧ can_hook s1 s2

/-- Main theorem: It's possible to hook all sheets of a certain color using 2k-2 hooks -/
theorem hook_all_of_one_color (t : Table) (h : t.num_colors ≥ 2) :
  ∃ (c : Nat), can_hook_color t c (2 * t.num_colors - 2) := by
  sorry

end hook_all_of_one_color_l1182_118218


namespace x_value_l1182_118247

theorem x_value (x y : ℚ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 := by
  sorry

end x_value_l1182_118247


namespace replacement_cost_theorem_l1182_118266

/-- Calculate the total cost of replacing cardio machines in multiple gyms -/
def total_replacement_cost (num_gyms : ℕ) (bikes_per_gym : ℕ) (treadmills_per_gym : ℕ) (ellipticals_per_gym : ℕ) (bike_cost : ℝ) : ℝ :=
  let total_bikes := num_gyms * bikes_per_gym
  let total_treadmills := num_gyms * treadmills_per_gym
  let total_ellipticals := num_gyms * ellipticals_per_gym
  let treadmill_cost := bike_cost * 1.5
  let elliptical_cost := treadmill_cost * 2
  total_bikes * bike_cost + total_treadmills * treadmill_cost + total_ellipticals * elliptical_cost

/-- Theorem: The total cost to replace all cardio machines in 20 gyms is $455,000 -/
theorem replacement_cost_theorem :
  total_replacement_cost 20 10 5 5 700 = 455000 := by
  sorry

end replacement_cost_theorem_l1182_118266


namespace quadratic_roots_l1182_118234

theorem quadratic_roots (x : ℝ) : x * (x + 1) = 3 * (x + 1) ↔ x = 3 ∨ x = -1 := by
  sorry

end quadratic_roots_l1182_118234


namespace seven_numbers_divisible_by_three_l1182_118203

theorem seven_numbers_divisible_by_three (S : Finset ℕ) (h : S.card = 7) :
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c) % 3 = 0 := by
  sorry

end seven_numbers_divisible_by_three_l1182_118203


namespace inscribed_quadrilateral_perimeter_l1182_118253

/-- Given a rectangle with sides a and b, and an inscribed quadrilateral with vertices on each side
of the rectangle, the perimeter of the quadrilateral is greater than or equal to 2√(a² + b²). -/
theorem inscribed_quadrilateral_perimeter (a b : ℝ) (x y z t : ℝ)
  (hx : 0 ≤ x ∧ x ≤ a) (hy : 0 ≤ y ∧ y ≤ b) (hz : 0 ≤ z ∧ z ≤ a) (ht : 0 ≤ t ∧ t ≤ b) :
  let perimeter := Real.sqrt ((a - x)^2 + t^2) + Real.sqrt ((b - t)^2 + z^2) +
                   Real.sqrt ((a - z)^2 + (b - y)^2) + Real.sqrt (x^2 + y^2)
  perimeter ≥ 2 * Real.sqrt (a^2 + b^2) := by sorry

end inscribed_quadrilateral_perimeter_l1182_118253


namespace luna_pink_crayons_percentage_l1182_118222

/-- Given information about Mara's and Luna's crayons, prove that 20% of Luna's crayons are pink -/
theorem luna_pink_crayons_percentage
  (mara_total : ℕ)
  (mara_pink_percent : ℚ)
  (luna_total : ℕ)
  (total_pink : ℕ)
  (h1 : mara_total = 40)
  (h2 : mara_pink_percent = 1/10)
  (h3 : luna_total = 50)
  (h4 : total_pink = 14)
  : (luna_total - (mara_pink_percent * mara_total).floor) / luna_total = 1/5 := by
  sorry

#eval (50 : ℚ) / 5  -- Expected output: 10

end luna_pink_crayons_percentage_l1182_118222


namespace mark_bench_press_l1182_118242

-- Define the given conditions
def dave_weight : ℝ := 175
def dave_bench_press_multiplier : ℝ := 3
def craig_bench_press_percentage : ℝ := 0.20
def emma_bench_press_percentage : ℝ := 0.75
def emma_bench_press_increase : ℝ := 15
def john_bench_press_multiplier : ℝ := 2
def mark_bench_press_difference : ℝ := 50

-- Define the theorem
theorem mark_bench_press :
  let dave_bench_press := dave_weight * dave_bench_press_multiplier
  let craig_bench_press := craig_bench_press_percentage * dave_bench_press
  let emma_bench_press := emma_bench_press_percentage * dave_bench_press + emma_bench_press_increase
  let combined_craig_emma := craig_bench_press + emma_bench_press
  let mark_bench_press := combined_craig_emma - mark_bench_press_difference
  mark_bench_press = 463.75 := by
  sorry

end mark_bench_press_l1182_118242


namespace product_of_first_six_terms_l1182_118294

/-- A geometric sequence with the given property -/
def GeometricSequenceWithProperty (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) * a (2 * n) = 3^n

/-- The theorem to be proved -/
theorem product_of_first_six_terms
  (a : ℕ → ℝ)
  (h : GeometricSequenceWithProperty a) :
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 729 := by
  sorry

end product_of_first_six_terms_l1182_118294


namespace largest_811_triple_l1182_118273

/-- Converts a base-10 number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

/-- Converts a list of base-8 digits to a base-10 number -/
def fromBase8 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a list of digits to a base-10 number -/
def toBase10 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is an 8-11 triple -/
def is811Triple (m : ℕ) : Prop :=
  let base8Digits := toBase8 m
  toBase10 base8Digits = 3 * m

/-- The largest 8-11 triple -/
def largestTriple : ℕ := 705

theorem largest_811_triple :
  is811Triple largestTriple ∧
  ∀ m : ℕ, m > largestTriple → ¬is811Triple m :=
by sorry

end largest_811_triple_l1182_118273


namespace last_four_digits_of_m_l1182_118212

theorem last_four_digits_of_m (M : ℕ) (h1 : M > 0) 
  (h2 : ∃ (a b c d e : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    M % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (M^2) % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e) :
  M % 10000 = 9687 := by
sorry

end last_four_digits_of_m_l1182_118212


namespace min_toothpicks_removal_l1182_118277

/-- Represents a triangular figure made of toothpicks -/
structure TriangularFigure where
  toothpicks : ℕ
  triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : TriangularFigure) : ℕ :=
  15

/-- Theorem: For a triangular figure with 40 toothpicks and at least 35 triangles,
    the minimum number of toothpicks to remove to eliminate all triangles is 15 -/
theorem min_toothpicks_removal (figure : TriangularFigure) 
    (h1 : figure.toothpicks = 40) 
    (h2 : figure.triangles ≥ 35) : 
  min_toothpicks_to_remove figure = 15 :=
by
  sorry


end min_toothpicks_removal_l1182_118277


namespace polynomial_factorization_l1182_118257

theorem polynomial_factorization (x y z : ℝ) : 
  x^3 * (y - z) + y^3 * (z - x) + z^3 * (x - y) = (x + y + z) * (x - y) * (y - z) * (z - x) := by
  sorry

end polynomial_factorization_l1182_118257


namespace emilys_waist_size_conversion_l1182_118228

/-- Conversion of Emily's waist size from inches to centimeters -/
theorem emilys_waist_size_conversion (inches_per_foot : ℝ) (cm_per_foot : ℝ) (waist_inches : ℝ) :
  inches_per_foot = 12 →
  cm_per_foot = 30.48 →
  waist_inches = 28 →
  ∃ (waist_cm : ℝ), abs (waist_cm - (waist_inches / inches_per_foot * cm_per_foot)) < 0.1 ∧ waist_cm = 71.1 := by
  sorry

end emilys_waist_size_conversion_l1182_118228
