import Mathlib

namespace NUMINAMATH_CALUDE_inequality_theorem_l2940_294068

theorem inequality_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * (x + y + z + Real.sqrt (x^2 + y^2 + z^2))) / 
  ((x^2 + y^2 + z^2) * (y*z + z*x + x*y)) ≤ (3 + Real.sqrt 3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2940_294068


namespace NUMINAMATH_CALUDE_inequality_proof_l2940_294065

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2*y*z)) + (y^2 / (y^2 + 2*z*x)) + (z^2 / (z^2 + 2*x*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2940_294065


namespace NUMINAMATH_CALUDE_function_monotonicity_and_extrema_l2940_294079

noncomputable section

variable (a : ℝ)
variable (k : ℝ)

def f (x : ℝ) : ℝ := (x - a - 1) * Real.exp x - (1/2) * x^2 + a * x

theorem function_monotonicity_and_extrema (h : a > 0) :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 0 → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, (∀ x, f a x₁ ≤ f a x) ∧ (∀ x, f a x₂ ≥ f a x) →
    (∀ a, a > 0 → f a x₁ - f a x₂ < k * a^3) ↔ k ≥ -1/6) :=
sorry

end

end NUMINAMATH_CALUDE_function_monotonicity_and_extrema_l2940_294079


namespace NUMINAMATH_CALUDE_jack_needs_five_rocks_l2940_294097

/-- The number of rocks needed to equalize weights on a see-saw -/
def rocks_needed (jack_weight anna_weight rock_weight : ℕ) : ℕ :=
  (jack_weight - anna_weight) / rock_weight

/-- Theorem: Jack needs 5 rocks to equalize weights with Anna -/
theorem jack_needs_five_rocks :
  rocks_needed 60 40 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jack_needs_five_rocks_l2940_294097


namespace NUMINAMATH_CALUDE_sticker_distribution_count_l2940_294032

/-- The number of ways to partition n identical objects into k or fewer non-negative integer parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution_count : 
  partition_count num_stickers num_sheets = 30 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_count_l2940_294032


namespace NUMINAMATH_CALUDE_employee_count_l2940_294005

/-- Proves the number of employees given salary information -/
theorem employee_count 
  (avg_salary : ℝ) 
  (salary_increase : ℝ) 
  (manager_salary : ℝ) 
  (h1 : avg_salary = 1700)
  (h2 : salary_increase = 100)
  (h3 : manager_salary = 3800) :
  ∃ (E : ℕ), 
    (E : ℝ) * (avg_salary + salary_increase) = E * avg_salary + manager_salary ∧ 
    E = 20 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l2940_294005


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2940_294054

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℚ) : Prop := m₁ = m₂

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
def slope_general_form (a b : ℚ) : ℚ := -a / b

/-- The slope of a line in the form y = mx + b is m -/
def slope_slope_intercept_form (m : ℚ) : ℚ := m

theorem parallel_lines_m_value :
  ∀ m : ℚ, 
  parallel (slope_general_form 2 m) (slope_slope_intercept_form 3) →
  m = -2/3 := by
sorry


end NUMINAMATH_CALUDE_parallel_lines_m_value_l2940_294054


namespace NUMINAMATH_CALUDE_tetrahedron_fits_in_box_l2940_294039

theorem tetrahedron_fits_in_box :
  ∀ (tetra_edge box_length box_width box_height : ℝ),
    tetra_edge = 12 →
    box_length = 9 ∧ box_width = 13 ∧ box_height = 15 →
    ∃ (cube_edge : ℝ),
      cube_edge = tetra_edge / Real.sqrt 2 ∧
      cube_edge ≤ box_length ∧
      cube_edge ≤ box_width ∧
      cube_edge ≤ box_height :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_fits_in_box_l2940_294039


namespace NUMINAMATH_CALUDE_bert_pencil_usage_l2940_294092

/-- The number of words Bert writes to use up a pencil -/
def words_per_pencil (puzzles_per_day : ℕ) (days_per_pencil : ℕ) (words_per_puzzle : ℕ) : ℕ :=
  puzzles_per_day * days_per_pencil * words_per_puzzle

/-- Theorem stating that Bert writes 1050 words to use up a pencil -/
theorem bert_pencil_usage :
  words_per_pencil 1 14 75 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_bert_pencil_usage_l2940_294092


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l2940_294073

theorem square_difference_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 60) 
  (diff_eq : x - y = 16) : 
  x^2 - y^2 = 960 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l2940_294073


namespace NUMINAMATH_CALUDE_club_member_age_difference_l2940_294045

/-- Given a club with 10 members, prove that replacing one member
    results in a 50-year difference between the old and new member's ages
    if the average age remains the same after 5 years. -/
theorem club_member_age_difference
  (n : ℕ) -- number of club members
  (a : ℝ) -- average age of members 5 years ago
  (o : ℝ) -- age of the old (replaced) member
  (n' : ℝ) -- age of the new member
  (h1 : n = 10) -- there are 10 members
  (h2 : n * a = (n - 1) * (a + 5) + n') -- average age remains the same after 5 years and replacement
  : |o - n'| = 50 := by
  sorry


end NUMINAMATH_CALUDE_club_member_age_difference_l2940_294045


namespace NUMINAMATH_CALUDE_rectangular_field_fencing_costs_l2940_294060

/-- Given a rectangular field with sides in the ratio of 3:4 and an area of 8112 sq.m,
    prove the perimeter and fencing costs for different materials. -/
theorem rectangular_field_fencing_costs 
  (ratio : ℚ) 
  (area : ℝ) 
  (wrought_iron_cost : ℝ) 
  (wooden_cost : ℝ) 
  (chain_link_cost : ℝ) :
  ratio = 3 / 4 →
  area = 8112 →
  wrought_iron_cost = 45 →
  wooden_cost = 35 →
  chain_link_cost = 25 →
  ∃ (perimeter : ℝ) 
    (wrought_iron_total : ℝ) 
    (wooden_total : ℝ) 
    (chain_link_total : ℝ),
    perimeter = 364 ∧
    wrought_iron_total = 16380 ∧
    wooden_total = 12740 ∧
    chain_link_total = 9100 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_fencing_costs_l2940_294060


namespace NUMINAMATH_CALUDE_ice_cream_cost_l2940_294029

theorem ice_cream_cost (people : ℕ) (meal_cost : ℚ) (total_amount : ℚ) 
  (h1 : people = 3)
  (h2 : meal_cost = 10)
  (h3 : total_amount = 45)
  (h4 : total_amount ≥ people * meal_cost) :
  (total_amount - people * meal_cost) / people = 5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l2940_294029


namespace NUMINAMATH_CALUDE_number_division_problem_l2940_294093

theorem number_division_problem (x y : ℝ) : 
  (x - 5) / y = 7 → (x - 24) / 10 = 3 → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2940_294093


namespace NUMINAMATH_CALUDE_ratio_problem_l2940_294046

theorem ratio_problem (first_part : ℝ) (ratio_percent : ℝ) (second_part : ℝ) :
  first_part = 5 →
  ratio_percent = 25 →
  first_part / (first_part + second_part) = ratio_percent / 100 →
  second_part = 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2940_294046


namespace NUMINAMATH_CALUDE_right_triangle_roots_l2940_294028

/-- Given complex numbers a and b, and complex roots z₁ and z₂ of z² + az + b = 0
    such that 0, z₁, and z₂ form a right triangle with z₂ opposite the right angle,
    prove that a²/b = 2 -/
theorem right_triangle_roots (a b z₁ z₂ : ℂ) 
    (h_root : z₁^2 + a*z₁ + b = 0 ∧ z₂^2 + a*z₂ + b = 0)
    (h_right_triangle : z₂ = z₁ * Complex.I) : 
    a^2 / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_roots_l2940_294028


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2940_294072

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (2 - 3 * Complex.I) / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2940_294072


namespace NUMINAMATH_CALUDE_marbles_cost_marbles_cost_value_l2940_294091

def total_spent : ℚ := 20.52
def football_cost : ℚ := 4.95
def baseball_cost : ℚ := 6.52

theorem marbles_cost : ℚ :=
  total_spent - (football_cost + baseball_cost)

#check marbles_cost

theorem marbles_cost_value : marbles_cost = 9.05 := by sorry

end NUMINAMATH_CALUDE_marbles_cost_marbles_cost_value_l2940_294091


namespace NUMINAMATH_CALUDE_cream_needed_proof_l2940_294088

/-- The amount of additional cream needed when given a total required amount and an available amount -/
def additional_cream_needed (total_required : ℕ) (available : ℕ) : ℕ :=
  total_required - available

/-- Theorem stating that given 300 lbs total required and 149 lbs available, 151 lbs additional cream is needed -/
theorem cream_needed_proof :
  additional_cream_needed 300 149 = 151 := by
  sorry

end NUMINAMATH_CALUDE_cream_needed_proof_l2940_294088


namespace NUMINAMATH_CALUDE_v3_at_neg_one_l2940_294064

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 + 0.35*x + 1.8*x^2 - 3*x^3 + 6*x^4 - 5*x^5 + x^6

/-- v3 in Horner's method for f(x) -/
def v3 (x : ℝ) : ℝ := (((x - 5)*x + 6)*x - 3)

/-- Theorem: v3 equals -15 when x = -1 -/
theorem v3_at_neg_one : v3 (-1) = -15 := by sorry

end NUMINAMATH_CALUDE_v3_at_neg_one_l2940_294064


namespace NUMINAMATH_CALUDE_stock_face_value_l2940_294024

/-- Calculates the face value of a stock given the discount rate, brokerage rate, and final cost price. -/
def calculate_face_value (discount_rate : ℚ) (brokerage_rate : ℚ) (final_cost : ℚ) : ℚ :=
  final_cost / ((1 - discount_rate) * (1 + brokerage_rate))

/-- Theorem stating that for a stock with 2% discount, 1/5% brokerage, and Rs 98.2 final cost, the face value is Rs 100. -/
theorem stock_face_value : 
  let discount_rate : ℚ := 2 / 100
  let brokerage_rate : ℚ := 1 / 500
  let final_cost : ℚ := 982 / 10
  calculate_face_value discount_rate brokerage_rate final_cost = 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_face_value_l2940_294024


namespace NUMINAMATH_CALUDE_farm_work_earnings_l2940_294023

/-- Calculates the total money collected given hourly rate, hours worked, and tips. -/
def total_money_collected (hourly_rate : ℕ) (hours_worked : ℕ) (tips : ℕ) : ℕ :=
  hourly_rate * hours_worked + tips

/-- Proves that given the specified conditions, the total money collected is $240. -/
theorem farm_work_earnings : total_money_collected 10 19 50 = 240 := by
  sorry

end NUMINAMATH_CALUDE_farm_work_earnings_l2940_294023


namespace NUMINAMATH_CALUDE_digits_9998_to_10000_of_1_998_l2940_294012

/-- The decimal expansion of 1/998 -/
def decimal_expansion_1_998 : ℕ → ℕ := sorry

/-- The function that extracts a 3-digit number from the decimal expansion -/
def extract_three_digits (start : ℕ) : ℕ := 
  100 * (decimal_expansion_1_998 start) + 
  10 * (decimal_expansion_1_998 (start + 1)) + 
  decimal_expansion_1_998 (start + 2)

/-- The theorem stating that the 9998th through 10000th digits of 1/998 form 042 -/
theorem digits_9998_to_10000_of_1_998 : 
  extract_three_digits 9998 = 42 := by sorry

end NUMINAMATH_CALUDE_digits_9998_to_10000_of_1_998_l2940_294012


namespace NUMINAMATH_CALUDE_color_congruent_triangle_l2940_294049

/-- A type representing the 1992 colors used to color the plane -/
def Color := Fin 1992

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle in the plane -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- A coloring of the plane -/
def Coloring := Point → Color

/-- Two triangles are congruent -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- A point is on the edge of a triangle (excluding vertices) -/
def on_edge (p : Point) (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem color_congruent_triangle 
  (coloring : Coloring) 
  (color_exists : ∀ c : Color, ∃ p : Point, coloring p = c) 
  (T : Triangle) : 
  ∃ T' : Triangle, congruent T T' ∧ 
    ∀ (e1 e2 : Fin 3), ∃ (p1 p2 : Point) (c : Color), 
      on_edge p1 T' ∧ on_edge p2 T' ∧ 
      coloring p1 = c ∧ coloring p2 = c := by sorry

end NUMINAMATH_CALUDE_color_congruent_triangle_l2940_294049


namespace NUMINAMATH_CALUDE_min_value_phi_l2940_294047

/-- Given real numbers a and b satisfying a^2 + b^2 - 4b + 3 = 0,
    and a function f(x) = a·sin(2x) + b·cos(2x) + 1 with maximum value φ(a,b),
    prove that the minimum value of φ(a,b) is 2. -/
theorem min_value_phi (a b : ℝ) (h : a^2 + b^2 - 4*b + 3 = 0) : 
  let f := fun (x : ℝ) ↦ a * Real.sin (2*x) + b * Real.cos (2*x) + 1
  let φ := fun (a b : ℝ) ↦ Real.sqrt (a^2 + b^2) + 1
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ φ a b ∧ 2 ≤ φ a b :=
by sorry

end NUMINAMATH_CALUDE_min_value_phi_l2940_294047


namespace NUMINAMATH_CALUDE_valid_sequences_count_l2940_294004

-- Define the square
def Square := {A : ℝ × ℝ | A = (1, 1) ∨ A = (-1, 1) ∨ A = (-1, -1) ∨ A = (1, -1)}

-- Define the transformations
inductive Transform
| L  -- 90° counterclockwise rotation
| R  -- 90° clockwise rotation
| H  -- reflection across x-axis
| V  -- reflection across y-axis

-- Define a sequence of transformations
def TransformSequence := List Transform

-- Function to check if a transformation is a reflection
def isReflection (t : Transform) : Bool :=
  match t with
  | Transform.H => true
  | Transform.V => true
  | _ => false

-- Function to count reflections in a sequence
def countReflections (seq : TransformSequence) : Nat :=
  seq.filter isReflection |>.length

-- Function to check if a sequence maps the square back to itself
def mapsToSelf (seq : TransformSequence) : Bool :=
  sorry  -- Implementation details omitted

-- Theorem statement
theorem valid_sequences_count (n : Nat) :
  (∃ (seqs : List TransformSequence),
    (∀ seq ∈ seqs,
      seq.length = 24 ∧
      mapsToSelf seq ∧
      Even (countReflections seq)) ∧
    seqs.length = n) :=
  sorry

#check valid_sequences_count

end NUMINAMATH_CALUDE_valid_sequences_count_l2940_294004


namespace NUMINAMATH_CALUDE_complement_of_intersection_l2940_294043

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 2, 4}

-- Define set N
def N : Finset Nat := {3, 4, 5}

-- Theorem statement
theorem complement_of_intersection (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2, 4} ∧ N = {3, 4, 5}) :
  (U \ (M ∩ N)) = {1, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l2940_294043


namespace NUMINAMATH_CALUDE_seminar_fee_calculation_l2940_294044

/-- Proves that the regular seminar fee is $150 given the problem conditions --/
theorem seminar_fee_calculation (F : ℝ) : 
  (∃ (total_spent discounted_fee : ℝ),
    -- 5% discount applied
    discounted_fee = F * 0.95 ∧
    -- 10 teachers registered
    -- $10 food allowance per teacher
    total_spent = 10 * discounted_fee + 10 * 10 ∧
    -- Total spent is $1525
    total_spent = 1525) →
  F = 150 := by
  sorry

end NUMINAMATH_CALUDE_seminar_fee_calculation_l2940_294044


namespace NUMINAMATH_CALUDE_alpha_sin_beta_lt_beta_sin_alpha_l2940_294026

theorem alpha_sin_beta_lt_beta_sin_alpha (α β : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) : 
  α * Real.sin β < β * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_alpha_sin_beta_lt_beta_sin_alpha_l2940_294026


namespace NUMINAMATH_CALUDE_min_value_of_z_l2940_294034

theorem min_value_of_z (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 12) :
  ∃ (z_min : ℝ), z_min = -5 ∧ ∀ (z : ℝ), z = 2 * x + Real.sqrt 3 * y → z ≥ z_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l2940_294034


namespace NUMINAMATH_CALUDE_unique_solution_is_four_l2940_294056

-- Define the equation
def equation (s x : ℝ) : Prop :=
  1 / (3 * x) = (s - x) / 9

-- State the theorem
theorem unique_solution_is_four :
  ∃! s : ℝ, (∃! x : ℝ, equation s x) ∧ s = 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_is_four_l2940_294056


namespace NUMINAMATH_CALUDE_log_equality_l2940_294086

theorem log_equality (x : ℝ) (h : x > 0) :
  (Real.log (2 * x) / Real.log (5 * x) = Real.log (8 * x) / Real.log (625 * x)) →
  (Real.log x / Real.log 2 = Real.log 5 / (2 * Real.log 2 - 3 * Real.log 5)) :=
by sorry

end NUMINAMATH_CALUDE_log_equality_l2940_294086


namespace NUMINAMATH_CALUDE_veg_eaters_count_l2940_294076

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  onlyVeg : ℕ
  onlyNonVeg : ℕ
  bothVegAndNonVeg : ℕ

/-- Calculates the total number of people who eat vegetarian food in the family -/
def totalVegEaters (diet : FamilyDiet) : ℕ :=
  diet.onlyVeg + diet.bothVegAndNonVeg

/-- Theorem stating that for a given family diet, the total number of vegetarian eaters
    is equal to the sum of those who eat only vegetarian and those who eat both -/
theorem veg_eaters_count (diet : FamilyDiet) :
  totalVegEaters diet = diet.onlyVeg + diet.bothVegAndNonVeg := by
  sorry

/-- Example family with the given dietary information -/
def exampleFamily : FamilyDiet where
  onlyVeg := 19
  onlyNonVeg := 9
  bothVegAndNonVeg := 12

#eval totalVegEaters exampleFamily

end NUMINAMATH_CALUDE_veg_eaters_count_l2940_294076


namespace NUMINAMATH_CALUDE_multiple_properties_l2940_294084

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ (∃ p : ℤ, a - b = 4 * p) := by
  sorry

end NUMINAMATH_CALUDE_multiple_properties_l2940_294084


namespace NUMINAMATH_CALUDE_fourth_term_is_twenty_l2940_294009

def sequence_term (n : ℕ) : ℕ := n + 2^n

theorem fourth_term_is_twenty : sequence_term 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_twenty_l2940_294009


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2940_294019

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b - a * b = 0) :
  a + b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4 * a₀ + b₀ - a₀ * b₀ = 0 ∧ a₀ + b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2940_294019


namespace NUMINAMATH_CALUDE_intersection_A_B_l2940_294071

def A : Set ℝ := {x | x^2 - 3*x ≤ 0}
def B : Set ℝ := {1, 2}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2940_294071


namespace NUMINAMATH_CALUDE_quadratic_less_than_linear_l2940_294085

theorem quadratic_less_than_linear (x : ℝ) : -1/2 * x^2 + 2*x < -x + 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_less_than_linear_l2940_294085


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2940_294094

theorem reciprocal_inequality (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2940_294094


namespace NUMINAMATH_CALUDE_incorrect_statement_E_l2940_294052

theorem incorrect_statement_E (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  -- Statement A
  (∀ x y : ℝ, x > 0 → y > 0 → x > y → x^2 > y^2) ∧
  -- Statement B
  (2 * a * b / (a + b) < Real.sqrt (a * b)) ∧
  -- Statement C
  (∀ p : ℝ, p > 0 → ∀ x y : ℝ, x > 0 → y > 0 → x * y = p → 
    x + y ≥ 2 * Real.sqrt p ∧ (x + y = 2 * Real.sqrt p ↔ x = y)) ∧
  -- Statement D
  ((a + b)^3 > (a^3 + b^3) / 2) ∧
  -- Statement E (negation)
  ¬((a + b)^2 / 4 > (a^2 + b^2) / 2) := by
sorry

end NUMINAMATH_CALUDE_incorrect_statement_E_l2940_294052


namespace NUMINAMATH_CALUDE_wage_before_raise_l2940_294063

/-- Given a 33.33% increase from x results in $40, prove that x equals $30. -/
theorem wage_before_raise (x : ℝ) : x * (1 + 33.33 / 100) = 40 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_wage_before_raise_l2940_294063


namespace NUMINAMATH_CALUDE_jebbs_take_home_pay_l2940_294030

/-- Calculates the take-home pay given a gross salary and various tax rates and deductions. -/
def calculateTakeHomePay (grossSalary : ℚ) : ℚ :=
  let federalTaxRate1 := 0.10
  let federalTaxRate2 := 0.15
  let federalTaxRate3 := 0.25
  let federalTaxThreshold1 := 2500
  let federalTaxThreshold2 := 5000
  let stateTaxRate1 := 0.05
  let stateTaxRate2 := 0.07
  let stateTaxThreshold := 3000
  let socialSecurityTaxRate := 0.062
  let socialSecurityTaxCap := 4800
  let medicareTaxRate := 0.0145
  let healthInsurance := 300
  let retirementContributionRate := 0.07

  let federalTax := 
    federalTaxRate1 * federalTaxThreshold1 +
    federalTaxRate2 * (federalTaxThreshold2 - federalTaxThreshold1) +
    federalTaxRate3 * (grossSalary - federalTaxThreshold2)

  let stateTax := 
    stateTaxRate1 * stateTaxThreshold +
    stateTaxRate2 * (grossSalary - stateTaxThreshold)

  let socialSecurityTax := socialSecurityTaxRate * (min grossSalary socialSecurityTaxCap)

  let medicareTax := medicareTaxRate * grossSalary

  let retirementContribution := retirementContributionRate * grossSalary

  let totalDeductions := 
    federalTax + stateTax + socialSecurityTax + medicareTax + healthInsurance + retirementContribution

  grossSalary - totalDeductions

/-- Theorem stating that Jebb's take-home pay is $3,958.15 given his gross salary and deductions. -/
theorem jebbs_take_home_pay :
  calculateTakeHomePay 6500 = 3958.15 := by
  sorry


end NUMINAMATH_CALUDE_jebbs_take_home_pay_l2940_294030


namespace NUMINAMATH_CALUDE_total_spent_is_20_l2940_294000

/-- The price of a bracelet in dollars -/
def bracelet_price : ℕ := 4

/-- The price of a keychain in dollars -/
def keychain_price : ℕ := 5

/-- The price of a coloring book in dollars -/
def coloring_book_price : ℕ := 3

/-- The number of bracelets Paula buys -/
def paula_bracelets : ℕ := 2

/-- The number of keychains Paula buys -/
def paula_keychains : ℕ := 1

/-- The number of coloring books Olive buys -/
def olive_coloring_books : ℕ := 1

/-- The number of bracelets Olive buys -/
def olive_bracelets : ℕ := 1

/-- The total amount spent by Paula and Olive -/
def total_spent : ℕ :=
  paula_bracelets * bracelet_price +
  paula_keychains * keychain_price +
  olive_coloring_books * coloring_book_price +
  olive_bracelets * bracelet_price

theorem total_spent_is_20 : total_spent = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_20_l2940_294000


namespace NUMINAMATH_CALUDE_work_completion_time_l2940_294002

/-- 
Given:
- a and b complete a work in 9 days
- a and b together can do the work in 6 days

Prove: a alone can complete the work in 18 days
-/
theorem work_completion_time (a b : ℝ) (h1 : a + b = 1 / 9) (h2 : a + b = 1 / 6) : a = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2940_294002


namespace NUMINAMATH_CALUDE_differential_savings_proof_l2940_294001

def annual_income : ℕ := 45000
def retirement_contribution : ℕ := 4000
def mortgage_interest : ℕ := 5000
def charitable_donations : ℕ := 2000
def previous_tax_rate : ℚ := 40 / 100

def taxable_income : ℕ := annual_income - retirement_contribution - mortgage_interest - charitable_donations

def tax_bracket_1 : ℕ := 10000
def tax_bracket_2 : ℕ := 25000
def tax_bracket_3 : ℕ := 50000

def tax_rate_1 : ℚ := 0 / 100
def tax_rate_2 : ℚ := 10 / 100
def tax_rate_3 : ℚ := 25 / 100
def tax_rate_4 : ℚ := 35 / 100

def new_tax (income : ℕ) : ℚ :=
  if income ≤ tax_bracket_1 then
    income * tax_rate_1
  else if income ≤ tax_bracket_2 then
    tax_bracket_1 * tax_rate_1 + (income - tax_bracket_1) * tax_rate_2
  else if income ≤ tax_bracket_3 then
    tax_bracket_1 * tax_rate_1 + (tax_bracket_2 - tax_bracket_1) * tax_rate_2 + (income - tax_bracket_2) * tax_rate_3
  else
    tax_bracket_1 * tax_rate_1 + (tax_bracket_2 - tax_bracket_1) * tax_rate_2 + (tax_bracket_3 - tax_bracket_2) * tax_rate_3 + (income - tax_bracket_3) * tax_rate_4

theorem differential_savings_proof :
  (annual_income * previous_tax_rate - new_tax taxable_income) = 14250 := by
  sorry

end NUMINAMATH_CALUDE_differential_savings_proof_l2940_294001


namespace NUMINAMATH_CALUDE_inequality_proof_l2940_294098

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^2 - b^2) / (a^2 + b^2) > (a - b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2940_294098


namespace NUMINAMATH_CALUDE_rays_dog_walks_63_blocks_l2940_294050

/-- Represents the distance of a single walk in blocks -/
structure Walk where
  to_destination : ℕ
  to_second_place : ℕ
  back_home : ℕ

/-- Calculates the total distance of a walk -/
def Walk.total (w : Walk) : ℕ := w.to_destination + w.to_second_place + w.back_home

/-- Represents Ray's daily dog walking routine -/
structure DailyWalk where
  morning : Walk
  afternoon : Walk
  evening : Walk

/-- Calculates the total distance of all walks in a day -/
def DailyWalk.total_distance (d : DailyWalk) : ℕ :=
  d.morning.total + d.afternoon.total + d.evening.total

/-- Ray's actual daily walk routine -/
def rays_routine : DailyWalk := {
  morning := { to_destination := 4, to_second_place := 7, back_home := 11 }
  afternoon := { to_destination := 3, to_second_place := 5, back_home := 8 }
  evening := { to_destination := 6, to_second_place := 9, back_home := 10 }
}

/-- Theorem stating that Ray's dog walks 63 blocks each day -/
theorem rays_dog_walks_63_blocks : DailyWalk.total_distance rays_routine = 63 := by
  sorry

end NUMINAMATH_CALUDE_rays_dog_walks_63_blocks_l2940_294050


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2940_294020

theorem functional_equation_solution (f g : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → f (x + y) = g (1/x + 1/y) * (x*y)^2008) :
  ∃ (c : ℝ), ∀ (x : ℝ), f x = c * x^2008 ∧ g x = c * x^2008 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2940_294020


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l2940_294080

theorem moving_circle_trajectory (x y : ℝ) : 
  (∃ (t : ℝ), x^2 + y^2 = 4 + t^2 ∧ t = 1 ∨ t = -1) ↔ 
  (x^2 + y^2 = 9 ∨ x^2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l2940_294080


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_range_of_a_inequality_proof_l2940_294078

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / Real.exp x - x + 1

theorem tangent_line_at_zero (h : ℝ) : 
  ∃ (m b : ℝ), m * h + b = f 1 h ∧ 
  ∀ x, m * x + b = 2 * x - 2 + f 1 0 := by sorry

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, f a x < 0) → a ≤ -1 := by sorry

theorem inequality_proof (x : ℝ) : 
  x > 0 → 2 / Real.exp x - 2 < (1/2) * x^2 - x := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_range_of_a_inequality_proof_l2940_294078


namespace NUMINAMATH_CALUDE_proposition_d_true_others_false_l2940_294007

theorem proposition_d_true_others_false :
  (∃ x : ℝ, 3 * x^2 - 4 = 6 * x) ∧
  ¬(∀ x : ℝ, (x - Real.sqrt 2)^2 > 0) ∧
  ¬(∀ x : ℚ, x^2 > 0) ∧
  ¬(∃ x : ℤ, 3 * x = 128) :=
by sorry

end NUMINAMATH_CALUDE_proposition_d_true_others_false_l2940_294007


namespace NUMINAMATH_CALUDE_alley_width_equals_ladder_height_l2940_294089

/-- Proof that the width of an alley equals the height of a ladder against one wall 
    when it forms specific angles with both walls. -/
theorem alley_width_equals_ladder_height 
  (l : ℝ) -- length of the ladder
  (x y : ℝ) -- heights on the walls
  (h_x_pos : x > 0)
  (h_y_pos : y > 0)
  (h_angle_Q : x / w = Real.sqrt 3) -- tan 60° = √3
  (h_angle_R : y / w = 1) -- tan 45° = 1
  : w = y :=
sorry

end NUMINAMATH_CALUDE_alley_width_equals_ladder_height_l2940_294089


namespace NUMINAMATH_CALUDE_quadratic_real_roots_alpha_range_l2940_294038

theorem quadratic_real_roots_alpha_range :
  ∀ α : ℝ, 
  (∃ x : ℝ, x^2 - 2*x + α = 0) →
  α ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_alpha_range_l2940_294038


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l2940_294041

theorem quadratic_inequality_problem (a c m : ℝ) :
  (∀ x, ax^2 + x + c > 0 ↔ 1 < x ∧ x < 3) →
  let A := {x | (-1/4)*x^2 + 2*x - 3 > 0}
  let B := {x | x + m > 0}
  A ⊆ B →
  (a = -1/4 ∧ c = -3/4) ∧ m ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l2940_294041


namespace NUMINAMATH_CALUDE_abc_sum_product_zero_l2940_294010

theorem abc_sum_product_zero (a b c : ℝ) (h : 1/a + 1/b + 1/c = 1/(a+b+c)) :
  (a+b)*(b+c)*(a+c) = 0 := by sorry

end NUMINAMATH_CALUDE_abc_sum_product_zero_l2940_294010


namespace NUMINAMATH_CALUDE_ball_arrangements_l2940_294096

def num_balls : ℕ := 5
def num_black_balls : ℕ := 2
def num_colored_balls : ℕ := 3
def balls_in_row : ℕ := 4

theorem ball_arrangements :
  (Nat.factorial balls_in_row) + 
  (num_colored_balls * (Nat.factorial balls_in_row) / 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ball_arrangements_l2940_294096


namespace NUMINAMATH_CALUDE_four_statements_incorrect_l2940_294033

/-- The alternating sum from 1 to 2002 -/
def alternating_sum : ℕ → ℤ
  | 0 => 0
  | n + 1 => if n % 2 = 0 then alternating_sum n + (n + 1) else alternating_sum n - (n + 1)

/-- The sum of n consecutive natural numbers starting from k -/
def consec_sum (n k : ℕ) : ℕ := n * (2 * k + n - 1) / 2

theorem four_statements_incorrect : 
  (¬ Even (alternating_sum 2002)) ∧ 
  (∃ (a b c : ℤ), Odd a ∧ Odd b ∧ Odd c ∧ (a * b) * (c - b) ≠ a) ∧
  (¬ Even (consec_sum 2002 1)) ∧
  (¬ ∃ (a b : ℤ), (a + b) * (a - b) = 2002) :=
by sorry

end NUMINAMATH_CALUDE_four_statements_incorrect_l2940_294033


namespace NUMINAMATH_CALUDE_largest_perimeter_is_164_l2940_294066

/-- Represents a rectangle with integer side lengths -/
structure IntRectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of an IntRectangle -/
def perimeter (r : IntRectangle) : ℕ :=
  2 * (r.length + r.width)

/-- Calculates the area of an IntRectangle -/
def area (r : IntRectangle) : ℕ :=
  r.length * r.width

/-- Checks if a rectangle satisfies the given condition -/
def satisfiesCondition (r : IntRectangle) : Prop :=
  4 * perimeter r = area r - 1

/-- Theorem stating that the largest possible perimeter of a rectangle satisfying the condition is 164 -/
theorem largest_perimeter_is_164 :
  ∀ r : IntRectangle, satisfiesCondition r → perimeter r ≤ 164 :=
by sorry

end NUMINAMATH_CALUDE_largest_perimeter_is_164_l2940_294066


namespace NUMINAMATH_CALUDE_sixth_term_value_l2940_294055

def sequence_property (s : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → s (n + 1) = (1 / 4) * (s n + s (n + 2))

theorem sixth_term_value (s : ℕ → ℚ) :
  sequence_property s →
  s 1 = 3 →
  s 5 = 48 →
  s 6 = 2001 / 14 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_value_l2940_294055


namespace NUMINAMATH_CALUDE_sum_of_squares_l2940_294090

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2940_294090


namespace NUMINAMATH_CALUDE_jimmy_passing_points_l2940_294062

/-- The minimum number of points required to pass the class -/
def passingScore : ℕ := 50

/-- The number of exams Jimmy took -/
def numExams : ℕ := 3

/-- The number of points Jimmy earned per exam -/
def pointsPerExam : ℕ := 20

/-- The number of points Jimmy lost for bad behavior -/
def pointsLost : ℕ := 5

/-- The maximum number of additional points Jimmy can lose while still passing -/
def maxAdditionalPointsLost : ℕ := 5

theorem jimmy_passing_points : 
  numExams * pointsPerExam - pointsLost - maxAdditionalPointsLost ≥ passingScore := by
  sorry

end NUMINAMATH_CALUDE_jimmy_passing_points_l2940_294062


namespace NUMINAMATH_CALUDE_time_at_15mph_is_3_hours_l2940_294036

/-- Represents the running scenario with three different speeds -/
structure RunningScenario where
  time_at_15mph : ℝ
  time_at_10mph : ℝ
  time_at_8mph : ℝ

/-- The total time of the run is 14 hours -/
def total_time (run : RunningScenario) : ℝ :=
  run.time_at_15mph + run.time_at_10mph + run.time_at_8mph

/-- The total distance covered is 164 miles -/
def total_distance (run : RunningScenario) : ℝ :=
  15 * run.time_at_15mph + 10 * run.time_at_10mph + 8 * run.time_at_8mph

/-- Theorem stating that the time spent running at 15 mph was 3 hours -/
theorem time_at_15mph_is_3_hours :
  ∃ (run : RunningScenario),
    total_time run = 14 ∧
    total_distance run = 164 ∧
    run.time_at_15mph = 3 ∧
    run.time_at_10mph ≥ 0 ∧
    run.time_at_8mph ≥ 0 :=
  sorry

end NUMINAMATH_CALUDE_time_at_15mph_is_3_hours_l2940_294036


namespace NUMINAMATH_CALUDE_total_pictures_l2940_294099

/-- 
Given that:
- Randy drew 5 pictures
- Peter drew 3 more pictures than Randy
- Quincy drew 20 more pictures than Peter

Prove that the total number of pictures drawn by Randy, Peter, and Quincy is 41.
-/
theorem total_pictures (randy peter quincy : ℕ) : 
  randy = 5 → 
  peter = randy + 3 → 
  quincy = peter + 20 → 
  randy + peter + quincy = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_pictures_l2940_294099


namespace NUMINAMATH_CALUDE_no_integer_solution_l2940_294048

theorem no_integer_solution : ¬∃ y : ℤ, (8 : ℝ)^3 + 4^3 + 2^10 = 2^y := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2940_294048


namespace NUMINAMATH_CALUDE_f_at_three_equals_five_l2940_294035

/-- A quadratic function f(x) = ax^2 + bx + 2 satisfying f(1) = 4 and f(2) = 5 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

/-- Theorem: Given f(x) = ax^2 + bx + 2 with f(1) = 4 and f(2) = 5, prove that f(3) = 5 -/
theorem f_at_three_equals_five (a b : ℝ) (h1 : f a b 1 = 4) (h2 : f a b 2 = 5) :
  f a b 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_at_three_equals_five_l2940_294035


namespace NUMINAMATH_CALUDE_audrey_twice_heracles_age_l2940_294070

def age_difference : ℕ := 7
def heracles_current_age : ℕ := 10

theorem audrey_twice_heracles_age (years : ℕ) : 
  (heracles_current_age + age_difference + years = 2 * heracles_current_age) → years = 3 := by
  sorry

end NUMINAMATH_CALUDE_audrey_twice_heracles_age_l2940_294070


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2940_294057

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 5 * Real.sin θ

-- Define the Cartesian equation of a circle
def circle_equation (x y : ℝ) (h k r : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_to_cartesian_circle :
  ∀ x y : ℝ, (∃ ρ θ : ℝ, polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ∃ h k r : ℝ, circle_equation x y h k r := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2940_294057


namespace NUMINAMATH_CALUDE_isosceles_probability_2020gon_l2940_294006

/-- The number of vertices in the regular polygon -/
def n : ℕ := 2020

/-- The probability of forming an isosceles triangle by randomly selecting
    three distinct vertices from a regular n-gon -/
def isosceles_probability (n : ℕ) : ℚ :=
  (n * ((n - 2) / 2)) / Nat.choose n 3

/-- Theorem stating that the probability of forming an isosceles triangle
    by randomly selecting three distinct vertices from a regular 2020-gon
    is 1/673 -/
theorem isosceles_probability_2020gon :
  isosceles_probability n = 1 / 673 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_probability_2020gon_l2940_294006


namespace NUMINAMATH_CALUDE_solution_set_characterization_l2940_294025

/-- A function that is even and monotonically increasing on (0,+∞) -/
def f (a b : ℝ) (x : ℝ) : ℝ := (x - 2) * (a * x + b)

/-- The property of f being an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The property of f being monotonically increasing on (0,+∞) -/
def is_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

/-- The solution set for f(2-x) > 0 -/
def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f (2 - x) > 0}

/-- The theorem stating the solution set for f(2-x) > 0 -/
theorem solution_set_characterization {a b : ℝ} (h_even : is_even (f a b))
    (h_incr : is_increasing_on_positive (f a b)) :
    solution_set (f a b) = {x | x < 0 ∨ x > 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l2940_294025


namespace NUMINAMATH_CALUDE_right_triangle_area_leg_sum_l2940_294011

theorem right_triangle_area_leg_sum (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → 
  a^2 + b^2 = c^2 → 
  (a * b) / 2 + a = 75 ∨ (a * b) / 2 + b = 75 →
  ((a = 6 ∧ b = 23 ∧ c = 25) ∨ (a = 23 ∧ b = 6 ∧ c = 25) ∨
   (a = 15 ∧ b = 8 ∧ c = 17) ∨ (a = 8 ∧ b = 15 ∧ c = 17)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_leg_sum_l2940_294011


namespace NUMINAMATH_CALUDE_total_population_of_three_cities_l2940_294031

/-- Given the populations of three cities with specific relationships, 
    prove that their total population is 56000. -/
theorem total_population_of_three_cities 
  (pop_lake_view pop_seattle pop_boise : ℕ) : 
  pop_lake_view = 24000 →
  pop_lake_view = pop_seattle + 4000 →
  pop_boise = (3 * pop_seattle) / 5 →
  pop_lake_view + pop_seattle + pop_boise = 56000 := by
  sorry

end NUMINAMATH_CALUDE_total_population_of_three_cities_l2940_294031


namespace NUMINAMATH_CALUDE_min_value_expression_l2940_294042

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a ≥ Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2940_294042


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2940_294061

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2940_294061


namespace NUMINAMATH_CALUDE_correct_expansion_l2940_294027

theorem correct_expansion (a b : ℝ) : (a - b) * (-a - b) = -a^2 + b^2 := by
  -- Definitions based on the given conditions (equations A, B, and C)
  have h1 : (a + b) * (a - b) = a^2 - b^2 := by sorry
  have h2 : (a + b) * (-a - b) = -(a + b)^2 := by sorry
  have h3 : (a - b) * (-a + b) = -(a - b)^2 := by sorry

  -- Proof of the correct expansion
  sorry

end NUMINAMATH_CALUDE_correct_expansion_l2940_294027


namespace NUMINAMATH_CALUDE_consecutive_integers_puzzle_l2940_294083

theorem consecutive_integers_puzzle (a b c d e : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  b = a + 1 → c = b + 1 → d = c + 1 → e = d + 1 →
  a < b → b < c → c < d → d < e →
  a + b = e - 1 →
  a * b = d + 1 →
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_puzzle_l2940_294083


namespace NUMINAMATH_CALUDE_ps_length_is_sqrt_461_l2940_294021

/-- A quadrilateral with two right angles and specific side lengths -/
structure RightQuadrilateral where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  angleQ_is_right : Bool
  angleR_is_right : Bool

/-- The length of PS in the right quadrilateral PQRS -/
def length_PS (quad : RightQuadrilateral) : ℝ :=
  sorry

/-- Theorem stating that for a quadrilateral PQRS with given side lengths and right angles, PS = √461 -/
theorem ps_length_is_sqrt_461 (quad : RightQuadrilateral) 
  (h1 : quad.PQ = 6)
  (h2 : quad.QR = 10)
  (h3 : quad.RS = 25)
  (h4 : quad.angleQ_is_right = true)
  (h5 : quad.angleR_is_right = true) :
  length_PS quad = Real.sqrt 461 :=
by sorry

end NUMINAMATH_CALUDE_ps_length_is_sqrt_461_l2940_294021


namespace NUMINAMATH_CALUDE_equation_solution_l2940_294053

theorem equation_solution :
  ∃ x : ℝ, (3 / 4 + 1 / x = 7 / 8) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2940_294053


namespace NUMINAMATH_CALUDE_relay_race_fifth_runner_l2940_294003

def relay_race (t1 t2 t3 t4 t5 : ℝ) : Prop :=
  t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧ t4 > 0 ∧ t5 > 0 ∧
  (t1/2 + t2 + t3 + t4 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.95 ∧
  (t1 + t2/2 + t3 + t4 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.90 ∧
  (t1 + t2 + t3/2 + t4 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.88 ∧
  (t1 + t2 + t3 + t4/2 + t5) / (t1 + t2 + t3 + t4 + t5) = 0.85

theorem relay_race_fifth_runner (t1 t2 t3 t4 t5 : ℝ) :
  relay_race t1 t2 t3 t4 t5 →
  (t1 + t2 + t3 + t4 + t5/2) / (t1 + t2 + t3 + t4 + t5) = 0.92 :=
by sorry

end NUMINAMATH_CALUDE_relay_race_fifth_runner_l2940_294003


namespace NUMINAMATH_CALUDE_probability_at_least_6_consecutive_heads_l2940_294058

def coin_flip_sequence := Fin 9 → Bool

def has_at_least_6_consecutive_heads (s : coin_flip_sequence) : Prop :=
  ∃ i, i + 5 < 9 ∧ (∀ j, i ≤ j ∧ j ≤ i + 5 → s j = true)

def total_sequences : ℕ := 2^9

def favorable_sequences : ℕ := 10

theorem probability_at_least_6_consecutive_heads :
  (favorable_sequences : ℚ) / total_sequences = 5 / 256 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_6_consecutive_heads_l2940_294058


namespace NUMINAMATH_CALUDE_neg_a_fourth_times_neg_a_squared_l2940_294037

theorem neg_a_fourth_times_neg_a_squared (a : ℝ) : -a^4 * (-a)^2 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_neg_a_fourth_times_neg_a_squared_l2940_294037


namespace NUMINAMATH_CALUDE_probability_not_green_l2940_294015

def total_balls : ℕ := 6 + 3 + 4 + 5
def non_green_balls : ℕ := 6 + 3 + 4

theorem probability_not_green (red_balls : ℕ) (yellow_balls : ℕ) (black_balls : ℕ) (green_balls : ℕ)
  (h_red : red_balls = 6)
  (h_yellow : yellow_balls = 3)
  (h_black : black_balls = 4)
  (h_green : green_balls = 5) :
  (red_balls + yellow_balls + black_balls : ℚ) / (red_balls + yellow_balls + black_balls + green_balls) = 13 / 18 :=
by sorry

end NUMINAMATH_CALUDE_probability_not_green_l2940_294015


namespace NUMINAMATH_CALUDE_tiling_comparison_l2940_294075

/-- Number of ways to tile a grid with rectangles -/
def tiling_count (grid_size : ℕ × ℕ) (tile_size : ℕ × ℕ) : ℕ := sorry

/-- Theorem: For any n > 1, the number of ways to tile a 3n × 3n grid with 1 × 3 rectangles
    is greater than the number of ways to tile a 2n × 2n grid with 1 × 2 rectangles -/
theorem tiling_comparison (n : ℕ) (h : n > 1) :
  tiling_count (3*n, 3*n) (1, 3) > tiling_count (2*n, 2*n) (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_tiling_comparison_l2940_294075


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2940_294067

theorem lcm_hcf_problem (A B : ℕ+) (h1 : B = 671) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 61) : A = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2940_294067


namespace NUMINAMATH_CALUDE_g_13_equals_201_l2940_294017

def g (n : ℕ) : ℕ := n^2 + n + 19

theorem g_13_equals_201 : g 13 = 201 := by sorry

end NUMINAMATH_CALUDE_g_13_equals_201_l2940_294017


namespace NUMINAMATH_CALUDE_candy_distribution_l2940_294013

theorem candy_distribution (left right : ℕ) : 
  left + right = 27 →
  right - left = (left + left) + 3 →
  left = 6 ∧ right = 21 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2940_294013


namespace NUMINAMATH_CALUDE_prime_factors_of_2008006_l2940_294069

theorem prime_factors_of_2008006 : 
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : Nat), 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ 
    Nat.Prime p₄ ∧ Nat.Prime p₅ ∧ Nat.Prime p₆ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₁ ≠ p₆ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₂ ≠ p₆ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₃ ≠ p₆ ∧
    p₄ ≠ p₅ ∧ p₄ ≠ p₆ ∧
    p₅ ≠ p₆ ∧
    2008006 = p₁ * p₂ * p₃ * p₄ * p₅ * p₆ ∧
    (∀ q : Nat, Nat.Prime q → q ∣ 2008006 → (q = p₁ ∨ q = p₂ ∨ q = p₃ ∨ q = p₄ ∨ q = p₅ ∨ q = p₆)) :=
by sorry


end NUMINAMATH_CALUDE_prime_factors_of_2008006_l2940_294069


namespace NUMINAMATH_CALUDE_tank_dimension_l2940_294014

/-- Proves that the third dimension of a rectangular tank is 2 feet given specific conditions -/
theorem tank_dimension (x : ℝ) : 
  (4 : ℝ) > 0 ∧ 
  (5 : ℝ) > 0 ∧ 
  x > 0 ∧
  (20 : ℝ) > 0 ∧
  1520 = 20 * (2 * (4 * 5) + 2 * (4 * x) + 2 * (5 * x)) →
  x = 2 := by
  sorry

#check tank_dimension

end NUMINAMATH_CALUDE_tank_dimension_l2940_294014


namespace NUMINAMATH_CALUDE_count_base7_with_456_l2940_294018

/-- Represents a positive integer in base 7 --/
def Base7Int : Type := ℕ+

/-- Checks if a Base7Int contains the digits 4, 5, or 6 --/
def containsDigit456 (n : Base7Int) : Prop := sorry

/-- The set of the smallest 2401 positive integers in base 7 --/
def smallestBase7Ints : Set Base7Int := {n | n.val ≤ 2401}

/-- The count of numbers in smallestBase7Ints that contain 4, 5, or 6 --/
def countWith456 : ℕ := sorry

theorem count_base7_with_456 : countWith456 = 2146 := by sorry

end NUMINAMATH_CALUDE_count_base7_with_456_l2940_294018


namespace NUMINAMATH_CALUDE_ground_mince_calculation_l2940_294087

/-- The total amount of ground mince used for lasagnas and cottage pies -/
def total_ground_mince (num_lasagnas : ℕ) (mince_per_lasagna : ℕ) 
                       (num_cottage_pies : ℕ) (mince_per_cottage_pie : ℕ) : ℕ :=
  num_lasagnas * mince_per_lasagna + num_cottage_pies * mince_per_cottage_pie

/-- Theorem stating the total amount of ground mince used -/
theorem ground_mince_calculation :
  total_ground_mince 100 2 100 3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_ground_mince_calculation_l2940_294087


namespace NUMINAMATH_CALUDE_calculator_time_saved_l2940_294008

/-- The time saved by using a calculator for math homework -/
theorem calculator_time_saved 
  (time_with_calc : ℕ)      -- Time per problem with calculator
  (time_without_calc : ℕ)   -- Time per problem without calculator
  (num_problems : ℕ)        -- Number of problems in the assignment
  (h1 : time_with_calc = 2) -- It takes 2 minutes per problem with calculator
  (h2 : time_without_calc = 5) -- It takes 5 minutes per problem without calculator
  (h3 : num_problems = 20)  -- The assignment has 20 problems
  : (time_without_calc - time_with_calc) * num_problems = 60 := by
  sorry

end NUMINAMATH_CALUDE_calculator_time_saved_l2940_294008


namespace NUMINAMATH_CALUDE_weight_loss_calculation_l2940_294074

/-- Proves that a measured weight loss of 9.22% with 2% added clothing weight
    corresponds to an actual weight loss of approximately 5.55% -/
theorem weight_loss_calculation (measured_loss : Real) (clothing_weight : Real) :
  measured_loss = 9.22 ∧ clothing_weight = 2 →
  ∃ actual_loss : Real,
    (100 - actual_loss) * (1 + clothing_weight / 100) = 100 - measured_loss ∧
    abs (actual_loss - 5.55) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_calculation_l2940_294074


namespace NUMINAMATH_CALUDE_sum_of_roots_even_l2940_294077

theorem sum_of_roots_even (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (h_distinct : ∃ (x y : ℤ), x ≠ y ∧ x^2 - 2*p*x + p*q = 0 ∧ y^2 - 2*p*y + p*q = 0) :
  ∃ (k : ℤ), 2 * p = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_even_l2940_294077


namespace NUMINAMATH_CALUDE_tenth_term_is_21_l2940_294095

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_first_three : a 1 + a 2 + a 3 = 15
  geometric : (a 2 + 5)^2 = (a 1 + 2) * (a 3 + 13)

/-- The 10th term of the arithmetic sequence is 21 -/
theorem tenth_term_is_21 (seq : ArithmeticSequence) : seq.a 10 = 21 := by
  sorry

#check tenth_term_is_21

end NUMINAMATH_CALUDE_tenth_term_is_21_l2940_294095


namespace NUMINAMATH_CALUDE_marbles_distribution_l2940_294016

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 35 →
  num_boys = 5 →
  marbles_per_boy = total_marbles / num_boys →
  marbles_per_boy = 7 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l2940_294016


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2940_294059

def has_three_integer_solutions (b : ℤ) : Prop :=
  ∃ x y z : ℤ, x < y ∧ y < z ∧
    (x^2 + b*x - 2 ≤ 0) ∧
    (y^2 + b*y - 2 ≤ 0) ∧
    (z^2 + b*z - 2 ≤ 0) ∧
    ∀ w : ℤ, (w^2 + b*w - 2 ≤ 0) → (w = x ∨ w = y ∨ w = z)

theorem quadratic_inequality_solutions :
  ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ b : ℤ, b ∈ s ↔ has_three_integer_solutions b :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2940_294059


namespace NUMINAMATH_CALUDE_max_plus_min_equals_16_l2940_294040

def f (x : ℝ) : ℝ := x^3 - 12*x + 8

theorem max_plus_min_equals_16 :
  ∃ (M m : ℝ),
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ M) ∧
    (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = M) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, m ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = m) ∧
    M + m = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_plus_min_equals_16_l2940_294040


namespace NUMINAMATH_CALUDE_min_value_expression_l2940_294082

theorem min_value_expression (x : ℝ) (h : x > 0) :
  4 * x + 9 / x^2 ≥ 3 * (36 : ℝ)^(1/3) ∧
  ∃ y > 0, 4 * y + 9 / y^2 = 3 * (36 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2940_294082


namespace NUMINAMATH_CALUDE_problem_solution_l2940_294022

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem problem_solution :
  (N ⊆ M) ∧
  (∀ a b : ℝ, (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M)) ∧
  (∃ p q : Prop, ¬(p ∧ q) ∧ (p ∨ q)) ∧
  (¬(∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2940_294022


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_l2940_294081

theorem quadratic_two_zeros (a b c : ℝ) (h : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_l2940_294081


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2940_294051

/-- A point in the second quadrant with given absolute values for its coordinates -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  second_quadrant : x < 0 ∧ y > 0
  abs_x : |x| = 2
  abs_y : |y| = 3

/-- The symmetric point with respect to the origin -/
def symmetric_point (p : SecondQuadrantPoint) : ℝ × ℝ := (-p.x, -p.y)

/-- Theorem stating that the symmetric point has coordinates (2, -3) -/
theorem symmetric_point_coordinates (p : SecondQuadrantPoint) : 
  symmetric_point p = (2, -3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2940_294051
