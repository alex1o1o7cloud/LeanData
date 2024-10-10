import Mathlib

namespace arithmetic_calculation_l572_57249

theorem arithmetic_calculation : 6 * (5 - 2) + 4 = 22 := by
  sorry

end arithmetic_calculation_l572_57249


namespace marcus_percentage_of_team_points_l572_57233

def marcus_points (three_point_goals two_point_goals free_throws four_point_goals : ℕ) : ℕ :=
  3 * three_point_goals + 2 * two_point_goals + free_throws + 4 * four_point_goals

def percentage (part whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

theorem marcus_percentage_of_team_points : 
  let marcus_total := marcus_points 5 10 8 2
  let team_total := 110
  abs (percentage marcus_total team_total - 46.36) < 0.01 := by sorry

end marcus_percentage_of_team_points_l572_57233


namespace parabola_shift_theorem_l572_57221

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
    b := p.b
    c := p.c + v }

/-- The theorem to be proved -/
theorem parabola_shift_theorem :
  let original := Parabola.mk (-2) 0 0
  let shifted_left := shift_horizontal original 3
  let final := shift_vertical shifted_left (-1)
  final = Parabola.mk (-2) 12 (-19) := by sorry

end parabola_shift_theorem_l572_57221


namespace pedro_has_200_squares_l572_57243

/-- The number of squares Jesus has -/
def jesus_squares : ℕ := 60

/-- The number of squares Linden has -/
def linden_squares : ℕ := 75

/-- The number of additional squares Pedro has compared to Jesus and Linden combined -/
def pedro_additional_squares : ℕ := 65

/-- The total number of squares Pedro has -/
def pedro_squares : ℕ := jesus_squares + linden_squares + pedro_additional_squares

theorem pedro_has_200_squares : pedro_squares = 200 := by
  sorry

end pedro_has_200_squares_l572_57243


namespace set_operations_l572_57298

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem set_operations :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 5}) ∧
  ((U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)}) ∧
  (A ∩ (U \ B) = {x | 2 ≤ x ∧ x < 3}) := by
  sorry

end set_operations_l572_57298


namespace intersection_when_m_zero_m_range_for_necessary_condition_l572_57266

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define predicates p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

-- Theorem for part (I)
theorem intersection_when_m_zero : 
  A ∩ B 0 = {x | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem for part (II)
theorem m_range_for_necessary_condition :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) ↔ m ≥ 4 ∨ m ≤ -2 := by sorry

end intersection_when_m_zero_m_range_for_necessary_condition_l572_57266


namespace karens_order_cost_l572_57224

/-- The cost of Karen's fast-food order -/
def fast_food_order_cost (burger_cost sandwich_cost smoothie_cost : ℕ) 
  (burger_quantity sandwich_quantity smoothie_quantity : ℕ) : ℕ :=
  burger_cost * burger_quantity + sandwich_cost * sandwich_quantity + smoothie_cost * smoothie_quantity

/-- Theorem stating that Karen's fast-food order costs $17 -/
theorem karens_order_cost : 
  fast_food_order_cost 5 4 4 1 1 2 = 17 := by
  sorry

end karens_order_cost_l572_57224


namespace eight_million_scientific_notation_l572_57285

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- States that 8 million in scientific notation is 8 * 10^6 -/
theorem eight_million_scientific_notation :
  to_scientific_notation 8000000 = ScientificNotation.mk 8 6 (by norm_num) :=
sorry

end eight_million_scientific_notation_l572_57285


namespace number_puzzle_l572_57256

theorem number_puzzle : ∃ x : ℝ, (100 - x = x + 40) ∧ (x = 30) := by sorry

end number_puzzle_l572_57256


namespace life_insurance_amount_l572_57265

/-- Calculates the life insurance amount given Bobby's salary and deductions --/
theorem life_insurance_amount
  (weekly_salary : ℝ)
  (federal_tax_rate : ℝ)
  (state_tax_rate : ℝ)
  (health_insurance : ℝ)
  (parking_fee : ℝ)
  (final_amount : ℝ)
  (h1 : weekly_salary = 450)
  (h2 : federal_tax_rate = 1/3)
  (h3 : state_tax_rate = 0.08)
  (h4 : health_insurance = 50)
  (h5 : parking_fee = 10)
  (h6 : final_amount = 184) :
  weekly_salary - (weekly_salary * federal_tax_rate) - (weekly_salary * state_tax_rate) - health_insurance - parking_fee - final_amount = 20 := by
  sorry

#check life_insurance_amount

end life_insurance_amount_l572_57265


namespace min_sum_distances_l572_57257

/-- The minimum sum of distances from a point on the unit circle to two specific lines -/
theorem min_sum_distances : 
  let P : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 + y^2 = 1
  let d1 : ℝ × ℝ → ℝ := λ (x, y) ↦ |3*x - 4*y - 10| / 5
  let d2 : ℝ × ℝ → ℝ := λ (x, y) ↦ |x - 3|
  ∃ (x y : ℝ), P (x, y) ∧ 
    ∀ (a b : ℝ), P (a, b) → d1 (x, y) + d2 (x, y) ≤ d1 (a, b) + d2 (a, b) ∧
    d1 (x, y) + d2 (x, y) = 5 - 4 * Real.sqrt 5 / 5 :=
by sorry

end min_sum_distances_l572_57257


namespace largest_c_for_two_in_range_l572_57236

theorem largest_c_for_two_in_range : 
  let f (x c : ℝ) := 3 * x^2 - 6 * x + c
  ∃ (c_max : ℝ), c_max = 5 ∧ 
    (∀ c : ℝ, (∃ x : ℝ, f x c = 2) → c ≤ c_max) ∧
    (∃ x : ℝ, f x c_max = 2) :=
by sorry

end largest_c_for_two_in_range_l572_57236


namespace franks_age_l572_57211

theorem franks_age (frank gabriel lucy : ℕ) : 
  gabriel = frank - 3 →
  frank + gabriel = 17 →
  lucy = gabriel + 5 →
  lucy = gabriel + frank →
  frank = 10 := by
sorry

end franks_age_l572_57211


namespace retail_discount_l572_57272

theorem retail_discount (wholesale_price retail_price : ℝ) 
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : ∃ selling_price, selling_price = wholesale_price * 1.2 ∧ 
                         selling_price = retail_price * (1 - (retail_price - selling_price) / retail_price)) :
  (retail_price - wholesale_price * 1.2) / retail_price = 0.1 := by
sorry

end retail_discount_l572_57272


namespace helmet_sales_theorem_l572_57297

/-- Represents the monthly growth rate of helmet sales -/
def monthly_growth_rate : ℝ := sorry

/-- Represents the optimal selling price of helmets -/
def optimal_selling_price : ℝ := sorry

/-- April sales volume -/
def april_sales : ℝ := 100

/-- June sales volume -/
def june_sales : ℝ := 144

/-- Cost price per helmet -/
def cost_price : ℝ := 30

/-- Reference selling price -/
def reference_price : ℝ := 40

/-- Reference monthly sales volume -/
def reference_sales : ℝ := 600

/-- Sales volume decrease per yuan increase in price -/
def sales_decrease_rate : ℝ := 10

/-- Target monthly profit -/
def target_profit : ℝ := 10000

theorem helmet_sales_theorem :
  (april_sales * (1 + monthly_growth_rate)^2 = june_sales) ∧
  ((optimal_selling_price - cost_price) * 
   (reference_sales - sales_decrease_rate * (optimal_selling_price - reference_price)) = target_profit) ∧
  (monthly_growth_rate = 0.2) ∧
  (optimal_selling_price = 50) := by sorry

end helmet_sales_theorem_l572_57297


namespace same_grade_percentage_is_42_5_l572_57284

/-- Represents the number of students in the class -/
def total_students : ℕ := 40

/-- Represents the number of students who received the same grade on both tests -/
def same_grade_students : ℕ := 17

/-- Calculates the percentage of students who received the same grade on both tests -/
def same_grade_percentage : ℚ :=
  (same_grade_students : ℚ) / (total_students : ℚ) * 100

/-- Proves that the percentage of students who received the same grade on both tests is 42.5% -/
theorem same_grade_percentage_is_42_5 :
  same_grade_percentage = 42.5 := by
  sorry


end same_grade_percentage_is_42_5_l572_57284


namespace remainder_2519_div_9_l572_57230

theorem remainder_2519_div_9 : 2519 % 9 = 8 := by
  sorry

end remainder_2519_div_9_l572_57230


namespace paint_wall_theorem_l572_57262

/-- The length of wall that can be painted by a group of boys in a given time -/
def wall_length (num_boys : ℕ) (days : ℝ) (rate : ℝ) : ℝ :=
  num_boys * days * rate

theorem paint_wall_theorem (rate : ℝ) :
  wall_length 8 3.125 rate = 50 →
  wall_length 6 5 rate = 106.67 := by
  sorry

end paint_wall_theorem_l572_57262


namespace max_xy_value_l572_57218

theorem max_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : Real.sqrt 3 = Real.sqrt (9^x * 3^y)) : 
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ Real.sqrt 3 = Real.sqrt (9^a * 3^b) → x * y ≥ a * b) ∧ 
  x * y = 1/8 :=
sorry

end max_xy_value_l572_57218


namespace functional_equation_implies_identity_l572_57289

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- Theorem stating that any function satisfying the functional equation is the identity function -/
theorem functional_equation_implies_identity (f : ℝ → ℝ) 
  (h : FunctionalEquation f) : ∀ x : ℝ, f x = x := by
  sorry

end functional_equation_implies_identity_l572_57289


namespace absolute_value_not_three_implies_not_three_l572_57241

theorem absolute_value_not_three_implies_not_three (x : ℝ) : |x| ≠ 3 → x ≠ 3 := by
  sorry

end absolute_value_not_three_implies_not_three_l572_57241


namespace non_degenerate_ellipse_condition_l572_57273

def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∀ x y : ℝ, 3 * x^2 + 9 * y^2 - 12 * x + 18 * y = k → k > -21

theorem non_degenerate_ellipse_condition :
  ∀ k : ℝ, is_non_degenerate_ellipse k ↔ k > -21 := by sorry

end non_degenerate_ellipse_condition_l572_57273


namespace inequality_proof_l572_57247

theorem inequality_proof (n : ℕ) : 
  2 * n * (n.factorial / (3 * n).factorial) ^ (1 / (2 * n)) < Real.log 3 := by
  sorry

end inequality_proof_l572_57247


namespace sandy_molly_age_ratio_l572_57263

/-- The ratio of Sandy's age to Molly's age -/
def age_ratio (sandy_age molly_age : ℕ) : ℚ :=
  sandy_age / molly_age

theorem sandy_molly_age_ratio :
  let sandy_age : ℕ := 56
  let molly_age : ℕ := sandy_age + 16
  age_ratio sandy_age molly_age = 7 / 9 := by
  sorry


end sandy_molly_age_ratio_l572_57263


namespace odd_functions_identification_l572_57222

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Define the given functions
def F1 (f : RealFunction) : RealFunction := fun x ↦ -|f x|
def F2 (f : RealFunction) : RealFunction := fun x ↦ x * f (x^2)
def F3 (f : RealFunction) : RealFunction := fun x ↦ -f (-x)
def F4 (f : RealFunction) : RealFunction := fun x ↦ f x - f (-x)

-- State the theorem
theorem odd_functions_identification (f : RealFunction) :
  ¬IsOdd (F1 f) ∧ IsOdd (F2 f) ∧ IsOdd (F4 f) :=
sorry

end odd_functions_identification_l572_57222


namespace average_of_numbers_l572_57286

def numbers : List ℤ := [-5, -2, 0, 4, 8]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 1 := by
  sorry

end average_of_numbers_l572_57286


namespace ellipse_properties_l572_57259

structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_b_lt_a : b < a
  h_e_eq : e = (a^2 - b^2).sqrt / a

def standard_equation (E : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2 / E.a^2 + y^2 / E.b^2 = 1

def vertices (E : Ellipse) : Set (ℝ × ℝ) :=
  {(-E.a, 0), (E.a, 0)}

def foci (E : Ellipse) : Set (ℝ × ℝ) :=
  {(-E.a * E.e, 0), (E.a * E.e, 0)}

def major_axis_length (E : Ellipse) : ℝ := 2 * E.a

def focal_distance (E : Ellipse) : ℝ := 2 * E.a * E.e

theorem ellipse_properties (E : Ellipse) (h_a : E.a = 5) (h_e : E.e = 4/5) :
  standard_equation E ∧
  vertices E = {(-5, 0), (5, 0)} ∧
  foci E = {(-4, 0), (4, 0)} ∧
  major_axis_length E = 10 ∧
  focal_distance E = 8 :=
sorry

end ellipse_properties_l572_57259


namespace intersection_x_difference_l572_57208

/-- The difference between the x-coordinates of the intersection points of two parabolas -/
theorem intersection_x_difference (f g : ℝ → ℝ) (h₁ : ∀ x, f x = 3*x^2 - 6*x + 5) 
  (h₂ : ∀ x, g x = -2*x^2 - 4*x + 6) : 
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ |x₁ - x₂| = 2 * Real.sqrt 6 / 5 :=
sorry

end intersection_x_difference_l572_57208


namespace final_ball_is_green_l572_57276

/-- Represents the colors of balls in the bag -/
inductive Color
| Red
| Green

/-- Represents the state of the bag -/
structure BagState where
  red : Nat
  green : Nat

/-- The process of drawing and modifying balls -/
def drawProcess (state : BagState) : BagState :=
  sorry

/-- The theorem to be proved -/
theorem final_ball_is_green (initial : BagState) 
  (h1 : initial.red = 2020) 
  (h2 : initial.green = 2021) :
  ∃ (final : BagState), 
    (final.red + final.green = 1) ∧ 
    (final.green = 1) ∧
    (∃ (n : Nat), (drawProcess^[n] initial) = final) :=
  sorry

end final_ball_is_green_l572_57276


namespace circle_radius_in_rectangle_l572_57290

theorem circle_radius_in_rectangle (r : ℝ) : 
  r > 0 → 
  (π * r^2 = 72 / 2) → 
  r = 6 / Real.sqrt π :=
by
  sorry

end circle_radius_in_rectangle_l572_57290


namespace particular_number_divisibility_l572_57255

theorem particular_number_divisibility (n : ℕ) : 
  n % 5 = 0 ∧ n / 5 = (320 / 4) + 220 → n / 3 = 500 := by
  sorry

end particular_number_divisibility_l572_57255


namespace equation_solution_l572_57209

theorem equation_solution : ∃! x : ℝ, (3 : ℝ) / (x - 3) = (4 : ℝ) / (x - 4) := by
  sorry

end equation_solution_l572_57209


namespace polygon_vertices_from_diagonals_l572_57231

/-- The number of vertices in a polygon given the number of diagonals from a single vertex -/
def num_vertices (diagonals_from_vertex : ℕ) : ℕ :=
  diagonals_from_vertex + 3

theorem polygon_vertices_from_diagonals (diagonals : ℕ) (h : diagonals = 6) :
  num_vertices diagonals = 9 := by
  sorry

end polygon_vertices_from_diagonals_l572_57231


namespace students_not_taking_languages_l572_57227

theorem students_not_taking_languages (total : ℕ) (french : ℕ) (spanish : ℕ) (both : ℕ) 
  (h1 : total = 28)
  (h2 : french = 5)
  (h3 : spanish = 10)
  (h4 : both = 4) :
  total - (french + spanish - both) = 17 := by
  sorry

#check students_not_taking_languages

end students_not_taking_languages_l572_57227


namespace remainder_2519_div_4_l572_57225

theorem remainder_2519_div_4 : 2519 % 4 = 3 := by
  sorry

end remainder_2519_div_4_l572_57225


namespace existence_of_integers_satisfying_inequality_l572_57240

theorem existence_of_integers_satisfying_inequality :
  ∃ (A B : ℤ), (999/1000 : ℝ) < A + B * Real.sqrt 2 ∧ A + B * Real.sqrt 2 < 1 := by
  sorry

end existence_of_integers_satisfying_inequality_l572_57240


namespace vector_parallel_condition_l572_57238

/-- Given vectors a, b, and c in ℝ², prove that if (a - c) is parallel to b, then k = 5. -/
theorem vector_parallel_condition (k : ℝ) : 
  let a : Fin 2 → ℝ := ![3, 1]
  let b : Fin 2 → ℝ := ![1, 3]
  let c : Fin 2 → ℝ := ![k, 7]
  (∃ (t : ℝ), (a - c) = t • b) → k = 5 := by
  sorry

end vector_parallel_condition_l572_57238


namespace angle_bisectors_rational_l572_57213

/-- Given a triangle with sides a = 84, b = 125, and c = 169, 
    the lengths of all angle bisectors are rational numbers -/
theorem angle_bisectors_rational (a b c : ℚ) (h1 : a = 84) (h2 : b = 125) (h3 : c = 169) :
  ∃ (fa fb fc : ℚ), 
    (fa = 2 * b * c / (b + c) * (((b^2 + c^2 - a^2) / (2 * b * c) + 1) / 2).sqrt) ∧
    (fb = 2 * a * c / (a + c) * (((a^2 + c^2 - b^2) / (2 * a * c) + 1) / 2).sqrt) ∧
    (fc = 2 * a * b / (a + b) * (((a^2 + b^2 - c^2) / (2 * a * b) + 1) / 2).sqrt) :=
by sorry

end angle_bisectors_rational_l572_57213


namespace greatest_multiple_under_1000_l572_57214

theorem greatest_multiple_under_1000 : ∃ (n : ℕ), n = 990 ∧ 
  (∀ m : ℕ, m < 1000 → m % 5 = 0 → m % 6 = 0 → m ≤ n) := by
  sorry

end greatest_multiple_under_1000_l572_57214


namespace hyperbola_y_axis_implies_m_negative_l572_57287

/-- A curve represented by the equation x²/m + y²/(1-m) = 1 -/
def is_curve (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2/m + y^2/(1-m) = 1

/-- The curve is a hyperbola with foci on the y-axis -/
def is_hyperbola_y_axis (m : ℝ) : Prop :=
  is_curve m ∧ ∃ c : ℝ, c > 0 ∧ ∀ x y : ℝ, y^2/(1-m) - x^2/(-m) = 1

/-- The theorem stating that if the curve is a hyperbola with foci on the y-axis, then m < 0 -/
theorem hyperbola_y_axis_implies_m_negative (m : ℝ) :
  is_hyperbola_y_axis m → m < 0 := by sorry

end hyperbola_y_axis_implies_m_negative_l572_57287


namespace arithmetic_sequence_problem_l572_57299

theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →
  a 2 + 4 * a 7 + a 12 = 96 →
  2 * a 3 + a 15 = 48 := by
  sorry

end arithmetic_sequence_problem_l572_57299


namespace six_heads_before_tail_l572_57281

/-- The probability of getting exactly n consecutive heads when flipping a fair coin -/
def prob_n_heads (n : ℕ) : ℚ :=
  1 / 2^n

/-- The probability of getting at least n consecutive heads before a tail when flipping a fair coin -/
def prob_at_least_n_heads (n : ℕ) : ℚ :=
  prob_n_heads n

theorem six_heads_before_tail (q : ℚ) :
  (q = prob_at_least_n_heads 6) → (q = 1 / 64) :=
by sorry

#eval (1 : ℕ) + (64 : ℕ)  -- Should output 65

end six_heads_before_tail_l572_57281


namespace log_sqrt8_512sqrt8_l572_57204

theorem log_sqrt8_512sqrt8 : Real.log (512 * Real.sqrt 8) / Real.log (Real.sqrt 8) = 7 := by
  sorry

end log_sqrt8_512sqrt8_l572_57204


namespace shelter_dogs_l572_57271

theorem shelter_dogs (dogs cats : ℕ) : 
  dogs * 7 = cats * 15 → 
  dogs * 11 = (cats + 8) * 15 → 
  dogs = 30 := by
sorry

end shelter_dogs_l572_57271


namespace smallest_sum_abcd_l572_57283

/-- Given positive integers A, B, C forming an arithmetic sequence,
    and integers B, C, D forming a geometric sequence,
    with C/B = 7/3, prove that the smallest possible value of A + B + C + D is 76. -/
theorem smallest_sum_abcd (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ d : ℤ, C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (∃ r : ℚ, C = r * B ∧ D = r * C) →  -- B, C, D form a geometric sequence
  C = (7 : ℚ) / 3 * B →  -- C/B = 7/3
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ d : ℤ, C' - B' = B' - A') →
    (∃ r : ℚ, C' = r * B' ∧ D' = r * C') →
    C' = (7 : ℚ) / 3 * B' →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 76 := by
sorry

end smallest_sum_abcd_l572_57283


namespace division_problem_l572_57205

theorem division_problem (divisor : ℕ) : 
  (15 / divisor = 4) ∧ (15 % divisor = 3) → divisor = 3 := by
  sorry

end division_problem_l572_57205


namespace unique_monotonic_function_l572_57270

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the property of being monotonic
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y ∨ f x > f y

-- Define the functional equation
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : PositiveReals, f (x * y) * f (f y / x) = 1

-- State the theorem
theorem unique_monotonic_function 
  (f : ℝ → ℝ) 
  (h1 : ∀ x : PositiveReals, f x > 0)
  (h2 : Monotonic f)
  (h3 : FunctionalEquation f) :
  ∀ x : PositiveReals, f x = 1 / x :=
sorry

end unique_monotonic_function_l572_57270


namespace repeating_decimal_equation_l572_57274

/-- A single-digit natural number -/
def SingleDigit (n : ℕ) : Prop := 0 < n ∧ n < 10

/-- Represents a repeating decimal of the form 0.ȳ -/
def RepeatingDecimal (y : ℕ) : ℚ := (y : ℚ) / 9

/-- The main theorem statement -/
theorem repeating_decimal_equation :
  ∀ x y : ℕ, SingleDigit y →
    (x / y + 1 = x + RepeatingDecimal y) ↔ ((x = 1 ∧ y = 3) ∨ (x = 0 ∧ y = 9)) :=
sorry

end repeating_decimal_equation_l572_57274


namespace right_triangle_yz_l572_57232

/-- In a right triangle XYZ, given angle X, angle Y, and hypotenuse XZ, calculate YZ --/
theorem right_triangle_yz (X Y Z : ℝ) (angleX : ℝ) (angleY : ℝ) (XZ : ℝ) : 
  angleX = 25 * π / 180 →  -- Convert 25° to radians
  angleY = π / 2 →         -- 90° in radians
  XZ = 18 →
  abs (Y - (XZ * Real.sin angleX)) < 0.0001 :=
by sorry

end right_triangle_yz_l572_57232


namespace max_a_value_l572_57261

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := True

-- Define the line equation
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

-- Define the condition for no lattice points
def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 200 → is_lattice_point x y →
    line_equation m x ≠ y

-- State the theorem
theorem max_a_value :
  ∀ a : ℚ, (∀ m : ℚ, 1/2 < m ∧ m < a → no_lattice_points m) →
    a ≤ 101/201 :=
sorry

end max_a_value_l572_57261


namespace convex_quadrilateral_count_lower_bound_l572_57260

/-- A set of points in a plane -/
structure PointSet where
  n : ℕ
  points : Fin n → ℝ × ℝ

/-- Predicate to check if three points are collinear -/
def collinear (p q r : ℝ × ℝ) : Prop := sorry

/-- Count of convex quadrilaterals in a set of points -/
def convexQuadrilateralCount (s : PointSet) : ℕ := sorry

theorem convex_quadrilateral_count_lower_bound (s : PointSet) 
  (h1 : s.n > 4)
  (h2 : ∀ p q r, p ≠ q ∧ q ≠ r ∧ p ≠ r → ¬collinear (s.points p) (s.points q) (s.points r)) :
  convexQuadrilateralCount s ≥ (s.n - 3) * (s.n - 4) / 2 := by sorry

end convex_quadrilateral_count_lower_bound_l572_57260


namespace symmetric_matrix_square_sum_l572_57216

theorem symmetric_matrix_square_sum (x y z : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; y, z]
  (∀ i j, B i j = B j i) →  -- B is symmetric
  B * B = (1 : Matrix (Fin 2) (Fin 2) ℝ) →  -- B^2 = I
  x^2 + 2*y^2 + z^2 = 2 := by
sorry

end symmetric_matrix_square_sum_l572_57216


namespace sin_value_from_tan_l572_57258

theorem sin_value_from_tan (α : Real) : 
  α > 0 ∧ α < Real.pi / 2 →  -- α is in the first quadrant
  Real.tan α = 3 / 4 →       -- tan α = 3/4
  Real.sin α = 3 / 5 :=      -- sin α = 3/5
by
  sorry

end sin_value_from_tan_l572_57258


namespace carla_cooks_three_steaks_l572_57242

/-- Represents the cooking scenario for Carla --/
structure CookingScenario where
  waffle_time : ℕ    -- Time to cook a batch of waffles in minutes
  steak_time : ℕ     -- Time to cook one steak in minutes
  total_time : ℕ     -- Total cooking time in minutes

/-- Calculates the number of steaks Carla needs to cook --/
def steaks_to_cook (scenario : CookingScenario) : ℕ :=
  (scenario.total_time - scenario.waffle_time) / scenario.steak_time

/-- Theorem stating that Carla needs to cook 3 steaks --/
theorem carla_cooks_three_steaks (scenario : CookingScenario) 
  (h1 : scenario.waffle_time = 10)
  (h2 : scenario.steak_time = 6)
  (h3 : scenario.total_time = 28) :
  steaks_to_cook scenario = 3 := by
  sorry

#eval steaks_to_cook { waffle_time := 10, steak_time := 6, total_time := 28 }

end carla_cooks_three_steaks_l572_57242


namespace solution_to_system_l572_57235

theorem solution_to_system (x y m : ℝ) 
  (eq1 : 4 * x + 2 * y = 3 * m)
  (eq2 : 3 * x + y = m + 2)
  (opposite : y = -x) : m = 1 := by
  sorry

end solution_to_system_l572_57235


namespace incorrect_inequality_l572_57264

theorem incorrect_inequality (a b : ℝ) (ha : -1 < a ∧ a < 2) (hb : -2 < b ∧ b < 3) :
  ¬ (∀ a b, -1 < a ∧ a < 2 ∧ -2 < b ∧ b < 3 → 2 < a * b ∧ a * b < 6) :=
by sorry

end incorrect_inequality_l572_57264


namespace no_pascal_row_with_four_distinct_elements_l572_57200

theorem no_pascal_row_with_four_distinct_elements : 
  ¬ ∃ (n : ℕ) (k m : ℕ) (a b c d : ℕ), 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (b = Nat.choose n k) ∧
    (d = Nat.choose n m) ∧
    (a = b / 2) ∧
    (c = d / 2) :=
by sorry

end no_pascal_row_with_four_distinct_elements_l572_57200


namespace oranges_for_juice_is_30_l572_57280

/-- Given a number of bags of oranges, oranges per bag, rotten oranges, and oranges to be sold,
    calculate the number of oranges kept for juice. -/
def oranges_for_juice (bags : ℕ) (oranges_per_bag : ℕ) (rotten : ℕ) (to_sell : ℕ) : ℕ :=
  bags * oranges_per_bag - rotten - to_sell

/-- Theorem stating that under the given conditions, 30 oranges will be kept for juice. -/
theorem oranges_for_juice_is_30 :
  oranges_for_juice 10 30 50 220 = 30 := by
  sorry

end oranges_for_juice_is_30_l572_57280


namespace equation_solutions_l572_57206

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 1, 2)} ∪ {(k, 2, 3*k) | k : ℕ} ∪ {(2, 3, 18)} ∪ {(1, 2*k, 3*k) | k : ℕ} ∪ {(2, 2, 6)}

theorem equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | x > 0 ∧ y > 0 ∧ z > 0 ∧ (1 : ℚ) / x + 2 / y - 3 / z = 1} = solution_set :=
by sorry

end equation_solutions_l572_57206


namespace circle_diameter_ratio_l572_57228

theorem circle_diameter_ratio (R S : ℝ) (harea : π * R^2 = 0.04 * π * S^2) :
  2 * R = 0.4 * (2 * S) := by
  sorry

end circle_diameter_ratio_l572_57228


namespace min_sum_bound_l572_57244

theorem min_sum_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b^2 / (6 * c^2) + c^3 / (9 * a^3) ≥ 3 / Real.rpow 162 (1/3) ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' / (3 * b') + b'^2 / (6 * c'^2) + c'^3 / (9 * a'^3) = 3 / Real.rpow 162 (1/3) :=
by sorry

end min_sum_bound_l572_57244


namespace circle_symmetry_l572_57212

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 4)^2 + (y + 1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), original_circle x y ∧ symmetry_line x y → symmetric_circle x y :=
sorry

end circle_symmetry_l572_57212


namespace renovation_project_dirt_calculation_l572_57201

theorem renovation_project_dirt_calculation (total material sand cement : ℚ)
  (h1 : sand = 0.17)
  (h2 : cement = 0.17)
  (h3 : total = 0.67)
  (h4 : material = total - (sand + cement)) :
  material = 0.33 := by
  sorry

end renovation_project_dirt_calculation_l572_57201


namespace city_transport_capacity_l572_57245

/-- Represents the capacity of different public transport vehicles in a small city -/
structure CityTransport where
  train_capacity : ℕ
  bus_capacity : ℕ
  tram_capacity : ℕ

/-- Calculates the total capacity of two buses and a tram given the conditions -/
def total_capacity (ct : CityTransport) : ℕ :=
  2 * ct.bus_capacity + ct.tram_capacity

/-- Theorem stating the total capacity of two buses and a tram in the city -/
theorem city_transport_capacity : ∃ (ct : CityTransport),
  ct.train_capacity = 120 ∧
  ct.bus_capacity = ct.train_capacity / 6 ∧
  ct.tram_capacity = (2 * ct.bus_capacity) * 2 / 3 ∧
  total_capacity ct = 67 := by
  sorry


end city_transport_capacity_l572_57245


namespace consecutive_numbers_percentage_l572_57217

theorem consecutive_numbers_percentage (a b c d e f g : ℤ) : 
  (a + b + c + d + e + f + g) / 7 = 9 ∧ 
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧ f = e + 1 ∧ g = f + 1 →
  a * 100 / g = 50 := by
sorry

end consecutive_numbers_percentage_l572_57217


namespace ball_return_to_start_l572_57237

theorem ball_return_to_start (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 5) : 
  ∃ m : ℕ, m > 0 ∧ (m * k) % n = 0 ∧ m = 3 :=
sorry

end ball_return_to_start_l572_57237


namespace water_per_pig_l572_57293

-- Define the given conditions
def pump_rate : ℚ := 3
def pumping_time : ℚ := 25
def corn_rows : ℕ := 4
def corn_plants_per_row : ℕ := 15
def water_per_corn_plant : ℚ := 1/2
def num_pigs : ℕ := 10
def num_ducks : ℕ := 20
def water_per_duck : ℚ := 1/4

-- Theorem to prove
theorem water_per_pig : 
  (pump_rate * pumping_time - 
   (corn_rows * corn_plants_per_row : ℚ) * water_per_corn_plant - 
   (num_ducks : ℚ) * water_per_duck) / (num_pigs : ℚ) = 4 := by
  sorry

end water_per_pig_l572_57293


namespace max_surface_area_rectangular_solid_l572_57275

theorem max_surface_area_rectangular_solid (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 + c^2 = 36) : 
  2*a*b + 2*a*c + 2*b*c ≤ 72 :=
sorry

end max_surface_area_rectangular_solid_l572_57275


namespace sum_altitudes_less_perimeter_l572_57278

/-- A triangle with sides a, b, c and corresponding altitudes h₁, h₂, h₃ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_h₁ : 0 < h₁
  pos_h₂ : 0 < h₂
  pos_h₃ : 0 < h₃
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b
  altitude_relation : h₁ * a = 2 * area ∧ h₂ * b = 2 * area ∧ h₃ * c = 2 * area
  area_pos : 0 < area

/-- The sum of the altitudes of a triangle is less than its perimeter -/
theorem sum_altitudes_less_perimeter (t : Triangle) : t.h₁ + t.h₂ + t.h₃ < t.a + t.b + t.c := by
  sorry

end sum_altitudes_less_perimeter_l572_57278


namespace sum_range_for_distinct_positive_numbers_l572_57279

theorem sum_range_for_distinct_positive_numbers (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_distinct : a ≠ b) 
  (h_eq : a^2 + a*b + b^2 = a + b) : 
  1 < a + b ∧ a + b < 4/3 := by
sorry

end sum_range_for_distinct_positive_numbers_l572_57279


namespace wire_ratio_proof_l572_57282

theorem wire_ratio_proof (total_length longer_piece shorter_piece : ℤ) : 
  total_length = 90 ∧ shorter_piece = 20 ∧ longer_piece = total_length - shorter_piece →
  shorter_piece / longer_piece = 2 / 7 := by
  sorry

end wire_ratio_proof_l572_57282


namespace series_sum_equals_one_third_l572_57288

/-- The sum of the infinite series ∑(k=1 to ∞) [2^k / (8^k - 1)] is equal to 1/3 -/
theorem series_sum_equals_one_third :
  ∑' k, (2 : ℝ)^k / ((8 : ℝ)^k - 1) = 1/3 := by sorry

end series_sum_equals_one_third_l572_57288


namespace happy_street_weekly_total_l572_57267

/-- The number of cars traveling down Happy Street each day of the week -/
structure WeeklyTraffic where
  tuesday : ℕ
  monday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Conditions for the traffic on Happy Street -/
def happy_street_traffic : WeeklyTraffic where
  tuesday := 25
  monday := 25 - (25 * 20 / 100)
  wednesday := (25 - (25 * 20 / 100)) + 2
  thursday := 10
  friday := 10
  saturday := 5
  sunday := 5

/-- The total number of cars traveling down Happy Street in a week -/
def total_weekly_traffic (w : WeeklyTraffic) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday + w.sunday

/-- Theorem stating that the total number of cars traveling down Happy Street in a week is 97 -/
theorem happy_street_weekly_total :
  total_weekly_traffic happy_street_traffic = 97 := by
  sorry

end happy_street_weekly_total_l572_57267


namespace prob_both_blue_is_one_third_l572_57254

/-- Represents a jar containing buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- The initial state of Jar C -/
def initial_jar_c : Jar := { red := 6, blue := 10 }

/-- The number of buttons removed from each color -/
def buttons_removed : ℕ := 2

/-- The final state of Jar C after removal -/
def final_jar_c : Jar := 
  { red := initial_jar_c.red - buttons_removed,
    blue := initial_jar_c.blue - buttons_removed }

/-- The state of Jar D after buttons are added -/
def jar_d : Jar := { red := buttons_removed, blue := buttons_removed }

/-- The probability of selecting a blue button from a jar -/
def prob_blue (jar : Jar) : ℚ :=
  jar.blue / (jar.red + jar.blue)

theorem prob_both_blue_is_one_third :
  prob_blue final_jar_c * prob_blue jar_d = 1/3 := by
  sorry

#eval prob_blue final_jar_c -- Expected: 2/3
#eval prob_blue jar_d -- Expected: 1/2
#eval prob_blue final_jar_c * prob_blue jar_d -- Expected: 1/3

end prob_both_blue_is_one_third_l572_57254


namespace binomial_15_12_l572_57223

theorem binomial_15_12 : Nat.choose 15 12 = 455 := by
  sorry

end binomial_15_12_l572_57223


namespace inequality_proof_l572_57250

theorem inequality_proof (a : ℝ) (n : ℕ) (h1 : a > -1) (h2 : a ≠ 0) (h3 : n ≥ 2) :
  (1 + a)^n > 1 + n * a := by
  sorry

end inequality_proof_l572_57250


namespace lines_perp_to_plane_are_parallel_planes_perp_to_line_are_parallel_l572_57203

/-- A 3D space -/
structure Space3D where
  -- Add necessary structure here

/-- A line in 3D space -/
structure Line3D (S : Space3D) where
  -- Add necessary structure here

/-- A plane in 3D space -/
structure Plane3D (S : Space3D) where
  -- Add necessary structure here

/-- Perpendicularity between a line and a plane -/
def perpendicular_line_plane (S : Space3D) (l : Line3D S) (p : Plane3D S) : Prop :=
  sorry

/-- Perpendicularity between a plane and a line -/
def perpendicular_plane_line (S : Space3D) (p : Plane3D S) (l : Line3D S) : Prop :=
  sorry

/-- Parallelism between two lines -/
def parallel_lines (S : Space3D) (l1 l2 : Line3D S) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel_planes (S : Space3D) (p1 p2 : Plane3D S) : Prop :=
  sorry

/-- Theorem: Two lines perpendicular to the same plane are parallel to each other -/
theorem lines_perp_to_plane_are_parallel (S : Space3D) (l1 l2 : Line3D S) (p : Plane3D S)
  (h1 : perpendicular_line_plane S l1 p) (h2 : perpendicular_line_plane S l2 p) :
  parallel_lines S l1 l2 :=
sorry

/-- Theorem: Two planes perpendicular to the same line are parallel to each other -/
theorem planes_perp_to_line_are_parallel (S : Space3D) (p1 p2 : Plane3D S) (l : Line3D S)
  (h1 : perpendicular_plane_line S p1 l) (h2 : perpendicular_plane_line S p2 l) :
  parallel_planes S p1 p2 :=
sorry

end lines_perp_to_plane_are_parallel_planes_perp_to_line_are_parallel_l572_57203


namespace cos_negative_third_quadrants_l572_57207

-- Define the quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define a function to determine the possible quadrants for a given cosine value
def possibleQuadrants (cosθ : ℝ) : Set Quadrant :=
  if cosθ > 0 then {Quadrant.First, Quadrant.Fourth}
  else if cosθ < 0 then {Quadrant.Second, Quadrant.Third}
  else {Quadrant.First, Quadrant.Second, Quadrant.Third, Quadrant.Fourth}

-- Theorem statement
theorem cos_negative_third_quadrants :
  let cosθ : ℝ := -1/3
  possibleQuadrants cosθ = {Quadrant.Second, Quadrant.Third} :=
by sorry


end cos_negative_third_quadrants_l572_57207


namespace seed_germination_problem_l572_57268

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  0.25 * x + 0.3 * 200 = 0.27 * (x + 200) → 
  x = 300 := by
sorry

end seed_germination_problem_l572_57268


namespace pure_imaginary_number_l572_57253

theorem pure_imaginary_number (x : ℝ) : 
  (((x - 2008) : ℂ) + (x + 2007)*I).re = 0 ∧ (((x - 2008) : ℂ) + (x + 2007)*I).im ≠ 0 → x = 2008 := by
  sorry

end pure_imaginary_number_l572_57253


namespace tim_added_fourteen_rulers_l572_57248

/-- Given an initial number of rulers and a final number of rulers,
    calculate the number of rulers added. -/
def rulers_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given 11 initial rulers and 25 final rulers,
    the number of rulers added is 14. -/
theorem tim_added_fourteen_rulers :
  rulers_added 11 25 = 14 := by
  sorry

end tim_added_fourteen_rulers_l572_57248


namespace average_children_in_families_with_children_l572_57226

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children_per_family : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children_per_family = 3)
  (h3 : childless_families = 3) :
  (total_families * average_children_per_family) / (total_families - childless_families) = 3.75 := by
sorry

end average_children_in_families_with_children_l572_57226


namespace triangle_circumcircle_intersection_l572_57291

-- Define the triangle
def triangle_PQR (P Q R : ℝ × ℝ) : Prop :=
  dist P Q = 37 ∧ dist Q R = 20 ∧ dist R P = 45

-- Define the circumcircle
def circumcircle (P Q R S : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ) (r : ℝ), dist O P = r ∧ dist O Q = r ∧ dist O R = r ∧ dist O S = r

-- Define the perpendicular bisector
def perp_bisector (P R S : ℝ × ℝ) : Prop :=
  dist P S = dist R S ∧ (S.1 - P.1) * (R.1 - P.1) + (S.2 - P.2) * (R.2 - P.2) = 0

-- Main theorem
theorem triangle_circumcircle_intersection 
  (P Q R S : ℝ × ℝ) 
  (h_triangle : triangle_PQR P Q R)
  (h_circumcircle : circumcircle P Q R S)
  (h_perp_bisector : perp_bisector P R S)
  (h_opposite_side : (S.1 - P.1) * (Q.1 - P.1) + (S.2 - P.2) * (Q.2 - P.2) < 0) :
  ∃ (a b : ℕ), 
    a = 15 ∧ 
    b = 27 ∧ 
    dist P S = a * Real.sqrt b ∧
    ⌊a + Real.sqrt b⌋ = 20 :=
  sorry

end triangle_circumcircle_intersection_l572_57291


namespace line_equation_through_point_with_slope_l572_57219

/-- A line passing through (-2, 3) with slope 2 has the equation 2x - y - 7 = 0 -/
theorem line_equation_through_point_with_slope :
  let point : ℝ × ℝ := (-2, 3)
  let slope : ℝ := 2
  let line_equation (x y : ℝ) := 2 * x - y - 7 = 0
  (∀ x y, line_equation x y ↔ y - point.2 = slope * (x - point.1)) ∧
  line_equation point.1 point.2 := by
  sorry

end line_equation_through_point_with_slope_l572_57219


namespace pieces_from_rod_l572_57210

/-- The number of pieces of a given length that can be cut from a rod. -/
def number_of_pieces (rod_length_m : ℕ) (piece_length_cm : ℕ) : ℕ :=
  (rod_length_m * 100) / piece_length_cm

/-- Theorem: The number of 85 cm pieces that can be cut from a 34-meter rod is 40. -/
theorem pieces_from_rod : number_of_pieces 34 85 = 40 := by
  sorry

end pieces_from_rod_l572_57210


namespace community_average_age_l572_57239

theorem community_average_age 
  (k : ℕ) 
  (h_k : k > 0) 
  (women : ℕ := 7 * k) 
  (men : ℕ := 8 * k) 
  (women_avg_age : ℚ := 30) 
  (men_avg_age : ℚ := 35) : 
  (women_avg_age * women + men_avg_age * men) / (women + men) = 98 / 3 := by
sorry

end community_average_age_l572_57239


namespace B_is_largest_l572_57269

def A : ℚ := 2008/2007 + 2008/2009
def B : ℚ := 2010/2009 + 2 * (2010/2009)
def C : ℚ := 2009/2008 + 2009/2010

theorem B_is_largest : B > A ∧ B > C := by
  sorry

end B_is_largest_l572_57269


namespace fraction_value_l572_57296

theorem fraction_value : 
  let a := 423134
  let b := 423133
  (a * 846267 - b) / (b * 846267 + a) = 1 := by sorry

end fraction_value_l572_57296


namespace student_performance_l572_57295

structure Student :=
  (name : String)
  (scores : Fin 6 → ℝ)

def class_avg : Fin 6 → ℝ
| 0 => 128.2
| 1 => 118.3
| 2 => 125.4
| 3 => 120.3
| 4 => 115.7
| 5 => 122.1

def student_A : Student :=
  ⟨"A", λ i => [138, 127, 131, 132, 128, 135].get i⟩

def student_B : Student :=
  ⟨"B", λ i => [130, 116, 128, 115, 126, 120].get i⟩

def student_C : Student :=
  ⟨"C", λ i => [108, 105, 113, 112, 115, 123].get i⟩

theorem student_performance :
  (∀ i : Fin 6, student_A.scores i > class_avg i) ∧
  (∃ i j : Fin 6, student_B.scores i > class_avg i ∧ student_B.scores j < class_avg j) ∧
  (∃ k : Fin 6, ∀ i j : Fin 6, i < j → j ≥ k →
    (student_C.scores j - class_avg j) > (student_C.scores i - class_avg i)) :=
by sorry

end student_performance_l572_57295


namespace trihedral_angle_sum_l572_57202

/-- Represents a trihedral angle with plane angles α, β, and γ. -/
structure TrihedralAngle where
  α : ℝ
  β : ℝ
  γ : ℝ
  α_pos : 0 < α
  β_pos : 0 < β
  γ_pos : 0 < γ

/-- The sum of any two plane angles of a trihedral angle is greater than the third. -/
theorem trihedral_angle_sum (t : TrihedralAngle) : t.α + t.β > t.γ ∧ t.β + t.γ > t.α ∧ t.α + t.γ > t.β := by
  sorry

end trihedral_angle_sum_l572_57202


namespace second_draw_pink_probability_l572_57277

/-- Represents a bag of marbles -/
structure Bag where
  red : ℕ
  green : ℕ
  pink : ℕ
  purple : ℕ

/-- The probability of drawing a pink marble in the second draw -/
def second_draw_pink_prob (bagA bagB bagC : Bag) : ℚ :=
  let total_A := bagA.red + bagA.green
  let total_B := bagB.pink + bagB.purple
  let total_C := bagC.pink + bagC.purple
  let prob_red := bagA.red / total_A
  let prob_green := bagA.green / total_A
  let prob_pink_B := bagB.pink / total_B
  let prob_pink_C := bagC.pink / total_C
  prob_red * prob_pink_B + prob_green * prob_pink_C

theorem second_draw_pink_probability :
  let bagA : Bag := { red := 5, green := 5, pink := 0, purple := 0 }
  let bagB : Bag := { red := 0, green := 0, pink := 8, purple := 2 }
  let bagC : Bag := { red := 0, green := 0, pink := 3, purple := 7 }
  second_draw_pink_prob bagA bagB bagC = 11 / 20 := by
  sorry

#eval second_draw_pink_prob
  { red := 5, green := 5, pink := 0, purple := 0 }
  { red := 0, green := 0, pink := 8, purple := 2 }
  { red := 0, green := 0, pink := 3, purple := 7 }

end second_draw_pink_probability_l572_57277


namespace missing_panels_l572_57215

/-- Calculates the number of missing solar panels in Faith's neighborhood. -/
theorem missing_panels (total_homes : Nat) (panels_per_home : Nat) (homes_with_panels : Nat) :
  total_homes = 20 →
  panels_per_home = 10 →
  homes_with_panels = 15 →
  total_homes * panels_per_home - homes_with_panels * panels_per_home = 50 :=
by
  sorry

#check missing_panels

end missing_panels_l572_57215


namespace permutation_17_14_l572_57252

/-- The falling factorial function -/
def fallingFactorial (n m : ℕ) : ℕ :=
  match m with
  | 0 => 1
  | m + 1 => n * fallingFactorial (n - 1) m

/-- The permutation function -/
def permutation (n m : ℕ) : ℕ := fallingFactorial n m

theorem permutation_17_14 :
  ∃ (n m : ℕ), permutation n m = (17 * 16 * 15 * 14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4) ∧ n = 17 ∧ m = 14 := by
  sorry

#check permutation_17_14

end permutation_17_14_l572_57252


namespace count_solutions_eq_two_l572_57220

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem count_solutions_eq_two :
  (∃ (A : Finset ℕ), (∀ n ∈ A, n + S n + S (S n) = 2050) ∧ A.card = 2 ∧
   ∀ n : ℕ, n + S n + S (S n) = 2050 → n ∈ A) := by sorry

end count_solutions_eq_two_l572_57220


namespace solve_equation_l572_57292

theorem solve_equation (x : ℝ) (h : x - 2*x + 3*x - 4*x = 120) : x = -60 := by
  sorry

end solve_equation_l572_57292


namespace birthday_gift_savings_is_86_l572_57234

/-- The amount of money Liam and Claire save for their mother's birthday gift -/
def birthday_gift_savings (liam_oranges : ℕ) (liam_price : ℚ) (claire_oranges : ℕ) (claire_price : ℚ) : ℚ :=
  (liam_oranges / 2 : ℚ) * liam_price + claire_oranges * claire_price

/-- Theorem stating that Liam and Claire save $86 for their mother's birthday gift -/
theorem birthday_gift_savings_is_86 :
  birthday_gift_savings 40 (5/2) 30 (6/5) = 86 := by
  sorry

end birthday_gift_savings_is_86_l572_57234


namespace expression_value_l572_57246

theorem expression_value (x y : ℝ) (h : x^2 - 4*x - 1 = 0) :
  (2*x - 3)^2 - (x + y)*(x - y) - y^2 = 12 := by
  sorry

end expression_value_l572_57246


namespace equation_solution_l572_57229

theorem equation_solution (x : ℝ) (h : x ≠ 2/3) :
  (6*x + 2) / (3*x^2 + 6*x - 4) = 3*x / (3*x - 2) ↔ x = 1 / Real.sqrt 3 ∨ x = -1 / Real.sqrt 3 :=
by sorry

end equation_solution_l572_57229


namespace lesser_fraction_l572_57251

theorem lesser_fraction (x y : ℚ) 
  (sum_eq : x + y = 13/14)
  (prod_eq : x * y = 1/8) :
  min x y = (13 - Real.sqrt 57) / 28 := by
sorry

end lesser_fraction_l572_57251


namespace vector_projection_l572_57294

theorem vector_projection (a b : ℝ × ℝ) (h1 : a = (3, 2)) (h2 : b = (-2, 1)) :
  let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  proj = -4 * Real.sqrt 5 / 5 := by
  sorry

end vector_projection_l572_57294
