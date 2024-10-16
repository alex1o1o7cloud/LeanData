import Mathlib

namespace NUMINAMATH_CALUDE_dividend_calculation_l2312_231298

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 18) 
  (h2 : quotient = 9) 
  (h3 : remainder = 4) : 
  divisor * quotient + remainder = 166 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2312_231298


namespace NUMINAMATH_CALUDE_turtle_ratio_l2312_231281

theorem turtle_ratio (total : ℕ) (green : ℕ) (h1 : total = 3200) (h2 : green = 800) :
  (total - green) / green = 3 := by
  sorry

end NUMINAMATH_CALUDE_turtle_ratio_l2312_231281


namespace NUMINAMATH_CALUDE_base_7_326_equals_base_4_2213_l2312_231250

def base_7_to_decimal (x y z : ℕ) : ℕ := x * 7^2 + y * 7 + z

def decimal_to_base_4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_7_326_equals_base_4_2213 :
  decimal_to_base_4 (base_7_to_decimal 3 2 6) = [2, 2, 1, 3] := by
  sorry

end NUMINAMATH_CALUDE_base_7_326_equals_base_4_2213_l2312_231250


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2312_231284

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * (-(2 + a) / 2) = (2 - a * Complex.I) / (1 + Complex.I)) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2312_231284


namespace NUMINAMATH_CALUDE_intersection_max_value_l2312_231273

/-- The polynomial function f(x) = x^6 - 10x^5 + 30x^4 - 20x^3 + 50x^2 - 24x + 48 -/
def f (x : ℝ) : ℝ := x^6 - 10*x^5 + 30*x^4 - 20*x^3 + 50*x^2 - 24*x + 48

/-- The line function g(x) = 8x -/
def g (x : ℝ) : ℝ := 8*x

theorem intersection_max_value :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (∀ x : ℝ, f x = g x → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (∃ x : ℝ, f x = g x ∧ ∀ y : ℝ, f y = g y → y ≤ x) →
  (∃ x : ℝ, f x = g x ∧ ∀ y : ℝ, f y = g y → x = 6) :=
sorry

end NUMINAMATH_CALUDE_intersection_max_value_l2312_231273


namespace NUMINAMATH_CALUDE_zeros_properties_l2312_231265

noncomputable def f (x : ℝ) : ℝ := (2 * x / (x - 2))^2 - 3^x

noncomputable def g (x : ℝ) : ℝ := 2 * (Real.log x / Real.log 3) - 4 / (x - 2) - 2

theorem zeros_properties (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 2) (h₂ : x₂ > 2) 
  (hf₁ : f x₁ = 0) (hf₂ : f x₂ = 0) 
  (hg₁ : g x₁ = 0) (hg₂ : g x₂ = 0) :
  x₂ > 3 ∧ 2*x₁ + 2*x₂ = x₁*x₂ ∧ x₁*x₂ > 16 :=
by sorry

end NUMINAMATH_CALUDE_zeros_properties_l2312_231265


namespace NUMINAMATH_CALUDE_clerical_staff_percentage_l2312_231202

def total_employees : ℕ := 3600
def initial_clerical_ratio : ℚ := 1 / 6
def clerical_reduction_ratio : ℚ := 1 / 4

theorem clerical_staff_percentage : 
  let initial_clerical := (initial_clerical_ratio * total_employees : ℚ)
  let reduced_clerical := initial_clerical - (clerical_reduction_ratio * initial_clerical)
  let remaining_employees := total_employees - (initial_clerical - reduced_clerical)
  (reduced_clerical / remaining_employees) * 100 = 450 / 3450 * 100 := by
  sorry

end NUMINAMATH_CALUDE_clerical_staff_percentage_l2312_231202


namespace NUMINAMATH_CALUDE_age_sum_proof_l2312_231212

theorem age_sum_proof (A B C : ℕ) 
  (h1 : A = B + C + 16) 
  (h2 : A^2 = (B + C)^2 + 1632) : 
  A + B + C = 102 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l2312_231212


namespace NUMINAMATH_CALUDE_fifth_term_of_specific_arithmetic_sequence_l2312_231200

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifth_term_of_specific_arithmetic_sequence :
  let a₁ : ℝ := 1
  let d : ℝ := 2
  arithmeticSequence a₁ d 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_specific_arithmetic_sequence_l2312_231200


namespace NUMINAMATH_CALUDE_ants_crushed_calculation_l2312_231288

/-- The number of ants crushed by a man's foot, given the original number of ants and the number of ants left alive -/
def antsCrushed (originalAnts : ℕ) (antsAlive : ℕ) : ℕ :=
  originalAnts - antsAlive

/-- Theorem stating that 60 ants were crushed when 102 ants were originally present and 42 ants remained alive -/
theorem ants_crushed_calculation :
  antsCrushed 102 42 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ants_crushed_calculation_l2312_231288


namespace NUMINAMATH_CALUDE_estimate_passing_papers_l2312_231259

theorem estimate_passing_papers (total_papers : ℕ) (sample_size : ℕ) (passing_in_sample : ℕ) 
  (h1 : total_papers = 5000)
  (h2 : sample_size = 400)
  (h3 : passing_in_sample = 360) :
  ⌊(total_papers : ℝ) * (passing_in_sample : ℝ) / (sample_size : ℝ)⌋ = 4500 := by
sorry

end NUMINAMATH_CALUDE_estimate_passing_papers_l2312_231259


namespace NUMINAMATH_CALUDE_sum_of_angles_l2312_231289

theorem sum_of_angles (α β : Real) : 
  (∃ x y : Real, x^2 - 3 * Real.sqrt 3 * x + 4 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  (0 < α ∧ α < π/2) →
  (0 < β ∧ β < π/2) →
  α + β = 2*π/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_angles_l2312_231289


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l2312_231216

theorem cylinder_radius_problem (rounds1 rounds2 : ℕ) (radius2 : ℝ) (radius1 : ℝ) :
  rounds1 = 70 →
  rounds2 = 49 →
  radius2 = 20 →
  rounds1 * (2 * Real.pi * radius1) = rounds2 * (2 * Real.pi * radius2) →
  radius1 = 14 := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l2312_231216


namespace NUMINAMATH_CALUDE_bank_through_window_l2312_231290

/-- Represents a letter as seen through a clear glass window from the inside --/
inductive MirroredLetter
  | Normal (c : Char)
  | Inverted (c : Char)

/-- Represents a word as seen through a clear glass window from the inside --/
def MirroredWord := List MirroredLetter

/-- Converts a character to its mirrored version --/
def mirrorChar (c : Char) : MirroredLetter :=
  match c with
  | 'B' => MirroredLetter.Inverted 'В'
  | 'A' => MirroredLetter.Normal 'A'
  | 'N' => MirroredLetter.Inverted 'И'
  | 'K' => MirroredLetter.Inverted 'И'
  | _ => MirroredLetter.Normal c

/-- Converts a string to its mirrored version --/
def mirrorWord (s : String) : MirroredWord :=
  s.toList.reverse.map mirrorChar

/-- Converts a MirroredWord to a string --/
def mirroredWordToString (w : MirroredWord) : String :=
  w.map (fun l => match l with
    | MirroredLetter.Normal c => c
    | MirroredLetter.Inverted c => c
  ) |>.asString

theorem bank_through_window :
  mirroredWordToString (mirrorWord "BANK") = "ИAИВ" := by
  sorry

#eval mirroredWordToString (mirrorWord "BANK")

end NUMINAMATH_CALUDE_bank_through_window_l2312_231290


namespace NUMINAMATH_CALUDE_unique_solution_l2312_231257

/-- A polynomial that satisfies the given functional equation -/
def functional_equation (p : ℝ → ℝ) : Prop :=
  ∀ x, p (p x) = 2 * x * p x + 3 * x^2

/-- The theorem stating that p(x) = 3x is the unique solution to the functional equation -/
theorem unique_solution :
  ∃! p : ℝ → ℝ, functional_equation p ∧ ∀ x, p x = 3 * x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2312_231257


namespace NUMINAMATH_CALUDE_arthur_muffins_l2312_231208

theorem arthur_muffins (initial_muffins : ℕ) (multiplier : ℚ) : 
  initial_muffins = 80 →
  multiplier = 5/2 →
  (multiplier * initial_muffins : ℚ) - initial_muffins = 120 :=
by sorry

end NUMINAMATH_CALUDE_arthur_muffins_l2312_231208


namespace NUMINAMATH_CALUDE_prism_volume_l2312_231275

/-- Given a right rectangular prism with dimensions a, b, and c satisfying certain conditions,
    prove that its volume is 200 cubic centimeters. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 50) (h2 : b * c = 80) (h3 : a * c = 100)
    (h4 : ∃ n : ℕ, (a * c : ℝ) = n ^ 2) : a * b * c = 200 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2312_231275


namespace NUMINAMATH_CALUDE_janelle_marbles_l2312_231211

/-- Calculates the total number of marbles Janelle has after a series of transactions. -/
def total_marbles (initial_green : ℕ) (blue_bags : ℕ) (marbles_per_bag : ℕ) 
  (gifted_red : ℕ) (gift_green : ℕ) (gift_blue : ℕ) (gift_red : ℕ) 
  (returned_blue : ℕ) : ℕ :=
  let total_blue := blue_bags * marbles_per_bag
  let remaining_green := initial_green - gift_green
  let remaining_blue := total_blue - gift_blue + returned_blue
  let remaining_red := gifted_red - gift_red
  remaining_green + remaining_blue + remaining_red

/-- Proves that Janelle ends up with 197 marbles given the initial conditions and transactions. -/
theorem janelle_marbles : 
  total_marbles 26 12 15 7 9 12 3 8 = 197 := by
  sorry

end NUMINAMATH_CALUDE_janelle_marbles_l2312_231211


namespace NUMINAMATH_CALUDE_jacob_age_jacob_age_proof_l2312_231236

theorem jacob_age : ℕ → Prop :=
  fun j : ℕ =>
    ∃ t : ℕ,
      t = j / 2 ∧  -- Tony's age is half of Jacob's age
      t + 6 = 18 ∧ -- In 6 years, Tony will be 18 years old
      j = 24       -- Jacob's current age is 24

-- The proof of the theorem
theorem jacob_age_proof : ∃ j : ℕ, jacob_age j :=
  sorry

end NUMINAMATH_CALUDE_jacob_age_jacob_age_proof_l2312_231236


namespace NUMINAMATH_CALUDE_melinda_doughnuts_count_l2312_231233

/-- The cost of one doughnut in dollars -/
def doughnut_cost : ℚ := 45/100

/-- The total cost of Harold's purchase in dollars -/
def harold_total : ℚ := 491/100

/-- The number of doughnuts Harold bought -/
def harold_doughnuts : ℕ := 3

/-- The number of coffees Harold bought -/
def harold_coffees : ℕ := 4

/-- The total cost of Melinda's purchase in dollars -/
def melinda_total : ℚ := 759/100

/-- The number of coffees Melinda bought -/
def melinda_coffees : ℕ := 6

/-- The number of doughnuts Melinda bought -/
def melinda_doughnuts : ℕ := 5

theorem melinda_doughnuts_count : 
  ∃ (coffee_cost : ℚ), 
    (harold_doughnuts : ℚ) * doughnut_cost + (harold_coffees : ℚ) * coffee_cost = harold_total ∧
    (melinda_doughnuts : ℚ) * doughnut_cost + (melinda_coffees : ℚ) * coffee_cost = melinda_total :=
by sorry

end NUMINAMATH_CALUDE_melinda_doughnuts_count_l2312_231233


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2312_231223

/-- The curve C defined by y = x^3 -/
def C : ℝ → ℝ := fun x ↦ x^3

/-- The point P through which the tangent line passes -/
def P : ℝ × ℝ := (1, 1)

/-- Predicate to check if a line passes through the fourth quadrant -/
def passes_through_fourth_quadrant (a b c : ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y < 0 ∧ a * x + b * y + c = 0

/-- The tangent line to curve C at point (x₀, C x₀) -/
def tangent_line (x₀ : ℝ) : ℝ → ℝ := fun x ↦ C x₀ + (3 * x₀^2) * (x - x₀)

theorem tangent_line_equation :
  ∃ x₀ : ℝ, 
    tangent_line x₀ P.1 = P.2 ∧ 
    ¬passes_through_fourth_quadrant 3 (-4) 1 ∧
    ∀ x, tangent_line x₀ x = 3*x - 4*(C x) + 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2312_231223


namespace NUMINAMATH_CALUDE_cost_of_ingredients_for_two_cakes_l2312_231244

/-- The cost of ingredients for two cakes given selling price, profit, and packaging cost -/
theorem cost_of_ingredients_for_two_cakes 
  (selling_price : ℝ) 
  (profit_per_cake : ℝ) 
  (packaging_cost : ℝ) : 
  selling_price = 15 → 
  profit_per_cake = 8 → 
  packaging_cost = 1 → 
  2 * selling_price - 2 * profit_per_cake - 2 * packaging_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_ingredients_for_two_cakes_l2312_231244


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2312_231240

theorem simplify_sqrt_expression :
  Real.sqrt 12 + 3 * Real.sqrt (1/3) = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2312_231240


namespace NUMINAMATH_CALUDE_evaluate_expression_l2312_231295

theorem evaluate_expression (y : ℚ) (h : y = -3) :
  (5 + y * (2 + y) - 4^2) / (y - 4 + y^2 - y) = -8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2312_231295


namespace NUMINAMATH_CALUDE_equal_sum_number_properties_l2312_231222

def is_equal_sum_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 + (n / 10) % 10 = (n / 100) % 10 + n % 10)

def transform (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  d * 1000 + c * 100 + b * 10 + a

def F (n : ℕ) : ℚ := (n + transform n : ℚ) / 101

def G (n : ℕ) : ℚ := (n - transform n : ℚ) / 99

theorem equal_sum_number_properties :
  (∀ n : ℕ, is_equal_sum_number n → 
    (F n - G n = 72 → n = 5236)) ∧
  (∃ n : ℕ, is_equal_sum_number n ∧ 
    (F n / 13).isInt ∧ (G n / 7).isInt ∧
    (∀ m : ℕ, is_equal_sum_number m ∧ 
      (F m / 13).isInt ∧ (G m / 7).isInt → m ≤ n) ∧
    n = 9647) := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_number_properties_l2312_231222


namespace NUMINAMATH_CALUDE_museum_visitors_theorem_l2312_231247

/-- Represents the inverse proportional relationship between visitors and ticket price -/
def inverse_proportion (v t k : ℝ) : Prop := v * t = k

/-- Given conditions of the problem -/
def museum_conditions (v₁ v₂ t₁ t₂ k : ℝ) : Prop :=
  t₁ = 20 ∧ v₁ = 150 ∧ t₂ = 30 ∧
  inverse_proportion v₁ t₁ k ∧
  inverse_proportion v₂ t₂ k

/-- Theorem statement -/
theorem museum_visitors_theorem (v₁ v₂ t₁ t₂ k : ℝ) :
  museum_conditions v₁ v₂ t₁ t₂ k → v₂ = 100 := by
  sorry

end NUMINAMATH_CALUDE_museum_visitors_theorem_l2312_231247


namespace NUMINAMATH_CALUDE_unique_one_third_area_point_l2312_231296

/-- Triangle in a plane --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- The circumcenter of a triangle --/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- The area of a triangle --/
def area (t : Triangle) : ℝ := sorry

/-- Theorem: The only point P for which the area of triangle PAB is one-third 
    of the area of triangle ABC is the circumcenter of ABC --/
theorem unique_one_third_area_point (t : Triangle) :
  ∀ P : ℝ × ℝ, area (Triangle.mk P t.A t.B) = (1/3) * area t → 
  P = circumcenter t := by
  sorry

end NUMINAMATH_CALUDE_unique_one_third_area_point_l2312_231296


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_segment_l2312_231264

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Calculates the distance between two points -/
def distance (A B : Point) : ℝ :=
  sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ :=
  sorry

/-- Main theorem -/
theorem quadrilateral_diagonal_segment (PQRS : Quadrilateral) (T : Point) : 
  (distance PQRS.P PQRS.Q = 15) →
  (distance PQRS.R PQRS.S = 20) →
  (distance PQRS.P PQRS.R = 22) →
  (triangleArea PQRS.P T PQRS.R = triangleArea PQRS.Q T PQRS.S) →
  (distance PQRS.Q T = 66 / 7) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_segment_l2312_231264


namespace NUMINAMATH_CALUDE_max_triangles_two_lines_l2312_231274

def points_on_line_a : ℕ := 5
def points_on_line_b : ℕ := 8

def triangles_type1 : ℕ := Nat.choose points_on_line_a 2 * Nat.choose points_on_line_b 1
def triangles_type2 : ℕ := Nat.choose points_on_line_a 1 * Nat.choose points_on_line_b 2

def total_triangles : ℕ := triangles_type1 + triangles_type2

theorem max_triangles_two_lines : total_triangles = 220 := by
  sorry

end NUMINAMATH_CALUDE_max_triangles_two_lines_l2312_231274


namespace NUMINAMATH_CALUDE_total_legs_of_three_spiders_l2312_231263

def human_legs : ℕ := 2

def spider1_legs : ℕ := 2 * (2 * human_legs)

def spider2_legs : ℕ := 3 * spider1_legs

def spider3_legs : ℕ := spider2_legs - 5

def total_spider_legs : ℕ := spider1_legs + spider2_legs + spider3_legs

theorem total_legs_of_three_spiders :
  total_spider_legs = 51 := by sorry

end NUMINAMATH_CALUDE_total_legs_of_three_spiders_l2312_231263


namespace NUMINAMATH_CALUDE_a_4_value_l2312_231226

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem a_4_value (a : ℕ → ℝ) :
  a 1 = 2 →
  geometric_sequence (λ n => 1 + a n) 3 →
  a 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_a_4_value_l2312_231226


namespace NUMINAMATH_CALUDE_x_calculation_l2312_231292

theorem x_calculation (m n p q x : ℝ) :
  x^2 + (2*m*p + 2*n*q)^2 + (2*m*q - 2*n*p)^2 = (m^2 + n^2 + p^2 + q^2)^2 →
  x = m^2 + n^2 - p^2 - q^2 ∨ x = -(m^2 + n^2 - p^2 - q^2) := by
sorry

end NUMINAMATH_CALUDE_x_calculation_l2312_231292


namespace NUMINAMATH_CALUDE_problem_statement_l2312_231294

theorem problem_statement (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (heq1 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2012)
  (heq2 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2012) :
  (a*b)^2012 - (c*d)^2012 = -2012 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2312_231294


namespace NUMINAMATH_CALUDE_clock_angle_at_3_15_l2312_231218

/-- The angle in degrees that the hour hand moves per minute -/
def hour_hand_speed : ℝ := 0.5

/-- The angle in degrees that the minute hand moves per minute -/
def minute_hand_speed : ℝ := 6

/-- The starting position of the hour hand at 3:00 in degrees -/
def hour_hand_start : ℝ := 90

/-- The number of minutes past 3:00 -/
def minutes_past : ℝ := 15

/-- Calculates the position of the hour hand at 3:15 -/
def hour_hand_position : ℝ := hour_hand_start + hour_hand_speed * minutes_past

/-- Calculates the position of the minute hand at 3:15 -/
def minute_hand_position : ℝ := minute_hand_speed * minutes_past

/-- The acute angle between the hour hand and minute hand at 3:15 -/
def clock_angle : ℝ := |hour_hand_position - minute_hand_position|

theorem clock_angle_at_3_15 : clock_angle = 7.5 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_15_l2312_231218


namespace NUMINAMATH_CALUDE_expression_value_l2312_231209

theorem expression_value (x y : ℝ) (h : x - 2*y + 3 = 0) : 
  (2*y - x)^2 - 2*x + 4*y - 1 = 14 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2312_231209


namespace NUMINAMATH_CALUDE_spacefarer_resources_sum_l2312_231219

def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem spacefarer_resources_sum :
  let crystal := base3_to_base10 [0, 2, 1, 2]
  let rare_metals := base3_to_base10 [2, 0, 1, 2]
  let alien_tech := base3_to_base10 [2, 0, 1]
  crystal + rare_metals + alien_tech = 145 := by
sorry

end NUMINAMATH_CALUDE_spacefarer_resources_sum_l2312_231219


namespace NUMINAMATH_CALUDE_abs_x_eq_3_system_solution_2_system_solution_3_l2312_231229

-- Part 1
theorem abs_x_eq_3 (x : ℝ) : |x| = 3 ↔ x = 3 ∨ x = -3 := by sorry

-- Part 2
theorem system_solution_2 (x y : ℝ) : 
  y * (x - 1) = 0 ∧ 2 * x + 5 * y = 7 ↔ 
  (x = 7/2 ∧ y = 0) ∨ (x = 1 ∧ y = 1) := by sorry

-- Part 3
theorem system_solution_3 (x y : ℝ) :
  x * y - 2 * x - y + 2 = 0 ∧ x + 6 * y = 3 ∧ 3 * x + y = 8 ↔
  (x = 1 ∧ y = 5) ∨ (x = 2 ∧ y = 2) := by sorry

end NUMINAMATH_CALUDE_abs_x_eq_3_system_solution_2_system_solution_3_l2312_231229


namespace NUMINAMATH_CALUDE_simplify_expression_l2312_231243

theorem simplify_expression (a b c : ℝ) (ha : a = 37/5) (hb : b = 5/37) :
  1.6 * (((1/a + 1/b - 2*c/(a*b)) * (a + b + 2*c)) / (1/a^2 + 1/b^2 + 2/(a*b) - 4*c^2/(a^2*b^2))) = 1.6 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2312_231243


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2312_231268

theorem arithmetic_sequence_length :
  ∀ (a₁ aₙ d n : ℤ),
    a₁ = -3 →
    aₙ = 45 →
    d = 4 →
    aₙ = a₁ + (n - 1) * d →
    n = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2312_231268


namespace NUMINAMATH_CALUDE_binomial_150_150_l2312_231285

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l2312_231285


namespace NUMINAMATH_CALUDE_x_power_six_plus_reciprocal_l2312_231237

theorem x_power_six_plus_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^6 + 1/x^6 = 322 := by
  sorry

end NUMINAMATH_CALUDE_x_power_six_plus_reciprocal_l2312_231237


namespace NUMINAMATH_CALUDE_brennan_pepper_usage_l2312_231297

/-- The amount of pepper Brennan used to make scrambled eggs -/
def pepper_used (initial : ℝ) (remaining : ℝ) : ℝ :=
  initial - remaining

/-- Theorem stating that Brennan used 0.16 grams of pepper -/
theorem brennan_pepper_usage :
  pepper_used 0.25 0.09 = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_brennan_pepper_usage_l2312_231297


namespace NUMINAMATH_CALUDE_last_two_digits_of_2007_power_20077_l2312_231278

theorem last_two_digits_of_2007_power_20077 : 2007^20077 % 100 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_2007_power_20077_l2312_231278


namespace NUMINAMATH_CALUDE_johns_average_speed_l2312_231224

/-- John's cycling and walking trip -/
def johns_trip (uphill_distance : ℝ) (uphill_time : ℝ) (downhill_time : ℝ) (walk_distance : ℝ) (walk_time : ℝ) : Prop :=
  let total_distance := 2 * uphill_distance + walk_distance
  let total_time := uphill_time + downhill_time + walk_time
  (total_distance / (total_time / 60)) = 6

theorem johns_average_speed :
  johns_trip 3 45 15 2 20 := by
  sorry

end NUMINAMATH_CALUDE_johns_average_speed_l2312_231224


namespace NUMINAMATH_CALUDE_area_increase_when_perimeter_increased_l2312_231272

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the set of possible area increases. -/
def possibleAreaIncreases : Set ℕ := {2, 4, 21, 36, 38}

/-- Theorem stating the possible area increases when the perimeter is increased by 4 cm. -/
theorem area_increase_when_perimeter_increased
  (r : Rectangle)
  (h_perimeter : perimeter r = 40)
  (h_area : area r ≤ 40)
  (r_new : Rectangle)
  (h_new_perimeter : perimeter r_new = 44)
  : (area r_new - area r) ∈ possibleAreaIncreases := by
  sorry

end NUMINAMATH_CALUDE_area_increase_when_perimeter_increased_l2312_231272


namespace NUMINAMATH_CALUDE_same_solution_implies_a_equals_one_l2312_231203

theorem same_solution_implies_a_equals_one :
  (∃ x : ℝ, 2 - a - x = 0 ∧ 2*x + 1 = 3) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_equals_one_l2312_231203


namespace NUMINAMATH_CALUDE_union_M_N_equals_geq_one_l2312_231235

-- Define set M
def M : Set ℝ := {x | x - 2 > 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

-- Theorem statement
theorem union_M_N_equals_geq_one : M ∪ N = {x | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_geq_one_l2312_231235


namespace NUMINAMATH_CALUDE_gathering_handshakes_l2312_231239

def num_dwarves : ℕ := 25
def num_elves : ℕ := 18

def handshakes_among_dwarves (n : ℕ) : ℕ := n * (n - 1) / 2
def handshakes_between_dwarves_and_elves (d e : ℕ) : ℕ := d * e

def total_handshakes (d e : ℕ) : ℕ :=
  handshakes_among_dwarves d + handshakes_between_dwarves_and_elves d e

theorem gathering_handshakes :
  total_handshakes num_dwarves num_elves = 750 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l2312_231239


namespace NUMINAMATH_CALUDE_number_of_digits_c_l2312_231248

theorem number_of_digits_c (a b c : ℕ) : 
  a < b → b < c → 
  (b + a) % (b - a) = 0 → 
  (c + b) % (c - b) = 0 → 
  a ≥ 10^2010 → a < 10^2011 →
  b ≥ 10^2011 → b < 10^2012 →
  c ≥ 10^4 ∧ c < 10^5 := by sorry

end NUMINAMATH_CALUDE_number_of_digits_c_l2312_231248


namespace NUMINAMATH_CALUDE_largest_divisible_by_six_under_9000_l2312_231242

theorem largest_divisible_by_six_under_9000 : 
  ∀ n : ℕ, n < 9000 ∧ 6 ∣ n → n ≤ 8994 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_by_six_under_9000_l2312_231242


namespace NUMINAMATH_CALUDE_quadratic_equations_roots_l2312_231228

theorem quadratic_equations_roots :
  (∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 3 ∧ ∀ x : ℝ, x^2 - 3*x = 0 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 5/4 ∧ x₂ = -1 ∧ ∀ x : ℝ, 4*x^2 - x - 5 = 0 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -2/3 ∧ ∀ x : ℝ, 3*x*(x-1) = 2-2*x ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_roots_l2312_231228


namespace NUMINAMATH_CALUDE_expansion_of_5000_power_150_l2312_231234

theorem expansion_of_5000_power_150 :
  ∃ (n : ℕ), 5000^150 = n * 10^450 ∧ 1 ≤ n ∧ n < 10 :=
by
  sorry

end NUMINAMATH_CALUDE_expansion_of_5000_power_150_l2312_231234


namespace NUMINAMATH_CALUDE_problem_solution_l2312_231210

theorem problem_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2312_231210


namespace NUMINAMATH_CALUDE_keith_attended_games_l2312_231254

theorem keith_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 20)
  (h2 : missed_games = 9) :
  total_games - missed_games = 11 := by
sorry

end NUMINAMATH_CALUDE_keith_attended_games_l2312_231254


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2312_231245

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let roots := {x : ℝ | f x = 0}
  (∃ x y : ℝ, x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∀ z : ℝ, z ∈ roots → z = x ∨ z = y) →
  x + y = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 - 15 * x + 20
  let roots := {x : ℝ | f x = 0}
  ∃ C D : ℝ, C ∈ roots ∧ D ∈ roots ∧ C ≠ D ∧
    (∀ z : ℝ, z ∈ roots → z = C ∨ z = D) ∧
    C + D = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2312_231245


namespace NUMINAMATH_CALUDE_abs_value_of_complex_l2312_231253

theorem abs_value_of_complex (z : ℂ) : z = (1 + 2 * Complex.I) / Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_complex_l2312_231253


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2312_231267

/-- The range of k for an ellipse with equation x²/(3-k) + y²/(5+k) = 1 and foci on the y-axis -/
theorem ellipse_k_range :
  ∀ k : ℝ,
  (∀ x y : ℝ, x^2 / (3 - k) + y^2 / (5 + k) = 1) →
  (5 + k > 3 - k) →
  (3 - k > 0) →
  (5 + k > 0) →
  -1 < k ∧ k < 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2312_231267


namespace NUMINAMATH_CALUDE_furniture_reimbursement_l2312_231204

/-- Calculates the reimbursement amount for an overcharged furniture purchase -/
theorem furniture_reimbursement
  (num_pieces : ℕ)
  (amount_paid : ℚ)
  (cost_per_piece : ℚ)
  (h1 : num_pieces = 150)
  (h2 : amount_paid = 20700)
  (h3 : cost_per_piece = 134) :
  amount_paid - (num_pieces : ℚ) * cost_per_piece = 600 := by
  sorry

end NUMINAMATH_CALUDE_furniture_reimbursement_l2312_231204


namespace NUMINAMATH_CALUDE_derivative_of_y_l2312_231277

noncomputable def y (x : ℝ) : ℝ := x / (1 - Real.cos x)

theorem derivative_of_y (x : ℝ) (h : x ≠ 0) :
  deriv y x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l2312_231277


namespace NUMINAMATH_CALUDE_apples_per_basket_l2312_231252

theorem apples_per_basket (total_apples : ℕ) (num_baskets : ℕ) 
  (h1 : total_apples = 495) 
  (h2 : num_baskets = 19) 
  (h3 : total_apples % num_baskets = 0) : 
  total_apples / num_baskets = 26 := by
sorry

end NUMINAMATH_CALUDE_apples_per_basket_l2312_231252


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_squared_l2312_231221

theorem sum_and_reciprocal_squared (x N : ℝ) (h1 : x ≠ 0) (h2 : x + 1/x = N) (h3 : x^2 + 1/x^2 = 2) : N = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_squared_l2312_231221


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l2312_231213

theorem intersection_point_is_solution (k : ℝ) (hk : k ≠ 0) :
  (∃ (x y : ℝ), y = 2 * x - 1 ∧ y = k * x ∧ x = 1 ∧ y = 1) →
  (∃! (x y : ℝ), 2 * x - y = 1 ∧ k * x - y = 0 ∧ x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l2312_231213


namespace NUMINAMATH_CALUDE_greatest_common_divisor_with_same_remainder_l2312_231201

theorem greatest_common_divisor_with_same_remainder (a b c : ℕ) (ha : a = 54) (hb : b = 87) (hc : c = 172) :
  ∃ (d : ℕ), d > 0 ∧ 
  (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
  (∀ (k : ℕ), k > d → ¬(∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s)) →
  d = 1 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_with_same_remainder_l2312_231201


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2312_231282

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -6) :
  x^3 + 1/x^3 = -198 := by sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2312_231282


namespace NUMINAMATH_CALUDE_fundraising_total_donation_l2312_231255

def total_donation (days : ℕ) (initial_donors : ℕ) (initial_donation : ℕ) : ℕ :=
  let rec donation_sum (d : ℕ) (donors : ℕ) (avg_donation : ℕ) (acc : ℕ) : ℕ :=
    if d = 0 then acc
    else donation_sum (d - 1) (donors * 2) (avg_donation + 5) (acc + donors * avg_donation)
  donation_sum days initial_donors initial_donation 0

theorem fundraising_total_donation :
  total_donation 5 10 10 = 8000 :=
by sorry

end NUMINAMATH_CALUDE_fundraising_total_donation_l2312_231255


namespace NUMINAMATH_CALUDE_mixed_doubles_selection_methods_l2312_231293

def male_athletes : ℕ := 5
def female_athletes : ℕ := 6
def selected_male : ℕ := 2
def selected_female : ℕ := 2

theorem mixed_doubles_selection_methods :
  (Nat.choose male_athletes selected_male) *
  (Nat.choose female_athletes selected_female) *
  (Nat.factorial selected_male) = 300 := by
sorry

end NUMINAMATH_CALUDE_mixed_doubles_selection_methods_l2312_231293


namespace NUMINAMATH_CALUDE_erased_number_theorem_l2312_231279

theorem erased_number_theorem :
  ∀ x : ℕ, x ∈ Finset.range 21 \ {0} →
  (∃ y ∈ Finset.range 21 \ {0, x},
    19 * y = (Finset.sum (Finset.range 21 \ {0, x}) id)) ↔
  x = 1 ∨ x = 20 :=
by sorry

end NUMINAMATH_CALUDE_erased_number_theorem_l2312_231279


namespace NUMINAMATH_CALUDE_phone_plan_monthly_fee_l2312_231249

theorem phone_plan_monthly_fee :
  let first_plan_per_minute : ℚ := 13/100
  let second_plan_monthly_fee : ℚ := 8
  let second_plan_per_minute : ℚ := 18/100
  let equal_minutes : ℕ := 280
  ∃ (F : ℚ),
    F + first_plan_per_minute * equal_minutes = 
    second_plan_monthly_fee + second_plan_per_minute * equal_minutes ∧
    F = 22 := by
  sorry

end NUMINAMATH_CALUDE_phone_plan_monthly_fee_l2312_231249


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2312_231258

/-- Calculate the interest rate given the principal, time, and total interest for a simple interest loan. -/
theorem simple_interest_rate_calculation
  (principal : ℝ)
  (time : ℝ)
  (total_interest : ℝ)
  (h_principal : principal = 5000)
  (h_time : time = 10)
  (h_total_interest : total_interest = 2000) :
  (total_interest * 100) / (principal * time) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2312_231258


namespace NUMINAMATH_CALUDE_second_class_size_l2312_231215

theorem second_class_size 
  (first_class_size : ℕ) 
  (first_class_avg : ℚ) 
  (second_class_avg : ℚ) 
  (total_avg : ℚ) 
  (h1 : first_class_size = 35)
  (h2 : first_class_avg = 40)
  (h3 : second_class_avg = 60)
  (h4 : total_avg = 51.25) :
  ∃ second_class_size : ℕ,
    (first_class_size * first_class_avg + second_class_size * second_class_avg) / 
    (first_class_size + second_class_size) = total_avg ∧
    second_class_size = 45 := by
  sorry


end NUMINAMATH_CALUDE_second_class_size_l2312_231215


namespace NUMINAMATH_CALUDE_original_speed_B_l2312_231207

/-- Two people traveling towards each other -/
structure TravelScenario where
  speed_A : ℝ
  speed_B : ℝ

/-- The condition that the meeting point remains the same after speed changes -/
def meeting_point_unchanged (s : TravelScenario) : Prop :=
  s.speed_A / s.speed_B = (5/4 * s.speed_A) / (s.speed_B + 10)

/-- The theorem stating that if the meeting point is unchanged, B's original speed is 40 km/h -/
theorem original_speed_B (s : TravelScenario) :
  meeting_point_unchanged s → s.speed_B = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_speed_B_l2312_231207


namespace NUMINAMATH_CALUDE_locus_of_equilateral_triangle_vertex_l2312_231286

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a rotation function
def rotate (p : ℝ × ℝ) (center : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  sorry

-- Define a function to check if a point is on a circle
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  sorry

-- Define a function to check if a triangle is equilateral
def isEquilateral (a b c : ℝ × ℝ) : Prop :=
  sorry

-- Theorem statement
theorem locus_of_equilateral_triangle_vertex (C : Circle) (P : ℝ × ℝ) :
  let locusM := {M : ℝ × ℝ | ∃ K, onCircle K C ∧ isEquilateral P K M}
  let rotated_circle_1 := {p : ℝ × ℝ | ∃ q, onCircle q C ∧ p = rotate q P (π/3)}
  let rotated_circle_2 := {p : ℝ × ℝ | ∃ q, onCircle q C ∧ p = rotate q P (-π/3)}
  if P = C.center then
    locusM = {p : ℝ × ℝ | onCircle p C}
  else
    locusM = rotated_circle_1 ∪ rotated_circle_2 :=
by sorry


end NUMINAMATH_CALUDE_locus_of_equilateral_triangle_vertex_l2312_231286


namespace NUMINAMATH_CALUDE_action_figure_shelves_l2312_231287

/-- Given a room with action figures and shelves, calculate the number of shelves. -/
theorem action_figure_shelves 
  (total_figures : ℕ) 
  (figures_per_shelf : ℕ) 
  (h1 : total_figures = 120) 
  (h2 : figures_per_shelf = 15) 
  (h3 : figures_per_shelf > 0) : 
  total_figures / figures_per_shelf = 8 := by
  sorry

end NUMINAMATH_CALUDE_action_figure_shelves_l2312_231287


namespace NUMINAMATH_CALUDE_max_value_expression_l2312_231230

theorem max_value_expression (a b c d : ℝ) 
  (ha : a ∈ Set.Icc (-9.5) 9.5)
  (hb : b ∈ Set.Icc (-9.5) 9.5)
  (hc : c ∈ Set.Icc (-9.5) 9.5)
  (hd : d ∈ Set.Icc (-9.5) 9.5) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 380 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2312_231230


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2312_231246

theorem cubic_polynomials_common_roots (c d : ℝ) : 
  (∃ u v : ℝ, u ≠ v ∧ 
    u^3 + c*u^2 + 10*u + 4 = 0 ∧ 
    u^3 + d*u^2 + 13*u + 5 = 0 ∧
    v^3 + c*v^2 + 10*v + 4 = 0 ∧ 
    v^3 + d*v^2 + 13*v + 5 = 0) → 
  c = 7 ∧ d = 8 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2312_231246


namespace NUMINAMATH_CALUDE_power_mod_eleven_l2312_231262

theorem power_mod_eleven : 5^2023 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l2312_231262


namespace NUMINAMATH_CALUDE_relationship_abc_l2312_231238

theorem relationship_abc : ∀ (a b c : ℝ), 
  a = 1/3 → b = Real.sin (1/3) → c = 1/Real.pi → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2312_231238


namespace NUMINAMATH_CALUDE_complex_simplification_l2312_231269

theorem complex_simplification :
  ((-5 - 3*Complex.I) - (2 - 7*Complex.I)) * 2 = -14 + 8*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2312_231269


namespace NUMINAMATH_CALUDE_girls_tryout_count_l2312_231225

theorem girls_tryout_count (boys : ℕ) (called_back : ℕ) (didnt_make_cut : ℕ) :
  boys = 4 →
  called_back = 26 →
  didnt_make_cut = 17 →
  ∃ girls : ℕ, girls + boys = called_back + didnt_make_cut ∧ girls = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_tryout_count_l2312_231225


namespace NUMINAMATH_CALUDE_pizza_calculation_l2312_231251

/-- The total number of pizzas made by Heather and Craig in two days -/
def total_pizzas (craig_day1 : ℕ) (heather_multiplier : ℕ) (craig_increase : ℕ) (heather_decrease : ℕ) : ℕ :=
  let craig_day2 := craig_day1 + craig_increase
  let heather_day1 := craig_day1 * heather_multiplier
  let heather_day2 := craig_day2 - heather_decrease
  craig_day1 + heather_day1 + craig_day2 + heather_day2

/-- Theorem stating the total number of pizzas made by Heather and Craig in two days -/
theorem pizza_calculation : total_pizzas 40 4 60 20 = 380 := by
  sorry

end NUMINAMATH_CALUDE_pizza_calculation_l2312_231251


namespace NUMINAMATH_CALUDE_masons_grandmother_age_l2312_231271

theorem masons_grandmother_age (mason_age sydney_age father_age grandmother_age : ℕ) :
  mason_age = 20 →
  sydney_age = 3 * mason_age →
  father_age = sydney_age + 6 →
  grandmother_age = 2 * father_age →
  grandmother_age = 132 := by
sorry

end NUMINAMATH_CALUDE_masons_grandmother_age_l2312_231271


namespace NUMINAMATH_CALUDE_infantry_column_problem_l2312_231231

/-- Given an infantry column of length 1 km, moving at speed x km/h,
    and Sergeant Kim moving at speed 3x km/h, if the infantry column
    covers 2.4 km while Kim travels to the front of the column and back,
    then Kim's total distance traveled is 3.6 km. -/
theorem infantry_column_problem (x : ℝ) (h : x > 0) :
  let column_length : ℝ := 1
  let column_speed : ℝ := x
  let kim_speed : ℝ := 3 * x
  let column_distance : ℝ := 2.4
  let time := column_distance / column_speed
  let kim_distance := kim_speed * time
  kim_distance = 3.6 := by sorry

end NUMINAMATH_CALUDE_infantry_column_problem_l2312_231231


namespace NUMINAMATH_CALUDE_painter_paintings_l2312_231206

/-- Given a painter who makes a certain number of paintings per day and already has some paintings,
    calculate the total number of paintings after a given number of days. -/
def total_paintings (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  paintings_per_day * days + initial_paintings

/-- Theorem: A painter who makes 2 paintings per day and already has 20 paintings
    will have 80 paintings in total after 30 days. -/
theorem painter_paintings : total_paintings 2 20 30 = 80 := by
  sorry

end NUMINAMATH_CALUDE_painter_paintings_l2312_231206


namespace NUMINAMATH_CALUDE_distinct_triangles_in_tetrahedron_l2312_231283

/-- The number of vertices in a regular tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- Calculates the number of combinations of k items chosen from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Theorem: The number of distinct triangles in a regular tetrahedron is 4 -/
theorem distinct_triangles_in_tetrahedron :
  choose tetrahedron_vertices triangle_vertices = 4 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_tetrahedron_l2312_231283


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2312_231276

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  A = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2312_231276


namespace NUMINAMATH_CALUDE_pentagon_triangles_l2312_231205

/-- The number of triangles formed in a pentagon when drawing diagonals from one vertex --/
def triangles_in_pentagon : ℕ := 3

/-- The number of vertices in a pentagon --/
def pentagon_vertices : ℕ := 5

/-- The number of diagonals drawn from one vertex in a pentagon --/
def diagonals_from_vertex : ℕ := 2

theorem pentagon_triangles :
  triangles_in_pentagon = diagonals_from_vertex + 1 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_triangles_l2312_231205


namespace NUMINAMATH_CALUDE_order_of_expressions_l2312_231299

theorem order_of_expressions :
  let a : ℝ := 3^(3/2)
  let b : ℝ := 3^(5/2)
  let c : ℝ := Real.log 3 / Real.log 0.5
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l2312_231299


namespace NUMINAMATH_CALUDE_chairlift_halfway_l2312_231227

def total_chairs : ℕ := 96
def current_chair : ℕ := 66

def halfway_chair (total : ℕ) (current : ℕ) : ℕ :=
  (current - total / 2 + total) % total

theorem chairlift_halfway :
  halfway_chair total_chairs current_chair = 18 := by
sorry

end NUMINAMATH_CALUDE_chairlift_halfway_l2312_231227


namespace NUMINAMATH_CALUDE_quadratic_symmetric_derivative_l2312_231280

-- Define a quadratic function symmetric about x = 1
def f (a b : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + b

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2 * a * (x - 1)

theorem quadratic_symmetric_derivative (a b : ℝ) :
  (f' a 0 = -2) → (f' a 2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetric_derivative_l2312_231280


namespace NUMINAMATH_CALUDE_solution_set_range_l2312_231266

-- Define the function f(x) for a given a
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 1) * x^2 - (a - 1) * x - 1

-- Define the property that f(x) < 0 for all real x
def always_negative (a : ℝ) : Prop := ∀ x : ℝ, f a x < 0

-- Define the set of a for which f(x) < 0 for all real x
def solution_set : Set ℝ := {a : ℝ | always_negative a}

-- State the theorem
theorem solution_set_range : solution_set = Set.Ioc (-3/5) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_range_l2312_231266


namespace NUMINAMATH_CALUDE_books_per_day_l2312_231241

def total_books : ℕ := 15
def total_days : ℕ := 3

theorem books_per_day : (total_books / total_days : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_books_per_day_l2312_231241


namespace NUMINAMATH_CALUDE_largest_prime_factor_57_largest_prime_factor_57_is_19_l2312_231256

def numbers : List Nat := [57, 75, 91, 143, 169]

def largest_prime_factor (n : Nat) : Nat :=
  sorry

theorem largest_prime_factor_57 :
  ∀ n ∈ numbers, n ≠ 57 → largest_prime_factor n < largest_prime_factor 57 :=
  sorry

theorem largest_prime_factor_57_is_19 :
  largest_prime_factor 57 = 19 :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_57_largest_prime_factor_57_is_19_l2312_231256


namespace NUMINAMATH_CALUDE_rectangle_area_l2312_231220

/-- Given a rectangle with perimeter 28 cm and width 6 cm, its area is 48 square centimeters. -/
theorem rectangle_area (perimeter width : ℝ) (h_perimeter : perimeter = 28) (h_width : width = 6) :
  let length := (perimeter - 2 * width) / 2
  width * length = 48 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2312_231220


namespace NUMINAMATH_CALUDE_cookies_eaten_total_l2312_231270

theorem cookies_eaten_total (charlie_cookies father_cookies mother_cookies : ℕ) 
  (h1 : charlie_cookies = 15)
  (h2 : father_cookies = 10)
  (h3 : mother_cookies = 5) :
  charlie_cookies + father_cookies + mother_cookies = 30 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_total_l2312_231270


namespace NUMINAMATH_CALUDE_tan_beta_value_l2312_231261

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3)
  (h2 : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
  Real.tan β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l2312_231261


namespace NUMINAMATH_CALUDE_train_length_l2312_231291

/-- Given a train traveling at 72 km/hr that crosses a pole in 9 seconds, prove its length is 180 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 72 → time = 9 → speed * time * (1000 / 3600) = 180 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2312_231291


namespace NUMINAMATH_CALUDE_line_slope_problem_l2312_231232

/-- Given a line passing through points (-1, -4) and (3, k) with slope k, prove k = 4/3 -/
theorem line_slope_problem (k : ℚ) : 
  (k - (-4)) / (3 - (-1)) = k → k = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l2312_231232


namespace NUMINAMATH_CALUDE_weight_of_new_person_new_person_weight_l2312_231217

theorem weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  let new_weight := replaced_weight + initial_count * average_increase
  new_weight

/-- Given a group of 8 people where one person weighing 65 kg is replaced by a new person, 
    and the average weight of the group increases by 3 kg, 
    the weight of the new person is 89 kg. -/
theorem new_person_weight : weight_of_new_person 8 3 65 = 89 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_new_person_weight_l2312_231217


namespace NUMINAMATH_CALUDE_ralph_squares_count_l2312_231214

/-- The number of matchsticks in a box -/
def total_matchsticks : ℕ := 50

/-- The number of matchsticks Elvis uses for one square -/
def elvis_square_size : ℕ := 4

/-- The number of matchsticks Ralph uses for one square -/
def ralph_square_size : ℕ := 8

/-- The number of squares Elvis makes -/
def elvis_squares : ℕ := 5

/-- The number of matchsticks left in the box -/
def remaining_matchsticks : ℕ := 6

/-- The number of squares Ralph makes -/
def ralph_squares : ℕ := 3

theorem ralph_squares_count : 
  elvis_square_size * elvis_squares + ralph_square_size * ralph_squares + remaining_matchsticks = total_matchsticks :=
by sorry

end NUMINAMATH_CALUDE_ralph_squares_count_l2312_231214


namespace NUMINAMATH_CALUDE_grid_polygon_segment_sums_equal_l2312_231260

-- Define a type for grid points
structure GridPoint where
  x : ℤ
  y : ℤ

-- Define a type for polygons on a grid
structure GridPolygon where
  vertices : List GridPoint
  convex : Bool
  verticesOnGrid : Bool
  sidesNotAligned : Bool

-- Define a function to calculate the sum of vertical segment lengths
def sumVerticalSegments (p : GridPolygon) : ℝ :=
  sorry

-- Define a function to calculate the sum of horizontal segment lengths
def sumHorizontalSegments (p : GridPolygon) : ℝ :=
  sorry

-- Theorem statement
theorem grid_polygon_segment_sums_equal (p : GridPolygon) :
  p.convex ∧ p.verticesOnGrid ∧ p.sidesNotAligned →
  sumVerticalSegments p = sumHorizontalSegments p :=
sorry

end NUMINAMATH_CALUDE_grid_polygon_segment_sums_equal_l2312_231260
