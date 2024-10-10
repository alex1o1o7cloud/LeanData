import Mathlib

namespace dvd_rental_cost_l1902_190265

theorem dvd_rental_cost (num_dvds : ℕ) (cost_per_dvd : ℚ) : 
  num_dvds = 4 → cost_per_dvd = 6/5 → num_dvds * cost_per_dvd = 24/5 := by
  sorry

end dvd_rental_cost_l1902_190265


namespace investment_interest_rate_l1902_190204

/-- Given an investment scenario, prove the interest rate for the second investment --/
theorem investment_interest_rate 
  (total_investment : ℝ) 
  (desired_interest : ℝ) 
  (first_investment : ℝ) 
  (first_rate : ℝ) 
  (h1 : total_investment = 10000)
  (h2 : desired_interest = 980)
  (h3 : first_investment = 6000)
  (h4 : first_rate = 0.09)
  : 
  let second_investment := total_investment - first_investment
  let first_interest := first_investment * first_rate
  let second_interest := desired_interest - first_interest
  let second_rate := second_interest / second_investment
  second_rate = 0.11 := by
sorry

end investment_interest_rate_l1902_190204


namespace sum_equals_four_l1902_190246

theorem sum_equals_four (x y : ℝ) (h : |x - 3| + |y + 2| = 0) : x + y + 3 = 4 := by
  sorry

end sum_equals_four_l1902_190246


namespace marble_difference_l1902_190216

/-- The number of marbles each person has -/
structure Marbles where
  laurie : ℕ
  kurt : ℕ
  dennis : ℕ

/-- Given conditions about the marbles -/
def marble_conditions (m : Marbles) : Prop :=
  m.laurie = 37 ∧ m.laurie = m.kurt + 12 ∧ m.dennis = 70

/-- Theorem stating the difference between Dennis's and Kurt's marbles -/
theorem marble_difference (m : Marbles) (h : marble_conditions m) :
  m.dennis - m.kurt = 45 := by
  sorry

end marble_difference_l1902_190216


namespace iron_wire_length_l1902_190261

/-- The length of each cut-off part of the wire in centimeters. -/
def cut_length : ℝ := 10

/-- The original length of the iron wire in centimeters. -/
def original_length : ℝ := 110

/-- The length of the remaining part of the wire after cutting both ends. -/
def remaining_length : ℝ := original_length - 2 * cut_length

/-- Theorem stating that the original length of the iron wire is 110 cm. -/
theorem iron_wire_length :
  (remaining_length = 4 * (2 * cut_length) + 10) →
  original_length = 110 :=
by sorry

end iron_wire_length_l1902_190261


namespace sequence_sum_proof_l1902_190260

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℚ := (n + 1) / 2

-- Define the geometric sequence {b_n}
def b (n : ℕ) : ℚ := 2^(n-1)

-- Define the sum of the first n terms of {b_n}
def T (n : ℕ) : ℚ := 2^n - 1

theorem sequence_sum_proof :
  -- Given conditions
  (a 3 = 2) ∧
  ((a 1 + a 2 + a 3) = 9/2) ∧
  (b 1 = a 1) ∧
  (b 4 = a 15) →
  -- Conclusion
  ∀ n : ℕ, T n = 2^n - 1 :=
by sorry

end sequence_sum_proof_l1902_190260


namespace alpha_computation_l1902_190218

theorem alpha_computation (α β : ℂ) :
  (α + β).re > 0 →
  (Complex.I * (α - 3 * β)).re > 0 →
  β = 4 + 3 * Complex.I →
  α = 3 - 3 * Complex.I := by
  sorry

end alpha_computation_l1902_190218


namespace fraction_increase_l1902_190276

theorem fraction_increase (a b : ℝ) (h : 3 * a - 4 * b ≠ 0) :
  (2 * (3 * a) * (3 * b)) / (3 * (3 * a) - 4 * (3 * b)) = 3 * ((2 * a * b) / (3 * a - 4 * b)) :=
by sorry

end fraction_increase_l1902_190276


namespace height_on_hypotenuse_l1902_190244

theorem height_on_hypotenuse (a b h : ℝ) (hyp : ℝ) : 
  a = 2 → b = 3 → a^2 + b^2 = hyp^2 → (a * b) / 2 = (hyp * h) / 2 → h = (6 * Real.sqrt 13) / 13 := by
  sorry

end height_on_hypotenuse_l1902_190244


namespace field_length_is_28_l1902_190232

/-- Proves that the length of a rectangular field is 28 meters given specific conditions --/
theorem field_length_is_28 (l w : ℝ) (h1 : l = 2 * w) (h2 : (7 : ℝ)^2 = (1/8) * (l * w)) : l = 28 :=
by sorry

end field_length_is_28_l1902_190232


namespace inverse_proportion_problem_l1902_190221

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : InverselyProportional x₁ y₁)
  (h2 : InverselyProportional x₂ y₂)
  (h3 : x₁ = 40)
  (h4 : y₁ = 8)
  (h5 : y₂ = 10) :
  x₂ = 32 := by
sorry

end inverse_proportion_problem_l1902_190221


namespace simplify_expression_l1902_190285

theorem simplify_expression (a : ℝ) (h : a ≠ 0) : -2 * a^3 / a = -2 * a^2 := by
  sorry

end simplify_expression_l1902_190285


namespace complex_equation_solution_l1902_190268

theorem complex_equation_solution (z : ℂ) (h : z * (1 + Complex.I) = 2) : z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l1902_190268


namespace hundredth_power_mod_125_l1902_190235

theorem hundredth_power_mod_125 (n : ℤ) : (n^100 : ℤ) % 125 = 0 ∨ (n^100 : ℤ) % 125 = 1 := by
  sorry

end hundredth_power_mod_125_l1902_190235


namespace f_pi_third_eq_half_l1902_190259

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sin x
  else if 1 ≤ x ∧ x ≤ Real.sqrt 2 then Real.cos x
  else Real.tan x

-- Theorem statement
theorem f_pi_third_eq_half : f (Real.pi / 3) = 1 / 2 := by
  sorry

end f_pi_third_eq_half_l1902_190259


namespace correct_calculation_l1902_190208

theorem correct_calculation (x y : ℝ) : 3 * x^4 * y / (x^2 * y) = 3 * x^2 := by
  sorry

end correct_calculation_l1902_190208


namespace path_area_theorem_l1902_190205

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Theorem: The area of a 2.5m wide path around a 60m x 55m field is 600 sq m -/
theorem path_area_theorem :
  path_area 60 55 2.5 = 600 := by sorry

end path_area_theorem_l1902_190205


namespace problem_statement_l1902_190226

theorem problem_statement (n : ℝ) (h : n + 1/n = 5) : n^2 + 1/n^2 + 7 = 30 := by
  sorry

end problem_statement_l1902_190226


namespace infinite_prime_pairs_l1902_190247

theorem infinite_prime_pairs : 
  ∃ (S : Set (ℕ × ℕ)), 
    (∀ (p q : ℕ), (p, q) ∈ S → Nat.Prime p ∧ Nat.Prime q) ∧ 
    (∀ (p q : ℕ), (p, q) ∈ S → p ∣ (2^(q-1) - 1) ∧ q ∣ (2^(p-1) - 1)) ∧ 
    Set.Infinite S :=
by sorry

end infinite_prime_pairs_l1902_190247


namespace invisible_dots_count_l1902_190231

/-- The number of dots on a standard six-sided die -/
def standardDieDots : ℕ := 21

/-- The total number of dots on four standard six-sided dice -/
def totalDots : ℕ := 4 * standardDieDots

/-- The list of visible face values on the stacked dice -/
def visibleFaces : List ℕ := [1, 1, 2, 3, 4, 4, 5, 6]

/-- The sum of the visible face values -/
def visibleDotsSum : ℕ := visibleFaces.sum

/-- Theorem: The number of dots not visible on four stacked standard six-sided dice -/
theorem invisible_dots_count : totalDots - visibleDotsSum = 58 := by
  sorry

end invisible_dots_count_l1902_190231


namespace triangle_area_l1902_190239

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define a median
def isMedian (t : Triangle) (M : ℝ × ℝ) (X Y Z : ℝ × ℝ) : Prop :=
  M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2) ∨ 
  M = ((Y.1 + Z.1) / 2, (Y.2 + Z.2) / 2) ∨ 
  M = ((Z.1 + X.1) / 2, (Z.2 + X.2) / 2)

-- Define the intersection point O
def intersectionPoint (XM YN : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the right angle intersection
def isRightAngle (XM YN : ℝ × ℝ) (O : ℝ × ℝ) : Prop := sorry

-- Define the length of a line segment
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_area (t : Triangle) (M N O : ℝ × ℝ) :
  isMedian t M t.X t.Y t.Z →
  isMedian t N t.X t.Y t.Z →
  O = intersectionPoint M N →
  isRightAngle M N O →
  length t.X M = 18 →
  length t.Y N = 24 →
  area t = 288 := by
  sorry

end triangle_area_l1902_190239


namespace max_min_product_l1902_190266

theorem max_min_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq : a + b + c = 12) (sum_prod : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 6 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 6 :=
sorry

end max_min_product_l1902_190266


namespace deposit_percentage_l1902_190292

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) :
  deposit = 150 →
  remaining = 1350 →
  (deposit / (deposit + remaining)) * 100 = 10 := by
sorry

end deposit_percentage_l1902_190292


namespace min_value_expression_l1902_190250

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / (2 * a) + 1 / b) ≥ Real.sqrt 2 + 3 / 2 := by
  sorry

end min_value_expression_l1902_190250


namespace contrapositive_squared_sum_l1902_190224

theorem contrapositive_squared_sum (x y : ℝ) : x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0 := by
  sorry

end contrapositive_squared_sum_l1902_190224


namespace max_sequence_length_l1902_190291

theorem max_sequence_length (x : ℕ) : 
  (68000 - 55 * x > 0) ∧ (34 * x - 42000 > 0) ↔ x = 1236 :=
sorry

end max_sequence_length_l1902_190291


namespace value_added_to_number_l1902_190209

theorem value_added_to_number : ∃ v : ℝ, 3 * (9 + v) = 9 + 24 ∧ v = 2 := by
  sorry

end value_added_to_number_l1902_190209


namespace power_of_square_l1902_190245

theorem power_of_square (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_square_l1902_190245


namespace exactly_two_correct_propositions_l1902_190229

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the relations
def parallel (a b : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def intersect (p1 p2 : Plane) (l : Line) : Prop := sorry

-- State the theorem
theorem exactly_two_correct_propositions 
  (l m n : Line) (α β γ : Plane) : 
  (∃! (correct : List Prop), 
    correct.length = 2 ∧ 
    correct ⊆ [
      (parallel m l ∧ perpendicular m α → perpendicular l α),
      (parallel m l ∧ parallel m α → parallel l α),
      (intersect α β l ∧ intersect β γ m ∧ intersect γ α n → 
        parallel l m ∧ parallel m n ∧ parallel l n),
      (intersect α β m ∧ intersect β γ l ∧ intersect α γ n ∧ 
        parallel n β → parallel m l)
    ] ∧
    (∀ p ∈ correct, p)) := by
  sorry

end exactly_two_correct_propositions_l1902_190229


namespace largest_prime_divisor_of_prime_square_difference_l1902_190278

theorem largest_prime_divisor_of_prime_square_difference (m n : ℕ) 
  (hm : Prime m) (hn : Prime n) (hmn : m ≠ n) :
  (∃ (p : ℕ) (hp : Prime p), p ∣ (m^2 - n^2)) ∧
  (∀ (q : ℕ) (hq : Prime q), q ∣ (m^2 - n^2) → q ≤ 2) :=
sorry

end largest_prime_divisor_of_prime_square_difference_l1902_190278


namespace sqrt_7x_equals_14_l1902_190294

theorem sqrt_7x_equals_14 (x : ℝ) (h : x / 2 - 5 = 9) : Real.sqrt (7 * x) = 14 := by
  sorry

end sqrt_7x_equals_14_l1902_190294


namespace intersection_of_M_and_N_l1902_190258

def M : Set ℕ := {1, 2, 4, 8, 16}
def N : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_M_and_N : M ∩ N = {2, 4, 8} := by
  sorry

end intersection_of_M_and_N_l1902_190258


namespace ellipse_sum_l1902_190225

theorem ellipse_sum (h k a b : ℝ) : 
  (h = 3) → 
  (k = -5) → 
  (a = 7) → 
  (b = 4) → 
  h + k + a + b = 9 := by
sorry

end ellipse_sum_l1902_190225


namespace cassidy_grounded_days_l1902_190298

/-- The number of days Cassidy is grounded for lying about her report card -/
def days_grounded_for_lying (total_days : ℕ) (grades_below_b : ℕ) (extra_days_per_grade : ℕ) : ℕ :=
  total_days - (grades_below_b * extra_days_per_grade)

/-- Theorem stating that Cassidy was grounded for 14 days for lying about her report card -/
theorem cassidy_grounded_days : 
  days_grounded_for_lying 26 4 3 = 14 := by
  sorry

end cassidy_grounded_days_l1902_190298


namespace younger_person_age_l1902_190211

/-- Given two persons whose ages differ by 20 years, and 5 years ago the elder one was 5 times as old as the younger one, the present age of the younger person is 10 years. -/
theorem younger_person_age (y e : ℕ) : 
  e = y + 20 →                  -- The ages differ by 20 years
  e - 5 = 5 * (y - 5) →         -- 5 years ago, elder was 5 times younger
  y = 10                        -- The younger person's age is 10
  := by sorry

end younger_person_age_l1902_190211


namespace yah_to_bah_conversion_l1902_190273

-- Define the exchange rates
def bah_to_rah_rate : ℚ := 30 / 18
def rah_to_yah_rate : ℚ := 25 / 10

-- Define the conversion function
def convert_yah_to_bah (yahs : ℚ) : ℚ :=
  yahs / (rah_to_yah_rate * bah_to_rah_rate)

-- Theorem statement
theorem yah_to_bah_conversion :
  convert_yah_to_bah 1250 = 300 := by
  sorry

end yah_to_bah_conversion_l1902_190273


namespace batsman_average_after_12th_innings_l1902_190263

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored) / (b.innings + 1)

theorem batsman_average_after_12th_innings
  (b : Batsman)
  (h1 : b.innings = 11)
  (h2 : newAverage b 65 = b.average + 3)
  : newAverage b 65 = 32 := by
  sorry

end batsman_average_after_12th_innings_l1902_190263


namespace trigonometric_equation_solution_l1902_190223

theorem trigonometric_equation_solution (k : ℤ) : 
  let x : ℝ := -Real.arccos (-4/5) + (2 * k + 1 : ℝ) * Real.pi
  let y : ℝ := -1/2
  3 * Real.sin x - 4 * Real.cos x = 4 * y^2 + 4 * y + 6 := by
  sorry

end trigonometric_equation_solution_l1902_190223


namespace rectangle_area_with_hole_l1902_190236

theorem rectangle_area_with_hole (x : ℝ) 
  (h : (3*x ≤ 2*x + 10) ∧ (x ≤ x + 3)) : 
  (2*x + 10) * (x + 3) - (3*x * x) = -x^2 + 16*x + 30 := by
sorry

end rectangle_area_with_hole_l1902_190236


namespace equality_proof_l1902_190227

theorem equality_proof : 2222 - 222 + 22 - 2 = 2020 := by
  sorry

end equality_proof_l1902_190227


namespace isosceles_triangle_condition_l1902_190287

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b/a = (1-cos B)/cos A, then A = C, implying it's an isosceles triangle. -/
theorem isosceles_triangle_condition
  (A B C : ℝ) (a b c : ℝ)
  (triangle_sum : A + B + C = π)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (sine_law : b / a = Real.sin B / Real.sin A)
  (hypothesis : b / a = (1 - Real.cos B) / Real.cos A) :
  A = C :=
sorry

end isosceles_triangle_condition_l1902_190287


namespace binomial_square_coefficient_l1902_190203

theorem binomial_square_coefficient (a : ℝ) : 
  (∃ r s : ℝ, (r * x + s)^2 = a * x^2 + 18 * x + 9) → a = 9 := by
  sorry

end binomial_square_coefficient_l1902_190203


namespace similar_rectangles_l1902_190219

theorem similar_rectangles (w1 l1 w2 : ℝ) (hw1 : w1 = 25) (hl1 : l1 = 40) (hw2 : w2 = 15) :
  let l2 := w2 * l1 / w1
  let perimeter := 2 * (w2 + l2)
  let area := w2 * l2
  (l2 = 24 ∧ perimeter = 78 ∧ area = 360) := by sorry

end similar_rectangles_l1902_190219


namespace difference_of_squares_divisible_by_18_l1902_190256

theorem difference_of_squares_divisible_by_18 (a b : ℤ) 
  (ha : Odd a) (hb : Odd b) : 
  ∃ k : ℤ, (3*a + 2)^2 - (3*b + 2)^2 = 18 * k := by
  sorry

end difference_of_squares_divisible_by_18_l1902_190256


namespace parallelogram_fourth_vertex_l1902_190288

/-- A parallelogram with three known vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ := (1, 1)
  v2 : ℝ × ℝ := (2, 2)
  v3 : ℝ × ℝ := (3, -1)

/-- The fourth vertex of the parallelogram -/
def fourth_vertex (p : Parallelogram) : Set (ℝ × ℝ) :=
  {(2, -2), (4, 0)}

/-- Theorem stating that the fourth vertex of the parallelogram is either (2, -2) or (4, 0) -/
theorem parallelogram_fourth_vertex (p : Parallelogram) :
  ∃ v4 : ℝ × ℝ, v4 ∈ fourth_vertex p :=
sorry

end parallelogram_fourth_vertex_l1902_190288


namespace tangents_intersect_on_AB_l1902_190286

-- Define the basic geometric objects
structure Point : Type :=
  (x y : ℝ)

structure Line : Type :=
  (a b c : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

-- Define the triangle
def Triangle (A B C : Point) : Prop := sorry

-- Define a point on a line segment
def PointOnSegment (D A B : Point) : Prop := sorry

-- Define incircle and excircle
def Incircle (ω : Circle) (A C D : Point) : Prop := sorry
def Excircle (Ω : Circle) (A C D : Point) : Prop := sorry

-- Define tangent line to a circle
def TangentLine (l : Line) (c : Circle) : Prop := sorry

-- Define intersection of lines
def Intersect (l₁ l₂ : Line) (P : Point) : Prop := sorry

-- Define a point on a line
def PointOnLine (P : Point) (l : Line) : Prop := sorry

-- Main theorem
theorem tangents_intersect_on_AB 
  (A B C D : Point) 
  (ω₁ ω₂ Ω₁ Ω₂ : Circle) 
  (AB : Line) :
  Triangle A B C →
  PointOnSegment D A B →
  Incircle ω₁ A C D →
  Incircle ω₂ B C D →
  Excircle Ω₁ A C D →
  Excircle Ω₂ B C D →
  PointOnLine A AB →
  PointOnLine B AB →
  ∃ (P Q : Point) (l₁ l₂ l₃ l₄ : Line),
    TangentLine l₁ ω₁ ∧ TangentLine l₁ ω₂ ∧
    TangentLine l₂ ω₁ ∧ TangentLine l₂ ω₂ ∧
    TangentLine l₃ Ω₁ ∧ TangentLine l₃ Ω₂ ∧
    TangentLine l₄ Ω₁ ∧ TangentLine l₄ Ω₂ ∧
    Intersect l₁ l₂ P ∧ Intersect l₃ l₄ Q ∧
    PointOnLine P AB ∧ PointOnLine Q AB :=
sorry

end tangents_intersect_on_AB_l1902_190286


namespace collinear_points_m_equals_four_l1902_190295

-- Define the points
def A : ℝ × ℝ := (-2, 12)
def B : ℝ × ℝ := (1, 3)
def C : ℝ → ℝ × ℝ := λ m ↦ (m, -6)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - p.1) = (r.2 - p.2) * (q.1 - p.1)

-- Theorem statement
theorem collinear_points_m_equals_four :
  collinear A B (C 4) := by sorry

end collinear_points_m_equals_four_l1902_190295


namespace triangle_area_l1902_190264

theorem triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 12) (h3 : c = 5) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 30 := by
  sorry

end triangle_area_l1902_190264


namespace least_b_value_l1902_190296

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The smallest prime factor of a positive integer -/
def smallest_prime_factor (n : ℕ+) : ℕ := sorry

theorem least_b_value (a b : ℕ+) 
  (ha_factors : num_factors a = 3)
  (hb_factors : num_factors b = a)
  (hb_div_a : a ∣ b)
  (ha_smallest_prime : smallest_prime_factor a = 3) :
  36 ≤ b :=
sorry

end least_b_value_l1902_190296


namespace parabola_point_order_l1902_190272

/-- Parabola function -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem parabola_point_order (a b c d : ℝ) : 
  f a = 2 → f b = 6 → f c = d → d < 1 → a < 0 → b > 0 → a < c ∧ c < b :=
by sorry

end parabola_point_order_l1902_190272


namespace rope_length_proof_l1902_190274

theorem rope_length_proof : 
  ∀ (L : ℝ), 
    (L / 4 - L / 6 = 2) →  -- Difference between parts is 2 meters
    (2 * L = 48)           -- Total length of two ropes is 48 meters
  := by sorry

end rope_length_proof_l1902_190274


namespace hyperbola_focal_distance_l1902_190207

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  asymptote1 : ℝ → ℝ
  asymptote2 : ℝ → ℝ
  point : ℝ × ℝ

/-- The distance between the foci of a hyperbola -/
def focalDistance (h : Hyperbola) : ℝ := sorry

/-- Theorem: For a hyperbola with asymptotes y = x + 3 and y = -x + 5, 
    passing through the point (4,6), the distance between its foci is 2√10 -/
theorem hyperbola_focal_distance :
  let h : Hyperbola := {
    asymptote1 := fun x ↦ x + 3,
    asymptote2 := fun x ↦ -x + 5,
    point := (4, 6)
  }
  focalDistance h = 2 * Real.sqrt 10 := by sorry

end hyperbola_focal_distance_l1902_190207


namespace root_sum_of_quadratic_l1902_190293

theorem root_sum_of_quadratic : ∃ (C D : ℝ), 
  (∀ x : ℝ, 3 * x^2 - 9 * x + 6 = 0 ↔ (x = C ∨ x = D)) ∧ 
  C + D = 3 := by
  sorry

end root_sum_of_quadratic_l1902_190293


namespace hyperbola_asymptote_slope_l1902_190289

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 3)^2) - Real.sqrt ((x - 7)^2 + (y + 3)^2) = 4

/-- The positive slope of an asymptote of the hyperbola -/
def positive_asymptote_slope : ℝ := 0.75

/-- Theorem stating that the positive slope of an asymptote of the given hyperbola is 0.75 -/
theorem hyperbola_asymptote_slope :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ positive_asymptote_slope = 0.75 :=
sorry

end hyperbola_asymptote_slope_l1902_190289


namespace g_100_value_l1902_190299

/-- A function satisfying the given property for all positive real numbers -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → x * g y - y * g x = g (x / y) + x - y

/-- The main theorem stating the value of g(100) -/
theorem g_100_value (g : ℝ → ℝ) (h : SatisfiesProperty g) : g 100 = -99 / 2 := by
  sorry


end g_100_value_l1902_190299


namespace sphere_surface_area_increase_l1902_190243

theorem sphere_surface_area_increase (r : ℝ) (h : r > 0) :
  let original_area := 4 * Real.pi * r^2
  let new_radius := 1.5 * r
  let new_area := 4 * Real.pi * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end sphere_surface_area_increase_l1902_190243


namespace inequality_proof_l1902_190253

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  1 / (8 * a^2 - 18 * a + 11) + 1 / (8 * b^2 - 18 * b + 11) + 1 / (8 * c^2 - 18 * c + 11) ≤ 3 := by
  sorry

end inequality_proof_l1902_190253


namespace square_area_ratio_l1902_190282

theorem square_area_ratio (x : ℝ) (hx : x > 0) : (x^2) / ((3*x)^2) = 1/9 := by
  sorry

end square_area_ratio_l1902_190282


namespace nonnegative_sum_one_inequality_l1902_190217

theorem nonnegative_sum_one_inequality (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum_one : x + y + z = 1) : 
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end nonnegative_sum_one_inequality_l1902_190217


namespace fence_perimeter_is_112_l1902_190230

-- Define the parameters of the fence
def total_posts : ℕ := 28
def posts_on_long_side : ℕ := 6
def gap_between_posts : ℕ := 4

-- Define the function to calculate the perimeter
def fence_perimeter : ℕ := 
  let posts_on_short_side := (total_posts - 2 * posts_on_long_side + 2) / 2 + 1
  let long_side_length := (posts_on_long_side - 1) * gap_between_posts
  let short_side_length := (posts_on_short_side - 1) * gap_between_posts
  2 * (long_side_length + short_side_length)

-- Theorem statement
theorem fence_perimeter_is_112 : fence_perimeter = 112 := by
  sorry

end fence_perimeter_is_112_l1902_190230


namespace sqrt_neg_two_squared_l1902_190201

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by
  sorry

end sqrt_neg_two_squared_l1902_190201


namespace seedling_probability_value_l1902_190215

/-- The germination rate of seeds in a batch -/
def germination_rate : ℝ := 0.9

/-- The survival rate of seedlings after germination -/
def survival_rate : ℝ := 0.8

/-- The probability that a randomly selected seed will grow into a seedling -/
def seedling_probability : ℝ := germination_rate * survival_rate

/-- Theorem stating that the probability of a randomly selected seed growing into a seedling is 0.72 -/
theorem seedling_probability_value : seedling_probability = 0.72 := by
  sorry

end seedling_probability_value_l1902_190215


namespace card_game_guarantee_l1902_190206

/-- Represents a cell on the 4x9 board --/
structure Cell :=
  (row : Fin 4)
  (col : Fin 9)

/-- Represents a pair of cells --/
structure CellPair :=
  (cell1 : Cell)
  (cell2 : Cell)

/-- Represents the state of the board --/
def Board := Fin 4 → Fin 9 → Bool

/-- A valid pairing of cells --/
def ValidPairing (board : Board) (pairs : List CellPair) : Prop :=
  ∀ p ∈ pairs,
    (board p.cell1.row p.cell1.col ≠ board p.cell2.row p.cell2.col) ∧
    ((p.cell1.row = p.cell2.row) ∨ (p.cell1.col = p.cell2.col))

/-- The main theorem --/
theorem card_game_guarantee (board : Board) :
  (∃ black_count : ℕ, black_count = 18 ∧ 
    (∀ r : Fin 4, ∀ c : Fin 9, (board r c = true) → black_count = black_count - 1)) →
  ∃ pairs : List CellPair, ValidPairing board pairs ∧ pairs.length ≥ 15 := by
  sorry

end card_game_guarantee_l1902_190206


namespace katy_brownies_theorem_l1902_190269

/-- The number of brownies Katy made -/
def total_brownies : ℕ := 15

/-- The number of brownies Katy ate on Monday -/
def monday_brownies : ℕ := 5

/-- The number of brownies Katy ate on Tuesday -/
def tuesday_brownies : ℕ := 2 * monday_brownies

theorem katy_brownies_theorem :
  total_brownies = monday_brownies + tuesday_brownies :=
by sorry

end katy_brownies_theorem_l1902_190269


namespace stewart_farm_ratio_l1902_190284

/-- Proves that the ratio of sheep to horses is 4:7 given the farm conditions --/
theorem stewart_farm_ratio (sheep : ℕ) (horse_food_per_day : ℕ) (total_horse_food : ℕ) :
  sheep = 32 →
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  (sheep : ℚ) / (total_horse_food / horse_food_per_day : ℚ) = 4 / 7 := by
  sorry

end stewart_farm_ratio_l1902_190284


namespace triangle_inequality_l1902_190234

-- Define a structure for a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  nondeg : min a b < c ∧ c < a + b -- Nondegenerate condition
  unit_perimeter : a + b + c = 1 -- Unit perimeter condition

-- Define the theorem
theorem triangle_inequality (t : Triangle) :
  |((t.a - t.b)/(t.c + t.a*t.b))| + |((t.b - t.c)/(t.a + t.b*t.c))| + |((t.c - t.a)/(t.b + t.a*t.c))| < 2 := by
  sorry

end triangle_inequality_l1902_190234


namespace joan_balloon_count_l1902_190281

theorem joan_balloon_count (total : ℕ) (melanie : ℕ) (joan : ℕ) : 
  total = 81 → melanie = 41 → joan + melanie = total → joan = 40 := by
  sorry

end joan_balloon_count_l1902_190281


namespace fifth_power_sum_l1902_190248

theorem fifth_power_sum (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : 1 / (x + y^2) + 1 / (x^2 + y) = 1 / 2) : 
  x^5 + y^5 = 123 := by
  sorry

end fifth_power_sum_l1902_190248


namespace max_area_CDFE_l1902_190262

/-- The area of quadrilateral CDFE in a square ABCD with side length 2,
    where E and F are on sides AB and AD respectively, and AE = AF = 2k. -/
def area_CDFE (k : ℝ) : ℝ := 2 * (1 - k)^2

/-- The theorem stating that the area of CDFE is maximized when k = 1/2,
    and the maximum area is 1/2. -/
theorem max_area_CDFE :
  ∀ k : ℝ, 0 < k → k < 1 →
  area_CDFE k ≤ area_CDFE (1/2) ∧ area_CDFE (1/2) = 1/2 :=
sorry

end max_area_CDFE_l1902_190262


namespace monotonic_sufficient_not_necessary_for_maximum_l1902_190254

theorem monotonic_sufficient_not_necessary_for_maximum 
  (f : ℝ → ℝ) (hf : ContinuousOn f (Set.Icc 0 1)) :
  (MonotoneOn f (Set.Icc 0 1) → ∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f y ≤ f x) ∧
  ¬(∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f y ≤ f x → MonotoneOn f (Set.Icc 0 1)) :=
sorry

end monotonic_sufficient_not_necessary_for_maximum_l1902_190254


namespace midpoint_trajectory_equation_l1902_190220

/-- The equation of the trajectory of the midpoint of the line connecting a fixed point to any point on a circle -/
theorem midpoint_trajectory_equation (P : ℝ × ℝ) (r : ℝ) :
  P = (4, -2) →
  r = 2 →
  ∀ (x y : ℝ), (∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 = r^2 ∧ x = (x₁ + P.1) / 2 ∧ y = (y₁ + P.2) / 2) →
  (x - 2)^2 + (y + 1)^2 = 1 := by
  sorry

end midpoint_trajectory_equation_l1902_190220


namespace function_characterization_l1902_190228

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- Theorem stating that any function satisfying the equation must be of the form f(x) = cx -/
theorem function_characterization (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end function_characterization_l1902_190228


namespace largest_n_divisible_by_103_l1902_190270

theorem largest_n_divisible_by_103 : 
  ∀ n : ℕ, n < 103 ∧ 103 ∣ (n^3 - 1) → n ≤ 52 :=
by sorry

end largest_n_divisible_by_103_l1902_190270


namespace turnip_bag_options_l1902_190257

def bag_weights : List Nat := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (turnip_weight : Nat) : Prop :=
  turnip_weight ∈ bag_weights ∧
  ∃ (onion_weights carrots_weights : List Nat),
    onion_weights ++ carrots_weights ++ [turnip_weight] = bag_weights ∧
    onion_weights.sum * 2 = carrots_weights.sum

theorem turnip_bag_options :
  ∀ w ∈ bag_weights, is_valid_turnip_weight w ↔ w = 13 ∨ w = 16 := by sorry

end turnip_bag_options_l1902_190257


namespace wage_recovery_percentage_l1902_190241

theorem wage_recovery_percentage (original_wage : ℝ) (h : original_wage > 0) :
  let decreased_wage := 0.7 * original_wage
  let required_increase := (original_wage / decreased_wage - 1) * 100
  ∃ ε > 0, abs (required_increase - 42.86) < ε :=
by
  sorry

end wage_recovery_percentage_l1902_190241


namespace blue_twice_prob_octahedron_l1902_190297

/-- A regular octahedron with colored faces -/
structure ColoredOctahedron where
  blue_faces : ℕ
  red_faces : ℕ
  total_faces : ℕ
  is_regular : Prop
  face_sum : blue_faces + red_faces = total_faces

/-- The probability of an event occurring twice in independent trials -/
def independent_event_twice_prob (single_prob : ℚ) : ℚ :=
  single_prob * single_prob

/-- The probability of rolling a blue face twice in succession on a colored octahedron -/
def blue_twice_prob (o : ColoredOctahedron) : ℚ :=
  independent_event_twice_prob ((o.blue_faces : ℚ) / (o.total_faces : ℚ))

theorem blue_twice_prob_octahedron :
  ∃ (o : ColoredOctahedron),
    o.blue_faces = 5 ∧
    o.red_faces = 3 ∧
    o.total_faces = 8 ∧
    o.is_regular ∧
    blue_twice_prob o = 25 / 64 := by
  sorry

end blue_twice_prob_octahedron_l1902_190297


namespace equation_solution_l1902_190202

theorem equation_solution (m n : ℝ) : 21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 := by
  sorry

end equation_solution_l1902_190202


namespace equation_one_solutions_l1902_190242

theorem equation_one_solutions (x : ℝ) :
  (x - 2)^2 = 4 → x = 4 ∨ x = 0 := by
  sorry

#check equation_one_solutions

end equation_one_solutions_l1902_190242


namespace peter_book_percentage_l1902_190240

theorem peter_book_percentage (total_books : ℕ) (brother_percentage : ℚ) (difference : ℕ) : 
  total_books = 20 →
  brother_percentage = 1/10 →
  difference = 6 →
  (↑(brother_percentage * ↑total_books + ↑difference) / ↑total_books : ℚ) = 2/5 := by
  sorry

end peter_book_percentage_l1902_190240


namespace alcohol_percentage_after_dilution_l1902_190210

/-- Calculates the alcohol percentage in a mixture after adding water -/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 9)
  (h2 : initial_alcohol_percentage = 57)
  (h3 : water_added = 3) :
  let alcohol_volume := initial_volume * (initial_alcohol_percentage / 100)
  let total_volume := initial_volume + water_added
  let new_alcohol_percentage := (alcohol_volume / total_volume) * 100
  new_alcohol_percentage = 42.75 := by
sorry

end alcohol_percentage_after_dilution_l1902_190210


namespace bounded_by_one_l1902_190283

/-- A function from integers to reals satisfying certain properties -/
def IntToRealFunction (f : ℤ → ℝ) : Prop :=
  (∀ n, f n ≥ 0) ∧ 
  (∀ m n, f (m * n) = f m * f n) ∧ 
  (∀ m n, f (m + n) ≤ max (f m) (f n))

/-- Theorem stating that any function satisfying IntToRealFunction is bounded above by 1 -/
theorem bounded_by_one (f : ℤ → ℝ) (hf : IntToRealFunction f) : 
  ∀ n, f n ≤ 1 := by
  sorry

end bounded_by_one_l1902_190283


namespace slope_of_solutions_l1902_190237

/-- The equation that defines the relationship between x and y -/
def equation (x y : ℝ) : Prop := (4 / x) + (6 / y) = 0

/-- Theorem stating that the slope between any two distinct solutions of the equation is -3/2 -/
theorem slope_of_solutions (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : equation x₁ y₁) (h₂ : equation x₂ y₂) (h_dist : (x₁, y₁) ≠ (x₂, y₂)) :
  (y₂ - y₁) / (x₂ - x₁) = -3/2 := by
sorry

end slope_of_solutions_l1902_190237


namespace centroid_circumcenter_distance_squared_l1902_190252

/-- Given a triangle with medians m_a, m_b, m_c and circumradius R,
    the squared distance between the centroid and circumcenter (SM^2)
    is equal to R^2 - (4/27)(m_a^2 + m_b^2 + m_c^2) -/
theorem centroid_circumcenter_distance_squared
  (m_a m_b m_c R : ℝ) :
  ∃ (SM : ℝ),
    SM^2 = R^2 - (4/27) * (m_a^2 + m_b^2 + m_c^2) :=
by sorry

end centroid_circumcenter_distance_squared_l1902_190252


namespace solution_set_theorem_range_of_a_theorem_l1902_190280

def f (x : ℝ) : ℝ := |x + 2| - 2 * |x - 1|

theorem solution_set_theorem (x : ℝ) :
  f x ≥ -2 ↔ -2/3 ≤ x ∧ x ≤ 6 :=
sorry

theorem range_of_a_theorem :
  (∀ x : ℝ, f x ≤ x - a) ↔ a ≤ -2 :=
sorry

end solution_set_theorem_range_of_a_theorem_l1902_190280


namespace digit_difference_in_base_d_l1902_190267

/-- Represents a digit in a given base --/
def Digit (d : ℕ) := { n : ℕ // n < d }

/-- Represents a two-digit number in a given base --/
def TwoDigitNumber (d : ℕ) (A B : Digit d) : ℕ := A.val * d + B.val

theorem digit_difference_in_base_d 
  (d : ℕ) 
  (h_d : d > 7) 
  (A B : Digit d) 
  (h_sum : TwoDigitNumber d A B + TwoDigitNumber d A A = 175) : 
  A.val - B.val = 2 := by
sorry

end digit_difference_in_base_d_l1902_190267


namespace calculate_ambulance_ride_cost_l1902_190249

/-- Given a hospital bill with various components, calculate the cost of the ambulance ride. -/
theorem calculate_ambulance_ride_cost (total_bill : ℝ) (medication_percent : ℝ) 
  (imaging_percent : ℝ) (surgical_percent : ℝ) (overnight_percent : ℝ) (doctor_percent : ℝ) 
  (food_fee : ℝ) (consultation_fee : ℝ) (therapy_fee : ℝ) 
  (h1 : total_bill = 18000)
  (h2 : medication_percent = 35)
  (h3 : imaging_percent = 15)
  (h4 : surgical_percent = 25)
  (h5 : overnight_percent = 10)
  (h6 : doctor_percent = 5)
  (h7 : food_fee = 300)
  (h8 : consultation_fee = 450)
  (h9 : therapy_fee = 600) :
  total_bill - (medication_percent / 100 * total_bill + 
                imaging_percent / 100 * total_bill + 
                surgical_percent / 100 * total_bill + 
                overnight_percent / 100 * total_bill + 
                doctor_percent / 100 * total_bill + 
                food_fee + consultation_fee + therapy_fee) = 450 := by
  sorry


end calculate_ambulance_ride_cost_l1902_190249


namespace square_1225_identity_l1902_190222

theorem square_1225_identity (x : ℤ) (h : x^2 = 1225) : (x + 2) * (x - 2) = 1221 := by
  sorry

end square_1225_identity_l1902_190222


namespace coal_consumption_factory_coal_consumption_l1902_190277

/-- Given a factory that burns coal at a constant daily rate, calculate the total coal burned over a longer period. -/
theorem coal_consumption (initial_coal : ℝ) (initial_days : ℝ) (total_days : ℝ) :
  initial_coal > 0 → initial_days > 0 → total_days > initial_days →
  (initial_coal / initial_days) * total_days = 
    initial_coal * (total_days / initial_days) := by
  sorry

/-- Specific instance of coal consumption calculation -/
theorem factory_coal_consumption :
  let initial_coal : ℝ := 37.5
  let initial_days : ℝ := 5
  let total_days : ℝ := 13
  (initial_coal / initial_days) * total_days = 97.5 := by
  sorry

end coal_consumption_factory_coal_consumption_l1902_190277


namespace equation_exists_l1902_190214

theorem equation_exists : ∃ (a b c d e f g h i : ℕ),
  (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧ 
  (e < 10) ∧ (f < 10) ∧ (g < 10) ∧ (h < 10) ∧ (i < 10) ∧
  (a + 100 * b + 10 * c + d = 10 * e + f + 100 * g + 10 * h + i) ∧
  (b = d) ∧ (g = h) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (a ≠ i) ∧
  (b ≠ c) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (b ≠ i) ∧
  (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ (c ≠ i) ∧
  (e ≠ f) ∧ (e ≠ g) ∧ (e ≠ i) ∧
  (f ≠ g) ∧ (f ≠ i) ∧
  (g ≠ i) :=
by
  sorry

end equation_exists_l1902_190214


namespace sum_of_squared_sums_of_roots_l1902_190233

theorem sum_of_squared_sums_of_roots (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := by
sorry

end sum_of_squared_sums_of_roots_l1902_190233


namespace new_average_age_after_move_l1902_190212

theorem new_average_age_after_move (room_a_initial_count : ℕ)
                                   (room_a_initial_avg : ℚ)
                                   (room_b_initial_count : ℕ)
                                   (room_b_initial_avg : ℚ)
                                   (moving_person_age : ℕ) :
  room_a_initial_count = 8 →
  room_a_initial_avg = 35 →
  room_b_initial_count = 5 →
  room_b_initial_avg = 30 →
  moving_person_age = 40 →
  let total_initial_a := room_a_initial_count * room_a_initial_avg
  let total_initial_b := room_b_initial_count * room_b_initial_avg
  let new_total_a := total_initial_a - moving_person_age
  let new_total_b := total_initial_b + moving_person_age
  let new_count_a := room_a_initial_count - 1
  let new_count_b := room_b_initial_count + 1
  let total_new_age := new_total_a + new_total_b
  let total_new_count := new_count_a + new_count_b
  (total_new_age / total_new_count : ℚ) = 33.08 := by
sorry

end new_average_age_after_move_l1902_190212


namespace prob_select_AB_correct_l1902_190251

/-- The number of students in the class -/
def num_students : ℕ := 5

/-- The number of students to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly A and B -/
def prob_select_AB : ℚ := 1 / 10

theorem prob_select_AB_correct :
  prob_select_AB = (1 : ℚ) / (num_students.choose num_selected) :=
by sorry

end prob_select_AB_correct_l1902_190251


namespace base5_multiplication_addition_l1902_190200

/-- Converts a base 5 number to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 5 --/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The main theorem --/
theorem base5_multiplication_addition :
  decimalToBase5 (base5ToDecimal [1, 3, 2] * base5ToDecimal [1, 3] + base5ToDecimal [4, 1]) =
  [0, 3, 1, 0, 1] := by sorry

end base5_multiplication_addition_l1902_190200


namespace intersection_distance_l1902_190213

/-- Given a linear function f(x) = ax + b, if the distance between the intersection points
    of y=x^2+2 and y=f(x) is √10, and the distance between the intersection points of
    y=x^2-1 and y=f(x)+1 is √42, then the distance between the intersection points of
    y=x^2 and y=f(x)+1 is √34. -/
theorem intersection_distance (a b : ℝ) : 
  let f := (fun x : ℝ => a * x + b)
  let d1 := Real.sqrt ((a^2 + 1) * (a^2 + 4*b - 8))
  let d2 := Real.sqrt ((a^2 + 1) * (a^2 + 4*b + 8))
  d1 = Real.sqrt 10 ∧ d2 = Real.sqrt 42 →
  Real.sqrt ((a^2 + 1) * (a^2 + 4*b + 4)) = Real.sqrt 34 :=
by sorry

end intersection_distance_l1902_190213


namespace complex_fraction_simplification_l1902_190238

theorem complex_fraction_simplification : ∀ (x : ℝ),
  x = (3 * (Real.sqrt 3 + Real.sqrt 7)) / (4 * Real.sqrt (3 + Real.sqrt 2)) →
  x ≠ 3 * Real.sqrt 7 / 4 ∧
  x ≠ 9 * Real.sqrt 2 / 16 ∧
  x ≠ 3 * Real.sqrt 3 / 4 ∧
  x ≠ 15 / 8 ∧
  x ≠ 9 / 4 :=
by sorry

end complex_fraction_simplification_l1902_190238


namespace subtract_inequality_l1902_190290

theorem subtract_inequality {a b c : ℝ} (h : a > b) : a - c > b - c := by
  sorry

end subtract_inequality_l1902_190290


namespace largest_blue_balls_l1902_190271

theorem largest_blue_balls (total : ℕ) (is_prime : ℕ → Prop) : 
  total = 72 →
  (∃ (red blue prime : ℕ), 
    red + blue = total ∧ 
    is_prime prime ∧ 
    red = blue + prime) →
  (∃ (max_blue : ℕ), 
    max_blue ≤ total ∧
    (∀ (blue : ℕ), 
      blue ≤ total →
      (∃ (red prime : ℕ), 
        red + blue = total ∧ 
        is_prime prime ∧ 
        red = blue + prime) →
      blue ≤ max_blue) ∧
    max_blue = 35) :=
by sorry

end largest_blue_balls_l1902_190271


namespace lcm_of_6_8_10_l1902_190275

theorem lcm_of_6_8_10 : Nat.lcm (Nat.lcm 6 8) 10 = 120 := by
  sorry

end lcm_of_6_8_10_l1902_190275


namespace race_distance_race_distance_proof_l1902_190255

/-- The total distance of a race where:
    - The ratio of speeds of contestants A and B is 2:4
    - A has a start of 300 m
    - A wins by 100 m
-/
theorem race_distance : ℝ :=
  let speed_ratio : ℚ := 2 / 4
  let head_start : ℝ := 300
  let winning_margin : ℝ := 100
  500

theorem race_distance_proof (speed_ratio : ℚ) (head_start winning_margin : ℝ) :
  speed_ratio = 2 / 4 →
  head_start = 300 →
  winning_margin = 100 →
  race_distance = 500 := by
  sorry

end race_distance_race_distance_proof_l1902_190255


namespace possible_pen_counts_l1902_190279

def total_money : ℕ := 11
def pen_cost : ℕ := 3
def notebook_cost : ℕ := 1

def valid_pen_count (x : ℕ) : Prop :=
  ∃ y : ℕ, x * pen_cost + y * notebook_cost = total_money

theorem possible_pen_counts : 
  (valid_pen_count 1 ∧ valid_pen_count 2 ∧ valid_pen_count 3) ∧
  (∀ x : ℕ, valid_pen_count x → x = 1 ∨ x = 2 ∨ x = 3) :=
sorry

end possible_pen_counts_l1902_190279
