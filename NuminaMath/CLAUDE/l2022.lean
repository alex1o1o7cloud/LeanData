import Mathlib

namespace orthic_triangle_right_angled_iff_45_or_135_angle_l2022_202287

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the orthic triangle of a given triangle
def orthicTriangle (t : Triangle) : Triangle := sorry

-- Define a right-angled triangle
def isRightAngled (t : Triangle) : Prop := sorry

-- Define the condition for an angle to be 45° or 135°
def has45or135Angle (t : Triangle) : Prop := sorry

-- Theorem statement
theorem orthic_triangle_right_angled_iff_45_or_135_angle (t : Triangle) :
  has45or135Angle t ↔ isRightAngled (orthicTriangle t) := by sorry

end orthic_triangle_right_angled_iff_45_or_135_angle_l2022_202287


namespace f_minimum_value_tangent_line_equation_l2022_202243

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - x - 2 * Real.log x + 1/2

-- Theorem for the minimum value of f(x)
theorem f_minimum_value :
  ∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≥ f x_min ∧ f x_min = -2 * Real.log 2 + 1/2 :=
sorry

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (x₀ : ℝ), x₀ > 0 ∧
  (2 * x₀ + f x₀ - 2 = 0) ∧
  ∀ (x : ℝ), 2 * x + f x₀ + (x - x₀) * (x₀ - 1 - 2 / x₀) - 2 = 0 :=
sorry

end f_minimum_value_tangent_line_equation_l2022_202243


namespace cost_price_proof_l2022_202288

/-- The cost price of an article satisfying the given profit and loss conditions. -/
def cost_price : ℝ := 49

/-- The selling price that results in a profit. -/
def profit_price : ℝ := 56

/-- The selling price that results in a loss. -/
def loss_price : ℝ := 42

theorem cost_price_proof :
  (profit_price - cost_price = cost_price - loss_price) →
  cost_price = 49 :=
by
  sorry

end cost_price_proof_l2022_202288


namespace city_population_l2022_202270

/-- If 96% of a city's population is 23040, then the total population is 24000. -/
theorem city_population (population : ℕ) : 
  (96 : ℚ) / 100 * population = 23040 → population = 24000 := by
  sorry

end city_population_l2022_202270


namespace total_area_is_68_l2022_202203

/-- Represents the dimensions of a rectangle -/
structure RectDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def rectangleArea (rect : RectDimensions) : ℕ :=
  rect.width * rect.height

/-- The dimensions of the four rectangles in the figure -/
def rect1 : RectDimensions := ⟨5, 7⟩
def rect2 : RectDimensions := ⟨3, 3⟩
def rect3 : RectDimensions := ⟨4, 1⟩
def rect4 : RectDimensions := ⟨5, 4⟩

/-- Theorem: The total area of the composite shape is 68 square units -/
theorem total_area_is_68 : 
  rectangleArea rect1 + rectangleArea rect2 + rectangleArea rect3 + rectangleArea rect4 = 68 := by
  sorry

end total_area_is_68_l2022_202203


namespace fencing_cost_per_meter_l2022_202266

/-- Proves that for a rectangular field with sides in the ratio 3:4, area of 7500 sq. m,
    and a total fencing cost of 87.5, the cost per metre of fencing is 0.25. -/
theorem fencing_cost_per_meter (length width : ℝ) (h1 : width / length = 4 / 3)
    (h2 : length * width = 7500) (h3 : 87.5 = 2 * (length + width) * cost_per_meter) :
  cost_per_meter = 0.25 := by
  sorry

end fencing_cost_per_meter_l2022_202266


namespace card_58_is_6_l2022_202238

/-- The sequence of playing cards -/
def card_sequence : ℕ → ℕ :=
  fun n => (n - 1) % 13 + 1

/-- The 58th card in the sequence -/
def card_58 : ℕ := card_sequence 58

theorem card_58_is_6 : card_58 = 6 := by
  sorry

end card_58_is_6_l2022_202238


namespace p_sufficient_not_necessary_for_q_l2022_202213

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (x - 1) / (x + 2) ≤ 0 → -2 ≤ x ∧ x ≤ 1) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 1 ∧ (x - 1) / (x + 2) > 0) :=
by sorry

end p_sufficient_not_necessary_for_q_l2022_202213


namespace regular_octagon_interior_angle_l2022_202267

/-- The measure of each interior angle in a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ := by
  sorry

end regular_octagon_interior_angle_l2022_202267


namespace parabola_vertex_l2022_202282

/-- A parabola is defined by the equation y = 3(x-7)^2 + 5. -/
def parabola (x y : ℝ) : Prop := y = 3 * (x - 7)^2 + 5

/-- The vertex of a parabola is the point where it reaches its minimum or maximum. -/
def is_vertex (x y : ℝ) : Prop := parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

/-- The vertex of the parabola y = 3(x-7)^2 + 5 has coordinates (7, 5). -/
theorem parabola_vertex : is_vertex 7 5 := by sorry

end parabola_vertex_l2022_202282


namespace min_value_of_f_l2022_202228

noncomputable def f (a : ℝ) : ℝ := a/2 - 1/4 + (Real.exp (-2*a))/2

theorem min_value_of_f :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (b : ℝ), b > 0 → f a ≤ f b) ∧
  a = Real.log 2 / 2 := by
sorry

end min_value_of_f_l2022_202228


namespace systematic_sampling_groups_for_56_and_8_l2022_202260

/-- Calculates the number of groups formed in systematic sampling -/
def systematicSamplingGroups (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

theorem systematic_sampling_groups_for_56_and_8 :
  systematicSamplingGroups 56 8 = 8 := by
  sorry

#eval systematicSamplingGroups 56 8

end systematic_sampling_groups_for_56_and_8_l2022_202260


namespace expression_evaluation_l2022_202253

theorem expression_evaluation : 
  let x : ℚ := 1/2
  (((x^2 - 2*x + 1) / (x^2 - 1) - 1 / (x + 1)) / ((2*x - 4) / (x^2 + x))) = 1/4 := by
  sorry

end expression_evaluation_l2022_202253


namespace imaginary_part_of_complex_fraction_l2022_202298

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + 2*i) / (i - 1)) = -3/2 := by
sorry

end imaginary_part_of_complex_fraction_l2022_202298


namespace money_sharing_l2022_202239

theorem money_sharing (john_share : ℕ) (jose_share : ℕ) (binoy_share : ℕ) 
  (h1 : john_share = 1400)
  (h2 : jose_share = 2 * john_share)
  (h3 : binoy_share = 3 * john_share) :
  john_share + jose_share + binoy_share = 8400 := by
  sorry

end money_sharing_l2022_202239


namespace dot_product_range_l2022_202278

-- Define the points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, -1)

-- Define the curve y = √(1-x^2)
def on_curve (P : ℝ × ℝ) : Prop :=
  P.2 = Real.sqrt (1 - P.1^2)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the vector from B to P
def BP (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 - B.1, P.2 - B.2)

-- Define the vector from B to A
def BA : ℝ × ℝ :=
  (A.1 - B.1, A.2 - B.2)

-- The main theorem
theorem dot_product_range :
  ∀ P : ℝ × ℝ, on_curve P →
  0 ≤ dot_product (BP P) BA ∧ dot_product (BP P) BA ≤ 1 + Real.sqrt 2 :=
by sorry

end dot_product_range_l2022_202278


namespace negation_inverse_implies_contrapositive_l2022_202273

-- Define propositions as functions from some universe U to Prop
variable {U : Type}
variable (p q r : U → Prop)

-- Define the negation relation
def is_negation (p q : U → Prop) : Prop :=
  ∀ x, q x ↔ ¬(p x)

-- Define the inverse relation
def is_inverse (q r : U → Prop) : Prop :=
  ∀ x, r x ↔ (¬q x)

-- Define the contrapositive relation
def is_contrapositive (p r : U → Prop) : Prop :=
  ∀ x y, (p x → p y) ↔ (¬p y → ¬p x)

-- The main theorem
theorem negation_inverse_implies_contrapositive (p q r : U → Prop) :
  is_negation p q → is_inverse q r → is_contrapositive p r :=
sorry

end negation_inverse_implies_contrapositive_l2022_202273


namespace chord_minimum_value_l2022_202219

theorem chord_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let line := {p : ℝ × ℝ | a * p.1 - b * p.2 + 2 = 0}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 + 1 = 0}
  let chord_length := 4
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ q ∈ line ∧ p ∈ circle ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) →
  (2/a + 3/b ≥ 4 + 2 * Real.sqrt 3) ∧ 
  (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ 2/a' + 3/b' = 4 + 2 * Real.sqrt 3) :=
by sorry

end chord_minimum_value_l2022_202219


namespace book_sale_fraction_l2022_202232

/-- Given a book sale where some books were sold for $2 each, 36 books remained unsold,
    and the total amount received was $144, prove that 2/3 of the books were sold. -/
theorem book_sale_fraction (B : ℕ) (h1 : B > 36) : 
  2 * (B - 36) = 144 → (B - 36 : ℚ) / B = 2/3 := by
  sorry

end book_sale_fraction_l2022_202232


namespace sum_of_composite_function_l2022_202205

def p (x : ℝ) : ℝ := 2 * abs x - 1

def q (x : ℝ) : ℝ := -abs x - 1

def xValues : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composite_function : 
  (xValues.map (fun x => q (p x))).sum = -42 := by sorry

end sum_of_composite_function_l2022_202205


namespace tangent_line_equation_l2022_202252

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (x - a)

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * a * x

-- Theorem statement
theorem tangent_line_equation (a : ℝ) (h : f' a 1 = 3) :
  ∃ (m b : ℝ), m * 1 - b = f a 1 ∧ 
                ∀ x, m * x - b = 3 * x - 2 :=
sorry

end tangent_line_equation_l2022_202252


namespace lucky_larry_calculation_l2022_202210

theorem lucky_larry_calculation (a b c d e : ℚ) : 
  a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 6 →
  a * b + c * d - c * e + 1 = a * (b + (c * (d - e))) →
  e = 23/4 := by
sorry

end lucky_larry_calculation_l2022_202210


namespace intersection_equals_open_interval_l2022_202258

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | |x - 1| < 2}
def N : Set ℝ := {x : ℝ | x < 2}

-- Define the open interval (-1, 2)
def open_interval : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem intersection_equals_open_interval : M ∩ N = open_interval := by
  sorry

end intersection_equals_open_interval_l2022_202258


namespace bird_cage_problem_l2022_202246

theorem bird_cage_problem (initial_birds : ℕ) : 
  (1 / 3 : ℚ) * (3 / 5 : ℚ) * (1 / 3 : ℚ) * initial_birds = 8 →
  initial_birds = 60 := by
sorry

end bird_cage_problem_l2022_202246


namespace exactly_one_of_each_survives_l2022_202255

-- Define the number of trees of each type
def num_trees_A : ℕ := 2
def num_trees_B : ℕ := 2

-- Define the survival rates
def survival_rate_A : ℚ := 2/3
def survival_rate_B : ℚ := 1/2

-- Define the probability of exactly one tree of type A surviving
def prob_one_A_survives : ℚ := 
  (num_trees_A.choose 1 : ℚ) * survival_rate_A * (1 - survival_rate_A)

-- Define the probability of exactly one tree of type B surviving
def prob_one_B_survives : ℚ := 
  (num_trees_B.choose 1 : ℚ) * survival_rate_B * (1 - survival_rate_B)

-- State the theorem
theorem exactly_one_of_each_survives : 
  prob_one_A_survives * prob_one_B_survives = 2/9 := by sorry

end exactly_one_of_each_survives_l2022_202255


namespace S_max_at_9_l2022_202248

/-- An arithmetic sequence -/
def arithmetic_sequence : ℕ → ℝ := sorry

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- Conditions of the problem -/
axiom S_18_positive : S 18 > 0
axiom S_19_negative : S 19 < 0

/-- Theorem: S_n is maximum when n = 9 -/
theorem S_max_at_9 : ∀ k : ℕ, S 9 ≥ S k := by sorry

end S_max_at_9_l2022_202248


namespace cubic_equation_solution_l2022_202250

theorem cubic_equation_solution : 
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x^2 + 4*x + 4 > 0 ∧ x = 6 := by
  sorry

end cubic_equation_solution_l2022_202250


namespace total_money_l2022_202292

/-- Given that r has two-thirds of the total amount and r has $2800, 
    prove that the total amount of money p, q, and r have among themselves is $4200. -/
theorem total_money (r_share : ℚ) (r_amount : ℕ) (total : ℕ) : 
  r_share = 2/3 → r_amount = 2800 → total = r_amount * 3/2 → total = 4200 := by
  sorry

end total_money_l2022_202292


namespace intersection_of_A_and_B_l2022_202285

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l2022_202285


namespace product_of_four_numbers_l2022_202231

theorem product_of_four_numbers (E F G H : ℝ) :
  E > 0 → F > 0 → G > 0 → H > 0 →
  E + F + G + H = 50 →
  E - 3 = F + 3 ∧ E - 3 = G * 3 ∧ E - 3 = H / 3 →
  E * F * G * H = 7461.9140625 := by
sorry

end product_of_four_numbers_l2022_202231


namespace prob_five_odd_in_seven_rolls_l2022_202263

/-- The probability of getting an odd number on a single roll of a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of rolls -/
def num_rolls : ℕ := 7

/-- The number of successful rolls (odd numbers) we want -/
def num_success : ℕ := 5

/-- The probability of getting exactly 5 odd numbers in 7 rolls of a fair 6-sided die -/
theorem prob_five_odd_in_seven_rolls :
  (Nat.choose num_rolls num_success : ℚ) * prob_odd ^ num_success * (1 - prob_odd) ^ (num_rolls - num_success) = 21/128 :=
sorry

end prob_five_odd_in_seven_rolls_l2022_202263


namespace original_ratio_l2022_202233

theorem original_ratio (x y : ℕ) (h1 : y = 16) (h2 : x + 12 = y) :
  ∃ (a b : ℕ), a = 1 ∧ b = 4 ∧ x * b = y * a :=
by sorry

end original_ratio_l2022_202233


namespace unique_m_exists_l2022_202234

theorem unique_m_exists : ∃! m : ℤ,
  30 ≤ m ∧ m ≤ 80 ∧
  ∃ k : ℤ, m = 6 * k ∧
  m % 8 = 2 ∧
  m % 5 = 2 ∧
  m = 42 := by sorry

end unique_m_exists_l2022_202234


namespace subtract_fraction_from_decimal_l2022_202295

theorem subtract_fraction_from_decimal : 7.31 - (1 / 5 : ℚ) = 7.11 := by sorry

end subtract_fraction_from_decimal_l2022_202295


namespace arithmetic_sequence_eighth_term_l2022_202251

/-- Given an arithmetic sequence {aₙ} where a₁ = 1 and the common difference d = 2,
    prove that a₈ = 15. -/
theorem arithmetic_sequence_eighth_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = 2) →  -- Common difference is 2
    a 1 = 1 →                    -- First term is 1
    a 8 = 15 := by
  sorry

end arithmetic_sequence_eighth_term_l2022_202251


namespace vector_computation_l2022_202294

def c : Fin 3 → ℝ := ![(-3), 5, 2]
def d : Fin 3 → ℝ := ![5, (-1), 3]

theorem vector_computation :
  (2 • c - 5 • d + c) = ![(-34), 20, (-9)] := by sorry

end vector_computation_l2022_202294


namespace intersection_at_one_point_l2022_202276

theorem intersection_at_one_point (c : ℝ) : 
  (∃! x : ℝ, c * x^2 - 5 * x + 3 = 2 * x + 5) ↔ c = -49/8 := by
sorry

end intersection_at_one_point_l2022_202276


namespace A_subseteq_C_l2022_202229

-- Define the universe
def U : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define set A
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 = 0}

-- Define set C
def C : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem A_subseteq_C : C ⊆ A := by sorry

end A_subseteq_C_l2022_202229


namespace function_properties_l2022_202218

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define g' as the derivative of g
variable (g' : ℝ → ℝ)
variable (h : ∀ x, HasDerivAt g (g' x) x)

-- Define the conditions
variable (cond1 : ∀ x, f x + g' x - 10 = 0)
variable (cond2 : ∀ x, f x - g' (4 - x) - 10 = 0)
variable (cond3 : ∀ x, g x = g (-x))  -- g is an even function

-- Theorem statement
theorem function_properties :
  (f 1 + f 3 = 20) ∧ (f 4 = 10) ∧ (f 2022 = 10) :=
by sorry

end function_properties_l2022_202218


namespace isaac_number_problem_l2022_202201

theorem isaac_number_problem (a b : ℤ) : 
  (2 * a + 3 * b = 100) → 
  ((a = 28 ∨ b = 28) → (a = 8 ∨ b = 8)) :=
by sorry

end isaac_number_problem_l2022_202201


namespace trapezoid_height_l2022_202277

/-- Given S = (1/2)(a+b)h and a+b ≠ 0, prove that h = 2S / (a+b) -/
theorem trapezoid_height (a b S h : ℝ) (h_eq : S = (1/2) * (a + b) * h) (h_ne_zero : a + b ≠ 0) :
  h = 2 * S / (a + b) := by
  sorry

end trapezoid_height_l2022_202277


namespace michelle_sandwiches_l2022_202224

theorem michelle_sandwiches (total : ℕ) (given : ℕ) (kept : ℕ) (remaining : ℕ) : 
  total = 20 → 
  given = 4 → 
  kept = 2 * given → 
  remaining = total - given - kept → 
  remaining = 8 := by
sorry

end michelle_sandwiches_l2022_202224


namespace margin_in_terms_of_selling_price_l2022_202214

theorem margin_in_terms_of_selling_price (n : ℕ) (C S M : ℝ) 
  (h_n : n > 0)
  (h_margin : M = (1/2) * (S - (1/n) * C))
  (h_cost : C = S - M) :
  M = ((n - 1) / (2 * n - 1)) * S := by
sorry

end margin_in_terms_of_selling_price_l2022_202214


namespace smallest_of_three_consecutive_odds_l2022_202204

theorem smallest_of_three_consecutive_odds (x y z : ℤ) : 
  (∃ k : ℤ, x = 2*k + 1) →  -- x is odd
  y = x + 2 →               -- y is the next consecutive odd number
  z = y + 2 →               -- z is the next consecutive odd number after y
  x + y + z = 69 →          -- their sum is 69
  x = 21 :=                 -- the smallest number (x) is 21
by
  sorry

end smallest_of_three_consecutive_odds_l2022_202204


namespace quadrilateral_area_is_28_l2022_202225

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first small triangle -/
  area1 : ℝ
  /-- Area of the second small triangle -/
  area2 : ℝ
  /-- Area of the third small triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ

/-- The theorem stating that if the areas of the three triangles are 4, 8, and 8,
    then the area of the quadrilateral is 28 -/
theorem quadrilateral_area_is_28 (t : PartitionedTriangle) 
  (h1 : t.area1 = 4) 
  (h2 : t.area2 = 8) 
  (h3 : t.area3 = 8) : 
  t.areaQuad = 28 := by
  sorry


end quadrilateral_area_is_28_l2022_202225


namespace cloth_sale_profit_per_meter_l2022_202247

/-- Calculates the profit per meter of cloth given the total length sold,
    total selling price, and cost price per meter. -/
def profit_per_meter (total_length : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  (total_selling_price - total_length * cost_price_per_meter) / total_length

/-- Proves that for the given cloth sale, the profit per meter is 25 rupees. -/
theorem cloth_sale_profit_per_meter :
  profit_per_meter 85 8925 80 = 25 := by
  sorry

end cloth_sale_profit_per_meter_l2022_202247


namespace periodic_scaled_function_l2022_202284

-- Define a real-valued function with period T
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Define F(x) = f(αx)
def F (f : ℝ → ℝ) (α : ℝ) (x : ℝ) : ℝ := f (α * x)

-- Theorem statement
theorem periodic_scaled_function
  (f : ℝ → ℝ) (T α : ℝ) (h_periodic : is_periodic f T) (h_pos : α > 0) :
  is_periodic (F f α) (T / α) :=
sorry

end periodic_scaled_function_l2022_202284


namespace prob_odd_after_removal_is_11_21_l2022_202227

/-- A standard die with faces numbered 1 to 6 -/
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- Total number of dots on a standard die -/
def totalDots : ℕ := standardDie.sum id

/-- Probability of removing a dot from a specific face -/
def probRemoveDot (face : ℕ) : ℚ := face / totalDots

/-- Probability of rolling an odd number after removing a dot -/
def probOddAfterRemoval : ℚ :=
  (1 / 6 * (probRemoveDot 2 + probRemoveDot 4 + probRemoveDot 6)) +
  (1 / 3 * (probRemoveDot 1 + probRemoveDot 3 + probRemoveDot 5))

theorem prob_odd_after_removal_is_11_21 : probOddAfterRemoval = 11 / 21 := by
  sorry

end prob_odd_after_removal_is_11_21_l2022_202227


namespace range_of_a_l2022_202245

theorem range_of_a (x a : ℝ) : 
  (∀ x, x^2 + 2*x - 3 ≤ 0 → x ≤ a) ∧ 
  (∃ x, x^2 + 2*x - 3 ≤ 0 ∧ x > a) →
  a ≥ 1 := by
sorry

end range_of_a_l2022_202245


namespace secondPlayerCanEnsureDivisibilityFor60_secondPlayerCannotEnsureDivisibilityFor14_l2022_202274

/-- Represents a strategy for the second player to choose digits -/
def Strategy := Nat → Nat → Nat

/-- Checks if a list of digits is divisible by 9 -/
def isDivisibleBy9 (digits : List Nat) : Prop :=
  (digits.sum % 9) = 0

/-- Generates all possible sequences of digits for the first player -/
def firstPlayerSequences (n : Nat) : List (List Nat) :=
  sorry

/-- Applies the second player's strategy to the first player's sequence -/
def applyStrategy (firstPlayerSeq : List Nat) (strategy : Strategy) : List Nat :=
  sorry

theorem secondPlayerCanEnsureDivisibilityFor60 :
  ∃ (strategy : Strategy),
    ∀ (firstPlayerSeq : List Nat),
      firstPlayerSeq.length = 30 →
      firstPlayerSeq.all (λ d => d ≥ 1 ∧ d ≤ 5) →
      isDivisibleBy9 (applyStrategy firstPlayerSeq strategy) :=
sorry

theorem secondPlayerCannotEnsureDivisibilityFor14 :
  ∀ (strategy : Strategy),
    ∃ (firstPlayerSeq : List Nat),
      firstPlayerSeq.length = 7 →
      firstPlayerSeq.all (λ d => d ≥ 1 ∧ d ≤ 5) →
      ¬isDivisibleBy9 (applyStrategy firstPlayerSeq strategy) :=
sorry

end secondPlayerCanEnsureDivisibilityFor60_secondPlayerCannotEnsureDivisibilityFor14_l2022_202274


namespace intersection_of_A_and_B_l2022_202200

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {-3, -1, 1, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1, 3} := by
  sorry

end intersection_of_A_and_B_l2022_202200


namespace geometric_sequence_ratio_sum_l2022_202271

theorem geometric_sequence_ratio_sum (k p r : ℝ) (hk : k ≠ 0) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  k * p^2 - k * r^2 = 5 * (k * p - k * r) → p + r = 5 := by
  sorry

end geometric_sequence_ratio_sum_l2022_202271


namespace negation_of_existential_proposition_l2022_202268

theorem negation_of_existential_proposition :
  (¬ ∃ n : ℕ, 2^n < 1000) ↔ (∀ n : ℕ, 2^n ≥ 1000) := by sorry

end negation_of_existential_proposition_l2022_202268


namespace greatest_divisor_with_remainders_l2022_202280

theorem greatest_divisor_with_remainders (d : ℕ) : d > 0 ∧ 
  (∃ q1 : ℤ, 4351 = d * q1 + 8) ∧ 
  (∃ r1 : ℤ, 5161 = d * r1 + 10) ∧ 
  (∀ n : ℕ, n > d → 
    (∃ q2 : ℤ, 4351 = n * q2 + 8) ∧ 
    (∃ r2 : ℤ, 5161 = n * r2 + 10) → n = d) → 
  d = 1 := by
sorry

end greatest_divisor_with_remainders_l2022_202280


namespace fencing_cost_l2022_202293

/-- Given a rectangular field with sides in ratio 3:4 and area 9408 sq. m,
    prove that the cost of fencing at 25 paise per metre is 98 rupees. -/
theorem fencing_cost (length width : ℝ) (area perimeter cost_per_metre total_cost : ℝ) : 
  length / width = 3 / 4 →
  area = 9408 →
  area = length * width →
  perimeter = 2 * (length + width) →
  cost_per_metre = 25 / 100 →
  total_cost = perimeter * cost_per_metre →
  total_cost = 98 := by
sorry

end fencing_cost_l2022_202293


namespace original_cat_count_l2022_202283

theorem original_cat_count (first_relocation second_relocation final_count : ℕ) 
  (h1 : first_relocation = 600)
  (h2 : second_relocation = (original_count - first_relocation) / 2)
  (h3 : final_count = 600)
  (h4 : final_count = original_count - first_relocation - second_relocation) :
  original_count = 1800 :=
by sorry

#check original_cat_count

end original_cat_count_l2022_202283


namespace movie_profit_calculation_l2022_202244

def movie_profit (actor_cost food_cost_per_person num_people equipment_rental_factor selling_price : ℚ) : ℚ :=
  let food_cost := food_cost_per_person * num_people
  let total_food_and_actors := actor_cost + food_cost
  let equipment_cost := equipment_rental_factor * total_food_and_actors
  let total_cost := actor_cost + food_cost + equipment_cost
  selling_price - total_cost

theorem movie_profit_calculation :
  movie_profit 1200 3 50 2 10000 = 5950 :=
by sorry

end movie_profit_calculation_l2022_202244


namespace max_y_value_l2022_202272

theorem max_y_value (x y : ℝ) (h : x^2 + y^2 = 10*x + 60*y) :
  y ≤ 30 + 5 * Real.sqrt 37 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 10*x₀ + 60*y₀ ∧ y₀ = 30 + 5 * Real.sqrt 37 := by
  sorry

end max_y_value_l2022_202272


namespace bong_paint_time_l2022_202261

def jay_time : ℝ := 2
def combined_time : ℝ := 1.2

theorem bong_paint_time :
  ∀ bong_time : ℝ,
  (1 / jay_time + 1 / bong_time = 1 / combined_time) →
  bong_time = 3 := by
sorry

end bong_paint_time_l2022_202261


namespace base_nine_representation_l2022_202269

theorem base_nine_representation (b : ℕ) : 
  (777 : ℕ) = 1 * b^3 + 0 * b^2 + 5 * b^1 + 3 * b^0 ∧ 
  b > 1 ∧ 
  b^3 ≤ 777 ∧ 
  777 < b^4 ∧
  (∃ (A C : ℕ), A ≠ C ∧ A < b ∧ C < b ∧ 
    777 = A * b^3 + C * b^2 + A * b^1 + C * b^0) →
  b = 9 := by
sorry

end base_nine_representation_l2022_202269


namespace intersection_when_a_neg_two_subset_condition_l2022_202299

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | (x - (a + 5)) / (x - a) > 0}

-- Theorem for part 1
theorem intersection_when_a_neg_two :
  A ∩ B (-2) = {x | 3 < x ∧ x ≤ 4} := by sorry

-- Theorem for part 2
theorem subset_condition (a : ℝ) :
  A ⊆ B a ↔ a < -6 ∨ a > 4 := by sorry

end intersection_when_a_neg_two_subset_condition_l2022_202299


namespace winnie_the_pooh_escalator_steps_l2022_202286

theorem winnie_the_pooh_escalator_steps :
  ∀ (u v L : ℝ),
    u > 0 →
    v > 0 →
    L > 0 →
    (L * u) / (u + v) = 55 →
    (L * u) / (u - v) = 1155 →
    L = 105 :=
by
  sorry

end winnie_the_pooh_escalator_steps_l2022_202286


namespace inequality_solution_set_l2022_202207

theorem inequality_solution_set (x : ℝ) :
  (8 * x^3 - 6 * x^2 + 5 * x - 1 < 4) ↔ (x < 1/2) := by
  sorry

end inequality_solution_set_l2022_202207


namespace ones_digit_of_34_power_power_4_cycle_seventeen_power_odd_main_theorem_l2022_202257

theorem ones_digit_of_34_power (n : ℕ) : n > 0 → (34^n) % 10 = (4^n) % 10 := by sorry

theorem power_4_cycle : ∀ n : ℕ, n > 0 → (4^n) % 10 = if n % 2 = 1 then 4 else 6 := by sorry

theorem seventeen_power_odd : (17^17) % 2 = 1 := by sorry

theorem main_theorem : (34^(34*(17^17))) % 10 = 4 := by sorry

end ones_digit_of_34_power_power_4_cycle_seventeen_power_odd_main_theorem_l2022_202257


namespace symmetric_lines_symmetric_line_equation_l2022_202202

/-- Given two lines in the 2D plane and a point, this theorem states that these lines are symmetric with respect to the given point. -/
theorem symmetric_lines (x y : ℝ) : 
  (2 * x + 3 * y - 6 = 0) ↔ (2 * (2 - x) + 3 * (-2 - y) - 6 = 0) := by
  sorry

/-- The equation of the line symmetric to 2x + 3y - 6 = 0 with respect to the point (1, -1) is 2x + 3y + 8 = 0. -/
theorem symmetric_line_equation : 
  ∀ x y : ℝ, (2 * x + 3 * y - 6 = 0) ↔ (2 * ((2 - x) - 1) + 3 * ((-2 - y) - (-1)) + 8 = 0) := by
  sorry

end symmetric_lines_symmetric_line_equation_l2022_202202


namespace four_digit_integer_problem_l2022_202256

theorem four_digit_integer_problem (n : ℕ) (a b c d : ℕ) :
  n = a * 1000 + b * 100 + c * 10 + d →
  a ≥ 1 →
  a ≤ 9 →
  b ≤ 9 →
  c ≤ 9 →
  d ≤ 9 →
  a + b + c + d = 16 →
  b + c = 10 →
  a - d = 2 →
  n % 11 = 0 →
  n = 4462 := by
sorry

end four_digit_integer_problem_l2022_202256


namespace short_trees_after_planting_verify_total_short_trees_l2022_202297

/-- The number of short trees in the park after planting -/
def total_short_trees (current_short_trees new_short_trees : ℕ) : ℕ :=
  current_short_trees + new_short_trees

/-- Theorem: The total number of short trees after planting is the sum of current and new short trees -/
theorem short_trees_after_planting 
  (current_short_trees : ℕ) (new_short_trees : ℕ) :
  total_short_trees current_short_trees new_short_trees = current_short_trees + new_short_trees :=
by sorry

/-- The correct number of short trees after planting, given the problem conditions -/
def correct_total : ℕ := 98

/-- Theorem: The total number of short trees after planting, given the problem conditions, is 98 -/
theorem verify_total_short_trees :
  total_short_trees 41 57 = correct_total :=
by sorry

end short_trees_after_planting_verify_total_short_trees_l2022_202297


namespace arithmetic_evaluation_l2022_202220

theorem arithmetic_evaluation : 5 + 2 * (8 - 3) = 15 := by
  sorry

end arithmetic_evaluation_l2022_202220


namespace sphere_volume_in_cube_l2022_202237

/-- The volume of a sphere inscribed in a cube with surface area 24 cm² is (4/3)π cm³ -/
theorem sphere_volume_in_cube (cube_surface_area : ℝ) (sphere_volume : ℝ) : 
  cube_surface_area = 24 →
  sphere_volume = (4/3) * Real.pi := by
  sorry

#check sphere_volume_in_cube

end sphere_volume_in_cube_l2022_202237


namespace shaded_area_calculation_l2022_202254

/-- The area of the shaded region formed by a sector of a circle and an equilateral triangle -/
theorem shaded_area_calculation (r : ℝ) (θ : ℝ) (a : ℝ) (h1 : r = 12) (h2 : θ = 112) (h3 : a = 12) :
  let sector_area := (θ / 360) * π * r^2
  let triangle_area := (Real.sqrt 3 / 4) * a^2
  abs ((sector_area - triangle_area) - 78.0211) < 0.0001 := by
sorry

end shaded_area_calculation_l2022_202254


namespace approx_cube_root_2370_l2022_202275

-- Define the approximation relation
def approx (x y : ℝ) := ∃ ε > 0, |x - y| < ε

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem approx_cube_root_2370 (h : approx (cubeRoot 2.37) 1.333) :
  approx (cubeRoot 2370) 13.33 := by
  sorry

end approx_cube_root_2370_l2022_202275


namespace arcsin_sin_eq_x_div_3_solutions_l2022_202217

theorem arcsin_sin_eq_x_div_3_solutions (x : ℝ) :
  -((3 * π) / 2) ≤ x ∧ x ≤ (3 * π) / 2 →
  (Real.arcsin (Real.sin x) = x / 3) ↔ 
  x ∈ ({-3*π, -2*π, -π, 0, π, 2*π, 3*π} : Set ℝ) :=
by sorry

end arcsin_sin_eq_x_div_3_solutions_l2022_202217


namespace modular_arithmetic_problem_l2022_202290

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (4 * a) % 65 = 1 ∧ 
                (13 * b) % 65 = 1 ∧ 
                (3 * a + 12 * b) % 65 = 42 :=
by sorry

end modular_arithmetic_problem_l2022_202290


namespace rope_triangle_probability_l2022_202259

/-- The probability of forming a triangle from three rope segments --/
theorem rope_triangle_probability (L : ℝ) (h : L > 0) : 
  (∫ x in (0)..(L/2), if x > L/4 then 1 else 0) / (L/2) = 1/2 := by
  sorry

end rope_triangle_probability_l2022_202259


namespace bridgets_score_l2022_202212

theorem bridgets_score (total_students : ℕ) (students_before : ℕ) (avg_before : ℚ) (avg_after : ℚ) 
  (h1 : total_students = 18)
  (h2 : students_before = 17)
  (h3 : avg_before = 76)
  (h4 : avg_after = 78) :
  (total_students : ℚ) * avg_after - (students_before : ℚ) * avg_before = 112 := by
  sorry

end bridgets_score_l2022_202212


namespace min_toothpicks_to_remove_correct_l2022_202236

/-- Represents a figure made of toothpicks and triangles -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ
  upward_side_length : ℕ
  downward_side_length : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ :=
  figure.upward_triangles * figure.upward_side_length

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_to_remove_correct (figure : ToothpickFigure) 
  (h1 : figure.total_toothpicks = 40)
  (h2 : figure.upward_triangles = 10)
  (h3 : figure.downward_triangles = 8)
  (h4 : figure.upward_side_length = 2)
  (h5 : figure.downward_side_length = 1) :
  min_toothpicks_to_remove figure = 20 := by
  sorry

#eval min_toothpicks_to_remove {
  total_toothpicks := 40,
  upward_triangles := 10,
  downward_triangles := 8,
  upward_side_length := 2,
  downward_side_length := 1
}

end min_toothpicks_to_remove_correct_l2022_202236


namespace literature_club_students_l2022_202264

theorem literature_club_students (total : ℕ) (english : ℕ) (french : ℕ) (both : ℕ) 
  (h_total : total = 120)
  (h_english : english = 72)
  (h_french : french = 52)
  (h_both : both = 12) :
  total - (english + french - both) = 8 := by
  sorry

end literature_club_students_l2022_202264


namespace odd_even_sum_difference_l2022_202230

/-- The sum of the first n odd natural numbers -/
def sum_odd (n : ℕ) : ℕ := n^2

/-- The sum of the first n even natural numbers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- The number of odd terms from 1 to 2023 -/
def n_odd : ℕ := (2023 - 1) / 2 + 1

/-- The number of even terms from 2 to 2022 -/
def n_even : ℕ := (2022 - 2) / 2 + 1

theorem odd_even_sum_difference : 
  sum_odd n_odd - sum_even n_even = 22 := by
  sorry

end odd_even_sum_difference_l2022_202230


namespace paint_wall_theorem_l2022_202216

/-- The number of people needed to paint a wall in a given time, assuming a constant rate of painting. -/
def people_needed (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) : ℕ :=
  (initial_people * initial_time) / new_time

/-- The additional number of people needed to paint a wall in a shorter time. -/
def additional_people_needed (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) : ℕ :=
  (people_needed initial_people initial_time new_time) - initial_people

theorem paint_wall_theorem (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) 
  (h1 : initial_people = 8) 
  (h2 : initial_time = 3) 
  (h3 : new_time = 2) :
  additional_people_needed initial_people initial_time new_time = 4 := by
  sorry

#check paint_wall_theorem

end paint_wall_theorem_l2022_202216


namespace nested_expression_evaluation_l2022_202208

theorem nested_expression_evaluation : (5*(5*(5*(5+1)+1)+1)+1) = 781 := by
  sorry

end nested_expression_evaluation_l2022_202208


namespace car_speed_time_relationship_car_q_graph_representation_l2022_202206

/-- Represents a car's travel characteristics -/
structure CarTravel where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The theorem stating the relationship between Car P and Car Q's travel characteristics -/
theorem car_speed_time_relationship 
  (p q : CarTravel) 
  (h1 : p.distance = q.distance) 
  (h2 : q.speed = 3 * p.speed) : 
  q.time = p.time / 3 := by
sorry

/-- The theorem proving the graphical representation of Car Q's travel -/
theorem car_q_graph_representation 
  (p q : CarTravel) 
  (h1 : p.distance = q.distance) 
  (h2 : q.speed = 3 * p.speed) : 
  q.speed = 3 * p.speed ∧ q.time = p.time / 3 := by
sorry

end car_speed_time_relationship_car_q_graph_representation_l2022_202206


namespace afternoon_eggs_count_l2022_202240

def initial_eggs : ℕ := 20
def morning_eggs : ℕ := 4
def remaining_eggs : ℕ := 13

theorem afternoon_eggs_count : initial_eggs - morning_eggs - remaining_eggs = 3 := by
  sorry

end afternoon_eggs_count_l2022_202240


namespace three_numbers_sum_l2022_202211

theorem three_numbers_sum (A B C : ℤ) : 
  A + B + C = 180 ∧ B = 3*C - 2 ∧ A = 2*C + 8 → A = 66 ∧ B = 85 ∧ C = 29 := by
  sorry

end three_numbers_sum_l2022_202211


namespace mice_meet_in_six_days_l2022_202221

/-- The thickness of the wall in feet -/
def wall_thickness : ℚ := 64 + 31/32

/-- The distance burrowed by both mice after n days -/
def total_distance (n : ℕ) : ℚ := 2^n - 1/(2^(n-1)) + 1

/-- The number of days it takes for the mice to meet -/
def days_to_meet : ℕ := 6

/-- Theorem stating that the mice meet after 6 days -/
theorem mice_meet_in_six_days :
  total_distance days_to_meet = wall_thickness :=
sorry

end mice_meet_in_six_days_l2022_202221


namespace trigonometric_identities_l2022_202291

open Real

theorem trigonometric_identities (α : ℝ) (h : 3 * sin α - 2 * cos α = 0) :
  ((cos α - sin α) / (cos α + sin α) + (cos α + sin α) / (cos α - sin α) = 5) ∧
  (sin α ^ 2 - 2 * sin α * cos α + 4 * cos α ^ 2 = 28 / 13) := by
  sorry

end trigonometric_identities_l2022_202291


namespace cryptarithmetic_puzzle_l2022_202235

theorem cryptarithmetic_puzzle (T W O F U R : ℕ) : 
  (T = 9) →
  (O % 2 = 1) →
  (T + T + W + W = F * 1000 + O * 100 + U * 10 + R) →
  (T ≠ W ∧ T ≠ O ∧ T ≠ F ∧ T ≠ U ∧ T ≠ R ∧
   W ≠ O ∧ W ≠ F ∧ W ≠ U ∧ W ≠ R ∧
   O ≠ F ∧ O ≠ U ∧ O ≠ R ∧
   F ≠ U ∧ F ≠ R ∧
   U ≠ R) →
  (T < 10 ∧ W < 10 ∧ O < 10 ∧ F < 10 ∧ U < 10 ∧ R < 10) →
  W = 1 := by
sorry

end cryptarithmetic_puzzle_l2022_202235


namespace cos_arithmetic_sequence_product_l2022_202209

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

def S (a₁ : ℝ) : Set ℝ := {x | ∃ n : ℕ+, x = Real.cos (arithmetic_sequence a₁ (2 * Real.pi / 3) n)}

theorem cos_arithmetic_sequence_product (a₁ : ℝ) :
  ∃ a b : ℝ, S a₁ = {a, b} → a * b = -1/2 := by sorry

end cos_arithmetic_sequence_product_l2022_202209


namespace science_score_calculation_l2022_202226

def average_score : ℝ := 95
def chinese_score : ℝ := 90
def math_score : ℝ := 98

theorem science_score_calculation :
  ∃ (science_score : ℝ),
    (chinese_score + math_score + science_score) / 3 = average_score ∧
    science_score = 97 := by sorry

end science_score_calculation_l2022_202226


namespace collinear_points_iff_k_eq_neg_ten_l2022_202241

/-- Three points in R² are collinear if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

/-- The theorem states that the points (1, -2), (3, k), and (6, 2k - 2) are collinear 
    if and only if k = -10. -/
theorem collinear_points_iff_k_eq_neg_ten :
  ∀ k : ℝ, collinear (1, -2) (3, k) (6, 2*k - 2) ↔ k = -10 := by
  sorry

end collinear_points_iff_k_eq_neg_ten_l2022_202241


namespace happy_family_cows_count_cow_ratio_l2022_202262

/-- The number of cows We the People has -/
def we_the_people_cows : ℕ := 17

/-- The total number of cows when both groups are together -/
def total_cows : ℕ := 70

/-- The number of cows Happy Good Healthy Family has -/
def happy_family_cows : ℕ := total_cows - we_the_people_cows

theorem happy_family_cows_count : happy_family_cows = 53 := by
  sorry

theorem cow_ratio : 
  (happy_family_cows : ℚ) / (we_the_people_cows : ℚ) = 53 / 17 := by
  sorry

end happy_family_cows_count_cow_ratio_l2022_202262


namespace certain_number_value_l2022_202265

theorem certain_number_value (t b c : ℝ) :
  (t + b + c + 14 + 15) / 5 = 12 ∧ (t + b + c + 29) / 4 = 15 → 14 = 14 := by
  sorry

end certain_number_value_l2022_202265


namespace chess_team_boys_l2022_202296

/-- Represents the number of boys on a chess team --/
def num_boys (total : ℕ) (attendees : ℕ) : ℕ :=
  total - 2 * (total - attendees)

/-- Theorem stating the number of boys on the chess team --/
theorem chess_team_boys (total : ℕ) (attendees : ℕ) 
  (h_total : total = 30)
  (h_attendees : attendees = 18)
  (h_attendance : ∃ (girls : ℕ), girls + (total - girls) = total ∧ 
                                  girls / 3 + (total - girls) = attendees) :
  num_boys total attendees = 12 := by
sorry

#eval num_boys 30 18  -- Should output 12

end chess_team_boys_l2022_202296


namespace election_win_percentage_l2022_202223

/-- The minimum percentage of votes needed to win an election --/
def min_win_percentage (total_votes : ℕ) (geoff_percentage : ℚ) (additional_votes_needed : ℕ) : ℚ :=
  ((geoff_percentage * total_votes + additional_votes_needed) / total_votes) * 100

/-- Theorem stating the minimum percentage of votes needed to win the election --/
theorem election_win_percentage :
  let total_votes : ℕ := 6000
  let geoff_percentage : ℚ := 1/200
  let additional_votes_needed : ℕ := 3000
  min_win_percentage total_votes geoff_percentage additional_votes_needed = 101/2 := by
  sorry

end election_win_percentage_l2022_202223


namespace final_algae_count_l2022_202279

/-- The number of algae plants in Milford Lake -/
def algae_count : ℕ → ℕ
| 0 => 809  -- Original count
| (n + 1) => algae_count n + 2454  -- Increase

theorem final_algae_count : algae_count 1 = 3263 := by
  sorry

end final_algae_count_l2022_202279


namespace tangent_line_equation_l2022_202281

theorem tangent_line_equation (x y : ℝ) :
  x < 0 ∧ y > 0 ∧  -- P is in the second quadrant
  y = x^3 - 10*x + 3 ∧  -- P is on the curve
  3*x^2 - 10 = 2  -- Slope of tangent line is 2
  →
  ∃ (a b : ℝ), a = 2 ∧ b = 19 ∧ ∀ (x' y' : ℝ), y' = a*x' + b  -- Equation of tangent line
  :=
by sorry

end tangent_line_equation_l2022_202281


namespace lcm_18_24_30_l2022_202222

theorem lcm_18_24_30 : Nat.lcm (Nat.lcm 18 24) 30 = 360 := by
  sorry

end lcm_18_24_30_l2022_202222


namespace ben_win_probability_l2022_202215

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 3/7) 
  (h2 : ∀ (tie_prob : ℚ), tie_prob = 0) : 
  1 - lose_prob = 4/7 := by
sorry

end ben_win_probability_l2022_202215


namespace average_side_lengths_of_squares_l2022_202289

theorem average_side_lengths_of_squares (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ = 25) (h₂ : a₂ = 36) (h₃ : a₃ = 64) (h₄ : a₄ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃ + Real.sqrt a₄) / 4 = 7.75 := by
  sorry

end average_side_lengths_of_squares_l2022_202289


namespace prime_iff_sum_four_integers_l2022_202249

theorem prime_iff_sum_four_integers (n : ℕ) (h : n ≥ 5) :
  Nat.Prime n ↔ ∀ (a b c d : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → n = a + b + c + d → a * b ≠ c * d := by
  sorry

end prime_iff_sum_four_integers_l2022_202249


namespace collinear_points_k_value_l2022_202242

/-- Given three points A, B, and C in 2D space, this function checks if they are collinear --/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  AB.1 * BC.2 = AB.2 * BC.1

/-- Theorem stating that if A(k, 12), B(4, 5), and C(10, k) are collinear, then k = 11 or k = -2 --/
theorem collinear_points_k_value (k : ℝ) :
  are_collinear (k, 12) (4, 5) (10, k) → k = 11 ∨ k = -2 := by
  sorry


end collinear_points_k_value_l2022_202242
