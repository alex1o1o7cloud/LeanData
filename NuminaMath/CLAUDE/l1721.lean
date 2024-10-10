import Mathlib

namespace chord_slope_l1721_172145

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- Definition of the midpoint of the chord -/
def is_midpoint (x y : ℝ) : Prop := x = 4 ∧ y = 2

/-- Theorem: The slope of the chord is -1/2 -/
theorem chord_slope (x1 y1 x2 y2 : ℝ) :
  is_on_ellipse x1 y1 → is_on_ellipse x2 y2 →
  is_midpoint ((x1 + x2) / 2) ((y1 + y2) / 2) →
  (y2 - y1) / (x2 - x1) = -1/2 := by sorry

end chord_slope_l1721_172145


namespace f_of_f_3_l1721_172110

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 1

theorem f_of_f_3 : f (f 3) = 1429 := by
  sorry

end f_of_f_3_l1721_172110


namespace park_entrance_cost_is_5_l1721_172163

def park_entrance_cost : ℝ → Prop :=
  λ cost =>
    let num_children := 4
    let num_parents := 2
    let num_grandmother := 1
    let attraction_cost_kid := 2
    let attraction_cost_adult := 4
    let total_paid := 55
    let total_family_members := num_children + num_parents + num_grandmother
    let total_attraction_cost := num_children * attraction_cost_kid + 
                                 (num_parents + num_grandmother) * attraction_cost_adult
    total_paid = total_family_members * cost + total_attraction_cost

theorem park_entrance_cost_is_5 : park_entrance_cost 5 := by
  sorry

end park_entrance_cost_is_5_l1721_172163


namespace find_novel_cost_l1721_172133

def novel_cost (initial_amount lunch_cost remaining_amount : ℚ) : Prop :=
  ∃ (novel_cost : ℚ),
    novel_cost > 0 ∧
    lunch_cost = 2 * novel_cost ∧
    initial_amount - (novel_cost + lunch_cost) = remaining_amount

theorem find_novel_cost :
  novel_cost 50 (2 * 7) 29 :=
sorry

end find_novel_cost_l1721_172133


namespace tangent_line_at_pi_l1721_172198

/-- The equation of the tangent line to y = x sin x at (π, 0) is y = -πx + π² -/
theorem tangent_line_at_pi (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => t * Real.sin t
  let f' : ℝ → ℝ := λ t => Real.sin t + t * Real.cos t
  let tangent_line : ℝ → ℝ := λ t => -π * t + π^2
  (∀ t, HasDerivAt f (f' t) t) →
  HasDerivAt f (f' π) π →
  tangent_line π = f π →
  tangent_line = λ t => -π * t + π^2 := by
sorry


end tangent_line_at_pi_l1721_172198


namespace assistant_coaches_average_age_l1721_172156

/-- The average age of assistant coaches in a sports club --/
theorem assistant_coaches_average_age 
  (total_members : ℕ) 
  (overall_average : ℕ) 
  (girls_count : ℕ) 
  (girls_average : ℕ) 
  (boys_count : ℕ) 
  (boys_average : ℕ) 
  (head_coaches_count : ℕ) 
  (head_coaches_average : ℕ) 
  (assistant_coaches_count : ℕ) 
  (h_total : total_members = 50)
  (h_overall : overall_average = 22)
  (h_girls : girls_count = 30)
  (h_girls_avg : girls_average = 18)
  (h_boys : boys_count = 15)
  (h_boys_avg : boys_average = 20)
  (h_head_coaches : head_coaches_count = 3)
  (h_head_coaches_avg : head_coaches_average = 30)
  (h_assistant_coaches : assistant_coaches_count = 2)
  (h_coaches_total : head_coaches_count + assistant_coaches_count = 5) :
  (total_members * overall_average - 
   girls_count * girls_average - 
   boys_count * boys_average - 
   head_coaches_count * head_coaches_average) / assistant_coaches_count = 85 := by
sorry


end assistant_coaches_average_age_l1721_172156


namespace sum_of_coefficients_l1721_172129

-- Define the polynomial
def p (x : ℝ) : ℝ := 3 * (x^8 - 2*x^5 + 4*x^3 - 6) - 5 * (x^4 - 3*x^2 + 2) + 2 * (x^6 + 5*x - 8)

-- Theorem: The sum of the coefficients of p is -3
theorem sum_of_coefficients : p 1 = -3 := by
  sorry

end sum_of_coefficients_l1721_172129


namespace sqrt_x_minus_one_meaningful_l1721_172105

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_meaningful_l1721_172105


namespace kiran_money_l1721_172103

/-- Given the ratios of money between Ravi, Giri, and Kiran, and Ravi's amount of money,
    prove that Kiran has $105. -/
theorem kiran_money (ravi giri kiran : ℚ) : 
  (ravi / giri = 6 / 7) →
  (giri / kiran = 6 / 15) →
  (ravi = 36) →
  (kiran = 105) := by
sorry

end kiran_money_l1721_172103


namespace rational_fraction_implies_integer_sum_squares_l1721_172166

theorem rational_fraction_implies_integer_sum_squares (a b c : ℕ+) 
  (h : ∃ (q : ℚ), (a.val : ℝ) * Real.sqrt 3 + b.val = q * ((b.val : ℝ) * Real.sqrt 3 + c.val)) :
  ∃ (n : ℤ), (a.val^2 + b.val^2 + c.val^2 : ℝ) / (a.val + b.val + c.val) = n := by
sorry

end rational_fraction_implies_integer_sum_squares_l1721_172166


namespace floor_sqrt_sum_eq_floor_sqrt_4n_plus_2_l1721_172121

theorem floor_sqrt_sum_eq_floor_sqrt_4n_plus_2 (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end floor_sqrt_sum_eq_floor_sqrt_4n_plus_2_l1721_172121


namespace fourth_term_is_two_l1721_172132

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q
  a6_eq_2 : a 6 = 2
  arithmetic_subseq : a 7 - a 5 = a 9 - a 7

/-- The fourth term of the geometric sequence is 2 -/
theorem fourth_term_is_two (seq : GeometricSequence) : seq.a 4 = 2 := by
  sorry

end fourth_term_is_two_l1721_172132


namespace max_value_relationship_l1721_172120

theorem max_value_relationship (x y : ℝ) : 
  (∀ a b : ℝ, 2005 - (a + b)^2 ≤ 2005 - (x + y)^2) → x = -y := by
  sorry

end max_value_relationship_l1721_172120


namespace solution_exists_in_interval_l1721_172106

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- State the theorem
theorem solution_exists_in_interval :
  ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ f x = 0 := by
  sorry

end solution_exists_in_interval_l1721_172106


namespace negation_of_implication_conjunction_implies_disjunction_disjunction_not_implies_conjunction_negation_of_universal_even_function_condition_l1721_172181

-- Define the propositions p and q
variable (p q : Prop)

-- Define the function f
variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- 1. Negation of implication
theorem negation_of_implication : ¬(p → q) ↔ (p ∧ ¬q) := by sorry

-- 2. Relationship between conjunction and disjunction
theorem conjunction_implies_disjunction : (p ∧ q) → (p ∨ q) := by sorry

theorem disjunction_not_implies_conjunction : ¬((p ∨ q) → (p ∧ q)) := by sorry

-- 3. Negation of universal quantifier
theorem negation_of_universal : 
  ¬(∀ x : ℝ, x > 2 → x^2 - 2*x > 0) ↔ (∃ x : ℝ, x > 2 ∧ x^2 - 2*x ≤ 0) := by sorry

-- 4. Even function condition
theorem even_function_condition : 
  (∀ x : ℝ, f x = f (-x)) → b = 0 := by sorry

end negation_of_implication_conjunction_implies_disjunction_disjunction_not_implies_conjunction_negation_of_universal_even_function_condition_l1721_172181


namespace symmetric_line_correct_l1721_172188

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Checks if a point satisfies the equation of a line -/
def on_line (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- The original line 2x - 3y + 2 = 0 -/
def original_line : Line :=
  { a := 2, b := -3, c := 2 }

/-- The symmetric line to be proven -/
def symmetric_line : Line :=
  { a := 2, b := 3, c := 2 }

theorem symmetric_line_correct :
  ∀ p : ℝ × ℝ, on_line symmetric_line p ↔ on_line original_line (reflect_x p) :=
sorry

end symmetric_line_correct_l1721_172188


namespace problem_solution_l1721_172100

theorem problem_solution : 18 * 36 + 45 * 18 - 9 * 18 = 1296 := by
  sorry

end problem_solution_l1721_172100


namespace solution_set_when_a_is_one_solution_set_for_any_a_l1721_172179

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 3*a*x + 2*a^2

-- Theorem for part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for part 2
theorem solution_set_for_any_a (a : ℝ) :
  ({x : ℝ | f a x < 0} = ∅ ∧ a = 0) ∨
  ({x : ℝ | f a x < 0} = {x : ℝ | a < x ∧ x < 2*a} ∧ a > 0) ∨
  ({x : ℝ | f a x < 0} = {x : ℝ | 2*a < x ∧ x < a} ∧ a < 0) := by sorry

end solution_set_when_a_is_one_solution_set_for_any_a_l1721_172179


namespace factors_of_81_l1721_172170

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end factors_of_81_l1721_172170


namespace product_real_parts_of_complex_equation_l1721_172125

theorem product_real_parts_of_complex_equation : ∃ (x₁ x₂ : ℂ),
  (x₁^2 - 4*x₁ = -4 - 4*I) ∧
  (x₂^2 - 4*x₂ = -4 - 4*I) ∧
  (x₁ ≠ x₂) ∧
  (Complex.re x₁ * Complex.re x₂ = 2) :=
by sorry

end product_real_parts_of_complex_equation_l1721_172125


namespace intersection_of_N_and_complement_of_M_l1721_172144

def M : Set ℝ := {x | x^2 - x - 6 ≥ 0}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

theorem intersection_of_N_and_complement_of_M :
  N ∩ (Set.univ \ M) = Set.Ioo (-2) 1 := by sorry

end intersection_of_N_and_complement_of_M_l1721_172144


namespace max_value_constrained_l1721_172122

theorem max_value_constrained (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (∃ x y z : ℝ, 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 ∧ 8 * x + 3 * y + 5 * z > 8 * a + 3 * b + 5 * c) ∨
  8 * a + 3 * b + 5 * c = 7 * Real.sqrt 2 :=
sorry

end max_value_constrained_l1721_172122


namespace paper_envelope_problem_l1721_172130

/-- 
Given that each paper envelope can contain 10 white papers and 12 paper envelopes are needed,
prove that the total number of clean white papers is 120.
-/
theorem paper_envelope_problem (papers_per_envelope : ℕ) (num_envelopes : ℕ) 
  (h1 : papers_per_envelope = 10) 
  (h2 : num_envelopes = 12) : 
  papers_per_envelope * num_envelopes = 120 := by
  sorry

end paper_envelope_problem_l1721_172130


namespace y_value_l1721_172152

theorem y_value (x y : ℝ) (h1 : x = 4) (h2 : y = 3 * x) : y = 12 := by
  sorry

end y_value_l1721_172152


namespace sum_reciprocals_and_diff_squares_l1721_172137

theorem sum_reciprocals_and_diff_squares (x y : ℝ) 
  (sum_eq : x + y = 12) 
  (prod_eq : x * y = 32) : 
  (1 / x + 1 / y = 3 / 8) ∧ (x^2 - y^2 = 48 * Real.sqrt 5) := by
  sorry

end sum_reciprocals_and_diff_squares_l1721_172137


namespace compute_fraction_power_l1721_172182

theorem compute_fraction_power : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end compute_fraction_power_l1721_172182


namespace divisibility_of_T_members_l1721_172123

/-- The set of all numbers which are the sum of the squares of four consecutive integers -/
def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

theorem divisibility_of_T_members :
  (∃ x ∈ T, 5 ∣ x) ∧ (∀ x ∈ T, ¬(7 ∣ x)) := by sorry

end divisibility_of_T_members_l1721_172123


namespace cymbal_triangle_tambourine_sync_l1721_172186

theorem cymbal_triangle_tambourine_sync (cymbal_beats : Nat) (triangle_beats : Nat) (tambourine_beats : Nat)
  (h1 : cymbal_beats = 13)
  (h2 : triangle_beats = 17)
  (h3 : tambourine_beats = 19) :
  Nat.lcm (Nat.lcm cymbal_beats triangle_beats) tambourine_beats = 4199 := by
  sorry

end cymbal_triangle_tambourine_sync_l1721_172186


namespace roxanne_change_l1721_172177

def lemonade_price : ℚ := 2
def lemonade_quantity : ℕ := 4

def sandwich_price : ℚ := 2.5
def sandwich_quantity : ℕ := 3

def watermelon_price : ℚ := 1.25
def watermelon_quantity : ℕ := 2

def chips_price : ℚ := 1.75
def chips_quantity : ℕ := 1

def cookie_price : ℚ := 0.75
def cookie_quantity : ℕ := 4

def pretzel_price : ℚ := 1
def pretzel_quantity : ℕ := 5

def salad_price : ℚ := 8
def salad_quantity : ℕ := 1

def payment : ℚ := 100

theorem roxanne_change :
  payment - (lemonade_price * lemonade_quantity +
             sandwich_price * sandwich_quantity +
             watermelon_price * watermelon_quantity +
             chips_price * chips_quantity +
             cookie_price * cookie_quantity +
             pretzel_price * pretzel_quantity +
             salad_price * salad_quantity) = 63.75 := by
  sorry

end roxanne_change_l1721_172177


namespace triangle_tangent_inequality_l1721_172140

/-- Given a triangle ABC with sides a, b, c and tangential segments x, y, z
    from vertices A, B, C to the incircle respectively, if a ≥ b ≥ c,
    then az + by + cx ≥ (a² + b² + c²)/2 ≥ ax + by + cz. -/
theorem triangle_tangent_inequality (a b c x y z : ℝ) 
    (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a = y + z) (h4 : b = x + z) (h5 : c = x + y) :
    a * z + b * y + c * x ≥ (a^2 + b^2 + c^2) / 2 ∧ 
    (a^2 + b^2 + c^2) / 2 ≥ a * x + b * y + c * z := by
  sorry


end triangle_tangent_inequality_l1721_172140


namespace henry_collection_cost_l1721_172141

/-- The amount of money Henry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total_needed : ℕ) (cost_per_figure : ℕ) : ℕ :=
  (total_needed - current) * cost_per_figure

/-- Proof that Henry needs $30 to finish his collection -/
theorem henry_collection_cost : money_needed 3 8 6 = 30 := by
  sorry

end henry_collection_cost_l1721_172141


namespace intersection_of_sets_l1721_172136

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 3, 4}
  let B : Set ℕ := {0, 1, 3}
  A ∩ B = {1, 3} := by
sorry

end intersection_of_sets_l1721_172136


namespace fractional_equation_solution_l1721_172169

theorem fractional_equation_solution :
  ∀ x : ℝ, x ≠ 2 → x ≠ 0 → (x / (x - 2) - 3 / x = 1) → x = 6 :=
by
  sorry

end fractional_equation_solution_l1721_172169


namespace complement_union_theorem_l1721_172189

universe u

def U : Finset ℕ := {1, 2, 3, 4}
def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {2, 3}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4} := by sorry

end complement_union_theorem_l1721_172189


namespace potato_cost_is_correct_l1721_172158

/-- The cost of one bag of potatoes from the farmer in rubles -/
def potato_cost : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase percentage -/
def andrey_increase : ℝ := 100

/-- Boris's first price increase percentage -/
def boris_first_increase : ℝ := 60

/-- Boris's second price increase percentage -/
def boris_second_increase : ℝ := 40

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The additional profit Boris made compared to Andrey in rubles -/
def additional_profit : ℝ := 1200

theorem potato_cost_is_correct : 
  potato_cost * bags_bought * (1 + boris_first_increase / 100) * boris_first_sale +
  potato_cost * bags_bought * (1 + boris_first_increase / 100) * (1 + boris_second_increase / 100) * boris_second_sale -
  potato_cost * bags_bought * (1 + andrey_increase / 100) * bags_bought = additional_profit := by
  sorry

end potato_cost_is_correct_l1721_172158


namespace roots_opposite_signs_l1721_172178

/-- Given an equation (x² + cx + d) / (2x - e) = (n - 2) / (n + 2),
    if the roots are numerically equal but have opposite signs,
    then n = (-4 - 2c) / (c - 2) -/
theorem roots_opposite_signs (c d e : ℝ) :
  let f (x : ℝ) := (x^2 + c*x + d) / (2*x - e)
  ∃ (n : ℝ), (∀ x, f x = (n - 2) / (n + 2)) →
  (∃ (r : ℝ), (f r = (n - 2) / (n + 2) ∧ f (-r) = (n - 2) / (n + 2))) →
  n = (-4 - 2*c) / (c - 2) := by
sorry

end roots_opposite_signs_l1721_172178


namespace x_squared_coefficient_zero_l1721_172161

/-- The coefficient of x^2 in the expansion of (x^2+ax+1)(x^2-3a+2) is zero when a = 1 -/
theorem x_squared_coefficient_zero (a : ℝ) : 
  (a = 1) ↔ ((-3 * a + 2 + 1) = 0) := by sorry

end x_squared_coefficient_zero_l1721_172161


namespace cube_root_unity_product_l1721_172111

/-- Given a cube root of unity ω, prove the equality for any complex numbers a, b, c -/
theorem cube_root_unity_product (ω : ℂ) (a b c : ℂ) 
  (h1 : ω^3 = 1) 
  (h2 : 1 + ω + ω^2 = 0) : 
  (a + b*ω + c*ω^2) * (a + b*ω^2 + c*ω) = a^2 + b^2 + c^2 - a*b - b*c - c*a := by
  sorry

end cube_root_unity_product_l1721_172111


namespace island_closed_path_theorem_l1721_172146

/-- Represents a rectangular county with a diagonal road --/
structure County where
  has_diagonal_road : Bool

/-- Represents a rectangular island composed of counties --/
structure Island where
  counties : List County
  is_rectangular : Bool

/-- Checks if the roads in the counties form a closed path without self-intersection --/
def forms_closed_path (island : Island) : Bool := sorry

/-- Theorem stating that a rectangular island with an odd number of counties can form a closed path
    if and only if it has at least 9 counties --/
theorem island_closed_path_theorem (island : Island) :
  island.is_rectangular ∧ 
  island.counties.length % 2 = 1 ∧
  island.counties.length ≥ 9 ∧
  (∀ c ∈ island.counties, c.has_diagonal_road) →
  forms_closed_path island :=
sorry

end island_closed_path_theorem_l1721_172146


namespace inside_implies_intersects_on_implies_tangent_outside_implies_no_intersection_l1721_172153

-- Define the circle C
def Circle (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = r^2}

-- Define the line l
def Line (x y : ℝ) : Set (ℝ × ℝ) := {p | x * p.1 + y * p.2 = x^2 + y^2}

-- Define point inside circle
def IsInside (p : ℝ × ℝ) (r : ℝ) : Prop := p.1^2 + p.2^2 < r^2

-- Define point on circle
def IsOn (p : ℝ × ℝ) (r : ℝ) : Prop := p.1^2 + p.2^2 = r^2

-- Define point outside circle
def IsOutside (p : ℝ × ℝ) (r : ℝ) : Prop := p.1^2 + p.2^2 > r^2

-- Define line intersects circle
def Intersects (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := ∃ p, p ∈ l ∧ p ∈ c

-- Define line tangent to circle
def IsTangent (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := ∃! p, p ∈ l ∧ p ∈ c

-- Define line does not intersect circle
def DoesNotIntersect (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := ∀ p, p ∈ l → p ∉ c

-- Theorem 1
theorem inside_implies_intersects (x y r : ℝ) (h1 : IsInside (x, y) r) (h2 : (x, y) ≠ (0, 0)) :
  Intersects (Line x y) (Circle r) := by sorry

-- Theorem 2
theorem on_implies_tangent (x y r : ℝ) (h : IsOn (x, y) r) :
  IsTangent (Line x y) (Circle r) := by sorry

-- Theorem 3
theorem outside_implies_no_intersection (x y r : ℝ) (h : IsOutside (x, y) r) :
  DoesNotIntersect (Line x y) (Circle r) := by sorry

end inside_implies_intersects_on_implies_tangent_outside_implies_no_intersection_l1721_172153


namespace usual_weekly_salary_proof_l1721_172199

/-- Calculates the weekly salary given daily rate and work days per week -/
def weeklySalary (dailyRate : ℚ) (workDaysPerWeek : ℕ) : ℚ :=
  dailyRate * workDaysPerWeek

/-- Represents a worker with a daily rate and standard work week -/
structure Worker where
  dailyRate : ℚ
  workDaysPerWeek : ℕ

theorem usual_weekly_salary_proof (w : Worker) 
    (h1 : w.workDaysPerWeek = 5)
    (h2 : w.dailyRate * 2 = 745) :
    weeklySalary w.dailyRate w.workDaysPerWeek = 1862.5 := by
  sorry

#eval weeklySalary (745 / 2) 5

end usual_weekly_salary_proof_l1721_172199


namespace not_necessarily_monotonic_increasing_l1721_172151

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
def strictly_increasing_by_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x < f (x + 1)

-- Define monotonic increasing
def monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

-- Theorem statement
theorem not_necessarily_monotonic_increasing 
  (h : strictly_increasing_by_one f) : 
  ¬ (monotonic_increasing f) :=
sorry

end not_necessarily_monotonic_increasing_l1721_172151


namespace prob_green_is_one_sixth_adding_two_green_balls_makes_prob_one_fourth_l1721_172160

/-- Represents the contents of the bag -/
structure BagContents where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (bag : BagContents) : ℕ :=
  bag.red + bag.yellow + bag.green

/-- Calculates the probability of drawing a green ball -/
def probGreen (bag : BagContents) : ℚ :=
  bag.green / (totalBalls bag)

/-- The initial bag contents -/
def initialBag : BagContents :=
  { red := 6, yellow := 9, green := 3 }

/-- Theorem stating the probability of drawing a green ball is 1/6 -/
theorem prob_green_is_one_sixth :
  probGreen initialBag = 1/6 := by sorry

/-- Adds green balls to the bag -/
def addGreenBalls (bag : BagContents) (n : ℕ) : BagContents :=
  { bag with green := bag.green + n }

/-- Theorem stating that adding 2 green balls makes the probability 1/4 -/
theorem adding_two_green_balls_makes_prob_one_fourth :
  probGreen (addGreenBalls initialBag 2) = 1/4 := by sorry

end prob_green_is_one_sixth_adding_two_green_balls_makes_prob_one_fourth_l1721_172160


namespace heart_op_ratio_l1721_172190

def heart_op (n m : ℕ) : ℕ := n^3 * m^2

theorem heart_op_ratio : 
  (heart_op 3 5 : ℚ) / (heart_op 5 3) = 5 / 9 := by sorry

end heart_op_ratio_l1721_172190


namespace probability_white_or_blue_is_half_l1721_172143

/-- Represents the number of marbles of each color in the basket -/
structure MarbleBasket where
  red : ℕ
  white : ℕ
  green : ℕ
  blue : ℕ

/-- Calculates the total number of marbles in the basket -/
def totalMarbles (basket : MarbleBasket) : ℕ :=
  basket.red + basket.white + basket.green + basket.blue

/-- Calculates the number of white and blue marbles in the basket -/
def whiteAndBlueMarbles (basket : MarbleBasket) : ℕ :=
  basket.white + basket.blue

/-- The probability of picking a white or blue marble from the basket -/
def probabilityWhiteOrBlue (basket : MarbleBasket) : ℚ :=
  whiteAndBlueMarbles basket / totalMarbles basket

theorem probability_white_or_blue_is_half :
  let basket : MarbleBasket := ⟨4, 3, 9, 10⟩
  probabilityWhiteOrBlue basket = 1/2 := by
  sorry

end probability_white_or_blue_is_half_l1721_172143


namespace boys_together_arrangements_l1721_172187

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 2

/-- The number of arrangements where all boys stand together -/
def arrangements_with_boys_together : ℕ := factorial num_boys * factorial (total_students - num_boys + 1)

theorem boys_together_arrangements :
  arrangements_with_boys_together = 36 :=
sorry

end boys_together_arrangements_l1721_172187


namespace function_monotonicity_l1721_172107

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

-- Define monotonically increasing function
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem function_monotonicity (a : ℝ) :
  MonotonicallyIncreasing (f a) ↔ (9/4 ≤ a ∧ a < 3) :=
sorry

end function_monotonicity_l1721_172107


namespace parallel_lines_l1721_172112

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m : ℝ) : Prop :=
  -m = -(3*m - 2)/m

/-- The first line equation: mx + y + 3 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop :=
  m*x + y + 3 = 0

/-- The second line equation: (3m - 2)x + my + 2 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop :=
  (3*m - 2)*x + m*y + 2 = 0

/-- The main theorem: lines are parallel iff m = 1 or m = 2 -/
theorem parallel_lines (m : ℝ) : parallel m ↔ (m = 1 ∨ m = 2) := by
  sorry

end parallel_lines_l1721_172112


namespace minimum_score_proof_l1721_172196

theorem minimum_score_proof (C ω : ℕ) (S : ℝ) : 
  S = 30 + 4 * C - ω →
  S > 80 →
  C + ω = 26 →
  ω ≤ 3 →
  ∀ (C' ω' : ℕ) (S' : ℝ), 
    (S' = 30 + 4 * C' - ω' ∧ 
     S' > 80 ∧ 
     C' + ω' = 26 ∧ 
     ω' ≤ 3) → 
    S ≤ S' →
  S = 119 :=
sorry

end minimum_score_proof_l1721_172196


namespace min_value_cyclic_fraction_l1721_172192

theorem min_value_cyclic_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / b + b / c + c / a ≥ 3 ∧ 
  (a / b + b / c + c / a = 3 ↔ a = b ∧ b = c) :=
by sorry

end min_value_cyclic_fraction_l1721_172192


namespace mango_purchase_problem_l1721_172150

/-- The problem of calculating the amount of mangoes purchased --/
theorem mango_purchase_problem (grape_kg : ℕ) (grape_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) :
  grape_kg = 8 →
  grape_rate = 80 →
  mango_rate = 55 →
  total_paid = 1135 →
  ∃ (mango_kg : ℕ), mango_kg * mango_rate + grape_kg * grape_rate = total_paid ∧ mango_kg = 9 :=
by sorry

end mango_purchase_problem_l1721_172150


namespace gcd_lcm_120_40_l1721_172172

theorem gcd_lcm_120_40 : 
  (Nat.gcd 120 40 = 40) ∧ (Nat.lcm 120 40 = 120) := by
  sorry

end gcd_lcm_120_40_l1721_172172


namespace manager_percentage_reduction_l1721_172176

theorem manager_percentage_reduction (total_employees : ℕ) (initial_percentage : ℚ) 
  (managers_leaving : ℕ) (target_percentage : ℚ) : 
  total_employees = 600 →
  initial_percentage = 99 / 100 →
  managers_leaving = 300 →
  target_percentage = 49 / 100 →
  (total_employees * initial_percentage - managers_leaving) / total_employees = target_percentage := by
  sorry

end manager_percentage_reduction_l1721_172176


namespace power_sum_tenth_l1721_172116

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^10 + b^10 = 123 -/
theorem power_sum_tenth (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end power_sum_tenth_l1721_172116


namespace david_orange_juice_purchase_l1721_172171

/-- Calculates the minimum cost to buy a given number of bottles -/
def min_cost (single_price : ℚ) (pack_price : ℚ) (total_bottles : ℕ) : ℚ :=
  let pack_count := total_bottles / 6
  let single_count := total_bottles % 6
  pack_count * pack_price + single_count * single_price

theorem david_orange_juice_purchase :
  min_cost (280/100) (1500/100) 22 = 5620/100 := by
  sorry

end david_orange_juice_purchase_l1721_172171


namespace arithmetic_sequence_sum_l1721_172134

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_roots : a 1 * a 99 = 21 ∧ a 1 + a 99 = 10) :
  a 3 + a 97 = 10 := by
sorry

end arithmetic_sequence_sum_l1721_172134


namespace most_cars_are_blue_l1721_172115

theorem most_cars_are_blue (total : ℕ) (red blue yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow := by
  sorry

end most_cars_are_blue_l1721_172115


namespace total_wash_time_l1721_172138

def wash_time_normal : ℕ := 4 + 7 + 4 + 9

def wash_time_suv : ℕ := 2 * wash_time_normal

def wash_time_minivan : ℕ := (3 * wash_time_normal) / 2

def num_normal_cars : ℕ := 3

def num_suvs : ℕ := 2

def num_minivans : ℕ := 1

def break_time : ℕ := 5

def total_vehicles : ℕ := num_normal_cars + num_suvs + num_minivans

theorem total_wash_time : 
  num_normal_cars * wash_time_normal + 
  num_suvs * wash_time_suv + 
  num_minivans * wash_time_minivan + 
  (total_vehicles - 1) * break_time = 229 := by
  sorry

end total_wash_time_l1721_172138


namespace gcd_of_quadratic_and_linear_l1721_172183

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 1836 * k) :
  Int.gcd (b^2 + 11*b + 28) (b + 6) = 2 := by
  sorry

end gcd_of_quadratic_and_linear_l1721_172183


namespace right_triangle_set_l1721_172191

theorem right_triangle_set : ∃! (a b c : ℝ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = Real.sqrt 2 ∧ b = 5 ∧ c = 2 * Real.sqrt 7) ∨
   (a = 6 ∧ b = 9 ∧ c = 15) ∨
   (a = 4 ∧ b = 12 ∧ c = 13)) ∧
  a^2 + b^2 = c^2 := by
  sorry

end right_triangle_set_l1721_172191


namespace total_cost_is_two_l1721_172147

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 1/10

/-- The cost of 3 pencils and 4 pens in dollars -/
def cost_3p4p : ℚ := 79/50

/-- The cost of a pen in dollars -/
def pen_cost : ℚ := (cost_3p4p - 3 * pencil_cost) / 4

/-- The total cost of 4 pencils and 5 pens in dollars -/
def total_cost : ℚ := 4 * pencil_cost + 5 * pen_cost

theorem total_cost_is_two : total_cost = 2 := by
  sorry

end total_cost_is_two_l1721_172147


namespace right_column_sum_equals_twenty_l1721_172135

/-- Represents a 3x3 grid of numbers -/
def Grid := Matrix (Fin 3) (Fin 3) ℕ

/-- Check if a grid contains only numbers from 1 to 9 without repetition -/
def isValidGrid (g : Grid) : Prop :=
  ∀ i j, g i j ∈ Finset.range 9 ∧ 
  ∀ i j i' j', g i j = g i' j' → i = i' ∧ j = j'

/-- Sum of the bottom row -/
def bottomRowSum (g : Grid) : ℕ :=
  g 2 0 + g 2 1 + g 2 2

/-- Sum of the rightmost column -/
def rightColumnSum (g : Grid) : ℕ :=
  g 0 2 + g 1 2 + g 2 2

theorem right_column_sum_equals_twenty (g : Grid) 
  (hValid : isValidGrid g) 
  (hBottomSum : bottomRowSum g = 20) 
  (hCorner : g 2 2 = 7) : 
  rightColumnSum g = 20 := by
  sorry

end right_column_sum_equals_twenty_l1721_172135


namespace ball_bounce_count_l1721_172167

/-- The number of bounces required for a ball to reach a height less than 2 feet -/
theorem ball_bounce_count (initial_height : ℝ) (bounce_ratio : ℝ) (target_height : ℝ) :
  initial_height = 20 →
  bounce_ratio = 2/3 →
  target_height = 2 →
  (∀ k : ℕ, k < 6 → initial_height * bounce_ratio^k ≥ target_height) ∧
  initial_height * bounce_ratio^6 < target_height :=
by sorry

end ball_bounce_count_l1721_172167


namespace numPaths_correct_l1721_172142

/-- The number of paths from (0,0) to (m,n) on Z^2, taking steps of +(1,0) or +(0,1) -/
def numPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem stating that numPaths gives the correct number of paths -/
theorem numPaths_correct (m n : ℕ) : 
  numPaths m n = Nat.choose (m + n) m := by sorry

end numPaths_correct_l1721_172142


namespace expression_simplification_l1721_172124

theorem expression_simplification (x : ℝ) : 
  3*x - 3*(2 - x) + 4*(2 + 3*x) - 5*(1 - 2*x) = 28*x - 3 := by
  sorry

end expression_simplification_l1721_172124


namespace pairing_theorem_l1721_172139

/-- Represents the pairing of boys and girls in the school event. -/
structure Pairing where
  boys : ℕ
  girls : ℕ
  first_pairing : ℕ := 3
  pairing_increment : ℕ := 2

/-- The relationship between boys and girls in the pairing. -/
def pairing_relationship (p : Pairing) : Prop :=
  p.boys = (p.girls - 1) / 2

/-- Theorem stating the relationship between boys and girls in the pairing. -/
theorem pairing_theorem (p : Pairing) : pairing_relationship p := by
  sorry

#check pairing_theorem

end pairing_theorem_l1721_172139


namespace factorial_difference_l1721_172185

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end factorial_difference_l1721_172185


namespace salary_degrees_in_circle_graph_l1721_172159

-- Define the percentages for each category
def transportation_percent : ℝ := 15
def research_dev_percent : ℝ := 9
def utilities_percent : ℝ := 5
def equipment_percent : ℝ := 4
def supplies_percent : ℝ := 2

-- Define the total degrees in a circle
def total_degrees : ℝ := 360

-- Theorem statement
theorem salary_degrees_in_circle_graph :
  let other_categories_percent := transportation_percent + research_dev_percent + 
                                  utilities_percent + equipment_percent + supplies_percent
  let salary_percent := 100 - other_categories_percent
  let salary_degrees := (salary_percent / 100) * total_degrees
  salary_degrees = 234 := by
sorry


end salary_degrees_in_circle_graph_l1721_172159


namespace half_of_one_point_six_million_l1721_172165

theorem half_of_one_point_six_million (x : ℝ) : 
  x = 1.6 * (10 : ℝ)^6 → (1/2 : ℝ) * x = 8 * (10 : ℝ)^5 := by
  sorry

end half_of_one_point_six_million_l1721_172165


namespace train_length_l1721_172154

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 15 → speed * (5/18) * time = 375 := by
  sorry

end train_length_l1721_172154


namespace impossible_to_make_all_divisible_by_three_l1721_172175

/-- Represents the state of numbers on the vertices of a 2018-sided polygon -/
def PolygonState := Fin 2018 → ℤ

/-- The initial state of the polygon -/
def initial_state : PolygonState :=
  fun i => if i.val = 2017 then 1 else 0

/-- The sum of all numbers on the vertices -/
def vertex_sum (state : PolygonState) : ℤ :=
  (Finset.univ.sum fun i => state i)

/-- Represents a legal move on the polygon -/
inductive LegalMove
  | add_subtract (i j : Fin 2018) : LegalMove

/-- Apply a legal move to a given state -/
def apply_move (state : PolygonState) (move : LegalMove) : PolygonState :=
  match move with
  | LegalMove.add_subtract i j =>
      fun k => if k = i then state k + 1
               else if k = j then state k - 1
               else state k

/-- Predicate to check if all numbers are divisible by 3 -/
def all_divisible_by_three (state : PolygonState) : Prop :=
  ∀ i, state i % 3 = 0

theorem impossible_to_make_all_divisible_by_three :
  ¬∃ (moves : List LegalMove), 
    all_divisible_by_three (moves.foldl apply_move initial_state) :=
  sorry


end impossible_to_make_all_divisible_by_three_l1721_172175


namespace complex_power_equality_l1721_172118

theorem complex_power_equality : (((1 - Complex.I) / Real.sqrt 2) ^ 44 : ℂ) = -1 := by sorry

end complex_power_equality_l1721_172118


namespace david_average_marks_l1721_172174

def david_marks : List ℝ := [72, 45, 72, 77, 75]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℝ) = 68.2 := by sorry

end david_average_marks_l1721_172174


namespace no_positive_abc_with_all_roots_l1721_172109

theorem no_positive_abc_with_all_roots : ¬ ∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (b^2 - 4*a*c ≥ 0) ∧ 
  (c^2 - 4*b*a ≥ 0) ∧ 
  (a^2 - 4*b*c ≥ 0) := by
  sorry

end no_positive_abc_with_all_roots_l1721_172109


namespace not_all_on_curve_implies_exists_off_curve_l1721_172113

-- Define the necessary types and functions
variable (X Y : Type) -- X and Y represent coordinate types
variable (C : Set (X × Y)) -- C represents the curve
variable (f : X → Y → Prop) -- f represents the equation f(x, y) = 0

-- The main theorem
theorem not_all_on_curve_implies_exists_off_curve :
  (¬ ∀ x y, f x y → (x, y) ∈ C) →
  ∃ x y, f x y ∧ (x, y) ∉ C := by
sorry

end not_all_on_curve_implies_exists_off_curve_l1721_172113


namespace gcd_1248_585_l1721_172157

theorem gcd_1248_585 : Nat.gcd 1248 585 = 39 := by
  sorry

end gcd_1248_585_l1721_172157


namespace sphere_radius_in_truncated_cone_l1721_172104

/-- Represents a truncated cone with horizontal bases -/
structure TruncatedCone where
  bottom_radius : ℝ
  top_radius : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Checks if a sphere is tangent to the truncated cone -/
def is_tangent (cone : TruncatedCone) (sphere : Sphere) : Prop :=
  -- This is a placeholder for the tangency condition
  True

theorem sphere_radius_in_truncated_cone (cone : TruncatedCone) (sphere : Sphere) :
  cone.bottom_radius = 10 ∧ 
  cone.top_radius = 3 ∧ 
  is_tangent cone sphere → 
  sphere.radius = Real.sqrt 30 := by
  sorry

end sphere_radius_in_truncated_cone_l1721_172104


namespace trick_decks_total_spent_l1721_172126

/-- The total amount spent by Frank and his friend on trick decks -/
def total_spent (deck_price : ℕ) (frank_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * frank_decks + deck_price * friend_decks

/-- Theorem stating the total amount spent by Frank and his friend -/
theorem trick_decks_total_spent :
  total_spent 7 3 2 = 35 := by
  sorry

end trick_decks_total_spent_l1721_172126


namespace remainder_less_than_divisor_l1721_172162

theorem remainder_less_than_divisor (a d : ℤ) (h : d ≠ 0) :
  ∃ (q r : ℤ), a = q * d + r ∧ 0 ≤ r ∧ r < |d| := by
  sorry

end remainder_less_than_divisor_l1721_172162


namespace small_semicircle_radius_l1721_172131

/-- Given a configuration of a large semicircle, a circle, and a small semicircle that are all
    pairwise tangent, this theorem proves that the radius of the smaller semicircle is 4 when
    the radius of the large semicircle is 12 and the radius of the circle is 6. -/
theorem small_semicircle_radius
  (R : ℝ) -- Radius of the large semicircle
  (r : ℝ) -- Radius of the circle
  (x : ℝ) -- Radius of the small semicircle
  (h1 : R = 12) -- Given radius of large semicircle
  (h2 : r = 6)  -- Given radius of circle
  (h3 : R > 0 ∧ r > 0 ∧ x > 0) -- All radii are positive
  (h4 : R > r ∧ R > x) -- Large semicircle is the largest
  (h5 : (R - x)^2 + r^2 = (r + x)^2) -- Pythagorean theorem for tangent circles
  : x = 4 := by
  sorry

end small_semicircle_radius_l1721_172131


namespace sufficient_condition_for_not_p_l1721_172164

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc 1 4, log_half x < 2*x + a

-- Theorem statement
theorem sufficient_condition_for_not_p (a : ℝ) :
  a < -11 → ∀ x ∈ Set.Icc 1 4, log_half x ≥ 2*x + a :=
by sorry

end sufficient_condition_for_not_p_l1721_172164


namespace cone_lateral_surface_area_l1721_172184

theorem cone_lateral_surface_area 
  (r : ℝ) (h : ℝ) (l : ℝ) (S : ℝ) 
  (h_r : r = 2) 
  (h_h : h = 4 * Real.sqrt 2) 
  (h_l : l^2 = r^2 + h^2) 
  (h_S : S = π * r * l) : 
  S = 12 * π := by
sorry

end cone_lateral_surface_area_l1721_172184


namespace stamp_collection_increase_l1721_172127

theorem stamp_collection_increase (initial_stamps final_stamps : ℕ) 
  (h1 : initial_stamps = 40)
  (h2 : final_stamps = 48) :
  (((final_stamps - initial_stamps : ℚ) / initial_stamps) * 100 : ℚ) = 20 := by
  sorry

end stamp_collection_increase_l1721_172127


namespace fraction_sum_equality_l1721_172155

theorem fraction_sum_equality : 
  (1 : ℚ) / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 2 / 15 = -2 / 15 := by
  sorry

end fraction_sum_equality_l1721_172155


namespace cylindrical_cans_radius_l1721_172197

theorem cylindrical_cans_radius (h : ℝ) (h_pos : h > 0) :
  let r₁ : ℝ := 15 -- radius of the second can
  let h₁ : ℝ := h -- height of the second can
  let h₂ : ℝ := (4 * h^2) / 3 -- height of the first can
  let v₁ : ℝ := π * r₁^2 * h₁ -- volume of the second can
  let r₂ : ℝ := (15 * Real.sqrt 3) / 2 -- radius of the first can
  v₁ = π * r₂^2 * h₂ -- volumes are equal
  := by sorry

end cylindrical_cans_radius_l1721_172197


namespace distance_to_focus_of_parabola_l1721_172173

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus_of_parabola (y : ℝ) :
  y^2 = 8 →  -- Point (1, y) satisfies the parabola equation
  Real.sqrt ((1 - 2)^2 + y^2) = 3 :=  -- Distance from (1, y) to focus (2, 0) is 3
by sorry

end distance_to_focus_of_parabola_l1721_172173


namespace geometric_sequence_product_l1721_172128

theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (m : ℕ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence definition
  (q ≠ 1 ∧ q ≠ -1) →                -- Common ratio not equal to ±1
  (a 1 = 1) →                       -- First term is 1
  (a m = a 1 * a 2 * a 3 * a 4 * a 5) →  -- Condition given in the problem
  m = 11 := by
sorry

end geometric_sequence_product_l1721_172128


namespace discount_fraction_proof_l1721_172101

/-- Given a purchase of two items with the following conditions:
  1. Each item's full price is $60.
  2. The total spent on both items is $90.
  3. The first item is bought at full price.
  4. The second item is discounted by a certain fraction.

  Prove that the discount fraction on the second item is 1/2. -/
theorem discount_fraction_proof (full_price : ℝ) (total_spent : ℝ) (discount_fraction : ℝ) :
  full_price = 60 →
  total_spent = 90 →
  total_spent = full_price + (1 - discount_fraction) * full_price →
  discount_fraction = (1 : ℝ) / 2 := by
  sorry

#check discount_fraction_proof

end discount_fraction_proof_l1721_172101


namespace factorization_equality_l1721_172149

theorem factorization_equality (x : ℝ) :
  (x^2 + 1) * (x^3 - x^2 + x - 1) = (x^2 + 1)^2 * (x - 1) := by
  sorry

end factorization_equality_l1721_172149


namespace rectangle_toothpicks_l1721_172168

/-- Calculates the number of toothpicks needed to form a rectangle --/
def toothpicks_in_rectangle (length width : ℕ) : ℕ :=
  let horizontal_rows := width + 1
  let vertical_columns := length + 1
  horizontal_rows * length + vertical_columns * width

/-- Theorem: A rectangle with length 20 and width 10 requires 430 toothpicks --/
theorem rectangle_toothpicks :
  toothpicks_in_rectangle 20 10 = 430 := by
  sorry

#eval toothpicks_in_rectangle 20 10

end rectangle_toothpicks_l1721_172168


namespace intersection_points_equality_l1721_172193

/-- Theorem: For a quadratic function y = ax^2 and two parallel lines intersecting
    this function, the difference between the x-coordinates of the intersection points
    satisfies (x3 - x1) = (x2 - x4). -/
theorem intersection_points_equality 
  (a : ℝ) 
  (x1 x2 x3 x4 : ℝ) 
  (h1 : x1 < x2) 
  (h2 : x3 < x4) 
  (h_parallel : ∃ (k b c : ℝ), 
    a * x1^2 = k * x1 + b ∧ 
    a * x2^2 = k * x2 + b ∧ 
    a * x3^2 = k * x3 + c ∧ 
    a * x4^2 = k * x4 + c) :
  x3 - x1 = x2 - x4 := by
  sorry

end intersection_points_equality_l1721_172193


namespace current_speed_l1721_172119

/-- Proves that given a man's speed with and against the current, the speed of the current can be determined. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 21)
  (h2 : speed_against_current = 12.4) :
  ∃ (current_speed : ℝ), current_speed = 4.3 := by
  sorry


end current_speed_l1721_172119


namespace bobby_candy_left_l1721_172117

theorem bobby_candy_left (initial_candy : ℕ) (eaten_candy : ℕ) (h1 : initial_candy = 30) (h2 : eaten_candy = 23) :
  initial_candy - eaten_candy = 7 := by
sorry

end bobby_candy_left_l1721_172117


namespace similar_triangle_point_coordinates_l1721_172195

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Defines a similarity transformation with a given ratio -/
def similarityTransform (p : Point) (ratio : ℝ) : Set Point :=
  { p' : Point | p'.x = p.x * ratio ∨ p'.x = p.x * (-ratio) ∧ 
                  p'.y = p.y * ratio ∨ p'.y = p.y * (-ratio) }

theorem similar_triangle_point_coordinates 
  (ABC : Triangle) 
  (C : Point) 
  (h1 : C = ABC.C) 
  (h2 : C.x = 4 ∧ C.y = 1) 
  (ratio : ℝ) 
  (h3 : ratio = 3) :
  ∃ (C' : Point), C' ∈ similarityTransform C ratio ∧ 
    ((C'.x = 12 ∧ C'.y = 3) ∨ (C'.x = -12 ∧ C'.y = -3)) :=
sorry

end similar_triangle_point_coordinates_l1721_172195


namespace volumes_and_cross_sections_l1721_172194

/-- Represents a geometric body -/
structure GeometricBody where
  volume : ℝ
  crossSectionArea : ℝ → ℝ  -- Function mapping height to cross-sectional area

/-- Zu Chongzhi's principle -/
axiom zu_chongzhi_principle (A B : GeometricBody) :
  (∀ h : ℝ, A.crossSectionArea h = B.crossSectionArea h) → A.volume = B.volume

/-- The main theorem to prove -/
theorem volumes_and_cross_sections (A B : GeometricBody) :
  (A.volume ≠ B.volume → ∃ h : ℝ, A.crossSectionArea h ≠ B.crossSectionArea h) ∧
  ∃ C D : GeometricBody, C.volume = D.volume ∧ ∃ h : ℝ, C.crossSectionArea h ≠ D.crossSectionArea h :=
sorry

end volumes_and_cross_sections_l1721_172194


namespace area_square_on_hypotenuse_l1721_172114

/-- Given a right triangle XYZ with right angle at Y, prove that the area of the square on XZ is 201
    when the sum of areas of a square on XY, a rectangle on YZ, and a square on XZ is 450,
    and YZ is 3 units longer than XY. -/
theorem area_square_on_hypotenuse (x y z : ℝ) (h1 : x^2 + y^2 = z^2)
    (h2 : y = x + 3) (h3 : x^2 + x * y + z^2 = 450) : z^2 = 201 := by
  sorry

end area_square_on_hypotenuse_l1721_172114


namespace circumscribed_circle_equation_l1721_172108

/-- The equation of a circle passing through three given points -/
def CircleEquation (x y : ℝ) := x^2 + y^2 - 6*x + 4 = 0

/-- Point A coordinates -/
def A : ℝ × ℝ := (1, 1)

/-- Point B coordinates -/
def B : ℝ × ℝ := (4, 2)

/-- Point C coordinates -/
def C : ℝ × ℝ := (2, -2)

/-- Theorem stating that the given equation represents the circumscribed circle of triangle ABC -/
theorem circumscribed_circle_equation :
  CircleEquation A.1 A.2 ∧
  CircleEquation B.1 B.2 ∧
  CircleEquation C.1 C.2 :=
sorry

end circumscribed_circle_equation_l1721_172108


namespace nails_to_buy_l1721_172148

theorem nails_to_buy (initial_nails : ℕ) (found_nails : ℕ) (total_needed : ℕ) : 
  initial_nails = 247 → found_nails = 144 → total_needed = 500 →
  total_needed - (initial_nails + found_nails) = 109 :=
by sorry

end nails_to_buy_l1721_172148


namespace quadratic_function_properties_l1721_172102

/-- A quadratic function with graph opening upwards and vertex at (1, -2) -/
def quadratic_function (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem quadratic_function_properties :
  ∃ (a : ℝ), a > 0 ∧
  (∀ x : ℝ, quadratic_function x = a * (x - 1)^2 - 2) ∧
  (∀ x : ℝ, quadratic_function x ≥ -2) ∧
  quadratic_function 1 = -2 := by
sorry


end quadratic_function_properties_l1721_172102


namespace expression_evaluation_l1721_172180

theorem expression_evaluation (a b c : ℝ) (ha : a = 3) (hb : b = 2) (hc : c = 5) :
  2 * ((a^2 + b)^2 - (a^2 - b)^2) * c^2 = 3600 := by
  sorry

end expression_evaluation_l1721_172180
