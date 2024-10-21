import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan1_better_for_B_l1303_130304

/-- The probability of player B winning a single game -/
noncomputable def p_B : ℝ := 1/4

/-- The probability of player A winning a single game -/
noncomputable def p_A : ℝ := 3/4

/-- The probability of player B winning under Plan 1 (2 out of 3 games) -/
noncomputable def prob_plan1 : ℝ := p_B^2 + 3 * p_B^2 * p_A

/-- The probability of player B winning under Plan 2 (3 out of 5 games) -/
noncomputable def prob_plan2 : ℝ := p_B^3 + 5 * p_B^3 * p_A + 10 * p_B^3 * p_A^2

/-- Theorem stating that Plan 1 maximizes B's winning probability -/
theorem plan1_better_for_B : prob_plan1 > prob_plan2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan1_better_for_B_l1303_130304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_inequalities_l1303_130387

-- Define an arithmetic sequence
noncomputable def arithmeticSequence (a₁ d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmeticSequence a₁ d n + d

-- Define the partial sum of an arithmetic sequence
noncomputable def partialSum (a₁ d : ℝ) : ℕ → ℝ
  | 0 => 0
  | n => (n : ℝ) * a₁ + (n * (n - 1) : ℝ) / 2 * d

-- State the theorem
theorem arithmetic_sequence_inequalities (a₁ d : ℝ) (i j k l : ℕ) 
  (h1 : i + l = j + k) (h2 : i ≤ j) (h3 : j ≤ k) (h4 : k ≤ l) :
  (arithmeticSequence a₁ d i) * (arithmeticSequence a₁ d l) ≤ 
    (arithmeticSequence a₁ d j) * (arithmeticSequence a₁ d k) ∧ 
  (partialSum a₁ d i) * (partialSum a₁ d l) ≤ 
    (partialSum a₁ d j) * (partialSum a₁ d k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_inequalities_l1303_130387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_product_greater_than_14_l1303_130359

def balls : Finset ℕ := {1, 2, 3, 4, 5, 6}

def is_valid_selection (a b : ℕ) : Prop :=
  a ∈ balls ∧ b ∈ balls ∧ Odd (a * b) ∧ a * b > 14

def total_outcomes : ℕ := balls.card * balls.card

def valid_outcomes : ℕ := balls.card * balls.card

theorem probability_odd_product_greater_than_14 :
  (valid_outcomes : ℚ) / total_outcomes = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_product_greater_than_14_l1303_130359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_l1303_130396

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 + x

theorem sum_lower_bound 
  (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) 
  (h₂ : x₂ > 0) 
  (h : f x₁ + f x₂ + x₁ * x₂ = 0) : 
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_l1303_130396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_part_theorem_l1303_130357

theorem complex_real_part_theorem (a : ℝ) : 
  (((a - Complex.I) / (3 + Complex.I)).re = 1/2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_part_theorem_l1303_130357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_l1303_130342

theorem binomial_expansion_sum (a : ℝ) (n : ℕ) : 
  a = ∫ x in (0)..(2), (1 - 3 * x^2) + 4 →
  Nat.choose n 2 = 15 →
  (1 + 1 / a)^n = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_l1303_130342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_equality_l1303_130306

/-- In a triangle ABC, prove that (b²cosA)/a + (c²cosB)/b + (a²cosC)/c = (a⁴ + b⁴ + c⁴)/(2abc) -/
theorem triangle_cosine_sum_equality (a b c : ℝ) (A B C : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_cos_A : Real.cos A = (b^2 + c^2 - a^2) / (2*b*c))
  (h_cos_B : Real.cos B = (a^2 + c^2 - b^2) / (2*a*c))
  (h_cos_C : Real.cos C = (a^2 + b^2 - c^2) / (2*a*b)) :
  (b^2 * Real.cos A) / a + (c^2 * Real.cos B) / b + (a^2 * Real.cos C) / c = (a^4 + b^4 + c^4) / (2*a*b*c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_equality_l1303_130306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_theorem_l1303_130374

/-- A convex 2020-gon with numbers 6 and 7 placed on vertices -/
structure Polygon2020 where
  vertices : Fin 2020 → Nat
  vertex_values : ∀ i, vertices i = 6 ∨ vertices i = 7
  consecutive_property : ∀ i, 
    (vertices i = 6 ∧ vertices ((i + 1) % 2020) = 7) ∨ 
    (vertices i = 7 ∧ vertices ((i + 1) % 2020) = 6)

/-- Sum of products of numbers on sides -/
def side_sum (p : Polygon2020) : Nat :=
  Finset.sum (Finset.range 2020) fun i => p.vertices i * p.vertices ((i + 1) % 2020)

/-- Sum of products of numbers on diagonals connecting vertices one apart -/
def diagonal_sum (p : Polygon2020) : Nat :=
  Finset.sum (Finset.range 2020) fun i => p.vertices i * p.vertices ((i + 2) % 2020)

/-- The theorem to be proved -/
theorem max_difference_theorem (p : Polygon2020) : 
  diagonal_sum p - side_sum p ≤ 1010 := by
  sorry

#check max_difference_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_theorem_l1303_130374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_intersection_iff_coprime_five_l1303_130369

/-- The set of all positive integers whose digits are 1 or 2 -/
def S : Set ℕ := {n : ℕ | ∀ d ∈ Nat.digits 10 n, d = 1 ∨ d = 2}

/-- The set of all integers divisible by n -/
def T (n : ℕ) : Set ℕ := {k : ℕ | n ∣ k}

/-- The main theorem -/
theorem infinite_intersection_iff_coprime_five (n : ℕ) :
  Set.Infinite (S ∩ T n) ↔ Nat.Coprime n 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_intersection_iff_coprime_five_l1303_130369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_representation_l1303_130347

theorem fraction_representation (n : ℕ) (a₁ a₂ : ℕ) : n = 22 →
  (∃ (k : ℕ), (3 : ℚ) / n = 0.1 + (a₁ * 10 + a₂) / 99 * (1 / 10^k)) ∧
  a₁ ≠ a₂ ∧ a₁ < 10 ∧ a₂ < 10 :=
by sorry

#check fraction_representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_representation_l1303_130347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_h_domain_intervals_l1303_130358

noncomputable def h (x : ℝ) : ℝ := (x^3 - 3*x^2 + 2*x + 4) / (x^2 - 5*x + 6)

theorem h_domain :
  Set.range h = {y | ∃ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ h x = y} :=
by sorry

theorem h_domain_intervals :
  Set.range h = Set.Iio 2 ∪ Set.Ioo 2 3 ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_h_domain_intervals_l1303_130358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l1303_130399

/-- Probability of heads for the biased coin -/
noncomputable def p_biased : ℝ := 4/5

/-- Probability of heads for the fair coin -/
noncomputable def p_fair : ℝ := 1/2

/-- Total number of tosses -/
def total_tosses : ℕ := 40

/-- Probability of even number of heads after n tosses -/
noncomputable def P (n : ℕ) : ℝ := 
  1/2 * (1 + (p_biased * p_fair) ^ n)

/-- The main theorem: probability of even number of heads after 40 tosses -/
theorem even_heads_probability : 
  P total_tosses = 1/2 * (1 + (2/5)^40) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l1303_130399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_expression_value_rhombus_area_l1303_130311

-- Define x and y as noncomputable
noncomputable def x : ℝ := 2 + Real.sqrt 2
noncomputable def y : ℝ := 2 - Real.sqrt 2

-- Theorem for part 1
theorem algebraic_expression_value : x^2 + 3*x*y + y^2 = 18 := by
  -- Proof steps would go here
  sorry

-- Theorem for part 2
theorem rhombus_area : (1/2 : ℝ) * x * y = 1 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_expression_value_rhombus_area_l1303_130311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_zero_l1303_130379

theorem product_zero (a : ℤ) (x : Fin 13 → ℤ) 
  (h1 : a = (Finset.univ : Finset (Fin 13)).prod (λ i => 1 + x i))
  (h2 : a = (Finset.univ : Finset (Fin 13)).prod (λ i => 1 - x i)) :
  a * (Finset.univ : Finset (Fin 13)).prod x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_zero_l1303_130379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joyce_apples_theorem_l1303_130307

/-- The number of apples Joyce ends up with after giving some away -/
noncomputable def joyce_final_apples (initial : ℝ) (given_to_larry : ℝ) (percentage_to_neighbors : ℝ) : ℝ :=
  let remaining_after_larry := initial - given_to_larry
  let given_to_neighbors := (percentage_to_neighbors / 100) * remaining_after_larry
  remaining_after_larry - given_to_neighbors

/-- Joyce ends up with 82.375 apples after giving some away -/
theorem joyce_apples_theorem :
  joyce_final_apples 350.5 218.7 37.5 = 82.375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joyce_apples_theorem_l1303_130307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_education_expense_l1303_130389

/-- Represents Mr. Kishore's financial situation --/
structure KishoreFinances where
  rent : ℚ
  milk : ℚ
  groceries : ℚ
  petrol : ℚ
  miscellaneous : ℚ
  savings : ℚ
  savings_rate : ℚ

/-- Calculates the amount spent on children's education --/
def education_expense (k : KishoreFinances) : ℚ :=
  let total_salary := k.savings / k.savings_rate
  let known_expenses := k.rent + k.milk + k.groceries + k.petrol + k.miscellaneous
  total_salary - known_expenses - k.savings

/-- Theorem stating the amount spent on children's education --/
theorem kishore_education_expense :
  let k : KishoreFinances := {
    rent := 5000,
    milk := 1500,
    groceries := 4500,
    petrol := 2000,
    miscellaneous := 3940,
    savings := 2160,
    savings_rate := 1/10
  }
  education_expense k = 2600 := by sorry

#eval education_expense {
  rent := 5000,
  milk := 1500,
  groceries := 4500,
  petrol := 2000,
  miscellaneous := 3940,
  savings := 2160,
  savings_rate := 1/10
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_education_expense_l1303_130389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1303_130334

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 2 then x^2 else x + 3

theorem solve_equation (a : ℝ) (h : f a + f 3 = 0) : a = -12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1303_130334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_office_letters_orderings_l1303_130318

open BigOperators Finset Nat

def possible_orderings : ℕ :=
  ∑ k in range 6, choose 5 k * (k + 1) * (k + 2)

theorem office_letters_orderings :
  possible_orderings = 552 := by
  unfold possible_orderings
  simp [sum_range_succ]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_office_letters_orderings_l1303_130318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l1303_130363

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line y = x + 1
def line_y_eq_x_plus_1 (x y : ℝ) : Prop := y = x + 1

-- Define the x-axis
def x_axis (y : ℝ) : Prop := y = 0

-- Define a point on a circle
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define a point outside a circle
def point_outside_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

-- Define a line passing through a point
structure Line where
  slope : ℝ
  y_intercept : ℝ

def line_through_point (l : Line) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  y = l.slope * x + l.y_intercept

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem circle_and_line_properties
  (C : Circle)
  (h1 : line_y_eq_x_plus_1 C.center.1 C.center.2)
  (h2 : ∃ (x : ℝ), point_on_circle (x, 0) C)
  (h3 : point_on_circle (-5, -2) C)
  (h4 : point_outside_circle (-4, -5) C)
  (l : Line)
  (h5 : line_through_point l (-2, -4))
  (h6 : ∃ (A B : ℝ × ℝ), point_on_circle A C ∧ point_on_circle B C ∧ line_through_point l A ∧ line_through_point l B ∧ distance A B = 2 * Real.sqrt 3)
  : (∀ (x y : ℝ), point_on_circle (x, y) C ↔ (x + 3)^2 + (y + 2)^2 = 4) ∧
    ((l.slope = 0 ∧ l.y_intercept = -2) ∨ (l.slope = -3/4 ∧ l.y_intercept = -5/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l1303_130363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l1303_130351

theorem log_inequality_range (a : ℝ) : 
  (0 < a ∧ a ≠ 1 ∧ (Real.log (3/5) / Real.log a < 1)) ↔ (0 < a ∧ a < 3/5) ∨ (a > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l1303_130351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_tagging_ratio_l1303_130312

theorem fish_tagging_ratio : 
  ∀ (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) (total_fish : ℕ),
  initial_tagged = 60 →
  second_catch = 50 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = 2 / 50 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_tagging_ratio_l1303_130312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1303_130327

/-- The function f(x) = (3x^2 + 8x + 15) / (3x + 4) -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 + 8*x + 15) / (3*x + 4)

/-- The proposed oblique asymptote function g(x) = x + 4/3 -/
noncomputable def g (x : ℝ) : ℝ := x + 4/3

/-- Theorem: The oblique asymptote of f(x) is g(x) -/
theorem oblique_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - g x| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1303_130327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l1303_130339

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

/-- Definition of a point being on the ellipse -/
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

/-- Left vertex of the ellipse -/
def A : ℝ × ℝ := (-2, 0)

/-- Right vertex of the ellipse -/
def B : ℝ × ℝ := (2, 0)

/-- Origin -/
def O : ℝ × ℝ := (0, 0)

/-- Function to calculate the area of a triangle given three points -/
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)

/-- Main theorem -/
theorem constant_triangle_area (P M N : ℝ × ℝ) :
  on_ellipse P → P ≠ A → P ≠ B → on_ellipse M → on_ellipse N →
  (∃ t : ℝ, M = (t * P.1, t * P.2)) →  -- M is on the ray OP
  (∃ s : ℝ, N = (s * (P.1 - 2), s * P.2)) →  -- N is on a ray parallel to BP
  triangle_area O M N = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l1303_130339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_latest_start_time_correct_l1303_130349

structure MyTask where
  name : String
  duration : Nat

def totalTaskTime (tasks : List MyTask) : Nat :=
  tasks.foldl (fun acc t => acc + t.duration) 0

def breakTime : Nat := 15
def breakCount : Nat := 3
def movieTime : Nat := 1200  -- 8:00 PM in minutes since midnight
def dinnerTime : Nat := 45
def lawnMowingDeadline : Nat := 1140  -- 7:00 PM in minutes since midnight
def homeArrivalTime : Nat := 1020  -- 5:00 PM in minutes since midnight

def tasks : List MyTask := [
  ⟨"Homework", 30⟩,
  ⟨"Clean room", 30⟩,
  ⟨"Take out trash", 5⟩,
  ⟨"Empty dishwasher", 10⟩,
  ⟨"Walk dog", 20⟩,
  ⟨"Help sister", 15⟩,
  ⟨"Mow lawn", 60⟩
]

def latestStartTime : Nat := 965  -- 4:05 PM in minutes since midnight

theorem latest_start_time_correct :
  latestStartTime = movieTime - (totalTaskTime tasks + breakTime + dinnerTime) ∧
  latestStartTime + totalTaskTime tasks + breakTime ≤ lawnMowingDeadline ∧
  latestStartTime ≥ homeArrivalTime := by
  sorry

#check latest_start_time_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_latest_start_time_correct_l1303_130349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tanker_fill_time_l1303_130315

/-- Represents the time (in minutes) it takes to fill the tanker under the given conditions -/
noncomputable def fill_time : ℝ := 30

/-- The rate at which pipe A fills the tanker (fraction per minute) -/
noncomputable def rate_A : ℝ := 1 / 60

/-- The rate at which pipe B fills the tanker (fraction per minute) -/
noncomputable def rate_B : ℝ := 1 / 40

/-- The rate at which both pipes A and B together fill the tanker (fraction per minute) -/
noncomputable def rate_AB : ℝ := rate_A + rate_B

theorem tanker_fill_time :
  rate_B * (fill_time / 2) + rate_AB * (fill_time / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tanker_fill_time_l1303_130315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_half_angle_l1303_130365

theorem tangent_half_angle (x : ℝ) : 
  Real.sin x + Real.cos x = 1/5 → Real.tan (x/2) = 2 ∨ Real.tan (x/2) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_half_angle_l1303_130365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_for_seven_l1303_130332

theorem factorial_ratio_for_seven :
  let n : ℕ := 7
  Nat.factorial (n + 2) / Nat.factorial n = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_for_seven_l1303_130332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_transformation_l1303_130364

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2

-- Define the coordinate transformation
noncomputable def t (x : ℝ) : ℝ := 2 * x
noncomputable def y (z : ℝ) : ℝ := (z + 1) / 2

-- State the theorem
theorem coordinate_transformation (x : ℝ) :
  f x = y (Real.cos (t x)) := by
  -- Expand the definitions
  simp [f, y, t]
  -- Use the trigonometric identity
  have h : Real.cos x ^ 2 = (1 + Real.cos (2 * x)) / 2 := by
    sorry -- This step requires the double angle formula for cosine
  -- Apply the identity and simplify
  rw [h]
  ring -- Simplify the algebraic expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_transformation_l1303_130364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_increasing_condition_l1303_130390

noncomputable def exponential_function (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_increasing_condition (a : ℝ) :
  a > 0 →
  a ≠ 1 →
  (∀ x y : ℝ, x < y → exponential_function a x < exponential_function a y) →
  a > 1 := by
  intro h_pos h_neq_one h_increasing
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_increasing_condition_l1303_130390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1303_130303

-- Define T(r) as the sum of the geometric series
noncomputable def T (r : ℝ) : ℝ := 20 / (1 - r)

-- State the theorem
theorem geometric_series_sum (b : ℝ) 
  (h1 : -1 < b) (h2 : b < 1) 
  (h3 : T b * T (-b) = 5040) : 
  T b + T (-b) = 504 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1303_130303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_constraint_l1303_130383

/-- Represents the selling price in yuan -/
def selling_price : ℝ := 0

/-- Represents the daily sales volume -/
def sales_volume (x : ℝ) : ℝ := 100 - 10 * (x - 10)

/-- Represents the daily profit in yuan -/
def daily_profit (x : ℝ) : ℝ := (x - 8) * sales_volume x

theorem profit_constraint :
  ∀ x : ℝ, daily_profit x > 320 → 12 < x ∧ x < 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_constraint_l1303_130383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_weight_allowed_l1303_130350

/-- Represents the weight of clothing items in ounces -/
structure ClothingWeight where
  weight : Nat

/-- Weight of a pair of socks in ounces -/
def sockWeight : ClothingWeight := ⟨2⟩

/-- Weight of underwear in ounces -/
def underwearWeight : ClothingWeight := ⟨4⟩

/-- Weight of a shirt in ounces -/
def shirtWeight : ClothingWeight := ⟨5⟩

/-- Weight of shorts in ounces -/
def shortsWeight : ClothingWeight := ⟨8⟩

/-- Weight of pants in ounces -/
def pantsWeight : ClothingWeight := ⟨10⟩

/-- Number of pants Tony is washing -/
def numPants : Nat := 1

/-- Number of shirts Tony is washing -/
def numShirts : Nat := 2

/-- Number of shorts Tony is washing -/
def numShorts : Nat := 1

/-- Number of pairs of socks Tony is washing -/
def numSocks : Nat := 3

/-- Number of underwear Tony can add -/
def numUnderwear : Nat := 4

/-- Multiplies a natural number with a ClothingWeight -/
def multWeight (n : Nat) (cw : ClothingWeight) : Nat := n * cw.weight

/-- Calculates the total weight of clothing in the washing machine -/
def totalWeight : Nat :=
  multWeight numPants pantsWeight +
  multWeight numShirts shirtWeight +
  multWeight numShorts shortsWeight +
  multWeight numSocks sockWeight +
  multWeight numUnderwear underwearWeight

/-- Theorem stating the maximum total weight of clothing allowed in the washing machine -/
theorem max_weight_allowed : totalWeight = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_weight_allowed_l1303_130350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slices_distribution_l1303_130336

theorem pizza_slices_distribution (total_slices ron_slices scott_fraction mark_slices remaining_friends : ℕ) 
  (h1 : total_slices = 24)
  (h2 : ron_slices = 5)
  (h3 : scott_fraction = 3)
  (h4 : mark_slices = 2)
  (h5 : remaining_friends = 3) :
  (total_slices - ron_slices - (total_slices - ron_slices) / scott_fraction - mark_slices) / remaining_friends = 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slices_distribution_l1303_130336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_regular_triangular_pyramid_l1303_130381

/-- The radius of a sphere touching all edges of a regular triangular pyramid -/
noncomputable def sphereRadius (a b : ℝ) : ℝ :=
  Real.sqrt ((4 * b^2 - 3 * a^2) / 12)

/-- Theorem: The radius of a sphere touching all edges of a regular triangular pyramid
    with base side length a and lateral edge b is sqrt((4b^2 - 3a^2) / 12) -/
theorem sphere_radius_regular_triangular_pyramid (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ r : ℝ, r > 0 ∧ r = sphereRadius a b ∧ 
  r * Real.sqrt 3 = Real.sqrt (b^2 - 3 * a^2 / 4) ∧
  r = a / (2 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_regular_triangular_pyramid_l1303_130381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_prism_volume_l1303_130373

/-- The volume of a right triangular prism with an isosceles right triangle base --/
theorem right_triangular_prism_volume 
  (leg : ℝ) 
  (height : ℝ) 
  (h_leg : leg = Real.sqrt 2) 
  (h_height : height = 3) : 
  (leg * leg / 2) * height = 3 := by
  -- Substitute the given values
  rw [h_leg, h_height]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

#check right_triangular_prism_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_prism_volume_l1303_130373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_area_l1303_130319

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x y : ℝ) : Prop := y = 2*x - 4

-- Define the circle (renamed to avoid conflict)
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Theorem statement
theorem parabola_line_intersection_area :
  -- Given conditions
  let focus : ℝ × ℝ := (2, 0)
  let vertex : ℝ × ℝ := (0, 0)
  -- Theorem parts
  (∀ x y, parabola x y ↔ y^2 = 8*x) ∧ 
  (∃ A B : ℝ × ℝ,
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧
    line A.1 A.2 ∧
    line B.1 B.2 ∧
    (1/2) * abs (A.1 * B.2 - A.2 * B.1) = 4 * Real.sqrt 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_area_l1303_130319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_343_l1303_130337

theorem divisibility_by_343 (x y : ℤ) 
  (h : (7 : ℤ) ∣ ((x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y))) :
  (343 : ℤ) ∣ ((x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_343_l1303_130337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1303_130322

open Set Real

def A : Set ℝ := {x | 0 < log x / log 4 ∧ log x / log 4 < 1}
def B : Set ℝ := Iic 2

theorem intersection_A_B : A ∩ B = Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1303_130322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_positive_function_condition_l1303_130346

theorem always_positive_function_condition (k : ℝ) : 
  (∀ x : ℝ, (3^(2*x) - (k + 1)*3^x + 2) > 0) ↔ k < 2^(-1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_positive_function_condition_l1303_130346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_interest_rate_problem_l1303_130326

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℝ) (h : principal > 0) (h2 : time > 0) :
  (interest * 100) / (principal * time) * principal * time / 100 = interest :=
by sorry

/-- Proof of the specific interest rate problem -/
theorem interest_rate_problem :
  (180 * 100) / (720 * 4) = 6.25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_interest_rate_problem_l1303_130326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_max_area_when_b_4_l1303_130316

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.a = t.b * Real.cos t.C + t.c * Real.sin t.B

-- Theorem 1: B = π/4
theorem triangle_angle_B (t : Triangle) (h : given_condition t) : t.B = Real.pi / 4 := by
  sorry

-- Function to calculate the area of the triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

-- Theorem 2: Maximum area when b = 4
theorem max_area_when_b_4 (t : Triangle) (h1 : given_condition t) (h2 : t.b = 4) :
  ∃ (max_area : ℝ), max_area = 4 * Real.sqrt 2 + 4 ∧ 
  ∀ (t' : Triangle), given_condition t' → t'.b = 4 → triangle_area t' ≤ max_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_max_area_when_b_4_l1303_130316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_and_intersection_circle_l1303_130329

noncomputable section

/-- Hyperbola C with center at origin and right focus at (2√3/3, 0) -/
def hyperbola_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1^2 - p.2^2 = 1}

/-- Right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (2 * Real.sqrt 3 / 3, 0)

/-- Asymptote equations of the hyperbola -/
def asymptote (x : ℝ) : Set ℝ :=
  {y : ℝ | y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x}

/-- Line l with equation y = kx + 1 -/
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- Intersection points of hyperbola C and line l -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  hyperbola_C ∩ line_l k

theorem hyperbola_equation_and_intersection_circle (k : ℝ) :
  (∀ x y, (x, y) ∈ hyperbola_C ↔ 3 * x^2 - y^2 = 1) ∧
  (∃ A B, A ∈ intersection_points k ∧ B ∈ intersection_points k ∧
    (A.1 * B.1 + A.2 * B.2 = 0 ↔ k = 1 ∨ k = -1)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_and_intersection_circle_l1303_130329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_inequality_l1303_130371

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The length of the median to side BC in a triangle -/
noncomputable def median_length (t : Triangle) : ℝ := 
  Real.sqrt ((2 * t.b ^ 2 + 2 * t.c ^ 2 - t.a ^ 2) / 4)

/-- Theorem: The median satisfies the given inequality -/
theorem median_inequality (t : Triangle) : 
  |((t.b + t.c) / 2) - (t.a / 2)| < median_length t ∧ 
  median_length t < (t.b + t.c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_inequality_l1303_130371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_surface_path_l1303_130370

/-- Represents a 3x3x3 cube with the central 3 cubes removed --/
structure Pipe :=
  (size : Nat)
  (hollow : Nat)
  (h_size : size = 3)
  (h_hollow : hollow = 1)

/-- Represents a path on the surface of the Pipe --/
structure SurfacePath :=
  (vertices : Finset (Fin 64))
  (edges : List (Fin 64 × Fin 64))
  (h_closed : edges.getLast?.map Prod.snd = edges.head?.map Prod.fst)
  (h_no_revisit : ∀ v, v ∈ vertices → (edges.filter (λ e => e.1 = v ∨ e.2 = v)).length ≤ 2)

/-- The main theorem stating the impossibility of drawing the required path --/
theorem no_valid_surface_path (p : Pipe) : ¬ ∃ (path : SurfacePath), path.vertices.card = 64 := by
  sorry

/-- Helper lemma: The number of surface diagonals is 64 --/
lemma surface_diagonals_count (p : Pipe) : 
  4 * 9 + 2 * 8 + 12 = 64 := by
  norm_num

/-- Helper lemma: The number of vertices is 64 --/
lemma vertex_count (p : Pipe) : 
  4 * 16 = 64 := by
  norm_num

/-- Helper lemma: In a chessboard coloring, there are 32 vertices of each color --/
lemma chessboard_coloring (p : Pipe) :
  64 / 2 = 32 := by
  norm_num

#check no_valid_surface_path

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_surface_path_l1303_130370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1303_130343

theorem triangle_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hSum : A + B + C = Real.pi) :
  Real.tan (A/2)^2 + Real.tan (B/2)^2 + Real.tan (C/2)^2 + 8 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1303_130343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_sum_l1303_130331

-- Define the series
def alternating_series (n : ℕ) : ℤ := 
  if n % 2 = 1 then (n + 1) / 2 else -(n / 2)

-- Define the sum of the series
def series_sum (n : ℕ) : ℤ := 
  Finset.sum (Finset.range n) (fun i => alternating_series (i + 1))

-- Theorem statement
theorem alternating_series_sum : series_sum 100 = -50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_sum_l1303_130331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_c_is_three_l1303_130393

noncomputable def c : ℕ → ℝ
  | 0 => 0
  | 1 => 0
  | n + 2 => ((n + 1) / (n + 2))^2 * c (n + 1) + 6 * (n + 1) / (n + 2)^2

theorem limit_of_c_is_three :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |c n - 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_c_is_three_l1303_130393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_percentage_l1303_130362

/-- Calculates the profit percentage for a car sale -/
noncomputable def profit_percentage (purchase_price repair_cost selling_price : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem stating that the profit percentage for the given scenario is approximately 12.55% -/
theorem car_sale_profit_percentage :
  let purchase_price : ℝ := 42000
  let repair_cost : ℝ := 13000
  let selling_price : ℝ := 61900
  abs (profit_percentage purchase_price repair_cost selling_price - 12.55) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_percentage_l1303_130362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_configuration_l1303_130325

-- Define the colors of pegs
inductive PegColor
  | Yellow
  | Red
  | Green
  | Blue
  | Orange
deriving Repr, DecidableEq

-- Define the triangular peg board
def TriangularBoard := Fin 6 → Fin 6 → Option PegColor

-- Define a valid configuration
def ValidConfiguration (board : TriangularBoard) : Prop :=
  -- Each row has at most one peg of each color
  (∀ row : Fin 6, ∀ color : PegColor,
    (Finset.filter (fun col => board row col = some color) (Finset.univ : Finset (Fin 6))).card ≤ 1) ∧
  -- Each column has at most one peg of each color
  (∀ col : Fin 6, ∀ color : PegColor,
    (Finset.filter (fun row => board row col = some color) (Finset.univ : Finset (Fin 6))).card ≤ 1) ∧
  -- Correct number of pegs for each color
  (Finset.sum (Finset.univ : Finset (Fin 6)) (fun row =>
    Finset.sum (Finset.univ : Finset (Fin 6)) (fun col =>
      if board row col = some PegColor.Yellow then 1 else 0)) = 6) ∧
  (Finset.sum (Finset.univ : Finset (Fin 6)) (fun row =>
    Finset.sum (Finset.univ : Finset (Fin 6)) (fun col =>
      if board row col = some PegColor.Red then 1 else 0)) = 5) ∧
  (Finset.sum (Finset.univ : Finset (Fin 6)) (fun row =>
    Finset.sum (Finset.univ : Finset (Fin 6)) (fun col =>
      if board row col = some PegColor.Green then 1 else 0)) = 4) ∧
  (Finset.sum (Finset.univ : Finset (Fin 6)) (fun row =>
    Finset.sum (Finset.univ : Finset (Fin 6)) (fun col =>
      if board row col = some PegColor.Blue then 1 else 0)) = 3) ∧
  (Finset.sum (Finset.univ : Finset (Fin 6)) (fun row =>
    Finset.sum (Finset.univ : Finset (Fin 6)) (fun col =>
      if board row col = some PegColor.Orange then 1 else 0)) = 2)

-- Theorem: There exists exactly one valid configuration
theorem unique_valid_configuration :
  ∃! board : TriangularBoard, ValidConfiguration board :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_configuration_l1303_130325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheet_dimension_proof_l1303_130308

/-- The length of the first dimension of a metallic sheet -/
def sheet_length : ℝ := 46

/-- The volume of the box formed by cutting squares from the corners of the sheet -/
def box_volume (length : ℝ) : ℝ :=
  (length - 16) * (36 - 16) * 8

theorem sheet_dimension_proof :
  sheet_length = 46 ∧ box_volume sheet_length = 4800 := by
  apply And.intro
  · rfl  -- This proves sheet_length = 46
  · norm_num [sheet_length, box_volume]  -- This simplifies and proves box_volume sheet_length = 4800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheet_dimension_proof_l1303_130308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_divisors_sum_l1303_130345

/-- The set of positive integer divisors of 294, excluding 1 -/
def divisors_294 : Set Nat := {d | d > 1 ∧ 294 % d = 0}

/-- A function representing the circular arrangement of divisors -/
noncomputable def circular_arrangement : divisors_294 → divisors_294 := sorry

/-- Predicate to check if two numbers have a common factor greater than 1 -/
def has_common_factor (a b : Nat) : Prop := ∃ (f : Nat), f > 1 ∧ a % f = 0 ∧ b % f = 0

theorem adjacent_divisors_sum :
  ∀ (x y : Nat),
    x ∈ divisors_294 →
    y ∈ divisors_294 →
    (circular_arrangement ⟨x, by sorry⟩ = ⟨21, by sorry⟩ ∨ 
     circular_arrangement ⟨y, by sorry⟩ = ⟨21, by sorry⟩) →
    has_common_factor x 21 →
    has_common_factor y 21 →
    has_common_factor x y →
    x + y = 49 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_divisors_sum_l1303_130345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_depth_l1303_130353

theorem box_depth (length width : ℕ) (num_cubes : ℕ) (depth : ℕ) : 
  length = 24 →
  width = 40 →
  num_cubes = 30 →
  (∃ (cube_side : ℕ), 
    cube_side > 0 ∧
    length % cube_side = 0 ∧
    width % cube_side = 0 ∧
    depth % cube_side = 0 ∧
    num_cubes * (cube_side ^ 3) = length * width * depth) →
  depth = 16 := by
  sorry

#check box_depth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_depth_l1303_130353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_l1303_130394

/-- The length of a tunnel given a train passing through it -/
theorem tunnel_length (train_speed : ℝ) (train_length : ℝ) (time_in_tunnel : ℝ) : 
  train_speed = 100 / 3 →
  train_length = 400 →
  time_in_tunnel = 30 →
  train_speed * time_in_tunnel - train_length = 600 := by
  intros h1 h2 h3
  -- Proof steps would go here
  sorry

#check tunnel_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_l1303_130394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_k_value_l1303_130328

/-- Given vectors OA, OB, OC, and collinear points A, B, C, prove k = -2/3 --/
theorem collinear_vectors_k_value (k : ℝ) : 
  let OA : Fin 3 → ℝ := ![k, 12, 1]
  let OB : Fin 3 → ℝ := ![4, 5, 1]
  let OC : Fin 3 → ℝ := ![-k, 10, 1]
  (∃ (t : ℝ), (t • (OC - OA) = OB - OA)) → k = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_k_value_l1303_130328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_fit_in_box_l1303_130388

/-- Represents a 3D box with dimensions length, width, and height. -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a ball with a given radius. -/
structure Ball where
  radius : ℝ

/-- Checks if a given number of balls can fit into a box. -/
def canFitBalls (box : Box) (ball : Ball) (n : ℕ) : Prop :=
  ∃ (arrangement : ℕ → ℝ × ℝ × ℝ),
    (∀ i j, i ≠ j → ‖arrangement i - arrangement j‖ ≥ 2 * ball.radius) ∧
    (∀ i, i < n → arrangement i ∈ Set.Icc (ball.radius, ball.radius, ball.radius) 
                                          (box.length - ball.radius, box.width - ball.radius, box.height - ball.radius))

theorem balls_fit_in_box :
  let box : Box := ⟨10, 10, 1⟩
  let ball : Ball := ⟨0.5⟩
  (canFitBalls box ball 105) ∧ (canFitBalls box ball 106) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_fit_in_box_l1303_130388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSumValue_l1303_130375

/-- The sum of the infinite series ∑(1/(n(n+3))) for n from 1 to infinity -/
noncomputable def infiniteSeriesSum : ℝ := ∑' n, 1 / (n * (n + 3))

/-- Theorem stating that the sum of the infinite series is equal to 11/18 -/
theorem infiniteSeriesSumValue : infiniteSeriesSum = 11/18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSumValue_l1303_130375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_range_l1303_130321

/-- Given a sequence {a_n} with sum of first n terms S_n = 3^n(lambda - n) - 6,
    if {a_n} is decreasing, then lambda ∈ (-∞, 2) -/
theorem sequence_sum_range (lambda : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = 3^n * (lambda - n) - 6) →
  (∀ n, a (n+1) < a n) →
  lambda ∈ Set.Iio 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_range_l1303_130321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sculpture_surface_area_l1303_130310

/-- Represents a cube in the sculpture --/
structure Cube where
  edge : ℕ
  deriving Repr

/-- Represents a layer in the sculpture --/
structure Layer where
  cubes : ℕ
  exposed_sides : ℕ
  deriving Repr

/-- Represents the entire sculpture --/
structure Sculpture where
  layers : List Layer
  cube : Cube
  deriving Repr

/-- Calculates the exposed surface area of a single layer --/
def layer_surface_area (l : Layer) (c : Cube) : ℕ :=
  l.cubes * l.exposed_sides * c.edge ^ 2

/-- Calculates the total exposed surface area of the sculpture --/
def total_surface_area (s : Sculpture) : ℕ :=
  (s.layers.map (λ l => layer_surface_area l s.cube)).sum

/-- The main theorem stating the total exposed surface area of the sculpture --/
theorem sculpture_surface_area :
  let cube : Cube := ⟨2⟩
  let layers : List Layer := [⟨1, 5⟩, ⟨4, 9⟩, ⟨6, 10⟩, ⟨9, 6⟩]
  let sculpture : Sculpture := ⟨layers, cube⟩
  total_surface_area sculpture = 180 := by sorry

#eval total_surface_area ⟨[⟨1, 5⟩, ⟨4, 9⟩, ⟨6, 10⟩, ⟨9, 6⟩], ⟨2⟩⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sculpture_surface_area_l1303_130310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_polygon_angle_l1303_130376

/-- The angle at each point of a star formed by extending every second side of a regular polygon -/
noncomputable def star_angle (n : ℕ) : ℝ :=
  (n - 3 : ℝ) * 360 / n

/-- Theorem: The angle at each point of the star where extensions intersect in a regular polygon with n sides -/
theorem star_polygon_angle (n : ℕ) (h1 : n > 5) (h2 : Odd n) :
  star_angle n = (n - 3 : ℝ) * 360 / n :=
by
  -- Unfold the definition of star_angle
  unfold star_angle
  -- The equality holds by definition
  rfl

#check star_polygon_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_polygon_angle_l1303_130376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_l1303_130384

/-- Represents the number of pages in a book -/
def total_pages (n : ℕ) : Prop := n = 500

/-- Represents the reading speed for the first half of the book in pages per day -/
def first_half_speed : ℕ := 10

/-- Represents the reading speed for the second half of the book in pages per day -/
def second_half_speed : ℕ := 5

/-- Represents the total number of days spent reading the book -/
def total_days : ℕ := 75

/-- Theorem stating that given the conditions, the book contains 500 pages -/
theorem book_pages : total_pages 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_l1303_130384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_expression_equals_fraction_l1303_130367

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem inverse_expression_equals_fraction :
  (i - i⁻¹ + (3 : ℂ))⁻¹ = (3 - 2*i) / 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_expression_equals_fraction_l1303_130367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_projection_is_point_or_line_l1303_130341

/-- A straight line in 3D space -/
structure StraightLine where
  point : EuclideanSpace ℝ (Fin 3)
  direction : EuclideanSpace ℝ (Fin 3)

/-- A plane in 3D space -/
structure Plane where
  point : EuclideanSpace ℝ (Fin 3)
  normal : EuclideanSpace ℝ (Fin 3)

/-- The projection of a point onto a plane -/
def project_point (p : EuclideanSpace ℝ (Fin 3)) (plane : Plane) : EuclideanSpace ℝ (Fin 3) :=
  sorry

/-- The projection of a straight line onto a plane -/
def project_line (line : StraightLine) (plane : Plane) : Set (EuclideanSpace ℝ (Fin 3)) :=
  sorry

/-- Predicate to check if a set of points is a single point -/
def is_point (s : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  ∃ p, s = {p}

/-- Predicate to check if a set of points forms a straight line -/
def is_straight_line (s : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  sorry

/-- Theorem: The projection of a straight line onto a plane is either a point or a straight line -/
theorem line_projection_is_point_or_line (line : StraightLine) (plane : Plane) :
  let projection := project_line line plane
  is_point projection ∨ is_straight_line projection :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_projection_is_point_or_line_l1303_130341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_absence_percentage_l1303_130398

theorem school_absence_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (boys_absent_fraction : ℚ) (girls_absent_fraction : ℚ) :
  total_students = 180 →
  boys = 120 →
  girls = 60 →
  boys_absent_fraction = 1 / 6 →
  girls_absent_fraction = 1 / 4 →
  (((boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students) * 100 : ℚ) = 35 / 180 * 100 := by
  sorry

#eval (35 / 180 : ℚ) * 100  -- This will evaluate to 19.44444...

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_absence_percentage_l1303_130398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sales_x_correct_increased_sales_range_l1303_130309

/-- Represents the sales model with price increase and volume decrease -/
structure SalesModel where
  p : ℝ  -- original price in yuan
  n : ℝ  -- original sales volume per month
  x : ℝ  -- price increase in tenths
  y : ℝ  -- sales volume decrease in tenths
  z : ℝ  -- ratio of new total sales to original
  a : ℝ  -- constant relating x and y
  h1 : 0 < x
  h2 : x ≤ 10
  h3 : 1/3 ≤ a
  h4 : a < 1
  h5 : y = a * x
  h6 : z = (1 + x/10) * (1 - y/10)

/-- The value of x that maximizes sales amount -/
noncomputable def max_sales_x (model : SalesModel) : ℝ := 5 * (1 - model.a) / model.a

/-- Theorem: The value of x that maximizes sales amount is 5(1-a)/a -/
theorem max_sales_x_correct (model : SalesModel) :
  max_sales_x model = 5 * (1 - model.a) / model.a := by
  sorry

/-- Theorem: When y = 2/3 * x, the range of x for increased sales is 0 < x < 5 -/
theorem increased_sales_range (model : SalesModel) (h : model.y = 2/3 * model.x) :
  model.z > 1 ↔ 0 < model.x ∧ model.x < 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sales_x_correct_increased_sales_range_l1303_130309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l1303_130360

-- Define the focal length and eccentricity
def focal_length : ℝ := 8
def eccentricity : ℝ := 0.8

-- Define the ellipse parameters
noncomputable def c : ℝ := focal_length / 2
noncomputable def a : ℝ := c / eccentricity
noncomputable def b : ℝ := Real.sqrt (a^2 - c^2)

-- Define the standard equations of the ellipse
def standard_equation_1 (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def standard_equation_2 (x y : ℝ) : Prop := y^2 / a^2 + x^2 / b^2 = 1

theorem ellipse_standard_equation :
  (∀ x y : ℝ, standard_equation_1 x y ∨ standard_equation_2 x y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l1303_130360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inlet_rate_calculation_l1303_130378

/-- Represents a water tank with a leak and an inlet pipe. -/
structure WaterTank where
  capacity : ℚ
  leakEmptyTime : ℚ
  leakAndInletEmptyTime : ℚ

/-- Calculates the rate of the inlet pipe in liters per minute. -/
def inletRate (tank : WaterTank) : ℚ :=
  (tank.capacity / tank.leakEmptyTime - tank.capacity / tank.leakAndInletEmptyTime) / 60

/-- Theorem stating the inlet rate for the given tank specifications. -/
theorem inlet_rate_calculation (tank : WaterTank) 
    (h1 : tank.capacity = 6480)
    (h2 : tank.leakEmptyTime = 6)
    (h3 : tank.leakAndInletEmptyTime = 8) : 
  inletRate tank = 4.5 := by
  sorry

def exampleTank : WaterTank := {
  capacity := 6480,
  leakEmptyTime := 6,
  leakAndInletEmptyTime := 8
}

#eval inletRate exampleTank

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inlet_rate_calculation_l1303_130378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l1303_130317

theorem polynomial_value_bound (n : ℕ) (P : Polynomial ℝ) (x : Fin (n + 1) → ℤ) : 
  (∀ i j : Fin (n + 1), i < j → x i < x j) →
  (P.degree = n) →
  (P.coeff n = 1) →
  ∃ j : Fin (n + 1), |P.eval (↑(x j))| ≥ (n.factorial : ℝ) / 2^n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l1303_130317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_of_f_l1303_130366

noncomputable def f (x : ℝ) : ℝ := (1/4) * x^4 - 2 * x^3 + (11/2) * x^2 - 6 * x + 9/4

theorem extrema_of_f :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3 - ε) (3 + ε), f x ≥ f 3) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≤ f 2) ∧
  f 1 = 0 ∧ f 3 = 0 ∧ f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_of_f_l1303_130366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1303_130314

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define our function f(x) = lg(x+2)
noncomputable def f (x : ℝ) : ℝ := lg (x + 2)

-- State the theorem about the domain of f
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1303_130314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hexagon_area_l1303_130330

/-- Represents a hexagon with specific properties -/
structure Hexagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  rectangle_width : ℝ
  rectangle_length : ℝ
  triangle_base : ℝ
  triangle_height : ℝ

/-- Calculates the area of the hexagon -/
noncomputable def hexagon_area (h : Hexagon) : ℝ :=
  h.rectangle_width * h.rectangle_length + 2 * (1/2 * h.triangle_base * h.triangle_height)

/-- Theorem stating that the area of the specific hexagon is 990 square units -/
theorem specific_hexagon_area :
  let h : Hexagon := {
    side1 := 18,
    side2 := 25,
    side3 := 30,
    side4 := 28,
    side5 := 25,
    side6 := 18,
    rectangle_width := 18,
    rectangle_length := 30,
    triangle_base := 18,
    triangle_height := 25
  }
  hexagon_area h = 990 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hexagon_area_l1303_130330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_pyramid_edges_can_be_larger_l1303_130320

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  apex : Point3D
  base1 : Point3D
  base2 : Point3D
  base3 : Point3D

/-- Calculates the sum of edge lengths of a triangular pyramid -/
noncomputable def sumEdgeLengths (p : TriangularPyramid) : ℝ :=
  distance p.apex p.base1 + distance p.apex p.base2 + distance p.apex p.base3 +
  distance p.base1 p.base2 + distance p.base2 p.base3 + distance p.base3 p.base1

/-- Checks if one pyramid is inside another -/
def isInside (inner outer : TriangularPyramid) : Prop :=
  inner.base1 = outer.base1 ∧ inner.base2 = outer.base2 ∧ inner.base3 = outer.base3 ∧
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧
    inner.apex = Point3D.mk
      (t * outer.apex.x + (1 - t) * outer.base1.x)
      (t * outer.apex.y + (1 - t) * outer.base1.y)
      (t * outer.apex.z + (1 - t) * outer.base1.z)

theorem inner_pyramid_edges_can_be_larger :
  ∃ (inner outer : TriangularPyramid),
    isInside inner outer ∧ sumEdgeLengths inner > sumEdgeLengths outer := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_pyramid_edges_can_be_larger_l1303_130320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1303_130352

/-- The maximum area of a triangle DEF with DE = 12 and DF:EF = 25:26 is 143/4 -/
theorem max_triangle_area (D E F : ℝ × ℝ) : 
  let de := Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2)
  let df := Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2)
  let ef := Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2)
  de = 12 ∧ df / ef = 25 / 26 →
  ∀ area : ℝ, area = Real.sqrt (let s := (de + df + ef) / 2; s * (s - de) * (s - df) * (s - ef)) →
  area ≤ 143 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1303_130352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_product_l1303_130348

noncomputable section

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - C.1)^2 + (B.2 - C.2)^2

-- Define a right angle
def RightAngle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the length of a line segment
noncomputable def Length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define a perpendicular line
def Perpendicular (A P B C : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (C.1 - B.1) + (P.2 - A.2) * (C.2 - B.2) = 0

-- Define vector representation
def VectorRep (A P B C : ℝ × ℝ) (l m : ℝ) : Prop :=
  P.1 - A.1 = l * (B.1 - A.1) + m * (C.1 - A.1) ∧
  P.2 - A.2 = l * (B.2 - A.2) + m * (C.2 - A.2)

theorem triangle_vector_product (A B C P : ℝ × ℝ) (l m : ℝ) :
  Triangle A B C →
  RightAngle A B C →
  Length A C = 1 →
  Length A B = 2 →
  Perpendicular A P B C →
  VectorRep A P B C l m →
  l * m = 4 / 25 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_product_l1303_130348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_problem_l1303_130324

/-- Calculates the binomial coefficient C(n,k) -/
def binomial (n k : ℕ) : ℕ := 
  Nat.choose n k

/-- Calculates a = C₂₀⁰ + C₂₀¹·2 + C₂₀²·2² + … + C₂₀²⁰·2²⁰ -/
def a : ℕ := 
  Finset.sum (Finset.range 21) (fun k => binomial 20 k * 2^k)

/-- Two numbers are congruent modulo m if they have the same remainder when divided by m -/
def congruent_mod (x y m : ℤ) : Prop := 
  ∃ k : ℤ, x - y = m * k

theorem congruence_problem (b : ℤ) (h : congruent_mod (a : ℤ) b 10) :
  congruent_mod b 2021 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_problem_l1303_130324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_range_of_a_l1303_130392

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + Real.log (x + 1)

def has_extreme_point (g : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), 
    (g x ≤ g x₀ ∨ g x ≥ g x₀) ∧ (g x = g x₀ → x = x₀)

theorem extreme_points_and_range_of_a :
  (∀ x > -1, has_extreme_point (f 2) (-Real.sqrt 2 / 2)) ∧
  (∀ x > -1, has_extreme_point (f 2) (Real.sqrt 2 / 2)) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, (deriv (f a)) x > x) → a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_range_of_a_l1303_130392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_m_l1303_130395

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 1 < x ∧ x < m + 1}

-- Define the conditions
def condition1 (m : ℝ) : Prop := A ∩ B m = B m
def condition2 (m : ℝ) : Prop := (A ∩ B m).Nonempty

-- Define the set of possible values for m
def M : Set ℝ := {m : ℝ | condition1 m ∧ condition2 m}

-- Theorem statement
theorem possible_values_of_m : M = Set.Icc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_m_l1303_130395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1303_130340

-- Define the curve C
noncomputable def curve_C (a : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (a + 4 * Real.cos θ, 1 + 4 * Real.sin θ)

-- Define the line
def line (x y : ℝ) : Prop :=
  3 * x + 4 * y - 5 = 0

-- Theorem statement
theorem intersection_point (a : ℝ) (h1 : a > 0) :
  (∃! p : ℝ × ℝ, ∃ θ : ℝ, p = curve_C a θ ∧ line p.1 p.2) → a = 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1303_130340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_in_terms_of_a_and_b_l1303_130354

theorem sin_x_in_terms_of_a_and_b (a b x : ℝ) 
  (h1 : Real.tan x = (3 * a * b) / (a^2 - b^2))
  (h2 : a > b)
  (h3 : b > 0)
  (h4 : 0 < x)
  (h5 : x < Real.pi / 2) :
  Real.sin x = (3 * a * b) / Real.sqrt (a^4 + 7 * a^2 * b^2 + b^4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_in_terms_of_a_and_b_l1303_130354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l1303_130372

theorem plane_equation (p1 p2 normal : ℝ × ℝ × ℝ) :
  p1 = (-1, 1, 1) →
  p2 = (1, -1, 1) →
  normal = (1, 2, 3) →
  ∃ (A B C D : ℤ),
    (A : ℝ) * p1.fst + (B : ℝ) * p1.snd.fst + (C : ℝ) * p1.snd.snd + (D : ℝ) = 0 ∧
    (A : ℝ) * p2.fst + (B : ℝ) * p2.snd.fst + (C : ℝ) * p2.snd.snd + (D : ℝ) = 0 ∧
    (A : ℝ) * normal.fst + (B : ℝ) * normal.snd.fst + (C : ℝ) * normal.snd.snd = 0 ∧
    A > 0 ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    A = 1 ∧ B = 1 ∧ C = -1 ∧ D = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l1303_130372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_area_theorem_l1303_130361

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def totalAreaPyramid (baseEdge : ℝ) (lateralEdge : ℝ) : ℝ :=
  4 * (1/2 * baseEdge * Real.sqrt (lateralEdge^2 - (baseEdge/2)^2))

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid
    with base edges of 8 units and lateral edges of 7 units is equal to 16√33 square units -/
theorem pyramid_area_theorem :
  totalAreaPyramid 8 7 = 16 * Real.sqrt 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_area_theorem_l1303_130361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_password_probability_l1303_130300

/-- The set of non-negative single-digit numbers -/
def NonNegativeSingleDigit : Finset ℕ := Finset.filter (fun n => n < 10) (Finset.range 10)

/-- The set of prime single-digit numbers -/
def PrimeSingleDigit : Finset ℕ := Finset.filter (fun n => n < 10 ∧ Nat.Prime n) (Finset.range 10)

/-- The set of positive single-digit numbers -/
def PositiveSingleDigit : Finset ℕ := Finset.filter (fun n => 0 < n ∧ n < 10) (Finset.range 10)

/-- The probability of selecting a prime single-digit number -/
def ProbPrimeSingleDigit : ℚ := (PrimeSingleDigit.card : ℚ) / (NonNegativeSingleDigit.card : ℚ)

/-- The probability of selecting a positive single-digit number -/
def ProbPositiveSingleDigit : ℚ := (PositiveSingleDigit.card : ℚ) / (NonNegativeSingleDigit.card : ℚ)

/-- The probability of Bob's password meeting the specified criteria -/
def PasswordProbability : ℚ := ProbPrimeSingleDigit * ProbPositiveSingleDigit

theorem password_probability :
  PasswordProbability = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_password_probability_l1303_130300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_payment_l1303_130386

/-- Given a 10 percent deposit of $110, prove that the remaining amount to be paid is $990 -/
theorem remaining_payment (deposit : ℚ) (deposit_percentage : ℚ) : 
  deposit = 110 → deposit_percentage = 1/10 → 
  (deposit / deposit_percentage - deposit : ℚ) = 990 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_payment_l1303_130386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l1303_130301

/-- Rectangle ABCD with AB = 4 and BC = 8 -/
structure Rectangle where
  AB : ℝ
  BC : ℝ
  h1 : AB = 4
  h2 : BC = 8

/-- Pentagon ABEFD formed by folding rectangle ABCD so that A coincides with C -/
def foldedPentagon (r : Rectangle) : Type := 
  { EF : ℝ // ∃ (x : ℝ), x ^ 2 + r.AB ^ 2 = (r.BC - x) ^ 2 ∧ EF ^ 2 = 4 + (r.BC - x) ^ 2 }

/-- The length of EF in the folded pentagon is 2√5 -/
theorem EF_length (r : Rectangle) : ∃ (p : foldedPentagon r), p.val = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l1303_130301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_correct_l1303_130305

-- Define the number of lamps of each color
def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4

-- Define the total number of lamps
def total_lamps : ℕ := num_red_lamps + num_blue_lamps

-- Define the number of lamps that are turned on
def num_on_lamps : ℕ := 4

-- Define the function to calculate the probability
noncomputable def probability_specific_arrangement : ℚ :=
  -- Define the numerator (favorable outcomes)
  let favorable_outcomes := (Nat.choose (total_lamps - 2) (num_red_lamps - 1)) * (Nat.choose (total_lamps - 2) (num_on_lamps - 1))
  -- Define the denominator (total outcomes)
  let total_outcomes := (Nat.choose total_lamps num_red_lamps) * (Nat.choose total_lamps num_on_lamps)
  -- Calculate the probability
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- Theorem statement
theorem probability_is_correct : probability_specific_arrangement = 8 / 98 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_correct_l1303_130305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_range_minimum_value_M_l1303_130344

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * Real.exp (a * x - 1) - 2 * a * x + f a x

-- State the theorems
theorem monotonicity_range (a : ℝ) :
  (∀ x, x > 0 → x < Real.log 3 → StrictMono (f a) ↔ StrictMono (F a)) →
  a < 0 →
  a ≤ -3 :=
by sorry

theorem minimum_value_M (a : ℝ) (M : ℝ) :
  a ≤ -(Real.exp 2)⁻¹ →
  a < 0 →
  (∃ m, ∀ x, x > 0 → g a x ≥ m) →
  (∀ ε > 0, ∃ x, x > 0 ∧ g a x < M + ε) →
  M = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_range_minimum_value_M_l1303_130344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_is_periodic_l1303_130380

-- Define the property of being a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the sine function
noncomputable def sine : ℝ → ℝ := Real.sin

-- Theorem statement
theorem sine_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric sine →
  IsPeriodic sine := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_is_periodic_l1303_130380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_mixture_mass_l1303_130368

/-- Given an initial paint mixture of 12 kg with 80% white paint by mass,
    when more white paint is added to achieve a 90% white paint composition by mass,
    the final total mass of the paint is 24 kg. -/
theorem paint_mixture_mass (initial_mass : ℝ) (initial_white_percent : ℝ) 
  (final_white_percent : ℝ) (final_mass : ℝ) : 
  initial_mass = 12 →
  initial_white_percent = 0.8 →
  final_white_percent = 0.9 →
  final_mass = initial_mass + (initial_mass * initial_white_percent) / 
    (final_white_percent - initial_white_percent) * (1 - final_white_percent) →
  final_mass = 24 := by
  sorry

#check paint_mixture_mass

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_mixture_mass_l1303_130368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_divisible_by_80_contradiction_l1303_130302

theorem number_divisible_by_80_contradiction :
  ¬ ∃ (x y z : ℕ),
    (x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9) ∧  -- x, y, z are single digits
    (100 * x + 10 * y + z) % 80 = 0 ∧  -- number is divisible by 80
    x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_divisible_by_80_contradiction_l1303_130302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_pool_fill_time_l1303_130385

/-- Represents the time taken to fill Thomas's swimming pool -/
noncomputable def fill_time : ℝ :=
  let pool_capacity : ℝ := 32000
  let hoses_count : ℕ := 5
  let normal_hose_flow : ℝ := 3
  let blocked_hose_flow : ℝ := 1.5
  let normal_hoses_count : ℕ := 4
  let total_flow_per_minute : ℝ := normal_hose_flow * (normal_hoses_count : ℝ) + blocked_hose_flow
  let total_flow_per_hour : ℝ := total_flow_per_minute * 60
  pool_capacity / total_flow_per_hour

theorem thomas_pool_fill_time : 
  ⌈fill_time⌉ = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_pool_fill_time_l1303_130385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_travel_time_l1303_130323

/-- The time taken for a boat to travel with the current -/
noncomputable def time_with_current (distance : ℝ) (time_against : ℝ) (boat_speed : ℝ) (current_speed : ℝ) : ℝ :=
  distance / (boat_speed + current_speed)

/-- The speed of the current -/
noncomputable def current_speed (distance : ℝ) (time_against : ℝ) (boat_speed : ℝ) : ℝ :=
  boat_speed - distance / time_against

theorem boat_travel_time 
  (distance : ℝ) 
  (time_against : ℝ) 
  (boat_speed : ℝ) 
  (h1 : distance = 96)
  (h2 : time_against = 8)
  (h3 : boat_speed = 15.6) :
  time_with_current distance time_against boat_speed (current_speed distance time_against boat_speed) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_travel_time_l1303_130323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equals_specific_values_l1303_130335

/-- The set of real numbers a for which the inequality (ax-1)(x+2a-1) > 0 
    has exactly 3 integer solutions -/
def SolutionSet : Set ℝ :=
  {a : ℝ | ∃ (S : Set ℤ), 
    (∀ x : ℤ, x ∈ S ↔ (a * x - 1) * (x + 2 * a - 1) > 0) ∧
    (∃ (l : List ℤ), S = l.toFinset ∧ l.length = 3)}

/-- The theorem stating that the solution set is {-1, -1/2} -/
theorem solution_set_equals_specific_values : 
  SolutionSet = {-1, -1/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equals_specific_values_l1303_130335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1303_130338

def a : ℕ → ℚ
| 0 => 2
| n + 1 => a n / (1 + a n)

theorem a_formula (n : ℕ) : a n = 2 / (2 * (n + 1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1303_130338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sum_relation_l1303_130391

theorem triangle_sum_relation (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = a^2)
  (h2 : y^2 + y*z + z^2 = b^2)
  (h3 : x^2 + x*z + z^2 = c^2) :
  let s := (a + b + c) / 2
  x*y + y*z + x*z = 4 * Real.sqrt ((s * (s - a) * (s - b) * (s - c)) / 3) := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sum_relation_l1303_130391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1303_130382

/-- The time taken to complete a task when two workers work together -/
noncomputable def combined_work_time (time1 time2 : ℝ) : ℝ :=
  1 / (1 / time1 + 1 / time2)

/-- Theorem: If Ravi can do a piece of work in 50 days and Prakash can do it in 75 days,
    then they will finish the work together in 30 days -/
theorem work_completion_time :
  combined_work_time 50 75 = 30 := by
  -- Unfold the definition of combined_work_time
  unfold combined_work_time
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1303_130382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1303_130313

/-- The area of a circular sector with radius r and central angle θ -/
noncomputable def sectorArea (r : ℝ) (θ : ℝ) : ℝ := (1/2) * r^2 * θ

/-- Theorem: The area of a circular sector with radius 6 and central angle π/3 is 6π -/
theorem sector_area_example : sectorArea 6 (Real.pi/3) = 6*Real.pi := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1303_130313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_value_at_7_l1303_130333

-- Define the polynomial P(x)
def P (a b c d e : ℝ) (x : ℂ) : ℂ :=
  (3 * x^4 - 15 * x^3 + a * x^2 + b * x + c) * (4 * x^3 - 36 * x^2 + d * x + e)

-- Define the set of roots
def roots : Set ℂ := {2, 3, 4, 5}

-- Theorem statement
theorem P_value_at_7 (a b c d e : ℝ) :
  (∃ z ∈ roots, (Complex.abs (P a b c d e z) = 0 ∧ 
    (∃ w ∈ roots, w ≠ z ∧ Complex.abs (P a b c d e w) = 0))) →
  P a b c d e 7 = 23040 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_value_at_7_l1303_130333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_value_range_l1303_130355

/-- The range of the exact value of an approximate number -/
theorem exact_value_range (a : ℝ) : 
  (∃ (n : ℕ), n = 170 ∧ Int.floor a = n) ↔ 169.5 ≤ a ∧ a < 170.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_value_range_l1303_130355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_supplies_cost_difference_l1303_130356

/-- The cost difference between two years of school supplies purchases. -/
theorem school_supplies_cost_difference 
  (pen_cost notebook_cost : ℤ)
  (total_spent : ℤ)
  (x y : ℤ) :
  pen_cost = 13 →
  notebook_cost = 17 →
  total_spent = 10000 →
  pen_cost * x + notebook_cost * y = total_spent →
  ∀ (a b : ℤ), (pen_cost * a + notebook_cost * b = total_spent) → (abs (x - y) ≤ abs (a - b)) →
  notebook_cost * y + pen_cost * x - (pen_cost * x + notebook_cost * y) = 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_supplies_cost_difference_l1303_130356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_complex_occupancy_l1303_130397

/-- Represents an apartment complex with identical buildings -/
structure ApartmentComplex where
  numBuildings : ℕ
  studioApts : ℕ
  twoPersApts : ℕ
  fourPersApts : ℕ

/-- Calculates the maximum occupancy of an apartment complex -/
def maxOccupancy (complex : ApartmentComplex) : ℕ :=
  complex.numBuildings * (complex.studioApts + 2 * complex.twoPersApts + 4 * complex.fourPersApts)

/-- Calculates the actual occupancy at a given percentage of maximum occupancy -/
def actualOccupancy (complex : ApartmentComplex) (percentage : ℚ) : ℕ :=
  (percentage * (maxOccupancy complex : ℚ)).floor.toNat

/-- The main theorem to prove -/
theorem apartment_complex_occupancy :
  ∃ (complex : ApartmentComplex),
    complex.numBuildings = 4 ∧
    complex.studioApts = 10 ∧
    complex.twoPersApts = 20 ∧
    complex.fourPersApts = 5 ∧
    actualOccupancy complex (3/4) = 210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_complex_occupancy_l1303_130397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_spheres_surface_area_l1303_130377

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A sphere in 3D space -/
structure Sphere where
  center : Point
  radius : ℝ

/-- A cube in 3D space -/
structure Cube where
  center : Point
  side_length : ℝ

/-- The circumference of a sphere -/
noncomputable def Sphere.circumference (s : Sphere) : ℝ := 2 * Real.pi * s.radius

/-- The surface area of a sphere -/
noncomputable def Sphere.surface_area (s : Sphere) : ℝ := 4 * Real.pi * s.radius^2

/-- A cube is inscribed in a sphere -/
def Cube.inscribed_in (c : Cube) (s : Sphere) : Prop :=
  c.side_length = 2 * s.radius

/-- A sphere is inscribed in a cube -/
def Sphere.inscribed_in (s : Sphere) (c : Cube) : Prop :=
  2 * s.radius = c.side_length

/-- Given a sphere with circumference 6π meters, a cube inscribed in this sphere,
    and a second sphere inscribed in this cube, prove that the surface area of
    the second sphere is 36π square meters. -/
theorem inscribed_spheres_surface_area (first_sphere : Sphere) (cube : Cube) (second_sphere : Sphere) :
  (first_sphere.circumference = 6 * Real.pi) →
  (cube.inscribed_in first_sphere) →
  (second_sphere.inscribed_in cube) →
  (second_sphere.surface_area = 36 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_spheres_surface_area_l1303_130377
