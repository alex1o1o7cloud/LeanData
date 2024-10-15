import Mathlib

namespace NUMINAMATH_CALUDE_identical_answers_possible_l2493_249311

/-- A person who either always tells the truth or always lies -/
inductive TruthTeller
  | Always
  | Never

/-- The response to a question, either Yes or No -/
inductive Response
  | Yes
  | No

/-- Given a question, determine the response of a TruthTeller -/
def respond (person : TruthTeller) (questionTruth : Bool) : Response :=
  match person, questionTruth with
  | TruthTeller.Always, true => Response.Yes
  | TruthTeller.Always, false => Response.No
  | TruthTeller.Never, true => Response.No
  | TruthTeller.Never, false => Response.Yes

theorem identical_answers_possible :
  ∃ (question : Bool),
    respond TruthTeller.Always question = respond TruthTeller.Never question :=
by sorry

end NUMINAMATH_CALUDE_identical_answers_possible_l2493_249311


namespace NUMINAMATH_CALUDE_fixed_point_existence_l2493_249376

theorem fixed_point_existence (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x : ℝ, x = a^(x - 2) - 3 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_existence_l2493_249376


namespace NUMINAMATH_CALUDE_part_a_part_b_part_b_max_exists_l2493_249304

-- Part (a)
def P (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x + 2

theorem part_a (k : ℝ) : P k 2 = 0 → k = 5 := by sorry

-- Part (b)
theorem part_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2*a + b + 4/(a*b) = 10 → a ≤ 4 := by sorry

theorem part_b_max_exists :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + b + 4/(a*b) = 10 ∧ a = 4 := by sorry

end NUMINAMATH_CALUDE_part_a_part_b_part_b_max_exists_l2493_249304


namespace NUMINAMATH_CALUDE_escalator_time_l2493_249395

/-- Time taken to cover an escalator's length -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) : 
  escalator_speed = 9 →
  person_speed = 3 →
  escalator_length = 200 →
  (escalator_length / (escalator_speed + person_speed)) = 200 / (9 + 3) := by
  sorry

end NUMINAMATH_CALUDE_escalator_time_l2493_249395


namespace NUMINAMATH_CALUDE_median_to_mean_l2493_249360

theorem median_to_mean (m : ℝ) : 
  let s : Finset ℝ := {m, m + 4, m + 7, m + 10, m + 16}
  m + 7 = 12 →
  (s.sum id) / s.card = 12.4 := by
sorry

end NUMINAMATH_CALUDE_median_to_mean_l2493_249360


namespace NUMINAMATH_CALUDE_minimum_value_of_F_l2493_249392

/-- A function is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem minimum_value_of_F (m n : ℝ) (f g : ℝ → ℝ) :
  (∀ x > 0, f x + n * g x + x + 2 ≤ 8) →
  OddFunction f →
  OddFunction g →
  ∃ c, c = -4 ∧ ∀ x < 0, m * f x + n * g x + x + 2 ≥ c :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_F_l2493_249392


namespace NUMINAMATH_CALUDE_problem_solving_questions_count_l2493_249346

-- Define the total number of multiple-choice questions
def total_mc : ℕ := 35

-- Define the fraction of multiple-choice questions already written
def mc_written_fraction : ℚ := 2/5

-- Define the fraction of problem-solving questions already written
def ps_written_fraction : ℚ := 1/3

-- Define the total number of remaining questions to write
def remaining_questions : ℕ := 31

-- Theorem to prove
theorem problem_solving_questions_count :
  ∃ (total_ps : ℕ),
    (total_ps : ℚ) * (1 - ps_written_fraction) + 
    (total_mc : ℚ) * (1 - mc_written_fraction) = remaining_questions ∧
    total_ps = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_questions_count_l2493_249346


namespace NUMINAMATH_CALUDE_travel_expense_fraction_l2493_249379

theorem travel_expense_fraction (initial_amount : ℝ) 
  (clothes_fraction : ℝ) (food_fraction : ℝ) (final_amount : ℝ) :
  initial_amount = 1499.9999999999998 →
  clothes_fraction = 1/3 →
  food_fraction = 1/5 →
  final_amount = 600 →
  let remaining_after_clothes := initial_amount * (1 - clothes_fraction)
  let remaining_after_food := remaining_after_clothes * (1 - food_fraction)
  (remaining_after_food - final_amount) / remaining_after_food = 1/4 := by
sorry

end NUMINAMATH_CALUDE_travel_expense_fraction_l2493_249379


namespace NUMINAMATH_CALUDE_base_5_103_eq_28_l2493_249371

/-- Converts a list of digits in base b to its decimal representation -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The decimal representation of 103 in base 5 -/
def base_5_103 : Nat := to_decimal [3, 0, 1] 5

theorem base_5_103_eq_28 : base_5_103 = 28 := by sorry

end NUMINAMATH_CALUDE_base_5_103_eq_28_l2493_249371


namespace NUMINAMATH_CALUDE_polynomial_independence_l2493_249317

-- Define the polynomials A and B
def A (x a : ℝ) : ℝ := x^2 + a*x
def B (x b : ℝ) : ℝ := 2*b*x^2 - 4*x - 1

-- Define the combined polynomial 2A + B
def combined_polynomial (x a b : ℝ) : ℝ := 2 * A x a + B x b

-- Theorem statement
theorem polynomial_independence (a b : ℝ) : 
  (∀ x : ℝ, ∃ c : ℝ, combined_polynomial x a b = c) ↔ (a = 2 ∧ b = -1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_independence_l2493_249317


namespace NUMINAMATH_CALUDE_num_four_digit_numbers_eq_twelve_l2493_249393

/-- The number of different four-digit numbers that can be formed using the cards "2", "0", "0", "9" (where "9" can also be used as "6") -/
def num_four_digit_numbers : ℕ :=
  (Nat.choose 3 2) * 2 * (Nat.factorial 2)

/-- Theorem stating that the number of different four-digit numbers is 12 -/
theorem num_four_digit_numbers_eq_twelve : num_four_digit_numbers = 12 := by
  sorry

#eval num_four_digit_numbers

end NUMINAMATH_CALUDE_num_four_digit_numbers_eq_twelve_l2493_249393


namespace NUMINAMATH_CALUDE_ellipse_smallest_area_l2493_249347

/-- Given an ellipse that contains two specific circles, prove its smallest possible area -/
theorem ellipse_smallest_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) :
  ∃ k : ℝ, k = Real.sqrt 3 / 3 ∧ 
    ∀ a' b' : ℝ, (∀ x y : ℝ, x^2/a'^2 + y^2/b'^2 = 1 → 
      ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) →
    π * a' * b' ≥ k * π :=
by sorry

end NUMINAMATH_CALUDE_ellipse_smallest_area_l2493_249347


namespace NUMINAMATH_CALUDE_remainder_2017_div_89_l2493_249327

theorem remainder_2017_div_89 : 2017 % 89 = 59 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2017_div_89_l2493_249327


namespace NUMINAMATH_CALUDE_baker_cakes_l2493_249323

theorem baker_cakes (cakes_sold : ℕ) (cakes_remaining : ℕ) (initial_cakes : ℕ) : 
  cakes_sold = 10 → cakes_remaining = 139 → initial_cakes = cakes_sold + cakes_remaining → initial_cakes = 149 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l2493_249323


namespace NUMINAMATH_CALUDE_bianca_carrots_l2493_249331

/-- Proves that Bianca threw out 10 carrots given the initial conditions -/
theorem bianca_carrots (initial : ℕ) (next_day : ℕ) (total : ℕ) (thrown_out : ℕ) 
  (h1 : initial = 23)
  (h2 : next_day = 47)
  (h3 : total = 60)
  (h4 : initial - thrown_out + next_day = total) : 
  thrown_out = 10 := by
  sorry

end NUMINAMATH_CALUDE_bianca_carrots_l2493_249331


namespace NUMINAMATH_CALUDE_jasmine_laps_l2493_249324

/-- Calculates the total number of laps swum over a period of weeks -/
def total_laps (laps_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  laps_per_day * days_per_week * num_weeks

/-- Proves that Jasmine swims 300 laps in 5 weeks -/
theorem jasmine_laps : total_laps 12 5 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_laps_l2493_249324


namespace NUMINAMATH_CALUDE_sum_of_areas_l2493_249390

/-- A sequence of circles tangent to two half-lines -/
structure TangentCircles where
  d₁ : ℝ
  r₁ : ℝ
  d : ℕ → ℝ
  r : ℕ → ℝ
  h₁ : d₁ > 0
  h₂ : r₁ > 0
  h₃ : ∀ n : ℕ, d n > 0
  h₄ : ∀ n : ℕ, r n > 0
  h₅ : d 1 = d₁
  h₆ : r 1 = r₁
  h₇ : ∀ n : ℕ, n > 1 → d n < d (n-1)
  h₈ : ∀ n : ℕ, r n / d n = r₁ / d₁

theorem sum_of_areas (tc : TangentCircles) :
  (∑' n, π * (tc.r n)^2) = (π/4) * (tc.r₁ * (tc.d₁ + tc.r₁)^2 / tc.d₁) :=
sorry

end NUMINAMATH_CALUDE_sum_of_areas_l2493_249390


namespace NUMINAMATH_CALUDE_optimal_price_reduction_and_profit_l2493_249382

/-- Represents the daily profit function for a flower shop -/
def profit_function (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

/-- Represents the constraints on the price reduction -/
def valid_price_reduction (x : ℝ) : Prop := 0 ≤ x ∧ x < 40

/-- Theorem stating the optimal price reduction and maximum profit -/
theorem optimal_price_reduction_and_profit :
  ∃ (x : ℝ), valid_price_reduction x ∧ 
    (∀ y, valid_price_reduction y → profit_function y ≤ profit_function x) ∧
    x = 15 ∧ profit_function x = 1250 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_reduction_and_profit_l2493_249382


namespace NUMINAMATH_CALUDE_max_ab_value_l2493_249314

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := 12 * x^2 - 2 * a * x - 2 * b

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f_deriv a b 1 = 0) :
  (∃ (max_ab : ℝ), max_ab = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → f_deriv a' b' 1 = 0 → a' * b' ≤ max_ab) :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l2493_249314


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l2493_249308

theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5) →  -- consecutive odd integers
  (a + b + c = -147) →                                  -- sum is -147
  (max a (max b c) = -47) :=                            -- largest is -47
by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l2493_249308


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_one_l2493_249339

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a ≥ 0}

-- Theorem statement
theorem intersection_nonempty_implies_a_greater_than_one (a : ℝ) :
  (∃ x, x ∈ A ∩ B a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_one_l2493_249339


namespace NUMINAMATH_CALUDE_nested_function_ratio_l2493_249377

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

def g (x : ℝ) : ℝ := 2 * x^2 - x + 1

theorem nested_function_ratio :
  f (g (f 1)) / g (f (g 1)) = 6801 / 281 := by
  sorry

end NUMINAMATH_CALUDE_nested_function_ratio_l2493_249377


namespace NUMINAMATH_CALUDE_least_number_divisible_l2493_249315

theorem least_number_divisible (n : ℕ) : n = 857 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    m + 7 = 24 * k₁ ∧ 
    m + 7 = 32 * k₂ ∧ 
    m + 7 = 36 * k₃ ∧ 
    m + 7 = 54 * k₄)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    n + 7 = 24 * k₁ ∧ 
    n + 7 = 32 * k₂ ∧ 
    n + 7 = 36 * k₃ ∧ 
    n + 7 = 54 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_l2493_249315


namespace NUMINAMATH_CALUDE_roadway_deck_concrete_amount_l2493_249307

/-- The amount of concrete needed for the roadway deck of a bridge -/
def roadway_deck_concrete (total_concrete : ℕ) (anchor_concrete : ℕ) (pillar_concrete : ℕ) : ℕ :=
  total_concrete - (2 * anchor_concrete + pillar_concrete)

/-- Theorem stating that the roadway deck needs 1600 tons of concrete -/
theorem roadway_deck_concrete_amount :
  roadway_deck_concrete 4800 700 1800 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_roadway_deck_concrete_amount_l2493_249307


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2493_249356

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -2; 4, 3]
  Matrix.det A = 23 := by sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2493_249356


namespace NUMINAMATH_CALUDE_ten_row_triangle_pieces_l2493_249343

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculates the nth triangular number -/
def triangularNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Represents the structure of a triangle made of rods and connectors -/
structure RodTriangle where
  rows : ℕ
  firstRowRods : ℕ
  rodIncrement : ℕ

/-- Calculates the total number of rods in a RodTriangle -/
def totalRods (t : RodTriangle) : ℕ :=
  arithmeticSum t.firstRowRods t.rodIncrement t.rows

/-- Calculates the total number of connectors in a RodTriangle -/
def totalConnectors (t : RodTriangle) : ℕ :=
  triangularNumber (t.rows + 1)

/-- Calculates the total number of pieces (rods and connectors) in a RodTriangle -/
def totalPieces (t : RodTriangle) : ℕ :=
  totalRods t + totalConnectors t

/-- Theorem: The total number of pieces in a ten-row triangle is 231 -/
theorem ten_row_triangle_pieces :
  totalPieces { rows := 10, firstRowRods := 3, rodIncrement := 3 } = 231 := by
  sorry

end NUMINAMATH_CALUDE_ten_row_triangle_pieces_l2493_249343


namespace NUMINAMATH_CALUDE_inequality_proof_l2493_249380

theorem inequality_proof (a b c : ℝ) (h : a^6 + b^6 + c^6 = 3) :
  a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2493_249380


namespace NUMINAMATH_CALUDE_pyramid_volume_l2493_249300

theorem pyramid_volume (base_side : ℝ) (height : ℝ) (volume : ℝ) :
  base_side = 1 / 2 →
  height = 1 →
  volume = (1 / 3) * (base_side ^ 2) * height →
  volume = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2493_249300


namespace NUMINAMATH_CALUDE_light_toggle_theorem_l2493_249316

/-- Represents the state of a light (on or off) -/
inductive LightState
| Off
| On

/-- Represents a position in the 5x5 grid -/
structure Position where
  row : Fin 5
  col : Fin 5

/-- The type of the 5x5 grid of lights -/
def Grid := Fin 5 → Fin 5 → LightState

/-- Toggles a light and its adjacent lights in the same row and column -/
def toggle (grid : Grid) (pos : Position) : Grid := sorry

/-- Checks if exactly one light is on in the grid -/
def exactlyOneOn (grid : Grid) : Prop := sorry

/-- The set of possible positions for the single on light -/
def possiblePositions : Set Position :=
  {⟨2, 2⟩, ⟨2, 4⟩, ⟨3, 3⟩, ⟨4, 2⟩, ⟨4, 4⟩}

/-- The initial grid with all lights off -/
def initialGrid : Grid := fun _ _ => LightState.Off

theorem light_toggle_theorem :
  ∀ (finalGrid : Grid),
    (∃ (toggleSequence : List Position),
      finalGrid = toggleSequence.foldl toggle initialGrid) →
    exactlyOneOn finalGrid →
    ∃ (pos : Position), finalGrid pos.row pos.col = LightState.On ∧ pos ∈ possiblePositions :=
sorry

end NUMINAMATH_CALUDE_light_toggle_theorem_l2493_249316


namespace NUMINAMATH_CALUDE_max_value_of_x2_plus_y2_l2493_249364

theorem max_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 = 2*x - 2*y + 2) :
  x^2 + y^2 ≤ 6 + 4 * Real.sqrt 2 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 6 + 4 * Real.sqrt 2 ∧ x₀^2 + y₀^2 = 2*x₀ - 2*y₀ + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_x2_plus_y2_l2493_249364


namespace NUMINAMATH_CALUDE_inequality_proof_l2493_249330

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a * b + b * c + c * a = 1) : 
  Real.sqrt (a^3 + a) + Real.sqrt (b^3 + b) + Real.sqrt (c^3 + c) ≥ 2 * Real.sqrt (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2493_249330


namespace NUMINAMATH_CALUDE_translation_theorem_l2493_249313

def f (x : ℝ) : ℝ := (x - 2)^2 + 2

def g (x : ℝ) : ℝ := (x - 1)^2 + 3

theorem translation_theorem :
  ∀ x : ℝ, g x = f (x + 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l2493_249313


namespace NUMINAMATH_CALUDE_square_difference_squared_l2493_249399

theorem square_difference_squared : (7^2 - 3^2)^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_squared_l2493_249399


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2493_249365

theorem expand_and_simplify (x : ℝ) : 3 * (x - 4) * (x + 9) = 3 * x^2 + 15 * x - 108 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2493_249365


namespace NUMINAMATH_CALUDE_triangle_theorem_l2493_249310

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that for a triangle ABC with parallel vectors (a, √3b) and (cos A, sin B),
    where a = √7 and b = 2, the angle A is π/3 and the area is (3√3)/2. -/
theorem triangle_theorem (t : Triangle) 
    (h1 : t.a * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.A) -- Vectors are parallel
    (h2 : t.a = Real.sqrt 7)
    (h3 : t.b = 2) :
    t.A = π / 3 ∧ (1 / 2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2493_249310


namespace NUMINAMATH_CALUDE_intersection_equals_A_l2493_249318

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_equals_A : A ∩ B = A := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_A_l2493_249318


namespace NUMINAMATH_CALUDE_not_divides_power_minus_one_l2493_249320

theorem not_divides_power_minus_one (n : ℕ) (h : n ≥ 2) : ¬(n ∣ 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_minus_one_l2493_249320


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_mod_7_l2493_249329

def factorial (n : ℕ) : ℕ := sorry

def sum_factorials (n : ℕ) : ℕ := sorry

theorem factorial_sum_remainder_mod_7 : sum_factorials 10 % 7 = 5 := by sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_mod_7_l2493_249329


namespace NUMINAMATH_CALUDE_square_area_ratio_l2493_249321

theorem square_area_ratio : 
  ∀ (s₂ : ℝ), s₂ > 0 →
  let s₁ := s₂ * Real.sqrt 2
  let s₃ := s₁ / 2
  let A₂ := s₂ ^ 2
  let A₃ := s₃ ^ 2
  A₃ / A₂ = 1 / 2 := by
    sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2493_249321


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l2493_249381

theorem ice_cream_combinations (n_flavors m_toppings : ℕ) 
  (h_flavors : n_flavors = 5) 
  (h_toppings : m_toppings = 7) : 
  n_flavors * Nat.choose m_toppings 3 = 175 := by
  sorry

#check ice_cream_combinations

end NUMINAMATH_CALUDE_ice_cream_combinations_l2493_249381


namespace NUMINAMATH_CALUDE_carlos_total_earnings_l2493_249359

-- Define the problem parameters
def hours_week1 : ℕ := 18
def hours_week2 : ℕ := 30
def extra_earnings : ℕ := 54

-- Define Carlos's hourly wage as a rational number
def hourly_wage : ℚ := 54 / 12

-- Theorem statement
theorem carlos_total_earnings :
  (hours_week1 : ℚ) * hourly_wage + (hours_week2 : ℚ) * hourly_wage = 216 := by
  sorry

#eval (hours_week1 : ℚ) * hourly_wage + (hours_week2 : ℚ) * hourly_wage

end NUMINAMATH_CALUDE_carlos_total_earnings_l2493_249359


namespace NUMINAMATH_CALUDE_soda_price_after_increase_l2493_249374

theorem soda_price_after_increase (candy_price : ℝ) (soda_price : ℝ) : 
  candy_price = 10 →
  candy_price + soda_price = 16 →
  9 = soda_price * 1.5 :=
by
  sorry

end NUMINAMATH_CALUDE_soda_price_after_increase_l2493_249374


namespace NUMINAMATH_CALUDE_absolute_value_equals_opposite_implies_nonpositive_l2493_249369

theorem absolute_value_equals_opposite_implies_nonpositive (a : ℝ) :
  (abs a = -a) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_opposite_implies_nonpositive_l2493_249369


namespace NUMINAMATH_CALUDE_range_of_a_l2493_249358

/-- Proposition p: The equation represents a hyperbola -/
def p (a : ℝ) : Prop := 2 - a > 0 ∧ a + 1 > 0

/-- Proposition q: The equation has real roots -/
def q (a : ℝ) : Prop := 16 + 4 * a ≥ 0

/-- The range of a given the negation of p and q is true -/
theorem range_of_a : ∀ a : ℝ, (¬p a ∧ q a) → (a ≤ -1 ∨ a ≥ 2) ∧ a ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2493_249358


namespace NUMINAMATH_CALUDE_special_function_property_l2493_249398

/-- A continuous function f: ℝ → ℝ satisfying f(x) · f(f(x)) = 1 for all real x, and f(1000) = 999 -/
def special_function (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ 
  (∀ x : ℝ, f x * f (f x) = 1) ∧
  f 1000 = 999

theorem special_function_property (f : ℝ → ℝ) (h : special_function f) : 
  f 500 = 1 / 500 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l2493_249398


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2493_249368

/-- Given a sphere with surface area 256π cm², its volume is (2048/3)π cm³. -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
  (4 * Real.pi * r^2 = 256 * Real.pi) → 
  ((4/3) * Real.pi * r^3 = (2048/3) * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2493_249368


namespace NUMINAMATH_CALUDE_exactly_one_tail_in_three_flips_l2493_249370

-- Define a fair coin
def fair_coin_prob : ℝ := 0.5

-- Define the number of flips
def num_flips : ℕ := 3

-- Define the number of tails we want
def num_tails : ℕ := 1

-- Define the binomial coefficient function
def choose (n k : ℕ) : ℕ := 
  Nat.choose n k

-- Define the probability of exactly k successes in n trials
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (choose n k : ℝ) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem exactly_one_tail_in_three_flips :
  binomial_probability num_flips num_tails fair_coin_prob = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_tail_in_three_flips_l2493_249370


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2493_249353

theorem cube_root_equation_solution : 
  ∃! x : ℝ, (7 - x / 3) ^ (1/3 : ℝ) = 5 ∧ x = -354 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2493_249353


namespace NUMINAMATH_CALUDE_katie_sold_four_bead_necklaces_l2493_249333

/-- The number of bead necklaces Katie sold at her garage sale. -/
def bead_necklaces : ℕ := sorry

/-- The number of gem stone necklaces Katie sold. -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars. -/
def cost_per_necklace : ℕ := 3

/-- The total earnings from the necklace sale in dollars. -/
def total_earnings : ℕ := 21

/-- Theorem stating that Katie sold 4 bead necklaces. -/
theorem katie_sold_four_bead_necklaces : 
  bead_necklaces = 4 :=
by sorry

end NUMINAMATH_CALUDE_katie_sold_four_bead_necklaces_l2493_249333


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2493_249342

/-- An isosceles triangle with a circle inscribed in it -/
structure IsoscelesTriangleWithInscribedCircle where
  -- Base of the isosceles triangle
  base : ℝ
  -- Height of the isosceles triangle
  height : ℝ
  -- Radius of the inscribed circle
  radius : ℝ
  -- The circle touches the base and both equal sides of the triangle
  touches_sides : True

/-- Theorem stating that for an isosceles triangle with base 20 and height 24, 
    the radius of the inscribed circle is 20/3 -/
theorem inscribed_circle_radius 
  (triangle : IsoscelesTriangleWithInscribedCircle)
  (h_base : triangle.base = 20)
  (h_height : triangle.height = 24) :
  triangle.radius = 20 / 3 := by
  sorry

#check inscribed_circle_radius

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2493_249342


namespace NUMINAMATH_CALUDE_extremum_implies_deriv_zero_not_always_converse_l2493_249305

open Set
open Function
open Topology

-- Define a structure for differentiable functions on ℝ
structure DiffFunction where
  f : ℝ → ℝ
  diff : Differentiable ℝ f

variable (f : DiffFunction)

-- Define what it means for a function to have an extremum
def has_extremum (f : DiffFunction) : Prop :=
  ∃ x₀ : ℝ, ∀ x : ℝ, f.f x ≤ f.f x₀ ∨ f.f x ≥ f.f x₀

-- Define what it means for f'(x) = 0 to have a solution
def deriv_has_zero (f : DiffFunction) : Prop :=
  ∃ x : ℝ, deriv f.f x = 0

-- State the theorem
theorem extremum_implies_deriv_zero (f : DiffFunction) : 
  has_extremum f → deriv_has_zero f :=
sorry

-- State that the converse is not always true
theorem not_always_converse : 
  ∃ f : DiffFunction, deriv_has_zero f ∧ ¬has_extremum f :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_deriv_zero_not_always_converse_l2493_249305


namespace NUMINAMATH_CALUDE_jerome_contacts_l2493_249362

/-- The number of people on Jerome's contact list -/
def total_contacts (classmates out_of_school_friends parents sisters : ℕ) : ℕ :=
  classmates + out_of_school_friends + parents + sisters

/-- Theorem stating the total number of contacts on Jerome's list -/
theorem jerome_contacts : ∃ (classmates out_of_school_friends parents sisters : ℕ),
  classmates = 20 ∧
  out_of_school_friends = classmates / 2 ∧
  parents = 2 ∧
  sisters = 1 ∧
  total_contacts classmates out_of_school_friends parents sisters = 33 := by
  sorry

end NUMINAMATH_CALUDE_jerome_contacts_l2493_249362


namespace NUMINAMATH_CALUDE_sum_equality_in_subset_l2493_249357

theorem sum_equality_in_subset (S : Finset ℕ) :
  S ⊆ Finset.range 38 →
  S.card = 10 →
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b = c + d := by
  sorry

end NUMINAMATH_CALUDE_sum_equality_in_subset_l2493_249357


namespace NUMINAMATH_CALUDE_power_function_decreasing_n_l2493_249397

/-- A power function f(x) = ax^n where a and n are constants and x > 0 -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), ∀ x > 0, f x = a * x^n

/-- A function f is decreasing on (0, +∞) if for all x, y in (0, +∞) with x < y, f(x) > f(y) -/
def isDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x > f y

/-- The main theorem -/
theorem power_function_decreasing_n (n : ℝ) :
  (isPowerFunction (fun x ↦ (n^2 - n - 1) * x^n)) ∧
  (isDecreasingOn (fun x ↦ (n^2 - n - 1) * x^n)) ↔
  n = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_decreasing_n_l2493_249397


namespace NUMINAMATH_CALUDE_square_sum_equals_34_l2493_249309

theorem square_sum_equals_34 (x y : ℕ+) 
  (h1 : x * y + x + y = 23)
  (h2 : x^2 * y + x * y^2 = 120) : 
  x^2 + y^2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_34_l2493_249309


namespace NUMINAMATH_CALUDE_simplest_fraction_with_conditions_l2493_249349

theorem simplest_fraction_with_conditions (a b : ℕ) : 
  (a : ℚ) / b = 45 / 56 →
  ∃ (x : ℕ), a = x^2 →
  ∃ (y : ℕ), b = y^3 →
  ∃ (c d : ℕ), (c : ℚ) / d = 1 ∧ 
    (∀ (e f : ℕ), (e : ℚ) / f = 45 / 56 → 
      (∃ (g : ℕ), e = g^2) → 
      (∃ (h : ℕ), f = h^3) → 
      (c : ℚ) / d ≤ (e : ℚ) / f) :=
by sorry

end NUMINAMATH_CALUDE_simplest_fraction_with_conditions_l2493_249349


namespace NUMINAMATH_CALUDE_euler_negative_two_i_in_third_quadrant_l2493_249372

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Real.cos z.im + Complex.I * Real.sin z.im)

-- Define the third quadrant
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- Theorem statement
theorem euler_negative_two_i_in_third_quadrant :
  third_quadrant (cexp (-2 * Complex.I)) := by
  sorry

end NUMINAMATH_CALUDE_euler_negative_two_i_in_third_quadrant_l2493_249372


namespace NUMINAMATH_CALUDE_valid_pairs_l2493_249391

def is_valid_pair (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∃ k : ℤ, (k : ℚ) = (a^2 + b : ℚ) / (b^2 - a : ℚ)) ∧
  (∃ m : ℤ, (m : ℚ) = (b^2 + a : ℚ) / (a^2 - b : ℚ))

theorem valid_pairs :
  ∀ a b : ℕ, is_valid_pair a b ↔
    ((a = 2 ∧ b = 2) ∨
     (a = 3 ∧ b = 3) ∨
     (a = 1 ∧ b = 2) ∨
     (a = 2 ∧ b = 1) ∨
     (a = 2 ∧ b = 3) ∨
     (a = 3 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l2493_249391


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l2493_249386

theorem quadratic_factorization_sum (d e f : ℤ) : 
  (∀ x, x^2 + 13*x + 40 = (x + d) * (x + e)) →
  (∀ x, x^2 - 19*x + 88 = (x - e) * (x - f)) →
  d + e + f = 24 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l2493_249386


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2493_249351

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 - 9 = 0} = {-3, 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2493_249351


namespace NUMINAMATH_CALUDE_valid_triplets_are_solution_set_l2493_249375

def is_valid_triplet (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧
  (b + c + 1) % a = 0 ∧
  (c + a + 1) % b = 0 ∧
  (a + b + 1) % c = 0

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(1, 1, 1), (1, 2, 2), (1, 1, 3), (2, 2, 5), (3, 3, 7), (1, 4, 6),
   (2, 6, 9), (3, 8, 12), (4, 10, 15), (5, 12, 18), (6, 14, 21)}

theorem valid_triplets_are_solution_set :
  ∀ a b c : ℕ, is_valid_triplet a b c ↔ (a, b, c) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_valid_triplets_are_solution_set_l2493_249375


namespace NUMINAMATH_CALUDE_gcd_lcm_336_1260_l2493_249361

theorem gcd_lcm_336_1260 : 
  (Nat.gcd 336 1260 = 84) ∧ (Nat.lcm 336 1260 = 5040) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_336_1260_l2493_249361


namespace NUMINAMATH_CALUDE_total_annual_interest_l2493_249336

def total_amount : ℝ := 1600
def interest_rate_x : ℝ := 0.06
def interest_rate_y : ℝ := 0.05
def lent_amount : ℝ := 1100
def lent_interest_rate : ℝ := 0.0500001

theorem total_annual_interest :
  ∀ x y : ℝ,
  x + y = total_amount →
  y = lent_amount →
  x * interest_rate_x + y * lent_interest_rate = 85.00011 := by
sorry

end NUMINAMATH_CALUDE_total_annual_interest_l2493_249336


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l2493_249303

theorem max_value_of_sum_of_squares (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 10) :
  (∃ (m : ℝ), ∀ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 10 →
    (x - y)^2 + (x - z)^2 + (x - w)^2 + (y - z)^2 + (y - w)^2 + (z - w)^2 ≤ m) ∧
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ 40 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l2493_249303


namespace NUMINAMATH_CALUDE_classroom_lights_theorem_l2493_249340

/-- The number of lamps in the classroom -/
def num_lamps : ℕ := 4

/-- The total number of possible states for the lights -/
def total_states : ℕ := 2^num_lamps

/-- The number of ways to turn on the lights, excluding the all-off state -/
def ways_to_turn_on : ℕ := total_states - 1

theorem classroom_lights_theorem : ways_to_turn_on = 15 := by
  sorry

end NUMINAMATH_CALUDE_classroom_lights_theorem_l2493_249340


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2493_249322

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k > 60 ∧ ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ m : ℤ, m ≤ 60 → (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2493_249322


namespace NUMINAMATH_CALUDE_dividend_calculation_l2493_249385

/-- Calculates the total dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : face_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.07) :
  let actual_price := face_value * (1 + premium_rate)
  let num_shares := investment / actual_price
  let dividend_per_share := face_value * dividend_rate
  num_shares * dividend_per_share = 840 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2493_249385


namespace NUMINAMATH_CALUDE_common_point_of_circumcircles_l2493_249326

-- Define the circle S
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define a point being outside a circle
def IsOutside (p : ℝ × ℝ) (s : Set (ℝ × ℝ)) : Prop :=
  p ∉ s

-- Define a line passing through a point
def LineThroughPoint (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | ∃ (t : ℝ), q = (p.1 + t, p.2 + t)}

-- Define the intersection of a line and a circle
def Intersect (l : Set (ℝ × ℝ)) (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ l ∧ p ∈ s}

-- Define the circumcircle of a triangle
def Circumcircle (p1 p2 p3 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry -- Actual definition would be more complex

-- Main theorem
theorem common_point_of_circumcircles
  (S : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ)
  (A B : ℝ × ℝ) :
  S = Circle center radius →
  IsOutside A S →
  IsOutside B S →
  ∃ (C : ℝ × ℝ), C ≠ B ∧
    ∀ (l : Set (ℝ × ℝ)) (M N : ℝ × ℝ),
      l = LineThroughPoint A →
      {M, N} ⊆ Intersect l S →
      C ∈ Circumcircle B M N :=
by sorry

end NUMINAMATH_CALUDE_common_point_of_circumcircles_l2493_249326


namespace NUMINAMATH_CALUDE_largest_common_divisor_36_60_l2493_249328

theorem largest_common_divisor_36_60 : 
  ∃ (n : ℕ), n > 0 ∧ n ∣ 36 ∧ n ∣ 60 ∧ ∀ (m : ℕ), m > 0 ∧ m ∣ 36 ∧ m ∣ 60 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_36_60_l2493_249328


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2493_249352

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 6 →
  b = 8 →
  c^2 = a^2 + b^2 →
  c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2493_249352


namespace NUMINAMATH_CALUDE_book_arrangement_l2493_249302

def arrange_books (total : ℕ) (group1 : ℕ) (group2 : ℕ) : Prop :=
  total = group1 + group2 ∧ 
  Nat.choose total group1 = Nat.choose total group2

theorem book_arrangement : 
  arrange_books 9 4 5 → Nat.choose 9 4 = 126 := by
sorry

end NUMINAMATH_CALUDE_book_arrangement_l2493_249302


namespace NUMINAMATH_CALUDE_sum_bound_l2493_249396

theorem sum_bound (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 4) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  21 ≤ x + y + z ∧ x + y + z ≤ 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_bound_l2493_249396


namespace NUMINAMATH_CALUDE_dad_steps_l2493_249337

theorem dad_steps (dad_masha_ratio : ℕ → ℕ → Prop)
                   (masha_yasha_ratio : ℕ → ℕ → Prop)
                   (masha_yasha_total : ℕ) :
  dad_masha_ratio 3 5 →
  masha_yasha_ratio 3 5 →
  masha_yasha_total = 400 →
  ∃ (dad_steps : ℕ), dad_steps = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_l2493_249337


namespace NUMINAMATH_CALUDE_area_of_four_triangles_l2493_249387

/-- The combined area of four right triangles with legs of 4 and 3 units is 24 square units. -/
theorem area_of_four_triangles :
  let triangle_area := (1 / 2) * 4 * 3
  4 * triangle_area = 24 := by sorry

end NUMINAMATH_CALUDE_area_of_four_triangles_l2493_249387


namespace NUMINAMATH_CALUDE_birdseed_box_content_l2493_249335

/-- The number of grams of seeds in each box of birdseed -/
def grams_per_box : ℕ := 225

/-- The number of boxes Leah bought -/
def boxes_bought : ℕ := 3

/-- The number of boxes Leah already had -/
def boxes_in_pantry : ℕ := 5

/-- The number of grams the parrot eats per week -/
def parrot_consumption : ℕ := 100

/-- The number of grams the cockatiel eats per week -/
def cockatiel_consumption : ℕ := 50

/-- The number of weeks Leah can feed her birds without going back to the store -/
def weeks_of_feeding : ℕ := 12

theorem birdseed_box_content :
  grams_per_box * (boxes_bought + boxes_in_pantry) = 
    (parrot_consumption + cockatiel_consumption) * weeks_of_feeding :=
by
  sorry

#eval grams_per_box

end NUMINAMATH_CALUDE_birdseed_box_content_l2493_249335


namespace NUMINAMATH_CALUDE_floor_of_7_9_l2493_249325

theorem floor_of_7_9 : ⌊(7.9 : ℝ)⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_floor_of_7_9_l2493_249325


namespace NUMINAMATH_CALUDE_math_competition_probabilities_l2493_249355

def number_of_students : ℕ := 5
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3
def students_to_select : ℕ := 2

-- Total number of possible selections
def total_selections : ℕ := Nat.choose number_of_students students_to_select

-- Number of ways to select exactly one boy
def exactly_one_boy_selections : ℕ := number_of_boys * number_of_girls

-- Number of ways to select at least one boy
def at_least_one_boy_selections : ℕ := total_selections - Nat.choose number_of_girls students_to_select

theorem math_competition_probabilities :
  (total_selections = 10) ∧
  (exactly_one_boy_selections / total_selections = 3 / 5) ∧
  (at_least_one_boy_selections / total_selections = 7 / 10) := by
  sorry

end NUMINAMATH_CALUDE_math_competition_probabilities_l2493_249355


namespace NUMINAMATH_CALUDE_sufficient_drivers_and_schedule_l2493_249354

-- Define the duration of trips and rest time
def one_way_trip_duration : ℕ := 160  -- in minutes
def round_trip_duration : ℕ := 320    -- in minutes
def min_rest_duration : ℕ := 60       -- in minutes

-- Define the schedule times (in minutes since midnight)
def driver_a_return : ℕ := 12 * 60 + 40
def driver_d_departure : ℕ := 13 * 60 + 5
def driver_b_return : ℕ := 16 * 60
def driver_a_second_departure : ℕ := 16 * 60 + 10
def driver_b_second_departure : ℕ := 17 * 60 + 30

-- Define the number of drivers
def num_drivers : ℕ := 4

-- Define the end time of the last trip
def last_trip_end : ℕ := 21 * 60 + 30

-- Theorem statement
theorem sufficient_drivers_and_schedule :
  (num_drivers = 4) ∧
  (driver_a_return + min_rest_duration ≤ driver_d_departure) ∧
  (driver_b_return + min_rest_duration ≤ driver_b_second_departure) ∧
  (driver_a_second_departure + round_trip_duration = last_trip_end) ∧
  (last_trip_end ≤ 24 * 60) → 
  (num_drivers ≥ 4) ∧ (last_trip_end = 21 * 60 + 30) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_drivers_and_schedule_l2493_249354


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l2493_249383

theorem reciprocal_of_sum : (1 / ((1 : ℚ) / 4 + (1 : ℚ) / 5)) = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l2493_249383


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l2493_249312

theorem shaded_area_fraction (total_squares : ℕ) (half_shaded : ℕ) (full_shaded : ℕ) :
  total_squares = 18 →
  half_shaded = 10 →
  full_shaded = 3 →
  (half_shaded / 2 + full_shaded : ℚ) / total_squares = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l2493_249312


namespace NUMINAMATH_CALUDE_condition_relationship_l2493_249345

theorem condition_relationship (x : ℝ) :
  (x > 1/3 → 1/x < 3) ∧ ¬(1/x < 3 → x > 1/3) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2493_249345


namespace NUMINAMATH_CALUDE_simplified_expression_equals_sqrt_two_minus_one_l2493_249338

theorem simplified_expression_equals_sqrt_two_minus_one :
  let x : ℝ := Real.sqrt 2 - 1
  (x^2 / (x^2 + 4*x + 4)) / (x / (x + 2)) - (x - 1) / (x + 2) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_sqrt_two_minus_one_l2493_249338


namespace NUMINAMATH_CALUDE_divisible_by_seven_l2493_249344

def number (x : ℕ) : ℕ := 
  666666666666666666666666666666666666666666666666666 * 10^51 + 
  x * 10^50 + 
  555555555555555555555555555555555555555555555555555

theorem divisible_by_seven (x : ℕ) : 
  x < 10 → (number x % 7 = 0 ↔ x = 2 ∨ x = 9) := by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l2493_249344


namespace NUMINAMATH_CALUDE_product_of_roots_l2493_249366

theorem product_of_roots (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂)
  (h₁ : x₁^2 - 2*x₁ = 1)
  (h₂ : x₂^2 - 2*x₂ = 1) : 
  x₁ * x₂ = -1 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2493_249366


namespace NUMINAMATH_CALUDE_train_passes_jogger_l2493_249389

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed train_speed : ℝ) (train_length initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 210 →
  initial_distance = 200 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 41 :=
by sorry

end NUMINAMATH_CALUDE_train_passes_jogger_l2493_249389


namespace NUMINAMATH_CALUDE_a_value_in_set_equality_l2493_249341

theorem a_value_in_set_equality (a b : ℝ) : 
  let A : Set ℝ := {a, b, 2}
  let B : Set ℝ := {2, b^2, 2*a}
  A ∩ B = A ∪ B → a = 0 ∨ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_a_value_in_set_equality_l2493_249341


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l2493_249301

theorem pentagon_area_sum (u v : ℤ) : 
  0 < v → v < u → (u^2 + 3*u*v = 451) → u + v = 21 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l2493_249301


namespace NUMINAMATH_CALUDE_depth_multiplier_is_fifteen_l2493_249332

/-- The depth of water in feet -/
def water_depth : ℕ := 255

/-- Ron's height in feet -/
def ron_height : ℕ := 13

/-- The difference between Dean's and Ron's heights in feet -/
def height_difference : ℕ := 4

/-- Dean's height in feet -/
def dean_height : ℕ := ron_height + height_difference

/-- The multiplier for Dean's height to find the depth of the water -/
def depth_multiplier : ℕ := water_depth / dean_height

theorem depth_multiplier_is_fifteen :
  depth_multiplier = 15 :=
by sorry

end NUMINAMATH_CALUDE_depth_multiplier_is_fifteen_l2493_249332


namespace NUMINAMATH_CALUDE_no_odd_multiples_of_6_or_8_up_to_60_l2493_249348

theorem no_odd_multiples_of_6_or_8_up_to_60 : 
  ¬∃ n : ℕ, n ≤ 60 ∧ n % 2 = 1 ∧ (n % 6 = 0 ∨ n % 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_odd_multiples_of_6_or_8_up_to_60_l2493_249348


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2493_249350

theorem inscribed_square_area (circle_area : ℝ) (square_area : ℝ) : 
  circle_area = 25 * Real.pi → 
  ∃ (r : ℝ), 
    circle_area = Real.pi * r^2 ∧ 
    square_area = 2 * r^2 →
    square_area = 50 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2493_249350


namespace NUMINAMATH_CALUDE_expected_black_balls_l2493_249363

/-- The expected number of black balls drawn when drawing 3 balls without replacement from a bag containing 5 red balls and 2 black balls. -/
theorem expected_black_balls (total : Nat) (red : Nat) (black : Nat) (drawn : Nat) 
  (h_total : total = 7)
  (h_red : red = 5)
  (h_black : black = 2)
  (h_drawn : drawn = 3)
  (h_sum : red + black = total) :
  (0 : ℚ) * (Nat.choose red drawn : ℚ) / (Nat.choose total drawn : ℚ) +
  (1 : ℚ) * (Nat.choose red (drawn - 1) * Nat.choose black 1 : ℚ) / (Nat.choose total drawn : ℚ) +
  (2 : ℚ) * (Nat.choose red (drawn - 2) * Nat.choose black 2 : ℚ) / (Nat.choose total drawn : ℚ) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expected_black_balls_l2493_249363


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l2493_249388

/-- For an isosceles right triangle with legs of length a, 
    the ratio of twice a leg to the hypotenuse is √2 -/
theorem isosceles_right_triangle_ratio (a : ℝ) (h : a > 0) : 
  (2 * a) / Real.sqrt (a^2 + a^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l2493_249388


namespace NUMINAMATH_CALUDE_place_eight_among_twelve_l2493_249319

/-- The number of ways to place black balls among white balls without adjacency. -/
def place_balls (white : ℕ) (black : ℕ) : ℕ :=
  Nat.choose (white + 1) black

/-- Theorem: Placing 8 black balls among 12 white balls without adjacency. -/
theorem place_eight_among_twelve :
  place_balls 12 8 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_place_eight_among_twelve_l2493_249319


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2493_249394

theorem cube_root_simplification :
  (20^3 + 30^3 + 50^3 : ℝ)^(1/3) = 10 * 160^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2493_249394


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l2493_249334

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.black + b.yellow + b.blue

/-- Defines the bags A, B, C, and D -/
def bagA : Bag := { white := 4, black := 5 }
def bagB : Bag := { yellow := 7, blue := 3 }
def bagC : Bag := { yellow := 3, blue := 6 }
def bagD : Bag := { yellow := 5, blue := 4 }

/-- Calculates the probability of drawing a yellow marble given the problem conditions -/
def yellowProbability : ℚ :=
  let pWhiteA := bagA.white / bagA.total
  let pBlackA := bagA.black / bagA.total
  let pYellowB := bagB.yellow / bagB.total
  let pBlueB := bagB.blue / bagB.total
  let pYellowC := bagC.yellow / bagC.total
  let pBlueC := bagC.blue / bagC.total
  let pYellowD := bagD.yellow / bagD.total
  pWhiteA * pYellowB + pBlackA * pYellowC + pWhiteA * pBlueB * pYellowD + pBlackA * pBlueC * pYellowD

/-- The main theorem stating that the probability of drawing a yellow marble is 1884/3645 -/
theorem yellow_marble_probability : yellowProbability = 1884 / 3645 := by
  sorry


end NUMINAMATH_CALUDE_yellow_marble_probability_l2493_249334


namespace NUMINAMATH_CALUDE_total_cards_l2493_249378

/-- The number of cards each person has -/
structure CardCounts where
  heike : ℕ
  anton : ℕ
  ann : ℕ
  bertrand : ℕ

/-- The conditions of the card counting problem -/
def card_problem (c : CardCounts) : Prop :=
  c.anton = 3 * c.heike ∧
  c.ann = 6 * c.heike ∧
  c.bertrand = 2 * c.heike ∧
  c.ann = 60

/-- The theorem stating that under the given conditions, 
    the total number of cards is 120 -/
theorem total_cards (c : CardCounts) : 
  card_problem c → c.heike + c.anton + c.ann + c.bertrand = 120 := by
  sorry


end NUMINAMATH_CALUDE_total_cards_l2493_249378


namespace NUMINAMATH_CALUDE_floor_of_three_point_six_l2493_249367

theorem floor_of_three_point_six : ⌊(3.6 : ℝ)⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_of_three_point_six_l2493_249367


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2493_249306

/-- The selling price of the computer table -/
def selling_price : ℝ := 5750

/-- The markup percentage applied by the shop owner -/
def markup_percentage : ℝ := 15

/-- The cost price of the computer table -/
def cost_price : ℝ := 5000

/-- Theorem stating that the given cost price is correct based on the selling price and markup -/
theorem cost_price_calculation : 
  selling_price = cost_price * (1 + markup_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2493_249306


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2493_249384

theorem quadratic_equation_result (a : ℝ) (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2493_249384


namespace NUMINAMATH_CALUDE_integer_solution_of_inequalities_l2493_249373

theorem integer_solution_of_inequalities :
  ∃! (x : ℤ), (3 * x - 4 ≤ 6 * x - 2) ∧ ((2 * x + 1) / 3 - 1 < (x - 1) / 2) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_of_inequalities_l2493_249373
