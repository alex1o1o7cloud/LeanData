import Mathlib

namespace NUMINAMATH_CALUDE_squared_difference_product_l1552_155299

theorem squared_difference_product (a b : ℝ) : 
  a = 4 + 2 * Real.sqrt 5 → 
  b = 4 - 2 * Real.sqrt 5 → 
  a^2 * b - a * b^2 = -16 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_squared_difference_product_l1552_155299


namespace NUMINAMATH_CALUDE_largest_valid_sequence_length_l1552_155273

def isPrimePower (n : ℕ) : Prop :=
  ∃ p k, Prime p ∧ n = p ^ k

def validSequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i, i ≤ n → isPrimePower (a i)) ∧
  (∀ i, 3 ≤ i ∧ i ≤ n → a i = a (i - 1) + a (i - 2))

theorem largest_valid_sequence_length :
  (∃ a : ℕ → ℕ, validSequence a 7) ∧
  (∀ n : ℕ, n > 7 → ¬∃ a : ℕ → ℕ, validSequence a n) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_sequence_length_l1552_155273


namespace NUMINAMATH_CALUDE_real_number_inequalities_l1552_155260

theorem real_number_inequalities (a b c : ℝ) :
  (∀ (c : ℝ), c ≠ 0 → (a * c^2 > b * c^2 → a > b)) ∧
  (a < b ∧ b < 0 → a^2 > a * b) ∧
  (∃ (a b c : ℝ), c > a ∧ a > b ∧ b > 0 ∧ a / (c - a) ≥ b / (c - b)) ∧
  (a > b ∧ b > 1 → a - 1 / b > b - 1 / a) :=
by sorry

end NUMINAMATH_CALUDE_real_number_inequalities_l1552_155260


namespace NUMINAMATH_CALUDE_generating_function_value_at_one_intersection_point_on_generating_function_l1552_155219

/-- Linear function -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- Generating function of two linear functions -/
def generatingFunction (f₁ f₂ : LinearFunction) (m n : ℝ) (x : ℝ) : ℝ :=
  m * (f₁.a * x + f₁.b) + n * (f₂.a * x + f₂.b)

/-- Theorem: The value of the generating function of y = x + 1 and y = 2x when x = 1 is 2 -/
theorem generating_function_value_at_one :
  ∀ (m n : ℝ), m + n = 1 →
  generatingFunction ⟨1, 1⟩ ⟨2, 0⟩ m n 1 = 2 := by
  sorry

/-- Theorem: The intersection point of two linear functions lies on their generating function -/
theorem intersection_point_on_generating_function (f₁ f₂ : LinearFunction) (m n : ℝ) :
  m + n = 1 →
  ∀ (x y : ℝ),
  (f₁.a * x + f₁.b = y ∧ f₂.a * x + f₂.b = y) →
  generatingFunction f₁ f₂ m n x = y := by
  sorry

end NUMINAMATH_CALUDE_generating_function_value_at_one_intersection_point_on_generating_function_l1552_155219


namespace NUMINAMATH_CALUDE_compound_proposition_falsehood_l1552_155297

theorem compound_proposition_falsehood (p q : Prop) : 
  ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_falsehood_l1552_155297


namespace NUMINAMATH_CALUDE_tree_height_difference_l1552_155256

-- Define the heights of the trees
def pine_height : ℚ := 49/4
def maple_height : ℚ := 75/4

-- Define the height difference
def height_difference : ℚ := maple_height - pine_height

-- Theorem to prove
theorem tree_height_difference :
  height_difference = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l1552_155256


namespace NUMINAMATH_CALUDE_same_solution_implies_m_equals_9_l1552_155208

theorem same_solution_implies_m_equals_9 :
  ∀ y m : ℝ,
  (y + 3 * m = 32) ∧ (y - 4 = 1) →
  m = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_m_equals_9_l1552_155208


namespace NUMINAMATH_CALUDE_cyclist_problem_solution_l1552_155289

/-- Represents the cyclist's journey to the bus stop -/
structure CyclistJourney where
  usual_speed : ℝ
  usual_time : ℝ
  reduced_speed_ratio : ℝ
  miss_time : ℝ
  bus_cover_ratio : ℝ

/-- Theorem stating the solution to the cyclist problem -/
theorem cyclist_problem_solution (journey : CyclistJourney) 
  (h1 : journey.reduced_speed_ratio = 4/5)
  (h2 : journey.miss_time = 5)
  (h3 : journey.bus_cover_ratio = 1/3)
  (h4 : journey.usual_time * journey.reduced_speed_ratio = journey.usual_time + journey.miss_time)
  (h5 : journey.usual_time > 0) :
  journey.usual_time = 20 ∧ 
  (journey.usual_time * journey.bus_cover_ratio = journey.usual_time * (1 - journey.bus_cover_ratio)) := by
  sorry

#check cyclist_problem_solution

end NUMINAMATH_CALUDE_cyclist_problem_solution_l1552_155289


namespace NUMINAMATH_CALUDE_quadratic_with_one_solution_l1552_155286

theorem quadratic_with_one_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →
  a + c = 11 →
  a < c →
  (a = (11 - Real.sqrt 21) / 2 ∧ c = (11 + Real.sqrt 21) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_with_one_solution_l1552_155286


namespace NUMINAMATH_CALUDE_cost_per_box_l1552_155218

-- Define the box dimensions
def box_length : ℝ := 20
def box_width : ℝ := 20
def box_height : ℝ := 12

-- Define the total volume of the collection
def total_volume : ℝ := 1920000

-- Define the minimum total cost for boxes
def min_total_cost : ℝ := 200

-- Theorem to prove
theorem cost_per_box :
  let box_volume := box_length * box_width * box_height
  let num_boxes := total_volume / box_volume
  let cost_per_box := min_total_cost / num_boxes
  cost_per_box = 0.5 := by sorry

end NUMINAMATH_CALUDE_cost_per_box_l1552_155218


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1552_155237

theorem arithmetic_calculation : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1552_155237


namespace NUMINAMATH_CALUDE_arrangement_theorem_l1552_155254

def number_of_arrangements (n m : ℕ) : ℕ := Nat.choose n m * Nat.factorial m

theorem arrangement_theorem : number_of_arrangements 6 4 = 360 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l1552_155254


namespace NUMINAMATH_CALUDE_bicycle_problem_l1552_155240

/-- Proves that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 12 ∧
  speed_ratio = 1.2 ∧
  time_difference = 1/6 →
  ∃ (speed_B : ℝ),
    speed_B = 12 ∧
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference :=
by sorry

end NUMINAMATH_CALUDE_bicycle_problem_l1552_155240


namespace NUMINAMATH_CALUDE_last_four_average_l1552_155247

theorem last_four_average (numbers : Fin 7 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 13)
  (h2 : numbers 4 + numbers 5 + numbers 6 = 55)
  (h3 : numbers 3 ^ 2 = numbers 6)
  (h4 : numbers 6 = 25) :
  (numbers 3 + numbers 4 + numbers 5 + numbers 6) / 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_last_four_average_l1552_155247


namespace NUMINAMATH_CALUDE_sequence_286_ends_l1552_155209

/-- A sequence of seven positive integers where each number differs by one from its neighbors -/
def ValidSequence (a b c d e f g : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧
  (b = a + 1 ∨ b = a - 1) ∧
  (c = b + 1 ∨ c = b - 1) ∧
  (d = c + 1 ∨ d = c - 1) ∧
  (e = d + 1 ∨ e = d - 1) ∧
  (f = e + 1 ∨ f = e - 1) ∧
  (g = f + 1 ∨ g = f - 1)

theorem sequence_286_ends (a b c d e f g : ℕ) :
  ValidSequence a b c d e f g →
  a + b + c + d + e + f + g = 2017 →
  (a = 286 ∨ g = 286) ∧ b ≠ 286 ∧ c ≠ 286 ∧ d ≠ 286 ∧ e ≠ 286 ∧ f ≠ 286 :=
by sorry

end NUMINAMATH_CALUDE_sequence_286_ends_l1552_155209


namespace NUMINAMATH_CALUDE_random_event_identification_l1552_155228

-- Define the three events
def event1 : Prop := ∃ x y : ℝ, x * y < 0 ∧ x + y < 0
def event2 : Prop := ∀ x y : ℝ, x * y < 0 → x * y > 0
def event3 : Prop := ∀ x y : ℝ, x * y < 0 → x / y < 0

-- Define what it means for an event to be certain
def is_certain (e : Prop) : Prop := e ∨ ¬e

-- Theorem stating that event1 is not certain, while event2 and event3 are certain
theorem random_event_identification :
  ¬(is_certain event1) ∧ (is_certain event2) ∧ (is_certain event3) :=
sorry

end NUMINAMATH_CALUDE_random_event_identification_l1552_155228


namespace NUMINAMATH_CALUDE_cannot_generate_AC_l1552_155222

/-- Represents a sequence of letters --/
inductive Sequence
| empty : Sequence
| cons : Char → Sequence → Sequence

/-- Checks if a sequence ends with the letter B --/
def endsWithB : Sequence → Bool := sorry

/-- Checks if a sequence starts with the letter A --/
def startsWithA : Sequence → Bool := sorry

/-- Counts the number of consecutive B's in a sequence --/
def countConsecutiveB : Sequence → Nat := sorry

/-- Counts the number of consecutive C's in a sequence --/
def countConsecutiveC : Sequence → Nat := sorry

/-- Applies Rule I: If a sequence ends with B, append C --/
def applyRuleI : Sequence → Sequence := sorry

/-- Applies Rule II: If a sequence starts with A, double the sequence after A --/
def applyRuleII : Sequence → Sequence := sorry

/-- Applies Rule III: Replace BBB with C anywhere in the sequence --/
def applyRuleIII : Sequence → Sequence := sorry

/-- Applies Rule IV: Remove CC anywhere in the sequence --/
def applyRuleIV : Sequence → Sequence := sorry

/-- Checks if a sequence is equal to "AC" --/
def isAC : Sequence → Bool := sorry

/-- Initial sequence "AB" --/
def initialSequence : Sequence := sorry

/-- Represents all sequences that can be generated from the initial sequence --/
inductive GeneratedSequence : Sequence → Prop
| initial : GeneratedSequence initialSequence
| rule1 {s : Sequence} : GeneratedSequence s → endsWithB s = true → GeneratedSequence (applyRuleI s)
| rule2 {s : Sequence} : GeneratedSequence s → startsWithA s = true → GeneratedSequence (applyRuleII s)
| rule3 {s : Sequence} : GeneratedSequence s → countConsecutiveB s ≥ 3 → GeneratedSequence (applyRuleIII s)
| rule4 {s : Sequence} : GeneratedSequence s → countConsecutiveC s ≥ 2 → GeneratedSequence (applyRuleIV s)

theorem cannot_generate_AC :
  ∀ s, GeneratedSequence s → isAC s = false := by sorry

end NUMINAMATH_CALUDE_cannot_generate_AC_l1552_155222


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l1552_155223

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, m₁ * x - y = b₁ ↔ m₂ * x - y = b₂) ↔ m₁ = m₂

/-- The value of m for which mx - y - 1 = 0 is parallel to x - 2y + 3 = 0 -/
theorem parallel_lines_m_value :
  (∀ x y, m * x - y - 1 = 0 ↔ x - 2 * y + 3 = 0) → m = 1 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_m_value_l1552_155223


namespace NUMINAMATH_CALUDE_vector_translation_result_l1552_155233

def vector_translation (a : ℝ × ℝ) (right : ℝ) (down : ℝ) : ℝ × ℝ :=
  (a.1 + right, a.2 - down)

theorem vector_translation_result :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := vector_translation a 2 1
  b = (3, 0) := by sorry

end NUMINAMATH_CALUDE_vector_translation_result_l1552_155233


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1552_155279

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x = 0 → x^2 - 2*x = 0) ∧ ¬(x^2 - 2*x = 0 → x = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1552_155279


namespace NUMINAMATH_CALUDE_smallest_m_correct_l1552_155296

/-- The smallest positive value of m for which 10x^2 - mx + 660 = 0 has integral solutions -/
def smallest_m : ℕ := 170

/-- A function representing the quadratic equation 10x^2 - mx + 660 = 0 -/
def quadratic (m : ℕ) (x : ℤ) : ℤ := 10 * x^2 - m * x + 660

theorem smallest_m_correct :
  (∃ x y : ℤ, quadratic smallest_m x = 0 ∧ quadratic smallest_m y = 0 ∧ x ≠ y) ∧
  (∀ m : ℕ, m < smallest_m → ¬∃ x y : ℤ, quadratic m x = 0 ∧ quadratic m y = 0 ∧ x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_correct_l1552_155296


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1552_155250

/-- Mike's current age -/
def m : ℕ := sorry

/-- Ana's current age -/
def a : ℕ := sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem age_ratio_problem :
  (m - 3 = 4 * (a - 3)) ∧ 
  (m - 7 = 5 * (a - 7)) →
  x = 77 ∧ 
  (m + x) * 2 = (a + x) * 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1552_155250


namespace NUMINAMATH_CALUDE_angle_A_value_l1552_155216

theorem angle_A_value (A B C : Real) (a b : Real) : 
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  A + B + C = π →    -- sum of angles in a triangle
  a > 0 →            -- side length is positive
  b > 0 →            -- side length is positive
  2 * a * Real.sin B = b →  -- given condition
  a / Real.sin A = b / Real.sin B →  -- law of sines
  A = π/6 := by
sorry

end NUMINAMATH_CALUDE_angle_A_value_l1552_155216


namespace NUMINAMATH_CALUDE_unique_solution_l1552_155287

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

/-- The main theorem stating that -3/4 is the unique solution -/
theorem unique_solution (a : ℝ) (h : a ≠ 0) :
  f a (1 - a) = f a (1 + a) ↔ a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1552_155287


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1552_155229

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum : ℕ → ℝ
  sum_def : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  term_def : ∀ n, a n = a 1 + (n - 1) * d

/-- Theorem: For an arithmetic sequence where S_5 = 3(a_2 + a_8), a_5 / a_3 = 5/6 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.sum 5 = 3 * (seq.a 2 + seq.a 8)) :
  seq.a 5 / seq.a 3 = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1552_155229


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1552_155294

def standard_die := Finset.range 6

def valid_roll (a b c : ℕ) : Prop :=
  (a - 1) * (b - 1) * (c - 1) * (6 - a) * (6 - b) * (6 - c) ≠ 0

def total_outcomes : ℕ := standard_die.card ^ 3

def successful_outcomes : ℕ := ({2, 3, 4, 5} : Finset ℕ).card ^ 3

theorem dice_roll_probability :
  (successful_outcomes : ℚ) / total_outcomes = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1552_155294


namespace NUMINAMATH_CALUDE_buddy_cards_bought_l1552_155263

/-- Represents the number of baseball cards Buddy has on each day --/
structure BuddyCards where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Represents the number of cards Buddy bought on Wednesday and Thursday --/
structure CardsBought where
  wednesday : ℕ
  thursday : ℕ

/-- The theorem statement --/
theorem buddy_cards_bought (cards : BuddyCards) (bought : CardsBought) : 
  cards.monday = 30 →
  cards.tuesday = cards.monday / 2 →
  cards.thursday = 32 →
  bought.thursday = cards.tuesday / 3 →
  cards.wednesday = cards.tuesday + bought.wednesday →
  cards.thursday = cards.wednesday + bought.thursday →
  bought.wednesday = 12 := by
  sorry


end NUMINAMATH_CALUDE_buddy_cards_bought_l1552_155263


namespace NUMINAMATH_CALUDE_triangle_inequality_with_heights_l1552_155248

theorem triangle_inequality_with_heights 
  (a b c h_a h_b h_c t : ℝ) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) 
  (heights_def : h_a * a = h_b * b ∧ h_b * b = h_c * c) 
  (t_bound : t ≥ (1 : ℝ) / 2) : 
  (t * a + h_a) + (t * b + h_b) > t * c + h_c ∧ 
  (t * b + h_b) + (t * c + h_c) > t * a + h_a ∧ 
  (t * c + h_c) + (t * a + h_a) > t * b + h_b :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_heights_l1552_155248


namespace NUMINAMATH_CALUDE_transport_cost_is_162_50_l1552_155245

/-- Calculates the transport cost for a refrigerator purchase given the following conditions:
  * purchase_price: The price Ramesh paid after discount
  * discount_rate: The discount rate on the labelled price
  * installation_cost: The cost of installation
  * profit_rate: The desired profit rate if no discount was offered
  * selling_price: The price to sell at to achieve the desired profit rate
-/
def calculate_transport_cost (purchase_price : ℚ) (discount_rate : ℚ) 
  (installation_cost : ℚ) (profit_rate : ℚ) (selling_price : ℚ) : ℚ :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let profit := labelled_price * profit_rate
  let calculated_selling_price := labelled_price + profit
  selling_price - calculated_selling_price - installation_cost

/-- Theorem stating that given the specific conditions of Ramesh's refrigerator purchase,
    the transport cost is 162.50 rupees. -/
theorem transport_cost_is_162_50 :
  calculate_transport_cost 12500 0.20 250 0.10 17600 = 162.50 := by
  sorry

end NUMINAMATH_CALUDE_transport_cost_is_162_50_l1552_155245


namespace NUMINAMATH_CALUDE_shark_observation_l1552_155202

theorem shark_observation (p_truth : ℝ) (p_shark : ℝ) (n : ℕ) :
  p_truth = 1/6 →
  p_shark = 0.027777777777777773 →
  p_shark = p_truth * (1 / n) →
  n = 6 := by sorry

end NUMINAMATH_CALUDE_shark_observation_l1552_155202


namespace NUMINAMATH_CALUDE_smallest_product_of_factors_l1552_155265

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_product_of_factors (x y : ℕ) : 
  x ≠ y →
  is_factor x 48 →
  is_factor y 48 →
  (Even x ∨ Even y) →
  ¬(is_factor (x * y) 48) →
  ∀ (a b : ℕ), a ≠ b ∧ is_factor a 48 ∧ is_factor b 48 ∧ (Even a ∨ Even b) ∧ ¬(is_factor (a * b) 48) →
  x * y ≤ a * b →
  x * y = 32 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_of_factors_l1552_155265


namespace NUMINAMATH_CALUDE_project_choices_l1552_155293

/-- The number of projects available to choose from -/
def num_projects : ℕ := 5

/-- The number of students choosing projects -/
def num_students : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of permutations of k items from n items -/
def permute (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

/-- The main theorem stating the number of ways students can choose projects -/
theorem project_choices : 
  (choose num_students 2) * (permute num_projects 3) + (permute num_projects num_students) = 480 :=
sorry

end NUMINAMATH_CALUDE_project_choices_l1552_155293


namespace NUMINAMATH_CALUDE_complex_on_y_axis_l1552_155298

theorem complex_on_y_axis (a : ℝ) : 
  let z : ℂ := (a - 3 * Complex.I) / (1 - Complex.I)
  (Complex.re z = 0) → a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_on_y_axis_l1552_155298


namespace NUMINAMATH_CALUDE_count_rectangles_l1552_155244

/-- The number of checkered rectangles containing exactly one gray cell -/
def num_rectangles (total_gray_cells : ℕ) (blue_cells : ℕ) (red_cells : ℕ) 
  (rectangles_per_blue : ℕ) (rectangles_per_red : ℕ) : ℕ :=
  blue_cells * rectangles_per_blue + red_cells * rectangles_per_red

/-- Theorem stating the number of checkered rectangles containing exactly one gray cell -/
theorem count_rectangles : 
  num_rectangles 40 36 4 4 8 = 176 := by
  sorry

end NUMINAMATH_CALUDE_count_rectangles_l1552_155244


namespace NUMINAMATH_CALUDE_angle_AOB_measure_l1552_155224

/-- A configuration of rectangles with specific properties -/
structure RectangleConfiguration where
  /-- The number of equal rectangles -/
  num_rectangles : ℕ
  /-- Assertion that one side of each rectangle is twice the other -/
  side_ratio : Prop
  /-- Assertion that points C, O, and B are collinear -/
  collinear_COB : Prop
  /-- Assertion that triangle ACO is right-angled and isosceles -/
  triangle_ACO_properties : Prop

/-- Theorem stating that given the specific configuration, angle AOB measures 135° -/
theorem angle_AOB_measure (config : RectangleConfiguration) 
  (h1 : config.num_rectangles = 5)
  (h2 : config.side_ratio)
  (h3 : config.collinear_COB)
  (h4 : config.triangle_ACO_properties) :
  ∃ (angle_AOB : ℝ), angle_AOB = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_AOB_measure_l1552_155224


namespace NUMINAMATH_CALUDE_unique_three_digit_square_l1552_155251

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a function to get the first two digits of a three-digit number
def first_two_digits (n : ℕ) : ℕ :=
  n / 10

-- Define a function to get the last digit of a three-digit number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the main theorem
theorem unique_three_digit_square : ∃! n : ℕ, 
  100 ≤ n ∧ n ≤ 999 ∧ 
  is_perfect_square n ∧ 
  is_perfect_square (first_two_digits n / last_digit n) ∧
  n = 361 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_square_l1552_155251


namespace NUMINAMATH_CALUDE_range_of_g_l1552_155246

def g (x : ℝ) : ℝ := -x^2 + 3*x - 3

theorem range_of_g :
  ∀ y ∈ Set.range (fun (x : ℝ) => g x), -31 ≤ y ∧ y ≤ -3/4 ∧
  ∃ x₁ x₂ : ℝ, -4 ≤ x₁ ∧ x₁ ≤ 4 ∧ -4 ≤ x₂ ∧ x₂ ≤ 4 ∧ g x₁ = -31 ∧ g x₂ = -3/4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l1552_155246


namespace NUMINAMATH_CALUDE_limit_sequence_is_zero_l1552_155234

/-- The limit of the sequence (n - (n^5 - 5)^(1/3)) * n * sqrt(n) as n approaches infinity is 0. -/
theorem limit_sequence_is_zero :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((n : ℝ) - ((n : ℝ)^5 - 5)^(1/3)) * n * (n : ℝ).sqrt| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_sequence_is_zero_l1552_155234


namespace NUMINAMATH_CALUDE_minimum_average_for_remaining_semesters_l1552_155243

def required_average : ℝ := 85
def num_semesters : ℕ := 5
def first_three_scores : List ℝ := [84, 88, 80]

theorem minimum_average_for_remaining_semesters :
  let total_required := required_average * num_semesters
  let current_total := first_three_scores.sum
  let remaining_semesters := num_semesters - first_three_scores.length
  let remaining_required := total_required - current_total
  (remaining_required / remaining_semesters : ℝ) = 86.5 := by
sorry

end NUMINAMATH_CALUDE_minimum_average_for_remaining_semesters_l1552_155243


namespace NUMINAMATH_CALUDE_a_range_l1552_155220

theorem a_range (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x : ℝ, |Real.sin x| > a)
  (h3 : ∀ x : ℝ, x ∈ [π/4, 3*π/4] → (Real.sin x)^2 + a * Real.sin x - 1 ≥ 0) :
  a ∈ Set.Ici (Real.sqrt 2 / 2) ∩ Set.Iio 1 :=
by sorry

end NUMINAMATH_CALUDE_a_range_l1552_155220


namespace NUMINAMATH_CALUDE_jimmy_lodging_cost_l1552_155253

def hostel_nights : ℕ := 3
def hostel_cost_per_night : ℕ := 15
def cabin_nights : ℕ := 2
def cabin_total_cost_per_night : ℕ := 45
def cabin_friends : ℕ := 3

theorem jimmy_lodging_cost :
  hostel_nights * hostel_cost_per_night +
  cabin_nights * (cabin_total_cost_per_night / cabin_friends) = 75 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_lodging_cost_l1552_155253


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1552_155215

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) / (z - Complex.I) = Complex.I → z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1552_155215


namespace NUMINAMATH_CALUDE_fraction_expressions_l1552_155204

theorem fraction_expressions (x z : ℚ) (h : x / z = 5 / 6) :
  ((x + 3 * z) / z = 23 / 6) ∧
  (z / (x - z) = -6) ∧
  ((2 * x + z) / z = 8 / 3) ∧
  (3 * x / (4 * z) = 5 / 8) ∧
  ((x - 2 * z) / z = -7 / 6) := by
  sorry

end NUMINAMATH_CALUDE_fraction_expressions_l1552_155204


namespace NUMINAMATH_CALUDE_nuts_problem_l1552_155225

/-- The number of nuts after one day's operation -/
def nuts_after_day (n : ℕ) : ℕ := 
  if 2 * n > 8 then 2 * n - 8 else 0

/-- The number of nuts after d days, starting with n nuts -/
def nuts_after_days (n : ℕ) (d : ℕ) : ℕ :=
  match d with
  | 0 => n
  | d + 1 => nuts_after_day (nuts_after_days n d)

theorem nuts_problem :
  nuts_after_days 7 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nuts_problem_l1552_155225


namespace NUMINAMATH_CALUDE_equation_represents_parabola_l1552_155211

/-- The equation y^4 - 6x^2 = 3y^2 - 2 represents a parabola -/
theorem equation_represents_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, y^4 - 6*x^2 = 3*y^2 - 2 ↔ a*y^2 + b*x + c = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_parabola_l1552_155211


namespace NUMINAMATH_CALUDE_expected_coffee_tea_difference_l1552_155292

/-- Represents the outcome of rolling a fair eight-sided die -/
inductive DieRoll
  | one | two | three | four | five | six | seven | eight

/-- Represents the drink Alice chooses based on her die roll -/
inductive Drink
  | coffee | tea | juice

/-- Function that determines the drink based on the die roll -/
def chooseDrink (roll : DieRoll) : Drink :=
  match roll with
  | DieRoll.one => Drink.juice
  | DieRoll.two => Drink.coffee
  | DieRoll.three => Drink.tea
  | DieRoll.four => Drink.coffee
  | DieRoll.five => Drink.tea
  | DieRoll.six => Drink.coffee
  | DieRoll.seven => Drink.tea
  | DieRoll.eight => Drink.coffee

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- Theorem stating the expected difference between coffee and tea days -/
theorem expected_coffee_tea_difference :
  let p_coffee : ℚ := 1/2
  let p_tea : ℚ := 3/8
  let expected_coffee_days : ℚ := p_coffee * daysInYear
  let expected_tea_days : ℚ := p_tea * daysInYear
  let difference : ℚ := expected_coffee_days - expected_tea_days
  ⌊difference⌋ = 45 := by sorry


end NUMINAMATH_CALUDE_expected_coffee_tea_difference_l1552_155292


namespace NUMINAMATH_CALUDE_eulers_formula_l1552_155277

/-- A convex polyhedron with vertices, edges, and faces. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for convex polyhedra. -/
theorem eulers_formula (p : ConvexPolyhedron) : p.vertices - p.edges + p.faces = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l1552_155277


namespace NUMINAMATH_CALUDE_sets_properties_l1552_155203

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Theorem statement
theorem sets_properties :
  (A ∩ B = {x | 2 < x ∧ x ≤ 3}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ 3 ∨ x ≥ 4}) ∧
  (∀ a : ℝ, B ∩ C a = C a → 2 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_sets_properties_l1552_155203


namespace NUMINAMATH_CALUDE_uniform_cost_ratio_l1552_155259

theorem uniform_cost_ratio : 
  ∀ (shirt_cost pants_cost tie_cost sock_cost : ℝ),
    pants_cost = 20 →
    tie_cost = shirt_cost / 5 →
    sock_cost = 3 →
    5 * (pants_cost + shirt_cost + tie_cost + sock_cost) = 355 →
    shirt_cost / pants_cost = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_uniform_cost_ratio_l1552_155259


namespace NUMINAMATH_CALUDE_percentage_calculation_l1552_155242

-- Define constants
def rupees_to_paise : ℝ → ℝ := (· * 100)

-- Theorem statement
theorem percentage_calculation (x : ℝ) : 
  (x / 100) * rupees_to_paise 160 = 80 → x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1552_155242


namespace NUMINAMATH_CALUDE_coin_problem_l1552_155282

theorem coin_problem (x y z : ℕ) : 
  x + y + z = 30 →
  10 * x + 15 * y + 20 * z = 500 →
  z > x :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l1552_155282


namespace NUMINAMATH_CALUDE_function_property_l1552_155235

/-- A function satisfying the given condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f y)^2)

/-- The main theorem -/
theorem function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1552_155235


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1552_155285

theorem sum_of_fractions : (1 : ℚ) / 3 + (5 : ℚ) / 12 = (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1552_155285


namespace NUMINAMATH_CALUDE_vector_equality_l1552_155226

/-- Given two vectors in ℝ², prove that if their sum and difference have equal magnitudes, 
    then the second component of the second vector must be 3/2. -/
theorem vector_equality (a b : ℝ × ℝ) (h : ‖a + b‖ = ‖a - b‖) 
    (ha : a = (1, 2)) (hb : b.1 = -3) : b.2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l1552_155226


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l1552_155214

def sequence_sum (n : ℕ) : ℕ → ℕ
| 0 => 5
| m + 1 => 2 * sequence_sum n m + (m + 1) + 5

theorem sequence_sum_formula (n : ℕ) :
  sequence_sum n n = 6 * 2^n - (n + 6) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l1552_155214


namespace NUMINAMATH_CALUDE_haley_initial_lives_l1552_155217

theorem haley_initial_lives : 
  ∀ (initial_lives : ℕ), 
    (initial_lives - 4 + 36 = 46) → 
    initial_lives = 14 := by
  sorry

end NUMINAMATH_CALUDE_haley_initial_lives_l1552_155217


namespace NUMINAMATH_CALUDE_sum_of_20th_and_30th_triangular_l1552_155261

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of the 20th and 30th triangular numbers is 675 -/
theorem sum_of_20th_and_30th_triangular : triangular_number 20 + triangular_number 30 = 675 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_20th_and_30th_triangular_l1552_155261


namespace NUMINAMATH_CALUDE_project_over_budget_proof_l1552_155227

/-- Calculates the amount a project is over budget given the total budget, 
    number of months, months passed, and actual expenditure. -/
def project_over_budget (total_budget : ℚ) (num_months : ℕ) 
                        (months_passed : ℕ) (actual_expenditure : ℚ) : ℚ :=
  actual_expenditure - (total_budget / num_months) * months_passed

/-- Proves that given the specific conditions of the problem, 
    the project is over budget by $280. -/
theorem project_over_budget_proof : 
  project_over_budget 12600 12 6 6580 = 280 := by
  sorry

#eval project_over_budget 12600 12 6 6580

end NUMINAMATH_CALUDE_project_over_budget_proof_l1552_155227


namespace NUMINAMATH_CALUDE_focus_of_given_parabola_l1552_155213

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  /-- The equation of the parabola in the form y = a(x - h)^2 + k -/
  equation : ℝ → ℝ
  /-- The coefficient 'a' determines the direction and width of the parabola -/
  a : ℝ
  /-- The horizontal shift of the vertex -/
  h : ℝ
  /-- The vertical shift of the vertex -/
  k : ℝ

/-- The focus of a parabola is a point from which all points on the parabola are equidistant to the directrix. -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Given parabola y = (x-3)^2 + 2 -/
def given_parabola : Parabola where
  equation := fun x ↦ (x - 3)^2 + 2
  a := 1
  h := 3
  k := 2

/-- Theorem: The focus of the parabola y = (x-3)^2 + 2 is at the point (3, 9/4) -/
theorem focus_of_given_parabola :
  focus given_parabola = (3, 9/4) := by sorry

end NUMINAMATH_CALUDE_focus_of_given_parabola_l1552_155213


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l1552_155231

-- Define the triangle ABC and point X
structure Triangle :=
  (A B C X : ℝ × ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  let d := λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt
  d t.A t.B = 75 ∧
  d t.A t.C = 45 ∧
  d t.B t.C = 90 ∧
  -- X is on the angle bisector of angle ACB
  (t.X.1 - t.A.1) / (t.C.1 - t.A.1) = (t.X.2 - t.A.2) / (t.C.2 - t.A.2) ∧
  (t.X.1 - t.B.1) / (t.C.1 - t.B.1) = (t.X.2 - t.B.2) / (t.C.2 - t.B.2)

-- Theorem statement
theorem angle_bisector_theorem (t : Triangle) (h : is_valid_triangle t) :
  let d := λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt
  d t.A t.X = 25 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_l1552_155231


namespace NUMINAMATH_CALUDE_student_selection_permutation_l1552_155249

theorem student_selection_permutation :
  (Nat.factorial 6) / (Nat.factorial 4) = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_permutation_l1552_155249


namespace NUMINAMATH_CALUDE_tickets_left_l1552_155201

/-- Given that Paul bought eleven tickets and spent three tickets,
    prove that he has eight tickets left. -/
theorem tickets_left (total : ℕ) (spent : ℕ) (left : ℕ) 
    (h1 : total = 11)
    (h2 : spent = 3)
    (h3 : left = total - spent) : left = 8 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_l1552_155201


namespace NUMINAMATH_CALUDE_wall_height_calculation_l1552_155262

/-- Given a brick and wall with specified dimensions, prove the height of the wall --/
theorem wall_height_calculation (brick_length brick_width brick_height : Real)
  (wall_length wall_width : Real) (num_bricks : Nat) :
  brick_length = 0.20 →
  brick_width = 0.10 →
  brick_height = 0.075 →
  wall_length = 25 →
  wall_width = 0.75 →
  num_bricks = 25000 →
  ∃ (wall_height : Real),
    wall_height = 2 ∧
    num_bricks * (brick_length * brick_width * brick_height) = wall_length * wall_width * wall_height :=
by sorry

end NUMINAMATH_CALUDE_wall_height_calculation_l1552_155262


namespace NUMINAMATH_CALUDE_equation_solution_l1552_155232

theorem equation_solution : ∃ x : ℝ, (144 / 0.144 = x / 0.0144) ∧ (x = 14.4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1552_155232


namespace NUMINAMATH_CALUDE_correct_multiplier_problem_solution_l1552_155275

theorem correct_multiplier (number_to_multiply : ℕ) (mistaken_multiplier : ℕ) (difference : ℕ) : ℕ :=
  let correct_multiplier := (mistaken_multiplier * number_to_multiply + difference) / number_to_multiply
  correct_multiplier

theorem problem_solution :
  correct_multiplier 135 34 1215 = 43 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplier_problem_solution_l1552_155275


namespace NUMINAMATH_CALUDE_system_solution_l1552_155271

theorem system_solution (x y : ℝ) : 
  x > 0 → y > 0 → 
  Real.log x / Real.log 4 + Real.log y / Real.log 4 = 1 + Real.log 9 / Real.log 4 →
  x + y = 20 →
  ((x = 2 ∧ y = 18) ∨ (x = 18 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1552_155271


namespace NUMINAMATH_CALUDE_strawberry_remainder_l1552_155272

/-- Given 3 kg and 300 g of strawberries, prove that after giving away 1 kg and 900 g, 
    the remaining amount is 1400 g. -/
theorem strawberry_remainder : 
  let total_kg : ℕ := 3
  let total_g : ℕ := 300
  let given_kg : ℕ := 1
  let given_g : ℕ := 900
  let g_per_kg : ℕ := 1000
  (total_kg * g_per_kg + total_g) - (given_kg * g_per_kg + given_g) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_remainder_l1552_155272


namespace NUMINAMATH_CALUDE_lawn_width_is_60_l1552_155274

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  roadWidth : ℝ
  costPerSqm : ℝ
  totalCost : ℝ

/-- Calculates the total area of the roads -/
def roadArea (l : LawnWithRoads) : ℝ :=
  l.roadWidth * l.length + l.roadWidth * l.width - l.roadWidth * l.roadWidth

/-- Theorem: Given the specifications, the width of the lawn is 60 meters -/
theorem lawn_width_is_60 (l : LawnWithRoads) 
    (h1 : l.length = 90)
    (h2 : l.roadWidth = 10)
    (h3 : l.costPerSqm = 3)
    (h4 : l.totalCost = 4200)
    (h5 : l.totalCost = l.costPerSqm * roadArea l) : 
  l.width = 60 := by
  sorry

end NUMINAMATH_CALUDE_lawn_width_is_60_l1552_155274


namespace NUMINAMATH_CALUDE_problem_statement_l1552_155269

theorem problem_statement (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1552_155269


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l1552_155278

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- State the theorem
theorem cubic_function_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ (y : ℝ), f a y ≤ f a x₁) ∧ 
    (∀ (y : ℝ), f a y ≥ f a x₂)) →
  (a < -3 ∨ a > 6) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l1552_155278


namespace NUMINAMATH_CALUDE_cube_split_theorem_l1552_155236

def is_split_number (m : ℕ) (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2 * k + 1 ∧ 
  ∃ (start : ℕ), (Finset.range m).sum (λ i => 2 * (start + i) + 1) = m^3 ∧
  ∃ (i : Fin m), n = 2 * (start + i) + 1

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) (h2 : is_split_number m 333) : m = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_theorem_l1552_155236


namespace NUMINAMATH_CALUDE_book_reading_percentage_l1552_155291

theorem book_reading_percentage (total_pages : ℕ) (second_night_percent : ℝ) 
  (third_night_percent : ℝ) (pages_left : ℕ) : ℝ :=
by
  have h1 : total_pages = 500 := by sorry
  have h2 : second_night_percent = 20 := by sorry
  have h3 : third_night_percent = 30 := by sorry
  have h4 : pages_left = 150 := by sorry
  
  -- Define the first night percentage
  let first_night_percent : ℝ := 20

  -- Prove that the first night percentage is correct
  have h5 : first_night_percent / 100 * total_pages + 
            second_night_percent / 100 * total_pages + 
            third_night_percent / 100 * total_pages = 
            total_pages - pages_left := by sorry

  exact first_night_percent

end NUMINAMATH_CALUDE_book_reading_percentage_l1552_155291


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1552_155255

theorem complex_fraction_equality : (Complex.I + 3) / (Complex.I + 1) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1552_155255


namespace NUMINAMATH_CALUDE_percentage_problem_l1552_155258

/-- Given a number where 10% of it is 40 and a certain percentage of it is 160, prove that percentage is 40% -/
theorem percentage_problem (n : ℝ) (p : ℝ) 
  (h1 : n * 0.1 = 40) 
  (h2 : n * (p / 100) = 160) : 
  p = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1552_155258


namespace NUMINAMATH_CALUDE_carbon_atoms_in_compound_l1552_155207

/-- Represents the number of atoms of each element in a compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight hydrogenWeight oxygenWeight : ℕ) : ℕ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

theorem carbon_atoms_in_compound :
  ∀ (c : Compound),
    c.hydrogen = 6 →
    c.oxygen = 1 →
    molecularWeight c 12 1 16 = 58 →
    c.carbon = 3 := by
  sorry

end NUMINAMATH_CALUDE_carbon_atoms_in_compound_l1552_155207


namespace NUMINAMATH_CALUDE_two_red_one_blue_probability_l1552_155200

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8

theorem two_red_one_blue_probability :
  let prob := (red_marbles * (red_marbles - 1) * blue_marbles + 
               red_marbles * blue_marbles * (red_marbles - 1) + 
               blue_marbles * red_marbles * (red_marbles - 1)) / 
              (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  prob = 44 / 95 := by
sorry

end NUMINAMATH_CALUDE_two_red_one_blue_probability_l1552_155200


namespace NUMINAMATH_CALUDE_total_unbroken_seashells_is_17_l1552_155266

/-- The number of unbroken seashells Tom found over three days -/
def total_unbroken_seashells : ℕ :=
  let day1_total := 7
  let day1_broken := 4
  let day2_total := 12
  let day2_broken := 5
  let day3_total := 15
  let day3_broken := 8
  (day1_total - day1_broken) + (day2_total - day2_broken) + (day3_total - day3_broken)

/-- Theorem stating that the total number of unbroken seashells is 17 -/
theorem total_unbroken_seashells_is_17 : total_unbroken_seashells = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_unbroken_seashells_is_17_l1552_155266


namespace NUMINAMATH_CALUDE_tan_difference_alpha_pi_8_l1552_155238

theorem tan_difference_alpha_pi_8 (α : ℝ) (h : 2 * Real.tan α = 3 * Real.tan (π / 8)) :
  Real.tan (α - π / 8) = (1 + 5 * Real.sqrt 2) / 49 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_alpha_pi_8_l1552_155238


namespace NUMINAMATH_CALUDE_initial_machines_count_l1552_155284

/-- The number of shirts produced by a group of machines -/
def shirts_produced (num_machines : ℕ) (time : ℕ) : ℕ := sorry

/-- The production rate of a single machine in shirts per minute -/
def machine_rate : ℚ := sorry

/-- The total production rate of all machines in shirts per minute -/
def total_rate : ℕ := 32

theorem initial_machines_count :
  ∃ (n : ℕ), 
    shirts_produced 8 10 = 160 ∧
    (n : ℚ) * machine_rate = total_rate ∧
    n = 16 :=
sorry

end NUMINAMATH_CALUDE_initial_machines_count_l1552_155284


namespace NUMINAMATH_CALUDE_unique_rational_root_l1552_155210

/-- The polynomial function we're examining -/
def f (x : ℚ) : ℚ := 3 * x^4 - 4 * x^3 - 10 * x^2 + 6 * x + 3

/-- A rational number is a root of f if f(x) = 0 -/
def is_root (x : ℚ) : Prop := f x = 0

/-- The statement that 1/3 is the only rational root of f -/
theorem unique_rational_root : 
  (is_root (1/3)) ∧ (∀ x : ℚ, is_root x → x = 1/3) := by sorry

end NUMINAMATH_CALUDE_unique_rational_root_l1552_155210


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l1552_155212

theorem largest_integer_inequality : 
  ∀ x : ℤ, x ≤ 10 ↔ (x : ℚ) / 4 + 5 / 6 < 7 / 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l1552_155212


namespace NUMINAMATH_CALUDE_quilt_square_transformation_l1552_155268

theorem quilt_square_transformation (length width : ℝ) (h1 : length = 6) (h2 : width = 24) :
  ∃ (side : ℝ), side^2 = length * width ∧ side = 12 := by
sorry

end NUMINAMATH_CALUDE_quilt_square_transformation_l1552_155268


namespace NUMINAMATH_CALUDE_fractional_equation_root_l1552_155252

theorem fractional_equation_root (m : ℝ) : 
  (∃ x : ℝ, x ≠ 2 ∧ x ≠ 2 ∧ (3 / (x - 2) + 1 = m / (4 - 2*x))) → m = -6 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l1552_155252


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l1552_155283

theorem consecutive_even_numbers_sum (x : ℤ) : 
  (x % 2 = 0) →                 -- x is even
  ((x + 2)^2 - x^2 = 84) →      -- difference of squares is 84
  (x + (x + 2) = 42) :=         -- sum is 42
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l1552_155283


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1552_155276

def M (a : ℕ) : Set ℕ := {3, 2^a}
def N (a b : ℕ) : Set ℕ := {a, b}

theorem union_of_M_and_N (a b : ℕ) :
  M a ∩ N a b = {2} →
  M a ∪ N a b = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1552_155276


namespace NUMINAMATH_CALUDE_min_mushrooms_collected_l1552_155221

/-- Represents the number of mushrooms collected by Vasya and Masha over two days -/
structure MushroomCollection where
  vasya_day1 : ℕ
  vasya_day2 : ℕ

/-- Calculates the total number of mushrooms collected by both Vasya and Masha -/
def total_mushrooms (c : MushroomCollection) : ℚ :=
  (c.vasya_day1 + c.vasya_day2 : ℚ) + 
  ((3/4 : ℚ) * c.vasya_day1 + (6/5 : ℚ) * c.vasya_day2)

/-- Checks if the collection satisfies the given conditions -/
def is_valid_collection (c : MushroomCollection) : Prop :=
  (3/4 : ℚ) * c.vasya_day1 + (6/5 : ℚ) * c.vasya_day2 = 
  (11/10 : ℚ) * (c.vasya_day1 + c.vasya_day2)

/-- The main theorem stating the minimum number of mushrooms collected -/
theorem min_mushrooms_collected :
  ∃ (c : MushroomCollection), 
    is_valid_collection c ∧ 
    (∀ (c' : MushroomCollection), is_valid_collection c' → 
      total_mushrooms c ≤ total_mushrooms c') ∧
    ⌈total_mushrooms c⌉ = 19 := by
  sorry


end NUMINAMATH_CALUDE_min_mushrooms_collected_l1552_155221


namespace NUMINAMATH_CALUDE_inequality_theorem_l1552_155264

theorem inequality_theorem (m n : ℕ) (hm : m > 0) (hn : n > 0) (hmn : m > n) (hn2 : n ≥ 2) :
  ((1 + 1 / m : ℝ) ^ m > (1 + 1 / n : ℝ) ^ n) ∧
  ((1 + 1 / m : ℝ) ^ (m + 1) < (1 + 1 / n : ℝ) ^ (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1552_155264


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l1552_155230

theorem positive_integer_solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 2 * p.2 = 1001) (Finset.product (Finset.range 1002) (Finset.range 1002))).card = 167 :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l1552_155230


namespace NUMINAMATH_CALUDE_equation_solution_l1552_155206

theorem equation_solution (a b c : ℤ) :
  (∀ x : ℝ, (x - a) * (x - 10) + 5 = (x + b) * (x + c)) ↔ (a = 4 ∨ a = 16) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l1552_155206


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1552_155205

theorem tangent_line_equation (x y : ℝ) : 
  (∃ x₀ y₀ : ℝ, 
    y₀ = Real.log x₀ ∧                           -- Point on the curve
    (1 : ℝ) * (1 : ℝ) = -1 ∧                     -- Perpendicularity condition
    (y - y₀) = (1 : ℝ) * (x - x₀))               -- Point-slope form of tangent line
  → x - y - 1 = 0 :=                             -- Equation of the tangent line
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1552_155205


namespace NUMINAMATH_CALUDE_number_with_quotient_and_remainder_l1552_155267

theorem number_with_quotient_and_remainder (x : ℕ) : 
  (x / 7 = 4) ∧ (x % 7 = 6) → x = 34 := by
  sorry

end NUMINAMATH_CALUDE_number_with_quotient_and_remainder_l1552_155267


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1552_155295

theorem absolute_value_inequality (x y : ℝ) 
  (h1 : |x - y| < 1) 
  (h2 : |2*x + y| < 1) : 
  |y| < 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1552_155295


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l1552_155280

theorem arctan_equation_solution (x : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/10) + Real.arctan (1/x) = π/4 → x = 37/3 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l1552_155280


namespace NUMINAMATH_CALUDE_logarithm_inequality_l1552_155288

theorem logarithm_inequality (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : 0 < c) 
  (h4 : c < 1) : 
  a * Real.log c / Real.log b < b * Real.log c / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l1552_155288


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1552_155281

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 2*k*x^2 + (8*k+1)*x + 8*k = 0 ∧ 2*k*y^2 + (8*k+1)*y + 8*k = 0) 
  ↔ 
  (k ≥ -1/16 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1552_155281


namespace NUMINAMATH_CALUDE_cosine_value_proof_l1552_155241

theorem cosine_value_proof (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 1/3)
  (h2 : Real.pi/2 ≤ α)
  (h3 : α ≤ Real.pi) :
  Real.cos α = -2 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cosine_value_proof_l1552_155241


namespace NUMINAMATH_CALUDE_dogwood_trees_planted_l1552_155290

/-- The number of dogwood trees planted in a park --/
theorem dogwood_trees_planted (initial_trees final_trees : ℕ) 
  (h1 : initial_trees = 34)
  (h2 : final_trees = 83) :
  final_trees - initial_trees = 49 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_l1552_155290


namespace NUMINAMATH_CALUDE_multiply_57_47_l1552_155257

theorem multiply_57_47 : 57 * 47 = 2820 := by
  sorry

end NUMINAMATH_CALUDE_multiply_57_47_l1552_155257


namespace NUMINAMATH_CALUDE_factor_expression_l1552_155239

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1552_155239


namespace NUMINAMATH_CALUDE_samantha_more_heads_prob_l1552_155270

def fair_coin_prob : ℚ := 1/2
def biased_coin_prob1 : ℚ := 3/5
def biased_coin_prob2 : ℚ := 2/3

def coin_set := (fair_coin_prob, biased_coin_prob1, biased_coin_prob2)

def prob_more_heads (coins : ℚ × ℚ × ℚ) : ℚ :=
  sorry

theorem samantha_more_heads_prob :
  prob_more_heads coin_set = 436/225 :=
sorry

end NUMINAMATH_CALUDE_samantha_more_heads_prob_l1552_155270
