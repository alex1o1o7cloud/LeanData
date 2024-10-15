import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_count_l158_15852

theorem rectangle_count (h v : ℕ) (h_eq : h = 5) (v_eq : v = 5) :
  (Nat.choose h 2) * (Nat.choose v 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_count_l158_15852


namespace NUMINAMATH_CALUDE_max_xy_perpendicular_vectors_l158_15841

theorem max_xy_perpendicular_vectors (x y : ℝ) :
  let a : ℝ × ℝ := (1, x - 1)
  let b : ℝ × ℝ := (y, 2)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  ∃ (m : ℝ), (∀ (x' y' : ℝ), 
    let a' : ℝ × ℝ := (1, x' - 1)
    let b' : ℝ × ℝ := (y', 2)
    (a'.1 * b'.1 + a'.2 * b'.2 = 0) → x' * y' ≤ m) ∧
  m = (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_xy_perpendicular_vectors_l158_15841


namespace NUMINAMATH_CALUDE_a_neg_three_sufficient_not_necessary_l158_15864

/-- Two lines in the plane, parameterized by a real number a -/
def line1 (a : ℝ) := {(x, y) : ℝ × ℝ | x + a * y + 2 = 0}
def line2 (a : ℝ) := {(x, y) : ℝ × ℝ | a * x + (a + 2) * y + 1 = 0}

/-- The condition for two lines to be perpendicular -/
def are_perpendicular (a : ℝ) : Prop :=
  a * (a + 3) = 0

/-- The statement to be proved -/
theorem a_neg_three_sufficient_not_necessary :
  (∀ a : ℝ, a = -3 → are_perpendicular a) ∧
  ¬(∀ a : ℝ, are_perpendicular a → a = -3) :=
sorry

end NUMINAMATH_CALUDE_a_neg_three_sufficient_not_necessary_l158_15864


namespace NUMINAMATH_CALUDE_reach_probability_is_15_1024_l158_15893

/-- Represents a point in a 2D grid --/
structure Point where
  x : Int
  y : Int

/-- Represents a step direction --/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of a single step in any direction --/
def stepProbability : Rat := 1 / 4

/-- The starting point --/
def start : Point := ⟨0, 0⟩

/-- The target point --/
def target : Point := ⟨3, 1⟩

/-- The maximum number of steps allowed --/
def maxSteps : Nat := 8

/-- Calculates the probability of reaching the target from the start in at most maxSteps --/
def reachProbability (start : Point) (target : Point) (maxSteps : Nat) : Rat :=
  sorry

/-- The main theorem to prove --/
theorem reach_probability_is_15_1024 : 
  reachProbability start target maxSteps = 15 / 1024 := by sorry

end NUMINAMATH_CALUDE_reach_probability_is_15_1024_l158_15893


namespace NUMINAMATH_CALUDE_jasons_quarters_l158_15813

/-- Given that Jason had 49 quarters initially and his dad gave him 25 quarters,
    prove that Jason now has 74 quarters. -/
theorem jasons_quarters (initial : ℕ) (given : ℕ) (total : ℕ) 
    (h1 : initial = 49) 
    (h2 : given = 25) 
    (h3 : total = initial + given) : 
  total = 74 := by
  sorry

end NUMINAMATH_CALUDE_jasons_quarters_l158_15813


namespace NUMINAMATH_CALUDE_age_difference_l158_15867

/-- Given that the total age of a and b is 17 years more than the total age of b and c,
    prove that c is 17 years younger than a. -/
theorem age_difference (a b c : ℕ) (h : a + b = b + c + 17) : a = c + 17 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l158_15867


namespace NUMINAMATH_CALUDE_range_of_a_l158_15889

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ∈ Set.union (Set.Iic (-2)) {1} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l158_15889


namespace NUMINAMATH_CALUDE_geometric_progression_properties_l158_15820

/-- Represents a geometric progression with given properties -/
structure GeometricProgression where
  ratio : ℚ
  fourthTerm : ℚ
  sum : ℚ

/-- The number of terms in the geometric progression -/
def numTerms (gp : GeometricProgression) : ℕ := sorry

/-- Theorem stating the properties of the specific geometric progression -/
theorem geometric_progression_properties :
  ∃ (gp : GeometricProgression),
    gp.ratio = 1/3 ∧
    gp.fourthTerm = 1/54 ∧
    gp.sum = 121/162 ∧
    numTerms gp = 5 := by sorry

end NUMINAMATH_CALUDE_geometric_progression_properties_l158_15820


namespace NUMINAMATH_CALUDE_gold_percentage_in_first_metal_l158_15896

theorem gold_percentage_in_first_metal
  (total_weight : Real)
  (desired_gold_percentage : Real)
  (first_metal_weight : Real)
  (second_metal_weight : Real)
  (second_metal_gold_percentage : Real)
  (h1 : total_weight = 12.4)
  (h2 : desired_gold_percentage = 0.5)
  (h3 : first_metal_weight = 6.2)
  (h4 : second_metal_weight = 6.2)
  (h5 : second_metal_gold_percentage = 0.4)
  (h6 : total_weight = first_metal_weight + second_metal_weight) :
  let total_gold := total_weight * desired_gold_percentage
  let second_metal_gold := second_metal_weight * second_metal_gold_percentage
  let first_metal_gold := total_gold - second_metal_gold
  let first_metal_gold_percentage := first_metal_gold / first_metal_weight
  first_metal_gold_percentage = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_gold_percentage_in_first_metal_l158_15896


namespace NUMINAMATH_CALUDE_dads_borrowed_nickels_l158_15877

/-- The number of nickels Mike's dad borrowed -/
def nickels_borrowed (initial_nickels remaining_nickels : ℕ) : ℕ :=
  initial_nickels - remaining_nickels

theorem dads_borrowed_nickels :
  let initial_nickels : ℕ := 87
  let remaining_nickels : ℕ := 12
  nickels_borrowed initial_nickels remaining_nickels = 75 := by
sorry

end NUMINAMATH_CALUDE_dads_borrowed_nickels_l158_15877


namespace NUMINAMATH_CALUDE_probability_one_of_each_color_is_9_28_l158_15876

def total_balls : ℕ := 9
def balls_per_color : ℕ := 3
def selected_balls : ℕ := 3

def probability_one_of_each_color : ℚ :=
  (balls_per_color ^ 3 : ℚ) / (total_balls.choose selected_balls)

theorem probability_one_of_each_color_is_9_28 :
  probability_one_of_each_color = 9 / 28 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_color_is_9_28_l158_15876


namespace NUMINAMATH_CALUDE_chelsea_needs_52_bullseyes_l158_15881

/-- Represents the archery competition scenario -/
structure ArcheryCompetition where
  total_shots : Nat
  chelsea_lead : Nat
  bullseye_score : Nat
  chelsea_min_score : Nat
  opponent_max_score : Nat

/-- Calculates the minimum number of consecutive bullseyes needed for Chelsea to win -/
def min_bullseyes_needed (comp : ArcheryCompetition) : Nat :=
  let remaining_shots := comp.total_shots / 2
  let chelsea_score := remaining_shots * comp.chelsea_min_score + comp.chelsea_lead
  let opponent_max := remaining_shots * comp.opponent_max_score
  let score_diff := opponent_max - chelsea_score
  (score_diff + comp.bullseye_score - comp.chelsea_min_score - 1) / (comp.bullseye_score - comp.chelsea_min_score) + 1

/-- The main theorem stating that 52 consecutive bullseyes are needed for Chelsea to win -/
theorem chelsea_needs_52_bullseyes :
  let comp := ArcheryCompetition.mk 120 60 10 3 10
  min_bullseyes_needed comp = 52 := by
  sorry

end NUMINAMATH_CALUDE_chelsea_needs_52_bullseyes_l158_15881


namespace NUMINAMATH_CALUDE_calculator_sale_loss_l158_15817

theorem calculator_sale_loss : 
  ∀ (x y : ℝ),
    x * (1 + 0.2) = 60 →
    y * (1 - 0.2) = 60 →
    60 + 60 - (x + y) = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_calculator_sale_loss_l158_15817


namespace NUMINAMATH_CALUDE_square_land_side_length_l158_15849

theorem square_land_side_length (area : ℝ) (h : area = Real.sqrt 900) :
  ∃ (side : ℝ), side * side = area ∧ side = 30 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l158_15849


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l158_15830

/-- Given a quadratic function f(x) = x^2 - ax - b with roots 2 and 3,
    prove that g(x) = bx^2 - ax - 1 has roots -1/2 and -1/3 -/
theorem quadratic_roots_relation (a b : ℝ) : 
  (∀ x, x^2 - a*x - b = 0 ↔ x = 2 ∨ x = 3) →
  (∀ x, b*x^2 - a*x - 1 = 0 ↔ x = -1/2 ∨ x = -1/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l158_15830


namespace NUMINAMATH_CALUDE_vanya_number_l158_15837

theorem vanya_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  let m := n / 10
  let d := n % 10
  (10 * d + m)^2 = 4 * n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_vanya_number_l158_15837


namespace NUMINAMATH_CALUDE_xyz_inequality_l158_15884

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x*y + y*z + z*x = 1) : 
  x*y*z*(x+y)*(y+z)*(z+x) ≥ (1-x^2)*(1-y^2)*(1-z^2) := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l158_15884


namespace NUMINAMATH_CALUDE_triangle_inequality_generalization_l158_15832

theorem triangle_inequality_generalization (x y z : ℝ) :
  (|x - y| + |y - z| + |z - x| ≤ 2 * Real.sqrt 2 * Real.sqrt (x^2 + y^2 + z^2)) ∧
  ((0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) → |x - y| + |y - z| + |z - x| ≤ 2 * Real.sqrt (x^2 + y^2 + z^2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_generalization_l158_15832


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l158_15843

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) :
  (∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
    (x^2 + a₁ * x + d₁) * (x^2 + a₂ * x + d₂) * (x^2 + a₃ * x + d₃)) →
  a₁ * d₁ + a₂ * d₂ + a₃ * d₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l158_15843


namespace NUMINAMATH_CALUDE_rice_box_theorem_l158_15898

/-- Represents the number of grains in each box -/
def grains_in_box (first_grain_count : ℕ) (common_difference : ℕ) (box_number : ℕ) : ℕ :=
  first_grain_count + (box_number - 1) * common_difference

/-- The total number of grains in all boxes -/
def total_grains (first_grain_count : ℕ) (common_difference : ℕ) (num_boxes : ℕ) : ℕ :=
  (num_boxes * (2 * first_grain_count + (num_boxes - 1) * common_difference)) / 2

theorem rice_box_theorem :
  (∃ (d : ℕ), total_grains 11 d 9 = 351 ∧ d = 7) ∧
  (∃ (d : ℕ), grains_in_box (23 - 2 * d) d 3 = 23 ∧ total_grains (23 - 2 * d) d 9 = 351 ∧ d = 8) :=
by sorry

end NUMINAMATH_CALUDE_rice_box_theorem_l158_15898


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l158_15885

theorem company_picnic_attendance 
  (total_employees : ℕ) 
  (men_attendance_rate : ℚ) 
  (women_attendance_rate : ℚ) 
  (men_percentage : ℚ) 
  (h1 : men_attendance_rate = 1/5) 
  (h2 : women_attendance_rate = 2/5) 
  (h3 : men_percentage = 7/20) :
  let women_percentage := 1 - men_percentage
  let men_attended := (men_attendance_rate * men_percentage * total_employees).floor
  let women_attended := (women_attendance_rate * women_percentage * total_employees).floor
  let total_attended := men_attended + women_attended
  (total_attended : ℚ) / total_employees = 33/100 :=
sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l158_15885


namespace NUMINAMATH_CALUDE_sample_customers_l158_15844

theorem sample_customers (samples_per_box : ℕ) (boxes_opened : ℕ) (leftover_samples : ℕ) :
  samples_per_box = 20 →
  boxes_opened = 12 →
  leftover_samples = 5 →
  (samples_per_box * boxes_opened - leftover_samples : ℕ) = 235 :=
by sorry

end NUMINAMATH_CALUDE_sample_customers_l158_15844


namespace NUMINAMATH_CALUDE_range_of_a_l158_15879

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then |Real.log x| else -(x - 3*a + 1)^2 + (2*a - 1)^2 + a

/-- The function g(x) defined as f(x) - b -/
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a x - b

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∃ b : ℝ, b > 0 ∧ (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    g a b x₁ = 0 ∧ g a b x₂ = 0 ∧ g a b x₃ = 0 ∧ g a b x₄ = 0)) →
  0 < a ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l158_15879


namespace NUMINAMATH_CALUDE_essay_pages_filled_l158_15845

theorem essay_pages_filled (johnny_words madeline_words timothy_words words_per_page : ℕ) 
  (h1 : johnny_words = 150)
  (h2 : madeline_words = 2 * johnny_words)
  (h3 : timothy_words = madeline_words + 30)
  (h4 : words_per_page = 260) : 
  (johnny_words + madeline_words + timothy_words) / words_per_page = 3 := by
  sorry

end NUMINAMATH_CALUDE_essay_pages_filled_l158_15845


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l158_15818

theorem greatest_integer_radius (A : ℝ) (h : A < 90 * Real.pi) : 
  ∀ r : ℕ, r * r * Real.pi ≤ A → r ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l158_15818


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l158_15850

theorem remainder_of_large_number (N : ℕ) (d : ℕ) (h : N = 9876543210123456789 ∧ d = 252) :
  N % d = 27 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l158_15850


namespace NUMINAMATH_CALUDE_vector_addition_path_l158_15880

-- Define a 2D vector
def Vector2D := ℝ × ℝ

-- Define vector addition
def vec_add (v w : Vector2D) : Vector2D :=
  (v.1 + w.1, v.2 + w.2)

-- Define vector from point to point
def vec_from_to (A B : Vector2D) : Vector2D :=
  (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem vector_addition_path (A B C D : Vector2D) :
  vec_add (vec_add (vec_from_to A B) (vec_from_to B C)) (vec_from_to C D) =
  vec_from_to A D :=
by sorry

end NUMINAMATH_CALUDE_vector_addition_path_l158_15880


namespace NUMINAMATH_CALUDE_expression_equivalence_l158_15829

theorem expression_equivalence (a b : ℝ) : (a) - (b) - 3 * (a + b) - b = a - 8 * b := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l158_15829


namespace NUMINAMATH_CALUDE_difference_of_ones_and_zeros_313_l158_15804

/-- The number of zeros in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

/-- The number of ones in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

theorem difference_of_ones_and_zeros_313 : 
  count_ones 313 - count_zeros 313 = 3 := by sorry

end NUMINAMATH_CALUDE_difference_of_ones_and_zeros_313_l158_15804


namespace NUMINAMATH_CALUDE_triangle_properties_l158_15855

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-1, -1)

-- Define the equation of angle bisector CD
def angle_bisector_eq (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the equation of the perpendicular bisector of AB
def perp_bisector_eq (x y : ℝ) : Prop := 4*x + 6*y - 3 = 0

-- Define vertex C
def C : ℝ × ℝ := (-1, 2)

theorem triangle_properties :
  (∀ x y : ℝ, angle_bisector_eq x y ↔ x + y - 1 = 0) ∧
  (∀ x y : ℝ, perp_bisector_eq x y ↔ 4*x + 6*y - 3 = 0) ∧
  C = (-1, 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l158_15855


namespace NUMINAMATH_CALUDE_quadratic_roots_square_relation_l158_15857

theorem quadratic_roots_square_relation (q : ℝ) : 
  (∃ (a b : ℝ), a ≠ b ∧ a^2 = b ∧ a^2 - 12*a + q = 0 ∧ b^2 - 12*b + q = 0) →
  (q = 27 ∨ q = -64) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_square_relation_l158_15857


namespace NUMINAMATH_CALUDE_division_example_exists_l158_15856

theorem division_example_exists : ∃ (D d q : ℕ+), 
  (D : ℚ) / (d : ℚ) = q ∧ 
  (q : ℚ) = (D : ℚ) / 5 ∧ 
  (q : ℚ) = 7 * (d : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_division_example_exists_l158_15856


namespace NUMINAMATH_CALUDE_investment_proof_l158_15873

-- Define the interest rates
def interest_rate_x : ℚ := 23 / 100
def interest_rate_y : ℚ := 17 / 100

-- Define the investment in fund X
def investment_x : ℚ := 42000

-- Define the interest difference
def interest_difference : ℚ := 200

-- Define the total investment
def total_investment : ℚ := 100000

-- Theorem statement
theorem investment_proof :
  ∃ (investment_y : ℚ),
    investment_y * interest_rate_y = investment_x * interest_rate_x + interest_difference ∧
    investment_x + investment_y = total_investment :=
by
  sorry


end NUMINAMATH_CALUDE_investment_proof_l158_15873


namespace NUMINAMATH_CALUDE_coffee_machine_discount_l158_15891

def coffee_machine_problem (original_price : ℝ) (home_cost : ℝ) (previous_coffees : ℕ) (previous_price : ℝ) (payoff_days : ℕ) : Prop :=
  let previous_daily_cost := previous_coffees * previous_price
  let daily_savings := previous_daily_cost - home_cost
  let total_savings := daily_savings * payoff_days
  let discount := original_price - total_savings
  original_price = 200 ∧ 
  home_cost = 3 ∧ 
  previous_coffees = 2 ∧ 
  previous_price = 4 ∧ 
  payoff_days = 36 →
  discount = 20

theorem coffee_machine_discount :
  coffee_machine_problem 200 3 2 4 36 :=
by sorry

end NUMINAMATH_CALUDE_coffee_machine_discount_l158_15891


namespace NUMINAMATH_CALUDE_floor_sqrt_15_plus_1_squared_l158_15840

theorem floor_sqrt_15_plus_1_squared : (⌊Real.sqrt 15⌋ + 1)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_15_plus_1_squared_l158_15840


namespace NUMINAMATH_CALUDE_polynomial_roots_l158_15839

theorem polynomial_roots : 
  let f : ℝ → ℝ := λ x => 3*x^4 + 2*x^3 - 7*x^2 + 2*x + 3
  let root1 : ℝ := ((-1 + 2*Real.sqrt 10)/3 + Real.sqrt (((-1 + 2*Real.sqrt 10)/3)^2 - 4))/2
  let root2 : ℝ := ((-1 + 2*Real.sqrt 10)/3 - Real.sqrt (((-1 + 2*Real.sqrt 10)/3)^2 - 4))/2
  let root3 : ℝ := ((-1 - 2*Real.sqrt 10)/3 + Real.sqrt (((-1 - 2*Real.sqrt 10)/3)^2 - 4))/2
  let root4 : ℝ := ((-1 - 2*Real.sqrt 10)/3 - Real.sqrt (((-1 - 2*Real.sqrt 10)/3)^2 - 4))/2
  (f root1 = 0) ∧ (f root2 = 0) ∧ (f root3 = 0) ∧ (f root4 = 0) ∧
  (∀ x : ℝ, f x = 0 → (x = root1 ∨ x = root2 ∨ x = root3 ∨ x = root4)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l158_15839


namespace NUMINAMATH_CALUDE_joggers_meeting_l158_15868

def lap_time_cathy : ℕ := 5
def lap_time_david : ℕ := 9
def lap_time_elena : ℕ := 8

def meeting_time : ℕ := 360
def cathy_laps : ℕ := 72

theorem joggers_meeting :
  (meeting_time % lap_time_cathy = 0) ∧
  (meeting_time % lap_time_david = 0) ∧
  (meeting_time % lap_time_elena = 0) ∧
  (∀ t : ℕ, t < meeting_time →
    ¬(t % lap_time_cathy = 0 ∧ t % lap_time_david = 0 ∧ t % lap_time_elena = 0)) ∧
  (cathy_laps = meeting_time / lap_time_cathy) :=
by sorry

end NUMINAMATH_CALUDE_joggers_meeting_l158_15868


namespace NUMINAMATH_CALUDE_performance_arrangement_count_l158_15890

/-- The number of ways to arrange n elements from a set of k elements --/
def A (k n : ℕ) : ℕ := sorry

/-- The number of ways to choose n elements from a set of k elements, where order matters --/
def P (k n : ℕ) : ℕ := sorry

/-- The number of ways to arrange 6 singing programs and 4 dance programs, 
    where no two dance programs can be adjacent --/
def arrangement_count : ℕ := P 7 4 * A 6 6

theorem performance_arrangement_count : 
  arrangement_count = P 7 4 * A 6 6 := by sorry

end NUMINAMATH_CALUDE_performance_arrangement_count_l158_15890


namespace NUMINAMATH_CALUDE_total_money_sum_l158_15861

theorem total_money_sum (J : ℕ) : 
  (3 * J = 60) → 
  (J + 3 * J + (2 * J - 7) = 113) := by
  sorry

end NUMINAMATH_CALUDE_total_money_sum_l158_15861


namespace NUMINAMATH_CALUDE_anna_guessing_ratio_l158_15812

theorem anna_guessing_ratio (c d : ℝ) 
  (h1 : c > 0 ∧ d > 0)  -- Ensure c and d are positive
  (h2 : 0.9 * c + 0.05 * d = 0.1 * c + 0.95 * d)  -- Equal number of cat and dog images
  (h3 : 0.95 * d = d - 0.05 * d)  -- 95% correct when guessing dog
  (h4 : 0.9 * c = c - 0.1 * c)  -- 90% correct when guessing cat
  : d / c = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_anna_guessing_ratio_l158_15812


namespace NUMINAMATH_CALUDE_min_cost_and_optimal_batch_funds_sufficient_l158_15874

/-- The total cost function for shipping and storage fees -/
def f (x : ℕ+) : ℚ := 144 / x.val + 4 * x.val

/-- The theorem stating the minimum cost and the optimal number of desks per batch -/
theorem min_cost_and_optimal_batch :
  (∀ x : ℕ+, x.val ≤ 36 → f x ≥ 48) ∧
  (∃ x : ℕ+, x.val ≤ 36 ∧ f x = 48 ∧ x.val = 6) := by
  sorry

/-- The theorem stating that the available funds are sufficient for the optimal arrangement -/
theorem funds_sufficient : 
  ∃ x : ℕ+, x.val ≤ 36 ∧ f x ≤ 480 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_and_optimal_batch_funds_sufficient_l158_15874


namespace NUMINAMATH_CALUDE_sum_twenty_ways_l158_15828

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the target sum
def target_sum : ℕ := 20

-- Define the minimum value on a die
def min_value : ℕ := 1

-- Define the maximum value on a die
def max_value : ℕ := 6

-- Function to calculate the number of ways to achieve the target sum
def ways_to_achieve_sum (n d s min max : ℕ) : ℕ :=
  sorry

-- Theorem stating that the number of ways to achieve a sum of 20 with 5 dice is 721
theorem sum_twenty_ways : ways_to_achieve_sum num_dice target_sum min_value max_value = 721 := by
  sorry

end NUMINAMATH_CALUDE_sum_twenty_ways_l158_15828


namespace NUMINAMATH_CALUDE_product_pure_imaginary_l158_15899

theorem product_pure_imaginary (x : ℝ) :
  (∃ y : ℝ, (x + Complex.I) * ((x + 1) + Complex.I) * ((x + 2) + 2 * Complex.I) = Complex.I * y) ↔
  x^3 + 3*x^2 - 9*x - 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_l158_15899


namespace NUMINAMATH_CALUDE_increase_both_averages_l158_15862

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem increase_both_averages :
  ∃ g ∈ group1,
    average (group1.filter (· ≠ g)) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end NUMINAMATH_CALUDE_increase_both_averages_l158_15862


namespace NUMINAMATH_CALUDE_profit_calculation_l158_15851

theorem profit_calculation (x : ℝ) 
  (h1 : 20 * cost_price = x * selling_price)
  (h2 : selling_price = 1.25 * cost_price) : 
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l158_15851


namespace NUMINAMATH_CALUDE_find_a_l158_15870

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem find_a : 
  (∀ x : ℝ, f (1/2 * x - 1) = 2*x - 5) → 
  f (7/4) = 6 := by sorry

end NUMINAMATH_CALUDE_find_a_l158_15870


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l158_15892

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 1) :
  1/x + 3/y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 1 ∧ 1/x₀ + 3/y₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l158_15892


namespace NUMINAMATH_CALUDE_weight_replaced_person_correct_l158_15897

/-- Represents the weight change scenario of a group of people -/
structure WeightChangeScenario where
  initial_count : ℕ
  average_increase : ℝ
  new_person_weight : ℝ

/-- Calculates the weight of the replaced person given a WeightChangeScenario -/
def weight_of_replaced_person (scenario : WeightChangeScenario) : ℝ :=
  scenario.new_person_weight - scenario.initial_count * scenario.average_increase

theorem weight_replaced_person_correct (scenario : WeightChangeScenario) :
  scenario.initial_count = 6 →
  scenario.average_increase = 2.5 →
  scenario.new_person_weight = 80 →
  weight_of_replaced_person scenario = 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_replaced_person_correct_l158_15897


namespace NUMINAMATH_CALUDE_angle_E_is_180_l158_15887

/-- A quadrilateral with specific angle relationships -/
structure SpecialQuadrilateral where
  E : ℝ  -- Angle E in degrees
  F : ℝ  -- Angle F in degrees
  G : ℝ  -- Angle G in degrees
  H : ℝ  -- Angle H in degrees
  angle_sum : E + F + G + H = 360  -- Sum of angles in a quadrilateral
  E_F_relation : E = 3 * F  -- Relationship between E and F
  E_G_relation : E = 2 * G  -- Relationship between E and G
  E_H_relation : E = 6 * H  -- Relationship between E and H

/-- The measure of angle E in the special quadrilateral is 180 degrees -/
theorem angle_E_is_180 (q : SpecialQuadrilateral) : q.E = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_E_is_180_l158_15887


namespace NUMINAMATH_CALUDE_intersection_condition_l158_15810

theorem intersection_condition (a : ℝ) : 
  let M := {x : ℝ | x - a = 0}
  let N := {x : ℝ | a * x - 1 = 0}
  (M ∩ N = N) → (a = 0 ∨ a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_l158_15810


namespace NUMINAMATH_CALUDE_raft_drift_time_l158_15866

/-- The time for a raft to drift from B to A, given boat travel times -/
theorem raft_drift_time (boat_speed : ℝ) (current_speed : ℝ) (distance : ℝ) 
  (h1 : distance / (boat_speed + current_speed) = 7)
  (h2 : distance / (boat_speed - current_speed) = 5)
  (h3 : boat_speed > 0)
  (h4 : current_speed > 0) :
  distance / current_speed = 35 := by
sorry

end NUMINAMATH_CALUDE_raft_drift_time_l158_15866


namespace NUMINAMATH_CALUDE_penny_excess_purchase_l158_15859

/-- Calculates the excess pounds of honey purchased above the minimum spend -/
def excess_honey_purchased (bulk_price : ℚ) (min_spend : ℚ) (tax_per_pound : ℚ) (total_paid : ℚ) : ℚ :=
  let total_price_per_pound := bulk_price + tax_per_pound
  let pounds_purchased := total_paid / total_price_per_pound
  let min_pounds := min_spend / bulk_price
  pounds_purchased - min_pounds

/-- Theorem stating that Penny's purchase exceeded the minimum spend by 32 pounds -/
theorem penny_excess_purchase :
  excess_honey_purchased 5 40 1 240 = 32 := by
  sorry

end NUMINAMATH_CALUDE_penny_excess_purchase_l158_15859


namespace NUMINAMATH_CALUDE_geometric_sum_example_l158_15808

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of the geometric sequence with first term 1/3 and common ratio 1/3 -/
theorem geometric_sum_example : geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_example_l158_15808


namespace NUMINAMATH_CALUDE_parabola_directrix_l158_15865

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 3 * x^2 - 6 * x + 1

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = -25 / 12

/-- Theorem: The directrix of the given parabola has the equation y = -25/12 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_equation x y → ∃ y_directrix : ℝ, directrix_equation y_directrix :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l158_15865


namespace NUMINAMATH_CALUDE_log_sum_equals_zero_l158_15834

theorem log_sum_equals_zero :
  Real.log 2 + Real.log 5 + Real.log 0.5 / Real.log 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_zero_l158_15834


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_79_l158_15827

theorem gcd_of_powers_of_79 :
  Nat.Prime 79 →
  Nat.gcd (79^7 + 1) (79^7 + 79^3 + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_79_l158_15827


namespace NUMINAMATH_CALUDE_f_properties_l158_15888

noncomputable def f (x : ℝ) := Real.log (|x| + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (f 0 = 0 ∧ ∀ x : ℝ, f x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l158_15888


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_union_implies_a_range_l158_15815

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + (a-1)*x + a^2 - 5 = 0}

-- Part 1: A ∩ B = {2} implies a = -3 or a = 1
theorem intersection_implies_a_values (a : ℝ) : 
  A ∩ B a = {2} → a = -3 ∨ a = 1 := by sorry

-- Part 2: A ∪ B = A implies a ≤ -3 or a > 7/3
theorem union_implies_a_range (a : ℝ) :
  A ∪ B a = A → a ≤ -3 ∨ a > 7/3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_union_implies_a_range_l158_15815


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l158_15801

theorem sum_of_a_and_b (a b : ℝ) : 
  (∀ x : ℝ, (x-a)/(x-b) > 0 ↔ x ∈ Set.Ioi 4 ∪ Set.Iic 1) → 
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l158_15801


namespace NUMINAMATH_CALUDE_rectangle_area_l158_15803

theorem rectangle_area (w l : ℕ) : 
  (2 * (w + l) = 60) →  -- Perimeter is 60 units
  (l = w + 1) →         -- Length and width are consecutive integers
  (w * l = 210)         -- Area is 210 square units
:= by sorry

end NUMINAMATH_CALUDE_rectangle_area_l158_15803


namespace NUMINAMATH_CALUDE_carly_butterfly_practice_l158_15800

/-- The number of days Carly practices butterfly stroke per week -/
def butterfly_days : ℕ := sorry

/-- Hours of butterfly stroke practice per day -/
def butterfly_hours_per_day : ℕ := 3

/-- Days of backstroke practice per week -/
def backstroke_days_per_week : ℕ := 6

/-- Hours of backstroke practice per day -/
def backstroke_hours_per_day : ℕ := 2

/-- Total hours of swimming practice per month -/
def total_practice_hours : ℕ := 96

/-- Number of weeks in a month -/
def weeks_per_month : ℕ := 4

theorem carly_butterfly_practice :
  butterfly_days = 4 ∧
  butterfly_days * butterfly_hours_per_day * weeks_per_month +
  backstroke_days_per_week * backstroke_hours_per_day * weeks_per_month =
  total_practice_hours :=
sorry

end NUMINAMATH_CALUDE_carly_butterfly_practice_l158_15800


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l158_15824

theorem scientific_notation_equality : 
  122254 = 1.22254 * (10 : ℝ) ^ 5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l158_15824


namespace NUMINAMATH_CALUDE_smallest_integer_solution_three_is_solution_three_is_smallest_solution_l158_15836

theorem smallest_integer_solution (x : ℤ) : (6 - 3 * x < 0) → x ≥ 3 :=
by sorry

theorem three_is_solution : 6 - 3 * 3 < 0 :=
by sorry

theorem three_is_smallest_solution : ∀ y : ℤ, y < 3 → 6 - 3 * y ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_three_is_solution_three_is_smallest_solution_l158_15836


namespace NUMINAMATH_CALUDE_max_d_value_l158_15872

def a (n : ℕ) : ℕ := 150 + (n + 1)^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (k : ℕ), d k = 2 ∧ ∀ (n : ℕ), d n ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l158_15872


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l158_15802

theorem condition_necessary_not_sufficient 
  (a₁ a₂ b₁ b₂ : ℝ) 
  (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ b₁ ≠ 0 ∧ b₂ ≠ 0) 
  (A : Set ℝ) 
  (hA : A = {x : ℝ | a₁ * x + b₁ > 0}) 
  (B : Set ℝ) 
  (hB : B = {x : ℝ | a₂ * x + b₂ > 0}) : 
  (∀ (A B : Set ℝ), A = B → a₁ / a₂ = b₁ / b₂) ∧ 
  ¬(∀ (A B : Set ℝ), a₁ / a₂ = b₁ / b₂ → A = B) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l158_15802


namespace NUMINAMATH_CALUDE_library_book_sorting_l158_15842

theorem library_book_sorting (total_removed : ℕ) (damaged : ℕ) (x : ℚ) 
  (h1 : total_removed = 69)
  (h2 : damaged = 11)
  (h3 : total_removed = damaged + (x * damaged - 8)) :
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_library_book_sorting_l158_15842


namespace NUMINAMATH_CALUDE_divisibility_condition_l158_15848

theorem divisibility_condition (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * y ∣ x^2 + 2*y - 1) ↔ 
  ((x = 1 ∧ y > 0) ∨ 
   (∃ t : ℕ, t > 0 ∧ x = 2*t - 1 ∧ y = t) ∨ 
   (x = 3 ∧ y = 8) ∨ 
   (x = 5 ∧ y = 8)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l158_15848


namespace NUMINAMATH_CALUDE_problem_statement_l158_15847

theorem problem_statement (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -2) :
  (1 - x) * (1 - y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l158_15847


namespace NUMINAMATH_CALUDE_banana_sharing_l158_15860

theorem banana_sharing (total_bananas : ℕ) (num_friends : ℕ) (bananas_per_friend : ℕ) :
  total_bananas = 21 →
  num_friends = 3 →
  total_bananas = num_friends * bananas_per_friend →
  bananas_per_friend = 7 := by
  sorry

end NUMINAMATH_CALUDE_banana_sharing_l158_15860


namespace NUMINAMATH_CALUDE_physics_class_size_l158_15819

theorem physics_class_size 
  (total_students : ℕ) 
  (physics_students : ℕ) 
  (math_students : ℕ) 
  (both_subjects : ℕ) :
  total_students = 100 →
  physics_students = math_students + both_subjects →
  physics_students = 2 * math_students →
  both_subjects = 10 →
  physics_students = 62 := by
sorry

end NUMINAMATH_CALUDE_physics_class_size_l158_15819


namespace NUMINAMATH_CALUDE_power_sixteen_divided_by_eight_l158_15882

theorem power_sixteen_divided_by_eight (m : ℕ) : m = 16^2023 → m / 8 = 2^8089 := by
  sorry

end NUMINAMATH_CALUDE_power_sixteen_divided_by_eight_l158_15882


namespace NUMINAMATH_CALUDE_somu_age_problem_l158_15878

/-- Somu's age problem -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_back : ℕ) :
  somu_age = 18 →
  somu_age = father_age / 3 →
  somu_age - years_back = (father_age - years_back) / 5 →
  years_back = 9 := by
  sorry

end NUMINAMATH_CALUDE_somu_age_problem_l158_15878


namespace NUMINAMATH_CALUDE_prob_angle_AQB_obtuse_l158_15875

/-- Pentagon ABCDE with vertices A(0,3), B(5,0), C(2π,0), D(2π,5), E(0,5) -/
def pentagon : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.1 ≤ 2*Real.pi ∧ p.2 ≥ 0 ∧ p.2 ≤ 5 ∧
       (p.2 ≥ 3 - 3/5 * p.1 ∨ p.1 ≥ 2*Real.pi)}

/-- Point A -/
def A : ℝ × ℝ := (0, 3)

/-- Point B -/
def B : ℝ × ℝ := (5, 0)

/-- Random point Q in the pentagon -/
def Q : ℝ × ℝ := sorry

/-- Angle AQB -/
def angle_AQB : ℝ := sorry

/-- Probability measure on the pentagon -/
def prob : MeasureTheory.Measure (ℝ × ℝ) := sorry

/-- The probability that angle AQB is obtuse -/
theorem prob_angle_AQB_obtuse :
  prob {q ∈ pentagon | angle_AQB > Real.pi/2} / prob pentagon = 17/128 := by sorry

end NUMINAMATH_CALUDE_prob_angle_AQB_obtuse_l158_15875


namespace NUMINAMATH_CALUDE_integer_division_property_l158_15869

theorem integer_division_property (n : ℕ+) : 
  (∃ k : ℤ, (2^(n : ℕ) + 1 : ℤ) = k * (n : ℤ)^2) ↔ n = 1 ∨ n = 3 := by
sorry

end NUMINAMATH_CALUDE_integer_division_property_l158_15869


namespace NUMINAMATH_CALUDE_cartesian_points_proof_l158_15846

/-- Given points P and Q in the Cartesian coordinate system, prove cos 2θ and sin(α + β) -/
theorem cartesian_points_proof (θ α β : Real) : 
  let P : Real × Real := (1/2, Real.cos θ ^ 2)
  let Q : Real × Real := (Real.sin θ ^ 2, -1)
  (P.1 * Q.1 + P.2 * Q.2 = -1/2) → 
  (Real.cos (2 * θ) = 1/3 ∧ 
   Real.sin (α + β) = -Real.sqrt 10 / 10) := by
sorry

end NUMINAMATH_CALUDE_cartesian_points_proof_l158_15846


namespace NUMINAMATH_CALUDE_fraction_decimal_digits_l158_15835

theorem fraction_decimal_digits : 
  let f : ℚ := 90 / (3^2 * 2^5)
  ∃ (d : ℕ) (n : ℕ), f = d.cast / 10^n ∧ (d % 10 ≠ 0) ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_decimal_digits_l158_15835


namespace NUMINAMATH_CALUDE_math_score_proof_l158_15894

theorem math_score_proof (a b c : ℕ) : 
  (a + b + c = 288) →  -- Sum of scores is 288
  (∃ k : ℕ, a = 2*k ∧ b = 2*k + 2 ∧ c = 2*k + 4) →  -- Consecutive even numbers
  b = 96  -- Mathematics score is 96
:= by sorry

end NUMINAMATH_CALUDE_math_score_proof_l158_15894


namespace NUMINAMATH_CALUDE_sin_2x_value_l158_15825

theorem sin_2x_value (x : ℝ) : 
  (Real.cos (4 * π / 5) * Real.cos (7 * π / 15) - Real.sin (9 * π / 5) * Real.sin (7 * π / 15) = 
   Real.cos (x + π / 2) * Real.cos x + 2 / 3) → 
  Real.sin (2 * x) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_2x_value_l158_15825


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l158_15883

theorem subcommittee_formation_count : 
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 7
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (Nat.choose total_republicans subcommittee_republicans) * 
  (Nat.choose total_democrats subcommittee_democrats) = 7350 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l158_15883


namespace NUMINAMATH_CALUDE_berries_taken_l158_15858

theorem berries_taken (stacy_initial : ℕ) (steve_initial : ℕ) (difference : ℕ) : 
  stacy_initial = 32 →
  steve_initial = 21 →
  difference = 7 →
  ∃ (berries_taken : ℕ), 
    steve_initial + berries_taken = stacy_initial - difference ∧
    berries_taken = 4 :=
by sorry

end NUMINAMATH_CALUDE_berries_taken_l158_15858


namespace NUMINAMATH_CALUDE_total_worth_is_22800_l158_15814

def engagement_ring_cost : ℝ := 4000
def car_cost : ℝ := 2000
def diamond_bracelet_cost : ℝ := 2 * engagement_ring_cost
def designer_gown_cost : ℝ := 0.5 * diamond_bracelet_cost
def jewelry_set_cost : ℝ := 1.2 * engagement_ring_cost

def total_worth : ℝ := engagement_ring_cost + car_cost + diamond_bracelet_cost + designer_gown_cost + jewelry_set_cost

theorem total_worth_is_22800 : total_worth = 22800 := by sorry

end NUMINAMATH_CALUDE_total_worth_is_22800_l158_15814


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l158_15805

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l158_15805


namespace NUMINAMATH_CALUDE_probability_score_le_6_is_13_35_l158_15811

structure Bag where
  red_balls : ℕ
  black_balls : ℕ

def score (red : ℕ) (black : ℕ) : ℕ :=
  red + 3 * black

def probability_score_le_6 (b : Bag) : ℚ :=
  let total_balls := b.red_balls + b.black_balls
  let drawn_balls := 4
  (Nat.choose b.red_balls 4 * Nat.choose b.black_balls 0 +
   Nat.choose b.red_balls 3 * Nat.choose b.black_balls 1) /
  Nat.choose total_balls drawn_balls

theorem probability_score_le_6_is_13_35 (b : Bag) :
  b.red_balls = 4 → b.black_balls = 3 → probability_score_le_6 b = 13 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_score_le_6_is_13_35_l158_15811


namespace NUMINAMATH_CALUDE_brennan_pepper_proof_l158_15821

/-- The amount of pepper Brennan used (in grams) -/
def pepper_used : ℝ := 0.16

/-- The amount of pepper Brennan has left (in grams) -/
def pepper_left : ℝ := 0.09

/-- The initial amount of pepper Brennan had (in grams) -/
def initial_pepper : ℝ := pepper_used + pepper_left

theorem brennan_pepper_proof : initial_pepper = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_brennan_pepper_proof_l158_15821


namespace NUMINAMATH_CALUDE_unique_number_digit_sum_l158_15838

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem unique_number_digit_sum :
  ∃! N : ℕ, 400 < N ∧ N < 600 ∧ N % 2 = 1 ∧ N % 5 = 0 ∧ N % 11 = 0 ∧ sumOfDigits N = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_number_digit_sum_l158_15838


namespace NUMINAMATH_CALUDE_boys_percentage_of_school_l158_15833

theorem boys_percentage_of_school (total_students : ℕ) (boys_representation : ℕ) 
  (h1 : total_students = 180)
  (h2 : boys_representation = 162)
  (h3 : boys_representation = (180 / 100) * (boys_percentage / 100 * total_students)) :
  boys_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_boys_percentage_of_school_l158_15833


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l158_15831

theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℤ) -- a is the arithmetic sequence
  (h1 : a 0 = 3^2) -- first term is 3^2
  (h2 : a 2 = 3^4) -- third term is 3^4
  (h3 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- arithmetic sequence
  : a 1 = 45 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l158_15831


namespace NUMINAMATH_CALUDE_arcsin_one_half_l158_15822

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_l158_15822


namespace NUMINAMATH_CALUDE_complex_division_l158_15826

/-- Given a complex number z = 1 + ai where a is a positive real number and |z| = √10,
    prove that z / (1 - 2i) = -1 + i -/
theorem complex_division (a : ℝ) (z : ℂ) (h1 : a > 0) (h2 : z = 1 + a * Complex.I) 
    (h3 : Complex.abs z = Real.sqrt 10) : 
  z / (1 - 2 * Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l158_15826


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l158_15853

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l158_15853


namespace NUMINAMATH_CALUDE_largest_sample_size_l158_15823

def population : Nat := 36

theorem largest_sample_size (X : Nat) : 
  (X > 0 ∧ 
   population % X = 0 ∧ 
   population % (X + 1) ≠ 0 ∧ 
   ∀ Y : Nat, Y > X → (population % Y = 0 → population % (Y + 1) = 0)) → 
  X = 9 := by sorry

end NUMINAMATH_CALUDE_largest_sample_size_l158_15823


namespace NUMINAMATH_CALUDE_interest_related_to_gender_l158_15854

/-- Represents the chi-square statistic -/
def chi_square : ℝ := 3.918

/-- Represents the critical value -/
def critical_value : ℝ := 3.841

/-- The probability that the chi-square statistic is greater than or equal to the critical value -/
def p_value : ℝ := 0.05

/-- The confidence level -/
def confidence_level : ℝ := 1 - p_value

theorem interest_related_to_gender :
  chi_square > critical_value →
  confidence_level = 0.95 →
  ∃ (relation : Prop), relation ∧ confidence_level = 0.95 :=
by sorry

end NUMINAMATH_CALUDE_interest_related_to_gender_l158_15854


namespace NUMINAMATH_CALUDE_minimum_students_l158_15895

theorem minimum_students (b g : ℕ) : 
  b > 0 → 
  g > 0 → 
  2 * (b / 2) = g * 2 / 3 → 
  ∀ b' g', b' > 0 → g' > 0 → 2 * (b' / 2) = g' * 2 / 3 → b' + g' ≥ b + g →
  b + g = 5 := by
sorry

end NUMINAMATH_CALUDE_minimum_students_l158_15895


namespace NUMINAMATH_CALUDE_root_quadratic_equation_property_l158_15807

theorem root_quadratic_equation_property (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → 2*m^2 - 4*m + 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_property_l158_15807


namespace NUMINAMATH_CALUDE_solution_set_is_closed_interval_l158_15863

def system_solution (x : ℝ) : Prop :=
  -2 * (x - 3) > 10 ∧ x^2 + 7*x + 12 ≤ 0

theorem solution_set_is_closed_interval :
  {x : ℝ | system_solution x} = {x : ℝ | -4 ≤ x ∧ x ≤ -3} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_is_closed_interval_l158_15863


namespace NUMINAMATH_CALUDE_dogsled_race_speed_difference_l158_15871

/-- Proves that the difference in average speeds between two teams is 5 mph
    given specific conditions of a dogsled race. -/
theorem dogsled_race_speed_difference
  (course_length : ℝ)
  (team_r_speed : ℝ)
  (time_difference : ℝ)
  (h1 : course_length = 300)
  (h2 : team_r_speed = 20)
  (h3 : time_difference = 3)
  : ∃ (team_a_speed : ℝ),
    team_a_speed = course_length / (course_length / team_r_speed - time_difference) ∧
    team_a_speed - team_r_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_dogsled_race_speed_difference_l158_15871


namespace NUMINAMATH_CALUDE_gears_can_rotate_l158_15886

/-- A gear system with n identical gears arranged in a closed loop. -/
structure GearSystem where
  n : ℕ
  is_closed_loop : n ≥ 2

/-- Represents the rotation direction of a gear. -/
inductive RotationDirection
  | Clockwise
  | Counterclockwise

/-- Function to determine if adjacent gears have opposite rotation directions. -/
def opposite_rotation (d1 d2 : RotationDirection) : Prop :=
  (d1 = RotationDirection.Clockwise ∧ d2 = RotationDirection.Counterclockwise) ∨
  (d1 = RotationDirection.Counterclockwise ∧ d2 = RotationDirection.Clockwise)

/-- Theorem stating that the gears can rotate if and only if the number of gears is even. -/
theorem gears_can_rotate (system : GearSystem) :
  (∃ (rotation : ℕ → RotationDirection), 
    (∀ i : ℕ, i < system.n → opposite_rotation (rotation i) (rotation ((i + 1) % system.n))) ∧
    opposite_rotation (rotation 0) (rotation (system.n - 1)))
  ↔ 
  Even system.n :=
sorry

end NUMINAMATH_CALUDE_gears_can_rotate_l158_15886


namespace NUMINAMATH_CALUDE_library_book_count_l158_15816

/-- The number of books in a library after two years of purchases -/
def total_books (initial : ℕ) (last_year : ℕ) (multiplier : ℕ) : ℕ :=
  initial + last_year + multiplier * last_year

/-- Theorem: The library now has 300 books -/
theorem library_book_count : total_books 100 50 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l158_15816


namespace NUMINAMATH_CALUDE_range_of_m_l158_15809

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 3| ≥ m + 4) → m ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l158_15809


namespace NUMINAMATH_CALUDE_nathan_tokens_used_l158_15806

/-- The total number of tokens used by Nathan at the arcade --/
def total_tokens (air_hockey_plays : ℕ) (basketball_plays : ℕ) (skee_ball_plays : ℕ)
                 (air_hockey_cost : ℕ) (basketball_cost : ℕ) (skee_ball_cost : ℕ) : ℕ :=
  air_hockey_plays * air_hockey_cost +
  basketball_plays * basketball_cost +
  skee_ball_plays * skee_ball_cost

/-- Theorem stating that Nathan used 64 tokens in total --/
theorem nathan_tokens_used :
  total_tokens 5 7 3 4 5 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_nathan_tokens_used_l158_15806
