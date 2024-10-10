import Mathlib

namespace premium_rate_calculation_l2466_246607

/-- Given a tempo insured to 4/5 of its original value of $87,500, with a premium of $910,
    the rate of the premium is 1.3%. -/
theorem premium_rate_calculation (original_value : ℝ) (insurance_ratio : ℝ) (premium : ℝ) :
  original_value = 87500 →
  insurance_ratio = 4 / 5 →
  premium = 910 →
  (premium / (insurance_ratio * original_value)) * 100 = 1.3 := by
  sorry

end premium_rate_calculation_l2466_246607


namespace equation_solvable_l2466_246683

/-- For a given real number b, this function represents the equation x - b = ∑_{k=0}^∞ x^k -/
def equation (b x : ℝ) : Prop :=
  x - b = (∑' k, x^k)

/-- This theorem states the conditions on b for which the equation has solutions -/
theorem equation_solvable (b : ℝ) : 
  (∃ x : ℝ, equation b x) ↔ (b ≤ -1 ∨ (-3/2 < b ∧ b ≤ -1)) :=
sorry

end equation_solvable_l2466_246683


namespace alphametic_puzzle_solution_l2466_246648

theorem alphametic_puzzle_solution :
  ∃! (T H E B G M A : ℕ),
    T ≠ H ∧ T ≠ E ∧ T ≠ B ∧ T ≠ G ∧ T ≠ M ∧ T ≠ A ∧
    H ≠ E ∧ H ≠ B ∧ H ≠ G ∧ H ≠ M ∧ H ≠ A ∧
    E ≠ B ∧ E ≠ G ∧ E ≠ M ∧ E ≠ A ∧
    B ≠ G ∧ B ≠ M ∧ B ≠ A ∧
    G ≠ M ∧ G ≠ A ∧
    M ≠ A ∧
    T < 10 ∧ H < 10 ∧ E < 10 ∧ B < 10 ∧ G < 10 ∧ M < 10 ∧ A < 10 ∧
    1000 * T + 100 * H + 10 * E + T + 1000 * B + 100 * E + 10 * T + A =
    10000 * G + 1000 * A + 100 * M + 10 * M + A ∧
    T = 4 ∧ H = 9 ∧ E = 4 ∧ B = 5 ∧ G = 1 ∧ M = 8 ∧ A = 0 :=
by sorry

end alphametic_puzzle_solution_l2466_246648


namespace otimes_equation_solution_l2466_246660

/-- Custom binary operator ⊗ -/
def otimes (a b : ℝ) : ℝ := -2 * a + b

/-- Theorem stating that if x ⊗ (-5) = 3, then x = -4 -/
theorem otimes_equation_solution (x : ℝ) (h : otimes x (-5) = 3) : x = -4 := by
  sorry

end otimes_equation_solution_l2466_246660


namespace intersection_empty_implies_a_range_l2466_246647

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - a| ≤ 1}
def B : Set ℝ := {x : ℝ | x^2 - 5*x + 4 ≥ 0}

-- State the theorem
theorem intersection_empty_implies_a_range (a : ℝ) : A a ∩ B = ∅ → 2 < a ∧ a < 3 := by
  sorry

end intersection_empty_implies_a_range_l2466_246647


namespace menu_restriction_l2466_246697

theorem menu_restriction (total_dishes : ℕ) (sugar_free_ratio : ℚ) (shellfish_free_ratio : ℚ)
  (h1 : sugar_free_ratio = 1 / 10)
  (h2 : shellfish_free_ratio = 3 / 4) :
  (sugar_free_ratio * shellfish_free_ratio : ℚ) = 3 / 40 := by
  sorry

end menu_restriction_l2466_246697


namespace exists_special_six_digit_number_l2466_246618

/-- A six-digit number is between 100000 and 999999 -/
def SixDigitNumber (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- The last six digits of a number -/
def LastSixDigits (n : ℕ) : ℕ := n % 1000000

theorem exists_special_six_digit_number :
  ∃ A : ℕ, SixDigitNumber A ∧
    ∀ k m : ℕ, 1 ≤ k → k < m → m ≤ 500000 →
      LastSixDigits (k * A) ≠ LastSixDigits (m * A) := by
  sorry

end exists_special_six_digit_number_l2466_246618


namespace solution_difference_l2466_246665

-- Define the equation
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + x - 20) = x + 3

-- Define the theorem
theorem solution_difference :
  ∃ (p q : ℝ), 
    p ≠ q ∧
    equation p ∧
    equation q ∧
    p > q ∧
    p - q = 5 :=
by
  sorry

end solution_difference_l2466_246665


namespace sqrt_D_irrational_l2466_246626

/-- Given even integers a and b where b = a + 2, and c = ab, √(a^2 + b^2 + c^2) is always irrational. -/
theorem sqrt_D_irrational (a b c : ℤ) : 
  Even a → Even b → b = a + 2 → c = a * b → 
  Irrational (Real.sqrt ((a^2 : ℝ) + b^2 + c^2)) := by
  sorry

end sqrt_D_irrational_l2466_246626


namespace function_properties_l2466_246695

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 8)*x

/-- The theorem stating the properties of the function and the results to be proved -/
theorem function_properties (a m : ℝ) :
  (∀ x, f a x ≤ 5 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∀ x, f a x ≥ m^2 - 4*m - 9) →
  a = 2 ∧ -1 ≤ m ∧ m ≤ 5 := by sorry

end function_properties_l2466_246695


namespace projection_vector_equals_result_l2466_246637

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 2]

theorem projection_vector_equals_result :
  let proj := (a • b) / (b • b) • b
  proj 0 = -3/5 ∧ proj 1 = 6/5 := by
  sorry

end projection_vector_equals_result_l2466_246637


namespace power_four_inequality_l2466_246613

theorem power_four_inequality (x y : ℝ) : x^4 + y^4 ≥ x*y*(x + y)^2 := by
  sorry

end power_four_inequality_l2466_246613


namespace mrs_excellent_class_size_l2466_246649

/-- Represents the number of students in Mrs. Excellent's class -/
def total_students : ℕ := 29

/-- Represents the number of girls in the class -/
def girls : ℕ := 13

/-- Represents the number of boys in the class -/
def boys : ℕ := girls + 3

/-- Represents the total number of jellybeans Mrs. Excellent has -/
def total_jellybeans : ℕ := 450

/-- Represents the number of jellybeans left after distribution -/
def leftover_jellybeans : ℕ := 10

theorem mrs_excellent_class_size :
  (girls * girls + boys * boys + leftover_jellybeans = total_jellybeans) ∧
  (girls + boys = total_students) := by
  sorry

#check mrs_excellent_class_size

end mrs_excellent_class_size_l2466_246649


namespace david_scott_age_difference_l2466_246666

/-- Represents the ages of three brothers -/
structure BrotherAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- Defines the conditions given in the problem -/
def satisfiesConditions (ages : BrotherAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- Theorem stating that David is 8 years older than Scott -/
theorem david_scott_age_difference (ages : BrotherAges) :
  satisfiesConditions ages → ages.david - ages.scott = 8 := by
  sorry

end david_scott_age_difference_l2466_246666


namespace sum_is_linear_l2466_246614

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies the described transformations to a parabola -/
def transform (p : Parabola) : ℝ → ℝ := 
  fun x => p.a * (x - 4)^2 + p.b * (x - 4) + p.c + 2

/-- Applies the described transformations to the reflection of a parabola -/
def transform_reflection (p : Parabola) : ℝ → ℝ := 
  fun x => -p.a * (x + 6)^2 - p.b * (x + 6) - p.c + 2

/-- The sum of the transformed parabola and its reflection -/
def sum_of_transformations (p : Parabola) : ℝ → ℝ :=
  fun x => transform p x + transform_reflection p x

theorem sum_is_linear (p : Parabola) : 
  ∀ x, sum_of_transformations p x = -20 * p.a * x + 52 * p.a - 10 * p.b + 4 :=
by sorry

end sum_is_linear_l2466_246614


namespace sara_initial_quarters_l2466_246690

/-- The number of quarters Sara had initially -/
def initial_quarters : ℕ := sorry

/-- The number of quarters Sara's dad gave her -/
def dads_gift : ℕ := 49

/-- The total number of quarters Sara has after her dad's gift -/
def total_quarters : ℕ := 70

/-- Theorem stating that Sara's initial number of quarters was 21 -/
theorem sara_initial_quarters : initial_quarters = 21 := by
  sorry

end sara_initial_quarters_l2466_246690


namespace line_point_distance_condition_l2466_246667

theorem line_point_distance_condition (a : ℝ) : 
  (∃ x y : ℝ, a * x + y + 2 = 0 ∧ 
    ((x + 3)^2 + y^2)^(1/2) = 2 * (x^2 + y^2)^(1/2)) → 
  a ≤ 0 ∨ a ≥ 4/3 := by
sorry

end line_point_distance_condition_l2466_246667


namespace johnny_money_left_l2466_246645

def johnny_savings (september october november : ℕ) : ℕ := september + october + november

theorem johnny_money_left (september october november spending : ℕ) 
  (h1 : september = 30)
  (h2 : october = 49)
  (h3 : november = 46)
  (h4 : spending = 58) :
  johnny_savings september october november - spending = 67 := by
  sorry

end johnny_money_left_l2466_246645


namespace plane_perp_necessary_not_sufficient_l2466_246662

/-- Two planes are perpendicular -/
def planes_perpendicular (α β : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (m : Line) (β : Plane) : Prop := sorry

/-- A line lies in a plane -/
def line_in_plane (m : Line) (α : Plane) : Prop := sorry

theorem plane_perp_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (h_different : α ≠ β)
  (h_m_in_α : line_in_plane m α) :
  (planes_perpendicular α β → line_perpendicular_to_plane m β) ∧
  ¬(line_perpendicular_to_plane m β → planes_perpendicular α β) :=
sorry

end plane_perp_necessary_not_sufficient_l2466_246662


namespace inequality_proof_l2466_246675

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end inequality_proof_l2466_246675


namespace train_length_is_100m_l2466_246630

-- Define the given constants
def train_speed : Real := 60  -- km/h
def bridge_length : Real := 80  -- meters
def crossing_time : Real := 10.799136069114471  -- seconds

-- Theorem to prove
theorem train_length_is_100m :
  let speed_ms : Real := train_speed * 1000 / 3600  -- Convert km/h to m/s
  let total_distance : Real := speed_ms * crossing_time
  let train_length : Real := total_distance - bridge_length
  train_length = 100 := by sorry

end train_length_is_100m_l2466_246630


namespace sequence_formula_l2466_246636

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) - a n = r * (a n - a (n - 1))

theorem sequence_formula (a : ℕ → ℝ) :
  geometric_sequence (λ n => a (n + 1) - a n) ∧
  (a 2 - a 1 = 1) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) - a n = (1 / 3) * (a n - a (n - 1))) →
  ∀ n : ℕ, n > 0 → a n = (3 / 2) * (1 - (1 / 3) ^ n) :=
by sorry

end sequence_formula_l2466_246636


namespace dans_eggs_l2466_246610

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Dan bought -/
def dans_dozens : ℕ := 9

/-- Theorem: Dan bought 108 eggs -/
theorem dans_eggs : dans_dozens * eggs_per_dozen = 108 := by
  sorry

end dans_eggs_l2466_246610


namespace range_of_a_l2466_246657

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 7)
  (h_f2 : f 2 > 1)
  (h_f2014 : f 2014 = (a + 3) / (a - 3)) :
  0 < a ∧ a < 3 := by
  sorry

end range_of_a_l2466_246657


namespace negative_roots_condition_l2466_246671

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - (a + 1) * x + a + 4

-- Define the condition for both roots being negative
def both_roots_negative (a : ℝ) : Prop :=
  ∀ x : ℝ, quadratic a x = 0 → x < 0

-- Theorem statement
theorem negative_roots_condition :
  ∀ a : ℝ, both_roots_negative a ↔ -4 < a ∧ a ≤ -3 :=
sorry

end negative_roots_condition_l2466_246671


namespace range_of_m_l2466_246617

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 < 0

-- Define the condition that ¬p is a necessary but not sufficient condition for ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  ∀ x, ¬(q x m) → ¬(p x) ∧ ∃ x, ¬(p x) ∧ (q x m)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∀ x, p x → q x m) ∧ not_p_necessary_not_sufficient_for_not_q m
  ↔ m ≥ 9 ∨ m ≤ -9 :=
sorry

end range_of_m_l2466_246617


namespace extracurricular_materials_selection_l2466_246677

theorem extracurricular_materials_selection : 
  let total_materials : ℕ := 6
  let materials_per_student : ℕ := 2
  let common_materials : ℕ := 1
  
  (total_materials.choose common_materials) * 
  ((total_materials - common_materials).choose (materials_per_student - common_materials)) = 60 :=
by sorry

end extracurricular_materials_selection_l2466_246677


namespace projection_problem_l2466_246659

/-- Given a projection that takes (2, -3) to (1, -3/2), 
    prove that the projection of (3, -2) is (24/13, -36/13) -/
theorem projection_problem (proj : ℝ × ℝ → ℝ × ℝ) 
  (h : proj (2, -3) = (1, -3/2)) :
  proj (3, -2) = (24/13, -36/13) := by
  sorry

end projection_problem_l2466_246659


namespace coin_sum_bounds_l2466_246603

def coin_values : List ℕ := [1, 1, 1, 5, 10, 10, 25, 50]

theorem coin_sum_bounds (coins : List ℕ) (h : coins = coin_values) :
  (∃ (a b : ℕ), a ∈ coins ∧ b ∈ coins ∧ a + b = 2) ∧
  (∃ (c d : ℕ), c ∈ coins ∧ d ∈ coins ∧ c + d = 75) ∧
  (∀ (x y : ℕ), x ∈ coins → y ∈ coins → 2 ≤ x + y ∧ x + y ≤ 75) :=
by sorry

end coin_sum_bounds_l2466_246603


namespace hyperbola_eccentricity_l2466_246601

/-- A hyperbola with foci F₁ and F₂, and endpoints of conjugate axis B₁ and B₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  B₁ : ℝ × ℝ
  B₂ : ℝ × ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The angle B₂F₁B₁ in a hyperbola -/
def angle_B₂F₁B₁ (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  angle_B₂F₁B₁ h = π/3 → eccentricity h = Real.sqrt 6 / 2 := by
  sorry

end hyperbola_eccentricity_l2466_246601


namespace solution_equality_implies_k_equals_one_l2466_246644

theorem solution_equality_implies_k_equals_one :
  ∀ x k : ℝ,
  (2 * x - 1 = 3 * x - 2) →
  (4 - (k * x + 2) / 3 = 3 * k - (2 - 2 * x) / 4) →
  k = 1 := by
sorry

end solution_equality_implies_k_equals_one_l2466_246644


namespace entertainment_percentage_l2466_246606

def monthly_salary : ℝ := 5000
def food_percentage : ℝ := 40
def rent_percentage : ℝ := 20
def conveyance_percentage : ℝ := 10
def savings : ℝ := 1000

theorem entertainment_percentage :
  let total_known_expenses := food_percentage + rent_percentage + conveyance_percentage
  let remaining_percentage := 100 - total_known_expenses
  let expected_savings := (remaining_percentage / 100) * monthly_salary
  let entertainment_expense := expected_savings - savings
  entertainment_expense / monthly_salary * 100 = 10 := by sorry

end entertainment_percentage_l2466_246606


namespace digit_multiplication_problem_l2466_246619

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if all elements in a list are different -/
def all_different (list : List Digit) : Prop :=
  ∀ i j, i ≠ j → list.get i ≠ list.get j

/-- Converts a three-digit number represented by digits to a natural number -/
def to_nat_3digit (a b c : Digit) : ℕ :=
  100 * a.val + 10 * b.val + c.val

/-- Converts a two-digit number represented by digits to a natural number -/
def to_nat_2digit (d e : Digit) : ℕ :=
  10 * d.val + e.val

/-- Converts a four-digit number represented by digits to a natural number -/
def to_nat_4digit (d1 d2 e1 e2 : Digit) : ℕ :=
  1000 * d1.val + 100 * d2.val + 10 * e1.val + e2.val

theorem digit_multiplication_problem (A B C D E : Digit) :
  all_different [A, B, C, D, E] →
  to_nat_3digit A B C * to_nat_2digit D E = to_nat_4digit D D E E →
  A.val + B.val = 7 := by
  sorry

end digit_multiplication_problem_l2466_246619


namespace extra_food_is_zero_point_four_l2466_246646

/-- The amount of cat food needed for one cat per day -/
def food_for_one_cat : ℝ := 0.5

/-- The total amount of cat food needed for two cats per day -/
def total_food_for_two_cats : ℝ := 0.9

/-- The extra amount of cat food needed for the second cat per day -/
def extra_food_for_second_cat : ℝ := total_food_for_two_cats - food_for_one_cat

theorem extra_food_is_zero_point_four :
  extra_food_for_second_cat = 0.4 := by
  sorry

end extra_food_is_zero_point_four_l2466_246646


namespace square_difference_given_sum_and_product_l2466_246679

theorem square_difference_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : (x - y)^2 = 4 := by
  sorry

end square_difference_given_sum_and_product_l2466_246679


namespace barrel_capacity_l2466_246696

theorem barrel_capacity (total_capacity : ℝ) (increase : ℝ) (decrease : ℝ)
  (h1 : total_capacity = 7000)
  (h2 : increase = 1000)
  (h3 : decrease = 4000) :
  ∃ (x y : ℝ),
    x + y = total_capacity ∧
    x = 6400 ∧
    y = 600 ∧
    x / (total_capacity + increase) + y / (total_capacity - decrease) = 1 :=
by sorry

end barrel_capacity_l2466_246696


namespace equal_angles_not_always_opposite_l2466_246612

-- Define the basic geometric concepts
variable (Line : Type) (Point : Type) (Angle : Type)
variable (opposite : Angle → Angle → Prop)
variable (equal : Angle → Angle → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (corresponding : Angle → Angle → Prop)

-- State the propositions
axiom opposite_angles_equal : ∀ (a b : Angle), opposite a b → equal a b
axiom perpendicular_lines_parallel : ∀ (l1 l2 l3 : Line), perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2
axiom corresponding_angles_equal : ∀ (a b : Angle), corresponding a b → equal a b

-- State the theorem to be proved
theorem equal_angles_not_always_opposite : ¬(∀ (a b : Angle), equal a b → opposite a b) :=
sorry

end equal_angles_not_always_opposite_l2466_246612


namespace happy_street_weekend_traffic_l2466_246615

/-- Number of cars passing Happy Street each day of the week -/
structure WeekTraffic where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  weekend_day : ℕ

/-- Conditions for the Happy Street traffic problem -/
def happy_street_conditions (w : WeekTraffic) : Prop :=
  w.tuesday = 25 ∧
  w.monday = w.tuesday - (w.tuesday / 5) ∧
  w.wednesday = w.monday + 2 ∧
  w.thursday = 10 ∧
  w.friday = 10 ∧
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + 2 * w.weekend_day = 97

theorem happy_street_weekend_traffic (w : WeekTraffic) 
  (h : happy_street_conditions w) : w.weekend_day = 5 := by
  sorry


end happy_street_weekend_traffic_l2466_246615


namespace subset_implies_a_value_l2466_246635

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 = 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- State the theorem
theorem subset_implies_a_value (a : ℝ) : B a ⊆ A → a ∈ ({0, 1, -1} : Set ℝ) := by
  sorry

end subset_implies_a_value_l2466_246635


namespace chessboard_selections_theorem_l2466_246622

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_valid : size = 4 ∨ size = 8)

/-- Represents a selection of squares on a chessboard -/
structure Selection (board : Chessboard) :=
  (count : Nat)
  (row_count : Nat)
  (col_count : Nat)
  (is_valid : count = row_count * board.size ∧ row_count = col_count)

/-- Counts the number of valid selections on a 4x4 board -/
def count_4x4_selections (board : Chessboard) (sel : Selection board) : Nat :=
  24

/-- Counts the number of valid selections on an 8x8 board with all black squares chosen -/
def count_8x8_selections (board : Chessboard) (sel : Selection board) : Nat :=
  576

/-- The main theorem to prove -/
theorem chessboard_selections_theorem (board4 : Chessboard) (board8 : Chessboard) 
  (sel4 : Selection board4) (sel8 : Selection board8) :
  board4.size = 4 ∧ 
  board8.size = 8 ∧ 
  sel4.count = 12 ∧ 
  sel4.row_count = 3 ∧
  sel8.count = 56 ∧
  sel8.row_count = 7 →
  count_8x8_selections board8 sel8 = (count_4x4_selections board4 sel4) ^ 2 :=
by sorry

end chessboard_selections_theorem_l2466_246622


namespace largest_prime_factor_of_sum_of_divisors_180_l2466_246676

def sum_of_divisors (n : ℕ) : ℕ := sorry

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end largest_prime_factor_of_sum_of_divisors_180_l2466_246676


namespace m_range_l2466_246688

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, x ≤ -1 → (m^2 - m) * 2^x - (1/2)^x < 1) → 
  -2 < m ∧ m < 3 := by
sorry

end m_range_l2466_246688


namespace smallest_solution_of_equation_l2466_246655

theorem smallest_solution_of_equation :
  ∃ (x : ℝ), x = 4 - Real.sqrt 2 ∧
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ (y : ℝ), (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x := by
  sorry

end smallest_solution_of_equation_l2466_246655


namespace brenda_skittles_count_l2466_246604

def final_skittles (initial bought given_away : ℕ) : ℕ :=
  initial + bought - given_away

theorem brenda_skittles_count : final_skittles 7 8 3 = 12 := by
  sorry

end brenda_skittles_count_l2466_246604


namespace collinear_points_reciprocal_sum_l2466_246672

theorem collinear_points_reciprocal_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ (t : ℝ), (2 + t * (a - 2), 2 + t * (-2)) = (0, b)) →
  1 / a + 1 / b = 1 / 2 := by
  sorry

end collinear_points_reciprocal_sum_l2466_246672


namespace unique_solution_system_l2466_246627

theorem unique_solution_system : 
  ∃! (x y z : ℕ+), 
    (x : ℝ)^2 = 2 * ((y : ℝ) + (z : ℝ)) ∧ 
    (x : ℝ)^6 = (y : ℝ)^6 + (z : ℝ)^6 + 31 * ((y : ℝ)^2 + (z : ℝ)^2) ∧
    x = 2 ∧ y = 1 ∧ z = 1 := by
  sorry

end unique_solution_system_l2466_246627


namespace max_sum_is_1446_l2466_246631

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- The set of numbers on each cube --/
def cube_numbers : Finset ℕ := {1, 3, 9, 27, 81, 243}

/-- A valid cube has all numbers from cube_numbers --/
def is_valid_cube (c : Cube) : Prop :=
  ∀ n ∈ cube_numbers, ∃ i : Fin 6, c.faces i = n

/-- The sum of visible faces when cubes are stacked --/
def visible_sum (cubes : Fin 4 → Cube) : ℕ :=
  sorry

/-- The maximum possible sum of visible faces --/
def max_visible_sum : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem max_sum_is_1446 :
  ∀ cubes : Fin 4 → Cube,
  (∀ i : Fin 4, is_valid_cube (cubes i)) →
  visible_sum cubes ≤ 1446 ∧
  ∃ optimal_cubes : Fin 4 → Cube,
    (∀ i : Fin 4, is_valid_cube (optimal_cubes i)) ∧
    visible_sum optimal_cubes = 1446 :=
by sorry

end max_sum_is_1446_l2466_246631


namespace twelve_rolls_in_case_l2466_246674

/-- Calculates the number of rolls in a case of paper towels given the case price, individual roll price, and savings percentage. -/
def rolls_in_case (case_price : ℚ) (roll_price : ℚ) (savings_percent : ℚ) : ℚ :=
  case_price / (roll_price * (1 - savings_percent / 100))

/-- Theorem stating that there are 12 rolls in the case under the given conditions. -/
theorem twelve_rolls_in_case :
  rolls_in_case 9 1 25 = 12 := by
  sorry

end twelve_rolls_in_case_l2466_246674


namespace great_wall_scientific_notation_l2466_246642

/-- Represents the length of the Great Wall in meters -/
def great_wall_length : ℝ := 6700010

/-- Converts a number to scientific notation with two significant figures -/
def to_scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

/-- Theorem stating that the scientific notation of the Great Wall's length
    is equal to 6.7 × 10^6 when rounded to two significant figures -/
theorem great_wall_scientific_notation :
  to_scientific_notation great_wall_length = (6.7, 6) :=
sorry

end great_wall_scientific_notation_l2466_246642


namespace opposite_of_negative_three_l2466_246689

theorem opposite_of_negative_three :
  ∀ x : ℤ, x = -3 → -x = 3 :=
by
  sorry

end opposite_of_negative_three_l2466_246689


namespace circle_area_ratio_l2466_246686

theorem circle_area_ratio (s r : ℝ) (hs : s > 0) (hr : r > 0) 
  (h : r = 0.8 * s) : 
  (π * (r / 2)^2) / (π * (s / 2)^2) = 0.64 := by
  sorry

end circle_area_ratio_l2466_246686


namespace remaining_files_correct_l2466_246620

/-- Calculates the number of remaining files on a flash drive -/
def remaining_files (music_files video_files deleted_files : ℕ) : ℕ :=
  music_files + video_files - deleted_files

/-- Theorem: The number of remaining files is correct given the initial conditions -/
theorem remaining_files_correct (music_files video_files deleted_files : ℕ) :
  remaining_files music_files video_files deleted_files =
  music_files + video_files - deleted_files :=
by sorry

end remaining_files_correct_l2466_246620


namespace difference_half_and_sixth_l2466_246656

theorem difference_half_and_sixth (x : ℝ) (hx : x = 1/2 - 1/6) : x = 1/3 := by
  sorry

end difference_half_and_sixth_l2466_246656


namespace point_coordinate_sum_l2466_246699

/-- Given a point P with coordinates (2, -1) in the standard coordinate system
    and (b-1, a+3) in another coordinate system with the same origin,
    prove that a + b = -1 -/
theorem point_coordinate_sum (a b : ℝ) 
  (h1 : b - 1 = 2) 
  (h2 : a + 3 = -1) : 
  a + b = -1 := by
  sorry

end point_coordinate_sum_l2466_246699


namespace cello_viola_pairs_l2466_246650

/-- The number of cellos in stock -/
def num_cellos : ℕ := 800

/-- The number of violas in stock -/
def num_violas : ℕ := 600

/-- The probability of randomly choosing a cello and a viola made from the same tree -/
def same_tree_prob : ℚ := 1 / 4800

/-- The number of cello-viola pairs made with wood from the same tree -/
def num_pairs : ℕ := 100

theorem cello_viola_pairs :
  num_pairs = (same_tree_prob * (num_cellos * num_violas : ℚ)).num := by
  sorry

end cello_viola_pairs_l2466_246650


namespace m_upper_bound_l2466_246608

/-- The function f(x) = a(x^2 + 1) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x^2 + 1)

theorem m_upper_bound
  (h1 : ∀ (a : ℝ), a ∈ Set.Ioo (-4) (-2))
  (h2 : ∀ (x : ℝ), x ∈ Set.Icc 1 3)
  (h3 : ∀ (m : ℝ) (a : ℝ) (x : ℝ),
    a ∈ Set.Ioo (-4) (-2) → x ∈ Set.Icc 1 3 →
    m * a - f a x > a^2 + Real.log x) :
  ∀ (m : ℝ), m ≤ -2 :=
sorry

end m_upper_bound_l2466_246608


namespace tangent_line_equation_l2466_246658

/-- The function f(x) = x^3 + 2x -/
def f (x : ℝ) : ℝ := x^3 + 2*x

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3*x^2 + 2

/-- The point of tangency -/
def point_of_tangency : ℝ × ℝ := (1, f 1)

/-- Theorem: The equation of the tangent line to y = f(x) at (1, f(1)) is 5x - y - 2 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | 5*x - y - 2 = 0} ↔ 
  y - (f 1) = (f_derivative 1) * (x - 1) :=
sorry

end tangent_line_equation_l2466_246658


namespace passengers_boarding_other_stops_eq_five_l2466_246623

/-- Calculates the number of passengers who got on the bus at other stops -/
def passengers_boarding_other_stops (initial : ℕ) (first_stop : ℕ) (getting_off : ℕ) (final : ℕ) : ℕ :=
  final - (initial + first_stop - getting_off)

/-- Theorem: Given the initial, first stop, getting off, and final passenger counts, 
    prove that 5 passengers got on at other stops -/
theorem passengers_boarding_other_stops_eq_five :
  passengers_boarding_other_stops 50 16 22 49 = 5 := by
  sorry

end passengers_boarding_other_stops_eq_five_l2466_246623


namespace triangle_c_coordinates_l2466_246687

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Defines the Euler line of a triangle -/
def Triangle.eulerLine (t : Triangle) : Line :=
  { a := 1, b := -1, c := 2 }

/-- Theorem: If triangle ABC has vertices A(2,0) and B(0,4), and its Euler line 
    is x-y+2=0, then the coordinates of C must be (-4,0) -/
theorem triangle_c_coordinates (t : Triangle) : 
  t.A = { x := 2, y := 0 } →
  t.B = { x := 0, y := 4 } →
  (t.eulerLine = { a := 1, b := -1, c := 2 }) →
  t.C = { x := -4, y := 0 } :=
by
  sorry

end triangle_c_coordinates_l2466_246687


namespace evaluate_expression_l2466_246609

theorem evaluate_expression : 500 * (500^500) * 500 = 500^502 := by sorry

end evaluate_expression_l2466_246609


namespace max_value_sum_of_sines_l2466_246602

open Real

theorem max_value_sum_of_sines :
  ∃ (x : ℝ), ∀ (y : ℝ), sin y + sin (y - π/3) ≤ sqrt 3 ∧
  sin x + sin (x - π/3) = sqrt 3 := by
  sorry

end max_value_sum_of_sines_l2466_246602


namespace min_value_implies_a_geq_two_l2466_246684

/-- The function f(x) defined as x^2 - 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

/-- Theorem: If the minimum value of f(x) in the interval [-1, 2] is f(2), then a ≥ 2 -/
theorem min_value_implies_a_geq_two (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f a x ≥ f a 2) → a ≥ 2 := by
  sorry

end min_value_implies_a_geq_two_l2466_246684


namespace remainder_1234567891011_div_210_l2466_246621

theorem remainder_1234567891011_div_210 : 1234567891011 % 210 = 31 := by
  sorry

end remainder_1234567891011_div_210_l2466_246621


namespace unique_increasing_function_l2466_246664

def f (x : ℕ) : ℤ := x^3 - 1

theorem unique_increasing_function :
  (∀ x y : ℕ, x < y → f x < f y) ∧
  f 2 = 7 ∧
  (∀ m n : ℕ, f (m * n) = f m + f n + f m * f n) ∧
  (∀ g : ℕ → ℤ, 
    (∀ x y : ℕ, x < y → g x < g y) →
    g 2 = 7 →
    (∀ m n : ℕ, g (m * n) = g m + g n + g m * g n) →
    ∀ x : ℕ, g x = f x) :=
by sorry

end unique_increasing_function_l2466_246664


namespace high_school_math_club_payment_l2466_246654

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem high_school_math_club_payment :
  ∀ B : ℕ, 
    B < 10 →
    is_divisible_by (2000 + 100 * B + 40) 15 →
    B = 7 := by
  sorry

end high_school_math_club_payment_l2466_246654


namespace correct_security_response_l2466_246651

/-- Represents an email with potentially suspicious characteristics -/
structure Email :=
  (sender : String)
  (content : String)
  (links : List String)

/-- Represents a website with potentially suspicious characteristics -/
structure Website :=
  (url : String)
  (content : String)
  (requestedInfo : List String)

/-- Represents an offer that may be unrealistic -/
structure Offer :=
  (description : String)
  (price : Nat)
  (originalPrice : Nat)

/-- Represents security measures to be followed -/
inductive SecurityMeasure
  | UseSecureNetwork
  | UseAntivirus
  | UpdateApplications
  | CheckHTTPS
  | UseComplexPasswords
  | Use2FA
  | RecognizeBankProtocols

/-- Represents the correct security response -/
structure SecurityResponse :=
  (trustSource : Bool)
  (enterInformation : Bool)
  (measures : List SecurityMeasure)

/-- Main theorem: Given suspicious conditions, prove the correct security response -/
theorem correct_security_response 
  (email : Email) 
  (website : Website) 
  (offer : Offer) : 
  (email.sender ≠ "official@aliexpress.com" ∧ 
   website.url ≠ "https://www.aliexpress.com" ∧ 
   offer.price < offer.originalPrice / 10) → 
  ∃ (response : SecurityResponse), 
    response.trustSource = false ∧ 
    response.enterInformation = false ∧ 
    response.measures.length ≥ 6 :=
by sorry

end correct_security_response_l2466_246651


namespace cos_alpha_value_l2466_246682

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (α / 2) + Real.cos (α / 2) = Real.sqrt 6 / 2) : 
  Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end cos_alpha_value_l2466_246682


namespace johnny_october_savings_l2466_246678

/-- Proves that Johnny saved $49 in October given his savings and spending information. -/
theorem johnny_october_savings :
  let september_savings : ℕ := 30
  let november_savings : ℕ := 46
  let video_game_cost : ℕ := 58
  let remaining_money : ℕ := 67
  let october_savings : ℕ := 49
  september_savings + october_savings + november_savings - video_game_cost = remaining_money :=
by
  sorry

#check johnny_october_savings

end johnny_october_savings_l2466_246678


namespace unique_integer_solution_inequality_proof_l2466_246681

-- Part 1
theorem unique_integer_solution (m : ℤ) 
  (h : ∃! (x : ℤ), |2*x - m| < 1 ∧ x = 2) : m = 4 := by
  sorry

-- Part 2
theorem inequality_proof (a b : ℝ) 
  (h1 : a * b = 4)
  (h2 : a > b)
  (h3 : b > 0) : 
  (a^2 + b^2) / (a - b) ≥ 4 * Real.sqrt 2 := by
  sorry

end unique_integer_solution_inequality_proof_l2466_246681


namespace sunflowers_per_packet_l2466_246611

theorem sunflowers_per_packet (eggplants_per_packet : ℕ) (eggplant_packets : ℕ) (sunflower_packets : ℕ) (total_plants : ℕ) :
  eggplants_per_packet = 14 →
  eggplant_packets = 4 →
  sunflower_packets = 6 →
  total_plants = 116 →
  total_plants = eggplants_per_packet * eggplant_packets + sunflower_packets * (total_plants - eggplants_per_packet * eggplant_packets) / sunflower_packets →
  (total_plants - eggplants_per_packet * eggplant_packets) / sunflower_packets = 10 :=
by sorry

end sunflowers_per_packet_l2466_246611


namespace quadratic_inequality_solution_set_l2466_246629

theorem quadratic_inequality_solution_set (a b c : ℝ) :
  a > 0 → (∀ x, a * x^2 + b * x + c > 0) ↔ b^2 - 4*a*c < 0 := by
  sorry

end quadratic_inequality_solution_set_l2466_246629


namespace determinant_solution_set_implies_a_value_l2466_246670

-- Define the determinant function
def det (x a : ℝ) : ℝ := a * x + 2

-- Define the inequality
def inequality (x a : ℝ) : Prop := det x a < 6

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | inequality x a}

-- Theorem statement
theorem determinant_solution_set_implies_a_value :
  (∀ x : ℝ, x > -1 ↔ x ∈ solution_set a) → a = -4 := by
  sorry

end determinant_solution_set_implies_a_value_l2466_246670


namespace shoe_price_ratio_l2466_246641

/-- Given a shoe with a marked price, a discount of 1/4 off, and a cost that is 2/3 of the actual selling price, 
    the ratio of the cost to the marked price is 1/2. -/
theorem shoe_price_ratio (marked_price : ℝ) (marked_price_pos : 0 < marked_price) : 
  let selling_price := (3/4) * marked_price
  let cost := (2/3) * selling_price
  cost / marked_price = 1/2 := by
sorry

end shoe_price_ratio_l2466_246641


namespace two_blue_probability_l2466_246616

def total_balls : ℕ := 15
def blue_balls : ℕ := 5
def red_balls : ℕ := 10
def drawn_balls : ℕ := 6
def target_blue : ℕ := 2

def probability_two_blue : ℚ := 2100 / 5005

theorem two_blue_probability :
  (Nat.choose blue_balls target_blue * Nat.choose red_balls (drawn_balls - target_blue)) /
  Nat.choose total_balls drawn_balls = probability_two_blue := by
  sorry

end two_blue_probability_l2466_246616


namespace smallest_b_for_real_root_l2466_246643

theorem smallest_b_for_real_root : 
  ∀ b : ℕ, (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≥ 10 :=
by sorry

end smallest_b_for_real_root_l2466_246643


namespace special_triangle_st_length_l2466_246653

/-- Triangle with given side lengths and a line segment parallel to one side passing through the incenter --/
structure SpecialTriangle where
  -- Side lengths of the triangle
  pq : ℝ
  pr : ℝ
  qr : ℝ
  -- Points S and T on sides PQ and PR respectively
  s : ℝ  -- distance PS
  t : ℝ  -- distance PT
  -- Conditions
  pq_positive : pq > 0
  pr_positive : pr > 0
  qr_positive : qr > 0
  s_on_pq : 0 < s ∧ s < pq
  t_on_pr : 0 < t ∧ t < pr
  st_parallel_qr : True  -- We can't directly express this geometric condition
  st_contains_incenter : True  -- We can't directly express this geometric condition

/-- The theorem stating that in the special triangle, ST has a specific value --/
theorem special_triangle_st_length (tri : SpecialTriangle) 
    (h_pq : tri.pq = 26) 
    (h_pr : tri.pr = 28) 
    (h_qr : tri.qr = 30) : 
  (tri.s - 0) / tri.pq + (tri.t - 0) / tri.pr = 135 / 7 := by
  sorry

end special_triangle_st_length_l2466_246653


namespace equation_D_is_quadratic_l2466_246639

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 1 = 0 -/
def equation_D (x : ℝ) : ℝ :=
  x^2 - 1

/-- Theorem: equation_D is a quadratic equation -/
theorem equation_D_is_quadratic : is_quadratic_equation equation_D :=
  sorry

end equation_D_is_quadratic_l2466_246639


namespace rectangle_area_reduction_l2466_246680

/-- Given a rectangle with initial dimensions 5 × 7 inches, if reducing one side by 2 inches
    results in an area of 21 square inches, then reducing the other side by 2 inches
    will result in an area of 25 square inches. -/
theorem rectangle_area_reduction (initial_width initial_length : ℝ)
  (h_initial_width : initial_width = 5)
  (h_initial_length : initial_length = 7)
  (h_reduced_area : (initial_width - 2) * initial_length = 21) :
  initial_width * (initial_length - 2) = 25 := by
  sorry

end rectangle_area_reduction_l2466_246680


namespace pie_eating_contest_l2466_246698

/-- The amount of pie Erik ate -/
def erik_pie : ℚ := 0.6666666666666666

/-- The amount of pie Frank ate -/
def frank_pie : ℚ := 0.3333333333333333

/-- The difference between Erik's and Frank's pie consumption -/
def pie_difference : ℚ := erik_pie - frank_pie

theorem pie_eating_contest :
  pie_difference = 0.3333333333333333 := by sorry

end pie_eating_contest_l2466_246698


namespace inequality_proof_l2466_246633

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : 1/a + 1/b + 1/c = 1) :
  a^(a*b*c) + b^(b*c*a) + c^(c*a*b) ≥ 27*b*c + 27*c*a + 27*a*b := by
  sorry

end inequality_proof_l2466_246633


namespace hyperbola_equation_l2466_246628

/-- Given a parabola and a hyperbola with specific properties, 
    prove that the standard equation of the hyperbola is x² - y²/2 = 1 -/
theorem hyperbola_equation 
  (parabola : ℝ → ℝ → Prop) 
  (hyperbola : ℝ → ℝ → ℝ → ℝ → Prop)
  (a b : ℝ)
  (A B F : ℝ × ℝ)
  (h_parabola : ∀ x y, parabola x y ↔ y^2 = 4 * Real.sqrt 3 * x)
  (h_hyperbola : ∀ x y, hyperbola a b x y ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_intersect : parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
                 hyperbola a b A.1 A.2 ∧ hyperbola a b B.1 B.2)
  (h_A_above_B : A.2 > B.2)
  (h_asymptote : ∀ x, b * x / a = Real.sqrt 2 * x)
  (h_F_focus : F = (Real.sqrt 3, 0))
  (h_equilateral : 
    (A.1 - F.1)^2 + (A.2 - F.2)^2 = 
    (B.1 - F.1)^2 + (B.2 - F.2)^2 ∧
    (A.1 - F.1)^2 + (A.2 - F.2)^2 = 
    (A.1 - B.1)^2 + (A.2 - B.2)^2) :
  ∀ x y, hyperbola a b x y ↔ x^2 - y^2 / 2 = 1 :=
by sorry

end hyperbola_equation_l2466_246628


namespace log_function_range_l2466_246640

theorem log_function_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ 2 → |Real.log x / Real.log a| > 1) ↔ 
  (a > 1/2 ∧ a < 1) ∨ (a > 1 ∧ a < 2) :=
sorry

end log_function_range_l2466_246640


namespace total_marbles_is_240_l2466_246668

/-- The number of marbles in a dozen -/
def dozen : ℕ := 12

/-- The number of red marbles Jessica has -/
def jessica_marbles : ℕ := 3 * dozen

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := 4 * jessica_marbles

/-- The number of red marbles Alex has -/
def alex_marbles : ℕ := jessica_marbles + 2 * dozen

/-- The total number of red marbles Jessica, Sandy, and Alex have -/
def total_marbles : ℕ := jessica_marbles + sandy_marbles + alex_marbles

theorem total_marbles_is_240 : total_marbles = 240 := by
  sorry

end total_marbles_is_240_l2466_246668


namespace range_of_a_l2466_246605

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) → 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * a - 1) ^ x₁ > (2 * a - 1) ^ x₂) → 
  1/2 < a ∧ a ≤ 2/3 := by
sorry

end range_of_a_l2466_246605


namespace worker_productivity_increase_l2466_246685

theorem worker_productivity_increase 
  (initial_value : ℝ) 
  (final_value : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : initial_value = 2500)
  (h2 : final_value = 2809)
  (h3 : final_value = initial_value * (1 + increase_percentage / 100)^2) :
  increase_percentage = 6 := by
sorry

end worker_productivity_increase_l2466_246685


namespace beakers_with_no_metal_ions_l2466_246673

theorem beakers_with_no_metal_ions (total_beakers : Nat) (copper_beakers : Nat) (silver_beakers : Nat)
  (drops_a_per_beaker : Nat) (drops_b_per_beaker : Nat) (total_drops_a : Nat) (total_drops_b : Nat) :
  total_beakers = 50 →
  copper_beakers = 10 →
  silver_beakers = 5 →
  drops_a_per_beaker = 3 →
  drops_b_per_beaker = 4 →
  total_drops_a = 106 →
  total_drops_b = 80 →
  total_beakers - copper_beakers - silver_beakers = 15 :=
by sorry

end beakers_with_no_metal_ions_l2466_246673


namespace certain_number_calculation_l2466_246625

theorem certain_number_calculation : ∀ (x y : ℕ),
  x + y = 36 →
  x = 19 →
  8 * x + 3 * y = 203 := by
  sorry

end certain_number_calculation_l2466_246625


namespace vectors_not_collinear_l2466_246638

/-- Given two vectors in ℝ³, construct two new vectors and prove they are not collinear -/
theorem vectors_not_collinear :
  let a : Fin 3 → ℝ := ![1, 2, -3]
  let b : Fin 3 → ℝ := ![2, -1, -1]
  let c₁ : Fin 3 → ℝ := fun i => 4 * a i + 3 * b i
  let c₂ : Fin 3 → ℝ := fun i => 8 * a i - b i
  ¬ ∃ (k : ℝ), c₁ = fun i => k * c₂ i :=
by
  sorry


end vectors_not_collinear_l2466_246638


namespace cost_price_per_meter_l2466_246632

/-- Proves that the cost price of one meter of cloth is 85 rupees -/
theorem cost_price_per_meter
  (total_length : ℕ)
  (total_selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 85)
  (h2 : total_selling_price = 8500)
  (h3 : profit_per_meter = 15) :
  (total_selling_price - total_length * profit_per_meter) / total_length = 85 := by
  sorry

end cost_price_per_meter_l2466_246632


namespace power_equality_l2466_246600

theorem power_equality (x : ℝ) (h : (10 : ℝ) ^ (2 * x) = 25) : (10 : ℝ) ^ (1 - x) = 2 := by
  sorry

end power_equality_l2466_246600


namespace min_stamps_for_35_cents_l2466_246663

/-- Represents the number of ways to make a certain amount of cents using 5-cent and 7-cent stamps -/
def stamp_combinations (cents : ℕ) : Set (ℕ × ℕ) :=
  {(x, y) | 5 * x + 7 * y = cents}

/-- The total number of stamps used in a combination -/
def total_stamps (combo : ℕ × ℕ) : ℕ :=
  combo.1 + combo.2

theorem min_stamps_for_35_cents :
  ∃ (combo : ℕ × ℕ),
    combo ∈ stamp_combinations 35 ∧
    ∀ (other : ℕ × ℕ), other ∈ stamp_combinations 35 →
      total_stamps combo ≤ total_stamps other ∧
      total_stamps combo = 5 :=
by sorry

end min_stamps_for_35_cents_l2466_246663


namespace f_bounded_g_bounded_l2466_246691

-- Define the functions f and g
def f (x : ℝ) := 3 * x - 4 * x^3
def g (x : ℝ) := 3 * x - 4 * x^3

-- Theorem for function f
theorem f_bounded : ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |f x| ≤ 1 := by
  sorry

-- Theorem for function g
theorem g_bounded : ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |g x| ≤ 1 := by
  sorry

end f_bounded_g_bounded_l2466_246691


namespace determinant_value_trig_expression_value_l2466_246652

-- Define the determinant function for 2x2 matrices
def det2 (a11 a12 a21 a22 : ℝ) : ℝ := a11 * a22 - a12 * a21

-- Problem 1
theorem determinant_value : 
  det2 (Real.cos (π/4)) 1 1 (Real.cos (π/3)) = (Real.sqrt 2 - 2) / 4 := by
  sorry

-- Problem 2
theorem trig_expression_value (a : ℝ) (h : Real.tan (π/4 + a) = -1/2) :
  (Real.sin (2*a) - 2 * (Real.cos a)^2) / (1 + Real.tan a) = 2/5 := by
  sorry

end determinant_value_trig_expression_value_l2466_246652


namespace matrix_inverse_l2466_246634

theorem matrix_inverse (x : ℝ) (h : x ≠ -12) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, x; -2, 6]
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![6 / (24 + 2*x), -x / (24 + 2*x); 2 / (24 + 2*x), 4 / (24 + 2*x)]
  A * A_inv = 1 ∧ A_inv * A = 1 :=
by sorry


end matrix_inverse_l2466_246634


namespace cupcake_frosting_l2466_246624

theorem cupcake_frosting (cagney_rate lacey_rate lacey_rest total_time : ℕ) :
  cagney_rate = 15 →
  lacey_rate = 25 →
  lacey_rest = 10 →
  total_time = 480 →
  (total_time : ℚ) / ((1 : ℚ) / cagney_rate + (1 : ℚ) / (lacey_rate + lacey_rest)) = 45 :=
by sorry

end cupcake_frosting_l2466_246624


namespace inequality_proof_l2466_246694

theorem inequality_proof (x y z : ℝ) (h : x * y * z + x + y + z = 4) :
  (y * z + 6)^2 + (z * x + 6)^2 + (x * y + 6)^2 ≥ 8 * (x * y * z + 5) := by
  sorry

end inequality_proof_l2466_246694


namespace larger_number_proof_l2466_246692

/-- Given two positive integers with the specified HCF and LCM factors, 
    prove that the larger of the two numbers is 3289 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf_condition : Nat.gcd a b = 23)
  (lcm_condition : ∃ k : ℕ+, Nat.lcm a b = 23 * 11 * 13 * 15^2 * k) :
  max a b = 3289 := by
  sorry

end larger_number_proof_l2466_246692


namespace max_roses_325_l2466_246669

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual_price : ℚ
  dozen_price : ℚ
  two_dozen_price : ℚ

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def max_roses (budget : ℚ) (pricing : RosePricing) : ℕ :=
  sorry

/-- The theorem stating that given the specific pricing and budget, 325 roses is the maximum that can be purchased -/
theorem max_roses_325 :
  let pricing : RosePricing := {
    individual_price := 23/10,
    dozen_price := 36,
    two_dozen_price := 50
  }
  max_roses 680 pricing = 325 := by sorry

end max_roses_325_l2466_246669


namespace triangle_inequality_l2466_246693

/-- For any triangle with sides a, b, c, semi-perimeter p, inradius r, and area S,
    where S = √(p(p-a)(p-b)(p-c)) and r = S/p, the following inequality holds:
    1/(p-a)² + 1/(p-b)² + 1/(p-c)² ≥ 1/r² -/
theorem triangle_inequality (a b c p r S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : p = (a + b + c) / 2)
  (h5 : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h6 : r = S / p) :
  1 / (p - a)^2 + 1 / (p - b)^2 + 1 / (p - c)^2 ≥ 1 / r^2 := by
  sorry

end triangle_inequality_l2466_246693


namespace exponential_inequality_l2466_246661

theorem exponential_inequality (x y a : ℝ) (hx : x > y) (hy : y > 1) (ha : 0 < a) (ha1 : a < 1) :
  a^x < a^y := by sorry

end exponential_inequality_l2466_246661
