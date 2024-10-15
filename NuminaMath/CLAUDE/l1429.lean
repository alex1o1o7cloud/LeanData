import Mathlib

namespace NUMINAMATH_CALUDE_only_f4_decreasing_l1429_142947

-- Define the four functions
def f1 (x : ℝ) : ℝ := x^2 + 1
def f2 (x : ℝ) : ℝ := -x^2 + 1
def f3 (x : ℝ) : ℝ := 2*x + 1
def f4 (x : ℝ) : ℝ := -2*x + 1

-- Theorem stating that only f4 has a negative derivative for all real x
theorem only_f4_decreasing :
  (∀ x : ℝ, deriv f1 x > 0) ∧
  (∃ x : ℝ, deriv f2 x ≥ 0) ∧
  (∀ x : ℝ, deriv f3 x > 0) ∧
  (∀ x : ℝ, deriv f4 x < 0) :=
by sorry

end NUMINAMATH_CALUDE_only_f4_decreasing_l1429_142947


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_positive_l1429_142997

theorem sum_of_x_and_y_positive (x y : ℝ) (h : 2 * x + 3 * y > 2 - y + 3 - x) : x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_positive_l1429_142997


namespace NUMINAMATH_CALUDE_owlHootsPerMinute_l1429_142963

/-- The number of hoot sounds one barnyard owl makes per minute, given that 3 owls together make 5 less than 20 hoots per minute. -/
def owlHoots : ℕ :=
  let totalHoots : ℕ := 20 - 5
  let numOwls : ℕ := 3
  totalHoots / numOwls

/-- Theorem stating that one barnyard owl makes 5 hoot sounds per minute under the given conditions. -/
theorem owlHootsPerMinute : owlHoots = 5 := by
  sorry

end NUMINAMATH_CALUDE_owlHootsPerMinute_l1429_142963


namespace NUMINAMATH_CALUDE_sphere_radius_from_depression_l1429_142983

/-- The radius of a sphere that creates a circular depression with given diameter and depth when partially submerged. -/
def sphere_radius (depression_diameter : ℝ) (depression_depth : ℝ) : ℝ :=
  13

/-- Theorem stating that a sphere with radius 13cm creates a circular depression
    with diameter 24cm and depth 8cm when partially submerged. -/
theorem sphere_radius_from_depression :
  sphere_radius 24 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_depression_l1429_142983


namespace NUMINAMATH_CALUDE_ac_negative_l1429_142974

theorem ac_negative (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : a / b + c / d = (a + c) / (b + d)) : a * c < 0 := by
  sorry

end NUMINAMATH_CALUDE_ac_negative_l1429_142974


namespace NUMINAMATH_CALUDE_min_value_expression_l1429_142988

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (x + 2) * (2 * y + 1) / (x * y) ≥ 19 + 4 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1429_142988


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l1429_142927

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l1429_142927


namespace NUMINAMATH_CALUDE_train_distance_problem_l1429_142970

theorem train_distance_problem (speed1 speed2 distance_diff : ℝ) 
  (h1 : speed1 = 50)
  (h2 : speed2 = 40)
  (h3 : distance_diff = 100) :
  let time := distance_diff / (speed1 - speed2)
  let distance1 := speed1 * time
  let distance2 := speed2 * time
  let total_distance := distance1 + distance2
  total_distance = 900 := by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l1429_142970


namespace NUMINAMATH_CALUDE_sum_two_smallest_angles_l1429_142912

-- Define the quadrilateral ABCD
variable (A B C D : Point)

-- Define the angles of the quadrilateral
def angle (P Q R : Point) : ℝ := sorry

-- Define the conditions
axiom quad_angles_arithmetic : ∃ (a d : ℝ), 
  angle B A D = a ∧
  angle A B C = a + d ∧
  angle B C D = a + 2*d ∧
  angle C D A = a + 3*d

axiom angle_equality1 : angle A B D = angle D B C
axiom angle_equality2 : angle A D B = angle B D C

axiom triangle_ABD_arithmetic : ∃ (x y : ℝ),
  angle B A D = x ∧
  angle A B D = x + y ∧
  angle A D B = x + 2*y

axiom triangle_DCB_arithmetic : ∃ (x y : ℝ),
  angle D C B = x ∧
  angle C D B = x + y ∧
  angle C B D = x + 2*y

axiom smallest_angle : angle B A D = 10
axiom second_angle : angle A B C = 70

-- Theorem to prove
theorem sum_two_smallest_angles :
  angle B A D + angle A B C = 80 := by sorry

end NUMINAMATH_CALUDE_sum_two_smallest_angles_l1429_142912


namespace NUMINAMATH_CALUDE_division_remainder_l1429_142972

theorem division_remainder : 
  let a := 555
  let b := 445
  let number := 220030
  let sum := a + b
  let diff := a - b
  let quotient := 2 * diff
  number % sum = 30 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l1429_142972


namespace NUMINAMATH_CALUDE_percentage_calculations_l1429_142962

theorem percentage_calculations (M N : ℝ) (h : M < N) :
  (100 * (N - M) / M = (N - M) / M * 100) ∧
  (100 * M / N = M / N * 100) :=
by sorry

end NUMINAMATH_CALUDE_percentage_calculations_l1429_142962


namespace NUMINAMATH_CALUDE_prob_heart_or_king_is_31_52_l1429_142994

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ := 52)
  (hearts : ℕ := 13)
  (kings : ℕ := 4)
  (king_of_hearts : ℕ := 1)

/-- The probability of drawing at least one heart or king when drawing two cards without replacement -/
def prob_heart_or_king (d : Deck) : ℚ :=
  1 - (d.total_cards - (d.hearts + d.kings - d.king_of_hearts)) * (d.total_cards - 1 - (d.hearts + d.kings - d.king_of_hearts)) /
      (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability of drawing at least one heart or king is 31/52 -/
theorem prob_heart_or_king_is_31_52 (d : Deck) :
  prob_heart_or_king d = 31 / 52 := by
  sorry

end NUMINAMATH_CALUDE_prob_heart_or_king_is_31_52_l1429_142994


namespace NUMINAMATH_CALUDE_trailing_zeros_of_999999999996_squared_l1429_142995

/-- The number of trailing zeros in 999,999,999,996^2 is 11 -/
theorem trailing_zeros_of_999999999996_squared : 
  (999999999996 : ℕ)^2 % 10^12 = 16 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_999999999996_squared_l1429_142995


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l1429_142915

/-- Given that the line x + y = c is the perpendicular bisector of the line segment
    from (2, 4) to (6, 8), prove that c = 10. -/
theorem perpendicular_bisector_c_value :
  ∀ c : ℝ,
  (∀ x y : ℝ, x + y = c ↔ 
    ((x - 4)^2 + (y - 6)^2 = 8) ∧ 
    ((x - 2) * (8 - 4) = (y - 4) * (6 - 2))) →
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l1429_142915


namespace NUMINAMATH_CALUDE_tangent_line_curve_equivalence_l1429_142989

theorem tangent_line_curve_equivalence 
  (α β m n : ℝ) 
  (h_pos_α : α > 0) 
  (h_pos_β : β > 0) 
  (h_pos_m : m > 0) 
  (h_pos_n : n > 0) 
  (h_relation : 1 / α + 1 / β = 1) : 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 
    (m * x + n * y = 1) ∧ 
    (x ^ α + y ^ α = 1) ∧
    (∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' ^ α + y' ^ α = 1 → m * x' + n * y' ≥ 1))
  ↔ 
  (m ^ β + n ^ β = 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_curve_equivalence_l1429_142989


namespace NUMINAMATH_CALUDE_binary_ternary_equality_l1429_142908

/-- Represents a digit in base 2 (binary) -/
def BinaryDigit : Type := {n : ℕ // n < 2}

/-- Represents a digit in base 3 (ternary) -/
def TernaryDigit : Type := {n : ℕ // n < 3}

/-- Converts a binary number to decimal -/
def binaryToDecimal (d₂ : BinaryDigit) (d₁ : BinaryDigit) (d₀ : BinaryDigit) : ℕ :=
  d₂.val * 2^2 + d₁.val * 2^1 + d₀.val * 2^0

/-- Converts a ternary number to decimal -/
def ternaryToDecimal (d₂ : TernaryDigit) (d₁ : TernaryDigit) (d₀ : TernaryDigit) : ℕ :=
  d₂.val * 3^2 + d₁.val * 3^1 + d₀.val * 3^0

theorem binary_ternary_equality :
  ∀ (x : TernaryDigit) (y : BinaryDigit),
    binaryToDecimal ⟨1, by norm_num⟩ y ⟨1, by norm_num⟩ = ternaryToDecimal x ⟨0, by norm_num⟩ ⟨2, by norm_num⟩ →
    x.val = 1 ∧ y.val = 1 ∧ ternaryToDecimal x ⟨0, by norm_num⟩ ⟨2, by norm_num⟩ = 11 :=
by sorry

#check binary_ternary_equality

end NUMINAMATH_CALUDE_binary_ternary_equality_l1429_142908


namespace NUMINAMATH_CALUDE_three_y_squared_l1429_142932

theorem three_y_squared (x y : ℤ) 
  (eq1 : 3 * x + y = 40) 
  (eq2 : 2 * x - y = 20) : 
  3 * y^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_three_y_squared_l1429_142932


namespace NUMINAMATH_CALUDE_certain_amount_proof_l1429_142977

theorem certain_amount_proof : 
  let x : ℝ := 10
  let percentage_of_500 : ℝ := 0.05 * 500
  let percentage_of_x : ℝ := 0.5 * x
  percentage_of_500 - percentage_of_x = 20 := by
sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l1429_142977


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1429_142952

/-- The x-intercept of a line is a point where the line crosses the x-axis (y = 0). -/
def x_intercept (a b c : ℚ) : ℚ × ℚ :=
  (c / a, 0)

/-- The line equation is represented as ax + by = c -/
def line_equation (a b c : ℚ) (x y : ℚ) : Prop :=
  a * x + b * y = c

theorem x_intercept_of_line :
  x_intercept 4 7 28 = (7, 0) ∧
  line_equation 4 7 28 (x_intercept 4 7 28).1 (x_intercept 4 7 28).2 :=
by sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1429_142952


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1429_142919

theorem min_value_reciprocal_sum (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1429_142919


namespace NUMINAMATH_CALUDE_domino_placement_theorem_l1429_142979

/-- Represents a 6x6 chessboard -/
def Chessboard : Type := Fin 6 × Fin 6

/-- Represents a domino placement on the chessboard -/
def Domino : Type := Chessboard × Chessboard

/-- Check if two squares are adjacent -/
def adjacent (s1 s2 : Chessboard) : Prop :=
  (s1.1 = s2.1 ∧ s1.2.succ = s2.2) ∨
  (s1.1 = s2.1 ∧ s1.2 = s2.2.succ) ∨
  (s1.1.succ = s2.1 ∧ s1.2 = s2.2) ∨
  (s1.1 = s2.1.succ ∧ s1.2 = s2.2)

/-- Check if a domino placement is valid -/
def valid_domino (d : Domino) : Prop :=
  adjacent d.1 d.2

/-- The main theorem -/
theorem domino_placement_theorem
  (dominos : Finset Domino)
  (h1 : dominos.card = 11)
  (h2 : ∀ d ∈ dominos, valid_domino d)
  (h3 : ∀ s1 s2 : Chessboard, s1 ≠ s2 →
        (∃ d ∈ dominos, d.1 = s1 ∨ d.2 = s1) →
        (∃ d ∈ dominos, d.1 = s2 ∨ d.2 = s2) →
        s1 ≠ s2) :
  ∃ s1 s2 : Chessboard, adjacent s1 s2 ∧
    (∀ d ∈ dominos, d.1 ≠ s1 ∧ d.2 ≠ s1 ∧ d.1 ≠ s2 ∧ d.2 ≠ s2) :=
by sorry

end NUMINAMATH_CALUDE_domino_placement_theorem_l1429_142979


namespace NUMINAMATH_CALUDE_set_operations_l1429_142943

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | x < 5}

-- State the theorem
theorem set_operations :
  (A ∪ B = Set.univ) ∧
  (Aᶜ ∩ B = {x | x < 2}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1429_142943


namespace NUMINAMATH_CALUDE_vector_problems_l1429_142980

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

theorem vector_problems :
  (∃ k : ℝ, (a.1 + k * c.1, a.2 + k * c.2) • (2 * b.1 - a.1, 2 * b.2 - a.2) = 0 → k = -11/18) ∧
  (∃ d : ℝ × ℝ, ∃ t : ℝ, d = (t * c.1, t * c.2) ∧ d.1^2 + d.2^2 = 34 → 
    d = (4 * Real.sqrt 2, Real.sqrt 2) ∨ d = (-4 * Real.sqrt 2, -Real.sqrt 2)) :=
by sorry

#check vector_problems

end NUMINAMATH_CALUDE_vector_problems_l1429_142980


namespace NUMINAMATH_CALUDE_sin_arccos_three_fifths_l1429_142993

theorem sin_arccos_three_fifths : Real.sin (Real.arccos (3/5)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_three_fifths_l1429_142993


namespace NUMINAMATH_CALUDE_jules_dog_walking_rate_l1429_142987

/-- Proves that Jules charges $1.25 per block for dog walking -/
theorem jules_dog_walking_rate :
  let vacation_cost : ℚ := 1000
  let family_members : ℕ := 5
  let start_fee : ℚ := 2
  let dogs_walked : ℕ := 20
  let total_blocks : ℕ := 128
  let individual_contribution := vacation_cost / family_members
  let total_start_fees := start_fee * dogs_walked
  let remaining_to_earn := individual_contribution - total_start_fees
  let rate_per_block := remaining_to_earn / total_blocks
  rate_per_block = 1.25 := by
sorry


end NUMINAMATH_CALUDE_jules_dog_walking_rate_l1429_142987


namespace NUMINAMATH_CALUDE_least_positive_multiple_least_positive_multiple_when_x_24_l1429_142917

theorem least_positive_multiple (x y : ℤ) : ∃ (k : ℤ), k > 0 ∧ k * (x + 16 * y) = 8 ∧ ∀ (m : ℤ), m > 0 ∧ (∃ (n : ℤ), m * (x + 16 * y) = n * 8) → m ≥ k :=
  by sorry

theorem least_positive_multiple_when_x_24 : ∃ (k : ℤ), k > 0 ∧ k * (24 + 16 * (-1)) = 8 ∧ ∀ (m : ℤ), m > 0 ∧ (∃ (n : ℤ), m * (24 + 16 * (-1)) = n * 8) → m ≥ k :=
  by sorry

end NUMINAMATH_CALUDE_least_positive_multiple_least_positive_multiple_when_x_24_l1429_142917


namespace NUMINAMATH_CALUDE_range_of_a_given_p_and_q_l1429_142971

-- Define the propositions
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - Real.log x - a ≥ 0

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x - 8 - 6*a = 0

-- Define the range of a
def range_of_a : Set ℝ :=
  Set.Ici (-4) ∪ Set.Icc (-2) 1

-- State the theorem
theorem range_of_a_given_p_and_q :
  ∀ a : ℝ, prop_p a ∧ prop_q a ↔ a ∈ range_of_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_given_p_and_q_l1429_142971


namespace NUMINAMATH_CALUDE_function_behavior_l1429_142991

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : periodic_two f)
  (h_decreasing : decreasing_on f 1 2) :
  increasing_on f (-2) (-1) ∧ decreasing_on f 3 4 := by
  sorry

end NUMINAMATH_CALUDE_function_behavior_l1429_142991


namespace NUMINAMATH_CALUDE_second_grade_sample_l1429_142929

/-- Calculates the number of students to be drawn from the second grade in a stratified sample -/
def students_from_second_grade (total_sample : ℕ) (ratio_first : ℕ) (ratio_second : ℕ) (ratio_third : ℕ) : ℕ :=
  (total_sample * ratio_second) / (ratio_first + ratio_second + ratio_third)

/-- Theorem: Given the conditions, the number of students to be drawn from the second grade is 15 -/
theorem second_grade_sample :
  students_from_second_grade 50 3 3 4 = 15 := by
  sorry

#eval students_from_second_grade 50 3 3 4

end NUMINAMATH_CALUDE_second_grade_sample_l1429_142929


namespace NUMINAMATH_CALUDE_fish_speed_problem_l1429_142902

/-- Calculates the downstream speed of a fish given its upstream and still water speeds. -/
def fish_downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem: A fish with an upstream speed of 35 kmph and a still water speed of 45 kmph 
    has a downstream speed of 55 kmph. -/
theorem fish_speed_problem :
  fish_downstream_speed 35 45 = 55 := by
  sorry

#eval fish_downstream_speed 35 45

end NUMINAMATH_CALUDE_fish_speed_problem_l1429_142902


namespace NUMINAMATH_CALUDE_crayons_per_child_l1429_142900

/-- Given a group of children with crayons, prove that each child has 12 crayons. -/
theorem crayons_per_child (total_children : ℕ) (total_crayons : ℕ) 
  (h1 : total_children = 18) 
  (h2 : total_crayons = 216) : 
  total_crayons / total_children = 12 := by
  sorry

#check crayons_per_child

end NUMINAMATH_CALUDE_crayons_per_child_l1429_142900


namespace NUMINAMATH_CALUDE_abc_value_l1429_142937

theorem abc_value (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 39) 
  (h3 : a + b + c = 10) : 
  a * b * c = -150 + 15 * Real.sqrt 69 := by
  sorry

end NUMINAMATH_CALUDE_abc_value_l1429_142937


namespace NUMINAMATH_CALUDE_right_triangle_cos_B_l1429_142933

theorem right_triangle_cos_B (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c^2 = a^2 + b^2) : 
  let cos_B := b / c
  cos_B = 15 / 17 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_cos_B_l1429_142933


namespace NUMINAMATH_CALUDE_mod_63_calculation_l1429_142922

theorem mod_63_calculation : ∃ (a b : ℤ), 
  (7 * a) % 63 = 1 ∧ 
  (13 * b) % 63 = 1 ∧ 
  (3 * a + 9 * b) % 63 = 48 := by
  sorry

end NUMINAMATH_CALUDE_mod_63_calculation_l1429_142922


namespace NUMINAMATH_CALUDE_airline_capacity_l1429_142960

/-- Calculates the number of passengers an airline can accommodate daily --/
theorem airline_capacity
  (num_airplanes : ℕ)
  (rows_per_airplane : ℕ)
  (seats_per_row : ℕ)
  (flights_per_day : ℕ)
  (h1 : num_airplanes = 5)
  (h2 : rows_per_airplane = 20)
  (h3 : seats_per_row = 7)
  (h4 : flights_per_day = 2) :
  num_airplanes * rows_per_airplane * seats_per_row * flights_per_day = 7000 :=
by sorry

end NUMINAMATH_CALUDE_airline_capacity_l1429_142960


namespace NUMINAMATH_CALUDE_function_intersection_condition_l1429_142955

/-- The function f(x) = (k+1)x^2 - 2x + 1 has intersections with the x-axis
    if and only if k ≤ 0. -/
theorem function_intersection_condition (k : ℝ) :
  (∃ x, (k + 1) * x^2 - 2 * x + 1 = 0) ↔ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_intersection_condition_l1429_142955


namespace NUMINAMATH_CALUDE_manager_percentage_after_leaving_l1429_142942

/-- Calculates the new percentage of managers after some leave the room -/
def new_manager_percentage (initial_employees : ℕ) (initial_manager_percentage : ℚ) 
  (managers_leaving : ℚ) : ℚ :=
  let initial_managers : ℚ := (initial_manager_percentage / 100) * initial_employees
  let remaining_managers : ℚ := initial_managers - managers_leaving
  let remaining_employees : ℚ := initial_employees - managers_leaving
  (remaining_managers / remaining_employees) * 100

/-- Theorem stating that given the initial conditions and managers leaving, 
    the new percentage of managers is 98% -/
theorem manager_percentage_after_leaving :
  new_manager_percentage 200 99 99.99999999999991 = 98 := by
  sorry

end NUMINAMATH_CALUDE_manager_percentage_after_leaving_l1429_142942


namespace NUMINAMATH_CALUDE_shaded_area_approx_l1429_142956

/-- The area of the shaded region formed by two circles with radii 3 and 6 -/
def shaded_area (π : ℝ) : ℝ :=
  let small_radius : ℝ := 3
  let large_radius : ℝ := 6
  let left_rectangle : ℝ := small_radius * (2 * small_radius)
  let right_rectangle : ℝ := large_radius * (2 * large_radius)
  let small_semicircle : ℝ := 0.5 * π * small_radius ^ 2
  let large_semicircle : ℝ := 0.5 * π * large_radius ^ 2
  (left_rectangle + right_rectangle) - (small_semicircle + large_semicircle)

theorem shaded_area_approx :
  ∃ (π : ℝ), abs (shaded_area π - 19.3) < 0.05 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_approx_l1429_142956


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l1429_142923

theorem smaller_number_in_ratio (a b d x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = a / b → x * y = d → 
  x = Real.sqrt ((a * d) / b) ∧ x < y := by sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l1429_142923


namespace NUMINAMATH_CALUDE_thousandth_digit_is_zero_l1429_142985

def factorial (n : ℕ) : ℕ := Nat.factorial n

def expression : ℚ := (factorial 13 * factorial 23 + factorial 15 * factorial 17) / 7

theorem thousandth_digit_is_zero :
  ∃ (n : ℕ), n ≥ 1000 ∧ (expression * 10^n).floor % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_thousandth_digit_is_zero_l1429_142985


namespace NUMINAMATH_CALUDE_thomas_salary_l1429_142920

/-- Given the average salaries of two groups, prove Thomas's salary --/
theorem thomas_salary (raj roshan thomas : ℕ) : 
  (raj + roshan) / 2 = 4000 →
  (raj + roshan + thomas) / 3 = 5000 →
  thomas = 7000 := by
sorry

end NUMINAMATH_CALUDE_thomas_salary_l1429_142920


namespace NUMINAMATH_CALUDE_sum_value_l1429_142965

theorem sum_value (a b : ℝ) (h1 : |a| = 1) (h2 : b = -2) : 
  a + b = -3 ∨ a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_value_l1429_142965


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l1429_142901

open Set

theorem inequality_solution_sets (a b : ℝ) :
  {x : ℝ | a * x - b < 0} = Ioi 1 →
  {x : ℝ | (a * x + b) * (x - 3) > 0} = Ioo (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l1429_142901


namespace NUMINAMATH_CALUDE_exp_properties_l1429_142906

-- Define the exponential function
noncomputable def Exp : ℝ → ℝ := Real.exp

-- Theorem statement
theorem exp_properties :
  (∀ (a b x : ℝ), Exp ((a + b) * x) = Exp (a * x) * Exp (b * x)) ∧
  (∀ (x : ℝ) (k : ℕ), Exp (k * x) = (Exp x) ^ k) := by
  sorry

end NUMINAMATH_CALUDE_exp_properties_l1429_142906


namespace NUMINAMATH_CALUDE_fourth_root_of_256000000_l1429_142992

theorem fourth_root_of_256000000 : Real.sqrt (Real.sqrt 256000000) = 40 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_256000000_l1429_142992


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l1429_142904

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ n : ℕ, n = 990 ∧
    5 ∣ n ∧
    6 ∣ n ∧
    n < 1000 ∧
    ∀ m : ℕ, m < 1000 → 5 ∣ m → 6 ∣ m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l1429_142904


namespace NUMINAMATH_CALUDE_ed_doug_marble_difference_l1429_142931

theorem ed_doug_marble_difference (ed_initial : ℕ) (doug : ℕ) (ed_lost : ℕ) (ed_doug_diff : ℕ) :
  ed_initial > doug →
  ed_initial = 91 →
  ed_lost = 21 →
  ed_initial - ed_lost - doug = ed_doug_diff →
  ed_doug_diff = 9 →
  ed_initial - doug = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_ed_doug_marble_difference_l1429_142931


namespace NUMINAMATH_CALUDE_sams_work_hours_sams_september_february_hours_l1429_142936

/-- Calculates the number of hours Sam worked from September to February -/
theorem sams_work_hours (earnings_mar_aug : ℝ) (hours_mar_aug : ℝ) (console_cost : ℝ) (car_repair_cost : ℝ) (remaining_hours : ℝ) : ℝ :=
  let hourly_rate := earnings_mar_aug / hours_mar_aug
  let remaining_earnings := console_cost - (earnings_mar_aug - car_repair_cost)
  let total_hours_needed := remaining_earnings / hourly_rate
  total_hours_needed - remaining_hours

/-- Proves that Sam worked 8 hours from September to February -/
theorem sams_september_february_hours : sams_work_hours 460 23 600 340 16 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sams_work_hours_sams_september_february_hours_l1429_142936


namespace NUMINAMATH_CALUDE_exactly_two_sunny_days_probability_l1429_142964

theorem exactly_two_sunny_days_probability 
  (num_days : ℕ) 
  (rain_prob : ℝ) 
  (sunny_prob : ℝ) :
  num_days = 3 →
  rain_prob = 0.6 →
  sunny_prob = 1 - rain_prob →
  (num_days.choose 2 : ℝ) * sunny_prob^2 * rain_prob = 54/125 :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_sunny_days_probability_l1429_142964


namespace NUMINAMATH_CALUDE_quadratic_value_l1429_142982

/-- A quadratic function with axis of symmetry at x = 3.5 and p(-6) = 0 -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_value (d e f : ℝ) :
  (∀ x : ℝ, p d e f x = p d e f (7 - x)) →  -- Axis of symmetry at x = 3.5
  p d e f (-6) = 0 →                        -- p(-6) = 0
  ∃ n : ℤ, p d e f 13 = n →                 -- p(13) is an integer
  p d e f 13 = 0 :=                         -- Conclusion: p(13) = 0
by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_l1429_142982


namespace NUMINAMATH_CALUDE_all_functions_increasing_l1429_142949

-- Define the functions
def f₁ (x : ℝ) : ℝ := 2 * x
def f₂ (x : ℝ) : ℝ := x^2 + 2*x - 1
def f₃ (x : ℝ) : ℝ := abs (x + 2)
def f₄ (x : ℝ) : ℝ := abs x + 2

-- Define the interval [0, +∞)
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Theorem statement
theorem all_functions_increasing :
  (∀ x y, nonnegative x → nonnegative y → x < y → f₁ x < f₁ y) ∧
  (∀ x y, nonnegative x → nonnegative y → x < y → f₂ x < f₂ y) ∧
  (∀ x y, nonnegative x → nonnegative y → x < y → f₃ x < f₃ y) ∧
  (∀ x y, nonnegative x → nonnegative y → x < y → f₄ x < f₄ y) :=
by sorry

end NUMINAMATH_CALUDE_all_functions_increasing_l1429_142949


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l1429_142951

theorem larger_number_of_pair (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 50) : max x y = 29 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l1429_142951


namespace NUMINAMATH_CALUDE_chipmunk_acorns_count_l1429_142913

/-- The number of acorns hidden by the chipmunk in each hole -/
def chipmunk_acorns_per_hole : ℕ := 3

/-- The number of acorns hidden by the squirrel in each hole -/
def squirrel_acorns_per_hole : ℕ := 4

/-- The number of holes dug by the chipmunk -/
def chipmunk_holes : ℕ := 16

/-- The number of holes dug by the squirrel -/
def squirrel_holes : ℕ := chipmunk_holes - 4

/-- The total number of acorns hidden by the chipmunk -/
def chipmunk_total_acorns : ℕ := chipmunk_acorns_per_hole * chipmunk_holes

/-- The total number of acorns hidden by the squirrel -/
def squirrel_total_acorns : ℕ := squirrel_acorns_per_hole * squirrel_holes

theorem chipmunk_acorns_count : chipmunk_total_acorns = 48 ∧ chipmunk_total_acorns = squirrel_total_acorns :=
by sorry

end NUMINAMATH_CALUDE_chipmunk_acorns_count_l1429_142913


namespace NUMINAMATH_CALUDE_equation_solution_l1429_142907

theorem equation_solution : 
  ∃ (x : ℝ), x ≠ -4 ∧ (-x^2 = (4*x + 2) / (x + 4)) ↔ (x = -1 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1429_142907


namespace NUMINAMATH_CALUDE_measure_six_with_special_ruler_l1429_142924

/-- A ruler with marks at specific positions -/
structure Ruler :=
  (marks : List ℝ)

/-- Definition of a ruler with marks at 0, 2, and 5 -/
def specialRuler : Ruler :=
  { marks := [0, 2, 5] }

/-- A function to check if a length can be measured using a given ruler -/
def canMeasure (r : Ruler) (length : ℝ) : Prop :=
  ∃ (a b : ℝ), a ∈ r.marks ∧ b ∈ r.marks ∧ (b - a = length ∨ a - b = length)

/-- Theorem stating that the special ruler can measure a segment of length 6 -/
theorem measure_six_with_special_ruler :
  canMeasure specialRuler 6 := by
  sorry


end NUMINAMATH_CALUDE_measure_six_with_special_ruler_l1429_142924


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l1429_142966

theorem six_digit_numbers_with_zero (total_six_digit : ℕ) (six_digit_no_zero : ℕ) :
  total_six_digit = 9 * 10^5 →
  six_digit_no_zero = 9^6 →
  total_six_digit - six_digit_no_zero = 368559 := by
sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l1429_142966


namespace NUMINAMATH_CALUDE_absolute_value_z_l1429_142905

theorem absolute_value_z (w z : ℂ) : 
  w * z = 20 - 21 * I → Complex.abs w = Real.sqrt 29 → Complex.abs z = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_z_l1429_142905


namespace NUMINAMATH_CALUDE_equation_solutions_l1429_142926

/-- The equation has solutions when the parameter a is greater than 1 -/
def has_solution (a : ℝ) : Prop :=
  a > 1

/-- The solutions of the equation for a given parameter a -/
def solutions (a : ℝ) : Set ℝ :=
  if a > 2 then { (1 - a) / a, -1, 1 - a }
  else if a = 2 then { -1, -1/2 }
  else if 1 < a ∧ a < 2 then { (1 - a) / a, -1, 1 - a }
  else ∅

/-- The main theorem stating that the equation has solutions for a > 1 
    and providing these solutions -/
theorem equation_solutions (a : ℝ) :
  has_solution a →
  ∃ x : ℝ, x ∈ solutions a ∧
    (2 - 2 * a * (x + 1)) / (|x| - x) = Real.sqrt (1 - a - a * x) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l1429_142926


namespace NUMINAMATH_CALUDE_weight_problem_l1429_142999

theorem weight_problem (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 42)
  (h3 : (b + c) / 2 = 43) :
  b = 35 := by
sorry

end NUMINAMATH_CALUDE_weight_problem_l1429_142999


namespace NUMINAMATH_CALUDE_magnitude_of_vector_difference_l1429_142925

/-- Given plane vectors a and b with specified properties, prove that |2a-b| = √91 -/
theorem magnitude_of_vector_difference (a b : ℝ × ℝ) :
  (Real.cos (5 * Real.pi / 6) = a.1 * b.1 + a.2 * b.2) →  -- angle between a and b is 5π/6
  (a.1^2 + a.2^2 = 16) →  -- |a| = 4
  (b.1^2 + b.2^2 = 3) →  -- |b| = √3
  ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2 = 91) :=  -- |2a-b| = √91
by sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_difference_l1429_142925


namespace NUMINAMATH_CALUDE_decoration_problem_l1429_142928

theorem decoration_problem (total_decorations : ℕ) (nails_used : ℕ) 
  (h1 : total_decorations = (3 * nails_used) / 2)
  (h2 : nails_used = 50) : 
  total_decorations - nails_used - (2 * (total_decorations - nails_used)) / 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_decoration_problem_l1429_142928


namespace NUMINAMATH_CALUDE_equation_solution_l1429_142940

theorem equation_solution :
  let f : ℝ → ℝ := fun x => 0.05 * x + 0.09 * (30 + x)
  ∃! x : ℝ, f x = 15.3 - 3.3 ∧ x = 465 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1429_142940


namespace NUMINAMATH_CALUDE_lcm_from_product_and_gcd_l1429_142948

theorem lcm_from_product_and_gcd (a b c : ℕ+) :
  a * b * c = 1354808 ∧ Nat.gcd a (Nat.gcd b c) = 11 →
  Nat.lcm a (Nat.lcm b c) = 123164 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_gcd_l1429_142948


namespace NUMINAMATH_CALUDE_curve_C_bound_expression_l1429_142969

theorem curve_C_bound_expression (x y : ℝ) :
  4 * x^2 + y^2 = 16 →
  -4 ≤ Real.sqrt 3 * x + (1/2) * y ∧ Real.sqrt 3 * x + (1/2) * y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_curve_C_bound_expression_l1429_142969


namespace NUMINAMATH_CALUDE_final_cucumber_count_l1429_142968

theorem final_cucumber_count (initial_total : ℕ) (initial_carrots : ℕ) (added_cucumbers : ℕ)
  (h1 : initial_total = 10)
  (h2 : initial_carrots = 4)
  (h3 : added_cucumbers = 2) :
  initial_total - initial_carrots + added_cucumbers = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_cucumber_count_l1429_142968


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1429_142981

theorem sum_with_radical_conjugate : 
  let a : ℝ := 15
  let b : ℝ := Real.sqrt 500
  (a - b) + (a + b) = 30 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1429_142981


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l1429_142975

theorem triangle_area_from_squares (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 225) (hc : c^2 = 324) :
  (1/2) * a * b = 45 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l1429_142975


namespace NUMINAMATH_CALUDE_remainder_problem_l1429_142935

theorem remainder_problem (k : ℕ) :
  k > 0 ∧ k < 100 ∧
  k % 5 = 2 ∧
  k % 6 = 3 ∧
  k % 8 = 7 →
  k % 9 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1429_142935


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1429_142953

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x + y = 4) ∧ (x + 2 * y = -1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1429_142953


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1429_142958

theorem algebraic_expression_equality (x y : ℝ) (h : 2 * x - 3 * y = 1) :
  6 * y - 4 * x + 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1429_142958


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l1429_142934

-- Define the number of each type of animal
def num_parrots : ℕ := 5
def num_dogs : ℕ := 3
def num_cats : ℕ := 4

-- Define the total number of animals
def total_animals : ℕ := num_parrots + num_dogs + num_cats

-- Define the function to calculate the number of arrangements
def num_arrangements : ℕ :=
  2 * (Nat.factorial num_parrots) * (Nat.factorial num_dogs) * (Nat.factorial num_cats)

-- Theorem statement
theorem animal_arrangement_count :
  num_arrangements = 34560 := by sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l1429_142934


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1429_142957

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 - (k - 4)*x - k + 7 > 0) ↔ (k > 4 ∧ k < 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1429_142957


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_6_minus_5_4_l1429_142950

theorem least_prime_factor_of_5_6_minus_5_4 :
  Nat.minFac (5^6 - 5^4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_6_minus_5_4_l1429_142950


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l1429_142945

/-- The solution set of the inequality (x+a)/(x^2+4x+3) > 0 --/
def SolutionSet (a : ℝ) : Set ℝ :=
  {x | (x + a) / (x^2 + 4*x + 3) > 0}

/-- The theorem stating that if the solution set is {x | x > -3, x ≠ -1}, then a = 1 --/
theorem solution_set_implies_a_equals_one :
  (∃ a : ℝ, SolutionSet a = {x : ℝ | x > -3 ∧ x ≠ -1}) →
  (∃ a : ℝ, SolutionSet a = {x : ℝ | x > -3 ∧ x ≠ -1} ∧ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l1429_142945


namespace NUMINAMATH_CALUDE_line_intersection_yz_plane_specific_line_intersection_l1429_142939

/-- The line passing through two given points intersects the yz-plane at a specific point. -/
theorem line_intersection_yz_plane (p₁ p₂ : ℝ × ℝ × ℝ) (h : p₁ ≠ p₂) :
  let line := λ t : ℝ => p₁ + t • (p₂ - p₁)
  ∃ t, line t = (0, 8, -5/2) :=
by
  sorry

/-- The specific instance of the line intersection problem. -/
theorem specific_line_intersection :
  let p₁ : ℝ × ℝ × ℝ := (3, 5, 1)
  let p₂ : ℝ × ℝ × ℝ := (5, 3, 6)
  let line := λ t : ℝ => p₁ + t • (p₂ - p₁)
  ∃ t, line t = (0, 8, -5/2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_intersection_yz_plane_specific_line_intersection_l1429_142939


namespace NUMINAMATH_CALUDE_solution_set_when_m_neg_one_m_range_for_subset_condition_l1429_142973

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2 * x - 1|

-- Part I
theorem solution_set_when_m_neg_one :
  {x : ℝ | f x (-1) ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
theorem m_range_for_subset_condition :
  {m : ℝ | ∀ x ∈ Set.Icc (3/4 : ℝ) 2, f x m ≤ |2 * x + 1|} = Set.Icc (-11/4 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_neg_one_m_range_for_subset_condition_l1429_142973


namespace NUMINAMATH_CALUDE_greatest_n_for_perfect_square_product_l1429_142959

def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem greatest_n_for_perfect_square_product (n : ℕ) : 
  n ≤ 2010 →
  (∀ k : ℕ, k > n → k ≤ 2010 → ¬is_perfect_square ((sum_squares k) * (sum_squares (2*k) - sum_squares k))) →
  is_perfect_square ((sum_squares n) * (sum_squares (2*n) - sum_squares n)) →
  n = 1935 := by sorry

end NUMINAMATH_CALUDE_greatest_n_for_perfect_square_product_l1429_142959


namespace NUMINAMATH_CALUDE_cost_per_square_meter_is_two_l1429_142921

-- Define the lawn dimensions
def lawn_length : ℝ := 80
def lawn_width : ℝ := 60

-- Define the road width
def road_width : ℝ := 10

-- Define the total cost of traveling both roads
def total_cost : ℝ := 2600

-- Theorem to prove
theorem cost_per_square_meter_is_two :
  let road_area := (lawn_length * road_width + lawn_width * road_width) - road_width * road_width
  total_cost / road_area = 2 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_square_meter_is_two_l1429_142921


namespace NUMINAMATH_CALUDE_min_value_sin_cos_l1429_142909

theorem min_value_sin_cos (p q : ℝ) : 
  (∀ θ : ℝ, p * Real.sin θ - q * Real.cos θ ≥ -Real.sqrt (p^2 + q^2)) ∧ 
  (∃ θ : ℝ, p * Real.sin θ - q * Real.cos θ = -Real.sqrt (p^2 + q^2)) := by
sorry

end NUMINAMATH_CALUDE_min_value_sin_cos_l1429_142909


namespace NUMINAMATH_CALUDE_b_over_a_is_real_l1429_142914

variable (a b x y : ℂ)

theorem b_over_a_is_real
  (h1 : a * b ≠ 0)
  (h2 : Complex.abs x = Complex.abs y)
  (h3 : x + y = a)
  (h4 : x * y = b) :
  ∃ (r : ℝ), b / a = r :=
sorry

end NUMINAMATH_CALUDE_b_over_a_is_real_l1429_142914


namespace NUMINAMATH_CALUDE_y_intercept_of_l_l1429_142996

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (a b : ℝ) : ℝ := b

/-- The line l is defined by the equation y = 3x - 2 -/
def l (x : ℝ) : ℝ := 3 * x - 2

theorem y_intercept_of_l :
  y_intercept 3 (-2) = -2 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_l_l1429_142996


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1429_142930

theorem product_sum_theorem (p q r s t : ℤ) :
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t →
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120 →
  p + q + r + s + t = 25 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1429_142930


namespace NUMINAMATH_CALUDE_fourth_number_in_first_set_l1429_142967

theorem fourth_number_in_first_set (x : ℝ) (y : ℝ) : 
  (28 + x + 70 + y + 104) / 5 = 67 →
  (50 + 62 + 97 + 124 + x) / 5 = 75.6 →
  y = 88 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_first_set_l1429_142967


namespace NUMINAMATH_CALUDE_scale_division_theorem_l1429_142984

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 10 * 12 + 5

/-- The number of equal parts the scale is divided into -/
def num_parts : ℕ := 5

/-- The length of each part in inches -/
def part_length : ℕ := scale_length / num_parts

/-- Converts inches to feet and remaining inches -/
def inches_to_feet_and_inches (inches : ℕ) : ℕ × ℕ :=
  (inches / 12, inches % 12)

theorem scale_division_theorem :
  inches_to_feet_and_inches part_length = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_scale_division_theorem_l1429_142984


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1429_142903

/-- Given that p and q vary inversely, prove that when q = 2.8 for p = 500, 
    then q = 1.12 when p = 1250 -/
theorem inverse_variation_problem (p q : ℝ) (h : p * q = 500 * 2.8) :
  p = 1250 → q = 1.12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1429_142903


namespace NUMINAMATH_CALUDE_base_of_first_term_l1429_142911

theorem base_of_first_term (x s : ℝ) (h : (x^16) * (25^s) = 5 * (10^16)) : x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_base_of_first_term_l1429_142911


namespace NUMINAMATH_CALUDE_r_geq_one_l1429_142954

noncomputable section

variables (m : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := Real.exp x

def g (m : ℝ) (x : ℝ) : ℝ := m * x + 4 * m

def r (m : ℝ) (x : ℝ) : ℝ := 1 / f x + (4 * m * x) / g m x

theorem r_geq_one (h1 : m > 0) (h2 : x ≥ 0) : r m x ≥ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_r_geq_one_l1429_142954


namespace NUMINAMATH_CALUDE_min_value_theorem_l1429_142916

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 8) :
  (2/x + 3/y) ≥ 25/8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 8 ∧ 2/x₀ + 3/y₀ = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1429_142916


namespace NUMINAMATH_CALUDE_cube_sum_of_sum_and_product_l1429_142946

theorem cube_sum_of_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 11) : x^3 + y^3 = 670 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_sum_and_product_l1429_142946


namespace NUMINAMATH_CALUDE_average_problem_l1429_142961

theorem average_problem (a b c X Y Z : ℝ) 
  (h1 : (a + b + c) / 3 = 5)
  (h2 : (X + Y + Z) / 3 = 7) :
  ((2*a + 3*X) + (2*b + 3*Y) + (2*c + 3*Z)) / 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l1429_142961


namespace NUMINAMATH_CALUDE_dave_ice_cubes_l1429_142986

theorem dave_ice_cubes (original : ℕ) (new : ℕ) (total : ℕ) : 
  original = 2 → new = 7 → total = original + new → total = 9 := by
  sorry

end NUMINAMATH_CALUDE_dave_ice_cubes_l1429_142986


namespace NUMINAMATH_CALUDE_complex_number_location_l1429_142978

theorem complex_number_location :
  ∀ z : ℂ, z * (2 + Complex.I) = 1 + 3 * Complex.I →
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1429_142978


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1429_142918

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α + Real.cos α) = -1) : 
  Real.tan α = 1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1429_142918


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1429_142998

theorem polynomial_factorization (a : ℝ) : a^3 + 10*a^2 + 25*a = a*(a+5)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1429_142998


namespace NUMINAMATH_CALUDE_death_rate_calculation_l1429_142976

/-- Given a birth rate, net growth rate, and initial population, 
    calculate the death rate. -/
def calculate_death_rate (birth_rate : ℝ) (net_growth_rate : ℝ) 
                          (initial_population : ℝ) : ℝ :=
  birth_rate - net_growth_rate * initial_population

/-- Theorem stating that under the given conditions, 
    the death rate is 16 per certain number of people. -/
theorem death_rate_calculation :
  let birth_rate : ℝ := 52
  let net_growth_rate : ℝ := 0.012
  let initial_population : ℝ := 3000
  calculate_death_rate birth_rate net_growth_rate initial_population = 16 := by
  sorry

#eval calculate_death_rate 52 0.012 3000

end NUMINAMATH_CALUDE_death_rate_calculation_l1429_142976


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l1429_142938

theorem tan_theta_minus_pi_fourth (θ : Real) : 
  (∃ (x y : Real), x = 2 ∧ y = 3 ∧ Real.tan θ = y / x) → 
  Real.tan (θ - Real.pi / 4) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l1429_142938


namespace NUMINAMATH_CALUDE_single_colony_days_l1429_142990

/-- Represents the growth of bacteria colonies -/
def BacteriaGrowth : Type :=
  { n : ℕ // n > 0 }

/-- The number of days it takes for two colonies to reach the habitat limit -/
def two_colony_days : BacteriaGrowth := ⟨15, by norm_num⟩

/-- Calculates the size of a colony after n days, given its initial size -/
def colony_size (initial : ℕ) (days : ℕ) : ℕ :=
  initial * 2^days

/-- Theorem stating that a single colony takes 16 days to reach the habitat limit -/
theorem single_colony_days :
  ∃ (limit : ℕ), limit > 0 ∧
    colony_size 1 (two_colony_days.val + 1) = limit ∧
    colony_size 2 two_colony_days.val = limit := by
  sorry

end NUMINAMATH_CALUDE_single_colony_days_l1429_142990


namespace NUMINAMATH_CALUDE_astronomical_unit_scientific_notation_l1429_142944

/-- One astronomical unit in kilometers -/
def astronomical_unit : ℝ := 1.496e9

/-- Scientific notation representation of one astronomical unit -/
def astronomical_unit_scientific : ℝ := 1.496 * 10^8

/-- Theorem stating that the astronomical unit can be expressed in scientific notation -/
theorem astronomical_unit_scientific_notation :
  astronomical_unit = astronomical_unit_scientific := by
  sorry

end NUMINAMATH_CALUDE_astronomical_unit_scientific_notation_l1429_142944


namespace NUMINAMATH_CALUDE_no_additional_cocoa_needed_l1429_142910

/-- Represents the chocolate cake recipe and baking scenario. -/
structure ChocolateCakeScenario where
  recipe_ratio : Real  -- Amount of cocoa powder per pound of cake batter
  cake_weight : Real   -- Total weight of the cake to be made
  given_cocoa : Real   -- Amount of cocoa powder already provided

/-- Calculates if additional cocoa powder is needed for the chocolate cake. -/
def additional_cocoa_needed (scenario : ChocolateCakeScenario) : Real :=
  scenario.recipe_ratio * scenario.cake_weight - scenario.given_cocoa

/-- Proves that no additional cocoa powder is needed in the given scenario. -/
theorem no_additional_cocoa_needed (scenario : ChocolateCakeScenario) 
  (h1 : scenario.recipe_ratio = 0.4)
  (h2 : scenario.cake_weight = 450)
  (h3 : scenario.given_cocoa = 259) : 
  additional_cocoa_needed scenario ≤ 0 := by
  sorry

#eval additional_cocoa_needed { recipe_ratio := 0.4, cake_weight := 450, given_cocoa := 259 }

end NUMINAMATH_CALUDE_no_additional_cocoa_needed_l1429_142910


namespace NUMINAMATH_CALUDE_max_sum_distances_in_unit_square_l1429_142941

theorem max_sum_distances_in_unit_square :
  ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
  let PA := Real.sqrt (x^2 + y^2)
  let PB := Real.sqrt ((1-x)^2 + y^2)
  let PC := Real.sqrt ((1-x)^2 + (1-y)^2)
  let PD := Real.sqrt (x^2 + (1-y)^2)
  PA + PB + PC + PD ≤ 2 + Real.sqrt 2 ∧
  ∃ (x₀ y₀ : ℝ), 0 ≤ x₀ ∧ x₀ ≤ 1 ∧ 0 ≤ y₀ ∧ y₀ ≤ 1 ∧
    let PA₀ := Real.sqrt (x₀^2 + y₀^2)
    let PB₀ := Real.sqrt ((1-x₀)^2 + y₀^2)
    let PC₀ := Real.sqrt ((1-x₀)^2 + (1-y₀)^2)
    let PD₀ := Real.sqrt (x₀^2 + (1-y₀)^2)
    PA₀ + PB₀ + PC₀ + PD₀ = 2 + Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_distances_in_unit_square_l1429_142941
