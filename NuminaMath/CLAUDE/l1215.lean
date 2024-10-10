import Mathlib

namespace binomial_square_constant_l1215_121584

/-- If x^2 + 80x + c is equal to the square of a binomial, then c = 1600 -/
theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 80*x + c = (x + a)^2) → c = 1600 := by
  sorry

end binomial_square_constant_l1215_121584


namespace expansion_coefficient_l1215_121576

theorem expansion_coefficient (a : ℝ) : 
  (∃ k : ℝ, k = 21 ∧ k = a^2 * 15 - 6 * a) ↔ (a = -1 ∨ a = 7/5) := by
sorry

end expansion_coefficient_l1215_121576


namespace complement_of_A_in_U_l1215_121593

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 4, 6}

theorem complement_of_A_in_U :
  U \ A = {1, 3, 5} := by sorry

end complement_of_A_in_U_l1215_121593


namespace shooting_sequences_l1215_121504

theorem shooting_sequences (n : Nat) (c₁ c₂ c₃ : Nat) 
  (h₁ : n = c₁ + c₂ + c₃) 
  (h₂ : c₁ = 3) 
  (h₃ : c₂ = 2) 
  (h₄ : c₃ = 3) :
  (Nat.factorial n) / (Nat.factorial c₁ * Nat.factorial c₂ * Nat.factorial c₃) = 560 :=
by sorry

end shooting_sequences_l1215_121504


namespace median_equations_l1215_121585

/-- Triangle ABC with given coordinates -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Equation of a line in general form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given triangle ABC -/
def givenTriangle : Triangle :=
  { A := (1, -4)
  , B := (6, 6)
  , C := (-2, 0) }

/-- Theorem stating the equations of the two medians -/
theorem median_equations (t : Triangle) 
  (h : t = givenTriangle) : 
  ∃ (l1 l2 : LineEquation),
    (l1.a = 6 ∧ l1.b = -8 ∧ l1.c = -13) ∧
    (l2.a = 7 ∧ l2.b = -1 ∧ l2.c = -11) :=
  sorry

end median_equations_l1215_121585


namespace pencils_per_row_l1215_121503

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 35)
  (h2 : num_rows = 7)
  (h3 : total_pencils = num_rows * pencils_per_row) :
  pencils_per_row = 5 := by
  sorry

end pencils_per_row_l1215_121503


namespace intersection_M_N_l1215_121529

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end intersection_M_N_l1215_121529


namespace intersection_of_A_and_B_l1215_121558

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end intersection_of_A_and_B_l1215_121558


namespace permutation_combination_equality_l1215_121549

/-- Given that A_n^2 = C_n^(n-3), prove that n = 8 --/
theorem permutation_combination_equality (n : ℕ) : 
  (n.factorial / (n - 2).factorial) = (n.factorial / ((3).factorial * (n - 3).factorial)) → n = 8 := by
  sorry

end permutation_combination_equality_l1215_121549


namespace rationalize_denominator_l1215_121571

theorem rationalize_denominator : 
  1 / (2 - Real.sqrt 2) = (2 + Real.sqrt 2) / 2 := by sorry

end rationalize_denominator_l1215_121571


namespace ellipse_intersection_slope_product_l1215_121577

/-- Given a line l passing through (-2,0) with slope k1 (k1 ≠ 0) intersecting 
    the ellipse x^2 + 2y^2 = 4 at points P1 and P2, and P being the midpoint of P1P2, 
    if k2 is the slope of OP, then k1 * k2 = -1/2 -/
theorem ellipse_intersection_slope_product (k1 : ℝ) (h1 : k1 ≠ 0) : 
  ∃ (P1 P2 P : ℝ × ℝ) (k2 : ℝ),
    (P1.1^2 + 2*P1.2^2 = 4) ∧ 
    (P2.1^2 + 2*P2.2^2 = 4) ∧
    (P1.2 = k1 * (P1.1 + 2)) ∧ 
    (P2.2 = k1 * (P2.1 + 2)) ∧
    (P = ((P1.1 + P2.1)/2, (P1.2 + P2.2)/2)) ∧
    (k2 = P.2 / P.1) →
    k1 * k2 = -1/2 := by
  sorry

end ellipse_intersection_slope_product_l1215_121577


namespace number_puzzle_l1215_121510

theorem number_puzzle : ∃ x : ℤ, x + 2 - 3 = 7 := by
  sorry

end number_puzzle_l1215_121510


namespace expression_value_l1215_121505

theorem expression_value : (2^1001 + 5^1002)^2 - (2^1001 - 5^1002)^2 = 40 * 10^1001 := by
  sorry

end expression_value_l1215_121505


namespace largest_prime_factor_l1215_121586

def expression : ℤ := 16^4 + 3 * 16^2 + 2 - 15^4

theorem largest_prime_factor (p : ℕ) : 
  Nat.Prime p ∧ p ∣ expression.natAbs ∧ 
  ∀ q : ℕ, Nat.Prime q ∧ q ∣ expression.natAbs → q ≤ p ↔ p = 241 := by
  sorry

end largest_prime_factor_l1215_121586


namespace problem_solution_l1215_121530

theorem problem_solution (a b : ℝ) 
  (h1 : a * b = 2 * (a + b) + 14) 
  (h2 : b - a = 3) : 
  b = 8 := by sorry

end problem_solution_l1215_121530


namespace factorial_equation_solutions_l1215_121522

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_equation_solutions :
  ∀ x y n : ℕ, (factorial x + factorial y) / factorial n = 3^n →
    ((x = 1 ∧ y = 2 ∧ n = 1) ∨ (x = 2 ∧ y = 1 ∧ n = 1)) := by
  sorry

end factorial_equation_solutions_l1215_121522


namespace complex_number_with_purely_imaginary_square_plus_three_l1215_121508

theorem complex_number_with_purely_imaginary_square_plus_three :
  ∃ (z : ℂ), (∀ (x : ℝ), (z^2 + 3).re = x → x = 0) ∧ z = (1 : ℂ) + (2 : ℂ) * I :=
by sorry

end complex_number_with_purely_imaginary_square_plus_three_l1215_121508


namespace joes_lifts_l1215_121545

theorem joes_lifts (total_weight first_lift : ℕ) 
  (h1 : total_weight = 1500)
  (h2 : first_lift = 600) : 
  2 * first_lift - (total_weight - first_lift) = 300 := by
  sorry

end joes_lifts_l1215_121545


namespace jesse_friends_bananas_l1215_121532

/-- Given a number of friends and bananas per friend, calculate the total number of bananas -/
def total_bananas (num_friends : ℝ) (bananas_per_friend : ℝ) : ℝ :=
  num_friends * bananas_per_friend

/-- Theorem: Jesse's friends have 63 bananas in total -/
theorem jesse_friends_bananas :
  total_bananas 3 21 = 63 := by
  sorry

end jesse_friends_bananas_l1215_121532


namespace balloon_radius_increase_l1215_121535

theorem balloon_radius_increase (c₁ c₂ : ℝ) (h₁ : c₁ = 24) (h₂ : c₂ = 30) :
  (c₂ / (2 * π)) - (c₁ / (2 * π)) = 3 / π := by sorry

end balloon_radius_increase_l1215_121535


namespace integer_roots_of_f_l1215_121564

def f (x : ℤ) : ℤ := 4*x^4 - 16*x^3 + 11*x^2 + 4*x - 3

theorem integer_roots_of_f :
  {x : ℤ | f x = 0} = {1, 3} := by sorry

end integer_roots_of_f_l1215_121564


namespace polynomial_factorization_and_range_l1215_121575

-- Define the polynomial and factored form
def P (x : ℝ) := x^3 - 2*x^2 - x + 2
def Q (a b c x : ℝ) := (x + a) * (x + b) * (x + c)

-- State the theorem
theorem polynomial_factorization_and_range :
  ∃ (a b c : ℝ),
    (∀ x, P x = Q a b c x) ∧
    (a > b) ∧ (b > c) ∧
    (a = 1) ∧ (b = -1) ∧ (c = -2) ∧
    (∀ x ∈ Set.Icc 0 3, a*x^2 + 2*b*x + c ∈ Set.Icc (-3) 1) ∧
    (∃ x₁ ∈ Set.Icc 0 3, a*x₁^2 + 2*b*x₁ + c = -3) ∧
    (∃ x₂ ∈ Set.Icc 0 3, a*x₂^2 + 2*b*x₂ + c = 1) :=
by sorry

end polynomial_factorization_and_range_l1215_121575


namespace percent_of_x_is_v_l1215_121521

theorem percent_of_x_is_v (x y z v : ℝ) 
  (h1 : 0.45 * z = 0.39 * y)
  (h2 : y = 0.75 * x)
  (h3 : v = 0.8 * z) :
  v = 0.52 * x :=
by sorry

end percent_of_x_is_v_l1215_121521


namespace alice_savings_difference_l1215_121506

def type_a_sales : ℝ := 1800
def type_b_sales : ℝ := 800
def type_c_sales : ℝ := 500
def basic_salary : ℝ := 500
def type_a_commission_rate : ℝ := 0.04
def type_b_commission_rate : ℝ := 0.06
def type_c_commission_rate : ℝ := 0.10
def monthly_expenses : ℝ := 600
def saving_goal : ℝ := 450
def usual_saving_rate : ℝ := 0.15

def total_commission : ℝ := 
  type_a_sales * type_a_commission_rate + 
  type_b_sales * type_b_commission_rate + 
  type_c_sales * type_c_commission_rate

def total_earnings : ℝ := basic_salary + total_commission

def net_earnings : ℝ := total_earnings - monthly_expenses

def actual_savings : ℝ := net_earnings * usual_saving_rate

theorem alice_savings_difference : saving_goal - actual_savings = 439.50 := by
  sorry

end alice_savings_difference_l1215_121506


namespace crayons_count_l1215_121523

/-- Given a group of children where each child has a certain number of crayons,
    calculate the total number of crayons. -/
def total_crayons (crayons_per_child : ℕ) (num_children : ℕ) : ℕ :=
  crayons_per_child * num_children

/-- Theorem stating that with 6 crayons per child and 12 children, 
    the total number of crayons is 72. -/
theorem crayons_count : total_crayons 6 12 = 72 := by
  sorry

end crayons_count_l1215_121523


namespace max_seated_people_is_14_l1215_121540

/-- Represents the state of the break room --/
structure BreakRoom where
  totalTables : Nat
  maxSeatsPerTable : Nat
  maxSeatsPerTableWithDistancing : Nat
  occupiedTables : List Nat
  totalChairs : Nat

/-- Calculates the maximum number of people that can be seated in the break room --/
def maxSeatedPeople (room : BreakRoom) : Nat :=
  sorry

/-- Theorem stating that the maximum number of people that can be seated is 14 --/
theorem max_seated_people_is_14 (room : BreakRoom) : 
  room.totalTables = 7 ∧ 
  room.maxSeatsPerTable = 6 ∧ 
  room.maxSeatsPerTableWithDistancing = 3 ∧ 
  room.occupiedTables = [2, 1, 1, 3] ∧ 
  room.totalChairs = 14 →
  maxSeatedPeople room = 14 :=
by sorry

end max_seated_people_is_14_l1215_121540


namespace function_inequality_l1215_121595

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
    (h_cond : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end function_inequality_l1215_121595


namespace cubic_equation_solutions_l1215_121570

theorem cubic_equation_solutions :
  let f : ℝ → ℝ := λ x => x^3 - 8
  let g : ℝ → ℝ := λ x => 16 * (x + 1)^(1/3)
  ∀ x : ℝ, f x = g x ↔ x = -2 ∨ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

end cubic_equation_solutions_l1215_121570


namespace interest_difference_l1215_121515

def initial_amount : ℝ := 1250

def compound_rate_year1 : ℝ := 0.08
def compound_rate_year2 : ℝ := 0.10
def compound_rate_year3 : ℝ := 0.12

def simple_rate_year1 : ℝ := 0.04
def simple_rate_year2 : ℝ := 0.06
def simple_rate_year3 : ℝ := 0.07
def simple_rate_year4 : ℝ := 0.09

def compound_interest (principal : ℝ) (rate1 rate2 rate3 : ℝ) : ℝ :=
  principal * (1 + rate1) * (1 + rate2) * (1 + rate3)

def simple_interest (principal : ℝ) (rate1 rate2 rate3 rate4 : ℝ) : ℝ :=
  principal * (1 + rate1 + rate2 + rate3 + rate4)

theorem interest_difference :
  compound_interest initial_amount compound_rate_year1 compound_rate_year2 compound_rate_year3 -
  simple_interest initial_amount simple_rate_year1 simple_rate_year2 simple_rate_year3 simple_rate_year4 = 88.2 := by
  sorry

end interest_difference_l1215_121515


namespace one_mile_equals_500_rods_l1215_121542

/-- Conversion factor from miles to furlongs -/
def mile_to_furlong : ℚ := 10

/-- Conversion factor from furlongs to rods -/
def furlong_to_rod : ℚ := 50

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_furlong * furlong_to_rod

/-- Theorem stating that one mile is equal to 500 rods -/
theorem one_mile_equals_500_rods : rods_in_mile = 500 := by
  sorry

end one_mile_equals_500_rods_l1215_121542


namespace problem_2009_2007_2008_l1215_121550

theorem problem_2009_2007_2008 : 2009 * (2007 / 2008) + 1 / 2008 = 2008 := by
  sorry

end problem_2009_2007_2008_l1215_121550


namespace set_difference_empty_implies_subset_l1215_121541

theorem set_difference_empty_implies_subset (A B : Set α) : 
  (A \ B = ∅) → (A ⊆ B) := by
  sorry

end set_difference_empty_implies_subset_l1215_121541


namespace symmetric_angle_660_l1215_121511

def is_symmetric_angle (θ : ℤ) : Prop :=
  ∃ k : ℤ, θ = -60 + 360 * k

theorem symmetric_angle_660 :
  is_symmetric_angle 660 ∧
  ¬ is_symmetric_angle (-660) ∧
  ¬ is_symmetric_angle 690 ∧
  ¬ is_symmetric_angle (-690) :=
sorry

end symmetric_angle_660_l1215_121511


namespace factor_expression_l1215_121572

theorem factor_expression (x : ℝ) : 92 * x^3 - 184 * x^6 = 92 * x^3 * (1 - 2 * x^3) := by
  sorry

end factor_expression_l1215_121572


namespace a_eq_two_sufficient_not_necessary_l1215_121518

/-- A quadratic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The property that f is increasing on [-1,∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, -1 ≤ x ∧ x < y → f a x < f a y

/-- The statement that a=2 is sufficient but not necessary for f to be increasing on [-1,∞) -/
theorem a_eq_two_sufficient_not_necessary :
  (is_increasing_on_interval 2) ∧
  (∃ a : ℝ, a ≠ 2 ∧ is_increasing_on_interval a) :=
sorry

end a_eq_two_sufficient_not_necessary_l1215_121518


namespace rectangular_prism_volume_increase_l1215_121590

theorem rectangular_prism_volume_increase (a b c : ℝ) : 
  (a * b * c = 8) → 
  ((a + 1) * (b + 1) * (c + 1) = 27) → 
  ((a + 2) * (b + 2) * (c + 2) = 64) := by
  sorry

end rectangular_prism_volume_increase_l1215_121590


namespace sixth_test_score_l1215_121520

def average_score : ℝ := 84
def num_tests : ℕ := 6
def known_scores : List ℝ := [83, 77, 92, 85, 89]

theorem sixth_test_score :
  let total_sum := average_score * num_tests
  let sum_of_known_scores := known_scores.sum
  total_sum - sum_of_known_scores = 78 := by
  sorry

end sixth_test_score_l1215_121520


namespace incident_ray_slope_l1215_121592

/-- Given a circle with center (2, -1) and a point P(-1, -3), prove that the slope
    of the line passing through P and the reflection of the circle's center
    across the x-axis is 4/3. -/
theorem incident_ray_slope (P : ℝ × ℝ) (C : ℝ × ℝ) :
  P = (-1, -3) →
  C = (2, -1) →
  let D : ℝ × ℝ := (C.1, -C.2)  -- Reflection of C across x-axis
  (D.2 - P.2) / (D.1 - P.1) = 4/3 := by
sorry

end incident_ray_slope_l1215_121592


namespace transformed_system_solution_l1215_121579

/-- Given a system of equations with solution, prove that a transformed system has a specific solution -/
theorem transformed_system_solution (a b m n : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 10 ∧ m * x - n * y = 8 ∧ x = 1 ∧ y = 2) →
  (∃ x y : ℝ, (1/2) * a * (x + y) + (1/3) * b * (x - y) = 10 ∧
              (1/2) * m * (x + y) - (1/3) * n * (x - y) = 8 ∧
              x = 4 ∧ y = -2) :=
by sorry

end transformed_system_solution_l1215_121579


namespace wrapping_paper_area_theorem_l1215_121565

/-- Represents a box with a square base -/
structure Box where
  base_length : ℝ
  height : ℝ

/-- Calculates the area of wrapping paper needed for a given box -/
def wrapping_paper_area (box : Box) : ℝ :=
  2 * (box.base_length + box.height)^2

/-- Theorem stating that the area of the wrapping paper is 2(w+h)^2 -/
theorem wrapping_paper_area_theorem (box : Box) :
  wrapping_paper_area box = 2 * (box.base_length + box.height)^2 := by
  sorry

end wrapping_paper_area_theorem_l1215_121565


namespace f_simplification_f_third_quadrant_f_specific_angle_l1215_121528

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 / 2 * Real.pi)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_third_quadrant (α : Real) 
  (h1 : Real.pi < α ∧ α < 3 * Real.pi / 2)  -- α is in the third quadrant
  (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 := by sorry

theorem f_specific_angle : f (-31 * Real.pi / 3) = -1 / 2 := by sorry

end f_simplification_f_third_quadrant_f_specific_angle_l1215_121528


namespace sine_difference_inequality_l1215_121553

theorem sine_difference_inequality (A B : Real) (hA : 0 ≤ A ∧ A ≤ π) (hB : 0 ≤ B ∧ B ≤ π) :
  |Real.sin A - Real.sin B| ≤ |Real.sin (A - B)| := by
  sorry

end sine_difference_inequality_l1215_121553


namespace twelve_returning_sequences_l1215_121574

-- Define the triangle T'
structure Triangle :=
  (v1 v2 v3 : ℝ × ℝ)

-- Define the set of transformations
inductive Transformation
  | Rotate60 : Transformation
  | Rotate120 : Transformation
  | Rotate240 : Transformation
  | ReflectYeqX : Transformation
  | ReflectYeqNegX : Transformation

-- Define a sequence of three transformations
def TransformationSequence := (Transformation × Transformation × Transformation)

-- Define the original triangle T'
def T' : Triangle :=
  { v1 := (1, 1), v2 := (5, 1), v3 := (1, 4) }

-- Function to check if a sequence of transformations returns T' to its original position
def returnsToOriginal (seq : TransformationSequence) : Prop :=
  sorry

-- Theorem stating that exactly 12 sequences return T' to its original position
theorem twelve_returning_sequences :
  ∃ (S : Finset TransformationSequence),
    (∀ seq ∈ S, returnsToOriginal seq) ∧
    (∀ seq, returnsToOriginal seq → seq ∈ S) ∧
    Finset.card S = 12 :=
  sorry

end twelve_returning_sequences_l1215_121574


namespace k_range_oa_perpendicular_ob_l1215_121567

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = -x
def line (k x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
def intersection_points (k : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    parabola x1 y1 ∧ line k x1 y1 ∧
    parabola x2 y2 ∧ line k x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)

-- Define the vertex O of the parabola
def vertex : ℝ × ℝ := (0, 0)

-- Theorem for the range of k
theorem k_range : 
  ∀ k : ℝ, intersection_points k ↔ k ≠ 0 :=
sorry

-- Define perpendicularity
def perpendicular (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) = 0

-- Theorem for perpendicularity of OA and OB
theorem oa_perpendicular_ob (k : ℝ) :
  k ≠ 0 → 
  ∃ (x1 y1 x2 y2 : ℝ),
    parabola x1 y1 ∧ line k x1 y1 ∧
    parabola x2 y2 ∧ line k x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) →
    perpendicular vertex (x1, y1) (x2, y2) :=
sorry

end k_range_oa_perpendicular_ob_l1215_121567


namespace remainder_x15_plus_1_div_x_plus_1_l1215_121556

theorem remainder_x15_plus_1_div_x_plus_1 (x : ℝ) : (x^15 + 1) % (x + 1) = 0 := by
  sorry

end remainder_x15_plus_1_div_x_plus_1_l1215_121556


namespace cubic_difference_evaluation_l1215_121597

theorem cubic_difference_evaluation : 
  2010^3 - 2007 * 2010^2 - 2007^2 * 2010 + 2007^3 = 36153 := by
  sorry

end cubic_difference_evaluation_l1215_121597


namespace walter_zoo_time_l1215_121580

theorem walter_zoo_time (S : ℝ) : 
  S > 0 ∧ 
  S + 8*S + 13 + S/2 = 185 → 
  S = 16 :=
by sorry

end walter_zoo_time_l1215_121580


namespace perfect_squares_mod_six_l1215_121591

theorem perfect_squares_mod_six :
  (∀ n : ℤ, n^2 % 6 ≠ 2) ∧
  (∃ K : Set ℤ, Set.Infinite K ∧ ∀ k ∈ K, ((6 * k + 3)^2) % 6 = 3) :=
by sorry

end perfect_squares_mod_six_l1215_121591


namespace job_arrangements_l1215_121599

/-- The number of ways to arrange n distinct candidates into k distinct jobs,
    where each job requires exactly one person and each person can take only one job. -/
def arrangements (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

/-- There are 3 different jobs, each requiring only one person,
    and each person taking on only one job.
    There are 4 candidates available for selection. -/
theorem job_arrangements : arrangements 4 3 = 24 := by
  sorry

end job_arrangements_l1215_121599


namespace hexagon_diagonal_area_bound_l1215_121543

/-- A convex hexagon is a six-sided polygon where all interior angles are less than or equal to 180 degrees. -/
structure ConvexHexagon where
  -- We assume the existence of a convex hexagon without explicitly defining its properties
  -- as the specific geometric representation is not crucial for this theorem.

/-- The theorem states that for any convex hexagon, there exists a diagonal that cuts off a triangle
    with an area less than or equal to one-sixth of the total area of the hexagon. -/
theorem hexagon_diagonal_area_bound (h : ConvexHexagon) (S : ℝ) (h_area : S > 0) :
  ∃ (triangle_area : ℝ), triangle_area ≤ S / 6 ∧ triangle_area > 0 := by
  sorry


end hexagon_diagonal_area_bound_l1215_121543


namespace larger_number_problem_l1215_121554

theorem larger_number_problem (L S : ℕ) (hL : L > S) :
  L - S = 1365 →
  L = 6 * S + 15 →
  L = 1635 := by
  sorry

end larger_number_problem_l1215_121554


namespace sue_votes_l1215_121559

theorem sue_votes (total_votes : ℕ) (candidate1_percent : ℚ) (candidate2_percent : ℚ)
  (h_total : total_votes = 1000)
  (h_cand1 : candidate1_percent = 20 / 100)
  (h_cand2 : candidate2_percent = 45 / 100) :
  (1 - (candidate1_percent + candidate2_percent)) * total_votes = 350 :=
by sorry

end sue_votes_l1215_121559


namespace binomial_expansion_sum_l1215_121560

theorem binomial_expansion_sum (n : ℕ) : 
  (∃ P S : ℕ, (P = (3 + 1)^n) ∧ (S = 2^n) ∧ (P + S = 272)) → n = 4 := by
  sorry

end binomial_expansion_sum_l1215_121560


namespace box_filling_theorem_l1215_121552

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  sorry

/-- The theorem stating that for a box with dimensions 36x45x18 inches, 
    the smallest number of identical cubes that can fill it is 40 -/
theorem box_filling_theorem : 
  let box : BoxDimensions := { length := 36, width := 45, depth := 18 }
  smallestNumberOfCubes box = 40 := by
  sorry

end box_filling_theorem_l1215_121552


namespace odd_function_property_l1215_121587

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_even : is_even_function (fun x ↦ f (x + 1)))
  (h_f_neg_one : f (-1) = -1) :
  f 2018 + f 2019 = -1 := by
  sorry

end odd_function_property_l1215_121587


namespace percentage_men_undeclared_l1215_121581

/-- Represents the percentages of students in different majors and categories -/
structure ClassComposition where
  men_science : ℝ
  men_humanities : ℝ
  men_business : ℝ
  men_double_science_humanities : ℝ
  men_double_science_business : ℝ
  men_double_humanities_business : ℝ

/-- Theorem stating the percentage of men with undeclared majors -/
theorem percentage_men_undeclared (c : ClassComposition) : 
  c.men_science = 24 ∧ 
  c.men_humanities = 13 ∧ 
  c.men_business = 18 ∧
  c.men_double_science_humanities = 13.5 ∧
  c.men_double_science_business = 9 ∧
  c.men_double_humanities_business = 6.75 →
  100 - (c.men_science + c.men_humanities + c.men_business + 
         c.men_double_science_humanities + c.men_double_science_business + 
         c.men_double_humanities_business) = 15.75 := by
  sorry

#check percentage_men_undeclared

end percentage_men_undeclared_l1215_121581


namespace no_roots_of_composite_l1215_121517

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating that if f(x) = 2x has no real roots, then f(f(x)) = 4x has no real roots -/
theorem no_roots_of_composite (a b c : ℝ) (ha : a ≠ 0) 
  (h : ∀ x : ℝ, f a b c x ≠ 2 * x) : 
  ∀ x : ℝ, f a b c (f a b c x) ≠ 4 * x := by
  sorry

end no_roots_of_composite_l1215_121517


namespace proportion_solution_l1215_121589

theorem proportion_solution (x : ℚ) : (3/4 : ℚ) / x = 7/8 → x = 6/7 := by
  sorry

end proportion_solution_l1215_121589


namespace sequence_term_16_l1215_121594

theorem sequence_term_16 (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n = (Real.sqrt 2) ^ (n - 1)) →
  ∃ n : ℕ, n > 0 ∧ a n = 16 ∧ n = 9 := by
  sorry

end sequence_term_16_l1215_121594


namespace car_oil_problem_l1215_121596

/-- Represents the relationship between remaining oil and distance traveled for a car -/
def oil_remaining (x : ℝ) : ℝ := 56 - 0.08 * x

/-- The initial amount of oil in the tank -/
def initial_oil : ℝ := 56

/-- The rate of oil consumption per kilometer -/
def consumption_rate : ℝ := 0.08

theorem car_oil_problem :
  (∀ x : ℝ, oil_remaining x = 56 - 0.08 * x) ∧
  (oil_remaining 350 = 28) ∧
  (∃ x : ℝ, oil_remaining x = 8 ∧ x = 600) := by
  sorry


end car_oil_problem_l1215_121596


namespace pasha_mistake_l1215_121526

theorem pasha_mistake : 
  ¬ ∃ (K R O S C T A P : ℕ),
    (K < 10 ∧ R < 10 ∧ O < 10 ∧ S < 10 ∧ C < 10 ∧ T < 10 ∧ A < 10 ∧ P < 10) ∧
    (K ≠ R ∧ K ≠ O ∧ K ≠ S ∧ K ≠ C ∧ K ≠ T ∧ K ≠ A ∧ K ≠ P ∧
     R ≠ O ∧ R ≠ S ∧ R ≠ C ∧ R ≠ T ∧ R ≠ A ∧ R ≠ P ∧
     O ≠ S ∧ O ≠ C ∧ O ≠ T ∧ O ≠ A ∧ O ≠ P ∧
     S ≠ C ∧ S ≠ T ∧ S ≠ A ∧ S ≠ P ∧
     C ≠ T ∧ C ≠ A ∧ C ≠ P ∧
     T ≠ A ∧ T ≠ P ∧
     A ≠ P) ∧
    (K * 10000 + R * 1000 + O * 100 + S * 10 + S + 2011 = 
     C * 10000 + T * 1000 + A * 100 + P * 10 + T) :=
by sorry

end pasha_mistake_l1215_121526


namespace job_completion_time_l1215_121562

/-- Given that Sylvia can complete a job in 45 minutes and Carla can complete
    the same job in 30 minutes, prove that together they can complete the job
    in 18 minutes. -/
theorem job_completion_time (sylvia_time carla_time : ℝ) 
    (h_sylvia : sylvia_time = 45)
    (h_carla : carla_time = 30) :
    1 / (1 / sylvia_time + 1 / carla_time) = 18 := by
  sorry


end job_completion_time_l1215_121562


namespace magazine_boxes_l1215_121569

theorem magazine_boxes (total_magazines : ℕ) (magazines_per_box : ℕ) (h1 : total_magazines = 63) (h2 : magazines_per_box = 9) :
  total_magazines / magazines_per_box = 7 :=
by
  sorry

end magazine_boxes_l1215_121569


namespace price_change_after_four_years_l1215_121531

theorem price_change_after_four_years (initial_price : ℝ) :
  let price_after_two_increases := initial_price * (1 + 0.2)^2
  let final_price := price_after_two_increases * (1 - 0.2)^2
  final_price = initial_price * (1 - 0.0784) :=
by sorry

end price_change_after_four_years_l1215_121531


namespace daily_earnings_of_c_l1215_121512

theorem daily_earnings_of_c (a b c : ℕ) 
  (h1 : a + b + c = 600)
  (h2 : a + c = 400)
  (h3 : b + c = 300) :
  c = 100 := by
sorry

end daily_earnings_of_c_l1215_121512


namespace choose_cooks_l1215_121546

theorem choose_cooks (n m : ℕ) (h1 : n = 10) (h2 : m = 3) :
  Nat.choose n m = 120 := by
  sorry

end choose_cooks_l1215_121546


namespace storm_rainfall_l1215_121516

/-- The total rainfall during a two-hour storm given specific conditions -/
theorem storm_rainfall (first_hour_rain : ℝ) (second_hour_increment : ℝ) : 
  first_hour_rain = 5 →
  second_hour_increment = 7 →
  (first_hour_rain + (2 * first_hour_rain + second_hour_increment)) = 22 := by
sorry

end storm_rainfall_l1215_121516


namespace exam_mean_score_l1215_121513

theorem exam_mean_score (score_below mean standard_deviation : ℝ) 
  (h1 : score_below = mean - 2 * standard_deviation)
  (h2 : 98 = mean + 3 * standard_deviation)
  (h3 : score_below = 58) : mean = 74 := by
  sorry

end exam_mean_score_l1215_121513


namespace unique_triple_solution_l1215_121598

theorem unique_triple_solution : 
  ∃! (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b * c = 2010 ∧ 
    b + c * a = 250 ∧
    a = 3 ∧ b = 223 ∧ c = 9 := by
  sorry

end unique_triple_solution_l1215_121598


namespace yuna_weekly_problems_l1215_121537

/-- The number of English problems Yuna solves per day -/
def problems_per_day : ℕ := 8

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of English problems Yuna solves in one week -/
def problems_per_week : ℕ := problems_per_day * days_in_week

theorem yuna_weekly_problems : problems_per_week = 56 := by
  sorry

end yuna_weekly_problems_l1215_121537


namespace sqrt_two_plus_abs_diff_solve_quadratic_equation_l1215_121533

-- Part 1
theorem sqrt_two_plus_abs_diff : 
  Real.sqrt 2 * (Real.sqrt 2 + 1) + |Real.sqrt 2 - Real.sqrt 3| = 2 + Real.sqrt 3 := by
  sorry

-- Part 2
theorem solve_quadratic_equation : 
  ∀ x : ℝ, 4 * x^2 = 25 ↔ x = 5/2 ∨ x = -5/2 := by
  sorry

end sqrt_two_plus_abs_diff_solve_quadratic_equation_l1215_121533


namespace pencils_in_drawer_proof_l1215_121525

/-- The number of pencils initially in the drawer -/
def initial_drawer_pencils : ℕ := 43

/-- The number of pencils initially on the desk -/
def initial_desk_pencils : ℕ := 19

/-- The number of pencils added to the desk -/
def added_desk_pencils : ℕ := 16

/-- The total number of pencils -/
def total_pencils : ℕ := 78

/-- Theorem stating that the initial number of pencils in the drawer is correct -/
theorem pencils_in_drawer_proof :
  initial_drawer_pencils = total_pencils - (initial_desk_pencils + added_desk_pencils) :=
by sorry

end pencils_in_drawer_proof_l1215_121525


namespace greatest_good_and_smallest_bad_l1215_121534

/-- Definition of a GOOD number -/
def isGood (M : ℕ) : Prop :=
  ∃ a b c d : ℕ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

/-- Definition of a BAD number -/
def isBad (M : ℕ) : Prop := ¬(isGood M)

/-- The greatest GOOD number -/
def greatestGood : ℕ := 576

/-- The smallest BAD number -/
def smallestBad : ℕ := 443

/-- Theorem stating that 576 is the greatest GOOD number and 443 is the smallest BAD number -/
theorem greatest_good_and_smallest_bad :
  (∀ M : ℕ, M > greatestGood → isBad M) ∧
  (∀ M : ℕ, M < smallestBad → isGood M) ∧
  isGood greatestGood ∧
  isBad smallestBad :=
sorry

end greatest_good_and_smallest_bad_l1215_121534


namespace dexter_cards_count_l1215_121527

/-- The number of boxes filled with basketball cards -/
def basketball_boxes : ℕ := 15

/-- The number of cards in each basketball box -/
def basketball_cards_per_box : ℕ := 20

/-- The difference in the number of boxes between basketball and football cards -/
def box_difference : ℕ := 7

/-- The number of cards in each football box -/
def football_cards_per_box : ℕ := 25

/-- The total number of cards Dexter has -/
def total_cards : ℕ := basketball_boxes * basketball_cards_per_box + 
  (basketball_boxes - box_difference) * football_cards_per_box

theorem dexter_cards_count : total_cards = 500 := by
  sorry

end dexter_cards_count_l1215_121527


namespace upstream_speed_calculation_l1215_121538

/-- Represents the speed of a man rowing in different conditions. -/
structure RowingSpeed where
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the upstream speed of a man given his rowing speeds in still water and downstream. -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given a man's speed in still water is 35 kmph and his downstream speed is 45 kmph, his upstream speed is 25 kmph. -/
theorem upstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 35) 
  (h2 : s.downstream = 45) : 
  upstreamSpeed s = 25 := by
  sorry

#eval upstreamSpeed { stillWater := 35, downstream := 45 }

end upstream_speed_calculation_l1215_121538


namespace soccer_club_girls_l1215_121539

theorem soccer_club_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  attended = 18 →
  boys + girls = total →
  boys + (girls / 3) = attended →
  boys + girls = total →
  girls = 18 := by
sorry

end soccer_club_girls_l1215_121539


namespace condition_analysis_l1215_121500

theorem condition_analysis (a b : ℝ) : 
  (∃ a b, a^2 = b^2 ∧ a^2 + b^2 ≠ 2*a*b) ∧ 
  (∀ a b, a^2 + b^2 = 2*a*b → a^2 = b^2) := by
  sorry

end condition_analysis_l1215_121500


namespace boxes_with_neither_l1215_121551

theorem boxes_with_neither (total : ℕ) (stickers : ℕ) (stamps : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : stickers = 9)
  (h3 : stamps = 5)
  (h4 : both = 3) : 
  total - (stickers + stamps - both) = 4 := by
  sorry

end boxes_with_neither_l1215_121551


namespace milk_powder_cost_july_l1215_121501

theorem milk_powder_cost_july (june_cost : ℝ) 
  (h1 : june_cost > 0)
  (h2 : 3 * (3 * june_cost + 0.4 * june_cost) / 2 = 5.1) : 
  0.4 * june_cost = 0.4 := by sorry

end milk_powder_cost_july_l1215_121501


namespace ngon_recovery_l1215_121588

/-- Represents a point in the plane with an associated number -/
structure MarkedPoint where
  x : ℝ
  y : ℝ
  number : ℕ

/-- Represents a regular n-gon with its center -/
structure RegularNGon where
  n : ℕ
  center : MarkedPoint
  vertices : Fin n → MarkedPoint

/-- Represents a triangle formed by two adjacent vertices and the center -/
structure Triangle where
  a : MarkedPoint
  b : MarkedPoint
  c : MarkedPoint

/-- Function to generate the list of triangles from a regular n-gon -/
def generateTriangles (ngon : RegularNGon) : List Triangle := sorry

/-- Function to get the multiset of numbers from a triangle -/
def getTriangleNumbers (triangle : Triangle) : Multiset ℕ := sorry

/-- Predicate to check if the original numbers can be uniquely recovered -/
def canRecover (ngon : RegularNGon) : Prop := sorry

theorem ngon_recovery (n : ℕ) :
  ∀ (ngon : RegularNGon),
    ngon.n = n →
    canRecover ngon ↔ Odd n :=
  sorry

end ngon_recovery_l1215_121588


namespace sin_cos_sum_equals_quarter_l1215_121568

theorem sin_cos_sum_equals_quarter :
  Real.sin (20 * π / 180) * Real.cos (70 * π / 180) +
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) = 1/4 := by
  sorry

end sin_cos_sum_equals_quarter_l1215_121568


namespace segment_sum_midpoint_inequality_l1215_121536

theorem segment_sum_midpoint_inequality
  (f : ℚ → ℤ) :
  ∃ (x y : ℚ), f x + f y ≤ 2 * f ((x + y) / 2) :=
sorry

end segment_sum_midpoint_inequality_l1215_121536


namespace total_bowling_balls_l1215_121563

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 := by
  sorry

end total_bowling_balls_l1215_121563


namespace simplify_fraction_l1215_121573

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 := by
  sorry

end simplify_fraction_l1215_121573


namespace odd_divisors_of_square_plus_one_l1215_121566

theorem odd_divisors_of_square_plus_one (x : ℤ) (d : ℤ) (h : d ∣ x^2 + 1) (hodd : Odd d) :
  ∃ (k : ℤ), d = 4 * k + 1 := by
sorry

end odd_divisors_of_square_plus_one_l1215_121566


namespace league_score_range_l1215_121555

/-- Represents a sports league -/
structure League where
  numTeams : ℕ
  pointsForWin : ℕ
  pointsForDraw : ℕ

/-- Calculate the total number of games in a double round-robin tournament -/
def totalGames (league : League) : ℕ :=
  league.numTeams * (league.numTeams - 1)

/-- Calculate the minimum possible total score for the league -/
def minTotalScore (league : League) : ℕ :=
  (totalGames league) * (2 * league.pointsForDraw)

/-- Calculate the maximum possible total score for the league -/
def maxTotalScore (league : League) : ℕ :=
  (totalGames league) * league.pointsForWin

/-- Theorem stating that the total score for a 15-team league with 3 points for a win
    and 1 point for a draw is between 420 and 630, inclusive -/
theorem league_score_range :
  let league := League.mk 15 3 1
  420 ≤ minTotalScore league ∧ maxTotalScore league ≤ 630 := by
  sorry

#eval minTotalScore (League.mk 15 3 1)
#eval maxTotalScore (League.mk 15 3 1)

end league_score_range_l1215_121555


namespace prob_queen_of_diamonds_l1215_121519

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)

/-- Represents a specific card -/
structure Card :=
  (rank : String)
  (suit : String)

/-- Definition of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4 }

/-- Probability of drawing a specific card from a deck -/
def prob_draw_card (d : Deck) (c : Card) : ℚ :=
  1 / d.total_cards

/-- Queen of Diamonds card -/
def queen_of_diamonds : Card :=
  { rank := "Queen",
    suit := "Diamonds" }

/-- Theorem: Probability of drawing Queen of Diamonds from a standard deck is 1/52 -/
theorem prob_queen_of_diamonds :
  prob_draw_card standard_deck queen_of_diamonds = 1 / 52 := by
  sorry

end prob_queen_of_diamonds_l1215_121519


namespace negation_equivalence_l1215_121502

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 + x₀ - 2 < 0) ↔ (∀ x₀ : ℝ, x₀^2 + x₀ - 2 ≥ 0) :=
by sorry

end negation_equivalence_l1215_121502


namespace rental_van_cost_increase_l1215_121548

theorem rental_van_cost_increase (C : ℝ) : 
  C / 8 - C / 9 = C / 72 := by sorry

end rental_van_cost_increase_l1215_121548


namespace park_trees_l1215_121561

/-- The number of trees in a rectangular park -/
def num_trees (length width tree_density : ℕ) : ℕ :=
  (length * width) / tree_density

/-- Proof that a park with given dimensions and tree density has 100,000 trees -/
theorem park_trees : num_trees 1000 2000 20 = 100000 := by
  sorry

end park_trees_l1215_121561


namespace smallest_integer_solution_l1215_121578

theorem smallest_integer_solution (x : ℤ) : (∀ y : ℤ, 7 + 3 * y < 26 → x ≤ y) ↔ x = 6 := by
  sorry

end smallest_integer_solution_l1215_121578


namespace S_infinite_l1215_121582

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- The set of natural numbers n such that σ(n)/n > σ(k)/k for all k < n -/
def S : Set ℕ :=
  {n : ℕ | ∀ k < n, (sigma n : ℚ) / n > (sigma k : ℚ) / k}

/-- Theorem stating that S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end S_infinite_l1215_121582


namespace two_distinct_zeros_implies_m_3_or_4_l1215_121524

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 1) * x - 1

-- Define the theorem
theorem two_distinct_zeros_implies_m_3_or_4 :
  ∀ m : ℝ,
  (∃ x y : ℝ, x ≠ y ∧ x ∈ Set.Icc (-1) 2 ∧ y ∈ Set.Icc (-1) 2 ∧ f m x = 0 ∧ f m y = 0) →
  (m = 3 ∨ m = 4) :=
by sorry

end two_distinct_zeros_implies_m_3_or_4_l1215_121524


namespace parabola_above_line_l1215_121557

theorem parabola_above_line (p : ℝ) : 
  (∀ x : ℝ, x^2 - 2*p*x + p + 1 ≥ -12*x + 5) ↔ (5 ≤ p ∧ p ≤ 8) := by
  sorry

end parabola_above_line_l1215_121557


namespace tangent_ratio_theorem_l1215_121507

theorem tangent_ratio_theorem (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (Real.cos θ ^ 2 + Real.sin θ * Real.cos θ) = 8/3 := by
  sorry

end tangent_ratio_theorem_l1215_121507


namespace m_less_than_one_l1215_121509

/-- Given that the solution set of |x| + |x-1| > m is ℝ and 
    f(x) = -(7-3m)^x is decreasing on ℝ, prove that m < 1 -/
theorem m_less_than_one (m : ℝ) 
  (h1 : ∀ x : ℝ, |x| + |x - 1| > m)
  (h2 : Monotone (fun x => -(7 - 3*m)^x)) : 
  m < 1 := by
  sorry

end m_less_than_one_l1215_121509


namespace scavenger_hunt_difference_l1215_121583

theorem scavenger_hunt_difference (lewis_items samantha_items tanya_items : ℕ) : 
  lewis_items = 20 →
  samantha_items = 4 * tanya_items →
  tanya_items = 4 →
  lewis_items - samantha_items = 4 := by
sorry

end scavenger_hunt_difference_l1215_121583


namespace alok_ice_cream_order_l1215_121544

/-- The number of ice-cream cups ordered by Alok -/
def ice_cream_cups (chapatis rice mixed_veg : ℕ) 
  (chapati_cost rice_cost mixed_veg_cost ice_cream_cost total_paid : ℕ) : ℕ :=
  (total_paid - (chapatis * chapati_cost + rice * rice_cost + mixed_veg * mixed_veg_cost)) / ice_cream_cost

/-- Theorem stating that Alok ordered 6 ice-cream cups -/
theorem alok_ice_cream_order : 
  ice_cream_cups 16 5 7 6 45 70 40 1051 = 6 := by
  sorry

end alok_ice_cream_order_l1215_121544


namespace problem_2_l1215_121514

theorem problem_2 (a : ℤ) (h : a = 67897) : a * (a + 1) - (a - 1) * (a + 2) = 2 := by
  sorry

end problem_2_l1215_121514


namespace equation_graph_is_axes_l1215_121547

/-- The set of points satisfying (x+y)^2 = x^2 + y^2 is equivalent to the union of the x-axis and y-axis -/
theorem equation_graph_is_axes (x y : ℝ) : 
  (x + y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end equation_graph_is_axes_l1215_121547
