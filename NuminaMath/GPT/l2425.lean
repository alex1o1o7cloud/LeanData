import Mathlib

namespace renaming_not_unnoticeable_l2425_242570

-- Define the conditions as necessary structures for cities and connections
structure City := (name : String)
structure Connection := (city1 city2 : City)

-- Definition of the king's list of connections
def kingList : List Connection := sorry  -- The complete list of connections

-- The renaming function represented generically
def rename (c1 c2 : City) : City := sorry  -- The renaming function which is unspecified here

-- The main theorem statement
noncomputable def renaming_condition (c1 c2 : City) : Prop :=
  -- This condition represents that renaming preserves the king's perception of connections
  ∀ c : City, sorry  -- The specific condition needs full details of renaming logic

-- The theorem to prove, which states that the renaming is not always unnoticeable
theorem renaming_not_unnoticeable : ∃ c1 c2 : City, ¬ renaming_condition c1 c2 := sorry

end renaming_not_unnoticeable_l2425_242570


namespace xiao_ming_valid_paths_final_valid_paths_l2425_242506

-- Definitions from conditions
def paths_segments := ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
def initial_paths := 256
def invalid_paths := 64

-- Theorem statement
theorem xiao_ming_valid_paths : initial_paths - invalid_paths = 192 :=
by sorry

theorem final_valid_paths : 192 * 2 = 384 :=
by sorry

end xiao_ming_valid_paths_final_valid_paths_l2425_242506


namespace expected_defective_chips_in_60000_l2425_242520

def shipmentS1 := (2, 5000)
def shipmentS2 := (4, 12000)
def shipmentS3 := (2, 15000)
def shipmentS4 := (4, 16000)

def total_defective_chips := shipmentS1.1 + shipmentS2.1 + shipmentS3.1 + shipmentS4.1
def total_chips := shipmentS1.2 + shipmentS2.2 + shipmentS3.2 + shipmentS4.2

def defective_ratio := total_defective_chips / total_chips
def shipment60000 := 60000

def expected_defectives (ratio : ℝ) (total_chips : ℝ) := ratio * total_chips

theorem expected_defective_chips_in_60000 :
  expected_defectives defective_ratio shipment60000 = 15 :=
by
  sorry

end expected_defective_chips_in_60000_l2425_242520


namespace dot_product_of_a_and_c_is_4_l2425_242507

def vector := (ℝ × ℝ)

def a : vector := (1, -2)
def b : vector := (-3, 2)

def three_a : vector := (3 * 1, 3 * -2)
def two_b_minus_a : vector := (2 * -3 - 1, 2 * 2 - -2)

def c : vector := (-(-three_a.fst + two_b_minus_a.fst), -(-three_a.snd + two_b_minus_a.snd))

def dot_product (u v : vector) : ℝ := u.fst * v.fst + u.snd * v.snd

theorem dot_product_of_a_and_c_is_4 : dot_product a c = 4 := 
by
  sorry

end dot_product_of_a_and_c_is_4_l2425_242507


namespace no_partition_exists_l2425_242581

noncomputable section

open Set

def partition_N (A B C : Set ℕ) : Prop := 
  A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧  -- Non-empty sets
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧  -- Disjoint sets
  A ∪ B ∪ C = univ ∧  -- Covers the whole ℕ
  (∀ a ∈ A, ∀ b ∈ B, a + b + 2008 ∈ C) ∧
  (∀ b ∈ B, ∀ c ∈ C, b + c + 2008 ∈ A) ∧
  (∀ c ∈ C, ∀ a ∈ A, c + a + 2008 ∈ B)

theorem no_partition_exists : ¬ ∃ (A B C : Set ℕ), partition_N A B C :=
by
  sorry

end no_partition_exists_l2425_242581


namespace sum_abcd_l2425_242582

variable (a b c d x : ℝ)

axiom eq1 : a + 2 = x
axiom eq2 : b + 3 = x
axiom eq3 : c + 4 = x
axiom eq4 : d + 5 = x
axiom eq5 : a + b + c + d + 10 = x

theorem sum_abcd : a + b + c + d = -26 / 3 :=
by
  -- We state the condition given in the problem
  sorry

end sum_abcd_l2425_242582


namespace white_tshirts_per_package_l2425_242552

theorem white_tshirts_per_package (p t : ℕ) (h1 : p = 28) (h2 : t = 56) :
  t / p = 2 :=
by 
  sorry

end white_tshirts_per_package_l2425_242552


namespace plane_equation_correct_l2425_242569

-- Define points A, B, and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := { x := 1, y := -1, z := 8 }
def B : Point3D := { x := -4, y := -3, z := 10 }
def C : Point3D := { x := -1, y := -1, z := 7 }

-- Define the vector BC
def vecBC (B C : Point3D) : Point3D :=
  { x := C.x - B.x, y := C.y - B.y, z := C.z - B.z }

-- Define the equation of the plane
def planeEquation (P : Point3D) (normal : Point3D) : ℝ × ℝ × ℝ × ℝ :=
  (normal.x, normal.y, normal.z, -(normal.x * P.x + normal.y * P.y + normal.z * P.z))

-- Calculate the equation of the plane passing through A and perpendicular to vector BC
def planeThroughAperpToBC : ℝ × ℝ × ℝ × ℝ :=
  let normal := vecBC B C
  planeEquation A normal

-- The expected result
def expectedPlaneEquation : ℝ × ℝ × ℝ × ℝ := (3, 2, -3, 23)

-- The theorem to be proved
theorem plane_equation_correct : planeThroughAperpToBC = expectedPlaneEquation := by
  sorry

end plane_equation_correct_l2425_242569


namespace circle_diameter_l2425_242518

theorem circle_diameter (A : ℝ) (hA : A = 25 * π) (r : ℝ) (h : A = π * r^2) : 2 * r = 10 := by
  sorry

end circle_diameter_l2425_242518


namespace max_distance_between_P_and_Q_l2425_242505

-- Definitions of the circle and ellipse
def is_on_circle (P : ℝ × ℝ) : Prop := P.1^2 + (P.2 - 6)^2 = 2
def is_on_ellipse (Q : ℝ × ℝ) : Prop := (Q.1^2) / 10 + Q.2^2 = 1

-- The maximum distance between any point on the circle and any point on the ellipse
theorem max_distance_between_P_and_Q :
  ∃ P Q : ℝ × ℝ, is_on_circle P ∧ is_on_ellipse Q ∧ dist P Q = 6 * Real.sqrt 2 :=
sorry

end max_distance_between_P_and_Q_l2425_242505


namespace area_of_circle_with_given_circumference_l2425_242513

-- Defining the given problem's conditions as variables
variables (C : ℝ) (r : ℝ) (A : ℝ)
  
-- The condition that circumference is 12π meters
def circumference_condition : Prop := C = 12 * Real.pi
  
-- The relationship between circumference and radius
def radius_relationship : Prop := C = 2 * Real.pi * r
  
-- The formula to calculate the area of the circle
def area_formula : Prop := A = Real.pi * r^2
  
-- The proof goal that we need to establish
theorem area_of_circle_with_given_circumference :
  circumference_condition C ∧ radius_relationship C r ∧ area_formula A r → A = 36 * Real.pi :=
by
  intros
  sorry -- Skipping the proof, to be done later

end area_of_circle_with_given_circumference_l2425_242513


namespace parabola_tangent_line_l2425_242595

theorem parabola_tangent_line (a : ℝ) : 
  (∀ x : ℝ, (y = ax^2 + 6 ↔ y = x)) → a = 1 / 24 :=
by
  sorry

end parabola_tangent_line_l2425_242595


namespace apple_multiple_l2425_242530

theorem apple_multiple (K Ka : ℕ) (M : ℕ) 
  (h1 : K + Ka = 340)
  (h2 : Ka = M * K + 10)
  (h3 : Ka = 274) : 
  M = 4 := 
by
  sorry

end apple_multiple_l2425_242530


namespace simplify_expression_l2425_242528

theorem simplify_expression (x : ℝ) : 
  3 - 5 * x - 7 * x^2 + 9 - 11 * x + 13 * x^2 - 15 + 17 * x + 19 * x^2 = 25 * x^2 + x - 3 := 
by
  sorry

end simplify_expression_l2425_242528


namespace total_population_l2425_242584

theorem total_population (b g t : ℕ) (h1 : b = 2 * g) (h2 : g = 4 * t) : b + g + t = 13 * t :=
by
  sorry

end total_population_l2425_242584


namespace domain_of_f_parity_of_f_range_of_f_l2425_242536

noncomputable def f (a x : ℝ) := Real.log (1 + x) / Real.log a - Real.log (1 - x) / Real.log a

variables {a x : ℝ}

-- The properties derived:
theorem domain_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  (-1 < x ∧ x < 1) ↔ ∃ y, f a x = y :=
sorry

theorem parity_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  f a (-x) = -f a x :=
sorry

theorem range_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  (f a x > 0 ↔ (a > 1 ∧ 0 < x ∧ x < 1) ∨ (0 < a ∧ a < 1 ∧ -1 < x ∧ x < 0)) :=
sorry

end domain_of_f_parity_of_f_range_of_f_l2425_242536


namespace solve_abs_inequality_l2425_242542

theorem solve_abs_inequality (x : ℝ) : 
  (3 ≤ abs (x + 2) ∧ abs (x + 2) ≤ 6) ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := 
by sorry

end solve_abs_inequality_l2425_242542


namespace gcd_of_powers_l2425_242577

theorem gcd_of_powers (m n : ℕ) (h1 : m = 2^2016 - 1) (h2 : n = 2^2008 - 1) : 
  Nat.gcd m n = 255 :=
by
  -- (Definitions and steps are omitted as only the statement is required)
  sorry

end gcd_of_powers_l2425_242577


namespace sum_of_numbers_l2425_242551

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 12) (h2 : (1 / x) = 3 * (1 / y)) :
  x + y = 8 :=
sorry

end sum_of_numbers_l2425_242551


namespace math_problem_l2425_242521

theorem math_problem : 2 + 5 * 4 - 6 + 3 = 19 := by
  sorry

end math_problem_l2425_242521


namespace min_value_of_2a_b_c_l2425_242535

-- Given conditions
variables (a b c : ℝ)
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (h : a * (a + b + c) + b * c = 4 + 2 * Real.sqrt 3)

-- Question to prove
theorem min_value_of_2a_b_c : 2 * a + b + c = 2 * Real.sqrt 3 + 2 :=
sorry

end min_value_of_2a_b_c_l2425_242535


namespace clocks_resynchronize_after_days_l2425_242579

/-- Arthur's clock gains 15 minutes per day. -/
def arthurs_clock_gain_per_day : ℕ := 15

/-- Oleg's clock gains 12 minutes per day. -/
def olegs_clock_gain_per_day : ℕ := 12

/-- The clocks display time in a 12-hour format, which is equivalent to 720 minutes. -/
def twelve_hour_format_in_minutes : ℕ := 720

/-- 
  After how many days will this situation first repeat given the 
  conditions of gain in Arthur's and Oleg's clocks and the 12-hour format.
-/
theorem clocks_resynchronize_after_days :
  ∃ (N : ℕ), N * arthurs_clock_gain_per_day % twelve_hour_format_in_minutes = 0 ∧
             N * olegs_clock_gain_per_day % twelve_hour_format_in_minutes = 0 ∧
             N = 240 :=
by
  sorry

end clocks_resynchronize_after_days_l2425_242579


namespace negation_of_implication_iff_l2425_242515

variable (a : ℝ)

theorem negation_of_implication_iff (p : a > 1 → a^2 > 1) :
  ¬(a > 1 → a^2 > 1) ↔ (a ≤ 1 → a^2 ≤ 1) :=
by sorry

end negation_of_implication_iff_l2425_242515


namespace factorize_polynomial_l2425_242500

theorem factorize_polynomial (x y : ℝ) :
  3 * x ^ 2 + 6 * x * y + 3 * y ^ 2 = 3 * (x + y) ^ 2 :=
by
  sorry

end factorize_polynomial_l2425_242500


namespace most_lines_of_symmetry_l2425_242512

def regular_pentagon_lines_of_symmetry : ℕ := 5
def kite_lines_of_symmetry : ℕ := 1
def regular_hexagon_lines_of_symmetry : ℕ := 6
def isosceles_triangle_lines_of_symmetry : ℕ := 1
def scalene_triangle_lines_of_symmetry : ℕ := 0

theorem most_lines_of_symmetry :
  regular_hexagon_lines_of_symmetry = max
    (max (max (max regular_pentagon_lines_of_symmetry kite_lines_of_symmetry)
              regular_hexagon_lines_of_symmetry)
        isosceles_triangle_lines_of_symmetry)
    scalene_triangle_lines_of_symmetry :=
sorry

end most_lines_of_symmetry_l2425_242512


namespace diana_apollo_probability_l2425_242564

theorem diana_apollo_probability :
  let outcomes := (6 * 6)
  let successful := (5 + 4 + 3 + 2 + 1)
  (successful / outcomes) = 5 / 12 := sorry

end diana_apollo_probability_l2425_242564


namespace sum_of_abs_of_coefficients_l2425_242537

theorem sum_of_abs_of_coefficients :
  ∃ a_0 a_2 a_4 a_1 a_3 a_5 : ℤ, 
    ((2*x - 1)^5 + (x + 2)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) ∧
    (|a_0| + |a_2| + |a_4| = 110) :=
by
  sorry

end sum_of_abs_of_coefficients_l2425_242537


namespace correct_exponentiation_incorrect_division_incorrect_multiplication_incorrect_addition_l2425_242526

theorem correct_exponentiation (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

-- Incorrect options for clarity
theorem incorrect_division (a : ℝ) : a^6 / a^2 ≠ a^3 :=
by sorry

theorem incorrect_multiplication (a : ℝ) : a^2 * a^3 ≠ a^6 :=
by sorry

theorem incorrect_addition (a : ℝ) : (a^2 + a^3) ≠ a^5 :=
by sorry

end correct_exponentiation_incorrect_division_incorrect_multiplication_incorrect_addition_l2425_242526


namespace independent_variable_range_l2425_242586

theorem independent_variable_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) ↔ x ≥ 0 ∧ x ≠ 1 := 
by
  sorry

end independent_variable_range_l2425_242586


namespace intervals_of_monotonic_increase_max_area_acute_triangle_l2425_242527

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ :=
  (sin x, (sqrt 3 / 2) * (sin x - cos x))

noncomputable def vector_b (x : ℝ) : ℝ × ℝ :=
  (cos x, sin x + cos x)

noncomputable def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  a.1 * b.1 + a.2 * b.2

-- Problem 1: Proving the intervals of monotonic increase for the function f(x)
theorem intervals_of_monotonic_increase :
  ∀ k : ℤ, ∀ x : ℝ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) →
  ∀ x₁ x₂ : ℝ, (k * π - π / 12 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ k * π + 5 * π / 12) → f x₁ ≤ f x₂ :=
sorry

-- Problem 2: Proving the maximum area of triangle ABC
theorem max_area_acute_triangle (A : ℝ) (a b c : ℝ) :
  (f A = 1 / 2) → (a = sqrt 2) →
  ∀ S : ℝ, S ≤ (1 + sqrt 2) / 2 :=
sorry

end intervals_of_monotonic_increase_max_area_acute_triangle_l2425_242527


namespace problem_statement_l2425_242567

noncomputable def P1 (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def P2 (β : ℝ) : ℝ × ℝ := (Real.cos β, -Real.sin β)
noncomputable def P3 (α β : ℝ) : ℝ × ℝ := (Real.cos (α + β), Real.sin (α + β))
noncomputable def A : ℝ × ℝ := (1, 0)

theorem problem_statement (α β : ℝ) :
  (Prod.fst (P1 α))^2 + (Prod.snd (P1 α))^2 = 1 ∧
  (Prod.fst (P2 β))^2 + (Prod.snd (P2 β))^2 = 1 ∧
  (Prod.fst (P1 α) * Prod.fst (P2 β) + Prod.snd (P1 α) * Prod.snd (P2 β)) = Real.cos (α + β) :=
by
  sorry

end problem_statement_l2425_242567


namespace value_of_a_plus_b_l2425_242566

variables (a b : ℝ)

theorem value_of_a_plus_b (ha : abs a = 1) (hb : abs b = 4) (hab : a * b < 0) : a + b = 3 ∨ a + b = -3 := by
  sorry

end value_of_a_plus_b_l2425_242566


namespace gcd_of_polynomials_l2425_242545

theorem gcd_of_polynomials (b : ℤ) (k : ℤ) (hk : k % 2 = 0) (hb : b = 1187 * k) : 
  Int.gcd (2 * b^2 + 31 * b + 67) (b + 15) = 1 :=
by 
  sorry

end gcd_of_polynomials_l2425_242545


namespace initial_saltwater_amount_l2425_242540

variable (x y : ℝ)
variable (h1 : 0.04 * x = (x - y) * 0.1)
variable (h2 : ((x - y) * 0.1 + 300 * 0.04) / (x - y + 300) = 0.064)

theorem initial_saltwater_amount : x = 500 :=
by
  sorry

end initial_saltwater_amount_l2425_242540


namespace sum_gt_two_l2425_242510

noncomputable def f (x : ℝ) : ℝ := ((x - 1) * Real.log x) / x

theorem sum_gt_two (x₁ x₂ : ℝ) (h₁ : f x₁ = f x₂) (h₂ : x₁ ≠ x₂) : x₁ + x₂ > 2 := 
sorry

end sum_gt_two_l2425_242510


namespace total_wax_required_l2425_242525

/-- Given conditions: -/
def wax_already_have : ℕ := 331
def wax_needed_more : ℕ := 22

/-- Prove the question (the total amount of wax required) -/
theorem total_wax_required :
  (wax_already_have + wax_needed_more) = 353 := by
  sorry

end total_wax_required_l2425_242525


namespace speed_of_current_l2425_242541

theorem speed_of_current (upstream_time : ℝ) (downstream_time : ℝ) :
    upstream_time = 25 / 60 ∧ downstream_time = 12 / 60 →
    ( (60 / downstream_time - 60 / upstream_time) / 2 ) = 1.3 :=
by
  -- Introduce the conditions
  intro h
  -- Simplify using given facts
  have h1 := h.1
  have h2 := h.2
  -- Calcuation of the speed of current
  sorry

end speed_of_current_l2425_242541


namespace implication_equivalence_l2425_242532

variable (P Q : Prop)

theorem implication_equivalence :
  (¬Q → ¬P) ∧ (¬P ∨ Q) ↔ (P → Q) :=
by sorry

end implication_equivalence_l2425_242532


namespace range_of_m_l2425_242560

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (x^2 - 4*|x| + 5 - m = 0) → (∃ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)) → (1 < m ∧ m < 5) :=
by
  sorry

end range_of_m_l2425_242560


namespace available_spaces_l2425_242576

noncomputable def numberOfBenches : ℕ := 50
noncomputable def capacityPerBench : ℕ := 4
noncomputable def peopleSeated : ℕ := 80

theorem available_spaces :
  let totalCapacity := numberOfBenches * capacityPerBench;
  let availableSpaces := totalCapacity - peopleSeated;
  availableSpaces = 120 := by
    sorry

end available_spaces_l2425_242576


namespace solve_problem1_solve_problem2_l2425_242543

-- Problem 1
theorem solve_problem1 (x : ℚ) : (3 * x - 1) ^ 2 = 9 ↔ x = 4 / 3 ∨ x = -2 / 3 := 
by sorry

-- Problem 2
theorem solve_problem2 (x : ℚ) : x * (2 * x - 4) = (2 - x) ^ 2 ↔ x = 2 ∨ x = -2 :=
by sorry

end solve_problem1_solve_problem2_l2425_242543


namespace expand_product_l2425_242585

noncomputable def a (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
noncomputable def b (x : ℝ) : ℝ := x^2 + x + 3

theorem expand_product (x : ℝ) : (a x) * (b x) = 2 * x^4 - x^3 + 4 * x^2 - 8 * x + 3 :=
by
  sorry

end expand_product_l2425_242585


namespace we_the_people_cows_l2425_242580

theorem we_the_people_cows (W : ℕ) (h1 : ∃ H : ℕ, H = 3 * W + 2) (h2 : W + 3 * W + 2 = 70) : W = 17 :=
sorry

end we_the_people_cows_l2425_242580


namespace choir_members_max_l2425_242539

theorem choir_members_max (x r m : ℕ) 
  (h1 : r * x + 3 = m)
  (h2 : (r - 3) * (x + 2) = m) 
  (h3 : m < 150) : 
  m = 759 :=
sorry

end choir_members_max_l2425_242539


namespace part1_part2_l2425_242558

section 
variable {a b : ℚ}

-- Define the new operation as given in the condition
def odot (a b : ℚ) : ℚ := a * (a + b) - 1

-- Prove the given results
theorem part1 : odot 3 (-2) = 2 :=
by
  -- Proof omitted
  sorry

theorem part2 : odot (-2) (odot 3 5) = -43 :=
by
  -- Proof omitted
  sorry

end

end part1_part2_l2425_242558


namespace joan_has_6_balloons_l2425_242548

theorem joan_has_6_balloons (initial_balloons : ℕ) (lost_balloons : ℕ) (h1 : initial_balloons = 8) (h2 : lost_balloons = 2) : initial_balloons - lost_balloons = 6 :=
sorry

end joan_has_6_balloons_l2425_242548


namespace no_overlapping_sale_days_l2425_242563

def bookstore_sale_days (d : ℕ) : Prop :=
  d % 4 = 0 ∧ 1 ≤ d ∧ d ≤ 31

def shoe_store_sale_days (d : ℕ) : Prop :=
  ∃ k : ℕ, d = 2 + 8 * k ∧ 1 ≤ d ∧ d ≤ 31

theorem no_overlapping_sale_days : 
  ∀ d : ℕ, bookstore_sale_days d → ¬ shoe_store_sale_days d :=
by
  intros d h1 h2
  sorry

end no_overlapping_sale_days_l2425_242563


namespace minimum_cubes_required_l2425_242592

def box_length := 12
def box_width := 16
def box_height := 6
def cube_volume := 3

def volume_box := box_length * box_width * box_height

theorem minimum_cubes_required : volume_box / cube_volume = 384 := by
  sorry

end minimum_cubes_required_l2425_242592


namespace total_students_in_lunchroom_l2425_242571

theorem total_students_in_lunchroom :
  (34 * 6) + 15 = 219 :=
by
  sorry

end total_students_in_lunchroom_l2425_242571


namespace find_fz_l2425_242568

noncomputable def v (x y : ℝ) : ℝ :=
  3^x * Real.sin (y * Real.log 3)

theorem find_fz (x y : ℝ) (C : ℂ) (z : ℂ) (hz : z = x + y * Complex.I) :
  ∃ f : ℂ → ℂ, f z = 3^z + C :=
by
  sorry

end find_fz_l2425_242568


namespace inequality_range_m_l2425_242550

theorem inequality_range_m:
  (∀ x ∈ Set.Icc (Real.sqrt 2) 4, (5 / 2) * x^2 ≥ m * (x - 1)) → m ≤ 10 :=
by 
  intros h 
  sorry

end inequality_range_m_l2425_242550


namespace cone_lateral_surface_area_ratio_l2425_242562

theorem cone_lateral_surface_area_ratio (r l S_lateral S_base : ℝ) (h1 : l = 3 * r)
  (h2 : S_lateral = π * r * l) (h3 : S_base = π * r^2) :
  S_lateral / S_base = 3 :=
by
  sorry

end cone_lateral_surface_area_ratio_l2425_242562


namespace evaluate_f_g3_l2425_242553

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 1
def g (x : ℝ) : ℝ := x + 3

theorem evaluate_f_g3 : f (g 3) = 97 := by
  sorry

end evaluate_f_g3_l2425_242553


namespace frac_sum_eq_one_l2425_242511

variable {x y : ℝ}

theorem frac_sum_eq_one (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : (1 / x) + (1 / y) = 1 :=
by sorry

end frac_sum_eq_one_l2425_242511


namespace fraction_subtraction_simplify_l2425_242554

theorem fraction_subtraction_simplify :
  (9 / 19 - 3 / 57 - 1 / 3) = 5 / 57 :=
by
  sorry

end fraction_subtraction_simplify_l2425_242554


namespace union_eq_interval_l2425_242502

def A := { x : ℝ | 1 < x ∧ x < 4 }
def B := { x : ℝ | (x - 3) * (x + 1) ≤ 0 }

theorem union_eq_interval : (A ∪ B) = { x : ℝ | -1 ≤ x ∧ x < 4 } :=
by
  sorry

end union_eq_interval_l2425_242502


namespace cake_volume_icing_area_sum_l2425_242599

-- Define the conditions based on the problem description
def cube_edge_length : ℕ := 4
def volume_of_piece := 16
def icing_area := 12

-- Define the statements to be proven
theorem cake_volume_icing_area_sum : 
  volume_of_piece + icing_area = 28 := 
sorry

end cake_volume_icing_area_sum_l2425_242599


namespace ratio_expression_value_l2425_242572

theorem ratio_expression_value (x y : ℝ) (h : x ≠ 0) (h' : y ≠ 0) (h_eq : x^2 - y^2 = x + y) : 
  x / y + y / x = 2 + 1 / (y^2 + y) :=
by
  sorry

end ratio_expression_value_l2425_242572


namespace resort_total_cost_l2425_242509

noncomputable def first_cabin_cost (P : ℝ) := P
noncomputable def second_cabin_cost (P : ℝ) := (1/2) * P
noncomputable def third_cabin_cost (P : ℝ) := (1/6) * P
noncomputable def land_cost (P : ℝ) := 4 * P
noncomputable def pool_cost (P : ℝ) := P

theorem resort_total_cost (P : ℝ) (h : P = 22500) :
  first_cabin_cost P + pool_cost P + second_cabin_cost P + third_cabin_cost P + land_cost P = 150000 :=
by
  sorry

end resort_total_cost_l2425_242509


namespace evaluate_expression_l2425_242557

theorem evaluate_expression : (527 * 527 - 526 * 528) = 1 := by
  sorry

end evaluate_expression_l2425_242557


namespace trajectory_of_midpoint_l2425_242598

-- Definitions based on the conditions identified in the problem
variables {x y x1 y1 : ℝ}

-- Condition that point P is on the curve y = 2x^2 + 1
def point_on_curve (x1 y1 : ℝ) : Prop :=
  y1 = 2 * x1^2 + 1

-- Definition of the midpoint M conditions
def midpoint_def (x y x1 y1 : ℝ) : Prop :=
  x = (x1 + 0) / 2 ∧ y = (y1 - 1) / 2

-- Final theorem statement to be proved
theorem trajectory_of_midpoint (x y x1 y1 : ℝ) :
  point_on_curve x1 y1 → midpoint_def x y x1 y1 → y = 4 * x^2 :=
sorry

end trajectory_of_midpoint_l2425_242598


namespace remainder_of_P_div_D_is_25158_l2425_242590

noncomputable def P (x : ℝ) := 4 * x^8 - 2 * x^6 + 5 * x^4 - x^3 + 3 * x - 15
def D (x : ℝ) := 2 * x - 6

theorem remainder_of_P_div_D_is_25158 : P 3 = 25158 := by
  sorry

end remainder_of_P_div_D_is_25158_l2425_242590


namespace yoongi_has_fewer_apples_l2425_242503

-- Define the number of apples Jungkook originally has and receives more.
def jungkook_original_apples := 6
def jungkook_received_apples := 3

-- Calculate the total number of apples Jungkook has.
def jungkook_total_apples := jungkook_original_apples + jungkook_received_apples

-- Define the number of apples Yoongi has.
def yoongi_apples := 4

-- State that Yoongi has fewer apples than Jungkook.
theorem yoongi_has_fewer_apples : yoongi_apples < jungkook_total_apples := by
  sorry

end yoongi_has_fewer_apples_l2425_242503


namespace inequality_solution_l2425_242501

theorem inequality_solution (x : ℝ) :
  (x < -2 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) ∨ 7 < x) →
  (1 / (x - 1)) - (4 / (x - 2)) + (4 / (x - 3)) - (1 / (x - 4)) < 1 / 30 :=
by
  sorry

end inequality_solution_l2425_242501


namespace gcd_390_455_546_l2425_242524

theorem gcd_390_455_546 : Nat.gcd (Nat.gcd 390 455) 546 = 13 := 
by
  sorry    -- this indicates the proof is not included

end gcd_390_455_546_l2425_242524


namespace total_cost_of_water_l2425_242574

-- Define conditions in Lean 4
def cost_per_liter : ℕ := 1
def liters_per_bottle : ℕ := 2
def number_of_bottles : ℕ := 6

-- Define the theorem to prove the total cost
theorem total_cost_of_water : (number_of_bottles * (liters_per_bottle * cost_per_liter)) = 12 :=
by
  sorry

end total_cost_of_water_l2425_242574


namespace find_num_yoYos_l2425_242516

variables (x y z w : ℕ)

def stuffed_animals_frisbees_puzzles := x + y + w = 80
def total_prizes := x + y + z + w + 180 + 60
def cars_and_robots := 180 + 60 = x + y + z + w + 15

theorem find_num_yoYos 
(h1 : stuffed_animals_frisbees_puzzles x y w)
(h2 : total_prizes = 300)
(h3 : cars_and_robots x y z w) : z = 145 :=
sorry

end find_num_yoYos_l2425_242516


namespace rs_value_l2425_242549

theorem rs_value (r s : ℝ) (hr : 0 < r) (hs: 0 < s) (h1 : r^2 + s^2 = 1) (h2 : r^4 + s^4 = 3 / 4) :
  r * s = Real.sqrt 2 / 4 :=
sorry

end rs_value_l2425_242549


namespace find_b_l2425_242578

-- Define what it means for b to be a solution
def is_solution (b : ℤ) : Prop :=
  b > 4 ∧ ∃ k : ℤ, 4 * b + 5 = k * k

-- State the problem
theorem find_b : ∃ b : ℤ, is_solution b ∧ ∀ b' : ℤ, is_solution b' → b' ≥ 5 := by
  sorry

end find_b_l2425_242578


namespace unique_sequence_l2425_242517

theorem unique_sequence (a : ℕ → ℕ) (h_distinct: ∀ m n, a m = a n → m = n)
    (h_divisible: ∀ n, a n % a (a n) = 0) : ∀ n, a n = n :=
by
  -- proof goes here
  sorry

end unique_sequence_l2425_242517


namespace average_score_of_class_l2425_242555

theorem average_score_of_class (total_students : ℕ)
  (perc_assigned_day perc_makeup_day : ℝ)
  (average_assigned_day average_makeup_day : ℝ)
  (h_total : total_students = 100)
  (h_perc_assigned_day : perc_assigned_day = 0.70)
  (h_perc_makeup_day : perc_makeup_day = 0.30)
  (h_average_assigned_day : average_assigned_day = 55)
  (h_average_makeup_day : average_makeup_day = 95) :
  ((perc_assigned_day * total_students * average_assigned_day + perc_makeup_day * total_students * average_makeup_day) / total_students) = 67 := by
  sorry

end average_score_of_class_l2425_242555


namespace find_f_3_l2425_242594

def quadratic_function (b c : ℝ) (x : ℝ) : ℝ :=
- x^2 + b * x + c

theorem find_f_3 (b c : ℝ) (h1 : quadratic_function b c 2 + quadratic_function b c 4 = 12138)
                       (h2 : 3*b + c = 6079) :
  quadratic_function b c 3 = 6070 := 
by
  sorry

end find_f_3_l2425_242594


namespace no_real_roots_ffx_l2425_242597

theorem no_real_roots_ffx 
  (b c : ℝ) 
  (h : ∀ x : ℝ, (x^2 + (b - 1) * x + (c - 1) ≠ 0 ∨ ∀x: ℝ, (b - 1)^2 - 4 * (c - 1) < 0)) 
  : ∀ x : ℝ, (x^2 + bx + c)^2 + b * (x^2 + bx + c) + c ≠ x :=
by
  sorry

end no_real_roots_ffx_l2425_242597


namespace score_order_l2425_242596

variable (A B C D : ℕ)

theorem score_order
  (h1 : A + C = B + D)
  (h2 : B > D)
  (h3 : C > A + B) :
  C > B ∧ B > A ∧ A > D :=
by 
  sorry

end score_order_l2425_242596


namespace abs_diff_of_m_and_n_l2425_242522

theorem abs_diff_of_m_and_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 :=
sorry

end abs_diff_of_m_and_n_l2425_242522


namespace hamburgers_served_l2425_242587

def hamburgers_made : ℕ := 9
def hamburgers_leftover : ℕ := 6

theorem hamburgers_served : ∀ (total : ℕ) (left : ℕ), total = hamburgers_made → left = hamburgers_leftover → total - left = 3 := 
by
  intros total left h_total h_left
  rw [h_total, h_left]
  rfl

end hamburgers_served_l2425_242587


namespace infinite_sorted_subsequence_l2425_242508

theorem infinite_sorted_subsequence : 
  ∀ (warriors : ℕ → ℕ), (∀ n, ∃ m, m > n ∧ warriors m < warriors n) 
  ∨ (∃ k, warriors k = 0) → 
  ∃ (remaining : ℕ → ℕ), (∀ i j, i < j → remaining i > remaining j) :=
by
  intros warriors h
  sorry

end infinite_sorted_subsequence_l2425_242508


namespace p_pow_four_minus_one_divisible_by_ten_l2425_242556

theorem p_pow_four_minus_one_divisible_by_ten
  (p : Nat) (prime_p : Nat.Prime p) (h₁ : p ≠ 2) (h₂ : p ≠ 5) : 
  10 ∣ (p^4 - 1) := 
by
  sorry

end p_pow_four_minus_one_divisible_by_ten_l2425_242556


namespace intersecting_graphs_value_l2425_242529

theorem intersecting_graphs_value (a b c d : ℝ) 
  (h1 : 5 = -|2 - a| + b) 
  (h2 : 3 = -|8 - a| + b) 
  (h3 : 5 = |2 - c| + d) 
  (h4 : 3 = |8 - c| + d) : 
  a + c = 10 :=
sorry

end intersecting_graphs_value_l2425_242529


namespace lotus_leaves_not_odd_l2425_242588

theorem lotus_leaves_not_odd (n : ℕ) (h1 : n > 1) (h2 : ∀ t : ℕ, ∃ r : ℕ, 0 ≤ r ∧ r < n ∧ (t * (t + 1) / 2 - 1) % n = r) : ¬ Odd n :=
sorry

end lotus_leaves_not_odd_l2425_242588


namespace no_valid_base_l2425_242534

theorem no_valid_base (b : ℤ) (n : ℤ) : b^2 + 2*b + 2 ≠ n^2 := by
  sorry

end no_valid_base_l2425_242534


namespace single_elimination_matches_l2425_242519

theorem single_elimination_matches (n : ℕ) (h : n = 512) :
  ∃ (m : ℕ), m = n - 1 ∧ m = 511 :=
by
  sorry

end single_elimination_matches_l2425_242519


namespace number_of_bracelets_l2425_242589

-- Define the conditions as constants
def metal_beads_nancy := 40
def pearl_beads_nancy := 60
def crystal_beads_rose := 20
def stone_beads_rose := 40
def beads_per_bracelet := 2

-- Define the number of sets each person can make
def sets_of_metal_beads := metal_beads_nancy / beads_per_bracelet
def sets_of_pearl_beads := pearl_beads_nancy / beads_per_bracelet
def sets_of_crystal_beads := crystal_beads_rose / beads_per_bracelet
def sets_of_stone_beads := stone_beads_rose / beads_per_bracelet

-- Define the theorem to prove
theorem number_of_bracelets : min sets_of_metal_beads (min sets_of_pearl_beads (min sets_of_crystal_beads sets_of_stone_beads)) = 10 := by
  -- Placeholder for the proof
  sorry

end number_of_bracelets_l2425_242589


namespace joe_commute_time_l2425_242593

theorem joe_commute_time
  (d : ℝ) -- total one-way distance from home to school
  (rw : ℝ) -- Joe's walking rate
  (rr : ℝ := 4 * rw) -- Joe's running rate (4 times walking rate)
  (walking_time_for_one_third : ℝ := 9) -- Joe takes 9 minutes to walk one-third distance
  (walking_time_two_thirds : ℝ := 2 * walking_time_for_one_third) -- time to walk two-thirds distance
  (running_time_two_thirds : ℝ := walking_time_two_thirds / 4) -- time to run two-thirds 
  : (2 * walking_time_two_thirds + running_time_two_thirds) = 40.5 := -- total travel time
by
  sorry

end joe_commute_time_l2425_242593


namespace cos_of_acute_angle_l2425_242523

theorem cos_of_acute_angle (θ : ℝ) (hθ1 : 0 < θ ∧ θ < π / 2) (hθ2 : Real.sin θ = 1 / 3) :
  Real.cos θ = 2 * Real.sqrt 2 / 3 :=
by
  -- The proof steps will be filled here
  sorry

end cos_of_acute_angle_l2425_242523


namespace solve_equation1_solve_equation2_l2425_242547

theorem solve_equation1 (x : ℝ) : 3 * (x - 1)^3 = 24 ↔ x = 3 := by
  sorry

theorem solve_equation2 (x : ℝ) : (x - 3)^2 = 64 ↔ x = 11 ∨ x = -5 := by
  sorry

end solve_equation1_solve_equation2_l2425_242547


namespace sequence_values_l2425_242544

theorem sequence_values (x y z : ℕ) 
    (h1 : x = 14 * 3) 
    (h2 : y = x - 1) 
    (h3 : z = y * 3) : 
    x = 42 ∧ y = 41 ∧ z = 123 := by 
    sorry

end sequence_values_l2425_242544


namespace length_of_train_B_l2425_242591

-- Given conditions
def lengthTrainA := 125  -- in meters
def speedTrainA := 54    -- in km/hr
def speedTrainB := 36    -- in km/hr
def timeToCross := 11    -- in seconds

-- Conversion factor from km/hr to m/s
def kmhr_to_mps (v : ℕ) : ℕ := v * 5 / 18

-- Relative speed of the trains in m/s
def relativeSpeed := kmhr_to_mps (speedTrainA + speedTrainB)

-- Distance covered in the given time
def distanceCovered := relativeSpeed * timeToCross

-- Proof statement
theorem length_of_train_B : distanceCovered - lengthTrainA = 150 := 
by
  -- Proof will go here
  sorry

end length_of_train_B_l2425_242591


namespace sampling_is_simple_random_l2425_242565

-- Definitions based on conditions
def total_students := 200
def students_sampled := 20
def sampling_method := "Simple Random Sampling"

-- The problem: given the random sampling of 20 students from 200, prove that the method is simple random sampling.
theorem sampling_is_simple_random :
  (total_students = 200 ∧ students_sampled = 20) → sampling_method = "Simple Random Sampling" := 
by
  sorry

end sampling_is_simple_random_l2425_242565


namespace Carol_saves_9_per_week_l2425_242573

variable (C : ℤ)

def Carol_savings (weeks : ℤ) : ℤ :=
  60 + weeks * C

def Mike_savings (weeks : ℤ) : ℤ :=
  90 + weeks * 3

theorem Carol_saves_9_per_week (h : Carol_savings C 5 = Mike_savings 5) : C = 9 :=
by
  dsimp [Carol_savings, Mike_savings] at h
  sorry

end Carol_saves_9_per_week_l2425_242573


namespace ordered_pair_solution_l2425_242514

theorem ordered_pair_solution :
  ∃ x y : ℤ, (x + y = (3 - x) + (3 - y)) ∧ (x - y = (x - 2) + (y - 2)) ∧ (x = 2) ∧ (y = 1) :=
by
  use 2, 1
  repeat { sorry }

end ordered_pair_solution_l2425_242514


namespace original_ratio_white_yellow_l2425_242538

-- Define the given conditions
variables (W Y : ℕ)
axiom total_balls : W + Y = 64
axiom erroneous_dispatch : W = 8 * (Y + 20) / 13

-- The theorem we need to prove
theorem original_ratio_white_yellow (W Y : ℕ) (h1 : W + Y = 64) (h2 : W = 8 * (Y + 20) / 13) : W = Y :=
by sorry

end original_ratio_white_yellow_l2425_242538


namespace no_intersection_abs_functions_l2425_242546

open Real

theorem no_intersection_abs_functions : 
  ∀ f g : ℝ → ℝ, 
  (∀ x, f x = |2 * x + 5|) → 
  (∀ x, g x = -|3 * x - 2|) → 
  (∀ y, ∀ x1 x2, f x1 = y ∧ g x2 = y → y = 0 ∧ x1 = -5/2 ∧ x2 = 2/3 → (x1 ≠ x2)) → 
  (∃ x, f x = g x) → 
  false := 
  by
    intro f g hf hg h
    sorry

end no_intersection_abs_functions_l2425_242546


namespace degree_to_radian_60_eq_pi_div_3_l2425_242531

theorem degree_to_radian_60_eq_pi_div_3 (pi : ℝ) (deg : ℝ) 
  (h : 180 * deg = pi) : 60 * deg = pi / 3 := 
by
  sorry

end degree_to_radian_60_eq_pi_div_3_l2425_242531


namespace necessary_but_not_sufficient_for_x_gt_4_l2425_242583

theorem necessary_but_not_sufficient_for_x_gt_4 (x : ℝ) : (x^2 > 16) → ¬ (x > 4) :=
by
  sorry

end necessary_but_not_sufficient_for_x_gt_4_l2425_242583


namespace perfect_square_trinomial_l2425_242561

theorem perfect_square_trinomial (m : ℤ) : 
  (∃ x y : ℝ, 16 * x^2 + m * x * y + 25 * y^2 = (4 * x + 5 * y)^2 ∨ 16 * x^2 + m * x * y + 25 * y^2 = (4 * x - 5 * y)^2) ↔ (m = 40 ∨ m = -40) :=
by
  sorry

end perfect_square_trinomial_l2425_242561


namespace permissible_range_n_l2425_242575

theorem permissible_range_n (n x y m : ℝ) (hn : n ≤ x) (hxy : x < y) (hy : y ≤ n+1)
  (hm_in: x < m ∧ m < y) (habs_eq : |y| = |m| + |x|): 
  -1 < n ∧ n < 1 := sorry

end permissible_range_n_l2425_242575


namespace inequality_ineqs_l2425_242559

theorem inequality_ineqs (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_cond : x * y + y * z + z * x = 1) :
  (27 / 4) * (x + y) * (y + z) * (z + x) 
  ≥ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2
  ∧ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2 
  ≥ 
  6 * Real.sqrt 3 := by
  sorry

end inequality_ineqs_l2425_242559


namespace melted_ice_cream_depth_l2425_242504

noncomputable def ice_cream_depth : ℝ :=
  let r1 := 3 -- radius of the sphere
  let r2 := 10 -- radius of the cylinder
  let V_sphere := (4/3) * Real.pi * r1^3 -- volume of the sphere
  let V_cylinder h := Real.pi * r2^2 * h -- volume of the cylinder
  V_sphere / (Real.pi * r2^2)

theorem melted_ice_cream_depth :
  ice_cream_depth = 9 / 25 :=
by
  sorry

end melted_ice_cream_depth_l2425_242504


namespace negation_of_proposition_l2425_242533

theorem negation_of_proposition :
  ¬ (∃ x_0 : ℝ, 2^x_0 < x_0^2) ↔ (∀ x : ℝ, 2^x ≥ x^2) :=
by sorry

end negation_of_proposition_l2425_242533
