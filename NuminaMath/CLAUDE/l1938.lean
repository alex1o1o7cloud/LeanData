import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1938_193849

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = 2) : 
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1938_193849


namespace NUMINAMATH_CALUDE_assignment_statement_valid_l1938_193862

-- Define what constitutes a valid variable name
def IsValidVariableName (name : String) : Prop := name.length > 0 ∧ name.all Char.isAlpha

-- Define what constitutes a valid arithmetic expression
inductive ArithmeticExpression
  | Var : String → ArithmeticExpression
  | Num : Int → ArithmeticExpression
  | Add : ArithmeticExpression → ArithmeticExpression → ArithmeticExpression
  | Mul : ArithmeticExpression → ArithmeticExpression → ArithmeticExpression
  | Sub : ArithmeticExpression → ArithmeticExpression → ArithmeticExpression

-- Define what constitutes a valid assignment statement
structure AssignmentStatement where
  lhs : String
  rhs : ArithmeticExpression
  valid : IsValidVariableName lhs

-- The statement we want to prove
theorem assignment_statement_valid :
  ∃ (stmt : AssignmentStatement),
    stmt.lhs = "A" ∧
    stmt.rhs = ArithmeticExpression.Sub
      (ArithmeticExpression.Add
        (ArithmeticExpression.Mul
          (ArithmeticExpression.Var "A")
          (ArithmeticExpression.Var "A"))
        (ArithmeticExpression.Var "A"))
      (ArithmeticExpression.Num 3) :=
by sorry


end NUMINAMATH_CALUDE_assignment_statement_valid_l1938_193862


namespace NUMINAMATH_CALUDE_count_prime_differences_l1938_193867

def is_in_set (n : ℕ) : Prop := ∃ k : ℕ, n = 10 * k - 3 ∧ k ≥ 1

def is_prime_difference (n : ℕ) : Prop := ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p - q

theorem count_prime_differences : 
  (∃! (s : Finset ℕ), (∀ n ∈ s, is_in_set n ∧ is_prime_difference n) ∧ s.card = 2) :=
sorry

end NUMINAMATH_CALUDE_count_prime_differences_l1938_193867


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_line_l1938_193873

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_parallel_line 
  (α β : Plane) (m n : Line) 
  (h1 : α ≠ β) 
  (h2 : m ≠ n) 
  (h3 : perpendicular m α) 
  (h4 : parallel n α) : 
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_line_l1938_193873


namespace NUMINAMATH_CALUDE_fraction_chain_l1938_193889

theorem fraction_chain (x y z w : ℚ) 
  (h1 : x / y = 5)
  (h2 : y / z = 1 / 4)
  (h3 : z / w = 7) :
  w / x = 4 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_chain_l1938_193889


namespace NUMINAMATH_CALUDE_cube_planes_divide_space_into_27_parts_l1938_193842

/-- Represents a cube in 3D space -/
structure Cube where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a plane in 3D space -/
structure Plane where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Function to generate planes through each face of a cube -/
def planes_through_cube_faces (c : Cube) : List Plane :=
  sorry

/-- Function to count the number of parts the space is divided into by the planes -/
def count_divided_parts (planes : List Plane) : Nat :=
  sorry

/-- Theorem stating that planes through each face of a cube divide space into 27 parts -/
theorem cube_planes_divide_space_into_27_parts (c : Cube) :
  count_divided_parts (planes_through_cube_faces c) = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_planes_divide_space_into_27_parts_l1938_193842


namespace NUMINAMATH_CALUDE_squares_characterization_l1938_193819

class MyGroup (G : Type) extends Group G where
  g : G
  h : G
  g_four : g ^ 4 = 1
  g_two_ne_one : g ^ 2 ≠ 1
  h_seven : h ^ 7 = 1
  h_ne_one : h ≠ 1
  gh_relation : g * h * g⁻¹ * h = 1
  subgroup_condition : ∀ (H : Subgroup G), g ∈ H → h ∈ H → H = ⊤

variable {G : Type} [MyGroup G]

def squares (G : Type) [MyGroup G] : Set G :=
  {x : G | ∃ y : G, y ^ 2 = x}

theorem squares_characterization :
  squares G = {1, (MyGroup.g : G) ^ 2, MyGroup.h, MyGroup.h ^ 2, MyGroup.h ^ 3, MyGroup.h ^ 4, MyGroup.h ^ 5, MyGroup.h ^ 6} := by
  sorry

end NUMINAMATH_CALUDE_squares_characterization_l1938_193819


namespace NUMINAMATH_CALUDE_total_team_score_l1938_193885

def team_score (connor_score amy_score jason_score emily_score : ℕ) : ℕ :=
  connor_score + amy_score + jason_score + emily_score

theorem total_team_score :
  ∀ (connor_score amy_score jason_score emily_score : ℕ),
    connor_score = 2 →
    amy_score = connor_score + 4 →
    jason_score = 2 * amy_score →
    emily_score = 3 * (connor_score + amy_score + jason_score) →
    team_score connor_score amy_score jason_score emily_score = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_total_team_score_l1938_193885


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l1938_193841

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l1938_193841


namespace NUMINAMATH_CALUDE_probability_odd_divisor_15_factorial_l1938_193817

theorem probability_odd_divisor_15_factorial (n : ℕ) (h : n = 15) :
  let factorial := n.factorial
  let total_divisors := (factorial.divisors.filter (λ x => x > 0)).card
  let odd_divisors := (factorial.divisors.filter (λ x => x > 0 ∧ x % 2 ≠ 0)).card
  (odd_divisors : ℚ) / total_divisors = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_odd_divisor_15_factorial_l1938_193817


namespace NUMINAMATH_CALUDE_julia_monday_playmates_l1938_193896

/-- The number of kids Julia played with on different days -/
structure JuliaPlaymates where
  wednesday : ℕ
  monday : ℕ

/-- Given conditions about Julia's playmates -/
def julia_conditions (j : JuliaPlaymates) : Prop :=
  j.wednesday = 4 ∧ j.monday = j.wednesday + 2

/-- Theorem: Julia played with 6 kids on Monday -/
theorem julia_monday_playmates (j : JuliaPlaymates) (h : julia_conditions j) : j.monday = 6 := by
  sorry

end NUMINAMATH_CALUDE_julia_monday_playmates_l1938_193896


namespace NUMINAMATH_CALUDE_college_students_count_l1938_193876

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 135) :
  boys + girls = 351 :=
sorry

end NUMINAMATH_CALUDE_college_students_count_l1938_193876


namespace NUMINAMATH_CALUDE_exactly_one_zero_iff_m_eq_zero_or_nine_l1938_193813

/-- A quadratic function of the form y = mx² - 6x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 6 * x + 1

/-- The discriminant of the quadratic function f -/
def discriminant (m : ℝ) : ℝ := (-6)^2 - 4 * m * 1

/-- The function f has exactly one zero -/
def has_exactly_one_zero (m : ℝ) : Prop :=
  (m = 0 ∧ ∃! x, f m x = 0) ∨
  (m ≠ 0 ∧ discriminant m = 0)

theorem exactly_one_zero_iff_m_eq_zero_or_nine (m : ℝ) :
  has_exactly_one_zero m ↔ m = 0 ∨ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_zero_iff_m_eq_zero_or_nine_l1938_193813


namespace NUMINAMATH_CALUDE_number_puzzle_l1938_193888

theorem number_puzzle : ∃ x : ℝ, x / 3 = x - 36 ∧ x = 54 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1938_193888


namespace NUMINAMATH_CALUDE_seven_times_prime_divisors_l1938_193843

theorem seven_times_prime_divisors (p : ℕ) (h_prime : Nat.Prime p) :
  (Nat.divisors (7 * p)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_seven_times_prime_divisors_l1938_193843


namespace NUMINAMATH_CALUDE_h_one_value_l1938_193871

/-- A polynomial of degree 3 with constant coefficients -/
structure CubicPolynomial where
  p : ℝ
  q : ℝ
  r : ℝ
  h_order : p < q ∧ q < r

/-- The function f(x) = x^3 + px^2 + qx + r -/
def f (c : CubicPolynomial) (x : ℝ) : ℝ :=
  x^3 + c.p * x^2 + c.q * x + c.r

/-- A polynomial h(x) whose roots are the squares of the reciprocals of the roots of f(x) -/
def h (c : CubicPolynomial) (x : ℝ) : ℝ :=
  sorry  -- Definition of h(x) is not explicitly given in the problem

/-- Theorem stating the value of h(1) in terms of p, q, and r -/
theorem h_one_value (c : CubicPolynomial) :
  h c 1 = (1 - c.p + c.q - c.r) * (1 - c.q + c.p - c.r) * (1 - c.r + c.p - c.q) / c.r^2 :=
sorry

end NUMINAMATH_CALUDE_h_one_value_l1938_193871


namespace NUMINAMATH_CALUDE_building_entrances_l1938_193892

/-- Represents a multi-story building with apartments --/
structure Building where
  floors : ℕ
  apartments_per_floor : ℕ
  total_apartments : ℕ

/-- Calculates the number of entrances in a building --/
def number_of_entrances (b : Building) : ℕ :=
  b.total_apartments / (b.floors * b.apartments_per_floor)

/-- Theorem: A building with 9 floors, 4 apartments per floor, and 180 total apartments has 5 entrances --/
theorem building_entrances :
  let b : Building := ⟨9, 4, 180⟩
  number_of_entrances b = 5 := by
sorry

end NUMINAMATH_CALUDE_building_entrances_l1938_193892


namespace NUMINAMATH_CALUDE_weight_of_A_l1938_193820

def avg_weight_ABC : ℝ := 60
def avg_weight_ABCD : ℝ := 65
def avg_weight_BCDE : ℝ := 64
def weight_difference_E_D : ℝ := 3

theorem weight_of_A (weight_A weight_B weight_C weight_D weight_E : ℝ) : 
  (weight_A + weight_B + weight_C) / 3 = avg_weight_ABC ∧
  (weight_A + weight_B + weight_C + weight_D) / 4 = avg_weight_ABCD ∧
  weight_E = weight_D + weight_difference_E_D ∧
  (weight_B + weight_C + weight_D + weight_E) / 4 = avg_weight_BCDE →
  weight_A = 87 := by
sorry

end NUMINAMATH_CALUDE_weight_of_A_l1938_193820


namespace NUMINAMATH_CALUDE_no_general_rational_solution_l1938_193805

theorem no_general_rational_solution (k : ℚ) : 
  ¬ ∃ (S : Set ℝ), ∀ (x : ℝ), x ∈ S → 
    ∃ (q : ℚ), x + k * Real.sqrt (x^2 + 1) - 1 / (x + k * Real.sqrt (x^2 + 1)) = q :=
by sorry

end NUMINAMATH_CALUDE_no_general_rational_solution_l1938_193805


namespace NUMINAMATH_CALUDE_planes_intersect_l1938_193825

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (are_skew : Line → Line → Prop)
variable (is_perpendicular_to_plane : Line → Plane → Prop)
variable (are_intersecting : Plane → Plane → Prop)

-- State the theorem
theorem planes_intersect (a b : Line) (α β : Plane) 
  (h1 : are_skew a b)
  (h2 : is_perpendicular_to_plane a α)
  (h3 : is_perpendicular_to_plane b β) :
  are_intersecting α β :=
sorry

end NUMINAMATH_CALUDE_planes_intersect_l1938_193825


namespace NUMINAMATH_CALUDE_extreme_point_implies_zero_derivative_zero_derivative_not_always_extreme_point_l1938_193851

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define differentiability for f
variable (hf : Differentiable ℝ f)

-- Define what it means for a point to be an extreme point
def IsExtremePoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem extreme_point_implies_zero_derivative
  (x₀ : ℝ) (h_extreme : IsExtremePoint f x₀) :
  deriv f x₀ = 0 :=
sorry

-- State that the converse is not always true
theorem zero_derivative_not_always_extreme_point :
  ¬ (∀ (g : ℝ → ℝ) (hg : Differentiable ℝ g) (x₀ : ℝ),
    deriv g x₀ = 0 → IsExtremePoint g x₀) :=
sorry

end NUMINAMATH_CALUDE_extreme_point_implies_zero_derivative_zero_derivative_not_always_extreme_point_l1938_193851


namespace NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l1938_193837

theorem sum_of_special_primes_is_prime (C D : ℕ+) : 
  Prime C.val → Prime D.val → Prime (C.val - D.val) → Prime (C.val + D.val) →
  Prime (C.val + D.val + (C.val - D.val) + C.val + D.val) := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l1938_193837


namespace NUMINAMATH_CALUDE_derivative_at_one_l1938_193879

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x + x^3) :
  deriv f 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1938_193879


namespace NUMINAMATH_CALUDE_gcd_1237_1957_l1938_193811

theorem gcd_1237_1957 : Nat.gcd 1237 1957 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1237_1957_l1938_193811


namespace NUMINAMATH_CALUDE_special_triangle_properties_l1938_193814

/-- Triangle ABC with specific conditions -/
structure SpecialTriangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side length opposite to A
  b : ℝ  -- Side length opposite to B
  c : ℝ  -- Side length opposite to C
  angle_sum : A + B + C = π
  side_condition : a + c = 3 * Real.sqrt 3 / 2
  side_b : b = Real.sqrt 3
  angle_condition : 2 * Real.cos A * Real.cos C * (Real.tan A * Real.tan C - 1) = 1

/-- Theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  t.B = π / 3 ∧ (1 / 2 * t.a * t.c * Real.sin t.B = 5 * Real.sqrt 3 / 16) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l1938_193814


namespace NUMINAMATH_CALUDE_one_less_than_three_times_l1938_193883

/-- The number that is 1 less than 3 times a real number a can be expressed as 3a - 1. -/
theorem one_less_than_three_times (a : ℝ) : ∃ x : ℝ, x = 3 * a - 1 ∧ x + 1 = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_one_less_than_three_times_l1938_193883


namespace NUMINAMATH_CALUDE_scallops_per_pound_is_eight_l1938_193839

/-- The number of jumbo scallops that weigh one pound -/
def scallops_per_pound : ℕ := by sorry

/-- The cost of one pound of jumbo scallops in dollars -/
def cost_per_pound : ℕ := 24

/-- The number of scallops paired per person -/
def scallops_per_person : ℕ := 2

/-- The number of people Nate is cooking for -/
def number_of_people : ℕ := 8

/-- The total cost of scallops for Nate in dollars -/
def total_cost : ℕ := 48

theorem scallops_per_pound_is_eight :
  scallops_per_pound = 8 := by sorry

end NUMINAMATH_CALUDE_scallops_per_pound_is_eight_l1938_193839


namespace NUMINAMATH_CALUDE_bounded_difference_l1938_193856

theorem bounded_difference (x y z : ℝ) :
  x - z < y ∧ x + z > y → -z < x - y ∧ x - y < z := by
  sorry

end NUMINAMATH_CALUDE_bounded_difference_l1938_193856


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1938_193853

theorem abs_neg_three_eq_three : abs (-3 : ℤ) = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1938_193853


namespace NUMINAMATH_CALUDE_ellipse_sum_property_l1938_193858

/-- Properties of an ellipse -/
structure Ellipse where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  a : ℝ  -- length of semi-major axis
  b : ℝ  -- length of semi-minor axis

/-- Theorem about the sum of center coordinates and axis lengths for a specific ellipse -/
theorem ellipse_sum_property (E : Ellipse) 
  (center_x : E.h = 3) 
  (center_y : E.k = -5) 
  (major_axis : E.a = 6) 
  (minor_axis : E.b = 2) : 
  E.h + E.k + E.a + E.b = 6 := by
  sorry

#check ellipse_sum_property

end NUMINAMATH_CALUDE_ellipse_sum_property_l1938_193858


namespace NUMINAMATH_CALUDE_unique_number_between_cube_roots_l1938_193869

theorem unique_number_between_cube_roots : ∃! (n : ℕ),
  n > 0 ∧ 24 ∣ n ∧ (9 : ℝ) < n ^ (1/3) ∧ n ^ (1/3) < (9.1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_unique_number_between_cube_roots_l1938_193869


namespace NUMINAMATH_CALUDE_circle_symmetry_l1938_193810

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the property of symmetry
def is_symmetric (circle1 circle2 : (ℝ → ℝ → Prop)) (line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    circle1 x1 y1 ∧ 
    circle2 x2 y2 ∧ 
    line ((x1 + x2) / 2) ((y1 + y2) / 2)

-- Define our target circle
def target_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- The main theorem
theorem circle_symmetry :
  is_symmetric given_circle target_circle symmetry_line :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1938_193810


namespace NUMINAMATH_CALUDE_appliance_price_difference_l1938_193828

theorem appliance_price_difference : 
  let in_store_price : ℚ := 109.99
  let tv_payment : ℚ := 24.99
  let tv_shipping : ℚ := 14.98
  let tv_price : ℚ := 4 * tv_payment + tv_shipping
  (tv_price - in_store_price) * 100 = 495 := by sorry

end NUMINAMATH_CALUDE_appliance_price_difference_l1938_193828


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1938_193807

/-- Given a line y = x - 1 intersecting an ellipse (x^2 / a^2) + (y^2 / (a^2 - 1)) = 1 
    where a > 1, if the circle with diameter AB (where A and B are intersection points) 
    passes through the left focus of the ellipse, then a = (√6 + √2) / 2 -/
theorem ellipse_intersection_theorem (a : ℝ) (h_a : a > 1) :
  let line := fun x : ℝ => x - 1
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / (a^2 - 1) = 1
  let intersection_points := {p : ℝ × ℝ | ellipse p.1 p.2 ∧ p.2 = line p.1}
  let circle := fun (c : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) => 
    (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2
  ∃ (A B : ℝ × ℝ) (c : ℝ × ℝ) (r : ℝ), 
    A ∈ intersection_points ∧ 
    B ∈ intersection_points ∧
    A ≠ B ∧
    circle c r A ∧
    circle c r B ∧
    circle c r (-1, 0) →
  a = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1938_193807


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1938_193850

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1938_193850


namespace NUMINAMATH_CALUDE_next_235_time_91_minutes_l1938_193801

def is_valid_time (h m : ℕ) : Prop :=
  h < 24 ∧ m < 60

def uses_digits_235_once (h m : ℕ) : Prop :=
  let digits := h.digits 10 ++ m.digits 10
  digits.count 2 = 1 ∧ digits.count 3 = 1 ∧ digits.count 5 = 1

def minutes_from_352_to (h m : ℕ) : ℕ :=
  if h < 3 ∨ (h = 3 ∧ m ≤ 52) then
    (h + 24 - 3) * 60 + (m - 52)
  else
    (h - 3) * 60 + (m - 52)

theorem next_235_time_91_minutes :
  ∃ (h m : ℕ), 
    is_valid_time h m ∧
    uses_digits_235_once h m ∧
    minutes_from_352_to h m = 91 ∧
    (∀ (h' m' : ℕ), 
      is_valid_time h' m' →
      uses_digits_235_once h' m' →
      minutes_from_352_to h' m' ≥ 91) :=
sorry

end NUMINAMATH_CALUDE_next_235_time_91_minutes_l1938_193801


namespace NUMINAMATH_CALUDE_relations_correctness_l1938_193859

-- Define the relations
def relation1 (a b c : ℝ) : Prop := (a > b) ↔ (a * c^2 > b * c^2)
def relation2 (a b : ℝ) : Prop := (a > b) → (1/a < 1/b)
def relation3 (a b c d : ℝ) : Prop := (a > b ∧ b > 0 ∧ c > d) → (a/d > b/c)
def relation4 (a b c : ℝ) : Prop := (a > b ∧ b > 1 ∧ c < 0) → (a^c < b^c)

-- State the theorem
theorem relations_correctness :
  (∃ a b c : ℝ, ¬(relation1 a b c)) ∧
  (∃ a b : ℝ, ¬(relation2 a b)) ∧
  (∃ a b c d : ℝ, ¬(relation3 a b c d)) ∧
  (∀ a b c : ℝ, relation4 a b c) :=
sorry

end NUMINAMATH_CALUDE_relations_correctness_l1938_193859


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1938_193898

theorem sum_of_fractions : (1 : ℚ) / 1 + (2 : ℚ) / 2 + (3 : ℚ) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1938_193898


namespace NUMINAMATH_CALUDE_intersection_implies_value_l1938_193874

theorem intersection_implies_value (a : ℝ) : 
  let A : Set ℝ := {2, a - 1}
  let B : Set ℝ := {a^2 - 7, -1}
  A ∩ B = {2} → a = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_value_l1938_193874


namespace NUMINAMATH_CALUDE_max_distance_difference_l1938_193802

def circle_C1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

def circle_C2 (x y : ℝ) : Prop := (x + 3)^2 + (y - 4)^2 = 9

def on_x_axis (x y : ℝ) : Prop := y = 0

theorem max_distance_difference :
  ∃ (max : ℝ),
    (∀ (Mx My Nx Ny Px Py : ℝ),
      circle_C1 Mx My →
      circle_C2 Nx Ny →
      on_x_axis Px Py →
      Real.sqrt ((Nx - Px)^2 + (Ny - Py)^2) -
      Real.sqrt ((Mx - Px)^2 + (My - Py)^2) ≤ max) ∧
    max = 4 + Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_difference_l1938_193802


namespace NUMINAMATH_CALUDE_largest_multiple_12_negation_gt_neg150_l1938_193893

theorem largest_multiple_12_negation_gt_neg150 :
  ∀ n : ℤ, (12 ∣ n) → -n > -150 → n ≤ 144 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_12_negation_gt_neg150_l1938_193893


namespace NUMINAMATH_CALUDE_M_intersect_N_l1938_193848

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem M_intersect_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1938_193848


namespace NUMINAMATH_CALUDE_unique_set_satisfying_condition_l1938_193895

theorem unique_set_satisfying_condition :
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    (a * b * c % d = 1) ∧
    (a * b * d % c = 1) ∧
    (a * c * d % b = 1) ∧
    (b * c * d % a = 1) →
    ({a, b, c, d} : Set ℕ) = {1, 2, 3, 4} :=
by sorry

end NUMINAMATH_CALUDE_unique_set_satisfying_condition_l1938_193895


namespace NUMINAMATH_CALUDE_number_of_possible_lists_l1938_193826

def num_balls : ℕ := 15
def list_length : ℕ := 4

theorem number_of_possible_lists :
  (num_balls ^ list_length : ℕ) = 50625 := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_lists_l1938_193826


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l1938_193816

theorem square_minus_product_plus_square : 7^2 - 4*5 + 2^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l1938_193816


namespace NUMINAMATH_CALUDE_power_of_81_three_fourths_l1938_193809

theorem power_of_81_three_fourths : (81 : ℝ) ^ (3/4 : ℝ) = 27 := by sorry

end NUMINAMATH_CALUDE_power_of_81_three_fourths_l1938_193809


namespace NUMINAMATH_CALUDE_triangle_existence_l1938_193857

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

-- Define the theorem
theorem triangle_existence (m : ℝ) : 
  (∀ a b c : ℝ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 
   a ≠ b ∧ b ≠ c ∧ a ≠ c →
   f m a + f m b > f m c ∧
   f m a + f m c > f m b ∧
   f m b + f m c > f m a) ↔
  m > 6 := by sorry

end NUMINAMATH_CALUDE_triangle_existence_l1938_193857


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l1938_193832

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ 13 ∣ n → n ≥ 104 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l1938_193832


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l1938_193878

/-- A circle with center (-1, 2) that is tangent to the x-axis has the equation (x + 1)^2 + (y - 2)^2 = 4 -/
theorem circle_tangent_to_x_axis :
  ∃ (r : ℝ),
    (∀ (x y : ℝ), (x + 1)^2 + (y - 2)^2 = 4 ↔ ((x + 1)^2 + (y - 2)^2 = r^2)) ∧
    (∀ (x : ℝ), ∃ (y : ℝ), (x + 1)^2 + (y - 2)^2 = 4 → y = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l1938_193878


namespace NUMINAMATH_CALUDE_problem_solution_l1938_193812

theorem problem_solution (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2004 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1938_193812


namespace NUMINAMATH_CALUDE_complex_power_modulus_l1938_193897

theorem complex_power_modulus : Complex.abs ((4 + 2*Complex.I)^5) = 160 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l1938_193897


namespace NUMINAMATH_CALUDE_max_distance_between_C1_and_C2_l1938_193847

-- Define the curves C1 and C2
def C1 (ρ θ : ℝ) : Prop := ρ + 6 * Real.sin θ + 8 / ρ = 0
def C2 (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

-- Define a point on C1
def point_on_C1 (x y : ℝ) : Prop :=
  ∃ (ρ θ : ℝ), C1 ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Define a point on C2
def point_on_C2 (x y : ℝ) : Prop := C2 x y

-- State the theorem
theorem max_distance_between_C1_and_C2 :
  ∃ (max_dist : ℝ),
    max_dist = Real.sqrt 65 / 2 + 1 ∧
    (∀ (x1 y1 x2 y2 : ℝ),
      point_on_C1 x1 y1 → point_on_C2 x2 y2 →
      Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) ≤ max_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      point_on_C1 x1 y1 ∧ point_on_C2 x2 y2 ∧
      Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = max_dist) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_C1_and_C2_l1938_193847


namespace NUMINAMATH_CALUDE_bus_stops_count_l1938_193815

/-- Represents a bus route in the city -/
structure BusRoute where
  stops : ℕ
  stops_ge_three : stops ≥ 3

/-- Represents the city's bus system -/
structure BusSystem where
  routes : Finset BusRoute
  route_count : routes.card = 57
  all_connected : ∀ (r₁ r₂ : BusRoute), r₁ ∈ routes → r₂ ∈ routes → ∃! (s : ℕ), s ≤ r₁.stops ∧ s ≤ r₂.stops
  stops_equal : ∀ (r₁ r₂ : BusRoute), r₁ ∈ routes → r₂ ∈ routes → r₁.stops = r₂.stops

theorem bus_stops_count (bs : BusSystem) : ∀ (r : BusRoute), r ∈ bs.routes → r.stops = 8 := by
  sorry

end NUMINAMATH_CALUDE_bus_stops_count_l1938_193815


namespace NUMINAMATH_CALUDE_remainder_two_power_33_mod_9_l1938_193818

theorem remainder_two_power_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_two_power_33_mod_9_l1938_193818


namespace NUMINAMATH_CALUDE_smallest_d_value_l1938_193866

theorem smallest_d_value : ∃ (d : ℝ), d ≥ 0 ∧ 
  (3 * Real.sqrt 5)^2 + (d + 3)^2 = (3 * d)^2 ∧
  (∀ (x : ℝ), x ≥ 0 ∧ (3 * Real.sqrt 5)^2 + (x + 3)^2 = (3 * x)^2 → d ≤ x) ∧
  d = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_value_l1938_193866


namespace NUMINAMATH_CALUDE_variance_transformation_l1938_193875

def data_variance (data : List ℝ) : ℝ := sorry

theorem variance_transformation (data : List ℝ) (h : data.length = 2010) 
  (h_var : data_variance data = 2) :
  data_variance (data.map (λ x => -3 * x + 1)) = 18 := by sorry

end NUMINAMATH_CALUDE_variance_transformation_l1938_193875


namespace NUMINAMATH_CALUDE_w_sequence_properties_l1938_193827

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sequence w_n
def w : ℕ → ℂ
  | 0 => 1
  | 1 => i
  | (n + 2) => 2 * w (n + 1) + 3 * w n

-- State the theorem
theorem w_sequence_properties :
  (∀ n : ℕ, w n = (1 + i) / 4 * 3^n + (3 - i) / 4 * (-1)^n) ∧
  (∀ n : ℕ, n ≥ 1 → |Complex.re (w n) - Complex.im (w n)| = 1) := by
  sorry


end NUMINAMATH_CALUDE_w_sequence_properties_l1938_193827


namespace NUMINAMATH_CALUDE_max_removable_edges_l1938_193846

/-- Represents a volleyball net grid with internal divisions -/
structure VolleyballNet where
  rows : Nat
  cols : Nat
  internalDivisions : Nat

/-- Calculates the total number of nodes in the volleyball net -/
def totalNodes (net : VolleyballNet) : Nat :=
  (net.rows + 1) * (net.cols + 1) + net.rows * net.cols

/-- Calculates the total number of edges in the volleyball net -/
def totalEdges (net : VolleyballNet) : Nat :=
  net.rows * (net.cols + 1) + net.cols * (net.rows + 1) + net.internalDivisions * net.rows * net.cols

/-- Theorem stating the maximum number of removable edges -/
theorem max_removable_edges (net : VolleyballNet) :
  net.rows = 10 → net.cols = 20 → net.internalDivisions = 4 →
  totalEdges net - (totalNodes net - 1) = 800 := by
  sorry


end NUMINAMATH_CALUDE_max_removable_edges_l1938_193846


namespace NUMINAMATH_CALUDE_cookie_recipe_ratio_l1938_193894

-- Define the total amount of sugar needed for the recipe
def total_sugar : ℚ := 3

-- Define the amount of sugar Katie still needs to add
def sugar_to_add : ℚ := 2.5

-- Define the amount of sugar Katie has already added
def sugar_already_added : ℚ := total_sugar - sugar_to_add

-- Define the ratio of sugar already added to total sugar needed
def sugar_ratio : ℚ × ℚ := (sugar_already_added, total_sugar)

-- Theorem to prove
theorem cookie_recipe_ratio :
  sugar_ratio = (1, 6) := by sorry

end NUMINAMATH_CALUDE_cookie_recipe_ratio_l1938_193894


namespace NUMINAMATH_CALUDE_twenty_apples_fourteen_cucumbers_l1938_193824

/-- Represents the cost of a single apple -/
def apple_cost : ℝ := sorry

/-- Represents the cost of a single banana -/
def banana_cost : ℝ := sorry

/-- Represents the cost of a single cucumber -/
def cucumber_cost : ℝ := sorry

/-- The cost of 10 apples equals the cost of 5 bananas -/
axiom ten_apples_five_bananas : 10 * apple_cost = 5 * banana_cost

/-- The cost of 5 bananas equals the cost of 7 cucumbers -/
axiom five_bananas_seven_cucumbers : 5 * banana_cost = 7 * cucumber_cost

/-- Theorem: The cost of 20 apples equals the cost of 14 cucumbers -/
theorem twenty_apples_fourteen_cucumbers : 20 * apple_cost = 14 * cucumber_cost := by
  sorry

end NUMINAMATH_CALUDE_twenty_apples_fourteen_cucumbers_l1938_193824


namespace NUMINAMATH_CALUDE_pirate_captain_age_l1938_193865

/-- Represents a health insurance card number -/
structure HealthInsuranceCard where
  main_number : Nat
  control_number : Nat
  h_main_digits : main_number < 10000000000000
  h_control_digits : control_number < 100

/-- Checks if a health insurance card number is valid -/
def is_valid_card (card : HealthInsuranceCard) : Prop :=
  (card.main_number + card.control_number) % 97 = 0

/-- Calculates the age based on birth year and current year -/
def calculate_age (birth_year : Nat) (current_year : Nat) : Nat :=
  current_year - birth_year

theorem pirate_captain_age :
  ∃ (card : HealthInsuranceCard),
    card.control_number = 67 ∧
    ∃ (x : Nat), x < 10 ∧ card.main_number = 1000000000000 * (10 + x) + 1271153044 ∧
    is_valid_card card ∧
    calculate_age (1900 + (10 + x)) 2011 = 65 := by
  sorry

end NUMINAMATH_CALUDE_pirate_captain_age_l1938_193865


namespace NUMINAMATH_CALUDE_two_hour_charge_is_174_l1938_193881

/-- Represents the pricing model for therapy sessions -/
structure TherapyPricing where
  first_hour : ℕ
  additional_hour : ℕ
  first_hour_premium : first_hour = additional_hour + 40

/-- Calculates the total charge for a given number of hours -/
def total_charge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  pricing.first_hour + (hours - 1) * pricing.additional_hour

/-- Theorem stating the correct charge for 2 hours given the conditions -/
theorem two_hour_charge_is_174 (pricing : TherapyPricing) 
  (h1 : total_charge pricing 5 = 375) : 
  total_charge pricing 2 = 174 := by
  sorry

end NUMINAMATH_CALUDE_two_hour_charge_is_174_l1938_193881


namespace NUMINAMATH_CALUDE_centers_connection_line_l1938_193808

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem centers_connection_line :
  ∃ (x1 y1 x2 y2 : ℝ),
    (∀ x y, circle1 x y ↔ (x - x1)^2 + (y - y1)^2 = (x1^2 + y1^2)) ∧
    (∀ x y, circle2 x y ↔ (x - x2)^2 + (y - y2)^2 = x2^2) ∧
    line_equation x1 y1 ∧
    line_equation x2 y2 :=
sorry

end NUMINAMATH_CALUDE_centers_connection_line_l1938_193808


namespace NUMINAMATH_CALUDE_hannah_leah_study_difference_l1938_193877

theorem hannah_leah_study_difference (daily_differences : List Int) 
  (h1 : daily_differences = [15, -5, 25, -15, 35, 0, 20]) 
  (days_in_week : Nat) (h2 : days_in_week = 7) : 
  Int.floor ((daily_differences.sum : ℚ) / days_in_week) = 10 := by
  sorry

end NUMINAMATH_CALUDE_hannah_leah_study_difference_l1938_193877


namespace NUMINAMATH_CALUDE_sin_1440_degrees_l1938_193864

theorem sin_1440_degrees : Real.sin (1440 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_1440_degrees_l1938_193864


namespace NUMINAMATH_CALUDE_center_octahedron_volume_ratio_l1938_193821

/-- A regular octahedron -/
structure RegularOctahedron where
  -- We don't need to define the structure fully, just declare it exists
  mk :: (dummy : Unit)

/-- The octahedron formed by the centers of faces of a regular octahedron -/
def center_octahedron (o : RegularOctahedron) : RegularOctahedron :=
  RegularOctahedron.mk ()

/-- The volume of an octahedron -/
def volume (o : RegularOctahedron) : ℝ :=
  sorry

/-- The theorem stating the volume ratio of the center octahedron to the original octahedron -/
theorem center_octahedron_volume_ratio (o : RegularOctahedron) :
  volume (center_octahedron o) / volume o = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_center_octahedron_volume_ratio_l1938_193821


namespace NUMINAMATH_CALUDE_square_nailing_theorem_l1938_193831

/-- Represents a paper square on the table -/
structure Square where
  color : Nat
  position : Real × Real

/-- Represents the arrangement of squares on the table -/
def Arrangement := List Square

/-- Checks if two squares can be nailed with one nail -/
def can_nail_together (s1 s2 : Square) : Prop := sorry

/-- The main theorem to be proved -/
theorem square_nailing_theorem (k : Nat) (arrangement : Arrangement) :
  (∀ (distinct_squares : List Square),
    distinct_squares.length = k →
    distinct_squares.Pairwise (λ s1 s2 => s1.color ≠ s2.color) →
    distinct_squares.Sublist arrangement →
    ∃ (s1 s2 : Square), s1 ∈ distinct_squares ∧ s2 ∈ distinct_squares ∧ can_nail_together s1 s2) →
  ∃ (color : Nat),
    let squares_of_color := arrangement.filter (λ s => s.color = color)
    ∃ (nails : List (Real × Real)), nails.length ≤ 2 * k - 2 ∧
      ∀ (s : Square), s ∈ squares_of_color →
        ∃ (nail : Real × Real), nail ∈ nails ∧ s.position = nail :=
sorry

end NUMINAMATH_CALUDE_square_nailing_theorem_l1938_193831


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l1938_193834

theorem cubic_polynomials_common_roots (a b : ℝ) : 
  (∃ r s : ℝ, r ≠ s ∧ 
    r^3 + a*r^2 + 15*r + 10 = 0 ∧ 
    r^3 + b*r^2 + 18*r + 12 = 0 ∧
    s^3 + a*s^2 + 15*s + 10 = 0 ∧ 
    s^3 + b*s^2 + 18*s + 12 = 0) →
  a = 3 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l1938_193834


namespace NUMINAMATH_CALUDE_polynomial_simplification_simplify_and_evaluate_l1938_193844

-- Part 1: Polynomial simplification
theorem polynomial_simplification (m n : ℝ) :
  6 * m * n - 2 * m - 3 * (m + 2 * m * n) = -5 * m := by sorry

-- Part 2: Simplify and evaluate
theorem simplify_and_evaluate :
  let a : ℝ := 1/2
  let b : ℝ := 3
  a^2 * b^3 - 1/2 * (4 * a * b + 6 * a^2 * b^3 - 1) + 2 * (a * b - a^2 * b^3) = -53/2 := by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_simplify_and_evaluate_l1938_193844


namespace NUMINAMATH_CALUDE_colten_chickens_l1938_193861

theorem colten_chickens (total : ℕ) (q s c : ℕ) : 
  total = 383 →
  q = 2 * s + 25 →
  s = 3 * c - 4 →
  q + s + c = total →
  c = 37 := by
sorry

end NUMINAMATH_CALUDE_colten_chickens_l1938_193861


namespace NUMINAMATH_CALUDE_thirty_five_only_math_l1938_193838

/-- Represents the number of students in various class combinations -/
structure ClassCounts where
  total : ℕ
  math : ℕ
  foreign : ℕ
  sport : ℕ
  all_three : ℕ

/-- Calculates the number of students taking only math class -/
def only_math (counts : ClassCounts) : ℕ :=
  counts.math - (counts.total - (counts.math + counts.foreign + counts.sport - counts.all_three))

/-- Theorem stating that 35 students take only math class given the specific class counts -/
theorem thirty_five_only_math (counts : ClassCounts) 
  (h_total : counts.total = 120)
  (h_math : counts.math = 85)
  (h_foreign : counts.foreign = 65)
  (h_sport : counts.sport = 50)
  (h_all_three : counts.all_three = 10) :
  only_math counts = 35 := by
  sorry

end NUMINAMATH_CALUDE_thirty_five_only_math_l1938_193838


namespace NUMINAMATH_CALUDE_octal_54321_to_decimal_l1938_193806

/-- Converts a base-8 digit to its base-10 equivalent -/
def octalToDecimal (digit : ℕ) : ℕ := digit

/-- Computes the value of a digit in a specific position in base 8 -/
def octalDigitValue (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (8 ^ position)

/-- Theorem: The base-10 equivalent of 54321 in base-8 is 22737 -/
theorem octal_54321_to_decimal : 
  octalToDecimal 1 + 
  octalDigitValue 2 1 + 
  octalDigitValue 3 2 + 
  octalDigitValue 4 3 + 
  octalDigitValue 5 4 = 22737 :=
by sorry

end NUMINAMATH_CALUDE_octal_54321_to_decimal_l1938_193806


namespace NUMINAMATH_CALUDE_solution_to_system_l1938_193803

theorem solution_to_system (x y z : ℝ) : 
  (3 * (x^2 + y^2 + z^2) = 1 ∧ 
   x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x + y + z)^3) → 
  ((x = 0 ∧ y = 0 ∧ z = 1/Real.sqrt 3) ∨ 
   (x = 0 ∧ y = 0 ∧ z = -1/Real.sqrt 3) ∨ 
   (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ 
   (x = 1/3 ∧ y = 1/3 ∧ z = -1/3) ∨ 
   (x = 1/3 ∧ y = -1/3 ∧ z = 1/3) ∨ 
   (x = 1/3 ∧ y = -1/3 ∧ z = -1/3) ∨ 
   (x = -1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ 
   (x = -1/3 ∧ y = 1/3 ∧ z = -1/3) ∨ 
   (x = -1/3 ∧ y = -1/3 ∧ z = 1/3) ∨ 
   (x = -1/3 ∧ y = -1/3 ∧ z = -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_to_system_l1938_193803


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1938_193882

theorem min_value_of_expression (x y z : ℝ) :
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + z^2 + 6 * z + 10 ≥ -7/2 ∧
  ∃ (x₀ y₀ z₀ : ℝ), 3 * x₀^2 + 3 * x₀ * y₀ + y₀^2 - 3 * x₀ + 3 * y₀ + z₀^2 + 6 * z₀ + 10 = -7/2 ∧
    x₀ = 3/2 ∧ y₀ = -3/2 ∧ z₀ = -3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1938_193882


namespace NUMINAMATH_CALUDE_sequence_sum_l1938_193829

theorem sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ + 13*x₇ = 3)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ + 15*x₇ = 15)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ + 17*x₇ = 85) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ + 19*x₇ = 213 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1938_193829


namespace NUMINAMATH_CALUDE_sqrt_product_equals_two_l1938_193860

theorem sqrt_product_equals_two : Real.sqrt (2/3) * Real.sqrt 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_two_l1938_193860


namespace NUMINAMATH_CALUDE_opposite_face_points_are_diametrically_opposite_l1938_193855

-- Define a cube
structure Cube where
  side_length : ℝ
  center : ℝ × ℝ × ℝ

-- Define a point on the surface of a cube
structure CubePoint where
  coordinates : ℝ × ℝ × ℝ
  cube : Cube
  on_surface : Bool

-- Define the concept of diametrically opposite points
def diametrically_opposite (p1 p2 : CubePoint) (c : Cube) : Prop :=
  ∃ (t : ℝ), 
    p1.coordinates = (1 - t) • c.center + t • p2.coordinates ∧ 
    0 ≤ t ∧ t ≤ 1

-- Define opposite faces
def opposite_faces (f1 f2 : CubePoint → Prop) (c : Cube) : Prop :=
  ∀ (p1 p2 : CubePoint), f1 p1 → f2 p2 → diametrically_opposite p1 p2 c

-- Theorem statement
theorem opposite_face_points_are_diametrically_opposite 
  (c : Cube) (p s : CubePoint) (f1 f2 : CubePoint → Prop) :
  opposite_faces f1 f2 c →
  f1 p →
  f2 s →
  diametrically_opposite p s c :=
sorry

end NUMINAMATH_CALUDE_opposite_face_points_are_diametrically_opposite_l1938_193855


namespace NUMINAMATH_CALUDE_incorrect_statement_about_immunity_l1938_193854

-- Define the three lines of defense
inductive LineOfDefense
| First
| Second
| Third

-- Define the types of immunity
inductive ImmunityType
| NonSpecific
| Specific

-- Define the components of each line of defense
def componentsOfDefense (line : LineOfDefense) : String :=
  match line with
  | .First => "skin and mucous membranes"
  | .Second => "antimicrobial substances and phagocytic cells in body fluids"
  | .Third => "immune organs and immune cells"

-- Define the type of immunity for each line of defense
def immunityTypeOfDefense (line : LineOfDefense) : ImmunityType :=
  match line with
  | .First => .NonSpecific
  | .Second => .NonSpecific
  | .Third => .Specific

-- Theorem to prove
theorem incorrect_statement_about_immunity :
  ¬(∀ (line : LineOfDefense), immunityTypeOfDefense line = .NonSpecific) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_about_immunity_l1938_193854


namespace NUMINAMATH_CALUDE_melanie_coin_count_l1938_193823

/-- Represents the number of coins Melanie has or receives -/
structure CoinCount where
  dimes : ℕ
  nickels : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in dollars -/
def coinValue (coins : CoinCount) : ℚ :=
  (coins.dimes * 10 + coins.nickels * 5 + coins.quarters * 25) / 100

/-- Adds two CoinCount structures -/
def addCoins (a b : CoinCount) : CoinCount :=
  { dimes := a.dimes + b.dimes,
    nickels := a.nickels + b.nickels,
    quarters := a.quarters + b.quarters }

def initial : CoinCount := { dimes := 19, nickels := 12, quarters := 8 }
def fromDad : CoinCount := { dimes := 39, nickels := 22, quarters := 15 }
def fromSister : CoinCount := { dimes := 15, nickels := 7, quarters := 12 }
def fromMother : CoinCount := { dimes := 25, nickels := 10, quarters := 0 }
def fromGrandmother : CoinCount := { dimes := 0, nickels := 30, quarters := 3 }

theorem melanie_coin_count :
  let final := addCoins initial (addCoins fromDad (addCoins fromSister (addCoins fromMother fromGrandmother)))
  final.dimes = 98 ∧
  final.nickels = 81 ∧
  final.quarters = 38 ∧
  coinValue final = 2335 / 100 := by
  sorry

end NUMINAMATH_CALUDE_melanie_coin_count_l1938_193823


namespace NUMINAMATH_CALUDE_positive_solution_of_equation_l1938_193852

theorem positive_solution_of_equation : ∃ x : ℝ, x > 0 ∧ 
  (1/2) * (4 * x^2 - 2) = (x^2 - 75*x - 15) * (x^2 + 35*x + 7) ∧
  x = (75 + Real.sqrt 5681) / 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_of_equation_l1938_193852


namespace NUMINAMATH_CALUDE_min_balls_for_three_colors_l1938_193884

theorem min_balls_for_three_colors (num_colors : Nat) (balls_per_color : Nat) 
  (h1 : num_colors = 4) (h2 : balls_per_color = 13) :
  (2 * balls_per_color + 1) = 27 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_three_colors_l1938_193884


namespace NUMINAMATH_CALUDE_smallest_w_sum_of_digits_17_l1938_193899

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest positive integer w such that 10^w - 74 has a sum of digits equal to 17 is 3 -/
theorem smallest_w_sum_of_digits_17 :
  ∀ w : ℕ+, sum_of_digits (10^(w.val) - 74) = 17 → w.val ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_sum_of_digits_17_l1938_193899


namespace NUMINAMATH_CALUDE_base_7_sum_theorem_l1938_193890

def base_7_to_decimal (a b c : Nat) : Nat :=
  7^2 * a + 7 * b + c

theorem base_7_sum_theorem (A B C : Nat) :
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
  A < 7 ∧ B < 7 ∧ C < 7 ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  base_7_to_decimal A B C + base_7_to_decimal B C A + base_7_to_decimal C A B = base_7_to_decimal A A A + 1 →
  B + C = 6 := by
sorry

end NUMINAMATH_CALUDE_base_7_sum_theorem_l1938_193890


namespace NUMINAMATH_CALUDE_exponential_inequality_l1938_193863

theorem exponential_inequality (a b c : ℝ) :
  0 < 0.8 ∧ 0.8 < 1 ∧ 5.2 > 1 →
  0.8^5.5 < 0.8^5.2 ∧ 0.8^5.2 < 5.2^0.1 :=
by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1938_193863


namespace NUMINAMATH_CALUDE_original_class_strength_l1938_193836

/-- Given an adult class, prove that the original strength was 12 students. -/
theorem original_class_strength
  (original_avg : ℝ)
  (new_students : ℕ)
  (new_avg : ℝ)
  (avg_decrease : ℝ)
  (h1 : original_avg = 40)
  (h2 : new_students = 12)
  (h3 : new_avg = 32)
  (h4 : avg_decrease = 4)
  : ∃ (x : ℕ), x = 12 ∧ 
    (x : ℝ) * original_avg + (new_students : ℝ) * new_avg = 
    ((x : ℝ) + new_students) * (original_avg - avg_decrease) :=
by
  sorry


end NUMINAMATH_CALUDE_original_class_strength_l1938_193836


namespace NUMINAMATH_CALUDE_smallest_valid_number_l1938_193872

def is_valid (n : ℕ) : Prop :=
  n > 9 ∧
  ¬(n % 7 = 0) ∧
  ∀ (i : ℕ), i < (String.length (toString n)) →
    ((n.div (10^i) % 10) ≠ 7) ∧
    (((n - (n.div (10^i) % 10) * 10^i + 7 * 10^i) % 7 = 0))

theorem smallest_valid_number :
  is_valid 13264513 ∧ ∀ (m : ℕ), m < 13264513 → ¬(is_valid m) := by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l1938_193872


namespace NUMINAMATH_CALUDE_zero_descriptions_l1938_193804

theorem zero_descriptions (x : ℝ) :
  (x = 0) ↔ 
  (∀ (y : ℝ), x ≤ y ∧ x ≥ y → y = x) ∧ 
  (∀ (y : ℝ), x + y = y) ∧
  (∀ (y : ℝ), x * y = x) :=
sorry

end NUMINAMATH_CALUDE_zero_descriptions_l1938_193804


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l1938_193891

theorem quadratic_integer_roots (p q : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q →
  (∃ x y : ℤ, x^2 + 5*p*x + 7*q = 0 ∧ y^2 + 5*p*y + 7*q = 0) ↔ 
  ((p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l1938_193891


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1938_193840

theorem quadratic_discriminant (a b c : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  |x₂ - x₁| = 2 →
  b^2 - 4*a*c = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1938_193840


namespace NUMINAMATH_CALUDE_new_average_age_with_teacher_l1938_193880

theorem new_average_age_with_teacher 
  (num_students : ℕ) 
  (student_avg_age : ℝ) 
  (teacher_age : ℕ) 
  (h1 : num_students = 50) 
  (h2 : student_avg_age = 14) 
  (h3 : teacher_age = 65) : 
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = 15 := by
sorry

end NUMINAMATH_CALUDE_new_average_age_with_teacher_l1938_193880


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1938_193870

theorem contrapositive_equivalence (a b x : ℝ) :
  (x ≥ a^2 + b^2 → x ≥ 2*a*b) ↔ (x < 2*a*b → x < a^2 + b^2) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1938_193870


namespace NUMINAMATH_CALUDE_n_times_n_plus_one_div_by_three_l1938_193868

theorem n_times_n_plus_one_div_by_three (n : ℕ) (h : 1 ≤ n ∧ n ≤ 99) : 
  3 ∣ (n * (n + 1)) := by
sorry

end NUMINAMATH_CALUDE_n_times_n_plus_one_div_by_three_l1938_193868


namespace NUMINAMATH_CALUDE_cost_per_set_is_20_l1938_193835

/-- Represents the manufacturing and sales scenario for horseshoe sets -/
structure HorseshoeManufacturing where
  initialOutlay : ℕ
  sellingPrice : ℕ
  setsSold : ℕ
  profit : ℕ

/-- Calculates the cost per set given the manufacturing scenario -/
def costPerSet (h : HorseshoeManufacturing) : ℚ :=
  (h.sellingPrice * h.setsSold - h.profit - h.initialOutlay) / h.setsSold

/-- Theorem stating that the cost per set is $20 given the specific scenario -/
theorem cost_per_set_is_20 (h : HorseshoeManufacturing) 
  (h_initial : h.initialOutlay = 10000)
  (h_price : h.sellingPrice = 50)
  (h_sold : h.setsSold = 500)
  (h_profit : h.profit = 5000) :
  costPerSet h = 20 := by
  sorry

#eval costPerSet { initialOutlay := 10000, sellingPrice := 50, setsSold := 500, profit := 5000 }

end NUMINAMATH_CALUDE_cost_per_set_is_20_l1938_193835


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l1938_193886

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a^2 - a + 2 = 0) → 
  (b^3 - 2*b^2 - b + 2 = 0) → 
  (c^3 - 2*c^2 - c + 2 = 0) → 
  (1/(a+2) + 1/(b+2) + 1/(c+2) = 3/2) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l1938_193886


namespace NUMINAMATH_CALUDE_horner_method_f_2_l1938_193822

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^5 - 5x^4 + 3x^3 - 2x^2 + x -/
def f (x : ℚ) : ℚ :=
  horner [1, 0, -2, 3, -5, 3] x

theorem horner_method_f_2 :
  f 2 = 34 := by sorry

end NUMINAMATH_CALUDE_horner_method_f_2_l1938_193822


namespace NUMINAMATH_CALUDE_power_of_two_pairs_l1938_193845

theorem power_of_two_pairs (m n : ℕ+) :
  (∃ a : ℕ, m + n = 2^(a+1)) ∧
  (∃ b : ℕ, m * n + 1 = 2^b) →
  (∃ a : ℕ, (m = 2^(a+1) - 1 ∧ n = 1) ∨ (m = 2^a + 1 ∧ n = 2^a - 1) ∨
             (m = 1 ∧ n = 2^(a+1) - 1) ∨ (m = 2^a - 1 ∧ n = 2^a + 1)) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_pairs_l1938_193845


namespace NUMINAMATH_CALUDE_butter_profit_percentage_l1938_193833

/-- Calculates the profit percentage for a butter mixture sale --/
theorem butter_profit_percentage
  (butter1_weight : ℝ)
  (butter1_price : ℝ)
  (butter2_weight : ℝ)
  (butter2_price : ℝ)
  (selling_price : ℝ)
  (h1 : butter1_weight = 44)
  (h2 : butter1_price = 150)
  (h3 : butter2_weight = 36)
  (h4 : butter2_price = 125)
  (h5 : selling_price = 194.25) :
  let total_cost := butter1_weight * butter1_price + butter2_weight * butter2_price
  let total_weight := butter1_weight + butter2_weight
  let total_selling_price := total_weight * selling_price
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 40 := by
sorry

end NUMINAMATH_CALUDE_butter_profit_percentage_l1938_193833


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1938_193887

theorem imaginary_part_of_z (z : ℂ) (h : (2 + Complex.I) * z = 2 - 4 * Complex.I) : 
  Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1938_193887


namespace NUMINAMATH_CALUDE_bananas_left_l1938_193830

/-- The number of bananas originally in the jar -/
def original_bananas : ℕ := 46

/-- The number of bananas Denise removes from the jar -/
def removed_bananas : ℕ := 5

/-- Theorem stating the number of bananas left in the jar after Denise removes some -/
theorem bananas_left : original_bananas - removed_bananas = 41 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l1938_193830


namespace NUMINAMATH_CALUDE_sum_of_2001_numbers_positive_l1938_193800

theorem sum_of_2001_numbers_positive 
  (a : Fin 2001 → ℝ) 
  (h : ∀ (i j k l : Fin 2001), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l → 
    a i + a j + a k + a l > 0) : 
  Finset.sum Finset.univ a > 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_2001_numbers_positive_l1938_193800
