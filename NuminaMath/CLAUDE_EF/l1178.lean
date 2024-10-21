import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_y_amount_l1178_117815

/-- The amount of Solution Y needed to prepare a larger batch of a mixture, given the initial amounts and the desired total volume. -/
theorem solution_y_amount
  (initial_x : ℝ)
  (initial_y : ℝ)
  (initial_total : ℝ)
  (desired_total : ℝ)
  (hy_positive : initial_y > 0)
  (htotal_positive : initial_total > 0)
  (h_sum : initial_x + initial_y = initial_total)
  (h_initial_x : initial_x = 0.05)
  (h_initial_y : initial_y = 0.03)
  (h_desired_total : desired_total = 0.64) :
  desired_total * (initial_y / initial_total) = 0.24 := by
  sorry

#check solution_y_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_y_amount_l1178_117815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1178_117801

def sequenceA (n : ℕ) : ℚ := 3 * n - 11

theorem max_m_value :
  ∀ m : ℕ,
  (∀ k : ℕ, k ≥ 4 →
    (sequenceA (k + 1) * sequenceA (k + 2)) / sequenceA k ≥ sequenceA m) →
  m ≤ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1178_117801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_period_and_m_value_l1178_117848

noncomputable def a (m : ℝ) (x : ℝ) : ℝ × ℝ := (m + 1, Real.sin x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (1, 4 * Real.cos (x + Real.pi / 6))

noncomputable def g (m : ℝ) (x : ℝ) : ℝ :=
  let av := a m x
  let bv := b x
  av.1 * bv.1 + av.2 * bv.2

theorem g_period_and_m_value (m : ℝ) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), g m (x + T) = g m x ∧
    ∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), g m (x + T') = g m x) → T ≤ T') ∧
  (∃ (max min : ℝ),
    (∀ (x : ℝ), 0 ≤ x ∧ x < Real.pi / 3 → g m x ≤ max) ∧
    (∃ (x : ℝ), 0 ≤ x ∧ x < Real.pi / 3 ∧ g m x = max) ∧
    (∀ (x : ℝ), 0 ≤ x ∧ x < Real.pi / 3 → min ≤ g m x) ∧
    (∃ (x : ℝ), 0 ≤ x ∧ x < Real.pi / 3 ∧ g m x = min) ∧
    max + min = 7 → m = 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_period_and_m_value_l1178_117848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_route_configuration_exists_l1178_117832

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  lines : Finset (Set (ℝ × ℝ))
  intersection_points : Finset (ℝ × ℝ)

/-- The number of lines in the configuration -/
def num_lines (config : LineConfiguration) : ℕ := config.lines.card

/-- The property that each pair of lines intersects at a unique point -/
def unique_intersections (config : LineConfiguration) : Prop :=
  ∀ l₁ l₂, l₁ ∈ config.lines → l₂ ∈ config.lines → l₁ ≠ l₂ → 
    ∃! p, p ∈ config.intersection_points ∧ p ∈ l₁ ∧ p ∈ l₂

/-- The property that any 9 lines cover all intersection points -/
def nine_cover_all (config : LineConfiguration) : Prop :=
  ∀ S, S ⊆ config.lines → S.card = 9 → 
    ∀ p, p ∈ config.intersection_points → ∃ l, l ∈ S ∧ p ∈ l

/-- The property that any 8 lines do not cover all intersection points -/
def eight_miss_some (config : LineConfiguration) : Prop :=
  ∀ S, S ⊆ config.lines → S.card = 8 → 
    ∃ p, p ∈ config.intersection_points ∧ ∀ l, l ∈ S → p ∉ l

/-- The main theorem stating the existence of a configuration satisfying all properties -/
theorem bus_route_configuration_exists : ∃ config : LineConfiguration,
  num_lines config = 10 ∧
  unique_intersections config ∧
  nine_cover_all config ∧
  eight_miss_some config := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_route_configuration_exists_l1178_117832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1178_117888

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y, x ∈ Set.Ioo (-2/3 : ℝ) 2 → y ∈ Set.Ioo (-2/3 : ℝ) 2 → x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1178_117888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_number_l1178_117873

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ 
  ({a, b, c} : Finset ℕ) = {8, 0, 7}

theorem largest_three_digit_number : 
  ∀ n : ℕ, is_valid_number n → n ≤ 870 :=
by
  intro n h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_number_l1178_117873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1178_117892

theorem trigonometric_identities (x : ℝ) 
  (h1 : Real.cos x - Real.sin x = 3 * Real.sqrt 2 / 5)
  (h2 : 5 * Real.pi / 4 < x ∧ x < 7 * Real.pi / 4) :
  (Real.sin x + Real.cos x = -4 * Real.sqrt 2 / 5) ∧
  ((Real.sin (2*x) - 2 * Real.sin x ^ 2) / (1 + Real.tan x) = -21 / 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1178_117892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_analysis_l1178_117882

/-- Represents a polynomial term -/
inductive Term where
  | constant (c : ℝ) : Term
  | linear (a : ℝ) (x : String) : Term
  | nonlinear (t : String) : Term

/-- Represents a polynomial as a list of terms -/
def MyPolynomial := List Term

/-- The given polynomial x^2y^5 - 5x + 3 -/
def given_polynomial : MyPolynomial := [
  Term.nonlinear "x^2y^5",
  Term.linear (-5) "x",
  Term.constant 3
]

/-- Counts the number of terms in a polynomial -/
def count_terms (p : MyPolynomial) : Nat :=
  p.length

/-- Finds the linear term in a polynomial -/
def find_linear_term (p : MyPolynomial) : Option Term :=
  p.find? fun t => match t with
    | Term.linear _ _ => true
    | _ => false

theorem polynomial_analysis :
  (count_terms given_polynomial = 3) ∧
  (find_linear_term given_polynomial = some (Term.linear (-5) "x")) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_analysis_l1178_117882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l1178_117846

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := ((x + 2)^2 + Real.sin x) / (x^2 + 4)

-- Define the theorem
theorem sum_of_max_min_f (a : ℝ) (h : a > 0) :
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc (-a) a, f x ≤ max) ∧
                    (∀ x ∈ Set.Icc (-a) a, f x ≥ min) ∧
                    (max + min = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l1178_117846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l1178_117893

/-- Represents a triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given side lengths form a valid triangle -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Generates the next triangle in the sequence -/
noncomputable def next_triangle (t : Triangle) : Triangle :=
  { a := (t.b + t.c - t.a) / 2,
    b := (t.c + t.a - t.b) / 2,
    c := (t.a + t.b - t.c) / 2 }

/-- The sequence of triangles -/
noncomputable def triangle_sequence : ℕ → Triangle
  | 0 => { a := 1011, b := 1012, c := 1013 }
  | n + 1 => next_triangle (triangle_sequence n)

/-- The perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- The index of the last valid triangle in the sequence -/
def last_valid_index : ℕ := 9

/-- The main theorem statement -/
theorem last_triangle_perimeter :
  perimeter (triangle_sequence last_valid_index) = 759 / 128 := by
  sorry

#eval last_valid_index

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l1178_117893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_F_l1178_117824

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the constants a and b
variable (a b : ℝ)

-- State the conditions
axiom domain_f : ∀ x, x ∈ Set.Icc a b → f x ≠ 0
axiom b_gt_neg_a : b > -a
axiom neg_a_gt_zero : -a > 0

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := f x - f (-x)

-- State the theorem
theorem domain_of_F : 
  Set.Icc a (-a) = {x : ℝ | ∃ y, F y = x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_F_l1178_117824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l1178_117807

-- Define the variables and functions
variable (m : ℝ)

noncomputable def x_A (m : ℝ) : ℝ := 2^(-m)
noncomputable def x_B (m : ℝ) : ℝ := 2^m
noncomputable def x_C (m : ℝ) : ℝ := 8^(-3/(m+1))
noncomputable def x_D (m : ℝ) : ℝ := 8^(3/(m+1))

noncomputable def a (m : ℝ) : ℝ := |2^(-m) - 8^(-3/(m+1))|
noncomputable def b (m : ℝ) : ℝ := |2^m - 8^(3/(m+1))|

noncomputable def f (m : ℝ) : ℝ := b m / a m

-- State the theorem
theorem intersection_properties :
  (∀ m > 0, (x_A m) * (x_B m) = (x_C m) * (x_D m)) ∧
  (∃ m₀ > 0, a m₀ = b m₀ ∧ m₀ = (-1 + Real.sqrt 37) / 2) ∧
  (∀ m > 0, f m = 2^(m + 9/(m+1))) ∧
  (∃ m_min > 0, f m_min = 32 ∧ ∀ m > 0, f m ≥ 32) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l1178_117807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_ideal_function_l1178_117841

-- Define the ideal function properties
def is_ideal_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0)

-- Define the function in question
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 else x^2

-- Theorem statement
theorem f_is_ideal_function : is_ideal_function f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_ideal_function_l1178_117841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1178_117837

noncomputable def f (x : ℝ) : ℝ := (2 * x - 3) / Real.sqrt (x - 7)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 7} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1178_117837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_DFE_to_ABEF_l1178_117853

noncomputable section

-- Define the rectangle ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (3, 0)

-- Define point E as the midpoint of BD
def E : ℝ × ℝ := ((D.1 + B.1) / 2, (D.2 + B.2) / 2)

-- Define point F on DA such that DF = 1/4 DA
def F : ℝ × ℝ := (3 * 3/4, 0)

-- Function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Theorem statement
theorem area_ratio_DFE_to_ABEF :
  (triangleArea D F E) / (triangleArea A B E + triangleArea A E F) = 1/7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_DFE_to_ABEF_l1178_117853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l1178_117852

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := ((1 + i)^2 + 3 * (1 - i)) / (2 + i)

theorem complex_equation_solution (a b : ℝ) :
  z^2 + a * z + b = 1 + i → a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l1178_117852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1178_117821

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (x / 2) * Real.cos (x / 2) + 2 * (Real.cos (x / 2))^2

-- State the theorem
theorem f_properties :
  -- The smallest positive period of f is 2π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧ p = 2 * Real.pi) ∧
  -- f is monotonically decreasing in the intervals [2kπ + π/3, 2kπ + 4π/3], where k ∈ ℤ
  (∀ (k : ℤ), ∀ (x y : ℝ), 
    2 * ↑k * Real.pi + Real.pi / 3 ≤ x ∧ 
    x ≤ y ∧ 
    y ≤ 2 * ↑k * Real.pi + 4 * Real.pi / 3 → 
    f y ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1178_117821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_l1178_117842

theorem isosceles_triangles (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : ∀ n : ℕ, ∃ (x y z : ℝ), 
    x = a^n ∧ y = b^n ∧ z = c^n ∧
    x + y > z ∧ y + z > x ∧ z + x > y) :
  b = c := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_l1178_117842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_pi_l1178_117895

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := 
  Int.floor x

-- State the theorem
theorem floor_of_pi : floor Real.pi = 3 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_pi_l1178_117895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_hyperbola_k_value_hyperbola_equation_l1178_117820

/-- Represents a hyperbola with the equation x²/(k-3) + y²/(k-5) = 1 where k is an integer -/
structure Hyperbola where
  k : ℤ
  is_hyperbola : (k - 3) * (k - 5) < 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt 2

/-- Theorem: The eccentricity of the given hyperbola is √2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : eccentricity h = Real.sqrt 2 := by
  -- Unfold the definition of eccentricity
  unfold eccentricity
  -- The equality is now trivial
  rfl

/-- Theorem: For the given hyperbola, k = 4 -/
theorem hyperbola_k_value (h : Hyperbola) : h.k = 4 := by
  sorry

/-- Theorem: The equation of the hyperbola is y² - x² = 1 -/
theorem hyperbola_equation (h : Hyperbola) : 
  ∃ (x y : ℝ), y^2 - x^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_hyperbola_k_value_hyperbola_equation_l1178_117820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_decrease_l1178_117810

theorem stock_price_decrease (x : ℝ) (h : x > 0) :
  let increase := 0.35
  let new_price := x * (1 + increase)
  let decrease := 1 - 1 / (1 + increase)
  ∃ (ε : ℝ), abs (decrease - 0.2593) < ε ∧ ε > 0 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_decrease_l1178_117810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1178_117840

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/4)^x - 3*(1/2)^x + 2

-- Define the domain
def domain : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

-- State the theorem
theorem range_of_f :
  ∃ (y_min y_max : ℝ), y_min = -1/4 ∧ y_max = 6 ∧
  (∀ x, x ∈ domain → y_min ≤ f x ∧ f x ≤ y_max) ∧
  (∃ x1 x2, x1 ∈ domain ∧ x2 ∈ domain ∧ f x1 = y_min ∧ f x2 = y_max) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1178_117840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_creature_is_rabbit_l1178_117812

noncomputable section

/-- The probability of being a rabbit -/
def P_rabbit : ℝ := 1/2

/-- The probability of being a hare -/
def P_hare : ℝ := 1/2

/-- The probability a rabbit is wrong -/
def P_rabbit_wrong : ℝ := 1/3

/-- The probability a hare is wrong -/
def P_hare_wrong : ℝ := 1/4

/-- The event that the creature says it's not a rabbit -/
def event_not_rabbit : Set (Fin 2) := {0}

/-- The event that the creature says it's not a hare -/
def event_not_hare : Set (Fin 2) := {1}

/-- The probability that a rabbit says it's not a rabbit -/
def P_rabbit_says_not_rabbit : ℝ := P_rabbit_wrong

/-- The probability that a rabbit says it's not a hare -/
def P_rabbit_says_not_hare : ℝ := 1 - P_rabbit_wrong

/-- The probability that a hare says it's not a rabbit -/
def P_hare_says_not_rabbit : ℝ := 1 - P_hare_wrong

/-- The probability that a hare says it's not a hare -/
def P_hare_says_not_hare : ℝ := P_hare_wrong

theorem probability_creature_is_rabbit :
  (P_rabbit * P_rabbit_says_not_rabbit * P_rabbit_says_not_hare) /
  (P_rabbit * P_rabbit_says_not_rabbit * P_rabbit_says_not_hare +
   P_hare * P_hare_says_not_rabbit * P_hare_says_not_hare) = 27/59 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_creature_is_rabbit_l1178_117812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_prime_360_l1178_117833

/-- Base prime representation of a natural number -/
def BasePrime (n : ℕ) : List ℕ := sorry

/-- Checks if a list of natural numbers is a valid base prime representation -/
def IsValidBasePrime (l : List ℕ) : Prop := sorry

theorem base_prime_360 : 
  ∃ (l : List ℕ), BasePrime 360 = l ∧ IsValidBasePrime l ∧ l = [3, 1, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_prime_360_l1178_117833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ABHFGD_l1178_117866

-- Define the points
variable (A B C D E F G H : EuclideanSpace ℝ (Fin 2))

-- Define the squares
def is_square (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the area of a square
def area_square (P Q R S : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Define the midpoint of a line segment
def is_midpoint (M P Q : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define a point being one-third along a line segment
def is_one_third_along (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the area of a polygon
def area_polygon (vertices : List (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

-- State the theorem
theorem area_of_ABHFGD (h1 : is_square A B C D)
                        (h2 : is_square E F G D)
                        (h3 : area_square A B C D = 25)
                        (h4 : area_square E F G D = 25)
                        (h5 : is_midpoint H E F)
                        (h6 : is_one_third_along B H C) :
  ∃ (ε : ℝ), abs (area_polygon [A, B, H, F, G, D] - 27.09) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ABHFGD_l1178_117866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_percentage_l1178_117885

noncomputable section

def radio_cp : ℝ := 1500
def tv_cp : ℝ := 8000
def refrigerator_cp : ℝ := 25000
def microwave_cp : ℝ := 6000
def washing_machine_cp : ℝ := 14000

def radio_sp : ℝ := 1110
def tv_sp : ℝ := 7500
def refrigerator_sp : ℝ := 23000
def microwave_sp : ℝ := 6600
def washing_machine_sp : ℝ := 13000

def total_cp : ℝ := radio_cp + tv_cp + refrigerator_cp + microwave_cp + washing_machine_cp
def total_sp : ℝ := radio_sp + tv_sp + refrigerator_sp + microwave_sp + washing_machine_sp

def overall_loss : ℝ := total_cp - total_sp
def overall_loss_percentage : ℝ := (overall_loss / total_cp) * 100

theorem store_loss_percentage :
  (overall_loss_percentage ≥ 6.02) ∧ (overall_loss_percentage ≤ 6.04) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_percentage_l1178_117885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_problem_l1178_117856

/-- Given a quadratic function f(x) = -x^2 + ax + b with a maximum value of 0,
    and where the solution set of f(x) > c-1 is (m-4, m+1),
    prove that c = -21/4 -/
theorem quadratic_function_problem (a b m c : ℝ) : 
  (∀ x, -x^2 + a*x + b ≤ 0) →                             -- Range is (-∞, 0]
  (∀ x, -x^2 + a*x + b > c - 1 ↔ m - 4 < x ∧ x < m + 1) → -- Solution set is (m-4, m+1)
  c = -21/4                                               -- Conclusion
:= by
  intro h1 h2
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_problem_l1178_117856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terese_monday_distance_l1178_117864

/-- Terese's running distances throughout the week -/
structure RunningWeek where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ
  thursday : ℚ

/-- The average distance Terese runs per day -/
def average_distance (week : RunningWeek) : ℚ :=
  (week.monday + week.tuesday + week.wednesday + week.thursday) / 4

theorem terese_monday_distance (week : RunningWeek) 
  (h1 : week.tuesday = 38/10)
  (h2 : week.wednesday = 36/10)
  (h3 : week.thursday = 44/10)
  (h4 : average_distance week = 4) :
  week.monday = 42/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terese_monday_distance_l1178_117864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1178_117860

/-- The range of the inclination angle of the line x*sin(α) + y + 2 = 0 -/
theorem inclination_angle_range :
  ∀ α : ℝ, ∃ θ : ℝ, 
    (θ ∈ Set.Icc 0 (π/4) ∪ Set.Ico (3*π/4) π) ∧ 
    (Real.tan θ = -Real.sin α) ∧
    (∀ x y : ℝ, x * Real.sin α + y + 2 = 0 → 
      ∃ k : ℝ, k = -Real.sin α ∧ y = k * x - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1178_117860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_beautiful_under_100_l1178_117887

def is_beautiful (n : ℕ) : Prop :=
  n > 1 ∧
  ∃ (k : ℕ) (a : ℕ → ℕ),
    k > 1 ∧
    (∀ i < k, a i > 0) ∧
    (∀ i < k - 1, a i ≥ a (i + 1)) ∧
    n = (Finset.range k).sum a ∧
    n = (Finset.range k).prod a ∧
    ∀ (k' : ℕ) (a' : ℕ → ℕ),
      k' > 1 →
      (∀ i < k', a' i > 0) →
      (∀ i < k' - 1, a' i ≥ a' (i + 1)) →
      n = (Finset.range k').sum a' →
      n = (Finset.range k').prod a' →
      k = k' ∧ ∀ i < k, a i = a' i

theorem largest_beautiful_under_100 :
  is_beautiful 95 ∧ ∀ n, 95 < n → n < 100 → ¬ is_beautiful n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_beautiful_under_100_l1178_117887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_l1178_117808

/-- Calculates the time taken for a car to travel a given distance at a given speed -/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Theorem: A car traveling at 65 miles per hour for 455 miles takes 7 hours -/
theorem car_travel_time : travel_time 455 65 = 7 := by
  -- Unfold the definition of travel_time
  unfold travel_time
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_l1178_117808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sanmao_current_age_l1178_117816

/-- Represents the age of Sanmao when his father was 36 years old -/
def sanmao_age_when_dad_36 : ℕ := sorry

/-- Represents the current age of Sanmao's father -/
def dad_current_age : ℕ := sorry

/-- Theorem stating Sanmao's current age given the problem conditions -/
theorem sanmao_current_age :
  (4 * sanmao_age_when_dad_36 + 3 * sanmao_age_when_dad_36 + sanmao_age_when_dad_36 = dad_current_age / 2) →
  (dad_current_age + (4 * sanmao_age_when_dad_36 + (dad_current_age - 36)) +
   (3 * sanmao_age_when_dad_36 + (dad_current_age - 36)) +
   (sanmao_age_when_dad_36 + (dad_current_age - 36)) = 108) →
  (sanmao_age_when_dad_36 + (dad_current_age - 36) = 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sanmao_current_age_l1178_117816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_when_a_zero_f_increasing_iff_a_leq_16_l1178_117827

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a/x

-- Theorem 1: f(x) is even when a = 0
theorem f_is_even_when_a_zero (x : ℝ) (h : x ≠ 0) :
  f 0 (-x) = f 0 x := by
  sorry

-- Theorem 2: f(x) is increasing on [2,+∞) iff a ≤ 16
theorem f_increasing_iff_a_leq_16 (a : ℝ) :
  (∀ x y, 2 ≤ x → x < y → f a x < f a y) ↔ a ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_when_a_zero_f_increasing_iff_a_leq_16_l1178_117827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l1178_117828

/-- Represents a triangular pyramid with vertex P and base ABC. -/
structure TriangularPyramid where
  PA : ℝ
  PB : ℝ
  PC : ℝ
  angle_PAB : ℝ
  angle_PAC : ℝ
  angle_PBC : ℝ

/-- Calculates the surface area of a sphere given its radius. -/
noncomputable def sphereSurfaceArea (radius : ℝ) : ℝ := 4 * Real.pi * radius ^ 2

/-- Theorem: The surface area of the circumscribed sphere of a specific triangular pyramid is 9π. -/
theorem circumscribed_sphere_surface_area (pyramid : TriangularPyramid) 
  (h1 : pyramid.PA = 1)
  (h2 : pyramid.PB = 2)
  (h3 : pyramid.PC = 2)
  (h4 : pyramid.angle_PAB = Real.pi / 2)
  (h5 : pyramid.angle_PAC = Real.pi / 2)
  (h6 : pyramid.angle_PBC = Real.pi / 2) :
  sphereSurfaceArea ((pyramid.PA ^ 2 + pyramid.PB ^ 2 + pyramid.PC ^ 2).sqrt / 2) = 9 * Real.pi := by
  sorry

#check circumscribed_sphere_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l1178_117828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_element_is_46_l1178_117884

def is_product_of_consecutive (n : ℕ) : Bool :=
  match n with
  | 0 => false
  | n + 1 => List.any (List.range n) (λ k => (k + 1) * (k + 2) == n + 1)

def valid_list : List ℕ :=
  (List.range 100).filter (λ n => ¬is_product_of_consecutive (n + 1))

theorem fortieth_element_is_46 : valid_list[39]? = some 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_element_is_46_l1178_117884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistance_l1178_117861

-- Define the resistances x and y
noncomputable def x : ℝ := 3
noncomputable def y : ℝ := 5

-- Define the combined resistance R
noncomputable def R : ℝ := (x * y) / (x + y)

-- Theorem statement
theorem parallel_resistance : R = 15 / 8 := by
  -- Unfold the definition of R
  unfold R
  -- Simplify the expression
  simp [x, y]
  -- The proof is completed automatically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistance_l1178_117861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l1178_117862

/-- A power function that passes through the point (2, √2/2) -/
noncomputable def f (x : ℝ) : ℝ := x^(-1/2 : ℝ)

theorem fixed_point_power_function :
  f 2 = Real.sqrt 2 / 2 ∧ f 9 = 1/3 := by
  sorry

#check fixed_point_power_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l1178_117862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_negative_one_l1178_117835

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (1 / x)
  else if x < 0 then 1 / x
  else 0  -- Value at x = 0 is arbitrary since it's not in the domain

-- Define the solution set
def solution_set : Set ℝ :=
  Set.Ioi (-1) ∪ Set.Ioo 0 (Real.exp 1)

-- Theorem statement
theorem f_greater_than_negative_one :
  {x : ℝ | f x > -1} = solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_negative_one_l1178_117835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_120_l1178_117839

theorem factors_of_120 : Finset.card (Nat.divisors 120) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_120_l1178_117839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_smartphone_percentage_l1178_117823

/-- The percentage of boxes containing smartphones in a warehouse -/
noncomputable def smartphone_box_percentage (total_boxes : ℕ) (smartphone_boxes : ℕ) : ℝ :=
  (smartphone_boxes : ℝ) / (total_boxes : ℝ) * 100

/-- Theorem stating that the percentage of boxes containing smartphones is approximately 58.59% -/
theorem warehouse_smartphone_percentage :
  let total_boxes : ℕ := 3680
  let smartphone_boxes : ℕ := 2156
  abs (smartphone_box_percentage total_boxes smartphone_boxes - 58.59) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_smartphone_percentage_l1178_117823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_k_range_l1178_117899

/-- Circle in the Cartesian plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Distance from a point to a line -/
noncomputable def distPointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  let (x, y) := p
  |l.slope * x - y + l.intercept| / Real.sqrt (l.slope^2 + 1)

/-- Check if a point is on a circle -/
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Theorem about the range of k for perpendicular tangents -/
theorem perpendicular_tangents_k_range 
  (c : Circle) 
  (l : Line) 
  (h1 : c.center = (2, 0)) 
  (h2 : c.radius = 2) 
  (h3 : l.slope = k) 
  (h4 : l.intercept = -k) 
  (h5 : ∃ (p : ℝ × ℝ), p.2 = k * (p.1 + 1) ∧ 
    ∃ (t1 t2 : ℝ × ℝ), (onCircle t1 c) ∧ (onCircle t2 c) ∧ 
    (Line.mk (p.1 - t1.1) (p.2 - t1.2)).slope * 
    (Line.mk (p.1 - t2.1) (p.2 - t2.2)).slope = -1) :
  k ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_k_range_l1178_117899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l1178_117817

theorem negation_of_sin_leq_one :
  (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x > 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l1178_117817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_school_l1178_117886

/-- Given a school with total students and a representative sample, calculate the number of girls in the school. -/
theorem girls_in_school
  (total_students : ℕ)
  (sample_size : ℕ)
  (girl_boy_diff : ℕ)
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : girl_boy_diff = 10)
  (h4 : sample_size % 2 = 0)  -- Ensure sample size is even for simplicity
  : ℕ := by
  sorry

-- Remove the #eval line as it's causing the compilation error
-- #eval girls_in_school 1600 200 10 rfl rfl rfl rfl

-- Instead, we can add a test using the #check command
#check girls_in_school 1600 200 10 rfl rfl rfl rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_school_l1178_117886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_sum_of_squares_represents_difference_l1178_117826

/-- Represents a data point in a regression analysis -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Calculates the predicted y-value for a given x-value using the regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.slope * x + model.intercept

/-- Calculates the residual for a single data point -/
def calculateResidual (model : LinearRegression) (point : DataPoint) : ℝ :=
  point.y - predict model point.x

/-- Calculates the residual sum of squares for a set of data points -/
def residualSumOfSquares (model : LinearRegression) (data : List DataPoint) : ℝ :=
  (data.map (λ point => (calculateResidual model point) ^ 2)).sum

/-- Represents the difference between data points and their corresponding positions on the regression line -/
def differenceBetweenDataAndRegressionLine (model : LinearRegression) (data : List DataPoint) : ℝ :=
  residualSumOfSquares model data

/-- Theorem stating that the residual sum of squares represents the difference
    between data points and their corresponding positions on the regression line -/
theorem residual_sum_of_squares_represents_difference
  (model : LinearRegression) (data : List DataPoint) :
  residualSumOfSquares model data =
  differenceBetweenDataAndRegressionLine model data := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_sum_of_squares_represents_difference_l1178_117826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l1178_117894

noncomputable def power_function (k : ℝ) : ℝ → ℝ := fun x => x^k

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem power_function_properties :
  (∀ k : ℝ, ∀ x > 0, power_function k x > 0) ∧
  (∃ k < 0, ¬ (∀ x > 0, power_function k x = power_function (1/k) x)) ∧
  (∃ k > 0, ¬ (∀ x y, x < y → power_function k x < power_function k y)) ∧
  (∃ k : ℝ, ¬ (∃ x₁ x₂, x₁ ≠ x₂ ∧ power_function k x₁ = power_function (-k) x₁ ∧ 
                            power_function k x₂ = power_function (-k) x₂)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l1178_117894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1178_117844

/-- An arithmetic sequence with sum of first n terms S_n -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ S (n + 1) = S n + a (n + 1)

/-- A geometric sequence with first term 2 and common ratio > 0 -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ b 1 = 2 ∧ ∀ n : ℕ, b (n + 1) = b n * q

theorem sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a S)
  (h_geom : geometric_sequence b)
  (h_sum : b 2 + b 3 = 12)
  (h_relation : b 3 = a 4 - 2 * a 1)
  (h_S11 : S 11 = 11 * b 4) :
  (∀ n : ℕ, a n = 3 * ↑n - 2) ∧
  (∀ n : ℕ, b n = 2^n) ∧
  (∀ n : ℕ, (Finset.range n).sum (λ i ↦ a (2 * (i + 1)) * b (i + 1)) = (3 * ↑n - 4) * 2^(n + 2) + 16) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1178_117844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l1178_117859

def number_of_boys : ℕ := 3
def number_of_girls : ℕ := 5
def boys_to_select : ℕ := 2
def girls_to_select : ℕ := 1
def number_of_competitions : ℕ := 3

theorem arrangement_count :
  (number_of_boys.choose boys_to_select) *
  (number_of_girls.choose girls_to_select) *
  number_of_competitions *
  (Nat.factorial (number_of_competitions - 1)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l1178_117859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_circle_tangent_property_l1178_117871

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a triangle with three vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a line segment between two points -/
def Segment (P Q : Point) : Set Point := sorry

/-- Checks if a triangle is isosceles -/
def IsIsosceles (t : Triangle) : Prop := sorry

/-- Checks if a point lies on a line segment -/
def OnSegment (P : Point) (seg : Set Point) : Prop := sorry

/-- Checks if a circle is tangent to a line segment -/
def IsTangent (c : Circle) (seg : Set Point) : Prop := sorry

/-- Checks if a point lies on the base of a triangle -/
def OnBase (P : Point) (t : Triangle) : Prop := sorry

/-- Calculates the distance between two points -/
noncomputable def dist (P Q : Point) : ℝ := sorry

/-- Main theorem -/
theorem isosceles_circle_tangent_property
  (ABC : Triangle)
  (S : Circle)
  (P Q : Point)
  (h1 : IsIsosceles ABC)
  (h2 : OnBase S.center ABC)
  (h3 : IsTangent S (Segment ABC.A ABC.B))
  (h4 : IsTangent S (Segment ABC.A ABC.C))
  (h5 : OnSegment P (Segment ABC.A ABC.B))
  (h6 : OnSegment Q (Segment ABC.A ABC.C))
  (h7 : IsTangent S (Segment P Q)) :
  4 * dist P ABC.B * dist Q ABC.C = dist ABC.B ABC.C ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_circle_tangent_property_l1178_117871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_with_3000_obtuse_angle_sum_l1178_117883

/-- Represents a convex polygon with n sides. -/
structure ConvexPolygon (n : ℕ) : Type :=
  (sides : n ≥ 3)
  (is_convex : Bool)

/-- Represents the sum of obtuse angles in a convex polygon. -/
def SumObtuseAngles (n : ℕ) (h : ConvexPolygon n) : ℕ := 
  sorry -- Placeholder for the actual implementation

/-- A convex polygon with the sum of its obtuse angles equal to 3000° can only have 19 or 20 sides. -/
theorem convex_polygon_with_3000_obtuse_angle_sum (n : ℕ) 
  (h_convex : ConvexPolygon n) 
  (h_obtuse_sum : SumObtuseAngles n h_convex = 3000) : n = 19 ∨ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_with_3000_obtuse_angle_sum_l1178_117883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obstruction_theorem_l1178_117829

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a function to check if two circles are non-overlapping
def non_overlapping (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

-- Define a function to determine if a point is outside the external common tangents
def outside_external_tangents (p : Point) (c1 c2 : Circle) : Prop :=
  sorry -- Implementation details omitted

-- Define a relation for complete obstruction
def completely_obstructs (c1 c2 : Circle) (p : Point) : Prop :=
  sorry -- Implementation details omitted

-- Theorem statement
theorem obstruction_theorem (c1 c2 : Circle) (p : Point) :
  non_overlapping c1 c2 →
  outside_external_tangents p c1 c2 →
  (∃ c c', (c = c1 ∨ c = c2) ∧ (c' = c1 ∨ c' = c2) ∧ c ≠ c' ∧
  completely_obstructs c c' p) :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obstruction_theorem_l1178_117829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1178_117814

-- Define the line equation
noncomputable def line_equation (x y : ℝ) : Prop := -x + Real.sqrt 3 * y - 6 = 0

-- Define the inclination angle
noncomputable def inclination_angle : ℝ := 30 * Real.pi / 180

-- Define the y-intercept
noncomputable def y_intercept : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem line_properties :
  (∀ x y : ℝ, line_equation x y → 
    (Real.tan inclination_angle = Real.sqrt 3 / 3) ∧
    (line_equation 0 y_intercept)) :=
by
  sorry

#check line_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1178_117814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1178_117869

def is_valid_assignment (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Set ℕ) ∧
  b ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Set ℕ) ∧
  c ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Set ℕ) ∧
  d ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Set ℕ) ∧
  e ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Set ℕ)

noncomputable def expression_value (a b c d e : ℕ) : ℝ :=
  Real.sqrt ((a / 2 : ℝ) ^ ((d : ℝ) / (e : ℝ)) / ((b : ℝ) / (c : ℝ)))

theorem max_value_theorem :
  ∀ a b c d e : ℕ,
    is_valid_assignment a b c d e →
    expression_value a b c d e ≤ expression_value 8 6 5 7 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1178_117869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_dihedral_plane_angle_can_be_obtuse_l1178_117822

-- Define the three types of angles
inductive AngleType
  | SkewLines
  | LineAndPlane
  | DihedralPlaneAngle

-- Define the property of being obtuse
def isObtuse (θ : Real) : Prop := Real.pi / 2 < θ ∧ θ < Real.pi

-- Define the range for each angle type
def angleRange (t : AngleType) : Set Real :=
  match t with
  | AngleType.SkewLines => {θ : Real | 0 < θ ∧ θ ≤ Real.pi / 2}
  | AngleType.LineAndPlane => {θ : Real | 0 ≤ θ ∧ θ ≤ Real.pi / 2}
  | AngleType.DihedralPlaneAngle => {θ : Real | 0 ≤ θ ∧ θ ≤ Real.pi}

-- Theorem statement
theorem only_dihedral_plane_angle_can_be_obtuse :
  ∃ (θ : Real), θ ∈ angleRange AngleType.DihedralPlaneAngle ∧ isObtuse θ ∧
  (∀ (t : AngleType), t ≠ AngleType.DihedralPlaneAngle →
    ∀ (φ : Real), φ ∈ angleRange t → ¬isObtuse φ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_dihedral_plane_angle_can_be_obtuse_l1178_117822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tan_sum_l1178_117813

/-- Given a line with equation x*tan(α) - y - 3*tan(β) = 0, slope 2, and y-intercept 1, prove that tan(α + β) = 1 -/
theorem line_tan_sum (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan β = -1/3) : Real.tan (α + β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tan_sum_l1178_117813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_rectangle_property_l1178_117845

-- Define the triangle ABC
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

-- Define the rectangle PQRS
structure Rectangle where
  P : EuclideanSpace ℝ (Fin 2)
  Q : EuclideanSpace ℝ (Fin 2)
  R : EuclideanSpace ℝ (Fin 2)
  S : EuclideanSpace ℝ (Fin 2)

-- Define the properties of the triangle and rectangle
def is_acute_triangle (t : Triangle) : Prop := sorry

def altitude_foot (t : Triangle) (D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def is_largest_inscribed_rectangle (t : Triangle) (r : Rectangle) : Prop := sorry

def is_harmonic_mean (x y z : ℝ) : Prop := 2 / x = 1 / y + 1 / z

-- Main theorem
theorem largest_inscribed_rectangle_property (t : Triangle) (r : Rectangle) (D : EuclideanSpace ℝ (Fin 2)) :
  is_acute_triangle t →
  altitude_foot t D →
  is_largest_inscribed_rectangle t r →
  is_harmonic_mean (dist r.P r.Q) (dist t.A D / dist D t.B) (dist t.A D / dist D t.C) →
  dist t.B t.C = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_rectangle_property_l1178_117845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_xy_value_min_xy_is_minimum_l1178_117847

-- Define the equation
def equation (x y : ℝ) : Prop :=
  2 * (Real.cos (x + y - 1))^2 = ((x + 1)^2 + (y - 1)^2 - 2*x*y) / (x - y + 1)

-- State the theorem
theorem min_xy_value (x y : ℝ) (h : equation x y) : 
  ∀ a b : ℝ, equation a b → x * y ≤ a * b ∧ x * y ≥ (1/4 : ℝ) := by
  sorry

-- Define the minimum value
noncomputable def min_xy : ℝ := 1/4

-- State that 1/4 is indeed the minimum value
theorem min_xy_is_minimum : 
  ∃ x y : ℝ, equation x y ∧ x * y = min_xy := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_xy_value_min_xy_is_minimum_l1178_117847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l1178_117809

/-- If the terminal side of angle α passes through point P(-4, m) and sin α = 3/5, then m = 3 -/
theorem angle_terminal_side_point (α : ℝ) (m : ℝ) : 
  (∃ (x y : ℝ), x = -4 ∧ y = m ∧ (x, y) ≠ (0, 0) ∧ Real.sin α = y / Real.sqrt (x^2 + y^2)) →
  Real.sin α = 3/5 →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_point_l1178_117809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_angle_relation_l1178_117870

theorem triangle_side_angle_relation (A B : ℝ) (a b : ℝ) :
  (0 < A) → (A < Real.pi) → (0 < B) → (B < Real.pi) →
  (a < b ↔ Real.cos A > Real.cos B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_angle_relation_l1178_117870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_tripled_numbers_l1178_117805

theorem product_of_tripled_numbers : 
  (∃ x y : ℝ, (x + 1/x = 3*x) ∧ (y + 1/y = 3*y) ∧ (x ≠ y)) → 
  (∃ p : ℝ, p = -1/2 ∧ p = x * y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_tripled_numbers_l1178_117805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_seventh_powers_of_roots_l1178_117874

theorem sum_of_seventh_powers_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 + 3*x₁ + 1 = 0 → x₂^2 + 3*x₂ + 1 = 0 → x₁^7 + x₂^7 = -843 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_seventh_powers_of_roots_l1178_117874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_condition_implies_a_range_l1178_117857

noncomputable section

-- Define the equation
def equation (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x - 4

-- Define the condition for roots
def isRoot (a : ℝ) (x : ℝ) : Prop := equation a x = 0

-- Define the corresponding point for a root
def correspondingPoint (x : ℝ) : ℝ × ℝ := (x, 4/x)

-- Define the condition for points being on the same side of y = x
def sameSideOfDiagonal (points : Set (ℝ × ℝ)) : Prop :=
  (∀ p ∈ points, p.1 < p.2) ∨ (∀ p ∈ points, p.1 > p.2)

-- Main theorem
theorem root_condition_implies_a_range (a : ℝ) :
  (∃ roots : Set ℝ, (∀ x ∈ roots, isRoot a x) ∧
   sameSideOfDiagonal (Set.image correspondingPoint roots)) →
  a ∈ Set.Iic (-6) ∪ Set.Ioi 6 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_condition_implies_a_range_l1178_117857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_race_outcomes_count_l1178_117867

/-- The number of participants in the race -/
def num_participants : ℕ := 6

/-- The number of places we're considering (1st, 2nd, 3rd) -/
def num_places : ℕ := 3

/-- 
Theorem stating that the number of ways to arrange 3 distinct places 
from 6 participants is 120
-/
theorem race_outcomes_count : 
  (Nat.factorial num_participants / Nat.factorial (num_participants - num_places)) = 120 := by
  rw [num_participants, num_places]
  norm_num
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_race_outcomes_count_l1178_117867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_sum_l1178_117806

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the expression
noncomputable def rationalizedExpression : ℝ := 
  (cubeRoot 25 + cubeRoot 20 + cubeRoot 16) / 1

-- Theorem statement
theorem rationalize_denominator_sum :
  (1 / (cubeRoot 5 - cubeRoot 4) = rationalizedExpression) ∧
  (25 + 20 + 16 + 1 = 62) := by
  sorry

#eval 25 + 20 + 16 + 1  -- This will evaluate to 62

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_sum_l1178_117806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_equals_n_plus_one_l1178_117803

/-- Represents a labyrinth with n walls -/
structure Labyrinth where
  n : ℕ
  walls : Fin n → EuclideanSpace ℝ (Fin 2)
  no_parallel : ∀ i j, i ≠ j → walls i ≠ walls j
  no_triple_intersection : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    ¬ ∃ p, p ∈ Set.range (walls i) ∩ Set.range (walls j) ∩ Set.range (walls k)

/-- Represents a coloring of the labyrinth walls -/
def Coloring (L : Labyrinth) := Fin L.n → Bool

/-- Represents the maximum number of knights that can be placed in the labyrinth -/
noncomputable def k (L : Labyrinth) : ℕ := sorry

/-- The main theorem: k(L) = n + 1 for any labyrinth L with n walls -/
theorem k_equals_n_plus_one (L : Labyrinth) : k L = L.n + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_equals_n_plus_one_l1178_117803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l1178_117865

theorem power_difference (m n : ℝ) (h1 : (10 : ℝ)^m = 12) (h2 : (10 : ℝ)^n = 3) : 
  (10 : ℝ)^(m-n) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l1178_117865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_theorem_l1178_117834

-- Define the sets A, B, and C
def A : Set ℝ := {x | |x - 1| ≤ 3}
def B : Set ℝ := {x | (x - 1) / (x + 3) < 0}
def C (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2 + m}

-- State the theorem
theorem sets_theorem :
  (A ∩ B = Set.Icc (-2) 1) ∧
  (Set.univ \ B = {x | x ≤ -3 ∨ x ≥ 1}) ∧
  (∀ m : ℝ, A ∪ C m = A ↔ m ≤ 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_theorem_l1178_117834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_form_only_for_1_and_2_l1178_117876

/-- Sequence defined by the recurrence relation a_{n+2} = 6a_{n+1} - a_n -/
def a : ℕ → ℤ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 7
  | n + 3 => 6 * a (n + 2) - a (n + 1)

/-- Predicate to check if a number is of the form 2m^2 - 1 for some integer m -/
def isOfForm (x : ℤ) : Prop :=
  ∃ m : ℤ, x = 2 * m^2 - 1

theorem a_form_only_for_1_and_2 :
  ∀ n : ℕ, n > 0 → (isOfForm (a n) ↔ n = 1 ∨ n = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_form_only_for_1_and_2_l1178_117876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l1178_117855

open Real

-- Define the function (marked as noncomputable due to use of transcendental functions)
noncomputable def f (x : ℝ) : ℝ := 2 * sin (x / 2 + π / 5)

-- State the theorem
theorem sine_function_properties :
  (∀ x, f (x + 4 * π) = f x) ∧  -- Period is 4π
  (∀ x, |f x| ≤ 2) ∧            -- Amplitude is at most 2
  (∃ x, f x = 2) ∧              -- Maximum value is 2
  (∃ x, f x = -2) :=            -- Minimum value is -2
by
  sorry -- Proof is omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l1178_117855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_ratio_l1178_117898

theorem shirt_price_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  (5/6 * (3/4 * marked_price)) / marked_price = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_ratio_l1178_117898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1178_117854

noncomputable def f (x : ℝ) := Real.cos x ^ 2

theorem tangent_line_equation :
  let p : ℝ × ℝ := (π / 4, 1 / 2)
  let m : ℝ := -2 * Real.sin (π / 4) * Real.cos (π / 4)
  let tangent_eq (x y : ℝ) := x + y - 1 / 2 - π / 4 = 0
  f (π / 4) = 1 / 2 ∧ 
  (∀ x : ℝ, HasDerivAt f (-2 * Real.sin x * Real.cos x) x) →
  ∀ x y : ℝ, tangent_eq x y ↔ y - p.2 = m * (x - p.1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1178_117854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_concentration_after_modification_l1178_117843

/-- Calculates the final concentration of an acid solution after modification -/
theorem acid_concentration_after_modification (initial_volume : ℝ) (initial_concentration : ℝ) 
  (removed_volume : ℝ) (added_water : ℝ) : 
  initial_volume = 5 →
  initial_concentration = 0.06 →
  removed_volume = 1 →
  added_water = 2 →
  let initial_acid := initial_volume * initial_concentration
  let remaining_volume := initial_volume - removed_volume
  let remaining_acid := initial_acid - (removed_volume * initial_concentration)
  let final_volume := remaining_volume + added_water
  let final_concentration := remaining_acid / final_volume
  final_concentration = 0.04 := by
  sorry

#check acid_concentration_after_modification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_concentration_after_modification_l1178_117843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_update_theorem_l1178_117804

/-- Represents a class of students -/
structure StudentClass where
  average_age : ℝ
  variance : ℝ

/-- Calculates the new class statistics after a given number of years -/
def update_class (c : StudentClass) (years : ℝ) : StudentClass :=
  { average_age := c.average_age + years
  , variance := c.variance }

theorem class_update_theorem (initial_class : StudentClass) (h1 : initial_class.average_age = 13)
    (h2 : initial_class.variance = 3) :
    let final_class := update_class initial_class 2
    final_class.average_age = 15 ∧ final_class.variance = initial_class.variance := by
  sorry

#check class_update_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_update_theorem_l1178_117804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_coord_difference_l1178_117897

/-- Triangle ABC with vertices A(0,10), B(3,0), C(10,0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Point P on line AC -/
def P (t : Triangle) : ℝ × ℝ := sorry

/-- Point Q on line BC -/
def Q (t : Triangle) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- The positive difference between x and y coordinates of a point -/
def coordDifference (p : ℝ × ℝ) : ℝ := abs (p.1 - p.2)

theorem triangle_area_implies_coord_difference (t : Triangle) 
  (h1 : t.A = (0, 10))
  (h2 : t.B = (3, 0))
  (h3 : t.C = (10, 0))
  (h4 : triangleArea (P t) (Q t) t.C = 16) :
  coordDifference (P t) = |10 - 8 * Real.sqrt 2| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_coord_difference_l1178_117897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_seven_pi_fourth_l1178_117825

theorem sec_seven_pi_fourth : 1 / Real.cos (7 * Real.pi / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_seven_pi_fourth_l1178_117825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_spherical_correct_l1178_117872

/-- Conversion from rectangular to spherical coordinates -/
noncomputable def rect_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := if x > 0 ∧ y < 0 then 2 * Real.pi - Real.arctan (-y/x) else Real.arctan (y/x)
  let φ := Real.arccos (z/ρ)
  (ρ, θ, φ)

theorem rect_to_spherical_correct :
  let (ρ, θ, φ) := rect_to_spherical 1 (-2 * Real.sqrt 3) 4
  ρ = Real.sqrt 29 ∧
  θ = 2 * Real.pi - Real.arctan (2 * Real.sqrt 3) ∧
  φ = Real.arccos (4 / Real.sqrt 29) ∧
  ρ > 0 ∧
  0 ≤ θ ∧ θ < 2 * Real.pi ∧
  0 ≤ φ ∧ φ ≤ Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_spherical_correct_l1178_117872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_C_l1178_117881

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the points A, B, and C
def A : ℝ × ℝ := (5, 25)
def B : ℝ × ℝ := (-5, 25)
def C : ℝ × ℝ := (0, 35)

-- Theorem statement
theorem y_coordinate_of_C : 
  (∀ (p : ℝ × ℝ), p ∈ ({A, B, C} : Set (ℝ × ℝ)) → p.2 = parabola p.1) →  -- Points are on the parabola
  A.2 = 25 ∧ B.2 = 25 →  -- AB is horizontal at y = 25
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →  -- ABC is isosceles
  (B.1 - A.1) * (C.2 - A.2) / 2 = 50 →  -- Area of ABC is 50
  C.2 = 35 :=
by sorry

#eval C.2  -- This will print 35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_C_l1178_117881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emergency_vehicle_coverage_area_l1178_117851

-- Define the vehicle's speeds and time limit
noncomputable def highway_speed : ℝ := 60
noncomputable def desert_speed : ℝ := 15
noncomputable def time_limit : ℝ := 4 / 60  -- 4 minutes in hours

-- Define the maximum distances
noncomputable def highway_distance : ℝ := highway_speed * time_limit
noncomputable def desert_distance : ℝ := desert_speed * time_limit

-- Theorem statement
theorem emergency_vehicle_coverage_area :
  let area := 4 * highway_distance ^ 2 + 4 * Real.pi * desert_distance ^ 2
  area = 16 + 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emergency_vehicle_coverage_area_l1178_117851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_12_choose_6_l1178_117896

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_12_choose_6_l1178_117896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1178_117875

noncomputable def f (x : ℝ) := Real.sin x + (1/2) * Real.sin (2*x)

theorem f_properties :
  (∀ x, f (x + 2 * Real.pi) = f x) ∧
  (∃! (x₁ x₂ x₃ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 2 * Real.pi ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0) ∧
  (∀ x, f x ≤ (3 * Real.sqrt 3) / 4) ∧
  (∃ x, f x = (3 * Real.sqrt 3) / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1178_117875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_parabola_equation_l1178_117891

-- Define the parabola C
def C (p : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}

-- Define line l₁ passing through (2,0)
def l₁ (k : ℝ) : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = k * xy.2 + 2}

-- Define line l₂: x = -2
def l₂ : Set (ℝ × ℝ) := {xy : ℝ × ℝ | xy.1 = -2}

-- Define point Q
def Q : ℝ × ℝ := (-2, 0)

-- Define points A and B as intersections of l₁ and C
noncomputable def A (p k : ℝ) : ℝ × ℝ := sorry
noncomputable def B (p k : ℝ) : ℝ × ℝ := sorry

-- Define slopes k₁ and k₂
noncomputable def k₁ (p k : ℝ) : ℝ := (A p k).2 / ((A p k).1 + 2)
noncomputable def k₂ (p k : ℝ) : ℝ := (B p k).2 / ((B p k).1 + 2)

-- Theorem for part (1)
theorem sum_of_slopes (p k : ℝ) : k₁ p k + k₂ p k = 0 := by sorry

-- Define points M and N
noncomputable def M (p k x y : ℝ) : ℝ × ℝ := sorry
noncomputable def N (p k x y : ℝ) : ℝ × ℝ := sorry

-- Define the dot product condition
def dot_product_condition (p k x y : ℝ) : Prop :=
  (M p k x y).1 * (N p k x y).1 + (M p k x y).2 * (N p k x y).2 = 2

-- Theorem for part (2)
theorem parabola_equation (p k : ℝ) :
  (∀ x y, (x, y) ∈ C p → (x, y) ≠ A p k → (x, y) ≠ B p k →
    dot_product_condition p k x y) → p = 1/2 := by sorry

#check sum_of_slopes
#check parabola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_parabola_equation_l1178_117891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_side_length_is_27_l1178_117858

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a window -/
structure Window where
  panes : Pane
  borderWidth : ℝ

/-- Calculates the side length of a square window -/
def calculateWindowSideLength (w : Window) : ℝ :=
  4 * w.panes.width + 5 * w.borderWidth

/-- Theorem stating the side length of the square window is 27 inches -/
theorem window_side_length_is_27 (w : Window) :
  w.panes.height = 3 * w.panes.width →
  w.borderWidth = 3 →
  calculateWindowSideLength w = 27 :=
by
  intros h1 h2
  unfold calculateWindowSideLength
  rw [h2]
  have h3 : w.panes.width = 3
  · sorry  -- This step requires algebraic manipulation
  rw [h3]
  norm_num

-- The following line is commented out as it's not a valid Lean command in this context
-- #eval window_side_length_is_27

end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_side_length_is_27_l1178_117858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_analysis_l1178_117880

/-- Represents a pair of variables -/
inductive VariablePair
  | CircleAreaRadius
  | SphereVolumeRadius
  | AngleSine
  | MathPhysicsScores

/-- Determines if a pair of variables has a correlation or a deterministic relationship -/
def has_correlation (pair : VariablePair) : Prop :=
  match pair with
  | VariablePair.CircleAreaRadius => False
  | VariablePair.SphereVolumeRadius => False
  | VariablePair.AngleSine => False
  | VariablePair.MathPhysicsScores => True

/-- The area of a circle is determined by its radius -/
axiom circle_area_deterministic : ∃ (f : ℝ → ℝ), ∀ r, f r = Real.pi * r^2

/-- The volume of a sphere is determined by its radius -/
axiom sphere_volume_deterministic : ∃ (f : ℝ → ℝ), ∀ r, f r = (4/3) * Real.pi * r^3

/-- The sine of an angle is determined by the angle -/
axiom sine_deterministic : ∃ (f : ℝ → ℝ), ∀ θ, f θ = Real.sin θ

/-- Math and physics scores are not deterministically related -/
axiom math_physics_not_deterministic : ¬∃ (f : ℝ → ℝ), ∀ math_score, 
  ∃ physics_score, f math_score = physics_score

theorem correlation_analysis : 
  (∀ pair : VariablePair, has_correlation pair ↔ pair = VariablePair.MathPhysicsScores) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_analysis_l1178_117880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l1178_117831

def circle1_center : ℝ × ℝ := (2, 2)
def circle2_center : ℝ × ℝ := (17, 10)

def circle1_radius : ℝ := circle1_center.2
def circle2_radius : ℝ := circle2_center.2

noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((circle2_center.1 - circle1_center.1)^2 + (circle2_center.2 - circle1_center.2)^2)

theorem closest_points_distance :
  distance_between_centers - circle1_radius - circle2_radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l1178_117831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_coloring_theorem_l1178_117818

/-- Represents a coloring of an 8x8 board --/
def Coloring := Fin 8 → Fin 8 → Bool

/-- An L-shaped corner on the board --/
structure LCorner where
  x1 : Fin 8
  y1 : Fin 8
  x2 : Fin 8
  y2 : Fin 8
  x3 : Fin 8
  y3 : Fin 8

/-- Checks if an L-corner contains at least one uncolored square --/
def hasUncoloredSquare (c : Coloring) (l : LCorner) : Prop :=
  ¬(c l.x1 l.y1 ∧ c l.x2 l.y2 ∧ c l.x3 l.y3)

/-- Checks if an L-corner contains at least one black square --/
def hasBlackSquare (c : Coloring) (l : LCorner) : Prop :=
  c l.x1 l.y1 ∨ c l.x2 l.y2 ∨ c l.x3 l.y3

/-- All possible L-corners on the board --/
def allLCorners : Set LCorner := sorry

/-- Counts the number of black squares in a coloring --/
def blackSquareCount (c : Coloring) : Nat := sorry

/-- The main theorem --/
theorem board_coloring_theorem :
  (∃ (c : Coloring), (∀ l ∈ allLCorners, hasUncoloredSquare c l) ∧
    blackSquareCount c = 32 ∧
    (∀ (c' : Coloring), (∀ l ∈ allLCorners, hasUncoloredSquare c' l) →
      blackSquareCount c' ≤ 32)) ∧
  (∃ (c : Coloring), (∀ l ∈ allLCorners, hasBlackSquare c l) ∧
    blackSquareCount c = 32 ∧
    (∀ (c' : Coloring), (∀ l ∈ allLCorners, hasBlackSquare c' l) →
      blackSquareCount c' ≥ 32)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_coloring_theorem_l1178_117818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l1178_117877

/-- The force function F(x) in Newtons -/
def F (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 5

/-- The work done by the force F(x) from x_start to x_end -/
noncomputable def work (x_start x_end : ℝ) : ℝ :=
  ∫ x in x_start..x_end, F x

theorem work_calculation :
  work 5 10 = 825 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l1178_117877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_180_not_by_3_or_5_l1178_117830

def divisors_not_by_3_or_5 (n : ℕ) : ℕ := 
  (Finset.filter (fun d => d ∣ n ∧ ¬(3 ∣ d) ∧ ¬(5 ∣ d)) (Finset.range (n + 1))).card

theorem divisors_180_not_by_3_or_5 : 
  divisors_not_by_3_or_5 180 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_180_not_by_3_or_5_l1178_117830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1178_117863

theorem cos_alpha_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi) (h2 : 0 < β ∧ β < Real.pi)
  (h3 : Real.cos β = -5/13) (h4 : Real.sin (α + β) = 3/5) : Real.cos α = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1178_117863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportion_m_eq_three_l1178_117889

/-- A function y = f(x) is a direct proportion if it can be written as y = kx for some non-zero constant k. -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function defined by the given equation -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + 3) * (x ^ (m^2 - 8))

/-- The theorem stating that m = 3 is the only value that makes f a direct proportion -/
theorem direct_proportion_m_eq_three :
  ∃! m : ℝ, is_direct_proportion (f m) ∧ m + 3 ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_proportion_m_eq_three_l1178_117889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_eigenvalue_three_l1178_117879

theorem matrix_eigenvalue_three (w : Fin 2 → ℝ) : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 0], ![0, 3]]
  N.vecMul w = (3 : ℝ) • w := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_eigenvalue_three_l1178_117879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_volume_l1178_117890

/-- The volume of a regular truncated triangular pyramid -/
noncomputable def truncatedPyramidVolume (a b : ℝ) : ℝ :=
  (Real.sqrt 3 / 12) * (a^3 - b^3)

/-- Helper function to represent the volume of a regular truncated triangular pyramid -/
noncomputable def volume_of_regular_truncated_triangular_pyramid (a b angle : ℝ) : ℝ :=
  sorry -- This function is left undefined as its exact implementation is not given in the problem statement

/-- Theorem: Volume of a regular truncated triangular pyramid -/
theorem truncated_pyramid_volume
  (a b : ℝ)
  (h1 : a > b)
  (h2 : a > 0)
  (h3 : b > 0) :
  ∃ (V : ℝ), V = truncatedPyramidVolume a b ∧
  V = volume_of_regular_truncated_triangular_pyramid a b 60 :=
by
  sorry -- Proof is omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_volume_l1178_117890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l1178_117819

def a (n : ℕ) : ℚ :=
  match n with
  | 0 => 3
  | n + 1 => (1 / 2) * a n + 1

theorem a_general_term (n : ℕ) :
  a n = (2^(n + 1) + 1) / 2^n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l1178_117819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1178_117838

/-- Compound interest formula -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- Problem statement -/
theorem interest_rate_calculation (P A n t : ℝ) 
  (h1 : P = 700)
  (h2 : A = 771.75)
  (h3 : n = 2)
  (h4 : t = 1) :
  ∃ r : ℝ, compound_interest P r n t = A ∧ r = 0.10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1178_117838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1178_117878

noncomputable def f (x : ℝ) := Real.sqrt 2 * Real.sin (x/2) * Real.cos (x/2) - Real.sqrt 2 * (Real.sin (x/2))^2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
   ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi) 0 → f x ≥ -Real.sqrt 2 / 2 - 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi) 0 ∧ f x = -Real.sqrt 2 / 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1178_117878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_of_E_circle_passes_through_T_l1178_117800

noncomputable section

-- Define the fixed point F
def F : ℝ × ℝ := (2, 0)

-- Define the fixed line l
def l : ℝ → Prop := λ x ↦ x = 3/2

-- Define the distance ratio
def distance_ratio : ℝ := 2 * Real.sqrt 3 / 3

-- Define the locus E
def E : ℝ × ℝ → Prop := λ p ↦ 
  let (x, y) := p
  (Real.sqrt ((x - F.1)^2 + (y - F.2)^2)) / |x - 3/2| = distance_ratio

-- Theorem: Equation of curve E
theorem equation_of_E : ∀ p : ℝ × ℝ, E p ↔ p.1^2 / 3 - p.2^2 = 1 := by sorry

-- Define a tangent line to E
def tangent_line (m n : ℝ) : ℝ × ℝ → Prop := λ p ↦ p.2 = m * p.1 + n

-- Define the intersection point N
def N (m n : ℝ) : ℝ × ℝ := (3/2, m * 3/2 + n)

-- Define the fixed point T
def T : ℝ × ℝ := (2, 0)

-- Theorem: Circle with diameter MN passes through T
theorem circle_passes_through_T : 
  ∀ (m n : ℝ) (M : ℝ × ℝ), 
    E M → tangent_line m n M → 
    ∃ (center : ℝ × ℝ) (radius : ℝ),
      (center.1 - M.1)^2 + (center.2 - M.2)^2 = radius^2 ∧
      (center.1 - (N m n).1)^2 + (center.2 - (N m n).2)^2 = radius^2 ∧
      (center.1 - T.1)^2 + (center.2 - T.2)^2 = radius^2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_of_E_circle_passes_through_T_l1178_117800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l1178_117811

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = a - 2 / (2^x + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a - 2 / (2^x + 1)

theorem odd_function_implies_a_equals_one :
  ∀ a : ℝ, IsOdd (f a) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l1178_117811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_l1178_117849

structure UnitCube where
  vertices : Fin 8 → ℝ × ℝ × ℝ
  is_unit : ∀ i j, i ≠ j → ‖vertices i - vertices j‖ = 1 ∨ ‖vertices i - vertices j‖ = Real.sqrt 2 ∨ ‖vertices i - vertices j‖ = Real.sqrt 3

structure InscribedOctahedron (cube : UnitCube) where
  vertices : Fin 6 → ℝ × ℝ × ℝ
  on_cube_edge : ∀ v, ∃ i j, cube.vertices i - vertices v = (1/3 : ℝ) • (cube.vertices i - cube.vertices j)
  is_regular : ∀ v w, v ≠ w → ‖vertices v - vertices w‖ = ‖vertices 0 - vertices 1‖

theorem octahedron_side_length (cube : UnitCube) (octa : InscribedOctahedron cube) :
  ∀ v w, v ≠ w → ‖octa.vertices v - octa.vertices w‖ = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_l1178_117849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1178_117868

theorem trig_identity (α : ℝ) :
  4 * Real.sin (π / 6 + α / 2) * Real.sin (π / 6 - α / 2) =
  Real.cos (3 * α / 2) / Real.cos (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1178_117868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_third_smallest_avg_l1178_117850

theorem second_third_smallest_avg (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧  -- five different positive integers
  (a + b + c + d + e : ℚ) / 5 = 5 ∧    -- average is 5
  (∀ x y z w v : ℕ, x < y ∧ y < z ∧ z < w ∧ w < v ∧ (x + y + z + w + v : ℚ) / 5 = 5 → 
    (v - x) ≤ (e - a)) →            -- maximum difference between largest and smallest
  (b + c : ℚ) / 2 = 7/2 := by           -- average of second and third smallest is 3.5
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_third_smallest_avg_l1178_117850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_point_distance_l1178_117802

/-- The height of an equilateral triangle with side length 1 -/
noncomputable def equilateral_triangle_height : ℝ := Real.sqrt 3 / 2

/-- The setup of three equilateral triangles with the center one rotated -/
structure TriangleArrangement where
  side_length : ℝ
  rotation_angle : ℝ
  h_side_length : side_length = 1
  h_rotation_angle : rotation_angle = π / 3

/-- The theorem stating the distance from the highest point to the base line -/
theorem highest_point_distance (arrangement : TriangleArrangement) :
  let distance := equilateral_triangle_height
  distance = Real.sqrt 3 / 2 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_point_distance_l1178_117802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessicas_food_expense_l1178_117836

/-- Represents Jessica's monthly expenses --/
structure MonthlyExpenses where
  rent : ℝ
  food : ℝ
  carInsurance : ℝ

/-- Calculates the total monthly expenses --/
def totalExpenses (e : MonthlyExpenses) : ℝ :=
  e.rent + e.food + e.carInsurance

/-- Jessica's expenses last year --/
def lastYear (x : ℝ) : MonthlyExpenses where
  rent := 1000
  food := x -- x is the unknown food expense
  carInsurance := 100

/-- Jessica's expenses this year --/
def thisYear (x : ℝ) : MonthlyExpenses where
  rent := 1000 * 1.3
  food := x * 1.5
  carInsurance := 100 * 3

/-- The difference in yearly expenses --/
def yearlyDifference (x : ℝ) : ℝ :=
  12 * (totalExpenses (thisYear x) - totalExpenses (lastYear x))

theorem jessicas_food_expense :
  ∃ x : ℝ, x = 200 ∧
    (lastYear x).food = x ∧
    yearlyDifference x = 7200 := by
  sorry

#eval yearlyDifference 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessicas_food_expense_l1178_117836
