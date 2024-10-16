import Mathlib

namespace NUMINAMATH_CALUDE_acute_angle_theorem_l751_75115

def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

theorem acute_angle_theorem (θ : ℝ) (h1 : is_acute_angle θ) 
  (h2 : 4 * (90 - θ) = (180 - θ) + 60) : θ = 40 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_theorem_l751_75115


namespace NUMINAMATH_CALUDE_power_greater_than_square_plus_one_l751_75108

theorem power_greater_than_square_plus_one (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_greater_than_square_plus_one_l751_75108


namespace NUMINAMATH_CALUDE_soft_taco_price_is_correct_l751_75106

/-- The price of a hard shell taco -/
def hard_shell_price : ℝ := 5

/-- The number of hard shell tacos bought by the family -/
def family_hard_shells : ℕ := 4

/-- The number of soft tacos bought by the family -/
def family_soft_shells : ℕ := 3

/-- The number of other customers -/
def other_customers : ℕ := 10

/-- The number of soft tacos bought by each other customer -/
def soft_tacos_per_customer : ℕ := 2

/-- The total revenue during lunch rush -/
def total_revenue : ℝ := 66

/-- The price of a soft taco -/
def soft_taco_price : ℝ := 2

theorem soft_taco_price_is_correct :
  soft_taco_price * (family_soft_shells + other_customers * soft_tacos_per_customer) +
  hard_shell_price * family_hard_shells = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_soft_taco_price_is_correct_l751_75106


namespace NUMINAMATH_CALUDE_binary_11011_equals_27_l751_75137

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_equals_27_l751_75137


namespace NUMINAMATH_CALUDE_product_equality_l751_75152

noncomputable def P : ℝ := Real.sqrt 1011 + Real.sqrt 1012
noncomputable def Q : ℝ := -Real.sqrt 1011 - Real.sqrt 1012
noncomputable def R : ℝ := Real.sqrt 1011 - Real.sqrt 1012
noncomputable def S : ℝ := Real.sqrt 1012 - Real.sqrt 1011

theorem product_equality : (P * Q)^2 * R * S = 8136957 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l751_75152


namespace NUMINAMATH_CALUDE_sqrt_68_minus_sqrt_64_approx_l751_75179

theorem sqrt_68_minus_sqrt_64_approx : |Real.sqrt 68 - Real.sqrt 64 - 0.24| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_68_minus_sqrt_64_approx_l751_75179


namespace NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l751_75101

-- Define the function f
def f (x m n : ℝ) : ℝ := 2 * x^3 + 3 * m * x^2 + 3 * n * x - 6

-- Define the derivative of f
def f' (x m n : ℝ) : ℝ := 6 * x^2 + 6 * m * x + 3 * n

theorem extreme_values_and_monotonicity :
  ∃ (m n : ℝ),
    (f' 1 m n = 0 ∧ f' 2 m n = 0) ∧
    (m = -3 ∧ n = 4) ∧
    (∀ x, x < 1 → (f' x m n > 0)) ∧
    (∀ x, 1 < x ∧ x < 2 → (f' x m n < 0)) ∧
    (∀ x, x > 2 → (f' x m n > 0)) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l751_75101


namespace NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l751_75103

/-- A parabola is tangent to a hyperbola if and only if m is 4 or 8 -/
theorem parabola_tangent_hyperbola :
  ∀ m : ℝ,
  (∀ x y : ℝ, y = x^2 + 3 ∧ y^2 - m*x^2 = 4 →
    ∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y) ↔ (m = 4 ∨ m = 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l751_75103


namespace NUMINAMATH_CALUDE_log_sum_equality_l751_75119

theorem log_sum_equality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (1 / (1 + Real.log (c/a) / Real.log (a^2 * b))) +
  (1 / (1 + Real.log (a/b) / Real.log (b^2 * c))) +
  (1 / (1 + Real.log (b/c) / Real.log (c^2 * a))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l751_75119


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l751_75150

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote
def asymptote (a b x y : ℝ) : Prop := y = (b / a) * x ∨ y = -(b / a) * x

-- Define the perpendicularity condition
def perpendicular (F₁ F₂ A : ℝ × ℝ) : Prop :=
  (A.2 - F₂.2) * (F₂.1 - F₁.1) = (F₂.2 - F₁.2) * (A.1 - F₂.1)

-- Define the distance condition
def distance_condition (O F₁ A : ℝ × ℝ) : Prop :=
  let d := abs ((A.2 - F₁.2) * O.1 - (A.1 - F₁.1) * O.2 + A.1 * F₁.2 - A.2 * F₁.1) /
            Real.sqrt ((A.2 - F₁.2)^2 + (A.1 - F₁.1)^2)
  d = (1/3) * Real.sqrt (F₁.1^2 + F₁.2^2)

-- Main theorem
theorem hyperbola_asymptote_slope (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (A F₁ F₂ O : ℝ × ℝ) :
  hyperbola a b A.1 A.2 →
  asymptote a b A.1 A.2 →
  perpendicular F₁ F₂ A →
  distance_condition O F₁ A →
  (b / a = Real.sqrt 2 / 2) ∨ (b / a = -Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l751_75150


namespace NUMINAMATH_CALUDE_gcd_420_882_l751_75141

theorem gcd_420_882 : Nat.gcd 420 882 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_420_882_l751_75141


namespace NUMINAMATH_CALUDE_exponent_properties_l751_75176

theorem exponent_properties (a m n : ℝ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) (h4 : n > 1) :
  (a^(m/n) = (a^m)^(1/n)) ∧ 
  (a^0 = 1) ∧ 
  (a^(-m/n) = 1 / (a^m)^(1/n)) := by
sorry

end NUMINAMATH_CALUDE_exponent_properties_l751_75176


namespace NUMINAMATH_CALUDE_corrected_mean_l751_75127

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 40 →
  incorrect_value = 15 →
  correct_value = 45 →
  let original_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := original_sum + difference
  corrected_sum / n = 40.6 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l751_75127


namespace NUMINAMATH_CALUDE_f_max_value_f_specific_values_f_explicit_formula_f_max_over_interval_f_min_over_interval_l751_75168

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -(x + 2)^2 + 3

/-- The maximum value of f is 3 -/
theorem f_max_value : ∀ x : ℝ, f x ≤ 3 := by sorry

/-- f(-4) = f(0) = -1 -/
theorem f_specific_values : f (-4) = -1 ∧ f 0 = -1 := by sorry

/-- The explicit formula for f(x) -/
theorem f_explicit_formula : ∀ x : ℝ, f x = -(x + 2)^2 + 3 := by sorry

/-- The maximum value of f(x) over [-3, 3] is 3 -/
theorem f_max_over_interval : 
  ∀ x : ℝ, x ∈ Set.Icc (-3) 3 → f x ≤ 3 ∧ ∃ y ∈ Set.Icc (-3) 3, f y = 3 := by sorry

/-- The minimum value of f(x) over [-3, 3] is -22 -/
theorem f_min_over_interval : 
  ∀ x : ℝ, x ∈ Set.Icc (-3) 3 → f x ≥ -22 ∧ ∃ y ∈ Set.Icc (-3) 3, f y = -22 := by sorry

end NUMINAMATH_CALUDE_f_max_value_f_specific_values_f_explicit_formula_f_max_over_interval_f_min_over_interval_l751_75168


namespace NUMINAMATH_CALUDE_evaluate_power_l751_75130

theorem evaluate_power (x : ℝ) (h : x = 81) : x^(5/4) = 243 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_power_l751_75130


namespace NUMINAMATH_CALUDE_value_of_equation_l751_75110

theorem value_of_equation (x y V : ℝ) 
  (eq1 : x + |x| + y = V)
  (eq2 : x + |y| - y = 6)
  (eq3 : x + y = 12) :
  V = 18 := by
  sorry

end NUMINAMATH_CALUDE_value_of_equation_l751_75110


namespace NUMINAMATH_CALUDE_banquet_food_consumption_l751_75193

/-- The total food consumed at a banquet is at least the product of the minimum number of guests and the maximum food consumed per guest. -/
theorem banquet_food_consumption 
  (max_food_per_guest : ℝ) 
  (min_guests : ℕ) 
  (h1 : max_food_per_guest = 2) 
  (h2 : min_guests = 162) : 
  ℝ := by
  sorry

#eval (2 : ℝ) * 162  -- Expected output: 324

end NUMINAMATH_CALUDE_banquet_food_consumption_l751_75193


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l751_75117

theorem chord_length_concentric_circles 
  (area_ring : ℝ) 
  (radius_small : ℝ) 
  (chord_length : ℝ) :
  area_ring = 50 * Real.pi ∧ 
  radius_small = 5 →
  chord_length = 10 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l751_75117


namespace NUMINAMATH_CALUDE_grandfathers_age_l751_75147

theorem grandfathers_age (x : ℕ) (y z : ℕ) : 
  (6 * x = 6 * x) →  -- Current year
  (6 * x + y = 5 * (x + y)) →  -- In y years
  (6 * x + y + z = 4 * (x + y + z)) →  -- In y + z years
  (x > 0) →  -- Ming's age is positive
  (y > 0) →  -- First time gap is positive
  (z > 0) →  -- Second time gap is positive
  6 * x = 72 := by
  sorry

end NUMINAMATH_CALUDE_grandfathers_age_l751_75147


namespace NUMINAMATH_CALUDE_system_solution_l751_75104

theorem system_solution (x : Fin 1995 → ℤ) 
  (h : ∀ i : Fin 1995, x i ^ 2 = 1 + x ((i + 1993) % 1995) * x ((i + 1994) % 1995)) :
  (∀ i : Fin 1995, i % 3 = 1 → x i = 0) ∧
  (∀ i : Fin 1995, i % 3 ≠ 1 → x i = 1 ∨ x i = -1) ∧
  (∀ i : Fin 1995, i % 3 ≠ 1 → x i = -x ((i + 1) % 1995)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l751_75104


namespace NUMINAMATH_CALUDE_like_terms_power_l751_75126

theorem like_terms_power (a b : ℕ) : 
  (∀ x y : ℝ, ∃ c : ℝ, x^(a+1) * y^2 = c * x^3 * y^b) → 
  a^b = 4 := by
sorry

end NUMINAMATH_CALUDE_like_terms_power_l751_75126


namespace NUMINAMATH_CALUDE_students_before_yoongi_l751_75149

theorem students_before_yoongi (total_students : ℕ) (finished_after_yoongi : ℕ) : 
  total_students = 20 → finished_after_yoongi = 11 → 
  total_students - finished_after_yoongi - 1 = 8 := by
sorry

end NUMINAMATH_CALUDE_students_before_yoongi_l751_75149


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l751_75118

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ -1}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l751_75118


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l751_75164

/-- Given a line L1 with equation x - 2y + 1 = 0, 
    and a line of symmetry y = x,
    the line L2 symmetric to L1 with respect to y = x
    has the equation x + 2y - 1 = 0 -/
theorem symmetric_line_equation :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ x - 2*y + 1 = 0
  let symmetry_line : ℝ → ℝ → Prop := λ x y ↦ y = x
  let L2 : ℝ → ℝ → Prop := λ x y ↦ x + 2*y - 1 = 0
  ∀ x y : ℝ, L2 x y ↔ (∃ x' y' : ℝ, L1 x' y' ∧ 
    (x = (x' + y')/2 ∧ y = (x' + y')/2))
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l751_75164


namespace NUMINAMATH_CALUDE_locus_of_intersection_points_l751_75145

/-- The locus of intersection points of perpendiculars drawn from a circle's 
    intersections with two perpendicular lines. -/
theorem locus_of_intersection_points (u v x y : ℝ) :
  (u ≠ v ∨ u ≠ -v) →
  (∃ (r : ℝ), r > 0 ∧ r > |u| ∧ r > |v| ∧
    (x - u)^2 / (u^2 - v^2) - (y - v)^2 / (u^2 - v^2) = 1) ∨
  (u = v ∨ u = -v) →
    (x - y) * (x + y) = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_intersection_points_l751_75145


namespace NUMINAMATH_CALUDE_no_equal_arithmetic_operations_l751_75189

theorem no_equal_arithmetic_operations (v t : ℝ) (hv : v > 0) (ht : t > 0) : 
  ¬(v + t = v * t ∧ v + t = v / t) :=
by sorry

end NUMINAMATH_CALUDE_no_equal_arithmetic_operations_l751_75189


namespace NUMINAMATH_CALUDE_fraction_square_simplification_l751_75172

theorem fraction_square_simplification (a b c : ℝ) (ha : a ≠ 0) :
  (3 * b * c / (-2 * a^2))^2 = 9 * b^2 * c^2 / (4 * a^4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_square_simplification_l751_75172


namespace NUMINAMATH_CALUDE_incenter_centroid_parallel_implies_arithmetic_sequence_l751_75159

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle. -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle. -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Checks if two points form a line parallel to a side of the triangle. -/
def is_parallel_to_side (p q : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- The lengths of the sides of a triangle. -/
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- Checks if three numbers form an arithmetic sequence. -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := sorry

theorem incenter_centroid_parallel_implies_arithmetic_sequence (t : Triangle) :
  is_parallel_to_side (incenter t) (centroid t) t →
  let (a, b, c) := side_lengths t
  is_arithmetic_sequence a b c := by sorry

end NUMINAMATH_CALUDE_incenter_centroid_parallel_implies_arithmetic_sequence_l751_75159


namespace NUMINAMATH_CALUDE_quadratic_function_property_l751_75196

theorem quadratic_function_property (a b c : ℝ) :
  (∀ x, (1 < x ∧ x < c) → (a * x^2 + b * x + c < 0)) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l751_75196


namespace NUMINAMATH_CALUDE_expansion_terms_count_l751_75102

/-- The number of terms in the simplified expansion of (x+y+z)^2010 + (x-y-z)^2010 -/
def num_terms : ℕ := 1012036

/-- The exponent in the expression (x+y+z)^n + (x-y-z)^n -/
def exponent : ℕ := 2010

theorem expansion_terms_count :
  num_terms = (exponent / 2 + 1)^2 := by sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l751_75102


namespace NUMINAMATH_CALUDE_sozopolian_inequality_sozopolian_equality_l751_75177

/-- Definition of a Sozopolian set -/
def is_sozopolian (p a b c : ℕ) : Prop :=
  Nat.Prime p ∧ p % 2 = 1 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a * b + 1) % p = 0 ∧
  (b * c + 1) % p = 0 ∧
  (c * a + 1) % p = 0

theorem sozopolian_inequality (p a b c : ℕ) :
  is_sozopolian p a b c → p + 2 ≤ (a + b + c) / 3 :=
sorry

theorem sozopolian_equality (p : ℕ) :
  (∃ a b c : ℕ, is_sozopolian p a b c ∧ p + 2 = (a + b + c) / 3) ↔ p = 5 :=
sorry

end NUMINAMATH_CALUDE_sozopolian_inequality_sozopolian_equality_l751_75177


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l751_75109

theorem at_least_one_greater_than_one (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) : max x y > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l751_75109


namespace NUMINAMATH_CALUDE_assistant_productivity_increase_l751_75123

theorem assistant_productivity_increase 
  (original_bears : ℝ) 
  (original_hours : ℝ) 
  (bear_increase_rate : ℝ) 
  (hour_decrease_rate : ℝ) 
  (h₁ : bear_increase_rate = 0.8) 
  (h₂ : hour_decrease_rate = 0.1) 
  : (((1 + bear_increase_rate) * original_bears) / ((1 - hour_decrease_rate) * original_hours)) / 
    (original_bears / original_hours) - 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_assistant_productivity_increase_l751_75123


namespace NUMINAMATH_CALUDE_max_square_plots_l751_75175

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available internal fencing -/
def available_fencing : ℕ := 1994

/-- Calculates the number of square plots given the side length -/
def num_plots (dims : FieldDimensions) (side_length : ℕ) : ℕ :=
  (dims.width / side_length) * (dims.length / side_length)

/-- Calculates the required internal fencing for a given configuration -/
def required_fencing (dims : FieldDimensions) (side_length : ℕ) : ℕ :=
  (dims.width / side_length - 1) * dims.length + (dims.length / side_length - 1) * dims.width

/-- Theorem stating that 78 is the maximum number of square plots -/
theorem max_square_plots (dims : FieldDimensions) 
    (h_width : dims.width = 24) 
    (h_length : dims.length = 52) : 
    ∀ side_length : ℕ, 
      side_length > 0 → 
      dims.width % side_length = 0 → 
      dims.length % side_length = 0 → 
      required_fencing dims side_length ≤ available_fencing → 
      num_plots dims side_length ≤ 78 :=
  sorry

#check max_square_plots

end NUMINAMATH_CALUDE_max_square_plots_l751_75175


namespace NUMINAMATH_CALUDE_income_comparison_l751_75161

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = 0.8 * juan) 
  (h2 : mary = 1.28 * juan) : 
  (mary - tim) / tim = 0.6 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l751_75161


namespace NUMINAMATH_CALUDE_simplify_fraction_l751_75160

theorem simplify_fraction (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 2*b) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l751_75160


namespace NUMINAMATH_CALUDE_calcium_oxide_molecular_weight_l751_75148

/-- Represents an element with its atomic weight -/
structure Element where
  name : String
  atomic_weight : ℝ

/-- Represents a compound made of elements -/
structure Compound where
  name : String
  elements : List Element

/-- Calculates the molecular weight of a compound -/
def molecular_weight (c : Compound) : ℝ :=
  c.elements.map (λ e => e.atomic_weight) |>.sum

/-- Calcium element -/
def calcium : Element := ⟨"Calcium", 40⟩

/-- Oxygen element -/
def oxygen : Element := ⟨"Oxygen", 16⟩

/-- Calcium oxide compound -/
def calcium_oxide : Compound := ⟨"Calcium oxide", [calcium, oxygen]⟩

/-- Theorem: The molecular weight of Calcium oxide is 56 -/
theorem calcium_oxide_molecular_weight :
  molecular_weight calcium_oxide = 56 := by sorry

end NUMINAMATH_CALUDE_calcium_oxide_molecular_weight_l751_75148


namespace NUMINAMATH_CALUDE_quadratic_one_root_l751_75116

theorem quadratic_one_root (a b c d : ℝ) : 
  b = a - d →
  c = a - 3*d →
  a ≥ b →
  b ≥ c →
  c ≥ 0 →
  (∃! x : ℝ, a*x^2 + b*x + c = 0) →
  (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ x = -(1 + 3*Real.sqrt 22) / 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l751_75116


namespace NUMINAMATH_CALUDE_sequence_growth_l751_75184

theorem sequence_growth (a : ℕ → ℤ) 
  (h1 : a 1 > a 0) 
  (h2 : a 1 > 0) 
  (h3 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_growth_l751_75184


namespace NUMINAMATH_CALUDE_metal_sheet_dimensions_l751_75197

theorem metal_sheet_dimensions (a : ℝ) :
  (a > 0) →
  (2*a > 6) →
  (a > 6) →
  (3 * (2*a - 6) * (a - 6) = 168) →
  (a = 10) := by
sorry

end NUMINAMATH_CALUDE_metal_sheet_dimensions_l751_75197


namespace NUMINAMATH_CALUDE_quadratic_solution_l751_75174

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9) - 36 = 0) → b = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l751_75174


namespace NUMINAMATH_CALUDE_range_of_x_for_inequality_l751_75105

theorem range_of_x_for_inequality (x : ℝ) : 
  (∀ m : ℝ, m ∈ Set.Icc 0 1 → m * x^2 - 2*x - m ≥ 2) ↔ x ∈ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_for_inequality_l751_75105


namespace NUMINAMATH_CALUDE_parabola_vertex_and_a_range_l751_75155

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a^2 + 2*a

-- Define the line
def line (x : ℝ) : ℝ := 2*x - 2

-- Define the length of PQ
def PQ_length (a : ℝ) (m : ℝ) : ℝ := (m - (a + 1))^2 + 1

theorem parabola_vertex_and_a_range :
  (∀ x : ℝ, parabola 1 x ≥ 2) ∧
  (parabola 1 1 = 2) ∧
  (∀ a : ℝ, a > 0 → 
    (∀ m : ℝ, m < 3 → 
      (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < h → h < δ → 
        PQ_length a (m + h) < PQ_length a m
      ) → a ≥ 2
    )
  ) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_and_a_range_l751_75155


namespace NUMINAMATH_CALUDE_boris_candy_problem_l751_75163

/-- Given the initial number of candy pieces, the number eaten by the daughter,
    the number of bowls, and the number of pieces taken from each bowl,
    calculate the final number of pieces in one bowl. -/
def candyInBowl (initial : ℕ) (eaten : ℕ) (bowls : ℕ) (taken : ℕ) : ℕ :=
  (initial - eaten) / bowls - taken

theorem boris_candy_problem :
  candyInBowl 100 8 4 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_boris_candy_problem_l751_75163


namespace NUMINAMATH_CALUDE_abby_and_damon_weight_l751_75128

/-- The combined weight of Abby and Damon given the weights of other pairs -/
theorem abby_and_damon_weight
  (a b c d : ℝ)
  (h1 : a + b = 280)
  (h2 : b + c = 265)
  (h3 : c + d = 290)
  (h4 : b + d = 275) :
  a + d = 305 :=
by sorry

end NUMINAMATH_CALUDE_abby_and_damon_weight_l751_75128


namespace NUMINAMATH_CALUDE_rightmost_book_price_l751_75151

/-- Represents the price of a book at a given position. -/
def book_price (first_price : ℕ) (position : ℕ) : ℕ :=
  first_price + 3 * (position - 1)

/-- The theorem states that for a sequence of 41 books with the given conditions,
    the price of the rightmost book is $150. -/
theorem rightmost_book_price (first_price : ℕ) :
  (book_price first_price 41 = 
   book_price first_price 20 + 
   book_price first_price 21 + 
   book_price first_price 22) →
  book_price first_price 41 = 150 := by
sorry

end NUMINAMATH_CALUDE_rightmost_book_price_l751_75151


namespace NUMINAMATH_CALUDE_money_lasts_9_weeks_l751_75153

def lawn_mowing_earnings : ℕ := 9
def weed_eating_earnings : ℕ := 18
def weekly_spending : ℕ := 3

theorem money_lasts_9_weeks : 
  (lawn_mowing_earnings + weed_eating_earnings) / weekly_spending = 9 := by
  sorry

end NUMINAMATH_CALUDE_money_lasts_9_weeks_l751_75153


namespace NUMINAMATH_CALUDE_work_completion_l751_75199

/-- The number of days A and B worked together before A left the job -/
def days_worked_together : ℕ := 5

/-- A's work rate in terms of the total work per day -/
def rate_A : ℚ := 1 / 20

/-- B's work rate in terms of the total work per day -/
def rate_B : ℚ := 1 / 12

/-- The number of days B worked alone after A left -/
def days_B_alone : ℕ := 3

theorem work_completion :
  (days_worked_together : ℚ) * (rate_A + rate_B) + (days_B_alone : ℚ) * rate_B = 1 := by
  sorry

#check work_completion

end NUMINAMATH_CALUDE_work_completion_l751_75199


namespace NUMINAMATH_CALUDE_fans_with_all_items_count_l751_75146

/-- The capacity of the stadium --/
def stadium_capacity : ℕ := 5000

/-- The interval for hot dog coupons --/
def hot_dog_interval : ℕ := 60

/-- The interval for soda coupons --/
def soda_interval : ℕ := 40

/-- The interval for ice cream coupons --/
def ice_cream_interval : ℕ := 90

/-- The number of fans who received all three types of free items --/
def fans_with_all_items : ℕ := stadium_capacity / (Nat.lcm hot_dog_interval (Nat.lcm soda_interval ice_cream_interval))

theorem fans_with_all_items_count : fans_with_all_items = 13 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_count_l751_75146


namespace NUMINAMATH_CALUDE_martas_textbook_cost_l751_75120

/-- The total cost of Marta's textbooks --/
def total_cost (sale_price : ℕ) (sale_quantity : ℕ) (online_cost : ℕ) (bookstore_multiplier : ℕ) : ℕ :=
  sale_price * sale_quantity + online_cost + bookstore_multiplier * online_cost

/-- Theorem stating the total cost of Marta's textbooks --/
theorem martas_textbook_cost :
  total_cost 10 5 40 3 = 210 := by
  sorry

end NUMINAMATH_CALUDE_martas_textbook_cost_l751_75120


namespace NUMINAMATH_CALUDE_congruence_problem_l751_75162

theorem congruence_problem (x : ℤ) : 
  (4 * x + 9) % 19 = 3 → (3 * x + 8) % 19 = 13 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l751_75162


namespace NUMINAMATH_CALUDE_break_even_point_l751_75166

/-- The break-even point for a company producing exam preparation manuals -/
theorem break_even_point (Q : ℝ) :
  (Q > 0) →  -- Ensure Q is positive for division
  (300 : ℝ) = 100 + 100000 / Q →  -- Price equals average cost
  Q = 500 := by
sorry

end NUMINAMATH_CALUDE_break_even_point_l751_75166


namespace NUMINAMATH_CALUDE_initial_contribution_proof_l751_75186

/-- Proves that the initial total contribution was 300,000 rupees given the problem conditions. -/
theorem initial_contribution_proof (num_workers : ℕ) (extra_per_worker : ℕ) (total_with_extra : ℕ) :
  num_workers = 1200 →
  extra_per_worker = 50 →
  total_with_extra = 360000 →
  num_workers * ((total_with_extra - num_workers * extra_per_worker) / num_workers) = 300000 :=
by sorry

end NUMINAMATH_CALUDE_initial_contribution_proof_l751_75186


namespace NUMINAMATH_CALUDE_range_of_a_l751_75135

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  (prop_p a ∧ prop_q a) → (a ≤ -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l751_75135


namespace NUMINAMATH_CALUDE_orange_profit_problem_l751_75143

/-- The number of oranges needed to make a profit of 200 cents -/
def oranges_needed (buy_price : ℚ) (sell_price : ℚ) (profit_goal : ℚ) : ℕ :=
  (profit_goal / (sell_price - buy_price)).ceil.toNat

/-- The problem statement -/
theorem orange_profit_problem :
  let buy_price : ℚ := 14 / 4
  let sell_price : ℚ := 25 / 6
  let profit_goal : ℚ := 200
  oranges_needed buy_price sell_price profit_goal = 300 := by
sorry

end NUMINAMATH_CALUDE_orange_profit_problem_l751_75143


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l751_75124

/-- Given a circular arrangement of n people, this function calculates the clockwise distance between two positions -/
def clockwise_distance (n : ℕ) (a b : ℕ) : ℕ :=
  (b - a + n) % n

/-- Given a circular arrangement of n people, this function calculates the counterclockwise distance between two positions -/
def counterclockwise_distance (n : ℕ) (a b : ℕ) : ℕ :=
  (a - b + n) % n

theorem circular_seating_arrangement (n : ℕ) (h1 : n > 31) :
  clockwise_distance n 31 7 = counterclockwise_distance n 31 14 → n = 41 := by
  sorry

#eval clockwise_distance 41 31 7
#eval counterclockwise_distance 41 31 14

end NUMINAMATH_CALUDE_circular_seating_arrangement_l751_75124


namespace NUMINAMATH_CALUDE_sqrt_720_simplification_l751_75113

theorem sqrt_720_simplification : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_simplification_l751_75113


namespace NUMINAMATH_CALUDE_piano_lessons_cost_l751_75129

theorem piano_lessons_cost (piano_cost : ℝ) (num_lessons : ℕ) (lesson_cost : ℝ) (discount_rate : ℝ) :
  piano_cost = 500 →
  num_lessons = 20 →
  lesson_cost = 40 →
  discount_rate = 0.25 →
  piano_cost + (num_lessons : ℝ) * lesson_cost * (1 - discount_rate) = 1100 := by
  sorry

end NUMINAMATH_CALUDE_piano_lessons_cost_l751_75129


namespace NUMINAMATH_CALUDE_water_amount_theorem_l751_75144

/-- The amount of water in gallons that Timmy, Tommy, and Tina fill in a kiddie pool -/
def water_in_pool (tinas_pail : ℕ) (tommys_extra : ℕ) (timmys_multiplier : ℕ) (trips : ℕ) : ℕ :=
  let tommys_pail := tinas_pail + tommys_extra
  let timmys_pail := tommys_pail * timmys_multiplier
  (tinas_pail + tommys_pail + timmys_pail) * trips

/-- Theorem stating the amount of water in the pool after 3 trips each -/
theorem water_amount_theorem :
  water_in_pool 4 2 2 3 = 66 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_theorem_l751_75144


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l751_75195

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The condition that a_n^2 = a_(n-1) * a_(n+1) for n ≥ 2 -/
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n ^ 2 = a (n - 1) * a (n + 1)

/-- Definition of a geometric sequence -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem stating that the condition is necessary but not sufficient -/
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  ¬(∀ a : Sequence, Condition a → IsGeometric a) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l751_75195


namespace NUMINAMATH_CALUDE_restaurant_period_days_l751_75136

def pies_per_day : ℕ := 8
def total_pies : ℕ := 56

theorem restaurant_period_days : 
  total_pies / pies_per_day = 7 := by sorry

end NUMINAMATH_CALUDE_restaurant_period_days_l751_75136


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l751_75191

/-- Calculates the total cost of tickets for a high school musical performance. -/
def calculate_total_cost (adult_price : ℚ) (child_price : ℚ) (senior_price : ℚ) (student_price : ℚ)
  (num_adults : ℕ) (num_children : ℕ) (num_seniors : ℕ) (num_students : ℕ) : ℚ :=
  let adult_cost := num_adults * adult_price
  let child_cost := (num_children - 1) * child_price  -- Family package applied
  let senior_cost := num_seniors * senior_price * (1 - 1/10)  -- 10% senior discount
  let student_cost := 2 * student_price + (student_price / 2)  -- Student promotion
  adult_cost + child_cost + senior_cost + student_cost

/-- Theorem stating that the total cost for the given scenario is $103.30. -/
theorem total_cost_is_correct :
  calculate_total_cost 12 10 8 9 4 3 2 3 = 1033/10 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l751_75191


namespace NUMINAMATH_CALUDE_sequence_length_five_l751_75190

theorem sequence_length_five :
  ∃ (b₁ b₂ b₃ b₄ b₅ : ℕ), 
    b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < b₄ ∧ b₄ < b₅ ∧
    (2^433 + 1) / (2^49 + 1) = 2^b₁ + 2^b₂ + 2^b₃ + 2^b₄ + 2^b₅ := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_five_l751_75190


namespace NUMINAMATH_CALUDE_first_group_size_is_four_l751_75157

/-- The number of men in the first group -/
def first_group_size : ℕ := 4

/-- The length of cloth colored by the first group -/
def first_group_cloth_length : ℝ := 48

/-- The time taken by the first group to color their cloth -/
def first_group_time : ℝ := 2

/-- The number of men in the second group -/
def second_group_size : ℕ := 5

/-- The length of cloth colored by the second group -/
def second_group_cloth_length : ℝ := 36

/-- The time taken by the second group to color their cloth -/
def second_group_time : ℝ := 1.2

theorem first_group_size_is_four :
  first_group_size = 4 :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_is_four_l751_75157


namespace NUMINAMATH_CALUDE_max_perimeter_special_triangle_max_perimeter_achievable_l751_75111

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The maximum perimeter of a triangle ABC where a^2 = b^2 + c^2 - bc and a = 2 is 6 -/
theorem max_perimeter_special_triangle :
  ∀ t : Triangle,
    t.a^2 = t.b^2 + t.c^2 - t.b * t.c →
    t.a = 2 →
    perimeter t ≤ 6 :=
by
  sorry

/-- Corollary: There exists a triangle satisfying the conditions with perimeter equal to 6 -/
theorem max_perimeter_achievable :
  ∃ t : Triangle,
    t.a^2 = t.b^2 + t.c^2 - t.b * t.c ∧
    t.a = 2 ∧
    perimeter t = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_special_triangle_max_perimeter_achievable_l751_75111


namespace NUMINAMATH_CALUDE_total_students_in_halls_l751_75181

theorem total_students_in_halls (general : ℕ) (biology : ℕ) (math : ℕ) : 
  general = 30 ∧ 
  biology = 2 * general ∧ 
  math = (3 * (general + biology)) / 5 → 
  general + biology + math = 144 :=
by sorry

end NUMINAMATH_CALUDE_total_students_in_halls_l751_75181


namespace NUMINAMATH_CALUDE_sin_cos_difference_l751_75198

theorem sin_cos_difference (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = 1/5) → 
  (Real.sin α - Real.cos α = 7/5) := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_l751_75198


namespace NUMINAMATH_CALUDE_tangent_through_origin_l751_75132

/-- The curve y = x^α + 1 has a tangent line at (1, 2) that passes through the origin if and only if α = 2 -/
theorem tangent_through_origin (α : ℝ) : 
  (∃ (m : ℝ), (∀ x : ℝ, x^α + 1 = m * (x - 1) + 2) ∧ m * (-1) + 2 = 0) ↔ α = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_through_origin_l751_75132


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l751_75183

theorem square_sum_equals_one (a b : ℝ) (h : a = 1 - b) : a^2 + 2*a*b + b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l751_75183


namespace NUMINAMATH_CALUDE_nested_expression_value_l751_75139

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l751_75139


namespace NUMINAMATH_CALUDE_isosceles_triangle_relationship_l751_75142

/-- An isosceles triangle with perimeter 10 cm -/
structure IsoscelesTriangle where
  /-- Length of each equal side in cm -/
  x : ℝ
  /-- Length of the base in cm -/
  y : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : true
  /-- The perimeter is 10 cm -/
  perimeterIs10 : x + x + y = 10

/-- The relationship between y and x, and the range of x for the isosceles triangle -/
theorem isosceles_triangle_relationship (t : IsoscelesTriangle) :
  t.y = 10 - 2 * t.x ∧ 5/2 < t.x ∧ t.x < 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_relationship_l751_75142


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l751_75185

/-- The constant term in the expansion of (1/√x - x^2)^10 is 45 -/
theorem constant_term_binomial_expansion :
  let f (x : ℝ) := (1 / Real.sqrt x - x^2)^10
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 → f x = c + x * (f x - c) ∧ c = 45 :=
sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l751_75185


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l751_75169

-- Define the sequence b_n
def b (n : ℕ) : ℕ := n.factorial + 2 * n

-- Theorem statement
theorem max_gcd_consecutive_terms :
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (b n) (b (n + 1)) ≤ k ∧
  ∃ (m : ℕ), Nat.gcd (b m) (b (m + 1)) = k :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l751_75169


namespace NUMINAMATH_CALUDE_participation_schemes_count_l751_75107

/-- Represents the number of students -/
def num_students : ℕ := 5

/-- Represents the number of competitions -/
def num_competitions : ℕ := 4

/-- Represents the number of competitions student A cannot participate in -/
def restricted_competitions : ℕ := 2

/-- Calculates the number of different competition participation schemes -/
def participation_schemes : ℕ := 
  (num_competitions - restricted_competitions) * 
  (Nat.factorial num_students / Nat.factorial (num_students - (num_competitions - 1)))

/-- Theorem stating the number of different competition participation schemes -/
theorem participation_schemes_count : participation_schemes = 72 := by
  sorry

end NUMINAMATH_CALUDE_participation_schemes_count_l751_75107


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l751_75138

/-- Given a line L1 with equation 3x - y = 6 and a point P (-2, 3),
    prove that the line L2 with equation y = 3x + 9 is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - y = 6
  let P : ℝ × ℝ := (-2, 3)
  let L2 : ℝ → ℝ → Prop := λ x y => y = 3 * x + 9
  (∀ x y, L1 x y ↔ y = 3 * x - 6) →  -- L1 in slope-intercept form
  L2 P.1 P.2 ∧                      -- L2 passes through P
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → y₂ - y₁ = 3 * (x₂ - x₁)) →  -- Slope of L1 is 3
  (∀ x₁ y₁ x₂ y₂, L2 x₁ y₁ → L2 x₂ y₂ → y₂ - y₁ = 3 * (x₂ - x₁))    -- Slope of L2 is 3
  :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l751_75138


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l751_75125

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ,
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p > 30 ∧ 
    q > 30 ∧ 
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ a b : ℕ, 
      Nat.Prime a → 
      Nat.Prime b → 
      a > 30 → 
      b > 30 → 
      a ≠ b → 
      a * b ≥ 1147 :=
by
  sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l751_75125


namespace NUMINAMATH_CALUDE_always_returns_to_present_max_stations_visited_l751_75133

/-- Represents the time machine's movement on a circular track of 2009 stations. -/
def TimeMachine :=
  { s : ℕ // s ≤ 2009 }

/-- Moves the time machine to the next station according to the rules. -/
def nextStation (s : TimeMachine) : TimeMachine :=
  sorry

/-- Checks if a number is a power of 2. -/
def isPowerOfTwo (n : ℕ) : Bool :=
  sorry

/-- Returns the sequence of stations visited by the time machine starting from a given station. -/
def stationSequence (start : TimeMachine) : List TimeMachine :=
  sorry

/-- Theorem stating that the time machine always returns to station 1. -/
theorem always_returns_to_present (start : TimeMachine) :
  1 ∈ (stationSequence start).map (fun s => s.val) := by
  sorry

/-- Theorem stating the maximum number of stations the time machine can stop at. -/
theorem max_stations_visited :
  ∃ (start : TimeMachine), (stationSequence start).length = 812 ∧
  ∀ (s : TimeMachine), (stationSequence s).length ≤ 812 := by
  sorry

end NUMINAMATH_CALUDE_always_returns_to_present_max_stations_visited_l751_75133


namespace NUMINAMATH_CALUDE_equation_positive_root_m_value_l751_75158

theorem equation_positive_root_m_value (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ m / (x - 3) - 1 / (3 - x) = 2) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_equation_positive_root_m_value_l751_75158


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l751_75192

theorem students_liking_both_desserts
  (total : ℕ)
  (like_brownies : ℕ)
  (like_ice_cream : ℕ)
  (like_neither : ℕ)
  (h1 : total = 45)
  (h2 : like_brownies = 22)
  (h3 : like_ice_cream = 17)
  (h4 : like_neither = 13) :
  (like_brownies + like_ice_cream) - (total - like_neither) = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l751_75192


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l751_75131

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 400) :
  a 2 + a 8 = 160 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l751_75131


namespace NUMINAMATH_CALUDE_cube_shape_ratio_l751_75187

/-- A shape formed by joining nine unit cubes with specific arrangement -/
structure CubeShape where
  /-- Total number of cubes -/
  num_cubes : Nat
  /-- Volume of each unit cube -/
  unit_volume : ℝ
  /-- Surface area of each exposed face of a unit cube -/
  unit_surface_area : ℝ
  /-- Number of exposed faces for each corner cube -/
  exposed_faces_per_corner : Nat
  /-- Number of corner cubes -/
  num_corner_cubes : Nat
  /-- Condition: Total number of cubes is 9 -/
  h_num_cubes : num_cubes = 9
  /-- Condition: Volume of each unit cube is 1 -/
  h_unit_volume : unit_volume = 1
  /-- Condition: Surface area of each exposed face is 1 -/
  h_unit_surface_area : unit_surface_area = 1
  /-- Condition: Each corner cube has 3 exposed faces -/
  h_exposed_faces : exposed_faces_per_corner = 3
  /-- Condition: There are 8 corner cubes -/
  h_corner_cubes : num_corner_cubes = 8

/-- The ratio of volume to surface area for the CubeShape is 3:8 -/
theorem cube_shape_ratio (shape : CubeShape) :
  (shape.num_cubes * shape.unit_volume) / 
  (shape.num_corner_cubes * shape.exposed_faces_per_corner * shape.unit_surface_area) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_shape_ratio_l751_75187


namespace NUMINAMATH_CALUDE_median_in_70_74_interval_l751_75182

/-- Represents the frequency of scores in each interval -/
structure ScoreFrequency where
  interval : ℕ × ℕ
  count : ℕ

/-- The list of score frequencies for the test -/
def scoreDistribution : List ScoreFrequency := [
  ⟨(80, 84), 16⟩,
  ⟨(75, 79), 12⟩,
  ⟨(70, 74), 6⟩,
  ⟨(65, 69), 3⟩,
  ⟨(60, 64), 2⟩,
  ⟨(55, 59), 20⟩,
  ⟨(50, 54), 22⟩
]

/-- The total number of students -/
def totalStudents : ℕ := 81

/-- The position of the median in the ordered list of scores -/
def medianPosition : ℕ := (totalStudents + 1) / 2

/-- Function to find the interval containing the median score -/
def findMedianInterval (distribution : List ScoreFrequency) (medianPos : ℕ) : ℕ × ℕ :=
  sorry

/-- Theorem stating that the median score is in the interval 70-74 -/
theorem median_in_70_74_interval :
  findMedianInterval scoreDistribution medianPosition = (70, 74) := by
  sorry

end NUMINAMATH_CALUDE_median_in_70_74_interval_l751_75182


namespace NUMINAMATH_CALUDE_worker_earnings_l751_75180

/-- Calculates the total earnings for a worker based on survey completions and bonus structure -/
def calculate_earnings (regular_rate : ℚ) 
                       (simple_surveys : ℕ) 
                       (moderate_surveys : ℕ) 
                       (complex_surveys : ℕ) 
                       (non_cellphone_surveys : ℕ) : ℚ :=
  let simple_rate := regular_rate * (1 + 30 / 100)
  let moderate_rate := regular_rate * (1 + 50 / 100)
  let complex_rate := regular_rate * (1 + 75 / 100)
  let total_surveys := simple_surveys + moderate_surveys + complex_surveys + non_cellphone_surveys
  let survey_earnings := 
    regular_rate * non_cellphone_surveys +
    simple_rate * simple_surveys +
    moderate_rate * moderate_surveys +
    complex_rate * complex_surveys
  let tiered_bonus := 
    if total_surveys ≥ 100 then 250
    else if total_surveys ≥ 75 then 150
    else if total_surveys ≥ 50 then 100
    else 0
  let milestone_bonus := 
    (if simple_surveys ≥ 25 then 50 else 0) +
    (if moderate_surveys ≥ 15 then 75 else 0) +
    (if complex_surveys ≥ 5 then 125 else 0)
  survey_earnings + tiered_bonus + milestone_bonus

/-- The total earnings for the worker is 1765 -/
theorem worker_earnings : 
  calculate_earnings 10 30 20 10 40 = 1765 := by sorry

end NUMINAMATH_CALUDE_worker_earnings_l751_75180


namespace NUMINAMATH_CALUDE_triangle_data_uniqueness_l751_75100

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- sides
  (α β γ : ℝ)  -- angles

-- Define the different sets of data
def ratio_two_sides_included_angle (t : Triangle) : ℝ × ℝ × ℝ := sorry
def ratios_angle_bisectors (t : Triangle) : ℝ × ℝ × ℝ := sorry
def ratios_medians (t : Triangle) : ℝ × ℝ × ℝ := sorry
def ratios_two_altitudes_bases (t : Triangle) : ℝ × ℝ := sorry
def two_angles_ratio_side_sum (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define a predicate for unique determination of triangle shape
def uniquely_determines_shape (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → t1 = t2

-- State the theorem
theorem triangle_data_uniqueness :
  uniquely_determines_shape ratio_two_sides_included_angle ∧
  uniquely_determines_shape ratios_medians ∧
  uniquely_determines_shape ratios_two_altitudes_bases ∧
  uniquely_determines_shape two_angles_ratio_side_sum ∧
  ¬ uniquely_determines_shape ratios_angle_bisectors :=
sorry

end NUMINAMATH_CALUDE_triangle_data_uniqueness_l751_75100


namespace NUMINAMATH_CALUDE_drug_effectiveness_max_effective_hours_l751_75178

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x ≤ 4 then -1/2 * x^2 + 2*x + 8
  else if 4 < x ∧ x ≤ 16 then -x/2 - Real.log x / Real.log 2 + 12
  else 0

def is_effective (m : ℝ) (x : ℝ) : Prop := m * f x ≥ 12

theorem drug_effectiveness (m : ℝ) (h : m > 0) :
  (∀ x, 0 < x ∧ x ≤ 8 → is_effective m x) ↔ m ≥ 12/5 :=
sorry

theorem max_effective_hours :
  ∃ k : ℕ, k = 6 ∧ 
  (∀ x, 0 < x ∧ x ≤ ↑k → is_effective 2 x) ∧
  (∀ k' : ℕ, k' > k → ∃ x, 0 < x ∧ x ≤ ↑k' ∧ ¬is_effective 2 x) :=
sorry

end NUMINAMATH_CALUDE_drug_effectiveness_max_effective_hours_l751_75178


namespace NUMINAMATH_CALUDE_sphere_expansion_l751_75112

theorem sphere_expansion (r₁ r₂ : ℝ) (h : r₁ > 0) (h' : r₂ > 0) :
  (4 * π * r₂^2) = 4 * (4 * π * r₁^2) →
  ((4 / 3) * π * r₂^3) = 8 * ((4 / 3) * π * r₁^3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_expansion_l751_75112


namespace NUMINAMATH_CALUDE_inequality_proof_l751_75170

theorem inequality_proof (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  1 / Real.sqrt (1 + x^2) + 1 / Real.sqrt (1 + y^2) ≤ 2 / Real.sqrt (1 + x*y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l751_75170


namespace NUMINAMATH_CALUDE_max_goats_l751_75173

/-- Represents the number of coconuts Max can trade for one crab -/
def coconuts_per_crab : ℕ := 3

/-- Represents the number of crabs Max can trade for one goat -/
def crabs_per_goat : ℕ := 6

/-- Represents the initial number of coconuts Max has -/
def initial_coconuts : ℕ := 342

/-- Calculates the number of goats Max will have after trading all his coconuts -/
def goats_from_coconuts (coconuts : ℕ) (coconuts_per_crab : ℕ) (crabs_per_goat : ℕ) : ℕ :=
  (coconuts / coconuts_per_crab) / crabs_per_goat

/-- Theorem stating that Max will end up with 19 goats -/
theorem max_goats : 
  goats_from_coconuts initial_coconuts coconuts_per_crab crabs_per_goat = 19 := by
  sorry

end NUMINAMATH_CALUDE_max_goats_l751_75173


namespace NUMINAMATH_CALUDE_pendant_sales_theorem_l751_75165

/-- Parameters for the Asian Games mascot pendant sales problem -/
structure PendantSales where
  cost : ℝ             -- Cost price of each pendant
  initial_price : ℝ    -- Initial selling price
  initial_sales : ℝ    -- Initial monthly sales
  price_sensitivity : ℝ -- Daily sales decrease per 1 yuan price increase

/-- Calculate profit based on price increase -/
def profit (p : PendantSales) (x : ℝ) : ℝ :=
  (p.initial_price + x - p.cost) * (p.initial_sales - 30 * p.price_sensitivity * x)

/-- Theorem for the Asian Games mascot pendant sales problem -/
theorem pendant_sales_theorem (p : PendantSales) 
  (h1 : p.cost = 13)
  (h2 : p.initial_price = 20)
  (h3 : p.initial_sales = 200)
  (h4 : p.price_sensitivity = 10) :
  (∃ x : ℝ, x^2 - 13*x + 22 = 0 ∧ profit p x = 1620) ∧
  (∃ x : ℝ, x = 53/2 ∧ ∀ y : ℝ, profit p y ≤ profit p x) ∧
  profit p (13/2) = 3645/2 := by
  sorry


end NUMINAMATH_CALUDE_pendant_sales_theorem_l751_75165


namespace NUMINAMATH_CALUDE_kenneth_theorem_l751_75114

def kenneth_problem (earnings : ℝ) (joystick_percentage : ℝ) : Prop :=
  let joystick_cost := earnings * (joystick_percentage / 100)
  let remaining := earnings - joystick_cost
  earnings = 450 ∧ joystick_percentage = 10 → remaining = 405

theorem kenneth_theorem : kenneth_problem 450 10 := by
  sorry

end NUMINAMATH_CALUDE_kenneth_theorem_l751_75114


namespace NUMINAMATH_CALUDE_turtleneck_sweater_profit_l751_75167

/-- Represents the pricing strategy and profit calculation for a store selling turtleneck sweaters -/
theorem turtleneck_sweater_profit (C : ℝ) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_discount := 0.08
  let SP1 := C * (1 + initial_markup)
  let SP2 := SP1 * (1 + new_year_markup)
  let SPF := SP2 * (1 - february_discount)
  let profit := SPF - C
  profit = 0.38 * C :=
by sorry

end NUMINAMATH_CALUDE_turtleneck_sweater_profit_l751_75167


namespace NUMINAMATH_CALUDE_max_value_at_point_one_two_l751_75121

/-- The feasible region defined by the given constraints -/
def FeasibleRegion (x y : ℝ) : Prop :=
  x + 2*y ≤ 5 ∧ 2*x + y ≤ 4 ∧ x ≥ 0 ∧ y ≥ 0

/-- The objective function to be maximized -/
def ObjectiveFunction (x y : ℝ) : ℝ := 3*x + 4*y

/-- Theorem stating that the maximum value of the objective function
    in the feasible region is 11, achieved at (1, 2) -/
theorem max_value_at_point_one_two :
  ∃ (max : ℝ), max = 11 ∧
  ∃ (x₀ y₀ : ℝ), x₀ = 1 ∧ y₀ = 2 ∧
  FeasibleRegion x₀ y₀ ∧
  ObjectiveFunction x₀ y₀ = max ∧
  ∀ (x y : ℝ), FeasibleRegion x y → ObjectiveFunction x y ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_at_point_one_two_l751_75121


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_l751_75122

theorem smallest_four_digit_multiple : ∃ (n : ℕ), 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  (2 ∣ n) ∧ (3 ∣ n) ∧ (8 ∣ n) ∧ (9 ∣ n) ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ (2 ∣ m) ∧ (3 ∣ m) ∧ (8 ∣ m) ∧ (9 ∣ m) → m ≥ n) ∧
  n = 1008 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_l751_75122


namespace NUMINAMATH_CALUDE_blanch_dinner_slices_l751_75140

/-- Calculates the number of pizza slices eaten for dinner given the initial number of slices and consumption throughout the day. -/
def pizza_slices_for_dinner (initial_slices breakfast_slices lunch_slices snack_slices remaining_slices : ℕ) : ℕ :=
  initial_slices - (breakfast_slices + lunch_slices + snack_slices + remaining_slices)

/-- Proves that Blanch ate 5 slices of pizza for dinner given the conditions of the problem. -/
theorem blanch_dinner_slices :
  pizza_slices_for_dinner 15 4 2 2 2 = 5 := by
  sorry

#eval pizza_slices_for_dinner 15 4 2 2 2

end NUMINAMATH_CALUDE_blanch_dinner_slices_l751_75140


namespace NUMINAMATH_CALUDE_largest_r_same_range_l751_75154

/-- A quadratic polynomial function -/
def f (r : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 3 * x + r

/-- The theorem stating the largest value of r for which f and f ∘ f have the same range -/
theorem largest_r_same_range :
  ∃ (r_max : ℝ), r_max = 15/8 ∧
  ∀ (r : ℝ), Set.range (f r) = Set.range (f r ∘ f r) ↔ r ≤ r_max :=
sorry

end NUMINAMATH_CALUDE_largest_r_same_range_l751_75154


namespace NUMINAMATH_CALUDE_purple_sequins_count_l751_75188

/-- The number of purple sequins in each row on Jane's costume. -/
def purple_sequins_per_row : ℕ :=
  let total_sequins : ℕ := 162
  let blue_rows : ℕ := 6
  let blue_per_row : ℕ := 8
  let purple_rows : ℕ := 5
  let green_rows : ℕ := 9
  let green_per_row : ℕ := 6
  let blue_sequins : ℕ := blue_rows * blue_per_row
  let green_sequins : ℕ := green_rows * green_per_row
  let purple_sequins : ℕ := total_sequins - (blue_sequins + green_sequins)
  purple_sequins / purple_rows

theorem purple_sequins_count : purple_sequins_per_row = 12 := by
  sorry

end NUMINAMATH_CALUDE_purple_sequins_count_l751_75188


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l751_75134

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_negative_two :
  reciprocal (-2) = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l751_75134


namespace NUMINAMATH_CALUDE_peter_walk_time_l751_75194

/-- Calculates the remaining time to walk given total distance, walking speed, and distance already walked -/
def remaining_walk_time (total_distance : ℝ) (walking_speed : ℝ) (distance_walked : ℝ) : ℝ :=
  (total_distance - distance_walked) * walking_speed

theorem peter_walk_time :
  let total_distance : ℝ := 2.5
  let walking_speed : ℝ := 20
  let distance_walked : ℝ := 1
  remaining_walk_time total_distance walking_speed distance_walked = 30 := by
sorry

end NUMINAMATH_CALUDE_peter_walk_time_l751_75194


namespace NUMINAMATH_CALUDE_black_car_overtake_time_l751_75156

/-- Proves that the time for the black car to overtake the red car is 3 hours. -/
theorem black_car_overtake_time (red_speed black_speed initial_distance : ℝ) 
  (h1 : red_speed = 40)
  (h2 : black_speed = 50)
  (h3 : initial_distance = 30)
  (h4 : red_speed > 0)
  (h5 : black_speed > red_speed) :
  (initial_distance / (black_speed - red_speed)) = 3 := by
  sorry

#check black_car_overtake_time

end NUMINAMATH_CALUDE_black_car_overtake_time_l751_75156


namespace NUMINAMATH_CALUDE_tenisha_dogs_l751_75171

/-- The initial number of dogs Tenisha had -/
def initial_dogs : ℕ := 40

/-- The proportion of female dogs -/
def female_ratio : ℚ := 3/5

/-- The proportion of female dogs that give birth -/
def birth_ratio : ℚ := 3/4

/-- The number of puppies each female dog gives birth to -/
def puppies_per_dog : ℕ := 10

/-- The number of puppies donated -/
def donated_puppies : ℕ := 130

/-- The number of puppies remaining after donation -/
def remaining_puppies : ℕ := 50

theorem tenisha_dogs :
  (initial_dogs : ℚ) * female_ratio * birth_ratio * puppies_per_dog =
  (donated_puppies + remaining_puppies : ℚ) :=
sorry

end NUMINAMATH_CALUDE_tenisha_dogs_l751_75171
