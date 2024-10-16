import Mathlib

namespace NUMINAMATH_CALUDE_expected_pollen_allergy_l1825_182533

theorem expected_pollen_allergy (total_sample : ℕ) (allergy_ratio : ℚ) 
  (h1 : total_sample = 400) 
  (h2 : allergy_ratio = 1 / 4) : 
  ↑total_sample * allergy_ratio = 100 := by
  sorry

end NUMINAMATH_CALUDE_expected_pollen_allergy_l1825_182533


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1825_182586

theorem no_positive_integer_solutions (A : ℕ) : 
  A > 0 → A < 10 → ¬∃ x : ℕ, x > 0 ∧ x^2 - (2*A + 1)*x + (A + 1)*(10 + A) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1825_182586


namespace NUMINAMATH_CALUDE_exists_irrational_less_than_neg_two_l1825_182517

theorem exists_irrational_less_than_neg_two : ∃ x : ℝ, Irrational x ∧ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_exists_irrational_less_than_neg_two_l1825_182517


namespace NUMINAMATH_CALUDE_empty_jar_weight_l1825_182563

/-- Represents the weight of a jar with water -/
structure JarWeight where
  empty : ℝ  -- Weight of the empty jar
  water : ℝ  -- Weight of water when fully filled

/-- The weight of the jar when partially filled -/
def partialWeight (j : JarWeight) (fraction : ℝ) : ℝ :=
  j.empty + fraction * j.water

theorem empty_jar_weight (j : JarWeight) :
  (partialWeight j (1/5) = 560) →
  (partialWeight j (4/5) = 740) →
  j.empty = 500 := by
  sorry

end NUMINAMATH_CALUDE_empty_jar_weight_l1825_182563


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l1825_182591

theorem cos_alpha_plus_pi_sixth (α : ℝ) (h : Real.sin (α - π/3) = 1/3) :
  Real.cos (α + π/6) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l1825_182591


namespace NUMINAMATH_CALUDE_great_dane_weight_l1825_182527

theorem great_dane_weight (chihuahua pitbull great_dane : ℕ) : 
  chihuahua + pitbull + great_dane = 439 →
  pitbull = 3 * chihuahua →
  great_dane = 3 * pitbull + 10 →
  great_dane = 307 := by
sorry

end NUMINAMATH_CALUDE_great_dane_weight_l1825_182527


namespace NUMINAMATH_CALUDE_max_y_value_l1825_182578

theorem max_y_value (x y : ℝ) (h : x^2 + y^2 = 10*x + 60*y) :
  y ≤ 30 + 5 * Real.sqrt 37 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 10*x₀ + 60*y₀ ∧ y₀ = 30 + 5 * Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l1825_182578


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1825_182525

/-- A triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : b = a + 2
  h2 : c = b + 2
  h3 : Even a
  h4 : a + b > c
  h5 : a + c > b
  h6 : b + c > a

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ := t.a + t.b + t.c

/-- The statement that 18 is the smallest possible perimeter of an EvenTriangle -/
theorem smallest_even_triangle_perimeter :
  ∀ t : EvenTriangle, perimeter t ≥ 18 ∧ ∃ t₀ : EvenTriangle, perimeter t₀ = 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1825_182525


namespace NUMINAMATH_CALUDE_gcd_192_144_320_l1825_182513

theorem gcd_192_144_320 : Nat.gcd 192 (Nat.gcd 144 320) = 16 := by sorry

end NUMINAMATH_CALUDE_gcd_192_144_320_l1825_182513


namespace NUMINAMATH_CALUDE_macy_running_goal_l1825_182564

/-- Calculates the remaining miles to run given a weekly goal, daily run distance, and number of days run. -/
def remaining_miles (weekly_goal : ℕ) (daily_run : ℕ) (days_run : ℕ) : ℕ :=
  weekly_goal - daily_run * days_run

/-- Proves that given a weekly goal of 24 miles and a daily run of 3 miles, the remaining distance to run after 6 days is 6 miles. -/
theorem macy_running_goal :
  remaining_miles 24 3 6 = 6 := by
  sorry

#eval remaining_miles 24 3 6

end NUMINAMATH_CALUDE_macy_running_goal_l1825_182564


namespace NUMINAMATH_CALUDE_smallest_seven_digit_binary_l1825_182516

theorem smallest_seven_digit_binary : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m.digits 2 = [1, 0, 0, 0, 0, 0, 0] → m ≥ n) ∧
  n.digits 2 = [1, 0, 0, 0, 0, 0, 0] ∧
  n = 64 := by
  sorry

end NUMINAMATH_CALUDE_smallest_seven_digit_binary_l1825_182516


namespace NUMINAMATH_CALUDE_even_function_alpha_beta_values_l1825_182555

theorem even_function_alpha_beta_values (α β : Real) :
  let f : Real → Real := λ x => 
    if x < 0 then Real.sin (x + α) else Real.cos (x + β)
  (∀ x, f (-x) = f x) →
  α = π / 3 ∧ β = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_even_function_alpha_beta_values_l1825_182555


namespace NUMINAMATH_CALUDE_unique_root_condition_l1825_182566

theorem unique_root_condition (k : ℝ) : 
  (∃! x : ℝ, (x / (x + 3) + x / (x + 4) = k * x)) ↔ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_root_condition_l1825_182566


namespace NUMINAMATH_CALUDE_sequence_a_formula_l1825_182522

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) - sequence_a n

theorem sequence_a_formula (n : ℕ) : sequence_a n = (2^n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_formula_l1825_182522


namespace NUMINAMATH_CALUDE_range_of_m_l1825_182587

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : ∀ x y, x > 0 → y > 0 → (2 * y / x + 8 * x / y ≥ m^2 + 2*m)) : 
  m ∈ Set.Icc (-4 : ℝ) 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1825_182587


namespace NUMINAMATH_CALUDE_candidates_scientific_notation_l1825_182595

/-- The number of candidates for the high school entrance examination in Guangdong Province in 2023 -/
def candidates : ℝ := 1108200

/-- The scientific notation representation of the number of candidates -/
def scientific_notation : ℝ := 1.1082 * (10 ^ 6)

/-- Theorem stating that the number of candidates is equal to its scientific notation representation -/
theorem candidates_scientific_notation : candidates = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_candidates_scientific_notation_l1825_182595


namespace NUMINAMATH_CALUDE_ellipse_right_triangle_l1825_182543

-- Define the ellipse
def Γ (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the vertices
def A (a b : ℝ) : ℝ × ℝ := (-a, 0)
def B (a b : ℝ) : ℝ × ℝ := (a, 0)
def C (a b : ℝ) : ℝ × ℝ := (0, b)
def D (a b : ℝ) : ℝ × ℝ := (0, -b)

-- Define the theorem
theorem ellipse_right_triangle (a b : ℝ) (P Q R : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  Γ a b P.1 P.2 ∧
  Γ a b Q.1 Q.2 ∧
  P.1 ≥ 0 ∧ P.2 ≥ 0 ∧
  Q.1 ≥ 0 ∧ Q.2 ≥ 0 ∧
  (∃ k : ℝ, Q = k • (A a b - P)) ∧
  (∃ t : ℝ, R = t • ((P.1 / 2, P.2 / 2) : ℝ × ℝ)) ∧
  Γ a b R.1 R.2 →
  ‖Q‖^2 + ‖R‖^2 = ‖B a b - C a b‖^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_right_triangle_l1825_182543


namespace NUMINAMATH_CALUDE_actual_tissue_diameter_l1825_182510

/-- Given a circular piece of tissue magnified by an electron microscope, 
    this theorem proves that the actual diameter of the tissue is 0.001 centimeters. -/
theorem actual_tissue_diameter 
  (magnification : ℝ) 
  (magnified_diameter : ℝ) 
  (h1 : magnification = 1000)
  (h2 : magnified_diameter = 1) : 
  magnified_diameter / magnification = 0.001 := by
  sorry

end NUMINAMATH_CALUDE_actual_tissue_diameter_l1825_182510


namespace NUMINAMATH_CALUDE_pencil_order_cost_l1825_182507

/-- Calculates the cost of pencils with a potential discount -/
def pencilCost (boxSize : ℕ) (boxPrice : ℚ) (discountThreshold : ℕ) (discountRate : ℚ) (quantity : ℕ) : ℚ :=
  let basePrice := (quantity : ℚ) * boxPrice / (boxSize : ℚ)
  if quantity > discountThreshold then
    basePrice * (1 - discountRate)
  else
    basePrice

theorem pencil_order_cost :
  pencilCost 200 40 1000 (1/10) 2400 = 432 :=
by sorry

end NUMINAMATH_CALUDE_pencil_order_cost_l1825_182507


namespace NUMINAMATH_CALUDE_negation_inverse_implies_contrapositive_l1825_182579

-- Define propositions as functions from some universe U to Prop
variable {U : Type}
variable (p q r : U → Prop)

-- Define the negation relation
def is_negation (p q : U → Prop) : Prop :=
  ∀ x, q x ↔ ¬(p x)

-- Define the inverse relation
def is_inverse (q r : U → Prop) : Prop :=
  ∀ x, r x ↔ (¬q x)

-- Define the contrapositive relation
def is_contrapositive (p r : U → Prop) : Prop :=
  ∀ x y, (p x → p y) ↔ (¬p y → ¬p x)

-- The main theorem
theorem negation_inverse_implies_contrapositive (p q r : U → Prop) :
  is_negation p q → is_inverse q r → is_contrapositive p r :=
sorry

end NUMINAMATH_CALUDE_negation_inverse_implies_contrapositive_l1825_182579


namespace NUMINAMATH_CALUDE_folded_line_length_squared_l1825_182521

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  is_positive : side_length > 0

-- Define the folding operation
def fold (t : EquilateralTriangle) (fold_point : ℝ) :=
  0 < fold_point ∧ fold_point < t.side_length

-- Theorem statement
theorem folded_line_length_squared 
  (t : EquilateralTriangle) 
  (h_side : t.side_length = 10) 
  (h_fold : fold t 3) : 
  ∃ (l : ℝ), l^2 = 37/4 ∧ l > 0 := by
  sorry

end NUMINAMATH_CALUDE_folded_line_length_squared_l1825_182521


namespace NUMINAMATH_CALUDE_set_operation_result_l1825_182577

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem set_operation_result :
  (M ∩ N) ∪ (U \ N) = {0, 1, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l1825_182577


namespace NUMINAMATH_CALUDE_min_value_theorem_l1825_182583

theorem min_value_theorem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 8) 
  (h2 : e * f * g * h = 16) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 + 
  (a * b)^2 + (c * d)^2 + (e * f)^2 + (g * h)^2 ≥ 64 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1825_182583


namespace NUMINAMATH_CALUDE_second_part_interest_rate_l1825_182573

theorem second_part_interest_rate 
  (total_investment : ℝ) 
  (first_part : ℝ) 
  (first_rate : ℝ) 
  (total_interest : ℝ) :
  total_investment = 4000 →
  first_part = 2800 →
  first_rate = 0.03 →
  total_interest = 144 →
  (first_part * first_rate + (total_investment - first_part) * 0.05 = total_interest) := by
sorry

end NUMINAMATH_CALUDE_second_part_interest_rate_l1825_182573


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1825_182542

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 3 ↔ x - a < 1 ∧ x - 2*b > 3) → 
  a = 2 ∧ b = -2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1825_182542


namespace NUMINAMATH_CALUDE_exactly_one_true_l1825_182599

def p : Prop := ∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0

def q : Prop := ∀ (f : ℝ → ℝ) (a b : ℝ), a < b →
  (∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c) →
  (∃ c ∈ Set.Ioo a b, ∃ ε > 0, ∀ x ∈ Set.Icc a b ∩ Set.Ioo (c - ε) (c + ε), f x ≤ f c)

theorem exactly_one_true : (p ∧ q) ∨ (p ∨ q) ∨ (¬p) ∧ ¬((p ∧ q) ∧ (p ∨ q)) ∧ ¬((p ∧ q) ∧ (¬p)) ∧ ¬((p ∨ q) ∧ (¬p)) := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_true_l1825_182599


namespace NUMINAMATH_CALUDE_loss_fraction_for_apple_l1825_182575

/-- Calculates the fraction of loss given the cost price and selling price -/
def fractionOfLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

/-- Theorem stating that for given cost price 17 and selling price 16, 
    the fraction of loss is 1/17 -/
theorem loss_fraction_for_apple : 
  fractionOfLoss 17 16 = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_loss_fraction_for_apple_l1825_182575


namespace NUMINAMATH_CALUDE_roberts_soccer_kicks_l1825_182515

theorem roberts_soccer_kicks (kicks_before_break kicks_after_break kicks_remaining : ℕ) :
  kicks_before_break = 43 →
  kicks_after_break = 36 →
  kicks_remaining = 19 →
  kicks_before_break + kicks_after_break + kicks_remaining = 98 := by
  sorry

end NUMINAMATH_CALUDE_roberts_soccer_kicks_l1825_182515


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l1825_182539

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Represents the height of an object in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Converts a Height to total inches -/
def heightToInches (h : Height) : ℕ := feetToInches h.feet + h.inches

/-- Calculates the total height when placing an object on a base -/
def totalHeight (objectHeight : Height) (baseHeight : ℕ) : ℕ :=
  heightToInches objectHeight + baseHeight

theorem sculpture_and_base_height :
  let sculptureHeight : Height := { feet := 2, inches := 10 }
  let baseHeight : ℕ := 4
  totalHeight sculptureHeight baseHeight = 38 := by sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l1825_182539


namespace NUMINAMATH_CALUDE_solution_set_l1825_182550

theorem solution_set (x : ℝ) : 
  x > 4 → x^3 - 8*x^2 + 16*x > 64 ∧ x^2 - 4*x + 5 > 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l1825_182550


namespace NUMINAMATH_CALUDE_absolute_value_sum_simplification_l1825_182561

theorem absolute_value_sum_simplification (x : ℝ) : 
  |x - 1| + |x - 2| + |x + 3| = 
    if x < -3 then -3*x
    else if x < 1 then 6 - x
    else if x < 2 then 4 + x
    else 3*x := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_simplification_l1825_182561


namespace NUMINAMATH_CALUDE_seven_layer_tower_lights_l1825_182548

/-- Represents a tower with lights -/
structure LightTower where
  layers : ℕ
  top_lights : ℕ
  total_lights : ℕ

/-- The sum of a geometric sequence -/
def geometricSum (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a₁ * (r^n - 1) / (r - 1)

/-- The theorem statement -/
theorem seven_layer_tower_lights (tower : LightTower) :
  tower.layers = 7 ∧
  tower.total_lights = 381 ∧
  (∀ i : ℕ, i < 7 → geometricSum tower.top_lights 2 (i + 1) ≤ tower.total_lights) →
  tower.top_lights = 3 := by
  sorry

end NUMINAMATH_CALUDE_seven_layer_tower_lights_l1825_182548


namespace NUMINAMATH_CALUDE_num_parallelepipeds_is_29_l1825_182540

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of four points in 3D space -/
def FourPoints := Fin 4 → Point3D

/-- Predicate to check if four points are non-coplanar -/
def NonCoplanar (points : FourPoints) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧
    ∀ (i : Fin 4), a * (points i).x + b * (points i).y + c * (points i).z + d = 0

/-- The number of distinct parallelepipeds that can be formed -/
def NumParallelepipeds (points : FourPoints) : ℕ := 29

/-- Theorem stating that the number of distinct parallelepipeds is 29 -/
theorem num_parallelepipeds_is_29 (points : FourPoints) (h : NonCoplanar points) :
  NumParallelepipeds points = 29 := by
  sorry

end NUMINAMATH_CALUDE_num_parallelepipeds_is_29_l1825_182540


namespace NUMINAMATH_CALUDE_sufficient_necessary_condition_l1825_182538

-- Define the interval (1, 4]
def OpenClosedInterval := { x : ℝ | 1 < x ∧ x ≤ 4 }

-- Define the inequality function
def InequalityFunction (m : ℝ) (x : ℝ) := x^2 - m*x + m > 0

-- State the theorem
theorem sufficient_necessary_condition :
  ∀ m : ℝ, (∀ x ∈ OpenClosedInterval, InequalityFunction m x) ↔ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_necessary_condition_l1825_182538


namespace NUMINAMATH_CALUDE_cubic_root_sum_cube_l1825_182501

theorem cubic_root_sum_cube (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cube_l1825_182501


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l1825_182530

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then -x^2 + 2*a*x - 2*a else a*x + 1

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ Set.Icc (-2 : ℝ) 0 ∧ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l1825_182530


namespace NUMINAMATH_CALUDE_trivia_team_score_l1825_182532

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_points : ℕ) :
  total_members = 12 →
  absent_members = 4 →
  total_points = 64 →
  (total_points / (total_members - absent_members) = 8) :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_score_l1825_182532


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1825_182546

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x

-- Define the theorem
theorem root_sum_theorem (h k : ℝ) 
  (h_root : p h = 1) 
  (k_root : p k = 5) : 
  h + k = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1825_182546


namespace NUMINAMATH_CALUDE_ages_solution_l1825_182598

/-- Represents the ages of Rahul and Deepak -/
structure Ages where
  rahul : ℕ
  deepak : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The ratio between Rahul and Deepak's age is 4:3
  4 * ages.deepak = 3 * ages.rahul ∧
  -- In 6 years, Rahul will be 26 years old
  ages.rahul + 6 = 26 ∧
  -- In 6 years, Deepak's age will be equal to half the sum of Rahul's present and future ages
  ages.deepak + 6 = (ages.rahul + (ages.rahul + 6)) / 2 ∧
  -- Five years after that, the sum of their ages will be 59
  (ages.rahul + 11) + (ages.deepak + 11) = 59

/-- The theorem to prove -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧ ages.rahul = 20 ∧ ages.deepak = 17 := by
  sorry

end NUMINAMATH_CALUDE_ages_solution_l1825_182598


namespace NUMINAMATH_CALUDE_integral_inequality_l1825_182593

open MeasureTheory

theorem integral_inequality 
  (f g : ℝ → ℝ) 
  (hf_pos : ∀ x, 0 ≤ f x) 
  (hg_pos : ∀ x, 0 ≤ g x)
  (hf_cont : Continuous f) 
  (hg_cont : Continuous g)
  (hf_incr : MonotoneOn f (Set.Icc 0 1))
  (hg_decr : AntitoneOn g (Set.Icc 0 1)) :
  ∫ x in (Set.Icc 0 1), f x * g x ≤ ∫ x in (Set.Icc 0 1), f x * g (1 - x) :=
sorry

end NUMINAMATH_CALUDE_integral_inequality_l1825_182593


namespace NUMINAMATH_CALUDE_truck_license_combinations_l1825_182571

/-- The number of possible letters for a truck license -/
def num_letters : ℕ := 3

/-- The number of digits in a truck license -/
def num_digits : ℕ := 6

/-- The number of possible digits (0-9) for each position -/
def digits_per_position : ℕ := 10

/-- The total number of possible truck license combinations -/
def total_combinations : ℕ := num_letters * (digits_per_position ^ num_digits)

theorem truck_license_combinations :
  total_combinations = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_truck_license_combinations_l1825_182571


namespace NUMINAMATH_CALUDE_complement_of_intersection_l1825_182574

def A : Set ℕ := {4, 5, 7}
def B : Set ℕ := {3, 4, 7, 8}
def U : Set ℕ := A ∪ B

theorem complement_of_intersection (A B : Set ℕ) (U : Set ℕ) (h : U = A ∪ B) :
  (A ∩ B)ᶜ = {3, 5, 8} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l1825_182574


namespace NUMINAMATH_CALUDE_base6_addition_example_l1825_182581

/-- Addition in base 6 -/
def base6_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 6 to base 10 -/
def base6_to_base10 (n : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 6 -/
def base10_to_base6 (n : ℕ) : ℕ := sorry

theorem base6_addition_example : base6_add 152 35 = 213 := by sorry

end NUMINAMATH_CALUDE_base6_addition_example_l1825_182581


namespace NUMINAMATH_CALUDE_problem_statement_l1825_182528

theorem problem_statement (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (heq : a + b = 1/a + 1/b) : 
  (a + b ≥ 2) ∧ ¬(a^2 + a < 2 ∧ b^2 + b < 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1825_182528


namespace NUMINAMATH_CALUDE_M_greater_than_N_l1825_182594

theorem M_greater_than_N (a : ℝ) : 2 * a * (a - 2) > (a + 1) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_M_greater_than_N_l1825_182594


namespace NUMINAMATH_CALUDE_correct_average_calculation_l1825_182568

theorem correct_average_calculation (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 15 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n : ℚ) * initial_avg - wrong_num + correct_num = n * 16 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l1825_182568


namespace NUMINAMATH_CALUDE_integer_solution_correct_rational_solution_correct_l1825_182557

-- Define the equation
def equation (x y : ℚ) : Prop := 2 * x^3 + x * y - 7 = 0

-- Define the set of integer solutions
def integer_solutions : Set (ℤ × ℤ) :=
  {(1, 5), (-1, -9), (7, -97), (-7, -99)}

-- Define the rational solution function
def rational_solution (x : ℚ) : ℚ := 7 / x - 2 * x^2

-- Theorem for integer solutions
theorem integer_solution_correct :
  ∀ (x y : ℤ), (x, y) ∈ integer_solutions → equation (x : ℚ) (y : ℚ) :=
sorry

-- Theorem for rational solutions
theorem rational_solution_correct :
  ∀ (x : ℚ), x ≠ 0 → equation x (rational_solution x) :=
sorry

end NUMINAMATH_CALUDE_integer_solution_correct_rational_solution_correct_l1825_182557


namespace NUMINAMATH_CALUDE_cost_per_bushel_approx_12_l1825_182529

-- Define the given constants
def apple_price : ℚ := 0.40
def apples_per_bushel : ℕ := 48
def profit : ℚ := 15
def apples_sold : ℕ := 100

-- Define the function to calculate the cost per bushel
def cost_per_bushel : ℚ :=
  let revenue := apple_price * apples_sold
  let cost := revenue - profit
  let bushels_sold := apples_sold / apples_per_bushel
  cost / bushels_sold

-- Theorem statement
theorem cost_per_bushel_approx_12 : 
  ∃ ε > 0, |cost_per_bushel - 12| < ε :=
sorry

end NUMINAMATH_CALUDE_cost_per_bushel_approx_12_l1825_182529


namespace NUMINAMATH_CALUDE_new_year_duty_arrangement_l1825_182552

theorem new_year_duty_arrangement (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 7 ∧ k = 4 ∧ m = 2 →
  (Nat.choose n m) * (Nat.descFactorial (n - m) (k - m)) = 420 :=
by sorry

end NUMINAMATH_CALUDE_new_year_duty_arrangement_l1825_182552


namespace NUMINAMATH_CALUDE_evaluate_expression_l1825_182505

theorem evaluate_expression : ((4^4 - 4*(4-2)^4)^4) = 136048896 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1825_182505


namespace NUMINAMATH_CALUDE_unique_number_property_l1825_182506

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l1825_182506


namespace NUMINAMATH_CALUDE_marks_leftover_amount_marks_leftover_is_980_l1825_182553

/-- Calculates the amount Mark has leftover each week after his raise and new expenses -/
theorem marks_leftover_amount (old_wage : ℝ) (raise_percentage : ℝ) 
  (hours_per_day : ℝ) (days_per_week : ℝ) (old_bills : ℝ) (trainer_cost : ℝ) : ℝ :=
  let new_wage := old_wage * (1 + raise_percentage / 100)
  let weekly_hours := hours_per_day * days_per_week
  let weekly_earnings := new_wage * weekly_hours
  let weekly_expenses := old_bills + trainer_cost
  weekly_earnings - weekly_expenses

/-- Proves that Mark has $980 leftover each week after his raise and new expenses -/
theorem marks_leftover_is_980 : 
  marks_leftover_amount 40 5 8 5 600 100 = 980 := by
  sorry

end NUMINAMATH_CALUDE_marks_leftover_amount_marks_leftover_is_980_l1825_182553


namespace NUMINAMATH_CALUDE_small_cylinder_radius_l1825_182537

/-- Proves that the radius of smaller cylinders is √(24/5) meters given the specified conditions -/
theorem small_cylinder_radius 
  (large_diameter : ℝ) 
  (large_height : ℝ) 
  (small_height : ℝ) 
  (num_small_cylinders : ℕ) 
  (h_large_diameter : large_diameter = 6)
  (h_large_height : large_height = 8)
  (h_small_height : small_height = 5)
  (h_num_small_cylinders : num_small_cylinders = 3)
  : ∃ (small_radius : ℝ), small_radius = Real.sqrt (24 / 5) := by
  sorry

#check small_cylinder_radius

end NUMINAMATH_CALUDE_small_cylinder_radius_l1825_182537


namespace NUMINAMATH_CALUDE_plot_length_l1825_182554

/-- The length of a rectangular plot given specific conditions -/
theorem plot_length (breadth : ℝ) (length : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 32 →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  2 * (length + breadth) * cost_per_meter = total_cost →
  length = 66 := by
sorry

end NUMINAMATH_CALUDE_plot_length_l1825_182554


namespace NUMINAMATH_CALUDE_milk_bottle_boxes_l1825_182526

/-- Given a total number of milk bottles, bottles per bag, and bags per box,
    calculate the total number of boxes. -/
def calculate_boxes (total_bottles : ℕ) (bottles_per_bag : ℕ) (bags_per_box : ℕ) : ℕ :=
  total_bottles / (bottles_per_bag * bags_per_box)

/-- Theorem stating that given 8640 milk bottles, with 12 bottles per bag and 6 bags per box,
    the total number of boxes is equal to 120. -/
theorem milk_bottle_boxes :
  calculate_boxes 8640 12 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_milk_bottle_boxes_l1825_182526


namespace NUMINAMATH_CALUDE_product_pass_rate_l1825_182565

/-- The pass rate of a product going through two independent processing steps -/
def pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

/-- Theorem stating that the pass rate of a product going through two independent processing steps
    with defect rates a and b is (1-a)·(1-b) -/
theorem product_pass_rate (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  pass_rate a b = (1 - a) * (1 - b) := by
  sorry

end NUMINAMATH_CALUDE_product_pass_rate_l1825_182565


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1825_182549

theorem expand_and_simplify (x : ℝ) : 5 * (x + 6) * (x + 2) * (x + 7) = 5*x^3 + 75*x^2 + 340*x + 420 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1825_182549


namespace NUMINAMATH_CALUDE_same_terminal_side_as_405_degrees_l1825_182551

theorem same_terminal_side_as_405_degrees : ∀ (k : ℤ),
  ∃ (n : ℤ), 405 = n * 360 + 45 ∧ (k * 360 + 45) % 360 = 45 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_as_405_degrees_l1825_182551


namespace NUMINAMATH_CALUDE_total_cost_of_items_l1825_182556

/-- The total cost of items given their price relationships -/
theorem total_cost_of_items (chair_price : ℝ) : 
  chair_price > 0 →
  let table_price := 3 * chair_price
  let couch_price := 5 * table_price
  couch_price = 300 →
  chair_price + table_price + couch_price = 380 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_items_l1825_182556


namespace NUMINAMATH_CALUDE_max_value_of_f_l1825_182562

/-- Definition of the sum of the first n terms of the geometric sequence -/
def S (n : ℕ) (k : ℝ) : ℝ := 2^(n-1) + k

/-- Definition of the function f -/
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x^2 - 2*x + 1

/-- Theorem stating the maximum value of f -/
theorem max_value_of_f (k : ℝ) : 
  (∃ (n : ℕ), ∀ (m : ℕ), S m k = 2^(m-1) + k) → 
  (∃ (x : ℝ), ∀ (y : ℝ), f k y ≤ f k x) ∧ 
  (∃ (x : ℝ), f k x = 5/2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1825_182562


namespace NUMINAMATH_CALUDE_g_inv_composition_l1825_182502

-- Define the function g
def g : Fin 5 → Fin 5
| 1 => 4
| 2 => 3
| 3 => 1
| 4 => 5
| 5 => 2

-- Define the inverse function g⁻¹
def g_inv : Fin 5 → Fin 5
| 1 => 3
| 2 => 5
| 3 => 2
| 4 => 1
| 5 => 4

-- State the theorem
theorem g_inv_composition :
  g_inv (g_inv (g_inv 3)) = 4 := by sorry

end NUMINAMATH_CALUDE_g_inv_composition_l1825_182502


namespace NUMINAMATH_CALUDE_gcd_840_1764_l1825_182534

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l1825_182534


namespace NUMINAMATH_CALUDE_hula_hoop_ratio_l1825_182582

def nancy_time : ℕ := 10
def casey_time : ℕ := nancy_time - 3
def morgan_time : ℕ := 21

theorem hula_hoop_ratio : 
  ∃ (k : ℕ), k > 0 ∧ morgan_time = k * casey_time ∧ morgan_time / casey_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_hula_hoop_ratio_l1825_182582


namespace NUMINAMATH_CALUDE_min_value_w_l1825_182500

theorem min_value_w (x y z : ℝ) :
  x^2 + 4*y^2 + 8*x - 6*y + z - 20 ≥ z - 38.25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_w_l1825_182500


namespace NUMINAMATH_CALUDE_mean_squares_sum_l1825_182504

theorem mean_squares_sum (x y z : ℝ) : 
  (x + y + z) / 3 = 10 →
  (x * y * z) ^ (1/3 : ℝ) = 6 →
  3 / (1/x + 1/y + 1/z) = 4 →
  x^2 + y^2 + z^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_mean_squares_sum_l1825_182504


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1825_182596

theorem sufficient_not_necessary_condition :
  (∃ x : ℝ, x^2 - 2*x < 0 → abs x < 2) ∧
  (∃ x : ℝ, abs x < 2 ∧ ¬(x^2 - 2*x < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1825_182596


namespace NUMINAMATH_CALUDE_odd_function_property_l1825_182589

-- Define odd functions
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function F
def F (f g : ℝ → ℝ) (x : ℝ) : ℝ := 3 * f x + 5 * g x + 2

-- Theorem statement
theorem odd_function_property (f g : ℝ → ℝ) (a : ℝ) 
  (hf : OddFunction f) (hg : OddFunction g) (hFa : F f g a = 3) : 
  F f g (-a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1825_182589


namespace NUMINAMATH_CALUDE_slope_problem_l1825_182585

theorem slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (2*m - 1) / ((m + 1) - 2*m) = m) : m = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_problem_l1825_182585


namespace NUMINAMATH_CALUDE_solve_magazine_problem_l1825_182509

def magazine_problem (cost_price selling_price gain : ℚ) : Prop :=
  ∃ (num_magazines : ℕ), 
    (selling_price - cost_price) * num_magazines = gain ∧
    num_magazines > 0

theorem solve_magazine_problem : 
  magazine_problem 3 3.5 5 → ∃ (num_magazines : ℕ), num_magazines = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_magazine_problem_l1825_182509


namespace NUMINAMATH_CALUDE_triangle_properties_l1825_182567

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

/-- The main theorem about the specific acute triangle. -/
theorem triangle_properties (t : AcuteTriangle)
    (h1 : Real.sqrt 3 * t.a - 2 * t.b * Real.sin t.A = 0)
    (h2 : t.a + t.c = 5)
    (h3 : t.a > t.c)
    (h4 : t.b = Real.sqrt 7) :
    t.B = π/3 ∧ (1/2 * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1825_182567


namespace NUMINAMATH_CALUDE_motorcycle_friction_speed_relation_l1825_182520

/-- Proves that the minimum friction coefficient for a motorcycle riding on vertical walls
    is inversely proportional to the square of its speed. -/
theorem motorcycle_friction_speed_relation 
  (m : ℝ) -- mass of the motorcycle
  (g : ℝ) -- acceleration due to gravity
  (r : ℝ) -- radius of the circular room
  (s : ℝ) -- speed of the motorcycle
  (μ : ℝ → ℝ) -- friction coefficient as a function of speed
  (h_positive : m > 0 ∧ g > 0 ∧ r > 0 ∧ s > 0) -- positivity conditions
  (h_equilibrium : ∀ s, μ s * (m * s^2 / r) = m * g) -- equilibrium condition
  : ∃ (k : ℝ), ∀ s, μ s = k / s^2 :=
sorry

end NUMINAMATH_CALUDE_motorcycle_friction_speed_relation_l1825_182520


namespace NUMINAMATH_CALUDE_assignPositions_eq_95040_l1825_182597

/-- The number of ways to assign 5 distinct positions to 5 people chosen from a group of 12 people,
    where each person can only hold one position. -/
def assignPositions : ℕ := 12 * 11 * 10 * 9 * 8

/-- Theorem stating that the number of ways to assign the positions is 95,040. -/
theorem assignPositions_eq_95040 : assignPositions = 95040 := by
  sorry

end NUMINAMATH_CALUDE_assignPositions_eq_95040_l1825_182597


namespace NUMINAMATH_CALUDE_fred_initial_balloons_l1825_182584

/-- The number of green balloons Fred gave to Sandy -/
def balloons_given : ℕ := 221

/-- The number of green balloons Fred has left -/
def balloons_left : ℕ := 488

/-- The initial number of green balloons Fred had -/
def initial_balloons : ℕ := balloons_given + balloons_left

theorem fred_initial_balloons : initial_balloons = 709 := by
  sorry

end NUMINAMATH_CALUDE_fred_initial_balloons_l1825_182584


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1825_182519

theorem arithmetic_geometric_mean_inequality {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1825_182519


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_given_numbers_l1825_182592

theorem arithmetic_mean_of_given_numbers : 
  let numbers : List ℕ := [16, 24, 40, 32]
  (numbers.sum / numbers.length : ℚ) = 28 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_given_numbers_l1825_182592


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1825_182503

theorem complex_equation_sum (a b : ℝ) : 
  (a / (1 - Complex.I)) + (b / (1 - 2 * Complex.I)) = (1 + 3 * Complex.I) / 4 → 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1825_182503


namespace NUMINAMATH_CALUDE_number_difference_l1825_182545

theorem number_difference (L S : ℕ) (hL : L > S) (hDiv : L = 6 * S + 20) (hLValue : L = 1634) : 
  L - S = 1365 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l1825_182545


namespace NUMINAMATH_CALUDE_magnitude_v_l1825_182536

theorem magnitude_v (u v : ℂ) (h1 : u * v = 24 - 10 * I) (h2 : Complex.abs u = 5) :
  Complex.abs v = 26 / 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_v_l1825_182536


namespace NUMINAMATH_CALUDE_seconds_in_3h45m_is_13500_l1825_182570

/-- Converts hours to minutes -/
def hours_to_minutes (h : ℕ) : ℕ := h * 60

/-- Converts minutes to seconds -/
def minutes_to_seconds (m : ℕ) : ℕ := m * 60

/-- The number of seconds in 3 hours and 45 minutes -/
def seconds_in_3h45m : ℕ := minutes_to_seconds (hours_to_minutes 3 + 45)

theorem seconds_in_3h45m_is_13500 : seconds_in_3h45m = 13500 := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_3h45m_is_13500_l1825_182570


namespace NUMINAMATH_CALUDE_minjeong_marbles_l1825_182544

/-- Given that the total number of marbles is 43 and Yunjae has 5 more marbles than Minjeong,
    prove that Minjeong has 19 marbles. -/
theorem minjeong_marbles : 
  ∀ (y m : ℕ), y + m = 43 → y = m + 5 → m = 19 := by
  sorry

end NUMINAMATH_CALUDE_minjeong_marbles_l1825_182544


namespace NUMINAMATH_CALUDE_opposite_sides_range_l1825_182569

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Determines if two points are on opposite sides of a line -/
def oppositeSides (p1 p2 : Point2D) (a : ℝ) : Prop :=
  (3 * p1.x - 2 * p1.y + a) * (3 * p2.x - 2 * p2.y + a) < 0

/-- The theorem stating the range of 'a' for which the given points are on opposite sides of the line -/
theorem opposite_sides_range :
  ∀ a : ℝ, 
    oppositeSides (Point2D.mk 3 1) (Point2D.mk (-4) 6) a ↔ -7 < a ∧ a < 24 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l1825_182569


namespace NUMINAMATH_CALUDE_min_value_quadratic_function_l1825_182588

theorem min_value_quadratic_function :
  ∃ (min : ℝ), min = -11.25 ∧
  ∀ (x y : ℝ), 2 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_function_l1825_182588


namespace NUMINAMATH_CALUDE_connie_calculation_l1825_182576

theorem connie_calculation (x : ℝ) : 200 - x = 100 → 200 + x = 300 := by
  sorry

end NUMINAMATH_CALUDE_connie_calculation_l1825_182576


namespace NUMINAMATH_CALUDE_x_intercepts_count_l1825_182531

theorem x_intercepts_count : 
  (⌊(100000 : ℝ) / Real.pi⌋ - ⌊(10000 : ℝ) / Real.pi⌋ : ℤ) = 28647 := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l1825_182531


namespace NUMINAMATH_CALUDE_pyramid_side_length_l1825_182590

/-- Represents a pyramid with a rectangular base ABCD and vertex E above A -/
structure Pyramid where
  -- Base side lengths
  AB : ℝ
  BC : ℝ
  -- Angles
  BCE : ℝ
  ADE : ℝ

/-- Theorem: In a pyramid with given conditions, BC = 2√2 -/
theorem pyramid_side_length (p : Pyramid)
  (h_AB : p.AB = 4)
  (h_BCE : p.BCE = Real.pi / 3)  -- 60 degrees in radians
  (h_ADE : p.ADE = Real.pi / 4)  -- 45 degrees in radians
  : p.BC = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_side_length_l1825_182590


namespace NUMINAMATH_CALUDE_equal_strawberry_division_l1825_182558

def strawberry_division (brother_baskets : ℕ) (strawberries_per_basket : ℕ) : ℕ :=
  let brother_strawberries := brother_baskets * strawberries_per_basket
  let kimberly_strawberries := 8 * brother_strawberries
  let parents_strawberries := kimberly_strawberries - 93
  let total_strawberries := kimberly_strawberries + brother_strawberries + parents_strawberries
  total_strawberries / 4

theorem equal_strawberry_division :
  strawberry_division 3 15 = 168 := by
  sorry

end NUMINAMATH_CALUDE_equal_strawberry_division_l1825_182558


namespace NUMINAMATH_CALUDE_cookies_remaining_batches_l1825_182547

/-- Given the following conditions:
  * Each batch of cookies requires 2 cups of flour
  * 3 batches of cookies were baked
  * The initial amount of flour was 20 cups
  Prove that 7 additional batches of cookies can be made with the remaining flour -/
theorem cookies_remaining_batches 
  (flour_per_batch : ℕ) 
  (batches_baked : ℕ) 
  (initial_flour : ℕ) : 
  flour_per_batch = 2 →
  batches_baked = 3 →
  initial_flour = 20 →
  (initial_flour - flour_per_batch * batches_baked) / flour_per_batch = 7 :=
by sorry

end NUMINAMATH_CALUDE_cookies_remaining_batches_l1825_182547


namespace NUMINAMATH_CALUDE_number_of_children_is_six_l1825_182572

/-- Represents the age of the youngest child -/
def youngest_age : ℕ := 6

/-- Represents the common difference between ages -/
def age_difference : ℕ := 3

/-- Represents the sum of all ages -/
def total_age : ℕ := 60

/-- Calculates the sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Theorem stating that the number of children is 6 -/
theorem number_of_children_is_six :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sum n youngest_age age_difference = total_age ∧ n = 6 :=
sorry

end NUMINAMATH_CALUDE_number_of_children_is_six_l1825_182572


namespace NUMINAMATH_CALUDE_valid_plans_count_l1825_182514

/-- Represents the three universities --/
inductive University : Type
| Peking : University
| Tsinghua : University
| Renmin : University

/-- Represents the five students --/
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

/-- A recommendation plan is a function from Student to University --/
def RecommendationPlan := Student → University

/-- Checks if a recommendation plan is valid --/
def isValidPlan (plan : RecommendationPlan) : Prop :=
  (∃ s, plan s = University.Peking) ∧
  (∃ s, plan s = University.Tsinghua) ∧
  (∃ s, plan s = University.Renmin) ∧
  (plan Student.A ≠ University.Peking)

/-- The number of valid recommendation plans --/
def numberOfValidPlans : ℕ := sorry

theorem valid_plans_count : numberOfValidPlans = 100 := by sorry

end NUMINAMATH_CALUDE_valid_plans_count_l1825_182514


namespace NUMINAMATH_CALUDE_cosine_sine_expression_value_l1825_182524

theorem cosine_sine_expression_value : 
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cosine_sine_expression_value_l1825_182524


namespace NUMINAMATH_CALUDE_rectangular_equation_chord_length_l1825_182508

-- Define the polar equation of curve C
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ)^2 = 8 * Real.cos θ

-- Define the parametric equations of line l
def line_equation (t x y : ℝ) : Prop :=
  x = 2 + (1/2) * t ∧ y = (Real.sqrt 3 / 2) * t

-- Theorem for the rectangular equation of curve C
theorem rectangular_equation (x y : ℝ) :
  (∃ ρ θ, polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔
  y^2 = 8 * x :=
sorry

-- Theorem for the length of chord AB
theorem chord_length :
  ∃ t₁ t₂ x₁ y₁ x₂ y₂,
    line_equation t₁ x₁ y₁ ∧ line_equation t₂ x₂ y₂ ∧
    y₁^2 = 8 * x₁ ∧ y₂^2 = 8 * x₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 32/3 :=
sorry

end NUMINAMATH_CALUDE_rectangular_equation_chord_length_l1825_182508


namespace NUMINAMATH_CALUDE_triangle_formation_l1825_182535

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  ¬(can_form_triangle 3 6 3) ∧
  (can_form_triangle 3 4 5) ∧
  (can_form_triangle (Real.sqrt 3) 1 2) ∧
  (can_form_triangle 1.5 2.5 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l1825_182535


namespace NUMINAMATH_CALUDE_greatest_number_satisfying_conditions_l1825_182518

/-- A number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

/-- A number is composed of the square of two distinct prime factors -/
def is_product_of_two_distinct_prime_squares (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p^2 * q^2

/-- A number has an odd number of positive factors -/
def has_odd_number_of_factors (n : ℕ) : Prop :=
  Odd (Nat.card (Nat.divisors n))

/-- The main theorem -/
theorem greatest_number_satisfying_conditions : 
  (∀ n : ℕ, n < 200 → is_perfect_square n → 
    is_product_of_two_distinct_prime_squares n → 
    has_odd_number_of_factors n → n ≤ 196) ∧ 
  (196 < 200 ∧ is_perfect_square 196 ∧ 
    is_product_of_two_distinct_prime_squares 196 ∧ 
    has_odd_number_of_factors 196) := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_satisfying_conditions_l1825_182518


namespace NUMINAMATH_CALUDE_natural_numbers_difference_l1825_182560

theorem natural_numbers_difference (a b : ℕ) : 
  a + b = 20250 → 
  b % 15 = 0 → 
  a = b / 3 → 
  b - a = 10130 := by
sorry

end NUMINAMATH_CALUDE_natural_numbers_difference_l1825_182560


namespace NUMINAMATH_CALUDE_triangle_area_from_square_sides_l1825_182511

theorem triangle_area_from_square_sides (a b c : Real) 
  (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) : 
  (1/2) * a * b = 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_from_square_sides_l1825_182511


namespace NUMINAMATH_CALUDE_lois_final_book_count_l1825_182512

def calculate_final_books (initial_books : ℕ) : ℕ :=
  let books_after_giving := initial_books - (initial_books / 4)
  let nonfiction_books := (books_after_giving * 60) / 100
  let kept_nonfiction := nonfiction_books / 2
  let fiction_books := books_after_giving - nonfiction_books
  let kept_fiction := fiction_books - (fiction_books / 3)
  let new_books := 12
  kept_nonfiction + kept_fiction + new_books

theorem lois_final_book_count :
  calculate_final_books 150 = 76 := by
  sorry

end NUMINAMATH_CALUDE_lois_final_book_count_l1825_182512


namespace NUMINAMATH_CALUDE_min_max_bound_l1825_182559

theorem min_max_bound (x₁ x₂ x₃ : ℝ) (h_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0) 
  (h_sum : x₁ + x₂ + x₃ = 1) : 
  1 ≤ (x₁ + 3*x₂ + 5*x₃)*(x₁ + x₂/3 + x₃/5) ∧ 
  (x₁ + 3*x₂ + 5*x₃)*(x₁ + x₂/3 + x₃/5) ≤ 9/5 := by
  sorry

#check min_max_bound

end NUMINAMATH_CALUDE_min_max_bound_l1825_182559


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4141_l1825_182523

theorem largest_prime_factor_of_4141 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4141 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4141 → q ≤ p ∧ p = 101 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4141_l1825_182523


namespace NUMINAMATH_CALUDE_apple_juice_problem_l1825_182541

theorem apple_juice_problem (x y : ℝ) : 
  (x - 1 = y + 1) →  -- Equalizing condition
  (x + 9 = 30) →     -- First barrel full after transfer
  (y - 9 = 10) →     -- Second barrel one-third full after transfer
  (x = 21 ∧ y = 19 ∧ x + y = 40) := by
  sorry

end NUMINAMATH_CALUDE_apple_juice_problem_l1825_182541


namespace NUMINAMATH_CALUDE_secondPlayerCanEnsureDivisibilityFor60_secondPlayerCannotEnsureDivisibilityFor14_l1825_182580

/-- Represents a strategy for the second player to choose digits -/
def Strategy := Nat → Nat → Nat

/-- Checks if a list of digits is divisible by 9 -/
def isDivisibleBy9 (digits : List Nat) : Prop :=
  (digits.sum % 9) = 0

/-- Generates all possible sequences of digits for the first player -/
def firstPlayerSequences (n : Nat) : List (List Nat) :=
  sorry

/-- Applies the second player's strategy to the first player's sequence -/
def applyStrategy (firstPlayerSeq : List Nat) (strategy : Strategy) : List Nat :=
  sorry

theorem secondPlayerCanEnsureDivisibilityFor60 :
  ∃ (strategy : Strategy),
    ∀ (firstPlayerSeq : List Nat),
      firstPlayerSeq.length = 30 →
      firstPlayerSeq.all (λ d => d ≥ 1 ∧ d ≤ 5) →
      isDivisibleBy9 (applyStrategy firstPlayerSeq strategy) :=
sorry

theorem secondPlayerCannotEnsureDivisibilityFor14 :
  ∀ (strategy : Strategy),
    ∃ (firstPlayerSeq : List Nat),
      firstPlayerSeq.length = 7 →
      firstPlayerSeq.all (λ d => d ≥ 1 ∧ d ≤ 5) →
      ¬isDivisibleBy9 (applyStrategy firstPlayerSeq strategy) :=
sorry

end NUMINAMATH_CALUDE_secondPlayerCanEnsureDivisibilityFor60_secondPlayerCannotEnsureDivisibilityFor14_l1825_182580
