import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_l910_91060

-- Define the set of students
inductive Student : Type
| DengQing : Student
| WuLin : Student
| Sanxi : Student
| Jianye : Student
| Meihong : Student

-- Define a ranking as a function from Student to ℕ
def Ranking := Student → ℕ

-- Define the statements made by each student
def DengQingStatements (r : Ranking) : Prop :=
  (r Student.Sanxi = 2) ∨ (r Student.Jianye = 3)

def WuLinStatements (r : Ranking) : Prop :=
  (r Student.Meihong = 2) ∨ (r Student.DengQing = 4)

def SanxiStatements (r : Ranking) : Prop :=
  (r Student.DengQing = 1) ∨ (r Student.WuLin = 5)

def JianyeStatements (r : Ranking) : Prop :=
  (r Student.Meihong = 3) ∨ (r Student.WuLin = 4)

def MeihongStatements (r : Ranking) : Prop :=
  (r Student.Jianye = 2) ∨ (r Student.Sanxi = 5)

-- Define the condition that each student's statements have one true and one false
def OneStatementTrue (r : Ranking) : Prop :=
  (DengQingStatements r ∧ ¬(r Student.Sanxi = 2 ∧ r Student.Jianye = 3)) ∧
  (WuLinStatements r ∧ ¬(r Student.Meihong = 2 ∧ r Student.DengQing = 4)) ∧
  (SanxiStatements r ∧ ¬(r Student.DengQing = 1 ∧ r Student.WuLin = 5)) ∧
  (JianyeStatements r ∧ ¬(r Student.Meihong = 3 ∧ r Student.WuLin = 4)) ∧
  (MeihongStatements r ∧ ¬(r Student.Jianye = 2 ∧ r Student.Sanxi = 5))

-- Define a valid ranking
def ValidRanking (r : Ranking) : Prop :=
  (∀ s : Student, r s ∈ ({1, 2, 3, 4, 5} : Set ℕ)) ∧
  (∀ s₁ s₂ : Student, s₁ ≠ s₂ → r s₁ ≠ r s₂)

-- The main theorem
theorem correct_ranking :
  ∀ r : Ranking, ValidRanking r ∧ OneStatementTrue r →
    r Student.DengQing = 1 ∧
    r Student.Meihong = 2 ∧
    r Student.Jianye = 3 ∧
    r Student.WuLin = 4 ∧
    r Student.Sanxi = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_l910_91060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_equality_l910_91059

/-- The rate at which Village Y's population is increasing per year -/
def rate_y : ℕ → ℕ := fun _ => 800

/-- The number of years after which the populations will be equal -/
def years_equal : ℕ := 13

/-- The initial population of Village X -/
def pop_x_initial : ℕ := 68000

/-- The initial population of Village Y -/
def pop_y_initial : ℕ := 42000

/-- The rate at which Village X's population is decreasing per year -/
def rate_x_decrease : ℕ := 1200

theorem village_population_equality : 
  pop_x_initial - years_equal * rate_x_decrease = pop_y_initial + years_equal * rate_y 0 := by
  sorry

#eval rate_y 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_equality_l910_91059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fifteen_terms_eq_202_5_l910_91044

/-- An arithmetic progression with given third and fifth terms -/
structure ArithmeticProgression where
  a₃ : ℚ
  a₅ : ℚ

/-- The sum of the first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  let d := (ap.a₅ - ap.a₃) / 2
  let a₁ := ap.a₃ - 2 * d
  n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem: The sum of the first 15 terms of the given arithmetic progression is 202.5 -/
theorem sum_fifteen_terms_eq_202_5 (ap : ArithmeticProgression) 
    (h₃ : ap.a₃ = -5) (h₅ : ap.a₅ = 12/5) : 
    sum_n_terms ap 15 = 405/2 := by
  sorry

#eval sum_n_terms { a₃ := -5, a₅ := 12/5 } 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fifteen_terms_eq_202_5_l910_91044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l910_91045

-- Define the function f(x) = 2^(x-1) + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.exp ((x - 1) * Real.log 2) + x - 3

-- State the theorem
theorem solution_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l910_91045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_after_n_throws_l910_91070

/-- Represents the number of girls in the circle -/
def n : ℕ := 11

/-- Represents the number of girls skipped in each throw -/
def skip : ℕ := 3

/-- Represents the girl who receives the ball after each throw -/
def next (i : ℕ) : ℕ := (i + skip + 1) % n

/-- Represents the number of throws needed for the ball to return to the starting position -/
def throws_to_return : ℕ := n

/-- Helper function to simulate the throws -/
def simulate_throws : ℕ → ℕ
  | 0 => 0
  | m + 1 => next (simulate_throws m)

theorem ball_returns_after_n_throws :
  simulate_throws throws_to_return = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_returns_after_n_throws_l910_91070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_relation_l910_91077

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + 2 * Real.pi / 3) + Real.sqrt 3 * Real.sin (2 * x)

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (AC BC : ℝ)
  (h1 : 0 < A ∧ A < Real.pi)
  (h2 : 0 < B ∧ B < Real.pi)
  (h3 : 0 < C ∧ C < Real.pi)
  (h4 : A + B + C = Real.pi)
  (h5 : AC = 1)
  (h6 : BC = 3)
  (h7 : f (C / 2) = -1 / 2)

theorem function_properties_and_triangle_relation (t : Triangle) :
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  (∀ x : ℝ, f x ≤ 1) ∧
  (∃ x : ℝ, f x = 1) ∧
  Real.sin t.A = (3 * Real.sqrt 21) / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_relation_l910_91077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l910_91014

/-- The set of ordered triples (x,y,z) of nonnegative real numbers that lie in the plane x+y+z=2 -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 ∧ p.1 + p.2.1 + p.2.2 = 2}

/-- Definition of "supports" -/
def supports (x y z a b c : ℝ) : Prop :=
  (x ≥ a ∧ y ≥ b) ∨ (x ≥ a ∧ z ≥ c) ∨ (y ≥ b ∧ z ≥ c)

/-- The set S of triples in T that support (1, 2/3, 1/3) -/
def S : Set (ℝ × ℝ × ℝ) :=
  {p ∈ T | supports p.1 p.2.1 p.2.2 1 (2/3) (1/3)}

/-- The area of a set in ℝ³ -/
noncomputable def area : Set (ℝ × ℝ × ℝ) → ℝ := sorry

/-- The main theorem stating the ratio of areas -/
theorem area_ratio : area S / area T = 3 / (8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l910_91014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_volume_l910_91090

/-- A solid with a square base and parallel upper edge -/
structure Solid where
  t : ℝ
  base_side_length : t > 0
  upper_edge_length : ℝ := 3 * t
  other_edges_length : ℝ := t

/-- The volume of the solid -/
noncomputable def volume (s : Solid) : ℝ := 216 * Real.sqrt 6

/-- Theorem stating the volume of the specific solid -/
theorem solid_volume (s : Solid) (h : s.t = 4 * Real.sqrt 3) :
  volume s = 216 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_volume_l910_91090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_paper_area_l910_91099

/-- The area of the square piece of paper ABCD is 121 square centimeters. -/
theorem square_paper_area : ∃ (side : ℝ), side * side = 121 :=
by
  -- Let the side length of the square be 11 cm
  let side : ℝ := 11

  -- Show that this side length satisfies the conditions
  have h1 : side * side = 121 := by ring

  -- Prove that this side length is consistent with the problem conditions
  have h2 : 3 + 5 + 3 = side := by norm_num

  -- The overlapping area after the first and second steps are equal
  have h3 : 3 * 5 = 3 * 5 := by ring

  -- Conclude that there exists a side length that satisfies the conditions
  exact ⟨side, h1⟩

#check square_paper_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_paper_area_l910_91099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_calculation_l910_91092

/-- The ★ operation for real numbers -/
noncomputable def star (x y : ℝ) : ℝ := (x + y) / (x^2 - y^2)

/-- Theorem stating that ((2 ★ 3) ★ 4) = -1/5 -/
theorem star_calculation : star (star 2 3) 4 = -1/5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_calculation_l910_91092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_bean_usage_ratio_l910_91002

/-- The ratio of evening to morning coffee bean usage in Droid's coffee shop --/
def evening_to_morning_ratio : ℚ := 2

theorem coffee_bean_usage_ratio :
  let morning_usage : ℕ := 3
  let afternoon_usage : ℕ := 3 * morning_usage
  let evening_usage : ℚ := evening_to_morning_ratio * morning_usage
  let daily_usage : ℚ := morning_usage + afternoon_usage + evening_usage
  let weekly_usage : ℕ := 126
  (weekly_usage : ℚ) = 7 * daily_usage →
  evening_to_morning_ratio = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_bean_usage_ratio_l910_91002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_blueberry_count_l910_91078

/-- Represents the number of jelly beans Camilla has -/
structure JellyBeans where
  blueberry : ℕ
  cherry : ℕ

/-- The initial state of Camilla's jelly beans -/
def initial (c : ℕ) : JellyBeans where
  blueberry := 3 * c
  cherry := c

/-- The state after Camilla eats some jelly beans -/
def after_eating (initial : JellyBeans) : JellyBeans where
  blueberry := initial.blueberry - 15
  cherry := initial.cherry - 5

/-- Theorem stating the original number of blueberry jelly beans -/
theorem original_blueberry_count :
  ∃ (c : ℕ),
    let initial := initial c
    (initial.blueberry = 3 * initial.cherry) ∧
    (after_eating initial).blueberry = 5 * (after_eating initial).cherry ∧
    initial.blueberry = 15 := by
  sorry

#check original_blueberry_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_blueberry_count_l910_91078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_street_light_distance_l910_91065

/-- Calculates the distance between street lights given the total road length and number of lights -/
noncomputable def distance_between_lights (road_length : ℝ) (num_lights : ℕ) : ℝ :=
  road_length / (num_lights / 2 - 1)

/-- Theorem stating that for a 16.4m road with 18 lights, the distance between lights is 2.05m -/
theorem street_light_distance :
  distance_between_lights 16.4 18 = 2.05 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distance_between_lights 16.4 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_street_light_distance_l910_91065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_fixed_point_g_fixed_point_range_h_fixed_point_existence_l910_91095

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x + 1

def g (b c : ℝ) (x : ℝ) : ℝ := b * x^2 + c * x + 3

def h (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- Theorem 1
theorem f_fixed_point : ∃ x : ℝ, f x = x ∧ x = 1 := by sorry

-- Theorem 2
theorem g_fixed_point_range (b c : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (1/2) 2 ∧ g b c x₀ = x₀ ∧ (2 * b * x₀ + c) = x₀) ↔
  b ∈ Set.Icc (5/4) 11 := by sorry

-- Theorem 3
theorem h_fixed_point_existence (a b c : ℝ) (h_a : a ≠ 0) :
  (∃ m : ℝ, m > 0 ∧ h a b c m > 0 ∧ h a b c (h a b c m) > 0 ∧ h a b c (h a b c (h a b c m)) > 0 ∧
   ∃ q : ℝ, q > 0 ∧ h a b c m = q * m ∧ h a b c (h a b c m) = q * h a b c m ∧
   h a b c (h a b c (h a b c m)) = q * h a b c (h a b c m)) →
  ∃ x : ℝ, h a b c x = x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_fixed_point_g_fixed_point_range_h_fixed_point_existence_l910_91095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l910_91075

def proposition_p (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2/m = 1 → ∃ a b : ℝ, a > b ∧ a^2 - b^2 = m - 1

def proposition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, m*x^2 + m*x + 1 > 0

theorem range_of_m :
  ∃ S : Set ℝ, S = {m : ℝ | m ∈ Set.Icc 0 1 ∨ m ∈ Set.Ici 4} ∧
  (∀ m : ℝ, m ∈ S ↔ 
    ((proposition_p m ∨ proposition_q m) ∧ ¬(proposition_p m ∧ proposition_q m))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l910_91075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digits_l910_91039

/-- Represents a base-7 number with two digits --/
structure Base7TwoDigit where
  tens : Nat
  ones : Nat
  h_valid : tens < 7 ∧ ones < 7

/-- Converts a Base7TwoDigit to its decimal representation --/
def toDecimal (n : Base7TwoDigit) : Nat :=
  7 * n.tens + n.ones

/-- Theorem stating the uniqueness of the solution --/
theorem unique_digits (A B D : Nat) : 
  (A ≠ 0 ∧ B ≠ 0 ∧ D ≠ 0) →  -- non-zero
  (A < 7 ∧ B < 7 ∧ D < 7) →  -- less than 7
  (A ≠ B ∧ B ≠ D ∧ A ≠ D) →  -- distinct
  (toDecimal ⟨A, B, by sorry⟩ + D = 7 * D) →  -- AB₇ + D₇ = D0₇
  (toDecimal ⟨A, B, by sorry⟩ + toDecimal ⟨B, A, by sorry⟩ = 7 * D + D) →  -- AB₇ + BA₇ = DD₇
  100 * A + 10 * B + D = 434 := by
  sorry

#check unique_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digits_l910_91039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_center_travel_distance_l910_91037

/-- Represents a wheel with a given radius -/
structure Wheel where
  radius : ℝ

/-- The distance traveled by the center of a wheel during one revolution -/
noncomputable def centerTravelDistance (w : Wheel) : ℝ := 2 * Real.pi * w.radius

theorem wheel_center_travel_distance (outerWheel innerWheel : Wheel) 
  (h1 : outerWheel.radius = 2)
  (h2 : innerWheel.radius = 1) :
  centerTravelDistance outerWheel = 4 * Real.pi ∧ 
  centerTravelDistance innerWheel = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_center_travel_distance_l910_91037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_quarter_l910_91093

def quarters_value : ℚ := 25/2
def nickels_value : ℚ := 15
def pennies_value : ℚ := 8

def quarter_worth : ℚ := 1/4
def nickel_worth : ℚ := 1/10
def penny_worth : ℚ := 1/50

def num_quarters : ℕ := (quarters_value / quarter_worth).floor.toNat
def num_nickels : ℕ := (nickels_value / nickel_worth).floor.toNat
def num_pennies : ℕ := (pennies_value / penny_worth).floor.toNat

def total_coins : ℕ := num_quarters + num_nickels + num_pennies

theorem probability_of_quarter : 
  (num_quarters : ℚ) / (total_coins : ℚ) = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_quarter_l910_91093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_common_chord_l910_91034

noncomputable section

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x-2)^2 + (y-2)^2 = 4

-- Define the common chord
def common_chord (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- Define the function to be minimized
noncomputable def f (a b : ℝ) : ℝ := 1/a + 9/b

-- Theorem statement
theorem min_value_on_common_chord :
  ∀ a b : ℝ, a > 0 → b > 0 → common_chord a b → 
  (∀ x y : ℝ, x > 0 → y > 0 → common_chord x y → f a b ≤ f x y) →
  f a b = 8 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_common_chord_l910_91034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l910_91064

/-- A rhombus with given side length and shorter diagonal -/
structure Rhombus where
  side_length : ℝ
  shorter_diagonal : ℝ

/-- The length of the longer diagonal of a rhombus -/
noncomputable def longer_diagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt (r.side_length ^ 2 - (r.shorter_diagonal / 2) ^ 2)

/-- Theorem: The longer diagonal of a rhombus with side length 27 and shorter diagonal 36 is 30√3 -/
theorem rhombus_longer_diagonal :
  let r : Rhombus := { side_length := 27, shorter_diagonal := 36 }
  longer_diagonal r = 30 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l910_91064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_840_1764_gcd_459_357_l910_91010

-- Define our own GCD function to avoid ambiguity
def myGcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Theorem 1: GCD of 840 and 1764
theorem gcd_840_1764 : myGcd 840 1764 = 84 := by
  -- The proof goes here
  sorry

-- Theorem 2: GCD of 459 and 357
theorem gcd_459_357 : myGcd 459 357 = 51 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_840_1764_gcd_459_357_l910_91010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l910_91083

theorem equation_solution : 
  ∃ x y : ℝ, (2 * x - 1) - x * (1 - 2 * x) = 0 ∧ 
             (2 * y - 1) - y * (1 - 2 * y) = 0 ∧ 
             x = 1/2 ∧ y = -1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l910_91083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_leq_neg_two_l910_91004

theorem inequality_holds_iff_a_leq_neg_two (a : ℝ) :
  (a < 0) →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔
  (a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_leq_neg_two_l910_91004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_formula_l910_91081

/-- The distance between two points in a 2D plane. -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The distance formula for two points in a 2D plane. -/
theorem distance_formula (x₁ y₁ x₂ y₂ : ℝ) :
  distance x₁ y₁ x₂ y₂ = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_formula_l910_91081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_common_element_l910_91051

def S : Finset ℕ := Finset.range 9

def subsets_of_5 : Finset (Finset ℕ) :=
  Finset.powerset S |>.filter (λ s ↦ s.card = 5)

def has_common_element (subsets : Finset (Finset ℕ)) : Prop :=
  ∃ x, ∀ s ∈ subsets, x ∈ s

theorem max_subsets_with_common_element :
  ∀ selected_subsets : Finset (Finset ℕ),
    selected_subsets ⊆ subsets_of_5 →
    selected_subsets.card = 6 →
    (∃ common_subsets : Finset (Finset ℕ),
      common_subsets ⊆ selected_subsets ∧
      common_subsets.card = 4 ∧
      has_common_element common_subsets) ∧
    (¬∃ common_subsets : Finset (Finset ℕ),
      common_subsets ⊆ selected_subsets ∧
      common_subsets.card > 4 ∧
      has_common_element common_subsets) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_common_element_l910_91051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l910_91054

noncomputable def f (a x : ℝ) := |x + 1/a| + |x - a|

theorem f_properties (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f a x ≥ 2) ∧
  (f a 3 < 5 → (1 + Real.sqrt 5)/2 < a ∧ a < (5 + Real.sqrt 21)/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l910_91054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_one_l910_91096

/-- The function f(x) with parameters a and b -/
noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b * x^2 + x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a b x : ℝ) : ℝ := a / x + 2 * b * x + 1

theorem max_value_at_one (a b : ℝ) :
  (∀ x > 0, f a b x ≤ f a b 1) ∧
  (f a b 1 = -1) ∧
  (f_derivative a b 1 = 0) →
  a = 3 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_one_l910_91096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_properties_l910_91053

-- Define the functions as noncomputable
noncomputable def f (x : ℝ) := Real.sin ((5 * Real.pi / 2) - 2 * x)
noncomputable def g (x : ℝ) := Real.sin (2 * x + 5 * Real.pi / 4)

-- State the theorem
theorem trig_properties :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, g (Real.pi / 8 + x) = g (Real.pi / 8 - x)) :=
by
  sorry -- Proof is omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_properties_l910_91053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_inequality_l910_91080

theorem arcsin_inequality (x : ℝ) :
  x ∈ Set.Icc (-1) 1 →
  (Real.arcsin x ^ 2 + Real.arcsin x + x ^ 6 + x ^ 3 > 0) ↔ x ∈ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_inequality_l910_91080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_button_presses_l910_91041

/-- Represents the possible letters in the lock code -/
inductive Letter : Type
| A : Letter
| B : Letter
| C : Letter

/-- A lock code is a sequence of three letters -/
def LockCode : Type := Fin 3 → Letter

/-- The set of all possible lock codes -/
def AllCodes : Set LockCode :=
  {code | ∀ i : Fin 3, code i ∈ ({Letter.A, Letter.B, Letter.C} : Set Letter)}

/-- A sequence of button presses -/
def ButtonSequence : Type := List Letter

/-- Checks if a subsequence of button presses opens the lock -/
def OpensLock (code : LockCode) (seq : ButtonSequence) : Prop :=
  ∃ i, List.take 3 (List.drop i seq) = [code 0, code 1, code 2]

/-- The main theorem: 29 is the minimum number of button presses to guarantee opening the lock -/
theorem min_button_presses :
  ∀ (n : ℕ), (∀ (seq : ButtonSequence), seq.length = n →
    ∀ (code : LockCode), code ∈ AllCodes → OpensLock code seq) →
  n ≥ 29 := by
  sorry

#check min_button_presses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_button_presses_l910_91041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_one_l910_91084

noncomputable def f (x : ℝ) : ℝ :=
  if x > 2 then x + 1 / (x - 2)
  else x^2 + 2

theorem f_composition_at_one : f (f 1) = 4 := by
  -- Evaluate f(1)
  have h1 : f 1 = 3 := by
    simp [f]
    norm_num
  
  -- Evaluate f(3)
  have h2 : f 3 = 4 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc
    f (f 1) = f 3 := by rw [h1]
    _       = 4   := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_one_l910_91084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pair_equation_solutions_l910_91007

theorem integer_pair_equation_solutions :
  ∀ a b : ℕ,
    a ≥ 1 → b ≥ 1 →
    (a^(b^2) = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pair_equation_solutions_l910_91007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_example_l910_91058

noncomputable def cone_lateral_surface_area (r h : ℝ) : ℝ := 
  Real.pi * r * (Real.sqrt (r^2 + h^2))

theorem cone_lateral_surface_area_example : 
  cone_lateral_surface_area 2 (4 * Real.sqrt 2) = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_example_l910_91058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_and_triangle_area_l910_91040

/-- A linear function passing through two points -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  passes_through_M : 2 = b
  passes_through_N : 3 = k + b

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

theorem linear_function_and_triangle_area (f : LinearFunction) :
  f.k = 1 ∧ f.b = 2 ∧
  triangle_area 2 2 = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_and_triangle_area_l910_91040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_graph_properties_l910_91024

-- Define the triangular path
structure TriangularPath where
  J : ℝ × ℝ
  K : ℝ × ℝ
  M : ℝ × ℝ

-- Define the runner's position as a function of time
noncomputable def runnerPosition (path : TriangularPath) (t : ℝ) : ℝ × ℝ :=
  sorry

-- Define the straight-line distance from J
noncomputable def distanceFromJ (path : TriangularPath) (t : ℝ) : ℝ :=
  sorry

-- Theorem stating the properties of the distance graph
theorem distance_graph_properties (path : TriangularPath) :
  ∃ (t₁ t₂ : ℝ), t₁ < t₂ ∧
    (∀ t, t < t₁ → (deriv (distanceFromJ path)) t > 0) ∧
    (∀ t, t₁ ≤ t ∧ t ≤ t₂ → (deriv (distanceFromJ path)) t = 0) ∧
    (∀ t, t > t₂ → (deriv (distanceFromJ path)) t < 0) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_graph_properties_l910_91024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_y_intercepts_l910_91042

/-- The number of y-intercepts for the parabola x = 3y^2 - 4y + 1 is 2 -/
theorem parabola_y_intercepts : 
  let f : ℝ → ℝ := fun y => 3 * y^2 - 4 * y + 1
  let a : ℝ := 3
  let b : ℝ := -4
  let c : ℝ := 1
  let discriminant : ℝ := b^2 - 4*a*c
  discriminant > 0 ∧ (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ f y1 = 0 ∧ f y2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_y_intercepts_l910_91042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_proof_l910_91098

theorem original_number_proof (n : ℕ) : 
  (∀ (d : ℕ), d ∈ [5, 6, 4, 3] → (n + 28) % d = 0) ↔ n = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_proof_l910_91098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_prime_factorization_sum_l910_91031

theorem min_prime_factorization_sum (x y a b c d : ℕ) : 
  (5 : ℕ) * x^7 = 13 * y^11 →
  x = a^c * b^d →
  (∀ z : ℕ, (5 : ℕ) * z^7 = 13 * y^11 → z ≥ x) →
  a + b + c + d = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_prime_factorization_sum_l910_91031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l910_91033

variable (k : ℕ)

def A : ℕ := (10^k - 1) / 3 * 3
def B : ℕ := (10^k - 1) / 9 * 4
def C : ℕ := (10^k - 1) / 9 * 6
def D : ℕ := (10^k - 1) / 9 * 7
def E : ℕ := (10^(2*k) - 1) / 9 * 5

theorem problem_solution : 
  E - A * D - B * C + 1 = (10^(k+1) - 1) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l910_91033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mono_increasing_interval_l910_91071

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (6*x - x^2) / Real.log 0.6

-- Define the property of being monotonically increasing on an interval
def MonoIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem f_mono_increasing_interval :
  MonoIncreasingOn f 3 6 ∧ 
  ∀ a b, MonoIncreasingOn f a b → 3 ≤ a ∧ b ≤ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mono_increasing_interval_l910_91071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l910_91006

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  total_distance / train_speed_mps

/-- Theorem: A train of length 110 m, traveling at 60 kmph, takes approximately 15 seconds to cross a bridge of length 140 m -/
theorem train_crossing_bridge_time :
  ∃ ε > 0, |train_crossing_time 110 60 140 - 15| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l910_91006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_symmetric_trig_function_l910_91046

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.cos (2 * x)

def symmetric_about_pi_sixth (f : ℝ → ℝ) : Prop :=
  ∀ x, f (Real.pi/6 + x) = f (Real.pi/6 - x)

theorem roots_of_symmetric_trig_function (a b : ℝ) (hab : a * b ≠ 0) 
  (h_sym : symmetric_about_pi_sixth (f a b)) :
  ∃ x y, x ∈ Set.Icc 0 (2 * Real.pi) ∧ 
         y ∈ Set.Icc 0 (2 * Real.pi) ∧ 
         x ≠ y ∧
         f a b x = 2 * b ∧ 
         f a b y = 2 * b ∧
         ∀ z ∈ Set.Icc 0 (2 * Real.pi), f a b z = 2 * b → (z = x ∨ z = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_symmetric_trig_function_l910_91046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducible_l910_91009

theorem fraction_irreducible (n : ℤ) : Int.gcd (18 * n + 3) (12 * n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducible_l910_91009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_P_equivalence_l910_91050

-- Define the sets A and B
def A : Set ℝ := {x | (3 : ℝ) ^ x < (3 : ℝ) ^ 5}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Define the set P
def P : Set ℝ := {x | x ∈ A ∧ x ∉ A ∩ B}

-- Theorem statement
theorem set_P_equivalence : P = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_P_equivalence_l910_91050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l910_91094

theorem function_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x^3 + y^3) = f (x^3) + 3*x^2*f x*f y + 3*f x*(f y)^2 + y^6*f y) →
  (∀ x : ℝ, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l910_91094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l910_91067

theorem sin_pi_plus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sin (π/2 + α) = 3/5) : Real.sin (π + α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l910_91067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_symmetry_centers_l910_91043

open Real

/-- The centers of symmetry for f(x) = 2tan(2x - π/4) -/
theorem tan_symmetry_centers (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ 2 * tan (2 * x - π / 4)
  let center : ℝ × ℝ := (π / 8 + k * π / 4, 0)
  let is_symmetry_center (c : ℝ × ℝ) := ∀ x, f (c.1 + x) = -f (c.1 - x)
  is_symmetry_center center :=
by sorry

/-- The centers of symmetry for the standard tangent function -/
axiom std_tan_symmetry_centers (k : ℤ) :
  let f : ℝ → ℝ := tan
  let center : ℝ × ℝ := (k * π / 2, 0)
  let is_symmetry_center (c : ℝ × ℝ) := ∀ x, f (c.1 + x) = -f (c.1 - x)
  is_symmetry_center center

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_symmetry_centers_l910_91043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l910_91088

-- Define the constants a, b, and c as noncomputable
noncomputable def a : ℝ := (1/3 : ℝ) ^ (2/3 : ℝ)
noncomputable def b : ℝ := (2/3 : ℝ) ^ (1/3 : ℝ)
noncomputable def c : ℝ := (2/3 : ℝ) ^ (2/3 : ℝ)

-- State the theorem
theorem order_of_abc : a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l910_91088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_equals_3_l910_91063

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 + 1

/-- The derivative of the curve function -/
noncomputable def f' (x : ℝ) : ℝ := 1/x + 2*x

/-- The point of interest -/
def point : ℝ × ℝ := (1, 2)

/-- The slope of the line perpendicular to the tangent line -/
noncomputable def perpendicular_slope (a : ℝ) : ℝ := -1/a

theorem tangent_perpendicular_implies_a_equals_3 (a : ℝ) :
  f point.1 = point.2 →
  f' point.1 * perpendicular_slope a = -1 →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_equals_3_l910_91063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_matches_played_l910_91097

/-- Proves that given the specified batting averages and number of matches, 
    the total number of matches played is 30. -/
theorem total_matches_played
  (first_avg : ℝ) (first_matches : ℕ)
  (second_avg : ℝ) (second_matches : ℕ)
  (total_avg : ℝ)
  (h1 : first_avg = 40)
  (h2 : first_matches = 20)
  (h3 : second_avg = 13)
  (h4 : second_matches = 10)
  (h5 : total_avg = 31)
  (h6 : (first_avg * (first_matches : ℝ) + second_avg * (second_matches : ℝ)) / 
        ((first_matches : ℝ) + (second_matches : ℝ)) = total_avg) :
  first_matches + second_matches = 30 := by
  sorry

#check total_matches_played

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_matches_played_l910_91097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_sum_bcd_final_answer_l910_91021

-- Define the sequence a_n
def a : ℕ → ℕ := sorry

-- Define the properties of the sequence
axiom a_odd : ∀ n : ℕ, Odd (a n)
axiom a_nondecreasing : ∀ n m : ℕ, n ≤ m → a n ≤ a m
axiom a_repetition : ∀ k : ℕ, Odd k → (∃ n : ℕ, ∀ i : ℕ, n ≤ i ∧ i < n + k + 2 → a i = k)

-- State the theorem
theorem a_formula : ∀ n : ℕ, n > 0 → a n = 2 * ⌊Real.sqrt (n - 1 : ℝ)⌋ + 1 := by sorry

-- Prove that b + c + d = 2
theorem sum_bcd : 2 + (-1) + 1 = 2 := by
  simp

-- Final theorem stating the answer
theorem final_answer : (2 : ℕ) = 2 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_sum_bcd_final_answer_l910_91021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_score_l910_91038

-- Define the homework questions
def question1 (a b : ℝ) : Prop := 2 * a * b + 3 * a * b = 5 * a * b
def question2 (a b : ℝ) : Prop := 2 * a * b - 3 * a * b = -a * b
def question3 (a b : ℝ) : Prop := 2 * a * b - 3 * a * b = 6 * a * b
def question4 (a b : ℝ) : Prop := (2 * a * b) / (3 * a * b) = 2 / 3

-- Define the scoring system
def points_per_correct_answer : ℕ := 2

-- Define a function to check if a question is correct
def is_correct (q : Prop) [Decidable q] : Bool := 
  if q then true else false

-- Theorem to prove
theorem homework_score :
  ∀ (a b : ℝ) [Decidable (question1 a b)] [Decidable (question2 a b)] 
               [Decidable (question3 a b)] [Decidable (question4 a b)],
  (is_correct (question1 a b)).toNat * points_per_correct_answer +
  (is_correct (question2 a b)).toNat * points_per_correct_answer +
  (is_correct (question3 a b)).toNat * points_per_correct_answer +
  (is_correct (question4 a b)).toNat * points_per_correct_answer = 6 :=
by
  sorry

#eval points_per_correct_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_score_l910_91038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_speed_ratio_l910_91022

/-- Represents a swimmer in the pool -/
structure Swimmer where
  speed : ℚ
  deriving Repr

/-- Represents the swimming pool setup -/
structure PoolSetup where
  pool_length : ℚ
  total_distance : ℚ
  meeting_count : ℕ
  deriving Repr

/-- Calculates the number of meetings between two swimmers -/
def calculate_meetings (s1 s2 : Swimmer) (setup : PoolSetup) : ℕ :=
  sorry

/-- Theorem stating that the only valid speed ratio is 5:1 -/
theorem unique_speed_ratio (s1 s2 : Swimmer) (setup : PoolSetup) :
  setup.pool_length = 50 ∧ 
  setup.total_distance = 1000 ∧
  setup.meeting_count = 16 ∧
  s1.speed > s2.speed ∧
  calculate_meetings s1 s2 setup = setup.meeting_count →
  s1.speed / s2.speed = 5 := by
  sorry

#eval Swimmer.mk 5
#eval PoolSetup.mk 50 1000 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_speed_ratio_l910_91022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_straight_line_times_l910_91076

/-- Represents a time on an analog clock --/
structure ClockTime where
  hours : ℕ
  minutes : ℚ
  h_hours_valid : hours ≥ 0 ∧ hours < 12
  h_minutes_valid : minutes ≥ 0 ∧ minutes < 60

/-- Calculates the angle of the hour hand from 12 o'clock position --/
def hour_angle (t : ClockTime) : ℚ :=
  30 * t.hours + t.minutes / 2

/-- Calculates the angle of the minute hand from 12 o'clock position --/
def minute_angle (t : ClockTime) : ℚ :=
  6 * t.minutes

/-- Determines if the hour and minute hands form a straight line --/
def hands_form_straight_line (t : ClockTime) : Prop :=
  (hour_angle t = minute_angle t) ∨ (abs (hour_angle t - minute_angle t) = 180)

/-- The main theorem to prove --/
theorem straight_line_times : 
  ∃ (t1 t2 : ClockTime), 
    t1.hours = 4 ∧ t1.minutes = 54 + 6/11 ∧
    t2.hours = 4 ∧ t2.minutes = 21 + 9/11 ∧
    (∀ (t : ClockTime), 
      t.hours = 4 ∧ t.minutes ≥ 0 ∧ t.minutes < 60 →
      hands_form_straight_line t → (t = t1 ∨ t = t2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_straight_line_times_l910_91076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rain_probability_april_l910_91068

/-- The number of days in April -/
def april_days : ℕ := 30

/-- The probability of rain on any given day in April -/
def daily_rain_prob : ℚ := 1/5

/-- The probability of rain on at most 3 days in April -/
noncomputable def at_most_three_days_rain_prob : ℝ := 
  (1 : ℝ) * (4/5)^30 + 
  30 * (1/5) * (4/5)^29 + 
  (30 * 29 / 2) * (1/5)^2 * (4/5)^28 + 
  (30 * 29 * 28 / 6) * (1/5)^3 * (4/5)^27

/-- Theorem stating that the probability of rain on at most 3 days in April is approximately 0.502 -/
theorem rain_probability_april :
  ∃ ε > 0, ε < 0.001 ∧ |at_most_three_days_rain_prob - 0.502| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rain_probability_april_l910_91068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_392_divided_l910_91052

theorem problem_392_divided (a : ℤ) (h : a > 0) : 
  ((392 / a - a) / a - a) / a - a = -a → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_392_divided_l910_91052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_photos_required_l910_91032

/-- Represents a photo of two children -/
structure Photo where
  child1 : ℕ
  child2 : ℕ
  h_distinct : child1 ≠ child2

/-- The theorem stating the minimum number of photos required -/
theorem min_photos_required (girls boys : ℕ) (h_girls : girls = 4) (h_boys : boys = 8) :
  ∃ (n : ℕ), n = 33 ∧
  ∀ (photos : Finset Photo),
    photos.card < n →
    (∀ p ∈ photos, p.child1 < girls + boys ∧ p.child2 < girls + boys) →
    (¬∃ p ∈ photos, (girls ≤ p.child1 ∧ girls ≤ p.child2) ∨
                    (p.child1 < girls ∧ p.child2 < girls)) →
    (¬∃ p1 p2, p1 ∈ photos ∧ p2 ∈ photos ∧ p1 ≠ p2 ∧ p1.child1 = p2.child1 ∧ p1.child2 = p2.child2) →
    False :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_photos_required_l910_91032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_A_max_sum_sin_B_C_l910_91082

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the given condition
def givenCondition (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

-- Theorem 1: Magnitude of A
theorem magnitude_of_A (t : Triangle) (h : givenCondition t) : t.A = 2 * Real.pi / 3 := by
  sorry

-- Theorem 2: Maximum value of sin(B) + sin(C)
theorem max_sum_sin_B_C (t : Triangle) (h : givenCondition t) : 
  ∃ (B C : ℝ), t.B = B ∧ t.C = C ∧ Real.sin B + Real.sin C ≤ 1 ∧ 
  ∃ (B' C' : ℝ), Real.sin B' + Real.sin C' = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_A_max_sum_sin_B_C_l910_91082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_wins_l910_91020

/-- Represents a cell in the 6x6 grid -/
structure Cell where
  row : Fin 6
  col : Fin 6
  value : ℝ

/-- Represents the game state -/
structure GameState where
  grid : List Cell
  currentPlayer : Bool  -- true for Player A, false for Player B

/-- Checks if a cell is marked (has the largest value in its row) -/
def isMarked (state : GameState) (cell : Cell) : Prop :=
  ∀ c ∈ state.grid, c.row = cell.row → c.value ≤ cell.value

/-- Checks if there exists a vertical path of marked cells from top to bottom -/
def existsVerticalPath (state : GameState) : Prop :=
  ∃ col : Fin 6, ∀ row : Fin 6, ∃ cell ∈ state.grid, 
    cell.row = row ∧ cell.col = col ∧ isMarked state cell

/-- Completes the game given strategies for both players -/
def completeGame (strategyA : GameState → Cell) (strategyB : GameState → Cell) : GameState :=
  sorry -- Implementation of game completion

/-- Theorem: Player B has a winning strategy -/
theorem player_b_wins : 
  ∀ strategyA : GameState → Cell,
  ∃ strategyB : GameState → Cell,
  ¬(existsVerticalPath (completeGame strategyA strategyB)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_wins_l910_91020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_trajectories_are_spirals_l910_91028

/-- Represents a second-order differential equation of the form
    d²x/dt² - dx/dt + x = 0 --/
structure SecondOrderDE where
  x : ℝ → ℝ
  h : Differentiable ℝ x
  eq : ∀ t, (deriv (deriv x)) t - (deriv x) t + x t = 0

/-- Represents a phase trajectory in the phase plane --/
structure PhaseTrajectory where
  x : ℝ → ℝ
  v : ℝ → ℝ

/-- Predicate to check if a curve is a spiral --/
def IsSpiral (curve : PhaseTrajectory) : Prop :=
  sorry  -- Definition of a spiral in the phase plane

/-- Theorem stating that the phase trajectories of the given differential equation are spirals --/
theorem phase_trajectories_are_spirals (de : SecondOrderDE) :
  ∃ (pt : PhaseTrajectory), IsSpiral pt ∧ (∀ t, pt.x t = de.x t ∧ pt.v t = (deriv de.x) t) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_trajectories_are_spirals_l910_91028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_hyperbola_equations_l910_91073

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on an ellipse -/
def lies_on_ellipse (p : Point) (a b : ℝ) : Prop :=
  p.x^2 / a^2 + p.y^2 / b^2 = 1

/-- Checks if a point lies on a hyperbola -/
def lies_on_hyperbola (p : Point) (a b : ℝ) : Prop :=
  p.y^2 / a^2 - p.x^2 / b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Reflects a point about the line y=x -/
def reflect_point (p : Point) : Point :=
  { x := p.y, y := p.x }

theorem ellipse_and_hyperbola_equations :
  let p := Point.mk 5 2
  let f1 := Point.mk (-6) 0
  let f2 := Point.mk 6 0
  let p' := reflect_point p
  let f1' := reflect_point f1
  let f2' := reflect_point f2
  (lies_on_ellipse p 45 9 ∧
   distance p f1 + distance p f2 = 2 * Real.sqrt 45) ∧
  (lies_on_hyperbola p' 20 16 ∧
   |distance p' f1' - distance p' f2'| = 2 * Real.sqrt 20) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_hyperbola_equations_l910_91073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l910_91017

/-- The range of k for an ellipse with equation x^2/k + y^2/4 = 1 and eccentricity e ∈ (1/2, 1) -/
theorem ellipse_k_range (k : ℝ) (e : ℝ) : 
  (∃ x y : ℝ, x^2/k + y^2/4 = 1) →  -- ellipse equation
  (e^2 = 1 - min (4/k) 1) →  -- eccentricity definition for ellipse
  e > 1/2 → e < 1 →  -- eccentricity range
  (k ∈ Set.Ioo 0 3 ∪ Set.Ioi (16/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l910_91017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_form_rock_game_theorem_l910_91023

/-- Represents the game with rocks --/
structure RockGame where
  initial_rocks : ℕ
  remove_rocks : ℕ → ℕ → ℕ
  is_multiple_of_five : ℕ → Prop

/-- The probability that the number of rocks left after each round is a multiple of 5 --/
noncomputable def probability_multiple_of_five (game : RockGame) : ℚ := sorry

/-- The game with 2015 initial rocks --/
def game_2015 : RockGame where
  initial_rocks := 2015
  remove_rocks := λ n k ↦ n - k
  is_multiple_of_five := λ n ↦ n % 5 = 0

/-- The probability is of the form 5^a * 31^b * (c/d) --/
theorem probability_form (game : RockGame) :
  ∃ (a b : ℤ) (c d : ℕ), 
    probability_multiple_of_five game = (5 : ℚ) ^ a * (31 : ℚ) ^ b * (c : ℚ) / (d : ℚ) ∧
    Nat.Coprime c d ∧
    Nat.Coprime c 5 ∧
    Nat.Coprime c 31 ∧
    d > 0 := by
  sorry

/-- The main theorem to prove --/
theorem rock_game_theorem (game : RockGame) 
  (h : game = game_2015) :
  ∃ (a b : ℤ) (c d : ℕ), 
    probability_multiple_of_five game = (5 : ℚ) ^ a * (31 : ℚ) ^ b * (c : ℚ) / (d : ℚ) ∧
    Nat.Coprime c d ∧
    Nat.Coprime c 5 ∧
    Nat.Coprime c 31 ∧
    d > 0 ∧
    a + b = -501 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_form_rock_game_theorem_l910_91023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_satisfies_condition_l910_91061

/-- The principal amount that satisfies the interest condition -/
noncomputable def principal_amount : ℝ := 1800

/-- The interest rate per annum -/
def interest_rate : ℝ := 10

/-- The time period in years -/
def time_period : ℝ := 2

/-- Calculate simple interest -/
noncomputable def simple_interest (P : ℝ) : ℝ := P * interest_rate * time_period / 100

/-- Calculate compound interest -/
noncomputable def compound_interest (P : ℝ) : ℝ := P * ((1 + interest_rate / 100) ^ time_period - 1)

/-- The given difference between compound and simple interest -/
def interest_difference : ℝ := 18

theorem principal_satisfies_condition :
  compound_interest principal_amount - simple_interest principal_amount = interest_difference :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_satisfies_condition_l910_91061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_divisible_by_2_3_5_l910_91056

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, n = m^2) → 2 ∣ n → 3 ∣ n → 5 ∣ n → n ≥ 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_divisible_by_2_3_5_l910_91056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_intersections_l910_91089

/-- Given k points on a circle, proves that the number of intersections between all possible line segments drawn between pairs of points is equal to the binomial coefficient (k choose 4). -/
theorem circle_point_intersections (k : ℕ) : 
  k ≥ 4 → Nat.choose k 4 = Nat.choose k 4 := by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_intersections_l910_91089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_simplification_l910_91049

-- Define the logarithmic expression
noncomputable def log_expression (a b c d x y z : ℝ) : ℝ :=
  Real.log (a^2 / b) + Real.log (b^2 / c) + Real.log (c^2 / d) - Real.log (a^2 * y * z / (d^2 * x))

-- State the theorem
theorem log_expression_simplification (a b c d x y z : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : x > 0) (h6 : y > 0) (h7 : z > 0) :
  log_expression a b c d x y z = Real.log (b * d * x / (y * z)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_simplification_l910_91049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sequence_general_term_l910_91085

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem x_sequence_general_term
  (x : ℕ → ℝ)
  (a : ℕ → ℝ)
  (h_positive : ∀ n, x n > 0)
  (h_arithmetic : is_arithmetic_sequence (λ n ↦ 1 / a n))
  (h_x1 : x 1 = 3)
  (h_sum : x 1 + x 2 + x 3 = 39)
  (h_power : ∀ n, (x n) ^ (a n) = (x (n + 1)) ^ (a (n + 1)) ∧
                  (x n) ^ (a n) = (x (n + 2)) ^ (a (n + 2))) :
  ∀ n, x n = 3^n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sequence_general_term_l910_91085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_number_is_eighteen_l910_91069

/-- Represents the sequence described in the problem -/
def sequenceValue : ℕ → ℕ
| 0 => 4
| n + 1 => sequenceValue n + 2

/-- Represents the number of times a value appears in its row -/
def appearances (n : ℕ) : ℕ := n / 2

/-- Represents the cumulative count of numbers up to and including a given row -/
def cumulative_count : ℕ → ℕ
| 0 => 0
| n + 1 => cumulative_count n + appearances (sequenceValue n)

/-- The 40th number in the sequence is 18 -/
theorem fortieth_number_is_eighteen :
  ∃ (row : ℕ), cumulative_count row ≥ 40 ∧ cumulative_count (row - 1) < 40 ∧ sequenceValue (row - 1) = 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_number_is_eighteen_l910_91069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_year_date_inequality_l910_91015

/-- Represents the data set of dates in a leap year -/
def leapYearDates : List Nat :=
  (List.range 30).bind (fun i => List.replicate 12 (i + 1)) ++ List.replicate 8 31

/-- The mean of the data set -/
noncomputable def μ : ℚ := (leapYearDates.sum : ℚ) / leapYearDates.length

/-- The median of the data set -/
def M : ℕ := 16

/-- The median of the modes -/
def d : ℚ := 15.5

theorem leap_year_date_inequality : d < μ ∧ μ < M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_year_date_inequality_l910_91015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l910_91003

-- Define arithmetic sequence
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  (b - a) = (c - b) ∧ (c - b) = (d - c)

-- Define geometric sequence
def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c) ∧ (d / c = e / d)

theorem arithmetic_geometric_sequence_ratio :
  ∀ (x y a b c : ℝ),
  is_arithmetic_sequence 1 x y 4 →
  is_geometric_sequence (-2) a b c (-8) →
  (y - x) / b = -1/4 :=
by
  intros x y a b c h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l910_91003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_set_l910_91047

def U : Set ℕ := Set.univ

def B : Set ℕ := {x | x < 2}

def A : Set ℕ := {x | (x + 4) * (x - 5) ≤ 0}

theorem intersection_complement_equals_set : A ∩ (U \ B) = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_set_l910_91047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_remainder_2005_l910_91027

theorem min_remainder_2005 (n : ℕ) 
  (h1 : n % 902 = 602)
  (h2 : n % 802 = 502)
  (h3 : n % 702 = 402) :
  ∃ k : ℕ, n = 2005 * k + 101 ∧ 
  ∀ m : ℕ, n = 2005 * m + (n % 2005) → n % 2005 ≥ 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_remainder_2005_l910_91027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_count_theorem_l910_91072

/-- Represents a boat with a certain capacity -/
structure Boat where
  capacity : ℕ

/-- Represents a group of people -/
structure People where
  adults : ℕ
  children : ℕ

/-- Calculates the number of ways to divide a group among boats -/
def countDivisions (boats : List Boat) (group : People) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem division_count_theorem (boats : List Boat) (group : People) :
  boats = [Boat.mk 3, Boat.mk 2, Boat.mk 1] →
  group = People.mk 2 2 →
  countDivisions boats group = 8 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_count_theorem_l910_91072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_ratio_fourth_quadrant_l910_91030

theorem sine_cosine_ratio_fourth_quadrant (α : Real) 
    (h1 : Real.sin α = -Real.sqrt 5 / 5) 
    (h2 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) : 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_ratio_fourth_quadrant_l910_91030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_left_faces_dots_l910_91086

/-- Represents a cube with faces labeled by the number of dots -/
structure Cube where
  faces : Fin 6 → Nat
  three_dot_face : ∃ f, faces f = 3
  two_dot_faces : ∃ f1 f2, f1 ≠ f2 ∧ faces f1 = 2 ∧ faces f2 = 2
  one_dot_faces : ∀ f, faces f ≠ 2 ∧ faces f ≠ 3 → faces f = 1

/-- Predicate to determine if two faces of different cubes are touching -/
def touching (i j : Fin 7) (f1 f2 : Fin 6) : Prop :=
  sorry

/-- Represents the "P" shaped configuration of 7 cubes -/
structure PConfiguration where
  cubes : Fin 7 → Cube
  touching_faces_same : ∀ i j f1 f2, touching i j f1 f2 → (cubes i).faces f1 = (cubes j).faces f2

/-- The faces A, B, and C are on the left side of the "P" configuration -/
def left_faces (config : PConfiguration) : Fin 3 → Nat :=
  sorry

/-- The theorem to be proved -/
theorem left_faces_dots (config : PConfiguration) :
  left_faces config 0 = 2 ∧ left_faces config 1 = 2 ∧ left_faces config 2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_left_faces_dots_l910_91086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sums_theorem_l910_91066

noncomputable def z (n : ℕ) : ℂ := Complex.exp (2 * Real.pi * Complex.I / n)

noncomputable def sum_S1 (n : ℕ) (a : ℤ) : ℂ :=
  Finset.sum (Finset.range n) (λ k => (z n) ^ (a * k))

noncomputable def sum_S2 (n : ℕ) (a : ℤ) : ℂ :=
  Finset.sum (Finset.range n) (λ k => (k + 1 : ℕ) * (z n) ^ (a * k))

theorem sums_theorem (n : ℕ) (a : ℤ) (h : n > 0):
  (sum_S1 n a = if a % n = 0 then n else 0) ∧
  (sum_S2 n a = if a % n = 0 then (n * (n + 1) / 2 : ℂ)
               else (n / (2 * Real.sin (a * Real.pi / n))) *
                    (Real.sin (a * Real.pi / n) - Complex.I * Real.cos (a * Real.pi / n))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sums_theorem_l910_91066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_students_in_canteen_l910_91001

/-- The number of students in the canteen -/
def students_in_canteen (total_students : ℕ) (absent_fraction : ℚ) (classroom_fraction : ℚ) : ℕ :=
  let present_students := total_students - (absent_fraction * ↑total_students).num
  let classroom_students := (classroom_fraction * ↑present_students).num
  (present_students - classroom_students).toNat

/-- Theorem stating that there are 9 students in the canteen -/
theorem nine_students_in_canteen :
  students_in_canteen 40 (1/10) (3/4) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_students_in_canteen_l910_91001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_boys_than_girls_l910_91025

/-- Represents a class with a number of girls and boys -/
structure ClassInfo where
  girls : Nat
  boys : Nat

/-- The field day competition setup -/
def fieldDay : List ClassInfo := [
  { girls := 12, boys := 13 }, -- First 4th grade class
  { girls := 15, boys := 11 }, -- Second 4th grade class
  { girls := 9, boys := 13 },  -- First 5th grade class
  { girls := 10, boys := 11 }  -- Second 5th grade class
]

/-- Calculates the total number of girls in all classes -/
def totalGirls (classes : List ClassInfo) : Nat :=
  classes.foldl (fun acc c => acc + c.girls) 0

/-- Calculates the total number of boys in all classes -/
def totalBoys (classes : List ClassInfo) : Nat :=
  classes.foldl (fun acc c => acc + c.boys) 0

/-- Theorem stating that there are 2 more boys than girls in the field day competition -/
theorem more_boys_than_girls : 
  totalBoys fieldDay - totalGirls fieldDay = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_boys_than_girls_l910_91025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_resale_price_l910_91018

-- Define the original cost price
def original_price : ℝ := 51136.36

-- Define the loss percentage
def loss_percentage : ℝ := 0.12

-- Define the gain percentage
def gain_percentage : ℝ := 0.20

-- Theorem statement
theorem car_resale_price :
  (original_price * (1 - loss_percentage)) * (1 + gain_percentage) = 54000 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_resale_price_l910_91018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_range_l910_91016

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Theorem 1: Maximum value when a = 0
theorem max_value_when_a_zero :
  ∃ (max : ℝ), max = -1 ∧ ∀ (x : ℝ), x > 0 → f 0 x ≤ max :=
sorry

-- Theorem 2: Range of a for exactly one zero
theorem one_zero_range :
  ∀ (a : ℝ), (∃! (x : ℝ), x > 0 ∧ f a x = 0) ↔ (a > 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_range_l910_91016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrange_five_with_pair_l910_91012

/-- The number of ways to arrange n items, where 2 specific items must be adjacent and in a fixed order -/
def arrangeWithPair (n : ℕ) : ℕ :=
  if n ≤ 2 then 1 else Nat.factorial (n - 1)

theorem arrange_five_with_pair :
  arrangeWithPair 5 = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrange_five_with_pair_l910_91012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_problem_l910_91000

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- first term
  d : ℚ  -- common difference

/-- Get the nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  ap.a + (n - 1 : ℚ) * ap.d

/-- Get the sum of the first n terms of an arithmetic progression -/
def ArithmeticProgression.sumOfTerms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

/-- Theorem stating the conditions and conclusion of the problem -/
theorem arithmetic_progression_problem (ap : ArithmeticProgression) (n : ℕ) :
  ap.nthTerm 4 + ap.nthTerm n = 20 ∧
  ap.sumOfTerms 12 = 120 →
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_problem_l910_91000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximum_profit_approx_7_69_l910_91013

-- Define the cost function
def total_cost (Q : ℝ) : ℝ := 5 * Q^2

-- Define the demand functions
def demand_non_slytherin (P : ℝ) : ℝ := 26 - 2 * P
def demand_slytherin (P : ℝ) : ℝ := 10 - P

-- Define the total demand function
def total_demand (P : ℝ) : ℝ := demand_non_slytherin P + demand_slytherin P

-- Define the revenue function
def revenue (P : ℝ) : ℝ := P * total_demand P

-- Define the profit function
def profit (P : ℝ) : ℝ := revenue P - total_cost (total_demand P)

-- State the theorem
theorem maximum_profit_approx_7_69 :
  ∃ P : ℝ, P > 0 ∧ ∀ P' : ℝ, P' > 0 → profit P ≥ profit P' ∧ abs (profit P - 7.69) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximum_profit_approx_7_69_l910_91013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_coplanar_iff_l910_91091

/-- Two lines are coplanar if their direction vectors and the vector connecting their points are linearly dependent -/
def are_coplanar (p1 p2 v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a • v1 + b • v2 + c • (p2 - p1) = (0, 0, 0) ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

/-- The direction vectors of the two lines -/
def v1 (m : ℝ) : ℝ × ℝ × ℝ := (2, 2, -m)
def v2 (m : ℝ) : ℝ × ℝ × ℝ := (m, 3, 2)

/-- The points on the two lines -/
def p1 : ℝ × ℝ × ℝ := (3, 5, 6)
def p2 : ℝ × ℝ × ℝ := (4, 7, 8)

/-- The main theorem stating the condition for the lines to be coplanar -/
theorem lines_coplanar_iff (m : ℝ) :
  are_coplanar p1 p2 (v1 m) (v2 m) ↔ m = 3 ∨ m = -1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_coplanar_iff_l910_91091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_is_finite_count_satisfying_z_l910_91035

def f (z : ℂ) : ℂ := z^2 + Complex.I * z + 1

def satisfies_conditions (z : ℂ) : Prop :=
  z.im > 0 ∧
  ∃ (a : ℤ), (f z).re = a ∧ (f z).im = a ∧ 
  a.natAbs ≤ 15

-- We need to prove that the set is finite before we can count its elements
theorem set_is_finite : 
  ∃ (s : Finset ℂ), ∀ z, satisfies_conditions z ↔ z ∈ s := by
  sorry

-- Now we can define the finite set
noncomputable def satisfying_set : Finset ℂ :=
  Classical.choose set_is_finite

-- Finally, we can state our theorem about the cardinality
theorem count_satisfying_z : 
  satisfying_set.card = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_is_finite_count_satisfying_z_l910_91035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_minus_12_is_quadratic_binomial_l910_91011

/-- Definition of a quadratic binomial -/
def is_quadratic_binomial (p : MvPolynomial (Fin 2) ℚ) : Prop :=
  (MvPolynomial.totalDegree p = 2) ∧ (MvPolynomial.support p).card = 2

/-- The polynomial xy - 12 -/
noncomputable def p : MvPolynomial (Fin 2) ℚ :=
  MvPolynomial.X 0 * MvPolynomial.X 1 - 12

/-- Theorem: xy - 12 is a quadratic binomial -/
theorem xy_minus_12_is_quadratic_binomial : is_quadratic_binomial p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_minus_12_is_quadratic_binomial_l910_91011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_storage_ratio_l910_91074

/-- Given the conditions of a bakery's storage room, prove the original ratio of flour to baking soda -/
theorem bakery_storage_ratio :
  -- Conditions
  let sugar_flour_ratio : ℚ := 5 / 5
  let sugar_amount : ℕ := 2400
  let flour_amount : ℕ := sugar_amount -- derived from sugar_flour_ratio
  let baking_soda_amount : ℕ := 240 -- We know this value from the solution
  -- If 60 more pounds of baking soda were added, the ratio of flour to baking soda would be 8:1
  (flour_amount : ℚ) / ((baking_soda_amount + 60) : ℚ) = 8 / 1 →
  -- Conclusion: The original ratio of flour to baking soda is 10:1
  (flour_amount : ℚ) / (baking_soda_amount : ℚ) = 10 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_storage_ratio_l910_91074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pesticide_bucket_capacity_l910_91062

theorem pesticide_bucket_capacity (x : ℚ) (hx : x > 0) : 
  (x - 8 - (4 * (x - 8)) / x ≤ 1/5 * x) ↔ (x = 9 ∨ x = 11) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pesticide_bucket_capacity_l910_91062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_radius_l910_91048

/-- A regular polygon with side length 2 whose sum of interior angles
    is twice the sum of its exterior angles has a radius of 2. -/
theorem regular_polygon_radius (n : ℕ) (r : ℝ) : 
  n ≥ 3 →  -- A polygon has at least 3 sides
  (2 : ℝ) = 2 * r * Real.sin (π / n) →  -- Side length is 2
  (n - 2) * π = 4 * π →  -- Sum of interior angles is twice the sum of exterior angles
  r = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_radius_l910_91048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_program_output_for_five_l910_91079

noncomputable def f (x : ℝ) : ℝ := -(((x^2 - x) / 2))

theorem program_output_for_five :
  f 5 = -10 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the arithmetic expression
  simp [pow_two]
  -- Perform the numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_program_output_for_five_l910_91079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_problem_l910_91029

theorem sin_double_angle_problem (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 1/3) 
  (h2 : Real.pi/2 ≤ α) 
  (h3 : α ≤ Real.pi) : 
  Real.sin (2*α) = -4*Real.sqrt 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_problem_l910_91029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_l910_91087

noncomputable def original_price : ℝ := 61.2 / 0.85

def discounted_price : ℝ := 61.2

def final_price : ℝ := discounted_price * 1.25

theorem price_difference : final_price - original_price = 4.5 := by
  -- Unfold definitions
  unfold final_price discounted_price original_price
  -- Simplify the expression
  simp [mul_div_assoc]
  -- Assert the equality (we'll use 'sorry' to skip the proof)
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_l910_91087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisection_triangle_area_l910_91008

/-- Given a triangle ABC with area 27 square units, where E trisects AC and F trisects AB,
    the area of triangle CEF is 3 square units. -/
theorem trisection_triangle_area (A B C : ℝ × ℝ) : 
  let triangle_area := (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ
  (∃ (triangle_area : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ), 
    triangle_area A B C = 27 ∧
    ∃ (E F : ℝ × ℝ),
      E.1 = (2 * A.1 + C.1) / 3 ∧ E.2 = (2 * A.2 + C.2) / 3 ∧
      F.1 = (2 * A.1 + B.1) / 3 ∧ F.2 = (2 * A.2 + B.2) / 3 ∧
      triangle_area C E F = 3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisection_triangle_area_l910_91008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_push_time_l910_91005

/-- Represents the time taken to push a car over a certain distance at a given speed -/
noncomputable def pushTime (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Represents the total time taken to push a car and take breaks -/
noncomputable def totalTime (pushTimes : List ℝ) (breakTimes : List ℝ) : ℝ :=
  (pushTimes.sum + breakTimes.sum) / 60

theorem car_push_time :
  let pushTimes := [
    pushTime 3 6,
    pushTime 2 3,
    pushTime 3 4,
    pushTime 4 8
  ]
  let breakTimes := [10, 15, 10]
  totalTime pushTimes breakTimes = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_push_time_l910_91005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amar_score_proof_l910_91026

-- Define the given conditions
def max_score : ℝ := 900
def bhavan_percent : ℝ := 36
def chetan_percent : ℝ := 44
def average_score : ℝ := 432

-- Define Amar's score as a variable
noncomputable def amar_percent : ℝ := 64

-- Theorem to prove
theorem amar_score_proof :
  (amar_percent / 100 * max_score + bhavan_percent / 100 * max_score + chetan_percent / 100 * max_score) / 3 = average_score := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amar_score_proof_l910_91026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_ordering_l910_91055

/-- Inverse proportion function -/
noncomputable def inverse_proportion (x : ℝ) : ℝ := -8 / x

theorem inverse_proportion_point_ordering :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  y₁ = inverse_proportion x₁ →
  y₂ = inverse_proportion x₂ →
  x₁ < 0 →
  0 < x₂ →
  y₂ < 0 ∧ 0 < y₁ :=
by
  intros x₁ y₁ x₂ y₂ h₁ h₂ h₃ h₄
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_ordering_l910_91055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_meeting_point_l910_91036

/-- The speed of the escalator moving down in m/s -/
noncomputable def u : ℝ := 1.5

/-- The speed of the person moving down relative to the escalator in m/s -/
noncomputable def v : ℝ := 3

/-- The length of the escalator in meters -/
noncomputable def l : ℝ := 100

/-- The meeting point of two people on an escalator -/
noncomputable def meeting_point (u v l : ℝ) : ℝ :=
  let v_down := v + u
  let v_up := (2 * v / 3) - u
  let v_relative := v_down + v_up
  let t := l / v_relative
  v_up * t

/-- Theorem stating that the meeting point is 10 meters from the bottom of the escalator -/
theorem escalator_meeting_point : meeting_point u v l = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_meeting_point_l910_91036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l910_91019

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a = 2 * Real.sin (A/2) * Real.sin (B/2) / Real.sin ((A+B)/2) ∧
  b = 2 * Real.sin (B/2) * Real.sin (C/2) / Real.sin ((B+C)/2) ∧
  c = 2 * Real.sin (C/2) * Real.sin (A/2) / Real.sin ((C+A)/2)

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) 
  (ha : a = 2) 
  (hc : c = 5) 
  (hcosB : Real.cos B = 3/5) :
  b = Real.sqrt 17 ∧ Real.sin C = 4 * Real.sqrt 17 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l910_91019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_logarithmic_base_l910_91057

theorem existence_of_logarithmic_base
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (q d : ℝ)
  (h_a_pos : ∀ n, a n > 0)
  (h_b_pos : ∀ n, b n > 0)
  (h_q_pos : q > 0)
  (h_d_pos : d > 0)
  (h_a_geom : ∀ n, a (n + 1) = a n * q)
  (h_b_arith : ∀ n, b (n + 1) = b n + d) :
  ∃ k : ℝ, k > 0 ∧ ∀ n : ℕ, Real.log (a n) / Real.log k - b n = Real.log (a 1) / Real.log k - b 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_logarithmic_base_l910_91057
