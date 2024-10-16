import Mathlib

namespace NUMINAMATH_CALUDE_five_fourths_of_x_over_three_l2001_200161

theorem five_fourths_of_x_over_three (x : ℝ) : (5 / 4) * (x / 3) = 5 * x / 12 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_x_over_three_l2001_200161


namespace NUMINAMATH_CALUDE_binary_to_octal_equivalence_l2001_200162

-- Define the binary number
def binary_num : ℕ := 11011

-- Define the octal number
def octal_num : ℕ := 33

-- Theorem stating the equivalence of the binary and octal representations
theorem binary_to_octal_equivalence :
  (binary_num.digits 2).foldl (· + 2 * ·) 0 = (octal_num.digits 8).foldl (· + 8 * ·) 0 :=
by sorry

end NUMINAMATH_CALUDE_binary_to_octal_equivalence_l2001_200162


namespace NUMINAMATH_CALUDE_floor_pi_plus_four_l2001_200121

theorem floor_pi_plus_four : ⌊Real.pi + 4⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_pi_plus_four_l2001_200121


namespace NUMINAMATH_CALUDE_benzene_required_for_reaction_l2001_200149

-- Define the molecules and their molar ratios in the reaction
structure Reaction :=
  (benzene : ℚ)
  (methane : ℚ)
  (toluene : ℚ)
  (hydrogen : ℚ)

-- Define the balanced equation
def balanced_equation : Reaction := ⟨1, 1, 1, 1⟩

-- Theorem statement
theorem benzene_required_for_reaction 
  (methane_input : ℚ) 
  (hydrogen_output : ℚ) :
  methane_input = 2 →
  hydrogen_output = 2 →
  methane_input * balanced_equation.benzene / balanced_equation.methane = 2 :=
by sorry

end NUMINAMATH_CALUDE_benzene_required_for_reaction_l2001_200149


namespace NUMINAMATH_CALUDE_circumradius_of_specific_trapezoid_l2001_200103

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  longBase : ℝ
  shortBase : ℝ
  lateralSide : ℝ

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumradius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed circle of the given isosceles trapezoid is 5√2 -/
theorem circumradius_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := ⟨14, 2, 10⟩
  circumradius t = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_of_specific_trapezoid_l2001_200103


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2001_200140

/-- Given a hyperbola with equation x²/a² - y²/4 = 1 and an asymptote y = x/2,
    prove that the equation of the hyperbola is x²/16 - y²/4 = 1 -/
theorem hyperbola_equation (a : ℝ) :
  (∃ x y, x^2 / a^2 - y^2 / 4 = 1) →
  (∃ x, x / 2 = x / 2) →
  (∃ x y, x^2 / 16 - y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2001_200140


namespace NUMINAMATH_CALUDE_chloe_age_sum_of_digits_l2001_200165

/-- Represents a person's age -/
structure Age :=
  (value : ℕ)

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- Represents the family's ages and their properties -/
structure FamilyAges :=
  (joey : Age)
  (chloe : Age)
  (max : Age)
  (joey_chloe_diff : joey.value = chloe.value + 2)
  (max_age : max.value = 2)
  (joey_multiple_of_max : ∃ k : ℕ, joey.value = k * max.value)
  (future_multiples : ∃ n₁ n₂ n₃ n₄ n₅ : ℕ, 
    (joey.value + n₁) % (max.value + n₁) = 0 ∧
    (joey.value + n₂) % (max.value + n₂) = 0 ∧
    (joey.value + n₃) % (max.value + n₃) = 0 ∧
    (joey.value + n₄) % (max.value + n₄) = 0 ∧
    (joey.value + n₅) % (max.value + n₅) = 0)

theorem chloe_age_sum_of_digits (family : FamilyAges) :
  ∃ n : ℕ, n > 0 ∧ 
    (family.chloe.value + n) % (family.max.value + n) = 0 ∧
    sumOfDigits (family.chloe.value + n) = 10 :=
  sorry

end NUMINAMATH_CALUDE_chloe_age_sum_of_digits_l2001_200165


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l2001_200132

theorem max_value_of_expression (x : ℝ) : 
  x^6 / (x^10 + 3*x^8 - 5*x^6 + 10*x^4 + 25) ≤ 1 / (5 + 2 * Real.sqrt 30) :=
sorry

theorem max_value_achievable : 
  ∃ x : ℝ, x^6 / (x^10 + 3*x^8 - 5*x^6 + 10*x^4 + 25) = 1 / (5 + 2 * Real.sqrt 30) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l2001_200132


namespace NUMINAMATH_CALUDE_completing_square_result_l2001_200191

theorem completing_square_result (x : ℝ) : x^2 + 4*x + 3 = 0 ↔ (x + 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l2001_200191


namespace NUMINAMATH_CALUDE_strawberry_to_fruit_ratio_l2001_200120

-- Define the total garden size
def garden_size : ℕ := 64

-- Define the fruit section size (half of the garden)
def fruit_section : ℕ := garden_size / 2

-- Define the strawberry section size
def strawberry_section : ℕ := 8

-- Theorem to prove the ratio of strawberry section to fruit section
theorem strawberry_to_fruit_ratio :
  (strawberry_section : ℚ) / fruit_section = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_to_fruit_ratio_l2001_200120


namespace NUMINAMATH_CALUDE_wage_payment_days_l2001_200155

/-- Given a sum of money that can pay y's wages for 45 days and both x and y's wages for 20 days,
    prove that it can pay x's wages for 36 days. -/
theorem wage_payment_days (S : ℝ) (Wx Wy : ℝ) (S_positive : S > 0) (Wx_positive : Wx > 0) (Wy_positive : Wy > 0) :
  S = 45 * Wy ∧ S = 20 * (Wx + Wy) → S = 36 * Wx := by
  sorry

#check wage_payment_days

end NUMINAMATH_CALUDE_wage_payment_days_l2001_200155


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2001_200122

/-- An isosceles triangle with two sides of length 9 and base of length 4 has perimeter 22. -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), a = 9 → b = 9 → c = 4 → a + b + c = 22 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2001_200122


namespace NUMINAMATH_CALUDE_area_ratio_triangle_circle_l2001_200168

/-- Given a right-angled isosceles triangle and a circle with the same perimeter,
    the ratio of the area of the triangle to the area of the circle is π(3 - 2√2)/2 -/
theorem area_ratio_triangle_circle (l : ℝ) (r : ℝ) (h : ℝ) :
  l > 0 → r > 0 →
  h = Real.sqrt 2 * l →  -- Pythagorean theorem for right-angled isosceles triangle
  2 * l + h = 2 * Real.pi * r →  -- Same perimeter condition
  (1 / 2 * l^2) / (Real.pi * r^2) = Real.pi * (3 - 2 * Real.sqrt 2) / 2 := by
sorry


end NUMINAMATH_CALUDE_area_ratio_triangle_circle_l2001_200168


namespace NUMINAMATH_CALUDE_intersection_point_property_l2001_200124

theorem intersection_point_property (x₀ : ℝ) (h1 : x₀ ≠ 0) (h2 : Real.tan x₀ = -x₀) :
  (x₀^2 + 1) * (1 + Real.cos (2 * x₀)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_property_l2001_200124


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l2001_200159

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem to prove
theorem complement_of_union_is_four :
  (M ∪ N)ᶜ = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l2001_200159


namespace NUMINAMATH_CALUDE_basketball_not_tabletennis_l2001_200142

theorem basketball_not_tabletennis (total : ℕ) (basketball : ℕ) (tabletennis : ℕ) (neither : ℕ)
  (h1 : total = 40)
  (h2 : basketball = 24)
  (h3 : tabletennis = 16)
  (h4 : neither = 6) :
  basketball - (basketball + tabletennis - (total - neither)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_basketball_not_tabletennis_l2001_200142


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l2001_200101

/-- A quadrilateral inscribed in a circle with given side lengths -/
structure InscribedQuadrilateral where
  -- The radius of the circumscribed circle
  radius : ℝ
  -- The lengths of the four sides of the quadrilateral
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Theorem stating that for a quadrilateral inscribed in a circle with radius 300√2
    and three sides of lengths 300, 300, and 150√2, the fourth side has length 300√2 -/
theorem inscribed_quadrilateral_fourth_side
  (q : InscribedQuadrilateral)
  (h_radius : q.radius = 300 * Real.sqrt 2)
  (h_side1 : q.side1 = 300)
  (h_side2 : q.side2 = 300)
  (h_side3 : q.side3 = 150 * Real.sqrt 2) :
  q.side4 = 300 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l2001_200101


namespace NUMINAMATH_CALUDE_f_range_l2001_200117

noncomputable def f (x : ℝ) : ℝ := 1 - 2 / (Real.log x + 1)

theorem f_range (m n : ℝ) (hm : m > Real.exp 1) (hn : n > Real.exp 1)
  (h : f m = 2 * Real.log (Real.sqrt (Real.exp 1)) - f n) :
  5/7 ≤ f (m * n) ∧ f (m * n) < 1 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l2001_200117


namespace NUMINAMATH_CALUDE_linear_equations_l2001_200141

-- Define what a linear equation is
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define the equations
def eq1 : ℝ → ℝ := λ _ => 12
def eq2 : ℝ → ℝ := λ x => 5 * x + 3
def eq3 : ℝ → ℝ → ℝ := λ x y => 2 * x + 3 * y
def eq4 : ℝ → ℝ := λ a => 2 * a - 1
def eq5 : ℝ → ℝ := λ x => 2 * x^2 + x

-- Theorem statement
theorem linear_equations :
  (¬ is_linear_equation eq1) ∧
  (is_linear_equation eq2) ∧
  (¬ is_linear_equation (λ x => eq3 x 0)) ∧
  (is_linear_equation eq4) ∧
  (¬ is_linear_equation eq5) :=
sorry

end NUMINAMATH_CALUDE_linear_equations_l2001_200141


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l2001_200192

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l2001_200192


namespace NUMINAMATH_CALUDE_race_first_part_length_l2001_200199

theorem race_first_part_length 
  (total_length : ℝ)
  (second_part : ℝ)
  (third_part : ℝ)
  (last_part : ℝ)
  (h1 : total_length = 74.5)
  (h2 : second_part = 21.5)
  (h3 : third_part = 21.5)
  (h4 : last_part = 16) :
  total_length - (second_part + third_part + last_part) = 15.5 := by
sorry

end NUMINAMATH_CALUDE_race_first_part_length_l2001_200199


namespace NUMINAMATH_CALUDE_subset_union_equality_l2001_200119

theorem subset_union_equality (n : ℕ+) (A : Fin (n + 1) → Set (Fin n)) 
  (h : ∀ i, (A i).Nonempty) :
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) := by
  sorry

end NUMINAMATH_CALUDE_subset_union_equality_l2001_200119


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l2001_200182

theorem cos_double_angle_special_case (θ : Real) :
  3 * Real.cos (π / 2 - θ) + Real.cos (π + θ) = 0 → Real.cos (2 * θ) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l2001_200182


namespace NUMINAMATH_CALUDE_max_k_inequality_l2001_200177

theorem max_k_inequality (k : ℝ) : 
  (∀ (x y : ℤ), 4 * x^2 + y^2 + 1 ≥ k * x * (y + 1)) ↔ k ≤ 3 := by sorry

end NUMINAMATH_CALUDE_max_k_inequality_l2001_200177


namespace NUMINAMATH_CALUDE_total_slices_is_136_l2001_200150

/-- The number of slices in a small pizza -/
def small_slices : ℕ := 6

/-- The number of slices in a medium pizza -/
def medium_slices : ℕ := 8

/-- The number of slices in a large pizza -/
def large_slices : ℕ := 12

/-- The total number of pizzas bought -/
def total_pizzas : ℕ := 15

/-- The number of small pizzas ordered -/
def small_pizzas : ℕ := 4

/-- The number of medium pizzas ordered -/
def medium_pizzas : ℕ := 5

/-- Theorem stating that the total number of slices is 136 -/
theorem total_slices_is_136 : 
  small_pizzas * small_slices + 
  medium_pizzas * medium_slices + 
  (total_pizzas - small_pizzas - medium_pizzas) * large_slices = 136 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_is_136_l2001_200150


namespace NUMINAMATH_CALUDE_rhombus_properties_l2001_200137

-- Define the rhombus ABCD
def Rhombus (A B C D : ℝ × ℝ) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A B = 4 ∧ dist B C = 4 ∧ dist C D = 4 ∧ dist D A = 4

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the condition for point A on the semicircle
def OnSemicircle (A : ℝ × ℝ) : Prop :=
  (A.1 - 2)^2 + A.2^2 = 4 ∧ 2 ≤ A.1 ∧ A.1 ≤ 4

-- Main theorem
theorem rhombus_properties
  (A B C D : ℝ × ℝ)
  (h_rhombus : Rhombus A B C D)
  (h_OB : dist O B = 6)
  (h_OD : dist O D = 6)
  (h_A_semicircle : OnSemicircle A) :
  (∃ k, dist O A * dist O B = k) ∧
  (∃ y, -5 ≤ y ∧ y ≤ 5 ∧ C = (5, y)) :=
sorry

#check rhombus_properties

end NUMINAMATH_CALUDE_rhombus_properties_l2001_200137


namespace NUMINAMATH_CALUDE_baron_munchausen_claim_l2001_200158

theorem baron_munchausen_claim (weights : Finset ℕ) : 
  weights.card = 8 ∧ weights = Finset.range 8 →
  ∃ (A B C : Finset ℕ), 
    A ∪ B ∪ C = weights ∧
    A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
    A.card = 2 ∧ B.card = 5 ∧ C.card = 1 ∧
    (A.sum id = B.sum id) ∧
    (∀ w ∈ C, w = A.sum id - B.sum id) :=
by sorry

end NUMINAMATH_CALUDE_baron_munchausen_claim_l2001_200158


namespace NUMINAMATH_CALUDE_exterior_angle_HGI_exterior_angle_is_81_degrees_l2001_200164

-- Define the polygons
def Octagon : Type := Unit
def Decagon : Type := Unit

-- Define the properties of the polygons
axiom is_regular_octagon : Octagon → Prop
axiom is_regular_decagon : Decagon → Prop

-- Define the interior angles
def interior_angle_octagon (o : Octagon) (h : is_regular_octagon o) : ℝ := 135
def interior_angle_decagon (d : Decagon) (h : is_regular_decagon d) : ℝ := 144

-- Define the configuration
structure Configuration :=
  (o : Octagon)
  (d : Decagon)
  (ho : is_regular_octagon o)
  (hd : is_regular_decagon d)
  (share_side : Prop)

-- State the theorem
theorem exterior_angle_HGI (c : Configuration) : ℝ :=
  360 - interior_angle_octagon c.o c.ho - interior_angle_decagon c.d c.hd

-- The main theorem to prove
theorem exterior_angle_is_81_degrees (c : Configuration) :
  exterior_angle_HGI c = 81 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_HGI_exterior_angle_is_81_degrees_l2001_200164


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2001_200144

theorem quadratic_equation_solution (p q : ℝ) : p = 15 * q^2 - 5 ∧ p = 40 → q = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2001_200144


namespace NUMINAMATH_CALUDE_equation_solution_l2001_200195

theorem equation_solution : ∃! x : ℚ, x ≠ -3 ∧ (x^2 + 3*x + 5) / (x + 3) = x + 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2001_200195


namespace NUMINAMATH_CALUDE_last_s_replacement_l2001_200196

/-- Represents the rules of the cryptographic code --/
structure CryptoRules where
  firstShift : ℕ
  vowels : List Char
  vowelSequence : List ℕ

/-- Counts the occurrences of a character in a string --/
def countOccurrences (c : Char) (s : String) : ℕ := sorry

/-- Calculates the triangular number for a given n --/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Applies the shift to a character based on the rules --/
def applyShift (c : Char) (count : ℕ) (rules : CryptoRules) : Char := sorry

/-- Main theorem to prove --/
theorem last_s_replacement (message : String) (rules : CryptoRules) :
  let lastSCount := countOccurrences 's' message
  let shift := triangularNumber lastSCount % 26
  let newPos := (('s'.toNat - 'a'.toNat + 1 + shift) % 26) + 'a'.toNat - 1
  Char.ofNat newPos = 'g' := by sorry

end NUMINAMATH_CALUDE_last_s_replacement_l2001_200196


namespace NUMINAMATH_CALUDE_f_minus_two_equals_minus_twelve_l2001_200135

def symmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

theorem f_minus_two_equals_minus_twelve
  (f : ℝ → ℝ)
  (h_symmetric : symmetricAbout f 1)
  (h_def : ∀ x : ℝ, x ≥ 1 → f x = x * (1 - x)) :
  f (-2) = -12 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_two_equals_minus_twelve_l2001_200135


namespace NUMINAMATH_CALUDE_sum_equals_12x_l2001_200183

theorem sum_equals_12x (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y - x) : 
  x + y + z = 12 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_12x_l2001_200183


namespace NUMINAMATH_CALUDE_butter_production_theorem_l2001_200134

/-- Represents the problem of determining butter production from milk --/
structure MilkButterProblem where
  milk_price : ℚ
  butter_price : ℚ
  num_cows : ℕ
  milk_per_cow : ℕ
  num_customers : ℕ
  milk_per_customer : ℕ
  total_earnings : ℚ

/-- Calculates the number of sticks of butter that can be made from one gallon of milk --/
def sticks_per_gallon (p : MilkButterProblem) : ℚ :=
  let total_milk := p.num_cows * p.milk_per_cow
  let sold_milk := p.num_customers * p.milk_per_customer
  let milk_revenue := sold_milk * p.milk_price
  let butter_revenue := p.total_earnings - milk_revenue
  let milk_for_butter := total_milk - sold_milk
  let total_butter_sticks := butter_revenue / p.butter_price
  total_butter_sticks / milk_for_butter

/-- Theorem stating that for the given problem conditions, 2 sticks of butter can be made per gallon of milk --/
theorem butter_production_theorem (p : MilkButterProblem) 
  (h1 : p.milk_price = 3)
  (h2 : p.butter_price = 3/2)
  (h3 : p.num_cows = 12)
  (h4 : p.milk_per_cow = 4)
  (h5 : p.num_customers = 6)
  (h6 : p.milk_per_customer = 6)
  (h7 : p.total_earnings = 144) :
  sticks_per_gallon p = 2 := by
  sorry

end NUMINAMATH_CALUDE_butter_production_theorem_l2001_200134


namespace NUMINAMATH_CALUDE_parabola_x_axis_intersections_l2001_200130

/-- The number of intersection points between y = 3x^2 + 2x + 1 and the x-axis is 0 -/
theorem parabola_x_axis_intersections :
  let f (x : ℝ) := 3 * x^2 + 2 * x + 1
  (∃ x : ℝ, f x = 0) = False :=
by sorry

end NUMINAMATH_CALUDE_parabola_x_axis_intersections_l2001_200130


namespace NUMINAMATH_CALUDE_age_equation_solution_l2001_200160

theorem age_equation_solution (A : ℝ) (N : ℝ) (h1 : A = 64) :
  (1 / 2) * ((A + 8) * N - N * (A - 8)) = A ↔ N = 8 := by
  sorry

end NUMINAMATH_CALUDE_age_equation_solution_l2001_200160


namespace NUMINAMATH_CALUDE_range_of_a_l2001_200157

open Set

/-- The range of a for which ¬p is a necessary but not sufficient condition for ¬q -/
theorem range_of_a (a : ℝ) : 
  (a < 0) →
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) → 
    (x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0)) →
  (∃ x : ℝ, (x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0) ∧ 
    ¬(x^2 - 4*a*x + 3*a^2 < 0)) →
  (a ≤ -4 ∨ -2/3 ≤ a) :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_l2001_200157


namespace NUMINAMATH_CALUDE_complex_root_modulus_one_l2001_200139

theorem complex_root_modulus_one (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (∃ k : ℤ, n + 2 = 6 * k) :=
sorry

end NUMINAMATH_CALUDE_complex_root_modulus_one_l2001_200139


namespace NUMINAMATH_CALUDE_min_value_of_f_l2001_200178

/-- The function f(x) = 3x^2 - 18x + 2205 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2205

theorem min_value_of_f :
  ∃ (min : ℝ), min = 2178 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2001_200178


namespace NUMINAMATH_CALUDE_diophantine_equation_prime_sum_l2001_200123

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem diophantine_equation_prime_sum (a b : ℕ) :
  a > 0 ∧ b > 0 ∧ a^2 + b^2 + 25 = 15*a*b ∧ is_prime (a^2 + a*b + b^2) →
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_prime_sum_l2001_200123


namespace NUMINAMATH_CALUDE_distance_traveled_by_A_is_60km_l2001_200188

/-- Calculates the distance traveled by A until meeting B given initial conditions and speed doubling rule -/
def distanceTraveledByA (initialDistance : ℝ) (initialSpeedA : ℝ) (initialSpeedB : ℝ) : ℝ :=
  let firstHourDistance := initialSpeedA
  let secondHourDistance := 2 * initialSpeedA
  let thirdHourDistance := 4 * initialSpeedA * 0.75
  firstHourDistance + secondHourDistance + thirdHourDistance

/-- Theorem stating that A travels 60 km until meeting B -/
theorem distance_traveled_by_A_is_60km :
  distanceTraveledByA 90 10 5 = 60 := by
  sorry

#eval distanceTraveledByA 90 10 5

end NUMINAMATH_CALUDE_distance_traveled_by_A_is_60km_l2001_200188


namespace NUMINAMATH_CALUDE_product_of_powers_equals_power_of_sum_l2001_200110

theorem product_of_powers_equals_power_of_sum :
  (10 ^ 0.4) * (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.05) * (10 ^ 1.1) * (10 ^ (-0.1)) = 10 ^ 1.85 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_power_of_sum_l2001_200110


namespace NUMINAMATH_CALUDE_square_sum_theorem_l2001_200115

theorem square_sum_theorem (x y : ℝ) (h1 : x + 2*y = 4) (h2 : x*y = -8) : 
  x^2 + 4*y^2 = 48 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l2001_200115


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2001_200180

noncomputable def f (x : ℝ) : ℝ := Real.exp (abs x) * Real.sin x

theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Ioo (-π/4) (3*π/4)) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2001_200180


namespace NUMINAMATH_CALUDE_line_through_points_with_45_degree_slope_l2001_200185

/-- Given a line passing through points (3, m) and (2, 4) with a slope angle of 45°, prove that m = 5. -/
theorem line_through_points_with_45_degree_slope (m : ℝ) :
  (∃ (line : Set (ℝ × ℝ)), 
    (3, m) ∈ line ∧ 
    (2, 4) ∈ line ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → y - 4 = x - 2)) → 
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_with_45_degree_slope_l2001_200185


namespace NUMINAMATH_CALUDE_actual_car_mass_is_1331_l2001_200105

/-- The mass of a scaled model car -/
def model_mass : ℝ := 1

/-- The scale factor between the model and the actual car -/
def scale_factor : ℝ := 11

/-- Calculates the mass of the actual car given the model mass and scale factor -/
def actual_car_mass (model_mass : ℝ) (scale_factor : ℝ) : ℝ :=
  model_mass * (scale_factor ^ 3)

/-- Theorem stating that the mass of the actual car is 1331 kg -/
theorem actual_car_mass_is_1331 :
  actual_car_mass model_mass scale_factor = 1331 := by
  sorry

end NUMINAMATH_CALUDE_actual_car_mass_is_1331_l2001_200105


namespace NUMINAMATH_CALUDE_right_triangle_pq_length_l2001_200133

/-- Given a right triangle PQR with ∠P = 90°, QR = 15, and tan R = 5 cos Q, prove that PQ = 6√6 -/
theorem right_triangle_pq_length (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let qr := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  let pr := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  let cos_q := pq / qr
  let tan_r := pq / pr
  (P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2) = 0 →  -- right angle at P
  qr = 15 →
  tan_r = 5 * cos_q →
  pq = 6 * Real.sqrt 6 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_pq_length_l2001_200133


namespace NUMINAMATH_CALUDE_afternoon_evening_difference_is_24_l2001_200128

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 33

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 34

/-- The number of campers who went rowing in the evening -/
def evening_campers : ℕ := 10

/-- The difference between the number of campers rowing in the afternoon and evening -/
def afternoon_evening_difference : ℕ := afternoon_campers - evening_campers

theorem afternoon_evening_difference_is_24 : 
  afternoon_evening_difference = 24 := by sorry

end NUMINAMATH_CALUDE_afternoon_evening_difference_is_24_l2001_200128


namespace NUMINAMATH_CALUDE_world_book_day_solution_l2001_200166

/-- Represents the number of books bought by each student -/
structure BookCount where
  a : ℕ
  b : ℕ

/-- The conditions of the World Book Day problem -/
def worldBookDayProblem (bc : BookCount) : Prop :=
  bc.a + bc.b = 22 ∧ bc.a = 2 * bc.b + 1

/-- The theorem stating the solution to the World Book Day problem -/
theorem world_book_day_solution :
  ∃ (bc : BookCount), worldBookDayProblem bc ∧ bc.a = 15 ∧ bc.b = 7 := by
  sorry

end NUMINAMATH_CALUDE_world_book_day_solution_l2001_200166


namespace NUMINAMATH_CALUDE_rectangular_plot_width_l2001_200193

theorem rectangular_plot_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 432 →
  width = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_width_l2001_200193


namespace NUMINAMATH_CALUDE_males_not_listening_l2001_200171

/-- Represents the survey results -/
structure SurveyResults where
  total_listeners : ℕ
  total_non_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ

/-- Theorem stating that the number of males who don't listen is 85 -/
theorem males_not_listening (survey : SurveyResults)
  (h1 : survey.total_listeners = 160)
  (h2 : survey.total_non_listeners = 180)
  (h3 : survey.female_listeners = 75)
  (h4 : survey.male_non_listeners = 85) :
  survey.male_non_listeners = 85 := by
  sorry

#check males_not_listening

end NUMINAMATH_CALUDE_males_not_listening_l2001_200171


namespace NUMINAMATH_CALUDE_mans_rowing_speed_in_still_water_l2001_200113

/-- Proves that a man's rowing speed in still water is 25 km/hr given the conditions of downstream speed and time. -/
theorem mans_rowing_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 3) -- Current speed in km/hr
  (h2 : distance = 90) -- Distance in meters
  (h3 : time = 17.998560115190784) -- Time in seconds
  : ∃ (still_water_speed : ℝ), still_water_speed = 25 := by
  sorry


end NUMINAMATH_CALUDE_mans_rowing_speed_in_still_water_l2001_200113


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_one_l2001_200186

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_one_l2001_200186


namespace NUMINAMATH_CALUDE_divisibility_condition_l2001_200106

theorem divisibility_condition (n : ℕ+) :
  (5^(n.val - 1) + 3^(n.val - 1)) ∣ (5^n.val + 3^n.val) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2001_200106


namespace NUMINAMATH_CALUDE_front_top_area_ratio_l2001_200131

/-- A rectangular box with given properties -/
structure Box where
  volume : ℝ
  side_area : ℝ
  top_area : ℝ
  front_area : ℝ
  top_side_ratio : ℝ

/-- The theorem stating the ratio of front face area to top face area -/
theorem front_top_area_ratio (b : Box) 
  (h_volume : b.volume = 5184)
  (h_side_area : b.side_area = 288)
  (h_top_side_ratio : b.top_area = 1.5 * b.side_area) :
  b.front_area / b.top_area = 1 / 2 := by
  sorry

#check front_top_area_ratio

end NUMINAMATH_CALUDE_front_top_area_ratio_l2001_200131


namespace NUMINAMATH_CALUDE_stamp_arrangement_count_l2001_200179

/-- Represents a stamp with its denomination -/
structure Stamp where
  denomination : Nat
  deriving Repr

/-- Represents an arrangement of stamps -/
def Arrangement := List Stamp

/-- Checks if an arrangement is valid (sums to 15 cents) -/
def isValidArrangement (arr : Arrangement) : Bool :=
  (arr.map (·.denomination)).sum = 15

/-- Checks if two arrangements are considered equivalent -/
def areEquivalentArrangements (arr1 arr2 : Arrangement) : Bool :=
  sorry  -- Implementation details omitted

/-- The set of all possible stamps -/
def allStamps : List Stamp :=
  (List.range 12).map (λ i => ⟨i + 1⟩) ++ (List.range 12).map (λ i => ⟨i + 1⟩)

/-- Generates all valid arrangements -/
def generateValidArrangements (stamps : List Stamp) : List Arrangement :=
  sorry  -- Implementation details omitted

/-- Counts distinct arrangements after considering equivalence -/
def countDistinctArrangements (arrangements : List Arrangement) : Nat :=
  sorry  -- Implementation details omitted

theorem stamp_arrangement_count :
  countDistinctArrangements (generateValidArrangements allStamps) = 213 := by
  sorry

end NUMINAMATH_CALUDE_stamp_arrangement_count_l2001_200179


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2001_200109

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 4 * x + 1 = 0) ↔ (k ≤ 4 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2001_200109


namespace NUMINAMATH_CALUDE_line_through_point_and_circle_center_l2001_200116

/-- A line passing through two points on a plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in a plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The theorem stating that the equation of a line passing through a given point and the center of a given circle is x-2=0. -/
theorem line_through_point_and_circle_center 
  (M : ℝ × ℝ) 
  (C : Circle) 
  (h1 : M.1 = 2 ∧ M.2 = 3) 
  (h2 : C.center = (2, -3)) 
  (h3 : C.radius = 3) : 
  ∃ (l : Line), l.a = 1 ∧ l.b = 0 ∧ l.c = -2 :=
sorry

end NUMINAMATH_CALUDE_line_through_point_and_circle_center_l2001_200116


namespace NUMINAMATH_CALUDE_distance_implies_product_l2001_200145

/-- Given two points (3a, a-5) and (7, 2), if the distance between these points is 3√10,
    then the product of all possible values of a is 0.8. -/
theorem distance_implies_product (a₁ a₂ : ℝ) : 
  (3 * a₁ - 7)^2 + (a₁ - 7)^2 = 90 →
  (3 * a₂ - 7)^2 + (a₂ - 7)^2 = 90 →
  a₁ ≠ a₂ →
  a₁ * a₂ = 0.8 := by
sorry

end NUMINAMATH_CALUDE_distance_implies_product_l2001_200145


namespace NUMINAMATH_CALUDE_floor_abs_sum_l2001_200152

theorem floor_abs_sum : ⌊|(-5.3 : ℝ)|⌋ + |⌊(-5.3 : ℝ)⌋| = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l2001_200152


namespace NUMINAMATH_CALUDE_max_d_value_l2001_200153

def a (n : ℕ) : ℕ := 100 + n^n

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n-1))

theorem max_d_value :
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → d n ≤ 401 ∧ ∃ (m : ℕ), m ≥ N ∧ d m = 401 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l2001_200153


namespace NUMINAMATH_CALUDE_flea_return_probability_l2001_200108

/-- A flea jumps on a number line with the following properties:
    - It starts at 0
    - Each jump has a length of 1
    - The probability of jumping in the same direction as the previous jump is p
    - The probability of jumping in the opposite direction is 1-p -/
def FleaJump (p : ℝ) := 
  {flea : ℕ → ℝ // flea 0 = 0 ∧ ∀ n, |flea (n+1) - flea n| = 1}

/-- The probability that the flea returns to 0 -/
noncomputable def ReturnProbability (p : ℝ) : ℝ := sorry

/-- The theorem stating the probability of the flea returning to 0 -/
theorem flea_return_probability (p : ℝ) : 
  ReturnProbability p = if p = 1 then 0 else 1 := by sorry

end NUMINAMATH_CALUDE_flea_return_probability_l2001_200108


namespace NUMINAMATH_CALUDE_min_value_of_f_l2001_200111

/-- The function f(x) = x^2 + 8x + 12 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 12

/-- The minimum value of f(x) is -4 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -4 ∧ ∀ (x : ℝ), f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2001_200111


namespace NUMINAMATH_CALUDE_beef_for_community_event_l2001_200181

/-- The amount of beef needed for a given number of hamburgers -/
def beef_needed (hamburgers : ℕ) : ℚ :=
  (4 : ℚ) / 10 * hamburgers

theorem beef_for_community_event : beef_needed 35 = 14 := by
  sorry

end NUMINAMATH_CALUDE_beef_for_community_event_l2001_200181


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l2001_200102

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a)) ≥ 1 / Real.rpow 6 (1/3) :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b) + b / (6 * c) + c / (9 * a)) = 1 / Real.rpow 6 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l2001_200102


namespace NUMINAMATH_CALUDE_arc_length_for_36_degree_angle_l2001_200197

theorem arc_length_for_36_degree_angle (d : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  d = 4 → θ_deg = 36 → l = (θ_deg * π / 180) * (d / 2) → l = 2 * π / 5 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_36_degree_angle_l2001_200197


namespace NUMINAMATH_CALUDE_product_remainder_by_five_l2001_200138

theorem product_remainder_by_five : 
  (2685 * 4932 * 91406) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_five_l2001_200138


namespace NUMINAMATH_CALUDE_no_square_143_b_l2001_200148

theorem no_square_143_b : ¬ ∃ (b : ℤ), b > 4 ∧ ∃ (n : ℤ), b^2 + 4*b + 3 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_square_143_b_l2001_200148


namespace NUMINAMATH_CALUDE_intersection_A_B_l2001_200154

def A : Set ℝ := {x | x ≤ 2*x + 1 ∧ 2*x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2001_200154


namespace NUMINAMATH_CALUDE_expand_product_l2001_200184

theorem expand_product (x : ℝ) : (x + 3) * (x^2 + 4*x + 6) = x^3 + 7*x^2 + 18*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2001_200184


namespace NUMINAMATH_CALUDE_hyperbola_points_l2001_200136

def hyperbola (x y : ℝ) : Prop := y = -4 / x

theorem hyperbola_points :
  hyperbola (-2) 2 ∧
  ¬ hyperbola 1 4 ∧
  ¬ hyperbola (-1) (-4) ∧
  ¬ hyperbola (-2) (-2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_points_l2001_200136


namespace NUMINAMATH_CALUDE_simplify_expression_l2001_200175

theorem simplify_expression : 0.72 * 0.43 + 0.12 * 0.34 = 0.3504 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2001_200175


namespace NUMINAMATH_CALUDE_portion_to_whole_cup_ratio_l2001_200163

theorem portion_to_whole_cup_ratio : 
  ∀ (grains_per_cup : ℕ) 
    (tablespoons_per_portion : ℕ) 
    (teaspoons_per_tablespoon : ℕ) 
    (grains_per_teaspoon : ℕ),
  grains_per_cup = 480 →
  tablespoons_per_portion = 8 →
  teaspoons_per_tablespoon = 3 →
  grains_per_teaspoon = 10 →
  (tablespoons_per_portion * teaspoons_per_tablespoon * grains_per_teaspoon) * 2 = grains_per_cup :=
by
  sorry

end NUMINAMATH_CALUDE_portion_to_whole_cup_ratio_l2001_200163


namespace NUMINAMATH_CALUDE_range_of_f_l2001_200118

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ∈ domain, f x = y} = {y : ℝ | 2 ≤ y ∧ y ≤ 6} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2001_200118


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l2001_200194

theorem smallest_integer_solution (x : ℤ) : 3 * x - 7 ≤ 17 → x ≤ 8 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l2001_200194


namespace NUMINAMATH_CALUDE_evaluate_expression_l2001_200104

theorem evaluate_expression : (7 - 3)^2 + (7^2 - 3^2) = 56 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2001_200104


namespace NUMINAMATH_CALUDE_travelers_checks_worth_l2001_200127

-- Define the problem parameters
def total_checks : ℕ := 30
def small_denomination : ℕ := 50
def large_denomination : ℕ := 100
def spent_checks : ℕ := 18
def remaining_average : ℕ := 75

-- Define the theorem
theorem travelers_checks_worth :
  ∀ (x y : ℕ),
    x + y = total_checks →
    x ≥ spent_checks →
    (small_denomination * (x - spent_checks) + large_denomination * y) / (total_checks - spent_checks) = remaining_average →
    small_denomination * x + large_denomination * y = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_travelers_checks_worth_l2001_200127


namespace NUMINAMATH_CALUDE_square_greater_than_abs_square_l2001_200151

theorem square_greater_than_abs_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_abs_square_l2001_200151


namespace NUMINAMATH_CALUDE_max_triples_value_l2001_200129

/-- The size of the square table -/
def n : ℕ := 999

/-- Represents the color of a cell in the table -/
inductive CellColor
| White
| Red

/-- Represents a cell in the table -/
structure Cell where
  row : Fin n
  col : Fin n

/-- Represents the coloring of the table -/
def TableColoring := Fin n → Fin n → CellColor

/-- Counts the number of valid triples for a given table coloring -/
def countTriples (coloring : TableColoring) : ℕ := sorry

/-- The maximum number of valid triples possible -/
def maxTriples : ℕ := (4 * n^4) / 27

/-- Theorem stating that the maximum number of valid triples is (4 * 999⁴) / 27 -/
theorem max_triples_value :
  ∀ (coloring : TableColoring), countTriples coloring ≤ maxTriples :=
by sorry

end NUMINAMATH_CALUDE_max_triples_value_l2001_200129


namespace NUMINAMATH_CALUDE_carrie_cake_days_l2001_200100

/-- Proves that Carrie worked 4 days on the cake given the specified conditions. -/
theorem carrie_cake_days : 
  ∀ (hours_per_day : ℕ) (hourly_rate : ℕ) (supply_cost : ℕ) (profit : ℕ),
    hours_per_day = 2 →
    hourly_rate = 22 →
    supply_cost = 54 →
    profit = 122 →
    ∃ (days : ℕ), 
      days = 4 ∧ 
      profit = hours_per_day * hourly_rate * days - supply_cost :=
by
  sorry


end NUMINAMATH_CALUDE_carrie_cake_days_l2001_200100


namespace NUMINAMATH_CALUDE_unique_shapes_count_l2001_200173

-- Define a rectangle
structure Rectangle where
  vertices : Fin 4 → Point

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define an ellipse
structure Ellipse where
  foci : Point × Point
  major_axis : ℝ

-- Function to count unique shapes
def count_unique_shapes (R : Rectangle) : ℕ :=
  let circles := sorry
  let ellipses := sorry
  circles + ellipses

-- Theorem statement
theorem unique_shapes_count (R : Rectangle) :
  count_unique_shapes R = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_shapes_count_l2001_200173


namespace NUMINAMATH_CALUDE_cube_root_expansion_implication_l2001_200143

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

theorem cube_root_expansion_implication (n : ℕ) (hn : n > 0) :
  ∃! (a_n b_n c_n : ℤ), (1 + 4 * cubeRoot 2 - 4 * cubeRoot 4)^n = 
    a_n + b_n * cubeRoot 2 + c_n * cubeRoot 4 →
  (c_n = 0 → n = 0) := by
sorry

end NUMINAMATH_CALUDE_cube_root_expansion_implication_l2001_200143


namespace NUMINAMATH_CALUDE_michael_has_fifteen_robots_l2001_200190

/-- Calculates the number of flying robots Michael has given Tom's count and the multiplier. -/
def michaels_robots (toms_robots : ℕ) (multiplier : ℕ) : ℕ :=
  toms_robots * multiplier

/-- Proves that Michael has 15 flying robots given the conditions. -/
theorem michael_has_fifteen_robots :
  let toms_robots : ℕ := 3
  let multiplier : ℕ := 4
  michaels_robots toms_robots multiplier = 15 := by
  sorry

#eval michaels_robots 3 4  -- This should output 15

end NUMINAMATH_CALUDE_michael_has_fifteen_robots_l2001_200190


namespace NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2001_200167

theorem polynomial_multiplication_simplification (x : ℝ) :
  (3*x - 2) * (5*x^9 + 3*x^8 + 2*x^7 + x^6) = 15*x^10 - x^9 + 3*x^7 - 2*x^6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2001_200167


namespace NUMINAMATH_CALUDE_product_of_tangents_plus_one_l2001_200176

theorem product_of_tangents_plus_one (α : ℝ) :
  (1 + Real.tan (α * π / 12)) * (1 + Real.tan (α * π / 6)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_tangents_plus_one_l2001_200176


namespace NUMINAMATH_CALUDE_exists_partition_without_infinite_progression_l2001_200114

/-- A partition of natural numbers. -/
def Partition := ℕ → Bool

/-- Checks if a set contains an infinite arithmetic progression. -/
def HasInfiniteArithmeticProgression (p : Partition) : Prop :=
  ∃ a d : ℕ, d > 0 ∧ ∀ k : ℕ, p (a + k * d) = p a

/-- There exists a partition of natural numbers into two sets
    such that neither set contains an infinite arithmetic progression. -/
theorem exists_partition_without_infinite_progression :
  ∃ p : Partition, ¬HasInfiniteArithmeticProgression p ∧
                   ¬HasInfiniteArithmeticProgression (fun n => ¬(p n)) := by
  sorry

end NUMINAMATH_CALUDE_exists_partition_without_infinite_progression_l2001_200114


namespace NUMINAMATH_CALUDE_jills_salary_l2001_200107

/-- Represents a person's monthly finances -/
structure MonthlyFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  giftAmount : ℝ

/-- Conditions for Jill's monthly finances -/
def jillsFinances (f : MonthlyFinances) : Prop :=
  f.discretionaryIncome = f.netSalary / 5 ∧
  f.giftAmount = f.discretionaryIncome * 0.2 ∧
  f.giftAmount = 111

/-- Theorem: If Jill's finances meet the given conditions, her net monthly salary is $2775 -/
theorem jills_salary (f : MonthlyFinances) (h : jillsFinances f) : f.netSalary = 2775 := by
  sorry

end NUMINAMATH_CALUDE_jills_salary_l2001_200107


namespace NUMINAMATH_CALUDE_digit_120_is_1_l2001_200174

/-- Represents the decimal number formed by concatenating integers 1 to 51 -/
def x : ℚ :=
  0.123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051

/-- Returns the nth digit after the decimal point in a rational number -/
def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_120_is_1 : nthDigitAfterDecimal x 120 = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_120_is_1_l2001_200174


namespace NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l2001_200146

/-- A sequence of binary digits (0 or 1) -/
def BinarySequence := List Nat

/-- Count the number of (1,0) pairs with even number of digits between them -/
def countEvenPairs (seq : BinarySequence) : Nat :=
  sorry

/-- Count the number of (1,0) pairs with odd number of digits between them -/
def countOddPairs (seq : BinarySequence) : Nat :=
  sorry

/-- The main theorem: For any binary sequence, the number of (1,0) pairs
    with even number of digits between is greater than or equal to
    the number of (1,0) pairs with odd number of digits between -/
theorem even_pairs_ge_odd_pairs (seq : BinarySequence) :
  countEvenPairs seq ≥ countOddPairs seq :=
sorry

end NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l2001_200146


namespace NUMINAMATH_CALUDE_second_part_speed_l2001_200147

/-- Proves that given a total distance of 20 miles, where the first 10 miles are traveled at 12 miles per hour,
    and the average speed for the entire trip is 10.909090909090908 miles per hour,
    the speed for the second part of the trip is 10 miles per hour. -/
theorem second_part_speed
  (total_distance : ℝ)
  (first_part_distance : ℝ)
  (first_part_speed : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 20)
  (h2 : first_part_distance = 10)
  (h3 : first_part_speed = 12)
  (h4 : average_speed = 10.909090909090908)
  : ∃ (second_part_speed : ℝ),
    second_part_speed = 10 ∧
    average_speed = (first_part_distance / first_part_speed + (total_distance - first_part_distance) / second_part_speed) / (total_distance / average_speed) :=
by
  sorry

end NUMINAMATH_CALUDE_second_part_speed_l2001_200147


namespace NUMINAMATH_CALUDE_indeterminate_larger_number_l2001_200187

/-- Given two real numbers x and y and a constant k such that
    x * k = y + 1 and x + y = -64, prove that it's not possible
    to determine which of x or y is larger without additional information. -/
theorem indeterminate_larger_number (x y k : ℝ) 
    (h1 : x * k = y + 1) 
    (h2 : x + y = -64) : 
  ¬ (∀ x y : ℝ, (x * k = y + 1 ∧ x + y = -64) → x < y ∨ y < x) :=
by
  sorry


end NUMINAMATH_CALUDE_indeterminate_larger_number_l2001_200187


namespace NUMINAMATH_CALUDE_some_birds_are_white_l2001_200172

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (Swan : U → Prop)
variable (Bird : U → Prop)
variable (White : U → Prop)

-- State our theorem
theorem some_birds_are_white
  (h1 : ∀ x, Swan x → White x)  -- All swans are white
  (h2 : ∃ x, Bird x ∧ Swan x)   -- Some birds are swans
  : ∃ x, Bird x ∧ White x :=    -- Conclusion: Some birds are white
by sorry

end NUMINAMATH_CALUDE_some_birds_are_white_l2001_200172


namespace NUMINAMATH_CALUDE_cubic_plus_linear_increasing_l2001_200156

/-- The function f(x) = x^3 + x is strictly increasing on all real numbers. -/
theorem cubic_plus_linear_increasing : 
  ∀ x y : ℝ, x < y → (x^3 + x) < (y^3 + y) := by
sorry

end NUMINAMATH_CALUDE_cubic_plus_linear_increasing_l2001_200156


namespace NUMINAMATH_CALUDE_inequality_range_l2001_200170

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| ≥ a*x) ↔ a ∈ Set.Iic 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2001_200170


namespace NUMINAMATH_CALUDE_vector_2016_coordinates_l2001_200169

def matrix_transformation (x_n y_n : ℝ) : ℝ × ℝ :=
  (x_n, x_n + y_n)

def vector_sequence (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => (2, 0)
  | n + 1 => matrix_transformation (vector_sequence n).1 (vector_sequence n).2

theorem vector_2016_coordinates :
  vector_sequence 2015 = (2, 4030) := by
  sorry

end NUMINAMATH_CALUDE_vector_2016_coordinates_l2001_200169


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l2001_200198

-- Define the sets S and T
def S : Set ℝ := {0, 1, 2, 3}
def T : Set ℝ := {x | |x - 1| ≤ 1}

-- State the theorem
theorem intersection_of_S_and_T : S ∩ T = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l2001_200198


namespace NUMINAMATH_CALUDE_multiplication_and_subtraction_l2001_200126

theorem multiplication_and_subtraction : 10 * (5 - 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_and_subtraction_l2001_200126


namespace NUMINAMATH_CALUDE_exists_infinite_periodic_sequence_l2001_200112

/-- A sequence of natural numbers -/
def InfiniteSequence := ℕ → ℕ

/-- Property: every natural number appears infinitely many times in the sequence -/
def AppearsInfinitelyOften (s : InfiniteSequence) : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, ∃ i ≥ k, s i = n

/-- Property: the sequence is periodic modulo m for every positive integer m -/
def PeriodicModulo (s : InfiniteSequence) : Prop :=
  ∀ m : ℕ+, ∃ p : ℕ+, ∀ i : ℕ, s (i + p) ≡ s i [MOD m]

/-- Theorem: There exists a sequence of natural numbers that appears infinitely often
    and is periodic modulo every positive integer -/
theorem exists_infinite_periodic_sequence :
  ∃ s : InfiniteSequence, AppearsInfinitelyOften s ∧ PeriodicModulo s := by
  sorry

end NUMINAMATH_CALUDE_exists_infinite_periodic_sequence_l2001_200112


namespace NUMINAMATH_CALUDE_water_amount_l2001_200125

/-- The number of boxes -/
def num_boxes : ℕ := 10

/-- The number of bottles in each box -/
def bottles_per_box : ℕ := 50

/-- The capacity of each bottle in liters -/
def bottle_capacity : ℚ := 12

/-- The fraction of the bottle's capacity that is filled -/
def fill_fraction : ℚ := 3/4

/-- The total amount of water in liters contained in all boxes -/
def total_water : ℚ := num_boxes * bottles_per_box * bottle_capacity * fill_fraction

theorem water_amount : total_water = 4500 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_l2001_200125


namespace NUMINAMATH_CALUDE_box_volume_increase_l2001_200189

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 4320)
  (surface_area : 2 * (l * w + w * h + h * l) = 1704)
  (edge_sum : 4 * (l + w + h) = 208) :
  (l + 1) * (w + 1) * (h + 1) = 5225 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2001_200189
