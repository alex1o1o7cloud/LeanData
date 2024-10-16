import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l3505_350519

theorem expression_evaluation : 4 * (-3) + 60 / (-15) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3505_350519


namespace NUMINAMATH_CALUDE_tangent_at_P_tangent_through_P_l3505_350598

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Theorem for the tangent line at P
theorem tangent_at_P :
  let (x₀, y₀) := P
  let slope := (3 * x₀^2 - 3 : ℝ)
  (∀ x, f x₀ + slope * (x - x₀) = -2) :=
sorry

-- Theorem for the tangent lines passing through P
theorem tangent_through_P :
  let (x₀, y₀) := P
  (∃ x₁ : ℝ, 
    (∀ x, f x₁ + (3 * x₁^2 - 3) * (x - x₁) = -2) ∨
    (∀ x, f x₁ + (3 * x₁^2 - 3) * (x - x₁) = -9/4*x + 1/4)) :=
sorry

end NUMINAMATH_CALUDE_tangent_at_P_tangent_through_P_l3505_350598


namespace NUMINAMATH_CALUDE_square_difference_39_40_square_41_from_40_l3505_350577

theorem square_difference_39_40 :
  (40 : ℕ)^2 - (39 : ℕ)^2 = 79 :=
by
  sorry

-- Additional theorem to represent the given condition
theorem square_41_from_40 :
  (41 : ℕ)^2 = (40 : ℕ)^2 + 81 :=
by
  sorry

end NUMINAMATH_CALUDE_square_difference_39_40_square_41_from_40_l3505_350577


namespace NUMINAMATH_CALUDE_home_learning_percentage_l3505_350535

-- Define the percentage of students present in school
def students_present : ℝ := 30

-- Define the theorem
theorem home_learning_percentage :
  let total_percentage : ℝ := 100
  let non_home_learning : ℝ := 2 * students_present
  let home_learning : ℝ := total_percentage - non_home_learning
  home_learning = 40 := by sorry

end NUMINAMATH_CALUDE_home_learning_percentage_l3505_350535


namespace NUMINAMATH_CALUDE_hex_F2E1_equals_62177_l3505_350520

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : Nat :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3 | '4' => 4 | '5' => 5 | '6' => 6 | '7' => 7
  | '8' => 8 | '9' => 9 | 'A' => 10 | 'B' => 11 | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_to_decimal (s : String) : Nat :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

/-- The hexadecimal number F2E1 -/
def hex_number : String := "F2E1"

/-- Theorem stating that F2E1 in hexadecimal is equal to 62177 in decimal -/
theorem hex_F2E1_equals_62177 : hex_to_decimal hex_number = 62177 := by
  sorry

end NUMINAMATH_CALUDE_hex_F2E1_equals_62177_l3505_350520


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_digit_sum_l3505_350596

/-- Function to create a number with n digits of 1 -/
def oneDigits (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Function to calculate the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sumOfDigits (n / 10)

/-- Theorem stating that there are infinitely many integers divisible by the sum of their digits -/
theorem infinitely_many_divisible_by_digit_sum :
  ∀ n : ℕ, ∃ k : ℕ,
    k > 0 ∧
    (∀ d : ℕ, d > 0 → d < 10 → k % d ≠ 0) ∧ 
    (k % (sumOfDigits k) = 0) :=
by
  intro n
  use oneDigits (3^n)
  sorry

/-- Lemma: The number created by oneDigits(3^n) has exactly 3^n digits, all of which are 1 -/
lemma oneDigits_all_ones (n : ℕ) :
  ∀ d : ℕ, d > 0 → d < 10 → (oneDigits (3^n)) % d ≠ 0 :=
by sorry

/-- Lemma: The sum of digits of oneDigits(3^n) is equal to 3^n -/
lemma sum_of_digits_oneDigits (n : ℕ) :
  sumOfDigits (oneDigits (3^n)) = 3^n :=
by sorry

/-- Lemma: oneDigits(3^n) is divisible by 3^n -/
lemma oneDigits_divisible (n : ℕ) :
  (oneDigits (3^n)) % (3^n) = 0 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_digit_sum_l3505_350596


namespace NUMINAMATH_CALUDE_expression_not_constant_l3505_350512

theorem expression_not_constant (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  ¬ ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 3 → x ≠ -2 → 
    (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - 
    (x^2 + 4 * x + 4) / ((x - 3) * (x + 2)) = c :=
by sorry

end NUMINAMATH_CALUDE_expression_not_constant_l3505_350512


namespace NUMINAMATH_CALUDE_steve_distance_theorem_l3505_350582

def steve_problem (distance : ℝ) : Prop :=
  let speed_to_work : ℝ := 17.5 / 2
  let speed_from_work : ℝ := 17.5
  let time_to_work : ℝ := distance / speed_to_work
  let time_from_work : ℝ := distance / speed_from_work
  (time_to_work + time_from_work = 6) ∧ (speed_from_work = 2 * speed_to_work)

theorem steve_distance_theorem : 
  ∃ (distance : ℝ), steve_problem distance ∧ distance = 35 := by
  sorry

end NUMINAMATH_CALUDE_steve_distance_theorem_l3505_350582


namespace NUMINAMATH_CALUDE_abc_books_sold_l3505_350571

theorem abc_books_sold (top_price : ℕ) (abc_price : ℕ) (top_sold : ℕ) (earnings_diff : ℕ) :
  top_price = 8 →
  abc_price = 23 →
  top_sold = 13 →
  earnings_diff = 12 →
  ∃ (abc_sold : ℕ), abc_sold * abc_price = top_sold * top_price - earnings_diff :=
by
  sorry

end NUMINAMATH_CALUDE_abc_books_sold_l3505_350571


namespace NUMINAMATH_CALUDE_fraction_simplification_l3505_350509

theorem fraction_simplification (d : ℝ) : (5 + 4 * d) / 7 + 3 = (26 + 4 * d) / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3505_350509


namespace NUMINAMATH_CALUDE_sum_of_special_system_l3505_350536

theorem sum_of_special_system (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (a + c)) :
  a + b + c = 1128 / 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_system_l3505_350536


namespace NUMINAMATH_CALUDE_max_toads_in_two_ponds_l3505_350583

/-- Represents a pond with frogs and toads -/
structure Pond where
  frogRatio : ℕ
  toadRatio : ℕ

/-- The maximum number of toads given two ponds and a total number of frogs -/
def maxToads (pond1 pond2 : Pond) (totalFrogs : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum number of toads in the given scenario -/
theorem max_toads_in_two_ponds :
  let pond1 : Pond := { frogRatio := 3, toadRatio := 4 }
  let pond2 : Pond := { frogRatio := 5, toadRatio := 6 }
  let totalFrogs : ℕ := 36
  maxToads pond1 pond2 totalFrogs = 46 := by
  sorry

end NUMINAMATH_CALUDE_max_toads_in_two_ponds_l3505_350583


namespace NUMINAMATH_CALUDE_symmetry_implies_t_zero_l3505_350573

/-- Line l in the Cartesian coordinate system -/
def line_l (x y : ℝ) : Prop :=
  8 * x + 6 * y + 1 = 0

/-- Circle C₁ in the Cartesian coordinate system -/
def circle_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 8*x - 2*y + 13 = 0

/-- Circle C₂ in the Cartesian coordinate system -/
def circle_C2 (t x y : ℝ) : Prop :=
  x^2 + y^2 + 8*t*x - 8*y + 16*t + 12 = 0

/-- The center of circle C₁ -/
def center_C1 : ℝ × ℝ :=
  (-4, 1)

/-- The center of circle C₂ -/
def center_C2 (t : ℝ) : ℝ × ℝ :=
  (-4*t, 4)

/-- Theorem: When circle C₁ and circle C₂ are symmetric about line l, t = 0 -/
theorem symmetry_implies_t_zero :
  ∀ t : ℝ, (∃ x y : ℝ, line_l x y ∧ 
    ((x - (-4))^2 + (y - 1)^2 = (x - (-4*t))^2 + (y - 4)^2) ∧
    ((8*x + 6*y + 1 = 0) → 
      ((-4 + (-4*t))/2 = x ∧ (1 + 4)/2 = y))) →
  t = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_t_zero_l3505_350573


namespace NUMINAMATH_CALUDE_taller_tree_height_taller_tree_height_proof_l3505_350522

/-- Given two trees where one is 20 feet taller than the other and their heights are in the ratio 2:3,
    the height of the taller tree is 60 feet. -/
theorem taller_tree_height : ℝ → ℝ → Prop :=
  fun h₁ h₂ => (h₁ = h₂ + 20 ∧ h₂ / h₁ = 2 / 3) → h₁ = 60

/-- Proof of the theorem -/
theorem taller_tree_height_proof : taller_tree_height 60 40 := by
  sorry

end NUMINAMATH_CALUDE_taller_tree_height_taller_tree_height_proof_l3505_350522


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3505_350518

theorem arithmetic_sequence_length
  (a : ℤ)  -- First term
  (l : ℤ)  -- Last term
  (d : ℤ)  -- Common difference
  (h1 : a = -22)
  (h2 : l = 50)
  (h3 : d = 7)
  : (l - a) / d + 1 = 11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3505_350518


namespace NUMINAMATH_CALUDE_integerRoot_of_infiniteSolutions_l3505_350563

/-- A polynomial of degree 3 with integer coefficients -/
def IntPolynomial3 : Type := ℤ → ℤ

/-- The property that xP(x) = yP(y) has infinitely many solutions for distinct integers x and y -/
def HasInfiniteSolutions (P : IntPolynomial3) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ x * P x = y * P y ∧ (abs x > n ∨ abs y > n)

/-- The property that a polynomial has an integer root -/
def HasIntegerRoot (P : IntPolynomial3) : Prop :=
  ∃ k : ℤ, P k = 0

/-- The main theorem -/
theorem integerRoot_of_infiniteSolutions (P : IntPolynomial3) 
  (h : HasInfiniteSolutions P) : HasIntegerRoot P :=
sorry

end NUMINAMATH_CALUDE_integerRoot_of_infiniteSolutions_l3505_350563


namespace NUMINAMATH_CALUDE_min_value_expression_l3505_350528

theorem min_value_expression (k : ℕ) (hk : k > 0) : 
  (10 : ℝ) / 3 + 32 / 10 ≤ (k : ℝ) / 3 + 32 / k :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3505_350528


namespace NUMINAMATH_CALUDE_equation_roots_existence_and_bounds_l3505_350557

theorem equation_roots_existence_and_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ x₁ x₂ : ℝ, 
    (1 / x₁ + 1 / (x₁ - a) + 1 / (x₁ + b) = 0) ∧
    (1 / x₂ + 1 / (x₂ - a) + 1 / (x₂ + b) = 0) ∧
    (a / 3 < x₁ ∧ x₁ < 2 * a / 3) ∧
    (-2 * b / 3 < x₂ ∧ x₂ < -b / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_existence_and_bounds_l3505_350557


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3505_350585

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define what it means for a point to be in the fourth quadrant
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- The point we want to prove is in the fourth quadrant
def point_to_check : Point := (1, -2)

-- Theorem statement
theorem point_in_fourth_quadrant : 
  is_in_fourth_quadrant point_to_check := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3505_350585


namespace NUMINAMATH_CALUDE_fish_tank_balls_l3505_350521

/-- The total number of balls in a fish tank with goldfish, platyfish, and angelfish -/
def total_balls (goldfish platyfish angelfish : ℕ) 
                (goldfish_balls platyfish_balls angelfish_balls : ℚ) : ℚ :=
  (goldfish : ℚ) * goldfish_balls + 
  (platyfish : ℚ) * platyfish_balls + 
  (angelfish : ℚ) * angelfish_balls

/-- Theorem stating the total number of balls in the fish tank -/
theorem fish_tank_balls : 
  total_balls 5 8 4 12.5 7.5 4.5 = 140.5 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_balls_l3505_350521


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l3505_350565

theorem integer_pairs_satisfying_equation :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 2) ∧ 
    s.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l3505_350565


namespace NUMINAMATH_CALUDE_tetrahedron_has_six_edges_l3505_350558

/-- A tetrahedron is a three-dimensional geometric shape with four triangular faces. -/
structure Tetrahedron where
  vertices : Finset (Fin 4)
  faces : Finset (Finset (Fin 3))
  is_valid : faces.card = 4 ∧ ∀ f ∈ faces, f.card = 3

/-- The number of edges in a tetrahedron -/
def num_edges (t : Tetrahedron) : ℕ := sorry

/-- Theorem: A tetrahedron has exactly 6 edges -/
theorem tetrahedron_has_six_edges (t : Tetrahedron) : num_edges t = 6 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_has_six_edges_l3505_350558


namespace NUMINAMATH_CALUDE_quotient_problem_l3505_350574

theorem quotient_problem (k : ℤ) (h : k = 4) : 16 / k = 4 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l3505_350574


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3505_350570

theorem arithmetic_calculation : 8 / 2 - 3 - 9 + 3 * 9 - 3^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3505_350570


namespace NUMINAMATH_CALUDE_line_segment_theorem_l3505_350555

/-- Represents a line segment on a straight line -/
structure LineSegment where
  left : ℝ
  right : ℝ
  h : left ≤ right

/-- Given a list of line segments, returns true if there exists a point common to at least n of them -/
def has_common_point (segments : List LineSegment) (n : ℕ) : Prop :=
  ∃ p : ℝ, (segments.filter (λ s => s.left ≤ p ∧ p ≤ s.right)).length ≥ n

/-- Given a list of line segments, returns true if there exist n pairwise disjoint segments -/
def has_disjoint_segments (segments : List LineSegment) (n : ℕ) : Prop :=
  ∃ disjoint : List LineSegment, disjoint.length = n ∧
    ∀ i j, i < j → j < disjoint.length →
      (disjoint.get ⟨i, by sorry⟩).right < (disjoint.get ⟨j, by sorry⟩).left

/-- The main theorem -/
theorem line_segment_theorem (segments : List LineSegment) 
    (h : segments.length = 50) :
    has_common_point segments 8 ∨ has_disjoint_segments segments 8 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_theorem_l3505_350555


namespace NUMINAMATH_CALUDE_circle_center_sum_l3505_350516

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 8*y + 13

/-- The center of a circle -/
def CircleCenter (h k : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 13) / 2

theorem circle_center_sum :
  ∀ h k : ℝ, CircleCenter h k CircleEquation → h + k = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3505_350516


namespace NUMINAMATH_CALUDE_max_sum_of_coeff_bound_l3505_350546

/-- A complex polynomial of degree 2 -/
def ComplexPoly (a b c : ℂ) : ℂ → ℂ := fun z ↦ a * z^2 + b * z + c

/-- The statement that |f(z)| ≤ 1 for all |z| ≤ 1 -/
def BoundedOnUnitDisk (f : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs z ≤ 1 → Complex.abs (f z) ≤ 1

/-- The main theorem -/
theorem max_sum_of_coeff_bound {a b c : ℂ} (h : BoundedOnUnitDisk (ComplexPoly a b c)) :
    Complex.abs a + Complex.abs b ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

#check max_sum_of_coeff_bound

end NUMINAMATH_CALUDE_max_sum_of_coeff_bound_l3505_350546


namespace NUMINAMATH_CALUDE_zeros_after_decimal_point_of_inverse_40_power_20_l3505_350530

theorem zeros_after_decimal_point_of_inverse_40_power_20 :
  let n : ℕ := 40
  let p : ℕ := 20
  let f : ℚ := 1 / (n^p : ℚ)
  (∃ (x : ℚ) (k : ℕ), f = x * 10^(-k : ℤ) ∧ x ≥ 1/10 ∧ x < 1 ∧ k = 38) :=
by sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_point_of_inverse_40_power_20_l3505_350530


namespace NUMINAMATH_CALUDE_jennifer_apples_l3505_350502

/-- The number of apples Jennifer started with -/
def initial_apples : ℕ := sorry

/-- The number of apples Jennifer found -/
def found_apples : ℕ := 74

/-- The total number of apples Jennifer ended up with -/
def total_apples : ℕ := 81

/-- Theorem stating that the initial number of apples plus the found apples equals the total apples -/
theorem jennifer_apples : initial_apples + found_apples = total_apples := by sorry

end NUMINAMATH_CALUDE_jennifer_apples_l3505_350502


namespace NUMINAMATH_CALUDE_abs_neg_three_l3505_350540

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_l3505_350540


namespace NUMINAMATH_CALUDE_quadratic_conditions_l3505_350550

/-- The quadratic function f(x) = x^2 - 4x - 3 + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x - 3 + a

/-- Theorem stating the conditions for the quadratic function -/
theorem quadratic_conditions :
  (∃ a : ℝ, f a 0 = 1 ∧ a = 4) ∧
  (∃ a : ℝ, (∀ x : ℝ, f a x = 0 → x = 0 ∨ x ≠ 0) ∧ (a = 3 ∨ a = 7)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_conditions_l3505_350550


namespace NUMINAMATH_CALUDE_min_abs_z_plus_i_l3505_350560

theorem min_abs_z_plus_i (z : ℂ) (h : Complex.abs (z^2 + 16) = Complex.abs (z * (z + 4*I))) :
  ∃ (w : ℂ), Complex.abs (w + I) = 3 ∧ ∀ (z : ℂ), Complex.abs (z^2 + 16) = Complex.abs (z * (z + 4*I)) → Complex.abs (z + I) ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_i_l3505_350560


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3505_350559

/-- Given two similar triangles with areas A₁ and A₂, where A₁ > A₂,
    prove that the corresponding side of the larger triangle is 12 feet. -/
theorem similar_triangles_side_length 
  (A₁ A₂ : ℝ) 
  (h_positive : A₁ > A₂) 
  (h_diff : A₁ - A₂ = 27) 
  (h_ratio : A₁ / A₂ = 9) 
  (h_small_side : ∃ (s : ℝ), s = 4 ∧ s * s / 2 ≤ A₂) : 
  ∃ (S : ℝ), S = 12 ∧ S * S / 2 ≤ A₁ := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3505_350559


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_l3505_350593

/-- Given a mixture of raisins and nuts with specific quantities and price ratios,
    prove that the cost of raisins is 1/4 of the total cost. -/
theorem raisin_cost_fraction (raisin_pounds almond_pounds cashew_pounds : ℕ) 
                              (raisin_price : ℚ) :
  raisin_pounds = 4 →
  almond_pounds = 3 →
  cashew_pounds = 2 →
  raisin_price > 0 →
  (raisin_pounds * raisin_price) / 
  (raisin_pounds * raisin_price + 
   almond_pounds * (2 * raisin_price) + 
   cashew_pounds * (3 * raisin_price)) = 1 / 4 := by
  sorry

#check raisin_cost_fraction

end NUMINAMATH_CALUDE_raisin_cost_fraction_l3505_350593


namespace NUMINAMATH_CALUDE_enchiladas_and_tacos_price_l3505_350533

-- Define the prices of enchiladas and tacos
noncomputable def enchilada_price : ℝ := sorry
noncomputable def taco_price : ℝ := sorry

-- Define the conditions
axiom condition1 : enchilada_price + 4 * taco_price = 3
axiom condition2 : 4 * enchilada_price + taco_price = 3.2

-- State the theorem
theorem enchiladas_and_tacos_price :
  4 * enchilada_price + 5 * taco_price = 5.55 := by sorry

end NUMINAMATH_CALUDE_enchiladas_and_tacos_price_l3505_350533


namespace NUMINAMATH_CALUDE_solve_equation_l3505_350590

theorem solve_equation (y : ℝ) : 7 - y = 12 ↔ y = -5 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3505_350590


namespace NUMINAMATH_CALUDE_perpendicular_polygon_perimeter_l3505_350597

/-- A polygon with adjacent sides perpendicular to each other -/
structure PerpendicularPolygon where
  a : ℝ  -- Sum of all vertical sides
  b : ℝ  -- Sum of all horizontal sides

/-- The perimeter of a perpendicular polygon -/
def perimeter (p : PerpendicularPolygon) : ℝ := 2 * (p.a + p.b)

/-- Theorem: The perimeter of a perpendicular polygon is 2(a+b) -/
theorem perpendicular_polygon_perimeter (p : PerpendicularPolygon) :
  perimeter p = 2 * (p.a + p.b) := by sorry

end NUMINAMATH_CALUDE_perpendicular_polygon_perimeter_l3505_350597


namespace NUMINAMATH_CALUDE_fraction_value_proof_l3505_350576

theorem fraction_value_proof (a b c : ℚ) (h1 : a = 5) (h2 : b = -3) (h3 : c = 4) :
  2 * c / (a + b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_proof_l3505_350576


namespace NUMINAMATH_CALUDE_base_prime_rep_441_l3505_350532

def base_prime_representation (n : ℕ) (primes : List ℕ) : List ℕ :=
  sorry

/-- The base prime representation of 441 using primes 2, 3, 5, and 7 is 0202 -/
theorem base_prime_rep_441 : 
  base_prime_representation 441 [2, 3, 5, 7] = [0, 2, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_rep_441_l3505_350532


namespace NUMINAMATH_CALUDE_line_equation_with_intercept_condition_l3505_350549

/-- The equation of a line passing through the intersection of two given lines,
    with its y-intercept being twice its x-intercept. -/
theorem line_equation_with_intercept_condition :
  ∃ (m b : ℝ),
    (∀ x y : ℝ, (2*x + y = 8 ∧ x - 2*y = -1) → (m*x + b = y)) ∧
    (2 * (b/m) = b) ∧
    ((m = 2 ∧ b = 0) ∨ (m = 2 ∧ b = -8)) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_with_intercept_condition_l3505_350549


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3505_350591

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- State the theorem
theorem complement_of_M_in_U : 
  Set.compl M = {x : ℝ | x < -2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3505_350591


namespace NUMINAMATH_CALUDE_triangle_inequality_l3505_350508

/-- Given a triangle with sides a, b, c and area S, prove that a^2 + b^2 + c^2 ≥ 4S√3,
    with equality if and only if the triangle is equilateral -/
theorem triangle_inequality (a b c S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0)
  (h_S : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 ∧
  (a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3 ↔ a = b ∧ b = c) := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequality_l3505_350508


namespace NUMINAMATH_CALUDE_decimal_to_binary_89_l3505_350551

theorem decimal_to_binary_89 :
  ∃ (b : List Bool),
    b.reverse.map (λ x => if x then 1 else 0) = [1, 0, 1, 1, 0, 0, 1] ∧
    b.foldr (λ x acc => 2 * acc + if x then 1 else 0) 0 = 89 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_89_l3505_350551


namespace NUMINAMATH_CALUDE_work_completion_time_l3505_350514

theorem work_completion_time 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (joint_work_days : ℕ) 
  (a_rate_def : a_rate = 1 / 5) 
  (b_rate_def : b_rate = 1 / 15) 
  (joint_work_days_def : joint_work_days = 2) : 
  (1 - (a_rate + b_rate) * joint_work_days) / b_rate = 7 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3505_350514


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3505_350523

/-- Theorem stating the relationship between k, a, m, and n for a parabola and line intersection -/
theorem parabola_line_intersection
  (a m n k b : ℝ)
  (ha : a ≠ 0)
  (h_intersect : ∃ (y₁ y₂ : ℝ),
    a * (1 - m) * (1 - n) = k * 1 + b ∧
    a * (6 - m) * (6 - n) = k * 6 + b) :
  k = a * (7 - m - n) := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3505_350523


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_theorem_l3505_350588

-- Define the theorem
theorem inequality_proof (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 - a*b) + Real.sqrt (b^2 + c^2 - b*c) ≥ Real.sqrt (a^2 + c^2 + a*c) := by
  sorry

-- Define the equality condition
def equality_condition (a b c : ℝ) : Prop :=
  (a * c = a * b + b * c) ∧ (a * b + a * c + b * c - 2 * b^2 ≥ 0)

-- Theorem for the equality condition
theorem equality_condition_theorem (a b c : ℝ) :
  (Real.sqrt (a^2 + b^2 - a*b) + Real.sqrt (b^2 + c^2 - b*c) = Real.sqrt (a^2 + c^2 + a*c)) ↔
  equality_condition a b c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_theorem_l3505_350588


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l3505_350526

theorem sum_of_fractions_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + c) / (a + b) + (b + d) / (b + c) + (c + a) / (c + d) + (d + b) / (d + a) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l3505_350526


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3505_350548

theorem trigonometric_identity :
  1 / Real.cos (70 * π / 180) - 2 / Real.sin (70 * π / 180) = 
  4 * Real.sin (10 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3505_350548


namespace NUMINAMATH_CALUDE_new_savings_amount_l3505_350517

def monthly_salary : ℕ := 6500
def initial_savings_rate : ℚ := 1/5
def expense_increase_rate : ℚ := 1/5

theorem new_savings_amount :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_expenses * expense_increase_rate
  let new_expenses := initial_expenses + expense_increase
  let new_savings := monthly_salary - new_expenses
  new_savings = 260 := by sorry

end NUMINAMATH_CALUDE_new_savings_amount_l3505_350517


namespace NUMINAMATH_CALUDE_xt_ty_ratio_is_one_l3505_350501

/-- Represents the shape described in the problem -/
structure Shape :=
  (total_squares : ℕ)
  (rectangle_squares : ℕ)
  (terrace_rows : ℕ)
  (terrace_squares_per_row : ℕ)

/-- Represents a line segment -/
structure LineSegment :=
  (length : ℝ)

/-- The problem setup -/
def problem_setup : Shape :=
  { total_squares := 12,
    rectangle_squares := 6,
    terrace_rows := 2,
    terrace_squares_per_row := 3 }

/-- The line RS that bisects the area horizontally -/
def RS : LineSegment :=
  { length := 6 }

/-- Theorem stating the ratio XT/TY = 1 -/
theorem xt_ty_ratio_is_one (shape : Shape) (rs : LineSegment) 
  (h1 : shape = problem_setup)
  (h2 : rs = RS)
  (h3 : rs.length = shape.total_squares / 2) :
  ∃ (xt ty : ℝ), xt = ty ∧ xt + ty = rs.length ∧ xt / ty = 1 :=
sorry

end NUMINAMATH_CALUDE_xt_ty_ratio_is_one_l3505_350501


namespace NUMINAMATH_CALUDE_reinforcement_arrival_days_l3505_350595

/-- Calculates the number of days that passed before reinforcement arrived -/
def days_before_reinforcement (initial_garrison : ℕ) (initial_provision_days : ℕ) 
  (reinforcement_size : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_garrison := initial_garrison + reinforcement_size
  let x := (initial_garrison * initial_provision_days - total_garrison * remaining_days) / initial_garrison
  x

/-- Theorem stating that 15 days passed before reinforcement arrived -/
theorem reinforcement_arrival_days :
  days_before_reinforcement 2000 62 2700 20 = 15 := by
  sorry

#eval days_before_reinforcement 2000 62 2700 20

end NUMINAMATH_CALUDE_reinforcement_arrival_days_l3505_350595


namespace NUMINAMATH_CALUDE_right_triangle_sin_value_l3505_350592

theorem right_triangle_sin_value (A B C : Real) (h1 : 0 < A) (h2 : A < π / 2) :
  (Real.cos B = 0) →
  (3 * Real.sin A = 4 * Real.cos A) →
  Real.sin A = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_value_l3505_350592


namespace NUMINAMATH_CALUDE_complement_of_A_l3505_350541

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {3, 5}

theorem complement_of_A : (U \ A) = {1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3505_350541


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l3505_350515

theorem max_value_theorem (x y : ℝ) :
  (3 * x + 4 * y + 6) / Real.sqrt (x^2 + 4 * y^2 + 4) ≤ Real.sqrt 61 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (3 * x + 4 * y + 6) / Real.sqrt (x^2 + 4 * y^2 + 4) = Real.sqrt 61 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l3505_350515


namespace NUMINAMATH_CALUDE_max_vertex_coordinate_sum_l3505_350544

/-- Given a parabola y = ax^2 + bx + c passing through (0,0), (2T,0), and (2T+1,35),
    where a and T are integers and T ≠ 0, the maximum sum of vertex coordinates is 34. -/
theorem max_vertex_coordinate_sum :
  ∀ (a T : ℤ) (b c : ℝ),
    T ≠ 0 →
    (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ 
      (x = 0 ∧ y = 0) ∨ 
      (x = 2 * T ∧ y = 0) ∨ 
      (x = 2 * T + 1 ∧ y = 35)) →
    (∃ (N : ℝ), N = T - a * T^2 ∧ 
      (∀ (N' : ℝ), (∃ (a' T' : ℤ) (b' c' : ℝ),
        T' ≠ 0 ∧
        (∀ x y : ℝ, y = a' * x^2 + b' * x + c' ↔ 
          (x = 0 ∧ y = 0) ∨ 
          (x = 2 * T' ∧ y = 0) ∨ 
          (x = 2 * T' + 1 ∧ y = 35)) ∧
        N' = T' - a' * T'^2) → N' ≤ N)) →
    (∃ (N : ℝ), N = 34 ∧
      (∀ (N' : ℝ), (∃ (a' T' : ℤ) (b' c' : ℝ),
        T' ≠ 0 ∧
        (∀ x y : ℝ, y = a' * x^2 + b' * x + c' ↔ 
          (x = 0 ∧ y = 0) ∨ 
          (x = 2 * T' ∧ y = 0) ∨ 
          (x = 2 * T' + 1 ∧ y = 35)) ∧
        N' = T' - a' * T'^2) → N' ≤ N)) :=
by sorry

end NUMINAMATH_CALUDE_max_vertex_coordinate_sum_l3505_350544


namespace NUMINAMATH_CALUDE_f_sin_A_lt_f_cos_B_l3505_350506

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) + f x = 0

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem f_sin_A_lt_f_cos_B
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_periodic : is_periodic_2 f)
  (h_increasing : is_increasing_on f 3 4)
  (A B : ℝ)
  (h_acute_A : 0 < A ∧ A < Real.pi / 2)
  (h_acute_B : 0 < B ∧ B < Real.pi / 2) :
  f (Real.sin A) < f (Real.cos B) :=
sorry

end NUMINAMATH_CALUDE_f_sin_A_lt_f_cos_B_l3505_350506


namespace NUMINAMATH_CALUDE_intersection_area_is_zero_l3505_350547

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three vertices -/
structure Triangle :=
  (v1 : Point)
  (v2 : Point)
  (v3 : Point)

/-- Calculate the area of intersection between two triangles -/
def areaOfIntersection (t1 t2 : Triangle) : ℝ := sorry

/-- The main theorem stating that the area of intersection is zero -/
theorem intersection_area_is_zero :
  let t1 := Triangle.mk (Point.mk 0 2) (Point.mk 2 1) (Point.mk 0 0)
  let t2 := Triangle.mk (Point.mk 2 2) (Point.mk 0 1) (Point.mk 2 0)
  areaOfIntersection t1 t2 = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_zero_l3505_350547


namespace NUMINAMATH_CALUDE_length_of_AB_is_8_l3505_350578

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = -4*x ∧ x < 0

-- Define point P
def P : ℝ × ℝ := (-3, -2)

-- Define the line l passing through P
def l (x y : ℝ) : Prop := y + 2 = x + 3

-- Define the property of P being the midpoint of AB
def is_midpoint (A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Main theorem
theorem length_of_AB_is_8 :
  ∀ A B : ℝ × ℝ,
  C A.1 A.2 → C B.1 B.2 →
  l A.1 A.2 → l B.1 B.2 →
  is_midpoint A B →
  ‖A - B‖ = 8 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_is_8_l3505_350578


namespace NUMINAMATH_CALUDE_pebble_distribution_correct_l3505_350539

/-- The number of friends who received pebbles from Janice --/
def num_friends : ℕ := 17

/-- The total weight of pebbles in grams --/
def total_weight : ℕ := 36000

/-- The weight of a small pebble in grams --/
def small_pebble_weight : ℕ := 200

/-- The weight of a large pebble in grams --/
def large_pebble_weight : ℕ := 300

/-- The number of small pebbles given to each friend --/
def small_pebbles_per_friend : ℕ := 3

/-- The number of large pebbles given to each friend --/
def large_pebbles_per_friend : ℕ := 5

/-- Theorem stating that the number of friends who received pebbles is correct --/
theorem pebble_distribution_correct : 
  num_friends * (small_pebbles_per_friend * small_pebble_weight + 
                 large_pebbles_per_friend * large_pebble_weight) ≤ total_weight ∧
  (num_friends + 1) * (small_pebbles_per_friend * small_pebble_weight + 
                       large_pebbles_per_friend * large_pebble_weight) > total_weight :=
by sorry

end NUMINAMATH_CALUDE_pebble_distribution_correct_l3505_350539


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3505_350513

theorem min_value_sum_reciprocals (n : ℕ) (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
  (1 / (1 + a^n)) + (1 / (1 + b^n)) ≥ 1 ∧ 
  ((1 / (1 + a^n)) + (1 / (1 + b^n)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

#check min_value_sum_reciprocals

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3505_350513


namespace NUMINAMATH_CALUDE_car_part_payment_l3505_350554

theorem car_part_payment (remaining_payment : ℝ) (part_payment_percentage : ℝ) 
  (h1 : remaining_payment = 5700)
  (h2 : part_payment_percentage = 0.05) : 
  (remaining_payment / (1 - part_payment_percentage)) * part_payment_percentage = 300 := by
  sorry

end NUMINAMATH_CALUDE_car_part_payment_l3505_350554


namespace NUMINAMATH_CALUDE_cube_shadow_problem_l3505_350569

theorem cube_shadow_problem (x : ℝ) : 
  let cube_edge : ℝ := 2
  let shadow_area : ℝ := 300
  let total_shadow_area : ℝ := shadow_area + cube_edge^2
  let shadow_side : ℝ := Real.sqrt total_shadow_area
  x = (cube_edge / (shadow_side - cube_edge)) →
  ⌊1000 * x⌋ = 706 := by
sorry

end NUMINAMATH_CALUDE_cube_shadow_problem_l3505_350569


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3505_350537

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (3*m + 2)*(-1) - (2*m - 1)*1 + 5*m + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3505_350537


namespace NUMINAMATH_CALUDE_weight_of_b_l3505_350561

theorem weight_of_b (a b c d : ℝ) 
  (h1 : (a + b + c + d) / 4 = 40)
  (h2 : (a + b) / 2 = 25)
  (h3 : (b + c) / 2 = 28)
  (h4 : (c + d) / 2 = 32) :
  b = 46 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l3505_350561


namespace NUMINAMATH_CALUDE_mary_stickers_l3505_350562

theorem mary_stickers (front_page : ℕ) (other_pages : ℕ) (stickers_per_page : ℕ) (remaining : ℕ) :
  front_page = 3 →
  other_pages = 6 →
  stickers_per_page = 7 →
  remaining = 44 →
  front_page + (other_pages * stickers_per_page) + remaining = 89 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_stickers_l3505_350562


namespace NUMINAMATH_CALUDE_min_midpoint_for_transformed_sine_l3505_350568

theorem min_midpoint_for_transformed_sine (f g : ℝ → ℝ) (x₁ x₂ : ℝ) :
  (∀ x, f x = Real.sin (x + π/3)) →
  (∀ x, g x = Real.sin (2*x + π/3)) →
  (x₁ ≠ x₂) →
  (g x₁ * g x₂ = -1) →
  (∃ m, m = |(x₁ + x₂)/2| ∧ ∀ y₁ y₂, y₁ ≠ y₂ → g y₁ * g y₂ = -1 → m ≤ |(y₁ + y₂)/2|) →
  |(x₁ + x₂)/2| = π/6 :=
by sorry

end NUMINAMATH_CALUDE_min_midpoint_for_transformed_sine_l3505_350568


namespace NUMINAMATH_CALUDE_probability_within_four_rings_l3505_350545

def P_first_ring : ℚ := 1 / 10
def P_second_ring : ℚ := 3 / 10
def P_third_ring : ℚ := 2 / 5
def P_fourth_ring : ℚ := 1 / 10

theorem probability_within_four_rings :
  P_first_ring + P_second_ring + P_third_ring + P_fourth_ring = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_within_four_rings_l3505_350545


namespace NUMINAMATH_CALUDE_train_speed_problem_l3505_350580

/-- Proves that given the conditions of the train problem, the speeds of the slower and faster trains are 60 km/hr and 70 km/hr respectively. -/
theorem train_speed_problem (distance : ℝ) (time : ℝ) (speed_diff : ℝ) (remaining_distance : ℝ)
  (h1 : distance = 300)
  (h2 : time = 2)
  (h3 : speed_diff = 10)
  (h4 : remaining_distance = 40) :
  ∃ (v1 v2 : ℝ), v1 = 60 ∧ v2 = 70 ∧ v2 = v1 + speed_diff ∧
  distance - remaining_distance = (v1 + v2) * time :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3505_350580


namespace NUMINAMATH_CALUDE_train_meeting_point_l3505_350507

/-- Two trains moving towards each other on a bridge --/
theorem train_meeting_point 
  (bridge_length : ℝ) 
  (train_a_speed : ℝ) 
  (train_b_speed : ℝ) 
  (h1 : bridge_length = 9000) 
  (h2 : train_a_speed = 15) 
  (h3 : train_b_speed = train_a_speed) :
  ∃ (meeting_time meeting_point : ℝ),
    meeting_time = 300 ∧ 
    meeting_point = bridge_length / 2 ∧
    meeting_point = train_a_speed * meeting_time :=
by sorry

end NUMINAMATH_CALUDE_train_meeting_point_l3505_350507


namespace NUMINAMATH_CALUDE_new_person_weight_l3505_350556

def group_weight_change (initial_count : ℕ) (leaving_weight : ℝ) (average_increase : ℝ) : ℝ :=
  let final_count : ℕ := initial_count
  let intermediate_count : ℕ := initial_count - 1
  (final_count : ℝ) * average_increase + leaving_weight

theorem new_person_weight 
  (initial_count : ℕ) 
  (leaving_weight : ℝ) 
  (average_increase : ℝ) 
  (h1 : initial_count = 15) 
  (h2 : leaving_weight = 90) 
  (h3 : average_increase = 3.7) : 
  group_weight_change initial_count leaving_weight average_increase = 55.5 := by
sorry

#eval group_weight_change 15 90 3.7

end NUMINAMATH_CALUDE_new_person_weight_l3505_350556


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_2x_plus_y_eq_7_l3505_350504

def is_solution (x y : ℕ) : Prop := 2 * x + y = 7

theorem positive_integer_solutions_of_2x_plus_y_eq_7 :
  {(x, y) : ℕ × ℕ | is_solution x y ∧ x > 0 ∧ y > 0} = {(1, 5), (2, 3), (3, 1)} := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_2x_plus_y_eq_7_l3505_350504


namespace NUMINAMATH_CALUDE_june_songs_total_l3505_350510

def songs_in_june (vivian_daily : ℕ) (clara_difference : ℕ) (total_days : ℕ) (weekend_days : ℕ) : ℕ :=
  let weekdays := total_days - weekend_days
  let vivian_total := vivian_daily * weekdays
  let clara_daily := vivian_daily - clara_difference
  let clara_total := clara_daily * weekdays
  vivian_total + clara_total

theorem june_songs_total :
  songs_in_june 10 2 30 8 = 396 := by
  sorry

end NUMINAMATH_CALUDE_june_songs_total_l3505_350510


namespace NUMINAMATH_CALUDE_max_sum_squares_sides_l3505_350543

/-- For any acute-angled triangle with side length a and angle α, 
    the sum of squares of the other two side lengths (b² + c²) 
    is less than or equal to a² / (2 sin²(α/2)). -/
theorem max_sum_squares_sides (a : ℝ) (α : ℝ) (h_acute : 0 < α ∧ α < π / 2) :
  ∀ b c : ℝ, 
  (0 < b ∧ 0 < c) → -- Ensure positive side lengths
  (b^2 + c^2 - 2*b*c*Real.cos α = a^2) → -- Cosine rule
  b^2 + c^2 ≤ a^2 / (2 * Real.sin (α/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squares_sides_l3505_350543


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l3505_350567

theorem complex_on_imaginary_axis (a : ℝ) : ∃ y : ℝ, (a + I) * (1 + a * I) = y * I := by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l3505_350567


namespace NUMINAMATH_CALUDE_definite_integral_quarter_circle_l3505_350587

theorem definite_integral_quarter_circle (f : ℝ → ℝ) :
  (∫ x in (0 : ℝ)..(Real.sqrt 2), f x) = π / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_definite_integral_quarter_circle_l3505_350587


namespace NUMINAMATH_CALUDE_construct_square_and_dodecagon_l3505_350599

/-- A point in a 2D plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in a 2D plane --/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a compass --/
structure Compass :=
  (create_circle : Point → ℝ → Circle)

/-- Represents a square --/
structure Square :=
  (vertices : Fin 4 → Point)

/-- Represents a regular dodecagon --/
structure RegularDodecagon :=
  (vertices : Fin 12 → Point)

/-- Theorem stating that a square and a regular dodecagon can be constructed using only a compass --/
theorem construct_square_and_dodecagon 
  (A B : Point) 
  (compass : Compass) : 
  ∃ (square : Square) (dodecagon : RegularDodecagon),
    (square.vertices 0 = A ∧ square.vertices 1 = B) ∧
    (dodecagon.vertices 0 = A ∧ dodecagon.vertices 1 = B) :=
sorry

end NUMINAMATH_CALUDE_construct_square_and_dodecagon_l3505_350599


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3505_350527

theorem quadratic_root_problem (a : ℝ) : 
  (3 : ℝ)^2 - (a + 2) * 3 + 2 * a = 0 → 
  ∃ x : ℝ, x^2 - (a + 2) * x + 2 * a = 0 ∧ x ≠ 3 ∧ x = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3505_350527


namespace NUMINAMATH_CALUDE_circle_tangency_distance_ratio_l3505_350579

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the distance function
variable (dist : Point → Point → ℝ)

-- Define the four circles
variable (A₁ A₂ A₃ A₄ : Circle)

-- Define the points
variable (P T₁ T₂ T₃ T₄ : Point)

-- Define the tangency and intersection relations
variable (tangent : Circle → Circle → Point → Prop)
variable (intersect : Circle → Circle → Point → Prop)

-- State the theorem
theorem circle_tangency_distance_ratio
  (h1 : tangent A₁ A₃ P)
  (h2 : tangent A₂ A₄ P)
  (h3 : intersect A₁ A₂ T₁)
  (h4 : intersect A₂ A₃ T₂)
  (h5 : intersect A₃ A₄ T₃)
  (h6 : intersect A₄ A₁ T₄)
  (h7 : T₁ ≠ P ∧ T₂ ≠ P ∧ T₃ ≠ P ∧ T₄ ≠ P) :
  (dist T₁ T₂ * dist T₂ T₃) / (dist T₁ T₄ * dist T₃ T₄) = (dist P T₂)^2 / (dist P T₄)^2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangency_distance_ratio_l3505_350579


namespace NUMINAMATH_CALUDE_parabola_line_intersection_ratio_l3505_350525

/-- Parabola type -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Line passing through a point with given slope angle -/
structure Line where
  slope_angle : ℝ
  point : ℝ × ℝ

/-- Intersection points of a line and a parabola -/
structure Intersection where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Theorem stating the ratio of distances from intersection points to focus -/
theorem parabola_line_intersection_ratio
  (C : Parabola)
  (l : Line)
  (i : Intersection)
  (h1 : l.slope_angle = π / 3) -- 60 degrees in radians
  (h2 : l.point = (C.p / 2, 0)) -- Focus of the parabola
  (h3 : i.A.1 > 0 ∧ i.A.2 > 0) -- A in first quadrant
  (h4 : i.B.1 > 0 ∧ i.B.2 < 0) -- B in fourth quadrant
  (h5 : i.A.2^2 = 2 * C.p * i.A.1) -- A satisfies parabola equation
  (h6 : i.B.2^2 = 2 * C.p * i.B.1) -- B satisfies parabola equation
  (h7 : i.A.2 - 0 = Real.sqrt 3 * (i.A.1 - C.p / 2)) -- A satisfies line equation
  (h8 : i.B.2 - 0 = Real.sqrt 3 * (i.B.1 - C.p / 2)) -- B satisfies line equation
  : (Real.sqrt ((i.A.1 - C.p / 2)^2 + i.A.2^2)) / (Real.sqrt ((i.B.1 - C.p / 2)^2 + i.B.2^2)) = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_ratio_l3505_350525


namespace NUMINAMATH_CALUDE_current_rate_calculation_l3505_350572

/-- Given a boat with speed in still water and its downstream travel distance and time,
    calculate the rate of the current. -/
theorem current_rate_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (travel_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 6.25 →
  travel_time = 0.25 →
  ∃ (current_rate : ℝ),
    current_rate = 5 ∧
    downstream_distance = (boat_speed + current_rate) * travel_time :=
by
  sorry


end NUMINAMATH_CALUDE_current_rate_calculation_l3505_350572


namespace NUMINAMATH_CALUDE_greatest_common_factor_372_72_under_50_l3505_350552

def is_greatest_common_factor (n : ℕ) : Prop :=
  n ∣ 372 ∧ n < 50 ∧ n ∣ 72 ∧
  ∀ m : ℕ, m ∣ 372 → m < 50 → m ∣ 72 → m ≤ n

theorem greatest_common_factor_372_72_under_50 :
  is_greatest_common_factor 12 := by
sorry

end NUMINAMATH_CALUDE_greatest_common_factor_372_72_under_50_l3505_350552


namespace NUMINAMATH_CALUDE_double_division_remainder_l3505_350581

def p (x : ℝ) : ℝ := x^10

def q1 (x : ℝ) : ℝ := 
  x^9 + 2*x^8 + 4*x^7 + 8*x^6 + 16*x^5 + 32*x^4 + 64*x^3 + 128*x^2 + 256*x + 512

theorem double_division_remainder (x : ℝ) : 
  ∃ (q2 : ℝ → ℝ) (r2 : ℝ), p x = (x - 2) * ((x - 2) * q2 x + q1 2) + r2 ∧ r2 = 5120 := by
  sorry

end NUMINAMATH_CALUDE_double_division_remainder_l3505_350581


namespace NUMINAMATH_CALUDE_max_value_of_trig_expression_l3505_350534

theorem max_value_of_trig_expression :
  ∀ α : Real, 0 ≤ α ∧ α ≤ π / 2 →
    (∀ β : Real, 0 ≤ β ∧ β ≤ π / 2 → 
      1 / (Real.sin β ^ 6 + Real.cos β ^ 6) ≤ 1 / (Real.sin α ^ 6 + Real.cos α ^ 6)) →
    1 / (Real.sin α ^ 6 + Real.cos α ^ 6) = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_expression_l3505_350534


namespace NUMINAMATH_CALUDE_company_bonus_problem_l3505_350584

/-- Represents the company bonus distribution problem -/
theorem company_bonus_problem (n : ℕ) 
  (h1 : 60 * n - 15 = 45 * n + 135) : 
  60 * n - 15 = 585 := by
  sorry

#check company_bonus_problem

end NUMINAMATH_CALUDE_company_bonus_problem_l3505_350584


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3505_350529

/-- The function representing y = 2x^2 --/
def f (x : ℝ) : ℝ := 2 * x^2

/-- The function representing y = 4x + c --/
def g (c : ℝ) (x : ℝ) : ℝ := 4 * x + c

/-- The condition for two identical solutions --/
def has_two_identical_solutions (c : ℝ) : Prop :=
  ∃! x : ℝ, f x = g c x ∧ ∀ y : ℝ, f y = g c y → y = x

theorem unique_solution_condition (c : ℝ) :
  has_two_identical_solutions c ↔ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3505_350529


namespace NUMINAMATH_CALUDE_checkerboard_coverage_l3505_350505

/-- A checkerboard is a rectangular grid of squares. -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ
  missing_squares : ℕ

/-- A domino covers exactly two adjacent squares. -/
def domino_area : ℕ := 2

/-- The total number of squares in a checkerboard. -/
def total_squares (board : Checkerboard) : ℕ :=
  board.rows * board.cols - board.missing_squares

/-- A checkerboard can be completely covered by dominoes if and only if
    it has an even number of squares. -/
theorem checkerboard_coverage (board : Checkerboard) :
  ∃ (n : ℕ), total_squares board = n * domino_area ↔ Even (total_squares board) := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_l3505_350505


namespace NUMINAMATH_CALUDE_value_of_a_l3505_350531

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 8) 
  (eq3 : c = 4) : 
  a = 0 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l3505_350531


namespace NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_power_units_digit_of_expression_l3505_350524

theorem units_digit_of_sum (a b : ℕ) : ∃ (x y : ℕ), 
  x = a % 10 ∧ 
  y = b % 10 ∧ 
  (a + b) % 10 = (x + y) % 10 :=
by sorry

theorem units_digit_of_power (base exp : ℕ) : 
  (base ^ exp) % 10 = (base % 10 ^ (exp % 4 + 4)) % 10 :=
by sorry

theorem units_digit_of_expression : (5^12 + 4^2) % 10 = 1 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_power_units_digit_of_expression_l3505_350524


namespace NUMINAMATH_CALUDE_eighteen_tons_equals_18000kg_l3505_350564

-- Define the conversion factor between tons and kilograms
def tons_to_kg (t : ℝ) : ℝ := 1000 * t

-- Theorem statement
theorem eighteen_tons_equals_18000kg : tons_to_kg 18 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_tons_equals_18000kg_l3505_350564


namespace NUMINAMATH_CALUDE_total_chocolates_in_month_l3505_350566

/-- Represents the number of chocolates Kantana buys for herself each Saturday -/
def self_chocolates_per_saturday : ℕ := 2

/-- Represents the number of chocolates Kantana buys for her sister each Saturday -/
def sister_chocolates_per_saturday : ℕ := 1

/-- Represents the number of Saturdays in a month -/
def saturdays_in_month : ℕ := 4

/-- Represents the number of chocolates Kantana bought for her friend Charlie -/
def charlie_chocolates : ℕ := 10

/-- Theorem stating the total number of chocolates Kantana bought in a month -/
theorem total_chocolates_in_month : 
  self_chocolates_per_saturday * saturdays_in_month + 
  sister_chocolates_per_saturday * saturdays_in_month + 
  charlie_chocolates = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_chocolates_in_month_l3505_350566


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3505_350575

theorem cubic_roots_sum (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ t : ℝ, t^3 - 9*t^2 + a*t - b = 0 ↔ t = x ∨ t = y ∨ t = z)) →
  a + b = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3505_350575


namespace NUMINAMATH_CALUDE_black_blue_difference_l3505_350553

/-- Represents Sam's pen collection -/
structure PenCollection where
  black : ℕ
  blue : ℕ
  red : ℕ
  pencils : ℕ

/-- Conditions for Sam's pen collection -/
def validCollection (c : PenCollection) : Prop :=
  c.black > c.blue ∧
  c.blue = 2 * c.pencils ∧
  c.pencils = 8 ∧
  c.red = c.pencils - 2 ∧
  c.black + c.blue + c.red = 48

/-- Theorem stating the difference between black and blue pens -/
theorem black_blue_difference (c : PenCollection) 
  (h : validCollection c) : c.black - c.blue = 10 := by
  sorry


end NUMINAMATH_CALUDE_black_blue_difference_l3505_350553


namespace NUMINAMATH_CALUDE_sector_area_from_arc_length_l3505_350500

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4 cm,
    prove that the area of the sector formed by this central angle is 4 cm². -/
theorem sector_area_from_arc_length (s : ℝ) (θ : ℝ) (r : ℝ) (A : ℝ) 
    (h1 : s = 4)
    (h2 : θ = 2)
    (h3 : s = r * θ)
    (h4 : A = 1/2 * r^2 * θ) : A = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_from_arc_length_l3505_350500


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l3505_350511

theorem quadratic_function_inequality (a b c : ℝ) (h1 : c > b) (h2 : b > a) 
  (h3 : a * 1^2 + 2 * b * 1 + c = 0) 
  (h4 : ∃ x, a * x^2 + 2 * b * x + c = -a) : 
  0 ≤ b / a ∧ b / a < 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l3505_350511


namespace NUMINAMATH_CALUDE_total_video_time_l3505_350589

def cat_video_length : ℕ := 4

def dog_video_length (cat : ℕ) : ℕ := 2 * cat

def gorilla_video_length (cat : ℕ) : ℕ := cat ^ 2

def elephant_video_length (cat dog gorilla : ℕ) : ℕ := cat + dog + gorilla

def penguin_video_length (cat dog gorilla elephant : ℕ) : ℕ := (cat + dog + gorilla + elephant) ^ 3

def dolphin_video_length (cat dog gorilla elephant penguin : ℕ) : ℕ :=
  cat + dog + gorilla + elephant + penguin

theorem total_video_time :
  let cat := cat_video_length
  let dog := dog_video_length cat
  let gorilla := gorilla_video_length cat
  let elephant := elephant_video_length cat dog gorilla
  let penguin := penguin_video_length cat dog gorilla elephant
  let dolphin := dolphin_video_length cat dog gorilla elephant penguin
  cat + dog + gorilla + elephant + penguin + dolphin = 351344 := by
  sorry

end NUMINAMATH_CALUDE_total_video_time_l3505_350589


namespace NUMINAMATH_CALUDE_total_amount_proof_l3505_350538

/-- The total amount of money shared by Debby, Maggie, and Alex -/
def total : ℝ := 22500

/-- Debby's share percentage -/
def debby_share : ℝ := 0.30

/-- Maggie's share percentage -/
def maggie_share : ℝ := 0.40

/-- Alex's share percentage -/
def alex_share : ℝ := 0.30

/-- Maggie's actual share amount -/
def maggie_amount : ℝ := 9000

theorem total_amount_proof :
  maggie_share * total = maggie_amount ∧
  debby_share + maggie_share + alex_share = 1 :=
sorry

end NUMINAMATH_CALUDE_total_amount_proof_l3505_350538


namespace NUMINAMATH_CALUDE_equal_goldfish_after_six_months_l3505_350586

/-- Number of goldfish Brent has after n months -/
def brent_goldfish (n : ℕ) : ℕ := 2 * 4^n

/-- Number of goldfish Gretel has after n months -/
def gretel_goldfish (n : ℕ) : ℕ := 162 * 3^n

/-- The number of months it takes for Brent and Gretel to have the same number of goldfish -/
def months_to_equal_goldfish : ℕ := 6

/-- Theorem stating that after 'months_to_equal_goldfish' months, 
    Brent and Gretel have the same number of goldfish -/
theorem equal_goldfish_after_six_months : 
  brent_goldfish months_to_equal_goldfish = gretel_goldfish months_to_equal_goldfish :=
by sorry

end NUMINAMATH_CALUDE_equal_goldfish_after_six_months_l3505_350586


namespace NUMINAMATH_CALUDE_parabola_vertex_l3505_350542

/-- The vertex of the parabola y = -(x-1)^2 + 4 is (1,4) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -(x - 1)^2 + 4 → (1, 4) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3505_350542


namespace NUMINAMATH_CALUDE_f_at_2_l3505_350503

-- Define the polynomial function
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

-- Theorem statement
theorem f_at_2 : f 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l3505_350503


namespace NUMINAMATH_CALUDE_largest_number_proof_l3505_350594

theorem largest_number_proof (w x y z : ℕ) : 
  w + x + y = 190 ∧ 
  w + x + z = 210 ∧ 
  w + y + z = 220 ∧ 
  x + y + z = 235 → 
  max w (max x (max y z)) = 95 := by
sorry

end NUMINAMATH_CALUDE_largest_number_proof_l3505_350594
