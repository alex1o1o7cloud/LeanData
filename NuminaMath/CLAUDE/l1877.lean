import Mathlib

namespace NUMINAMATH_CALUDE_spanning_rectangles_odd_l1877_187792

/-- Represents a 2 × 1 rectangle used to cover the cube surface -/
structure Rectangle :=
  (spans_two_faces : Bool)

/-- Represents the surface of a 9 × 9 × 9 cube -/
structure CubeSurface :=
  (side_length : Nat)
  (covering : List Rectangle)

/-- Axiom: The cube is 9 × 9 × 9 -/
axiom cube_size : ∀ (c : CubeSurface), c.side_length = 9

/-- Axiom: The surface is completely covered without gaps or overlaps -/
axiom complete_coverage : ∀ (c : CubeSurface), c.covering.length * 2 = 6 * c.side_length^2

/-- Main theorem: The number of rectangles spanning two faces is odd -/
theorem spanning_rectangles_odd (c : CubeSurface) : 
  Odd (c.covering.filter Rectangle.spans_two_faces).length :=
sorry

end NUMINAMATH_CALUDE_spanning_rectangles_odd_l1877_187792


namespace NUMINAMATH_CALUDE_eighteenth_decimal_is_nine_l1877_187784

/-- Represents the decimal expansion of a fraction -/
def DecimalExpansion := ℕ → Fin 10

/-- The decimal expansion of 10/11 -/
def decimal_expansion_10_11 : DecimalExpansion :=
  fun n => if n % 2 = 0 then 0 else 9

theorem eighteenth_decimal_is_nine
  (h : ∀ n : ℕ, decimal_expansion_10_11 (20 - n) = 9 → decimal_expansion_10_11 n = 9) :
  decimal_expansion_10_11 18 = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_eighteenth_decimal_is_nine_l1877_187784


namespace NUMINAMATH_CALUDE_or_equivalence_l1877_187793

-- Define the propositions
variable (p : Prop)  -- Athlete A's trial jump exceeded 2 meters
variable (q : Prop)  -- Athlete B's trial jump exceeded 2 meters

-- Define the statement "At least one of Athlete A or B exceeded 2 meters in their trial jump"
def atLeastOneExceeded (p q : Prop) : Prop :=
  p ∨ q

-- Theorem stating the equivalence
theorem or_equivalence :
  (p ∨ q) ↔ atLeastOneExceeded p q :=
sorry

end NUMINAMATH_CALUDE_or_equivalence_l1877_187793


namespace NUMINAMATH_CALUDE_shirt_total_price_l1877_187750

/-- The total price of 25 shirts given the conditions in the problem -/
theorem shirt_total_price : 
  ∀ (shirt_price sweater_price : ℝ),
  75 * sweater_price = 1500 →
  sweater_price = shirt_price + 4 →
  25 * shirt_price = 400 := by
    sorry

end NUMINAMATH_CALUDE_shirt_total_price_l1877_187750


namespace NUMINAMATH_CALUDE_right_isosceles_not_scalene_l1877_187781

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  sum_angles : angle_A + angle_B + angle_C = 180
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

def IsRightTriangle (t : Triangle) : Prop :=
  t.angle_C = 90

def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

def IsScalene (t : Triangle) : Prop :=
  t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.c ≠ t.a

def IsRightIsosceles (t : Triangle) : Prop :=
  IsRightTriangle t ∧ IsIsosceles t

theorem right_isosceles_not_scalene :
  ∀ t : Triangle, IsRightIsosceles t → ¬IsScalene t :=
by
  sorry


end NUMINAMATH_CALUDE_right_isosceles_not_scalene_l1877_187781


namespace NUMINAMATH_CALUDE_binomial_sum_mod_three_l1877_187727

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  if 0 ≤ k ∧ k ≤ n then
    Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  else
    0

/-- Sum of binomial coefficients with step 6 -/
def sumBinomialStep6 (n : ℕ) : ℕ :=
  Finset.sum (Finset.range ((n - 1) / 6 + 1)) (fun i => binomial n (6 * i + 1))

theorem binomial_sum_mod_three (n : ℕ) (h : n = 9002) :
  sumBinomialStep6 n % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_mod_three_l1877_187727


namespace NUMINAMATH_CALUDE_total_cards_l1877_187745

-- Define the number of people
def num_people : ℕ := 4

-- Define the number of cards each person has
def cards_per_person : ℕ := 14

-- Theorem: The total number of cards is 56
theorem total_cards : num_people * cards_per_person = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l1877_187745


namespace NUMINAMATH_CALUDE_train_length_l1877_187721

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 9 → speed * time * (1000 / 3600) = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l1877_187721


namespace NUMINAMATH_CALUDE_cubic_sum_l1877_187796

theorem cubic_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 8) / a = (b^3 + 8) / b ∧ (b^3 + 8) / b = (c^3 + 8) / c) : 
  a^3 + b^3 + c^3 = -24 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_l1877_187796


namespace NUMINAMATH_CALUDE_min_product_given_sum_l1877_187702

theorem min_product_given_sum (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 8 → a * b ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_product_given_sum_l1877_187702


namespace NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l1877_187703

/-- The volume of an oblique triangular prism -/
theorem oblique_triangular_prism_volume 
  (S d : ℝ) 
  (h_S : S > 0) 
  (h_d : d > 0) : 
  ∃ V : ℝ, V = (1/2) * d * S ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l1877_187703


namespace NUMINAMATH_CALUDE_equation_solution_l1877_187772

theorem equation_solution : ∃ x : ℝ, x = 25 ∧ Real.sqrt (1 + Real.sqrt (2 + x^2)) = (3 + Real.sqrt x) ^ (1/3 : ℝ) :=
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1877_187772


namespace NUMINAMATH_CALUDE_problem_solution_l1877_187739

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x else |Real.sin x|

theorem problem_solution (a : ℝ) :
  f a = (1/2) → (a = (1/4) ∨ a = -π/6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1877_187739


namespace NUMINAMATH_CALUDE_ninety_degrees_possible_l1877_187708

-- Define a pentagon with angles in arithmetic progression
def Pentagon (a d : ℝ) : Prop :=
  a > 60 ∧  -- smallest angle > 60 degrees
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 540  -- sum of angles in pentagon

-- Theorem statement
theorem ninety_degrees_possible (a d : ℝ) :
  Pentagon a d → ∃ k : ℕ, k < 5 ∧ a + k*d = 90 := by
  sorry


end NUMINAMATH_CALUDE_ninety_degrees_possible_l1877_187708


namespace NUMINAMATH_CALUDE_probability_all_colors_l1877_187753

/-- The probability of selecting 4 balls of all three colors from 11 balls (3 red, 3 black, 5 white) -/
theorem probability_all_colors (total : ℕ) (red : ℕ) (black : ℕ) (white : ℕ) (select : ℕ) : 
  total = 11 → red = 3 → black = 3 → white = 5 → select = 4 →
  (Nat.choose red 2 * Nat.choose black 1 * Nat.choose white 1 +
   Nat.choose black 2 * Nat.choose red 1 * Nat.choose white 1 +
   Nat.choose white 2 * Nat.choose red 1 * Nat.choose black 1) / 
  Nat.choose total select = 6 / 11 := by
sorry

end NUMINAMATH_CALUDE_probability_all_colors_l1877_187753


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l1877_187780

/-- A quadratic equation in one variable is of the form ax² + bx + c = 0, where a, b, and c are constants, and a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation (x-1)(x+2)=1 -/
def equation (x : ℝ) : ℝ := (x - 1) * (x + 2) - 1

theorem equation_is_quadratic : is_quadratic_equation equation := by
  sorry

#check equation_is_quadratic

end NUMINAMATH_CALUDE_equation_is_quadratic_l1877_187780


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l1877_187782

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x * (x - a) < 0}
def B : Set ℝ := {x | x^2 - 7*x - 18 < 0}

-- State the theorem
theorem set_inclusion_implies_a_range (a : ℝ) : A a ⊆ B → a ∈ Set.Icc (-2) 9 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_range_l1877_187782


namespace NUMINAMATH_CALUDE_sam_pennies_l1877_187735

theorem sam_pennies (initial_pennies final_pennies : ℕ) 
  (h1 : initial_pennies = 98) 
  (h2 : final_pennies = 191) : 
  final_pennies - initial_pennies = 93 := by
  sorry

end NUMINAMATH_CALUDE_sam_pennies_l1877_187735


namespace NUMINAMATH_CALUDE_johns_hourly_rate_is_10_l1877_187737

/-- Calculates John's hourly rate when earning the performance bonus -/
def johnsHourlyRateWithBonus (basePay dayHours bonusPay bonusHours : ℚ) : ℚ :=
  (basePay + bonusPay) / (dayHours + bonusHours)

/-- Theorem: John's hourly rate with bonus is $10 per hour -/
theorem johns_hourly_rate_is_10 :
  johnsHourlyRateWithBonus 80 8 20 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_johns_hourly_rate_is_10_l1877_187737


namespace NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l1877_187768

/-- Given a line segment connecting (1, -3) and (4, 6), parameterized by x = pt + q and y = rt + s,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 100 -/
theorem line_segment_parameter_sum_of_squares :
  ∀ (p q r s : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) →
  (q = 1 ∧ s = -3) →
  (p + q = 4 ∧ r + s = 6) →
  p^2 + q^2 + r^2 + s^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l1877_187768


namespace NUMINAMATH_CALUDE_nine_power_comparison_l1877_187798

theorem nine_power_comparison : 9^(10^10) > 9^20 := by
  sorry

end NUMINAMATH_CALUDE_nine_power_comparison_l1877_187798


namespace NUMINAMATH_CALUDE_complex_determinant_equation_l1877_187778

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_determinant_equation :
  ∀ z : ℂ, determinant z i 1 i = 1 + i → z = 2 - i := by sorry

end NUMINAMATH_CALUDE_complex_determinant_equation_l1877_187778


namespace NUMINAMATH_CALUDE_sum_is_zero_l1877_187718

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Three non-zero vectors with specified properties -/
structure ThreeVectors (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (a b c : V)
  (a_nonzero : a ≠ 0)
  (b_nonzero : b ≠ 0)
  (c_nonzero : c ≠ 0)
  (ab_noncollinear : ¬ ∃ (r : ℝ), a = r • b)
  (bc_noncollinear : ¬ ∃ (r : ℝ), b = r • c)
  (ca_noncollinear : ¬ ∃ (r : ℝ), c = r • a)
  (ab_parallel_c : ∃ (m : ℝ), a + b = m • c)
  (bc_parallel_a : ∃ (n : ℝ), b + c = n • a)

/-- The sum of three vectors with the given properties is zero -/
theorem sum_is_zero (v : ThreeVectors V) : v.a + v.b + v.c = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_is_zero_l1877_187718


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l1877_187751

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Theorem stating the volume and height of a specific tetrahedron -/
theorem tetrahedron_volume_and_height :
  let A₁ : Point3D := ⟨2, 3, 1⟩
  let A₂ : Point3D := ⟨4, 1, -2⟩
  let A₃ : Point3D := ⟨6, 3, 7⟩
  let A₄ : Point3D := ⟨7, 5, -3⟩
  (tetrahedronVolume A₁ A₂ A₃ A₄ = 70/3) ∧
  (tetrahedronHeight A₁ A₂ A₃ A₄ = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l1877_187751


namespace NUMINAMATH_CALUDE_tv_selection_probability_l1877_187758

def num_type_a : ℕ := 3
def num_type_b : ℕ := 2
def total_tvs : ℕ := num_type_a + num_type_b
def selection_size : ℕ := 2

theorem tv_selection_probability :
  let total_combinations := Nat.choose total_tvs selection_size
  let favorable_combinations := num_type_a * num_type_b
  (favorable_combinations : ℚ) / total_combinations = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_tv_selection_probability_l1877_187758


namespace NUMINAMATH_CALUDE_system_solutions_system_no_solutions_l1877_187738

-- Define the system of equations
def system (x y k : ℝ) : Prop :=
  3 * x - 4 * y = 9 ∧ 6 * x - 8 * y = k

-- Theorem statement
theorem system_solutions (k : ℝ) :
  (∃ x y, system x y k) ↔ k = 18 :=
by sorry

-- Corollary for no solutions
theorem system_no_solutions (k : ℝ) :
  (¬ ∃ x y, system x y k) ↔ k ≠ 18 :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_system_no_solutions_l1877_187738


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1877_187736

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (5 + 7 * i) / (2 + 3 * i) = 31 / 13 - (1 / 13) * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1877_187736


namespace NUMINAMATH_CALUDE_fourth_power_nested_roots_l1877_187724

theorem fourth_power_nested_roots : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 2)))^4 = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_nested_roots_l1877_187724


namespace NUMINAMATH_CALUDE_trapezoid_lower_side_length_l1877_187705

/-- Proves that the length of the lower side of a trapezoid is 17.65 cm given specific conditions -/
theorem trapezoid_lower_side_length 
  (height : ℝ) 
  (area : ℝ) 
  (side_difference : ℝ) 
  (h1 : height = 5.2)
  (h2 : area = 100.62)
  (h3 : side_difference = 3.4) : 
  ∃ (lower_side : ℝ), lower_side = 17.65 ∧ 
  area = (1/2) * (lower_side + (lower_side + side_difference)) * height :=
sorry

end NUMINAMATH_CALUDE_trapezoid_lower_side_length_l1877_187705


namespace NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l1877_187734

theorem sum_of_two_squares_equivalence (n : ℤ) : 
  (∃ (a b : ℤ), n = a^2 + b^2) ↔ (∃ (u v : ℤ), 2*n = u^2 + v^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l1877_187734


namespace NUMINAMATH_CALUDE_min_phi_for_even_sine_l1877_187704

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem -/
theorem min_phi_for_even_sine (ω φ : ℝ) (h_omega : ω ≠ 0) (h_phi : φ > 0) 
  (h_even : IsEven (fun x ↦ 2 * Real.sin (ω * x + φ))) :
  ∃ (k : ℤ), φ = k * Real.pi + Real.pi / 2 ∧ 
  ∀ (m : ℤ), (m * Real.pi + Real.pi / 2 > 0) → (k * Real.pi + Real.pi / 2 ≤ m * Real.pi + Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_min_phi_for_even_sine_l1877_187704


namespace NUMINAMATH_CALUDE_correlated_relationships_l1877_187709

/-- Represents a relationship between two variables -/
structure Relationship where
  has_correlation : Bool

/-- The relationship between carbon content in molten steel and smelting time -/
def steel_relationship : Relationship :=
  ⟨true⟩

/-- The relationship between a point on a curve and its coordinates -/
def curve_point_relationship : Relationship :=
  ⟨false⟩

/-- The relationship between citrus yield and temperature -/
def citrus_yield_relationship : Relationship :=
  ⟨true⟩

/-- The relationship between tree cross-section diameter and height -/
def tree_relationship : Relationship :=
  ⟨true⟩

/-- The relationship between a person's age and wealth -/
def age_wealth_relationship : Relationship :=
  ⟨true⟩

/-- The list of all relationships -/
def all_relationships : List Relationship :=
  [steel_relationship, curve_point_relationship, citrus_yield_relationship, tree_relationship, age_wealth_relationship]

theorem correlated_relationships :
  (all_relationships.filter (·.has_correlation)).length = 4 :=
sorry

end NUMINAMATH_CALUDE_correlated_relationships_l1877_187709


namespace NUMINAMATH_CALUDE_smallest_m_is_671_l1877_187720

def is_valid (m n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    a = 2015^(3*m+1) ∧
    b = 2015^(6*n+2) ∧
    a < b ∧
    a % 10^2014 = b % 10^2014

theorem smallest_m_is_671 :
  (∃ (n : ℕ), is_valid 671 n) ∧
  (∀ (m : ℕ), m < 671 → ¬∃ (n : ℕ), is_valid m n) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_671_l1877_187720


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_13_and_3_l1877_187794

theorem three_digit_divisible_by_13_and_3 : 
  (Finset.filter (fun n => n % 13 = 0 ∧ n % 3 = 0) (Finset.range 900)).card = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_13_and_3_l1877_187794


namespace NUMINAMATH_CALUDE_fraction_simplest_form_l1877_187771

theorem fraction_simplest_form (n : ℤ) : Int.gcd (39*n + 4) (26*n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplest_form_l1877_187771


namespace NUMINAMATH_CALUDE_perpendicular_probability_l1877_187706

/-- A square is a shape with 4 vertices -/
structure Square where
  vertices : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 4

/-- A line in a square is defined by two distinct vertices -/
structure Line (s : Square) where
  v1 : s.vertices
  v2 : s.vertices
  distinct : v1 ≠ v2

/-- Two lines are perpendicular if they form a right angle -/
def perpendicular (s : Square) (l1 l2 : Line s) : Prop := sorry

/-- The total number of possible line pairs in a square -/
def total_line_pairs (s : Square) : ℕ := sorry

/-- The number of perpendicular line pairs in a square -/
def perpendicular_line_pairs (s : Square) : ℕ := sorry

/-- The theorem to be proved -/
theorem perpendicular_probability (s : Square) : 
  (perpendicular_line_pairs s : ℚ) / (total_line_pairs s : ℚ) = 5 / 18 := sorry

end NUMINAMATH_CALUDE_perpendicular_probability_l1877_187706


namespace NUMINAMATH_CALUDE_lawn_mowing_time_l1877_187783

/-- Time required to mow a rectangular lawn -/
theorem lawn_mowing_time : 
  ∀ (length width swath_width overlap speed : ℝ),
  length = 90 →
  width = 150 →
  swath_width = 28 / 12 →
  overlap = 4 / 12 →
  speed = 5000 →
  (width / (swath_width - overlap) * length) / speed = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_time_l1877_187783


namespace NUMINAMATH_CALUDE_perfect_linearity_implies_R_squared_one_l1877_187757

/-- A scatter plot is perfectly linear if all its points fall on a straight line with non-zero slope -/
def is_perfectly_linear (scatter_plot : Set (ℝ × ℝ)) : Prop :=
  ∃ (m : ℝ) (b : ℝ), m ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ scatter_plot → y = m * x + b

/-- The coefficient of determination (R²) for a scatter plot -/
def R_squared (scatter_plot : Set (ℝ × ℝ)) : ℝ := sorry

theorem perfect_linearity_implies_R_squared_one (scatter_plot : Set (ℝ × ℝ)) :
  is_perfectly_linear scatter_plot → R_squared scatter_plot = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_linearity_implies_R_squared_one_l1877_187757


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l1877_187769

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l1877_187769


namespace NUMINAMATH_CALUDE_inequality_proof_l1877_187725

theorem inequality_proof (a b c d k : ℝ) 
  (h1 : |k| < 2) 
  (h2 : a^2 + b^2 - k*a*b = 1) 
  (h3 : c^2 + d^2 - k*c*d = 1) : 
  |a*c - b*d| ≤ 2 / Real.sqrt (4 - k^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1877_187725


namespace NUMINAMATH_CALUDE_response_rate_is_sixty_percent_l1877_187717

/-- The response rate percentage for a mail questionnaire -/
def response_rate (responses_needed : ℕ) (questionnaires_mailed : ℕ) : ℚ :=
  (responses_needed : ℚ) / (questionnaires_mailed : ℚ) * 100

/-- Theorem: The response rate is 60% when 750 responses are needed and 1250 questionnaires are mailed -/
theorem response_rate_is_sixty_percent :
  response_rate 750 1250 = 60 := by
  sorry

#eval response_rate 750 1250

end NUMINAMATH_CALUDE_response_rate_is_sixty_percent_l1877_187717


namespace NUMINAMATH_CALUDE_union_complement_problem_l1877_187716

open Set

theorem union_complement_problem (A B : Set ℝ) 
  (hA : A = {x : ℝ | -2 ≤ x ∧ x ≤ 3})
  (hB : B = {x : ℝ | x < -1 ∨ 4 < x}) :
  A ∪ (univ \ B) = {x : ℝ | -2 ≤ x ∧ x ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_problem_l1877_187716


namespace NUMINAMATH_CALUDE_minimum_pieces_to_capture_all_l1877_187799

/-- Represents a rhombus-shaped game board -/
structure RhombusBoard where
  angle : ℝ
  side_divisions : ℕ

/-- Represents a piece on the game board -/
structure GamePiece where
  position : ℕ × ℕ

/-- Represents the set of cells captured by a piece -/
def captured_cells (board : RhombusBoard) (piece : GamePiece) : Set (ℕ × ℕ) :=
  sorry

/-- The total number of cells on the board -/
def total_cells (board : RhombusBoard) : ℕ :=
  sorry

/-- Checks if a set of pieces captures all cells on the board -/
def captures_all_cells (board : RhombusBoard) (pieces : List GamePiece) : Prop :=
  sorry

theorem minimum_pieces_to_capture_all (board : RhombusBoard)
  (h1 : board.angle = 60)
  (h2 : board.side_divisions = 9) :
  ∃ (pieces : List GamePiece),
    pieces.length = 6 ∧
    captures_all_cells board pieces ∧
    ∀ (other_pieces : List GamePiece),
      captures_all_cells board other_pieces →
      other_pieces.length ≥ 6 :=
  sorry

end NUMINAMATH_CALUDE_minimum_pieces_to_capture_all_l1877_187799


namespace NUMINAMATH_CALUDE_three_distinct_roots_l1877_187728

theorem three_distinct_roots (p : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    ∀ t : ℝ, t^3 + p*t^2 - p*t - 1 = 0 ↔ t = x ∨ t = y ∨ t = z) ↔
  p > 1 ∨ p < -3 :=
sorry

end NUMINAMATH_CALUDE_three_distinct_roots_l1877_187728


namespace NUMINAMATH_CALUDE_reflection_theorem_l1877_187726

/-- A reflection in 2D space -/
structure Reflection2D where
  /-- The function that performs the reflection -/
  apply : ℝ × ℝ → ℝ × ℝ

/-- Theorem: Given a reflection that maps (3, -2) to (7, 6), it will map (0, 4) to (80/29, -84/29) -/
theorem reflection_theorem (r : Reflection2D) 
  (h1 : r.apply (3, -2) = (7, 6)) :
  r.apply (0, 4) = (80/29, -84/29) := by
sorry


end NUMINAMATH_CALUDE_reflection_theorem_l1877_187726


namespace NUMINAMATH_CALUDE_frank_riding_time_l1877_187713

-- Define the riding times for each person
def dave_time : ℝ := 10

-- Chuck's time is 5 times Dave's time
def chuck_time : ℝ := 5 * dave_time

-- Erica's time is 30% longer than Chuck's time
def erica_time : ℝ := chuck_time * (1 + 0.3)

-- Frank's time is 20% longer than Erica's time
def frank_time : ℝ := erica_time * (1 + 0.2)

-- Theorem to prove
theorem frank_riding_time : frank_time = 78 := by
  sorry

end NUMINAMATH_CALUDE_frank_riding_time_l1877_187713


namespace NUMINAMATH_CALUDE_largest_difference_l1877_187755

def P : ℕ := 3 * 1003^1004
def Q : ℕ := 1003^1004
def R : ℕ := 1002 * 1003^1003
def S : ℕ := 3 * 1003^1003
def T : ℕ := 1003^1003
def U : ℕ := 1003^1002 * Nat.factorial 1002

theorem largest_difference (P Q R S T U : ℕ) 
  (hP : P = 3 * 1003^1004)
  (hQ : Q = 1003^1004)
  (hR : R = 1002 * 1003^1003)
  (hS : S = 3 * 1003^1003)
  (hT : T = 1003^1003)
  (hU : U = 1003^1002 * Nat.factorial 1002) :
  P - Q > max (Q - R) (max (R - S) (max (S - T) (T - U))) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_l1877_187755


namespace NUMINAMATH_CALUDE_negative_double_negative_and_negative_absolute_are_opposite_l1877_187756

-- Define opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Theorem statement
theorem negative_double_negative_and_negative_absolute_are_opposite :
  are_opposite (-(-5)) (-|5|) := by
  sorry

end NUMINAMATH_CALUDE_negative_double_negative_and_negative_absolute_are_opposite_l1877_187756


namespace NUMINAMATH_CALUDE_initial_time_calculation_l1877_187719

theorem initial_time_calculation (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) 
  (h1 : distance = 180)
  (h2 : new_speed = 20)
  (h3 : time_ratio = 3/2) :
  let new_time := distance / new_speed
  let initial_time := new_time * time_ratio
  initial_time = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_initial_time_calculation_l1877_187719


namespace NUMINAMATH_CALUDE_parabola_unique_coefficients_l1877_187789

/-- A parabola passing through three given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ
  p1 : eq (-2) = -16
  p2 : eq 2 = 8
  p3 : eq 4 = 36
  form : ∀ x, eq x = x^2 + b*x + c

/-- The unique values of b and c for the parabola -/
theorem parabola_unique_coefficients (p : Parabola) : p.b = 6 ∧ p.c = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_unique_coefficients_l1877_187789


namespace NUMINAMATH_CALUDE_point_q_location_l1877_187761

/-- Given four points O, A, B, C on a straight line and a point Q on AB, prove that Q's position relative to O is 2a + 2.5b -/
theorem point_q_location (a b c : ℝ) (O A B C Q : ℝ) : 
  O < A ∧ A < B ∧ B < C ∧  -- Points are in order
  A - O = 2 * a ∧  -- OA = 2a
  B - A = 3 * b ∧  -- AB = 3b
  C - B = 4 * c ∧  -- BC = 4c
  A ≤ Q ∧ Q ≤ B ∧  -- Q is on segment AB
  (Q - A) / (B - Q) = 3 / 1  -- AQ:QB = 3:1
  → Q - O = 2 * a + 2.5 * b :=
by sorry

end NUMINAMATH_CALUDE_point_q_location_l1877_187761


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1877_187774

theorem negation_of_proposition :
  (¬ ∀ a : ℕ+, 2^(a : ℕ) ≥ (a : ℕ)^2) ↔ (∃ a : ℕ+, 2^(a : ℕ) < (a : ℕ)^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1877_187774


namespace NUMINAMATH_CALUDE_relationship_abc_l1877_187754

theorem relationship_abc (x : ℝ) (h : x > 2) : (1/3)^3 < Real.log x ∧ Real.log x < x^3 := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1877_187754


namespace NUMINAMATH_CALUDE_base_4_20312_equals_566_l1877_187795

def base_4_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base_4_20312_equals_566 :
  base_4_to_10 [2, 1, 3, 0, 2] = 566 := by
  sorry

end NUMINAMATH_CALUDE_base_4_20312_equals_566_l1877_187795


namespace NUMINAMATH_CALUDE_central_angle_regular_hexagon_l1877_187731

/-- The central angle of a regular hexagon is 60 degrees. -/
theorem central_angle_regular_hexagon :
  ∀ (full_circle_degrees : ℝ) (num_sides : ℕ),
    full_circle_degrees = 360 →
    num_sides = 6 →
    full_circle_degrees / num_sides = 60 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_regular_hexagon_l1877_187731


namespace NUMINAMATH_CALUDE_fruit_punch_water_amount_l1877_187773

/-- Represents the ratio of ingredients in the fruit punch -/
structure PunchRatio :=
  (water : ℚ)
  (orange_juice : ℚ)
  (cranberry_juice : ℚ)

/-- Calculates the amount of water needed for a given amount of punch -/
def water_needed (ratio : PunchRatio) (total_gallons : ℚ) (quarts_per_gallon : ℚ) : ℚ :=
  let total_parts := ratio.water + ratio.orange_juice + ratio.cranberry_juice
  let water_fraction := ratio.water / total_parts
  water_fraction * total_gallons * quarts_per_gallon

/-- Theorem stating the amount of water needed for the fruit punch -/
theorem fruit_punch_water_amount :
  let ratio : PunchRatio := ⟨5, 2, 1⟩
  let total_gallons : ℚ := 3
  let quarts_per_gallon : ℚ := 4
  water_needed ratio total_gallons quarts_per_gallon = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_punch_water_amount_l1877_187773


namespace NUMINAMATH_CALUDE_triangle_side_length_l1877_187748

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2/3 →
  b = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1877_187748


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_865_over_21_l1877_187707

theorem sqrt_fraction_sum_equals_sqrt_865_over_21 :
  Real.sqrt (9 / 49 + 16 / 9) = Real.sqrt 865 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_865_over_21_l1877_187707


namespace NUMINAMATH_CALUDE_problem_statement_l1877_187700

theorem problem_statement (a b : ℝ) 
  (h1 : a < b) (h2 : b < 0) (h3 : a^2 + b^2 = 4*a*b) : 
  (a + b) / (a - b) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1877_187700


namespace NUMINAMATH_CALUDE_class_project_funding_l1877_187723

/-- Calculates the total amount gathered by a class for a project --/
def total_amount_gathered (total_students : ℕ) (full_payment : ℕ) (half_paying_students : ℕ) : ℕ :=
  let full_paying_students := total_students - half_paying_students
  let full_amount := full_paying_students * full_payment
  let half_amount := half_paying_students * (full_payment / 2)
  full_amount + half_amount

/-- Proves that the class gathered $1150 for their project --/
theorem class_project_funding :
  total_amount_gathered 25 50 4 = 1150 := by
  sorry

end NUMINAMATH_CALUDE_class_project_funding_l1877_187723


namespace NUMINAMATH_CALUDE_family_ages_l1877_187776

/-- Problem statement about the ages of family members -/
theorem family_ages (mark_age john_age emma_age parents_current_age : ℕ)
  (h1 : mark_age = 18)
  (h2 : john_age = mark_age - 10)
  (h3 : emma_age = mark_age - 4)
  (h4 : parents_current_age = 7 * john_age)
  (h5 : parents_current_age = 25 + emma_age) :
  parents_current_age - mark_age = 38 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_l1877_187776


namespace NUMINAMATH_CALUDE_sphere_only_circular_all_views_l1877_187712

-- Define the geometric shapes
inductive Shape
| Cuboid
| Cylinder
| Cone
| Sphere

-- Define the views
inductive View
| Front
| Left
| Top

-- Function to determine if a view of a shape is circular
def isCircularView (s : Shape) (v : View) : Prop :=
  match s, v with
  | Shape.Sphere, _ => True
  | Shape.Cylinder, View.Top => True
  | Shape.Cone, View.Top => True
  | _, _ => False

-- Theorem stating that only the Sphere has circular views in all three perspectives
theorem sphere_only_circular_all_views :
  ∀ s : Shape, (∀ v : View, isCircularView s v) ↔ s = Shape.Sphere := by
  sorry

end NUMINAMATH_CALUDE_sphere_only_circular_all_views_l1877_187712


namespace NUMINAMATH_CALUDE_apples_sale_theorem_l1877_187763

/-- Calculate the total money made from selling boxes of apples -/
def total_money_from_apples (total_apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ) : ℕ :=
  ((total_apples / apples_per_box) * price_per_box)

/-- Theorem: Given 275 apples, with 20 apples per box sold at 8,000 won each,
    the total money made from selling all full boxes is 104,000 won -/
theorem apples_sale_theorem :
  total_money_from_apples 275 20 8000 = 104000 := by
  sorry

end NUMINAMATH_CALUDE_apples_sale_theorem_l1877_187763


namespace NUMINAMATH_CALUDE_composition_equality_l1877_187775

/-- Given two functions f and g, prove that if f(g(b)) = 4, then b = -1/2 -/
theorem composition_equality (f g : ℝ → ℝ) (b : ℝ) 
    (hf : ∀ x, f x = x / 3 + 2)
    (hg : ∀ x, g x = 5 - 2 * x)
    (h : f (g b) = 4) : 
  b = -1/2 := by
sorry

end NUMINAMATH_CALUDE_composition_equality_l1877_187775


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1877_187764

/-- An arithmetic sequence {a_n} with the given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1/3)
  (h3 : a 2 + a 5 = 4)
  (h4 : ∃ n : ℕ, a n = 27) :
  ∃ n : ℕ, n = 9 ∧ a n = 27 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1877_187764


namespace NUMINAMATH_CALUDE_cubic_expression_equals_zero_l1877_187759

theorem cubic_expression_equals_zero (k : ℝ) (h : k = 2) : (k^3 - 8) * (k + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equals_zero_l1877_187759


namespace NUMINAMATH_CALUDE_circle_radius_l1877_187797

/-- A circle with center (0, k) where k > 10, tangent to y = x, y = -x, y = 10, and x-axis has radius 20 -/
theorem circle_radius (k : ℝ) (h1 : k > 10) : 
  let circle := { (x, y) | x^2 + (y - k)^2 = (k - 10)^2 }
  (∀ (x y : ℝ), (x = y ∨ x = -y ∨ y = 10 ∨ y = 0) → 
    (x, y) ∈ circle → x^2 + (y - k)^2 = (k - 10)^2) →
  k - 10 = 20 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l1877_187797


namespace NUMINAMATH_CALUDE_erwans_shopping_trip_l1877_187732

/-- Proves that the price of each shirt is $80 given the conditions of Erwan's shopping trip -/
theorem erwans_shopping_trip (shoe_price : ℝ) (shirt_price : ℝ) :
  shoe_price = 200 →
  (shoe_price * 0.7 + 2 * shirt_price) * 0.95 = 285 →
  shirt_price = 80 :=
by sorry

end NUMINAMATH_CALUDE_erwans_shopping_trip_l1877_187732


namespace NUMINAMATH_CALUDE_fermat_like_equation_l1877_187733

theorem fermat_like_equation (a b c : ℕ) (h1 : Even c) (h2 : a^5 + 4*b^5 = c^5) : b = 0 := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_equation_l1877_187733


namespace NUMINAMATH_CALUDE_statement_D_no_related_factor_l1877_187752

-- Define a type for statements
inductive Statement
| A : Statement  -- A timely snow promises a good harvest
| B : Statement  -- If the upper beam is not straight, the lower beam will be crooked
| C : Statement  -- Smoking is harmful to health
| D : Statement  -- Magpies signify joy, crows signify mourning

-- Define what it means for a statement to have a related factor
def has_related_factor (s : Statement) : Prop :=
  ∃ (x y : Prop), (x → y) ∧ (s = Statement.A ∨ s = Statement.B ∨ s = Statement.C)

-- Theorem: Statement D does not have a related factor
theorem statement_D_no_related_factor :
  ¬ has_related_factor Statement.D :=
by
  sorry


end NUMINAMATH_CALUDE_statement_D_no_related_factor_l1877_187752


namespace NUMINAMATH_CALUDE_function_composition_nonnegative_implies_a_lower_bound_l1877_187760

theorem function_composition_nonnegative_implies_a_lower_bound 
  (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + 2 * x + 1) 
  (h2 : ∀ x, f (f x) ≥ 0) : 
  a ≥ (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_function_composition_nonnegative_implies_a_lower_bound_l1877_187760


namespace NUMINAMATH_CALUDE_x_lt_2_necessary_not_sufficient_for_x_lt_0_l1877_187766

theorem x_lt_2_necessary_not_sufficient_for_x_lt_0 :
  (∀ x : ℝ, x < 0 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_x_lt_2_necessary_not_sufficient_for_x_lt_0_l1877_187766


namespace NUMINAMATH_CALUDE_clara_cookies_sold_l1877_187714

/-- Calculates the total number of cookies sold by Clara -/
def total_cookies_sold (cookies_per_box : Fin 3 → ℕ) (boxes_sold : Fin 3 → ℕ) : ℕ :=
  (cookies_per_box 0) * (boxes_sold 0) + 
  (cookies_per_box 1) * (boxes_sold 1) + 
  (cookies_per_box 2) * (boxes_sold 2)

/-- Proves that Clara sells 3320 cookies in total -/
theorem clara_cookies_sold :
  let cookies_per_box : Fin 3 → ℕ := ![12, 20, 16]
  let boxes_sold : Fin 3 → ℕ := ![50, 80, 70]
  total_cookies_sold cookies_per_box boxes_sold = 3320 := by
  sorry

end NUMINAMATH_CALUDE_clara_cookies_sold_l1877_187714


namespace NUMINAMATH_CALUDE_heartsuit_five_three_l1877_187740

def heartsuit (x y : ℤ) : ℤ := 4 * x - 2 * y

theorem heartsuit_five_three : heartsuit 5 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_five_three_l1877_187740


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l1877_187746

/-- A line in the xy-plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := 3/2, point := (0, 6) } →
  b.point = (4, 2) →
  y_intercept b = -4 := by
sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l1877_187746


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l1877_187711

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : ℕ
  diagonal_shaded : Bool

/-- Calculate the shaded fraction of a quilt block -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  if q.diagonal_shaded && q.size = 3 then 1/6 else 0

/-- Theorem: The shaded fraction of a 3x3 quilt block with half-shaded diagonal squares is 1/6 -/
theorem quilt_shaded_fraction :
  ∀ (q : QuiltBlock), q.size = 3 → q.diagonal_shaded → shaded_fraction q = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l1877_187711


namespace NUMINAMATH_CALUDE_article_pricing_theorem_l1877_187765

/-- Represents the price and profit relationship for an article -/
structure ArticlePricing where
  cost_price : ℝ
  profit_price : ℝ
  loss_price : ℝ
  desired_profit_price : ℝ

/-- The main theorem about the article pricing -/
theorem article_pricing_theorem (a : ArticlePricing) 
  (h1 : a.profit_price - a.cost_price = a.cost_price - a.loss_price)
  (h2 : a.profit_price = 832)
  (h3 : a.desired_profit_price = 896) :
  a.cost_price * 1.4 = a.desired_profit_price :=
sorry

#check article_pricing_theorem

end NUMINAMATH_CALUDE_article_pricing_theorem_l1877_187765


namespace NUMINAMATH_CALUDE_line_properties_l1877_187767

/-- Point type representing a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing a parametric line -/
structure Line where
  p : Point
  α : ℝ

/-- Function to calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Function to get the x-intercept of a line -/
def xIntercept (l : Line) : Point := sorry

/-- Function to get the y-intercept of a line -/
def yIntercept (l : Line) : Point := sorry

/-- Function to convert a line to its polar form -/
def toPolarForm (l : Line) : ℝ → ℝ := sorry

theorem line_properties (P : Point) (l : Line) (h1 : P.x = 2 ∧ P.y = 1) 
    (h2 : l.p = P) 
    (h3 : ∀ t : ℝ, ∃ x y : ℝ, x = 2 + t * Real.cos l.α ∧ y = 1 + t * Real.sin l.α)
    (h4 : distance P (xIntercept l) * distance P (yIntercept l) = 4) :
  l.α = 3 * Real.pi / 4 ∧ 
  ∀ θ : ℝ, toPolarForm l θ * (Real.cos θ + Real.sin θ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_properties_l1877_187767


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1877_187791

theorem quadratic_max_value :
  let f : ℝ → ℝ := fun x ↦ -3 * x^2 + 6 * x + 4
  ∃ m : ℝ, (∀ x : ℝ, f x ≤ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1877_187791


namespace NUMINAMATH_CALUDE_num_kittens_is_eleven_l1877_187701

/-- The number of kittens -/
def num_kittens : ℕ := 11

/-- The weight of the two lightest kittens -/
def weight_lightest : ℕ := 80

/-- The weight of the four heaviest kittens -/
def weight_heaviest : ℕ := 200

/-- The total weight of all kittens -/
def total_weight : ℕ := 500

/-- Theorem stating that the number of kittens is 11 given the weight conditions -/
theorem num_kittens_is_eleven :
  (weight_lightest = 80) →
  (weight_heaviest = 200) →
  (total_weight = 500) →
  (num_kittens = 11) :=
by
  sorry

#check num_kittens_is_eleven

end NUMINAMATH_CALUDE_num_kittens_is_eleven_l1877_187701


namespace NUMINAMATH_CALUDE_no_solution_implies_m_geq_two_l1877_187715

theorem no_solution_implies_m_geq_two (m : ℝ) : 
  (∀ x : ℝ, ¬(2*x - 1 < 3 ∧ x > m)) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_geq_two_l1877_187715


namespace NUMINAMATH_CALUDE_triangle_problem_l1877_187747

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
def Triangle (a b c A B C : ℝ) : Prop :=
  -- Add necessary conditions for a valid triangle here
  True

theorem triangle_problem (a b c A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : Real.sqrt 3 * c * Real.cos A + a * Real.sin C = Real.sqrt 3 * c)
  (h_sum : b + c = 5)
  (h_area : (1/2) * b * c * Real.sin A = Real.sqrt 3) :
  A = π/3 ∧ a = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1877_187747


namespace NUMINAMATH_CALUDE_jerry_firecrackers_l1877_187779

theorem jerry_firecrackers (F : ℕ) : 
  (F ≥ 12) →
  (5 * (F - 12) / 6 = 30) →
  F = 48 := by sorry

end NUMINAMATH_CALUDE_jerry_firecrackers_l1877_187779


namespace NUMINAMATH_CALUDE_zeros_properties_l1877_187770

noncomputable def f (θ : ℝ) : ℝ := Real.sin (4 * θ) + Real.sin (3 * θ)

theorem zeros_properties (θ₁ θ₂ θ₃ : ℝ) 
  (h1 : 0 < θ₁ ∧ θ₁ < π) 
  (h2 : 0 < θ₂ ∧ θ₂ < π) 
  (h3 : 0 < θ₃ ∧ θ₃ < π) 
  (h4 : θ₁ ≠ θ₂ ∧ θ₁ ≠ θ₃ ∧ θ₂ ≠ θ₃) 
  (h5 : f θ₁ = 0) 
  (h6 : f θ₂ = 0) 
  (h7 : f θ₃ = 0) : 
  (θ₁ + θ₂ + θ₃ = 12 * π / 7) ∧ 
  (Real.cos θ₁ * Real.cos θ₂ * Real.cos θ₃ = 1 / 8) ∧ 
  (Real.cos θ₁ + Real.cos θ₂ + Real.cos θ₃ = -1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_zeros_properties_l1877_187770


namespace NUMINAMATH_CALUDE_cloth_cost_price_l1877_187790

/-- Given a shopkeeper selling cloth with the following conditions:
  * The shopkeeper sells 200 metres of cloth
  * The selling price is Rs. 12000
  * The shopkeeper incurs a loss of Rs. 6 per metre
  Prove that the cost price for one metre of cloth is Rs. 66 -/
theorem cloth_cost_price 
  (total_metres : ℕ) 
  (selling_price : ℕ) 
  (loss_per_metre : ℕ) 
  (h1 : total_metres = 200)
  (h2 : selling_price = 12000)
  (h3 : loss_per_metre = 6) :
  (selling_price + total_metres * loss_per_metre) / total_metres = 66 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l1877_187790


namespace NUMINAMATH_CALUDE_roots_squared_relation_l1877_187749

def f (x : ℝ) : ℝ := 2 * x^3 - x^2 + 4 * x - 3

def g (b c d x : ℝ) : ℝ := x^3 + b * x^2 + c * x + d

theorem roots_squared_relation (b c d : ℝ) :
  (∀ r : ℝ, f r = 0 → g b c d (r^2) = 0) →
  b = 15/4 ∧ c = 5/2 ∧ d = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_roots_squared_relation_l1877_187749


namespace NUMINAMATH_CALUDE_sum_proper_divisors_729_l1877_187730

def proper_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ x => x ≠ 0 ∧ n % x = 0)

theorem sum_proper_divisors_729 :
  (proper_divisors 729).sum id = 364 := by
  sorry

end NUMINAMATH_CALUDE_sum_proper_divisors_729_l1877_187730


namespace NUMINAMATH_CALUDE_glasses_in_smaller_box_l1877_187742

theorem glasses_in_smaller_box :
  ∀ (x : ℕ) (s l : ℕ),
  -- There are two different-sized boxes
  s = 1 →
  -- There are 16 more larger boxes than smaller boxes
  l = s + 16 →
  -- One box (smaller) contains x glasses, the other (larger) contains 16 glasses
  -- The total number of glasses is 480
  x * s + 16 * l = 480 →
  -- Prove that the number of glasses in the smaller box is 208
  x = 208 := by
sorry

end NUMINAMATH_CALUDE_glasses_in_smaller_box_l1877_187742


namespace NUMINAMATH_CALUDE_bear_path_discrepancy_l1877_187777

/-- Represents the circular path of a polar bear on an ice floe -/
structure BearPath where
  diameter_instrument : ℝ  -- Diameter measured by instruments
  diameter_footprint : ℝ   -- Diameter measured from footprints
  is_in_still_water : Prop -- The ice floe is in still water

/-- The difference in measured diameters is due to relative motion -/
theorem bear_path_discrepancy (path : BearPath) 
  (h_instrument : path.diameter_instrument = 8.5)
  (h_footprint : path.diameter_footprint = 9)
  (h_water : path.is_in_still_water) :
  ∃ (relative_motion : ℝ), 
    relative_motion > 0 ∧ 
    path.diameter_footprint - path.diameter_instrument = relative_motion :=
by sorry

end NUMINAMATH_CALUDE_bear_path_discrepancy_l1877_187777


namespace NUMINAMATH_CALUDE_zoo_incident_final_counts_l1877_187785

def wombat_count : ℕ := 9
def rhea_count : ℕ := 3
def porcupine_count : ℕ := 2

def carson_claw_per_wombat : ℕ := 4
def ava_claw_per_rhea : ℕ := 1
def liam_quill_per_porcupine : ℕ := 6

def carson_reduction_percent : ℚ := 25 / 100
def ava_reduction_percent : ℚ := 25 / 100
def liam_reduction_percent : ℚ := 50 / 100

def carson_initial_claws : ℕ := wombat_count * carson_claw_per_wombat
def ava_initial_claws : ℕ := rhea_count * ava_claw_per_rhea
def liam_initial_quills : ℕ := porcupine_count * liam_quill_per_porcupine

theorem zoo_incident_final_counts :
  (carson_initial_claws - Int.floor (↑carson_initial_claws * carson_reduction_percent) = 27) ∧
  (ava_initial_claws - Int.floor (↑ava_initial_claws * ava_reduction_percent) = 3) ∧
  (liam_initial_quills - Int.floor (↑liam_initial_quills * liam_reduction_percent) = 6) :=
by sorry

end NUMINAMATH_CALUDE_zoo_incident_final_counts_l1877_187785


namespace NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_21_l1877_187743

theorem constant_term_expansion (x : ℝ) : 
  (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7) = x^7 + 2*x^6 + 2*x^5 + 3*x^4 + x^5 + 2*x^4 + x^3 + 7*x^3 + 7*x^2 + 21 := by
  sorry

theorem constant_term_is_21 : 
  (λ x : ℝ => (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7)) 0 = 21 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_21_l1877_187743


namespace NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l1877_187744

theorem greatest_x_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 4 ∧ 
  (∀ (x : ℝ), -2 * x^2 + 12 * x - 16 ≥ 0 → x ≤ x_max) ∧
  (-2 * x_max^2 + 12 * x_max - 16 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l1877_187744


namespace NUMINAMATH_CALUDE_sqrt_abs_equation_solution_l1877_187710

theorem sqrt_abs_equation_solution :
  ∀ x y : ℝ, Real.sqrt (2 * x + 3 * y) + |x + 3| = 0 → x = -3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_equation_solution_l1877_187710


namespace NUMINAMATH_CALUDE_power_two_mod_nine_l1877_187787

theorem power_two_mod_nine : 2 ^ 46655 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_mod_nine_l1877_187787


namespace NUMINAMATH_CALUDE_ratio_limit_is_one_l1877_187788

/-- The ratio of the largest element (2^20) to the sum of other elements in the set {1, 2, 2^2, ..., 2^20} -/
def ratio (n : ℕ) : ℚ :=
  2^n / (2^n - 1)

/-- The limit of the ratio as n approaches infinity is 1 -/
theorem ratio_limit_is_one : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |ratio n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_ratio_limit_is_one_l1877_187788


namespace NUMINAMATH_CALUDE_shoe_probability_l1877_187786

/-- Represents the total number of shoe pairs -/
def total_pairs : ℕ := 16

/-- Represents the number of black shoe pairs -/
def black_pairs : ℕ := 8

/-- Represents the number of brown shoe pairs -/
def brown_pairs : ℕ := 5

/-- Represents the number of white shoe pairs -/
def white_pairs : ℕ := 3

/-- The probability of picking two shoes of the same color with one being left and the other right -/
theorem shoe_probability : 
  (black_pairs * black_pairs + brown_pairs * brown_pairs + white_pairs * white_pairs) / 
  (total_pairs * (2 * total_pairs - 1)) = 49 / 248 := by
  sorry

end NUMINAMATH_CALUDE_shoe_probability_l1877_187786


namespace NUMINAMATH_CALUDE_quadratic_root_property_l1877_187762

theorem quadratic_root_property : ∀ m n : ℝ,
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ x = m ∨ x = n) →
  m + n - m*n = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l1877_187762


namespace NUMINAMATH_CALUDE_unfactorable_quadratic_l1877_187741

/-- A quadratic trinomial that cannot be factored into linear binomials with integer coefficients -/
theorem unfactorable_quadratic (a b c : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_eval : a * 1991^2 + b * 1991 + c = p) :
  ¬ ∃ (d₁ d₂ e₁ e₂ : ℤ), ∀ x, a * x^2 + b * x + c = (d₁ * x + e₁) * (d₂ * x + e₂) :=
by sorry

end NUMINAMATH_CALUDE_unfactorable_quadratic_l1877_187741


namespace NUMINAMATH_CALUDE_book_chapters_l1877_187729

/-- Represents the number of pages in a book with arithmetic progression of chapter lengths -/
def book_pages (n : ℕ) : ℕ := n * (2 * 13 + (n - 1) * 3) / 2

/-- Theorem stating that a book with 95 pages, where the first chapter has 13 pages
    and each subsequent chapter has 3 more pages than the previous one, has 5 chapters -/
theorem book_chapters :
  ∃ (n : ℕ), n > 0 ∧ book_pages n = 95 ∧ n = 5 := by sorry

end NUMINAMATH_CALUDE_book_chapters_l1877_187729


namespace NUMINAMATH_CALUDE_square_tiles_count_l1877_187722

/-- Represents a box of pentagonal and square tiles -/
structure TileBox where
  pentagonal : ℕ
  square : ℕ

/-- The total number of tiles in the box -/
def TileBox.total (box : TileBox) : ℕ := box.pentagonal + box.square

/-- The total number of edges in the box -/
def TileBox.edges (box : TileBox) : ℕ := 5 * box.pentagonal + 4 * box.square

theorem square_tiles_count (box : TileBox) : 
  box.total = 30 ∧ box.edges = 122 → box.square = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_tiles_count_l1877_187722
