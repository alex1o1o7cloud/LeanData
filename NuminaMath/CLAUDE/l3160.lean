import Mathlib

namespace NUMINAMATH_CALUDE_no_periodic_sequence_exists_l3160_316082

-- Define a_n as the first non-zero digit from the unit place in n!
def a (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_periodic_sequence_exists :
  ∀ N : ℕ, ¬∃ T : ℕ, T > 0 ∧ ∀ k : ℕ, a (N + k + T) = a (N + k) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_sequence_exists_l3160_316082


namespace NUMINAMATH_CALUDE_bird_families_difference_l3160_316043

theorem bird_families_difference (total : ℕ) (flew_away : ℕ) 
  (h1 : total = 87) (h2 : flew_away = 7) : 
  total - flew_away - flew_away = 73 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_difference_l3160_316043


namespace NUMINAMATH_CALUDE_log_equality_l3160_316042

theorem log_equality (x k : ℝ) (h1 : Real.log 3 / Real.log 8 = x) (h2 : Real.log 81 / Real.log 2 = k * x) : k = 12 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l3160_316042


namespace NUMINAMATH_CALUDE_franks_change_is_four_l3160_316050

/-- The amount of change Frank receives from his purchase. -/
def franks_change (chocolate_bars : ℕ) (chips : ℕ) (chocolate_price : ℚ) (chips_price : ℚ) (money_given : ℚ) : ℚ :=
  money_given - (chocolate_bars * chocolate_price + chips * chips_price)

/-- Theorem stating that Frank's change is $4 given the problem conditions. -/
theorem franks_change_is_four :
  franks_change 5 2 2 3 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_franks_change_is_four_l3160_316050


namespace NUMINAMATH_CALUDE_scatter_diagram_placement_l3160_316059

/-- Represents a variable in a scatter diagram -/
inductive ScatterVariable
| Explanatory
| Predictor

/-- Represents an axis in a scatter diagram -/
inductive Axis
| X
| Y

/-- Determines the correct axis for a given scatter variable -/
def correct_axis_placement (v : ScatterVariable) : Axis :=
  match v with
  | ScatterVariable.Explanatory => Axis.X
  | ScatterVariable.Predictor => Axis.Y

/-- Theorem stating the correct placement of variables in a scatter diagram -/
theorem scatter_diagram_placement :
  (correct_axis_placement ScatterVariable.Explanatory = Axis.X) ∧
  (correct_axis_placement ScatterVariable.Predictor = Axis.Y) :=
by sorry

end NUMINAMATH_CALUDE_scatter_diagram_placement_l3160_316059


namespace NUMINAMATH_CALUDE_scenario_contradiction_characteristics_l3160_316072

/-- Represents a person's reaction to a statement --/
inductive Reaction
  | Cry
  | Laugh

/-- Represents a family member --/
inductive FamilyMember
  | Mother
  | Father

/-- Represents the characteristics of a contradiction --/
structure ContradictionCharacteristics where
  interpenetrating : Bool
  specific : Bool

/-- Given scenario where a child's "I love you" causes different reactions --/
def scenario : FamilyMember → Reaction
  | FamilyMember.Mother => Reaction.Cry
  | FamilyMember.Father => Reaction.Laugh

/-- Theorem stating that the contradiction in the scenario exhibits both 
    interpenetration of contradictory sides and specificity --/
theorem scenario_contradiction_characteristics :
  ∃ (c : ContradictionCharacteristics), 
    c.interpenetrating ∧ c.specific := by
  sorry

end NUMINAMATH_CALUDE_scenario_contradiction_characteristics_l3160_316072


namespace NUMINAMATH_CALUDE_slope_of_line_from_equation_l3160_316022

theorem slope_of_line_from_equation (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : x₁ ≠ x₂)
  (h₂ : (3 : ℝ) / x₁ + (4 : ℝ) / y₁ = 0)
  (h₃ : (3 : ℝ) / x₂ + (4 : ℝ) / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -(4 : ℝ) / 3 := by
sorry

end NUMINAMATH_CALUDE_slope_of_line_from_equation_l3160_316022


namespace NUMINAMATH_CALUDE_abs_x_lt_2_sufficient_not_necessary_for_quadratic_l3160_316068

theorem abs_x_lt_2_sufficient_not_necessary_for_quadratic :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ ¬(|x| < 2)) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_lt_2_sufficient_not_necessary_for_quadratic_l3160_316068


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l3160_316018

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  minor_axis_length : b = Real.sqrt 3
  foci_triangle : ∃ (c : ℝ), a = 2 * c ∧ a^2 = b^2 + c^2

/-- The point P -/
def P : ℝ × ℝ := (0, 2)

/-- Theorem about the special ellipse and its properties -/
theorem special_ellipse_properties (E : SpecialEllipse) :
  -- 1. Standard equation
  E.a^2 = 4 ∧ E.b^2 = 3 ∧
  -- 2. Existence of line l
  ∃ (k : ℝ), 
    -- 3. Equation of line l
    (k = Real.sqrt 2 / 2 ∨ k = -Real.sqrt 2 / 2) ∧
    -- Line passes through P and intersects the ellipse at two distinct points
    ∃ (M N : ℝ × ℝ), M ≠ N ∧
      M.1^2 / E.a^2 + M.2^2 / E.b^2 = 1 ∧
      N.1^2 / E.a^2 + N.2^2 / E.b^2 = 1 ∧
      M.2 = k * M.1 + P.2 ∧
      N.2 = k * N.1 + P.2 ∧
      -- Satisfying the dot product condition
      M.1 * N.1 + M.2 * N.2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l3160_316018


namespace NUMINAMATH_CALUDE_complex_power_four_l3160_316051

theorem complex_power_four (i : ℂ) : i^2 = -1 → 2 * i^4 = 2 := by sorry

end NUMINAMATH_CALUDE_complex_power_four_l3160_316051


namespace NUMINAMATH_CALUDE_inequalities_not_equivalent_l3160_316052

theorem inequalities_not_equivalent : 
  ¬(∀ x : ℝ, (x - 3) / (x^2 - 5*x + 6) < 2 ↔ 2*x^2 - 11*x + 15 > 0) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_equivalent_l3160_316052


namespace NUMINAMATH_CALUDE_fraction_inequality_l3160_316085

theorem fraction_inequality (x : ℝ) : (x - 1) / (x + 2) ≥ 0 ↔ x < -2 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3160_316085


namespace NUMINAMATH_CALUDE_total_laundry_pieces_l3160_316046

def start_time : Nat := 8
def end_time : Nat := 12
def pieces_per_hour : Nat := 20

theorem total_laundry_pieces :
  (end_time - start_time) * pieces_per_hour = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_laundry_pieces_l3160_316046


namespace NUMINAMATH_CALUDE_base_conversion_equality_l3160_316092

theorem base_conversion_equality (b : ℕ) : b > 0 → (
  4 * 5 + 3 = 1 * b^2 + 2 * b + 1 ↔ b = 4
) := by sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l3160_316092


namespace NUMINAMATH_CALUDE_point_belongs_to_transformed_plane_l3160_316017

/-- Plane equation coefficients -/
structure PlaneCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Apply similarity transformation to plane equation -/
def transformPlane (p : PlaneCoefficients) (k : ℝ) : PlaneCoefficients :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Check if a point satisfies a plane equation -/
def satisfiesPlane (point : Point3D) (plane : PlaneCoefficients) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Main theorem: Point A belongs to the image of plane a after similarity transformation -/
theorem point_belongs_to_transformed_plane 
  (A : Point3D) 
  (a : PlaneCoefficients) 
  (k : ℝ) 
  (h1 : A.x = 1/2) 
  (h2 : A.y = 1/3) 
  (h3 : A.z = 1) 
  (h4 : a.a = 2) 
  (h5 : a.b = -3) 
  (h6 : a.c = 3) 
  (h7 : a.d = -2) 
  (h8 : k = 1.5) : 
  satisfiesPlane A (transformPlane a k) :=
sorry

end NUMINAMATH_CALUDE_point_belongs_to_transformed_plane_l3160_316017


namespace NUMINAMATH_CALUDE_card_selection_count_l3160_316055

/-- Represents a standard deck of cards -/
def StandardDeck : Nat := 52

/-- Number of suits in a standard deck -/
def NumSuits : Nat := 4

/-- Number of ranks in a standard deck -/
def NumRanks : Nat := 13

/-- Number of cards to be chosen -/
def CardsToChoose : Nat := 5

/-- Number of cards that must be of the same suit -/
def SameSuitCards : Nat := 2

/-- Number of cards that must be of different suits -/
def DiffSuitCards : Nat := 3

theorem card_selection_count : 
  (Nat.choose NumSuits 1) * 
  (Nat.choose NumRanks SameSuitCards) * 
  (Nat.choose (NumSuits - 1) DiffSuitCards) * 
  ((Nat.choose (NumRanks - SameSuitCards) 1) ^ DiffSuitCards) = 414384 := by
  sorry

end NUMINAMATH_CALUDE_card_selection_count_l3160_316055


namespace NUMINAMATH_CALUDE_function_coefficient_l3160_316090

theorem function_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = a * x^3 - 2*x) →
  f (-1) = 4 →
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_function_coefficient_l3160_316090


namespace NUMINAMATH_CALUDE_length_of_PQ_l3160_316099

/-- The circle C with center (3, 2) and radius 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 2)^2 = 1}

/-- The line L defined by y = (3/4)x -/
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = (3/4) * p.1}

/-- The intersection points of C and L -/
def intersection := C ∩ L

/-- Assuming the intersection contains exactly two points -/
axiom two_intersection_points : ∃ P Q : ℝ × ℝ, P ≠ Q ∧ intersection = {P, Q}

/-- The length of the line segment PQ -/
noncomputable def PQ_length : ℝ := sorry

/-- The main theorem: The length of PQ is 4√6/5 -/
theorem length_of_PQ : PQ_length = 4 * Real.sqrt 6 / 5 := by sorry

end NUMINAMATH_CALUDE_length_of_PQ_l3160_316099


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l3160_316057

-- Define the complex plane
def ComplexPlane := ℂ

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the condition for the set of points
def SatisfiesCondition (z : ℂ) : Prop :=
  Complex.abs (z - i) + Complex.abs (z + i) = 3

-- Define the set of points satisfying the condition
def PointSet : Set ℂ :=
  {z : ℂ | SatisfiesCondition z}

-- Theorem statement
theorem trajectory_is_ellipse :
  ∃ (a b : ℝ) (center : ℂ), 
    a > 0 ∧ b > 0 ∧ a ≠ b ∧
    PointSet = {z : ℂ | (z.re - center.re)^2 / a^2 + (z.im - center.im)^2 / b^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l3160_316057


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3160_316074

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_arith : a 3 - 3 * a 1 = a 2 - a 3) :
  q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3160_316074


namespace NUMINAMATH_CALUDE_units_digit_of_4659_to_157_l3160_316096

theorem units_digit_of_4659_to_157 :
  (4659^157) % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_4659_to_157_l3160_316096


namespace NUMINAMATH_CALUDE_det_A_squared_minus_3A_l3160_316098

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_A_squared_minus_3A : Matrix.det ((A ^ 2) - 3 • A) = 88 := by
  sorry

end NUMINAMATH_CALUDE_det_A_squared_minus_3A_l3160_316098


namespace NUMINAMATH_CALUDE_f_at_2_l3160_316067

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_at_2 (a b : ℝ) : f a b (-2) = 3 → f a b 2 = -19 := by sorry

end NUMINAMATH_CALUDE_f_at_2_l3160_316067


namespace NUMINAMATH_CALUDE_quadratic_general_form_l3160_316094

theorem quadratic_general_form :
  ∀ x : ℝ, x^2 = 3*x + 1 ↔ x^2 - 3*x - 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_general_form_l3160_316094


namespace NUMINAMATH_CALUDE_gcf_252_96_l3160_316012

theorem gcf_252_96 : Nat.gcd 252 96 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_252_96_l3160_316012


namespace NUMINAMATH_CALUDE_max_chocolates_ben_l3160_316004

theorem max_chocolates_ben (total : ℕ) (ben carol : ℕ) (k : ℕ) : 
  total = 30 →
  ben + carol = total →
  carol = k * ben →
  k > 0 →
  ben ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_max_chocolates_ben_l3160_316004


namespace NUMINAMATH_CALUDE_remainder_2019_pow_2018_mod_100_l3160_316084

theorem remainder_2019_pow_2018_mod_100 : 2019^2018 ≡ 41 [ZMOD 100] := by sorry

end NUMINAMATH_CALUDE_remainder_2019_pow_2018_mod_100_l3160_316084


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l3160_316013

def polynomial (x : ℝ) : ℝ := 4 * (2 * x^8 + 3 * x^5 - 5) + 6 * (x^6 - 5 * x^3 + 4)

theorem sum_of_coefficients_is_zero : 
  polynomial 1 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l3160_316013


namespace NUMINAMATH_CALUDE_first_month_sale_correct_l3160_316062

/-- Represents the sales data for a grocery shop -/
structure SalesData where
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ
  average : ℕ

/-- Calculates the sale in the first month given the sales data -/
def calculate_first_month_sale (data : SalesData) : ℕ :=
  data.average * 6 - (data.month2 + data.month3 + data.month4 + data.month5 + data.month6)

/-- Theorem stating that the calculated first month sale is correct -/
theorem first_month_sale_correct (data : SalesData) 
  (h : data = { month2 := 6927, month3 := 6855, month4 := 7230, month5 := 6562, 
                month6 := 5091, average := 6500 }) : 
  calculate_first_month_sale data = 6335 := by
  sorry

#eval calculate_first_month_sale { month2 := 6927, month3 := 6855, month4 := 7230, 
                                   month5 := 6562, month6 := 5091, average := 6500 }

end NUMINAMATH_CALUDE_first_month_sale_correct_l3160_316062


namespace NUMINAMATH_CALUDE_ascending_order_abc_l3160_316087

theorem ascending_order_abc : 
  let a := (Real.sqrt 2 / 2) * (Real.sin (17 * π / 180) + Real.cos (17 * π / 180))
  let b := 2 * (Real.cos (13 * π / 180))^2 - 1
  let c := Real.sqrt 3 / 2
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ascending_order_abc_l3160_316087


namespace NUMINAMATH_CALUDE_honey_eaten_by_bears_l3160_316089

theorem honey_eaten_by_bears (initial_honey : Real) (remaining_honey : Real)
  (h1 : initial_honey = 0.36)
  (h2 : remaining_honey = 0.31) :
  initial_honey - remaining_honey = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_honey_eaten_by_bears_l3160_316089


namespace NUMINAMATH_CALUDE_rectangular_views_imply_prism_or_cylinder_l3160_316076

/-- A solid object in 3D space -/
structure Solid where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Front view of a solid -/
def frontView (s : Solid) : Set (ℝ × ℝ) :=
  sorry

/-- Side view of a solid -/
def sideView (s : Solid) : Set (ℝ × ℝ) :=
  sorry

/-- Predicate to check if a set is a rectangle -/
def isRectangle (s : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a solid is a rectangular prism -/
def isRectangularPrism (s : Solid) : Prop :=
  sorry

/-- Predicate to check if a solid is a cylinder -/
def isCylinder (s : Solid) : Prop :=
  sorry

/-- Theorem: If a solid has rectangular front and side views, it can be either a rectangular prism or a cylinder -/
theorem rectangular_views_imply_prism_or_cylinder (s : Solid) :
  isRectangle (frontView s) → isRectangle (sideView s) →
  isRectangularPrism s ∨ isCylinder s :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_views_imply_prism_or_cylinder_l3160_316076


namespace NUMINAMATH_CALUDE_merry_go_round_time_l3160_316015

theorem merry_go_round_time (dave_time chuck_time erica_time : ℝ) : 
  dave_time = 10 →
  chuck_time = 5 * dave_time →
  erica_time = chuck_time * 1.3 →
  erica_time = 65 := by
sorry

end NUMINAMATH_CALUDE_merry_go_round_time_l3160_316015


namespace NUMINAMATH_CALUDE_alcohol_solution_concentration_l3160_316080

/-- Given a 6-liter solution that is 40% alcohol, prove that adding 1.2 liters
    of pure alcohol will result in a solution that is 50% alcohol. -/
theorem alcohol_solution_concentration (initial_volume : ℝ) (initial_concentration : ℝ)
    (added_alcohol : ℝ) (target_concentration : ℝ) :
  initial_volume = 6 →
  initial_concentration = 0.4 →
  added_alcohol = 1.2 →
  target_concentration = 0.5 →
  (initial_volume * initial_concentration + added_alcohol) /
    (initial_volume + added_alcohol) = target_concentration := by
  sorry

#check alcohol_solution_concentration

end NUMINAMATH_CALUDE_alcohol_solution_concentration_l3160_316080


namespace NUMINAMATH_CALUDE_trumpet_trombone_difference_l3160_316078

/-- Represents the number of players for each instrument in the school band --/
structure BandComposition where
  flute : Nat
  trumpet : Nat
  trombone : Nat
  drummer : Nat
  clarinet : Nat
  french_horn : Nat

/-- Theorem stating the difference between trumpet and trombone players --/
theorem trumpet_trombone_difference (band : BandComposition) : 
  band.flute = 5 →
  band.trumpet = 3 * band.flute →
  band.trumpet > band.trombone →
  band.drummer = band.trombone + 11 →
  band.clarinet = 2 * band.flute →
  band.french_horn = band.trombone + 3 →
  band.flute + band.trumpet + band.trombone + band.drummer + band.clarinet + band.french_horn = 65 →
  band.trumpet - band.trombone = 8 := by
  sorry


end NUMINAMATH_CALUDE_trumpet_trombone_difference_l3160_316078


namespace NUMINAMATH_CALUDE_sequence_inequality_l3160_316029

/-- S(n,m) is the number of sequences of length n consisting of 0 and 1 
    where there exists a 0 in any consecutive m digits -/
def S (n m : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem sequence_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  S (2015 * n) n * S (2015 * m) m ≥ S (2015 * n) m * S (2015 * m) n := by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3160_316029


namespace NUMINAMATH_CALUDE_notebook_cost_l3160_316036

/-- Proves that the cost of each notebook is $1 given the conditions of Léa's purchases --/
theorem notebook_cost (book_cost : ℚ) (binder_cost : ℚ) (binder_count : ℕ) (notebook_count : ℕ) (total_cost : ℚ) :
  book_cost = 16 →
  binder_cost = 2 →
  binder_count = 3 →
  notebook_count = 6 →
  total_cost = 28 →
  (total_cost - (book_cost + binder_cost * binder_count)) / notebook_count = 1 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l3160_316036


namespace NUMINAMATH_CALUDE_celsius_to_fahrenheit_constant_is_zero_l3160_316033

/-- The conversion factor from Celsius to Fahrenheit -/
def celsius_to_fahrenheit_factor : ℚ := 9 / 5

/-- The change in Fahrenheit temperature -/
def fahrenheit_change : ℚ := 26

/-- The change in Celsius temperature -/
def celsius_change : ℚ := 14.444444444444445

/-- The constant in the Celsius to Fahrenheit conversion formula -/
def celsius_to_fahrenheit_constant : ℚ := 0

theorem celsius_to_fahrenheit_constant_is_zero :
  celsius_to_fahrenheit_constant = 0 := by sorry

end NUMINAMATH_CALUDE_celsius_to_fahrenheit_constant_is_zero_l3160_316033


namespace NUMINAMATH_CALUDE_square_side_length_l3160_316006

theorem square_side_length (r s : ℕ) : 
  (2*r + s = 2000) →
  (2*r + 5*s = 3030) →
  s = 258 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l3160_316006


namespace NUMINAMATH_CALUDE_coordinates_are_precise_l3160_316025

-- Define a type for location descriptions
inductive LocationDescription
  | indoor : String → String → String → LocationDescription  -- Building, room, etc.
  | roadSection : String → LocationDescription  -- Road name
  | coordinates : Float → Float → LocationDescription  -- Longitude and Latitude
  | direction : Float → String → LocationDescription  -- Angle and cardinal direction

-- Function to check if a location description is precise
def isPreciseLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.coordinates _ _ => True
  | _ => False

-- Theorem statement
theorem coordinates_are_precise (locations : List LocationDescription) :
  ∃ (loc : LocationDescription), loc ∈ locations ∧ isPreciseLocation loc ↔
    ∃ (lon lat : Float), LocationDescription.coordinates lon lat ∈ locations :=
sorry

end NUMINAMATH_CALUDE_coordinates_are_precise_l3160_316025


namespace NUMINAMATH_CALUDE_polynomial_identity_l3160_316001

theorem polynomial_identity (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃)) :
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3160_316001


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3160_316073

def repeating_decimal : ℚ := 0.157142857142857

theorem repeating_decimal_as_fraction :
  repeating_decimal = 10690 / 68027 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l3160_316073


namespace NUMINAMATH_CALUDE_equation_proof_l3160_316093

theorem equation_proof : 121 + 2 * 11 * 8 + 64 = 361 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3160_316093


namespace NUMINAMATH_CALUDE_perimeter_bounds_l3160_316056

/-- A quadrilateral inscribed in a circle with specific properties -/
structure InscribedQuadrilateral where
  AB : ℕ+
  BC : ℕ+
  CD : ℕ+
  DA : ℕ+
  DA_eq_2005 : DA = 2005
  right_angles : True  -- Represents ∠ABC = ∠ADC = 90°
  max_side_lt_2005 : max AB BC < 2005 ∧ max (max AB BC) CD < 2005

/-- The perimeter of the quadrilateral -/
def perimeter (q : InscribedQuadrilateral) : ℕ :=
  q.AB.val + q.BC.val + q.CD.val + q.DA.val

/-- Theorem stating the bounds on the perimeter -/
theorem perimeter_bounds (q : InscribedQuadrilateral) :
  4160 ≤ perimeter q ∧ perimeter q ≤ 7772 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_bounds_l3160_316056


namespace NUMINAMATH_CALUDE_calculate_expression_l3160_316008

theorem calculate_expression : (1 / 3 : ℚ) * 9 * 15 - 7 = 38 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l3160_316008


namespace NUMINAMATH_CALUDE_a_2008_mod_4_l3160_316032

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a n * sequence_a (n + 1) + 1

theorem a_2008_mod_4 : sequence_a 2008 % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_2008_mod_4_l3160_316032


namespace NUMINAMATH_CALUDE_circle_ratio_l3160_316026

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  π * b^2 - π * a^2 = 4 * (π * a^2) → a / b = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l3160_316026


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3160_316095

theorem min_value_quadratic (x : ℝ) :
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 5*x^2 + 20*x + 25 → y ≥ y_min ∧ ∃ (x_min : ℝ), 5*x_min^2 + 20*x_min + 25 = y_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3160_316095


namespace NUMINAMATH_CALUDE_triangle_theorem_l3160_316002

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi
  sine_law : a / (Real.sin A) = b / (Real.sin B)
  cosine_law : a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h : 2 * t.a * Real.sin t.A = (2 * t.b - t.c) * Real.sin t.B + (2 * t.c - t.b) * Real.sin t.C) :
  t.A = Real.pi / 3 ∧ 
  (Real.sin t.B + Real.sin t.C = Real.sqrt 3 → t.A = t.B ∧ t.B = t.C) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3160_316002


namespace NUMINAMATH_CALUDE_perimeter_ABCD_l3160_316047

-- Define the points A, B, C, D, E
variable (A B C D E : ℝ × ℝ)

-- Define the properties of the triangles
def is_right_angled (X Y Z : ℝ × ℝ) : Prop := sorry
def angle_equals_45_deg (X Y Z : ℝ × ℝ) : Prop := sorry
def is_45_45_90_triangle (X Y Z : ℝ × ℝ) : Prop := sorry

-- Define the distance function
def distance (X Y : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter function for a quadrilateral
def perimeter_quadrilateral (W X Y Z : ℝ × ℝ) : ℝ :=
  distance W X + distance X Y + distance Y Z + distance Z W

-- State the theorem
theorem perimeter_ABCD (h1 : is_right_angled A B E)
                       (h2 : is_right_angled B C E)
                       (h3 : is_right_angled C D E)
                       (h4 : angle_equals_45_deg A E B)
                       (h5 : angle_equals_45_deg B E C)
                       (h6 : angle_equals_45_deg C E D)
                       (h7 : distance A E = 32)
                       (h8 : is_45_45_90_triangle A B E)
                       (h9 : is_45_45_90_triangle B C E)
                       (h10 : is_45_45_90_triangle C D E) :
  perimeter_quadrilateral A B C D = 32 + 32 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_perimeter_ABCD_l3160_316047


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l3160_316041

theorem price_decrease_percentage (original_price new_price : ℚ) 
  (h1 : original_price = 1400)
  (h2 : new_price = 1064) :
  (original_price - new_price) / original_price * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l3160_316041


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l3160_316063

theorem reciprocal_of_sum : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l3160_316063


namespace NUMINAMATH_CALUDE_number_of_red_balls_l3160_316010

theorem number_of_red_balls (yellow_balls : ℕ) (white_balls : ℕ) (prob_yellow : ℚ) :
  yellow_balls = 18 →
  white_balls = 9 →
  prob_yellow = 3/10 →
  ∃ n : ℕ, (yellow_balls : ℚ) / (yellow_balls + white_balls + n) = prob_yellow ∧ n = 42 :=
by sorry

end NUMINAMATH_CALUDE_number_of_red_balls_l3160_316010


namespace NUMINAMATH_CALUDE_correct_staffing_arrangements_l3160_316077

def total_members : ℕ := 6
def positions_to_fill : ℕ := 4
def restricted_members : ℕ := 2
def restricted_positions : ℕ := 1

def staffing_arrangements (n m k r : ℕ) : ℕ :=
  (n.factorial / (n - m).factorial) - k * ((n - 1).factorial / (n - m).factorial)

theorem correct_staffing_arrangements :
  staffing_arrangements total_members positions_to_fill restricted_members restricted_positions = 240 := by
  sorry

end NUMINAMATH_CALUDE_correct_staffing_arrangements_l3160_316077


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3160_316039

open Real

theorem trigonometric_identities (α : ℝ) 
  (h : (sin α - 2 * cos α) / (sin α + 2 * cos α) = 3) : 
  ((sin α + 2 * cos α) / (5 * cos α - sin α) = -2/9) ∧
  ((sin α + cos α)^2 = 9/17) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3160_316039


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3160_316091

/-- Given the following definitions:
  a = 2√2 + 3√3 + 4√6
  b = -2√2 + 3√3 + 4√6
  c = 2√2 - 3√3 + 4√6
  d = -2√2 - 3√3 + 4√6
  Prove that (1/a + 1/b + 1/c + 1/d)² = 952576/70225 -/
theorem sum_of_reciprocals_squared (a b c d : ℝ) :
  a = 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  b = -2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  c = 2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  d = -2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  (1/a + 1/b + 1/c + 1/d)^2 = 952576/70225 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3160_316091


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l3160_316038

theorem trigonometric_expression_equality : 
  (Real.sin (15 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (5 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 
  1 / (2 * Real.cos (25 * π / 180)) := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l3160_316038


namespace NUMINAMATH_CALUDE_troy_needs_ten_dollars_l3160_316016

/-- The amount of additional money Troy needs to buy a new computer -/
def additional_money_needed (new_computer_cost initial_savings old_computer_price : ℕ) : ℕ :=
  new_computer_cost - (initial_savings + old_computer_price)

/-- Theorem: Troy needs $10 more to buy the new computer -/
theorem troy_needs_ten_dollars : 
  additional_money_needed 80 50 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_troy_needs_ten_dollars_l3160_316016


namespace NUMINAMATH_CALUDE_ping_pong_meeting_l3160_316024

theorem ping_pong_meeting (total_legs : ℕ) (square_stool_legs round_stool_legs : ℕ) :
  total_legs = 33 ∧ square_stool_legs = 4 ∧ round_stool_legs = 3 →
  ∃ (total_members square_stools round_stools : ℕ),
    total_members = square_stools + round_stools ∧
    total_members * 2 + square_stools * square_stool_legs + round_stools * round_stool_legs = total_legs ∧
    total_members = 6 :=
by sorry

end NUMINAMATH_CALUDE_ping_pong_meeting_l3160_316024


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l3160_316086

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel 
  (l m : Line) (α : Plane) 
  (h1 : l ≠ m) 
  (h2 : perp l α) 
  (h3 : perp m α) : 
  parallel l m :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l3160_316086


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3160_316034

theorem vector_difference_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : ‖a + b‖ = Real.sqrt 19) : 
  ‖a - b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3160_316034


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l3160_316031

theorem min_value_sum_of_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_9 : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l3160_316031


namespace NUMINAMATH_CALUDE_coefficient_of_x_is_14_l3160_316040

def expression (x : ℝ) : ℝ := 2 * (x - 6) + 5 * (3 - 3 * x^2 + 6 * x) - 6 * (3 * x - 5)

theorem coefficient_of_x_is_14 : 
  ∃ a b c : ℝ, ∀ x : ℝ, expression x = a * x^2 + 14 * x + c :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_is_14_l3160_316040


namespace NUMINAMATH_CALUDE_child_ticket_cost_l3160_316021

/-- Proves that the cost of a child ticket is $3.50 given the specified conditions -/
theorem child_ticket_cost (adult_price : ℝ) (total_tickets : ℕ) (total_cost : ℝ) (adult_tickets : ℕ) : ℝ :=
  let child_tickets := total_tickets - adult_tickets
  let child_price := (total_cost - (adult_price * adult_tickets)) / child_tickets
  by
    -- Assuming:
    have h1 : adult_price = 5.50 := by sorry
    have h2 : total_tickets = 21 := by sorry
    have h3 : total_cost = 83.50 := by sorry
    have h4 : adult_tickets = 5 := by sorry

    -- Proof goes here
    sorry

    -- Conclusion
    -- child_price = 3.50

end NUMINAMATH_CALUDE_child_ticket_cost_l3160_316021


namespace NUMINAMATH_CALUDE_exp_gt_m_ln_plus_two_l3160_316044

theorem exp_gt_m_ln_plus_two (x m : ℝ) (hx : x > 0) (hm : 0 < m) (hm1 : m ≤ 1) :
  Real.exp x > m * (Real.log x + 2) := by
  sorry

end NUMINAMATH_CALUDE_exp_gt_m_ln_plus_two_l3160_316044


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3160_316071

theorem quadratic_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (k - 1) * x^2 - 2 * x + 1 = 0 ∧ (k - 1) * y^2 - 2 * y + 1 = 0) ↔
  (k ≤ 2 ∧ k ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3160_316071


namespace NUMINAMATH_CALUDE_quadratic_roots_mean_l3160_316069

theorem quadratic_roots_mean (b c : ℝ) (r₁ r₂ : ℝ) : 
  (r₁ + r₂) / 2 = 9 →
  (r₁ * r₂).sqrt = 21 →
  r₁ + r₂ = -b →
  r₁ * r₂ = c →
  b = -18 ∧ c = 441 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_mean_l3160_316069


namespace NUMINAMATH_CALUDE_hall_tables_l3160_316088

theorem hall_tables (total_chairs : ℕ) (tables_with_three : ℕ) : 
  total_chairs = 91 → tables_with_three = 5 →
  ∃ (total_tables : ℕ), 
    (total_tables / 2 : ℚ) * 2 + 
    (tables_with_three : ℚ) * 3 + 
    ((total_tables : ℚ) - (total_tables / 2 : ℚ) - (tables_with_three : ℚ)) * 4 = 
    total_chairs ∧ 
    total_tables = 32 := by
  sorry

end NUMINAMATH_CALUDE_hall_tables_l3160_316088


namespace NUMINAMATH_CALUDE_parabola_equation_l3160_316065

/-- A parabola with vertex at the origin and directrix x = -1 has the equation y^2 = 4x -/
theorem parabola_equation (p : ℝ → ℝ → Prop) : 
  (∀ x y, p x y ↔ y^2 = 4*x) → 
  (∀ x, p x 0 ↔ x = 0) →  -- vertex at origin
  (∀ y, p (-1) y ↔ False) →  -- directrix at x = -1
  ∀ x y, p x y ↔ y^2 = 4*x := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3160_316065


namespace NUMINAMATH_CALUDE_brandon_skittles_count_l3160_316060

/-- Given Brandon's initial Skittles count and the number of Skittles he loses,
    prove that his final Skittles count is the difference between the initial count and the number lost. -/
theorem brandon_skittles_count (initial_count lost_count : ℕ) :
  initial_count - lost_count = initial_count - lost_count :=
by sorry

end NUMINAMATH_CALUDE_brandon_skittles_count_l3160_316060


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l3160_316011

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon :
  num_diagonals 25 = 275 := by sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l3160_316011


namespace NUMINAMATH_CALUDE_solve_for_A_l3160_316030

theorem solve_for_A : ∃ A : ℝ, (10 - A = 6) ∧ (A = 4) := by sorry

end NUMINAMATH_CALUDE_solve_for_A_l3160_316030


namespace NUMINAMATH_CALUDE_no_primes_satisfying_conditions_l3160_316049

theorem no_primes_satisfying_conditions : ¬∃ (p q : ℕ), 
  Prime p ∧ Prime q ∧ p > 3 ∧ q > 3 ∧ 
  (q ∣ (p^2 - 1)) ∧ (p ∣ (q^2 - 1)) := by
sorry

end NUMINAMATH_CALUDE_no_primes_satisfying_conditions_l3160_316049


namespace NUMINAMATH_CALUDE_math_books_same_box_probability_l3160_316027

/-- Represents a box with a given capacity -/
structure Box where
  capacity : ℕ

/-- Represents the collection of boxes -/
def boxes : List Box := [⟨4⟩, ⟨5⟩, ⟨6⟩]

/-- Total number of textbooks -/
def total_textbooks : ℕ := 15

/-- Number of mathematics textbooks -/
def math_textbooks : ℕ := 4

/-- Calculates the probability of all mathematics textbooks being in the same box -/
noncomputable def prob_math_books_same_box : ℚ := sorry

/-- Theorem stating the probability of all mathematics textbooks being in the same box -/
theorem math_books_same_box_probability :
  prob_math_books_same_box = 1 / 91 := by sorry

end NUMINAMATH_CALUDE_math_books_same_box_probability_l3160_316027


namespace NUMINAMATH_CALUDE_wheel_probability_l3160_316066

theorem wheel_probability (W X Y Z : ℝ) : 
  W = 3/8 → X = 1/4 → Y = 1/8 → W + X + Y + Z = 1 → Z = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_wheel_probability_l3160_316066


namespace NUMINAMATH_CALUDE_triangular_front_view_solids_l3160_316035

/-- A type representing different types of solids -/
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

/-- A predicate that determines if a solid can have a triangular front view -/
def has_triangular_front_view (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True
  | Solid.Cone => True
  | _ => False

/-- Theorem stating which solids can have a triangular front view -/
theorem triangular_front_view_solids :
  ∀ s : Solid, has_triangular_front_view s ↔ 
    (s = Solid.TriangularPyramid ∨ 
     s = Solid.SquarePyramid ∨ 
     s = Solid.TriangularPrism ∨ 
     s = Solid.Cone) :=
by sorry

end NUMINAMATH_CALUDE_triangular_front_view_solids_l3160_316035


namespace NUMINAMATH_CALUDE_rotation_of_negative_six_minus_three_i_l3160_316019

def rotate90Clockwise (z : ℂ) : ℂ := -Complex.I * z

theorem rotation_of_negative_six_minus_three_i :
  rotate90Clockwise (-6 - 3*Complex.I) = -3 + 6*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_rotation_of_negative_six_minus_three_i_l3160_316019


namespace NUMINAMATH_CALUDE_B_power_150_l3160_316045

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150 : B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end NUMINAMATH_CALUDE_B_power_150_l3160_316045


namespace NUMINAMATH_CALUDE_transportation_budget_theorem_l3160_316058

def total_budget : ℝ := 1200000

def known_percentages : List ℝ := [39, 27, 14, 9, 5, 3.5]

def transportation_percentage : ℝ := 100 - (known_percentages.sum)

theorem transportation_budget_theorem :
  (transportation_percentage = 2.5) ∧
  (transportation_percentage / 100 * 360 = 9) ∧
  (transportation_percentage / 100 * 360 * π / 180 = π / 20) ∧
  (transportation_percentage / 100 * total_budget = 30000) :=
by sorry

end NUMINAMATH_CALUDE_transportation_budget_theorem_l3160_316058


namespace NUMINAMATH_CALUDE_registration_methods_count_l3160_316000

theorem registration_methods_count :
  let num_students : ℕ := 4
  let num_activities : ℕ := 3
  let students_choose_one (s : ℕ) (a : ℕ) : ℕ := a^s
  students_choose_one num_students num_activities = 81 := by
  sorry

end NUMINAMATH_CALUDE_registration_methods_count_l3160_316000


namespace NUMINAMATH_CALUDE_trapezoid_equal_area_segment_l3160_316009

/-- 
Given a trapezoid with the following properties:
- One base is 120 units longer than the other
- The segment joining the midpoints of the legs divides the trapezoid into two regions with area ratio 3:4
- x is the length of the segment parallel to the bases that divides the trapezoid into two equal areas

This theorem states that the greatest integer not exceeding x^2/120 is 217.
-/
theorem trapezoid_equal_area_segment (b : ℝ) (h : ℝ) (x : ℝ) : 
  b > 0 → h > 0 →
  (b + 90) / (b + 30) = 3 / 4 →
  x > 90 →
  2 * ((x - 90) / 120 * h) * (90 + x) = h * (90 + 210) →
  ⌊x^2 / 120⌋ = 217 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_equal_area_segment_l3160_316009


namespace NUMINAMATH_CALUDE_buses_per_week_is_165_l3160_316003

/-- Calculates the number of buses leaving a station in a week -/
def total_buses_per_week (
  weekday_interval : ℕ
  ) (weekday_hours : ℕ
  ) (weekday_count : ℕ
  ) (weekend_interval : ℕ
  ) (weekend_hours : ℕ
  ) (weekend_count : ℕ
  ) : ℕ :=
  let weekday_buses := weekday_count * (weekday_hours * 60 / weekday_interval)
  let weekend_buses := weekend_count * (weekend_hours * 60 / weekend_interval)
  weekday_buses + weekend_buses

/-- Theorem stating that the total number of buses leaving the station in a week is 165 -/
theorem buses_per_week_is_165 :
  total_buses_per_week 40 14 5 20 10 2 = 165 := by
  sorry


end NUMINAMATH_CALUDE_buses_per_week_is_165_l3160_316003


namespace NUMINAMATH_CALUDE_arccos_sin_three_l3160_316054

theorem arccos_sin_three (x : ℝ) : x = Real.arccos (Real.sin 3) → x = 3 - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_three_l3160_316054


namespace NUMINAMATH_CALUDE_limit_to_e_l3160_316028

theorem limit_to_e (x : ℕ → ℝ) (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n| > 1/ε) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |(1 + 1 / x n) ^ (x n) - Real.exp 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_to_e_l3160_316028


namespace NUMINAMATH_CALUDE_total_machine_time_for_dolls_and_accessories_l3160_316079

/-- Calculates the total combined machine operation time for manufacturing dolls and accessories -/
def totalMachineTime (numDolls : ℕ) (numAccessoriesPerDoll : ℕ) (dollTime : ℕ) (accessoryTime : ℕ) : ℕ :=
  numDolls * dollTime + numDolls * numAccessoriesPerDoll * accessoryTime

/-- The number of dolls manufactured -/
def dollCount : ℕ := 12000

/-- The number of accessories per doll -/
def accessoriesPerDoll : ℕ := 2 + 3 + 1 + 5

/-- Time taken to manufacture one doll (in seconds) -/
def dollManufactureTime : ℕ := 45

/-- Time taken to manufacture one accessory (in seconds) -/
def accessoryManufactureTime : ℕ := 10

theorem total_machine_time_for_dolls_and_accessories :
  totalMachineTime dollCount accessoriesPerDoll dollManufactureTime accessoryManufactureTime = 1860000 := by
  sorry

end NUMINAMATH_CALUDE_total_machine_time_for_dolls_and_accessories_l3160_316079


namespace NUMINAMATH_CALUDE_range_of_f_l3160_316023

def f (x : ℝ) : ℝ := |x + 8| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Icc (-5) 17 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l3160_316023


namespace NUMINAMATH_CALUDE_same_sign_range_l3160_316053

theorem same_sign_range (m : ℝ) : 
  ((2 - m) * (|m| - 3) > 0) → (m ∈ Set.Ioo 2 3 ∪ Set.Iio (-3)) := by
  sorry

end NUMINAMATH_CALUDE_same_sign_range_l3160_316053


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3160_316037

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ := {x | -3 < x ∧ x < 2}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, x ∈ solution_set a b c ↔ f a b c x > 0) :
  a < 0 ∧
  a + b + c > 0 ∧
  (∀ x, x ∈ {x | -1/3 < x ∧ x < 1/2} ↔ f c b a x < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3160_316037


namespace NUMINAMATH_CALUDE_b_10_equals_64_l3160_316007

/-- Given two sequences {aₙ} and {bₙ} satisfying certain conditions, prove that b₁₀ = 64 -/
theorem b_10_equals_64 (a b : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a n + a (n + 1) = b n)
  (h3 : ∀ n, a n * a (n + 1) = 2^n) :
  b 10 = 64 := by
  sorry

end NUMINAMATH_CALUDE_b_10_equals_64_l3160_316007


namespace NUMINAMATH_CALUDE_apples_per_pie_l3160_316048

theorem apples_per_pie 
  (initial_apples : ℕ) 
  (handed_out : ℕ) 
  (num_pies : ℕ) 
  (h1 : initial_apples = 62) 
  (h2 : handed_out = 8) 
  (h3 : num_pies = 6) 
  (h4 : num_pies ≠ 0) : 
  (initial_apples - handed_out) / num_pies = 9 := by
sorry

end NUMINAMATH_CALUDE_apples_per_pie_l3160_316048


namespace NUMINAMATH_CALUDE_simple_interest_rate_change_l3160_316061

/-- Given the conditions of a simple interest problem, prove that the new interest rate is 8% -/
theorem simple_interest_rate_change
  (P : ℝ) (R1 T1 SI T2 : ℝ)
  (h1 : R1 = 5)
  (h2 : T1 = 8)
  (h3 : SI = 840)
  (h4 : T2 = 5)
  (h5 : P = (SI * 100) / (R1 * T1))
  (h6 : SI = (P * R1 * T1) / 100)
  (h7 : SI = (P * R2 * T2) / 100)
  : R2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_change_l3160_316061


namespace NUMINAMATH_CALUDE_faulty_clock_correct_time_fraction_l3160_316020

/-- Represents a 12-hour digital clock with a faulty display of '2' as '5' -/
structure FaultyClock where
  /-- The number of hours that display correctly -/
  correct_hours : ℕ
  /-- The number of minutes per hour that display correctly -/
  correct_minutes : ℕ

/-- The fraction of the day during which the faulty clock displays the correct time -/
def correct_time_fraction (clock : FaultyClock) : ℚ :=
  (clock.correct_hours : ℚ) / 12 * (clock.correct_minutes : ℚ) / 60

/-- The specific faulty clock described in the problem -/
def problem_clock : FaultyClock := {
  correct_hours := 10,
  correct_minutes := 44
}

theorem faulty_clock_correct_time_fraction :
  correct_time_fraction problem_clock = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_faulty_clock_correct_time_fraction_l3160_316020


namespace NUMINAMATH_CALUDE_gcf_of_90_and_105_l3160_316064

theorem gcf_of_90_and_105 : Nat.gcd 90 105 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_90_and_105_l3160_316064


namespace NUMINAMATH_CALUDE_chocolate_cake_price_is_12_l3160_316005

/-- The price of a chocolate cake given the order details and total payment -/
def chocolate_cake_price (num_chocolate : ℕ) (num_strawberry : ℕ) (strawberry_price : ℕ) (total_payment : ℕ) : ℕ :=
  (total_payment - num_strawberry * strawberry_price) / num_chocolate

theorem chocolate_cake_price_is_12 :
  chocolate_cake_price 3 6 22 168 = 12 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cake_price_is_12_l3160_316005


namespace NUMINAMATH_CALUDE_equation_solutions_l3160_316075

theorem equation_solutions : 
  (∃ s1 : Set ℝ, s1 = {x : ℝ | x^2 + 2*x - 8 = 0} ∧ s1 = {-4, 2}) ∧ 
  (∃ s2 : Set ℝ, s2 = {x : ℝ | x*(x-2) = x-2} ∧ s2 = {2, 1}) := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3160_316075


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3160_316081

theorem min_value_trig_expression (α γ : ℝ) :
  (3 * Real.cos α + 4 * Real.sin γ - 7)^2 + (3 * Real.sin α + 4 * Real.cos γ - 12)^2 ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3160_316081


namespace NUMINAMATH_CALUDE_cos_555_degrees_l3160_316014

theorem cos_555_degrees : Real.cos (555 * Real.pi / 180) = -(Real.sqrt 6 / 4 + Real.sqrt 2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_cos_555_degrees_l3160_316014


namespace NUMINAMATH_CALUDE_opposite_of_negative_l3160_316083

theorem opposite_of_negative (a : ℝ) : -(- a) = a := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_l3160_316083


namespace NUMINAMATH_CALUDE_problem_statement_l3160_316097

theorem problem_statement (a b : ℝ) 
  (h1 : a + b = -3) 
  (h2 : a^2 * b + a * b^2 = -30) : 
  a^2 - a*b + b^2 + 11 = -10 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3160_316097


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3160_316070

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x - 1 > 0 → x^2 - 1 > 0) ∧
  (∃ x, x^2 - 1 > 0 ∧ ¬(x - 1 > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3160_316070
