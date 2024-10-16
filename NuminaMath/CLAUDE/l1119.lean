import Mathlib

namespace NUMINAMATH_CALUDE_ursula_change_l1119_111974

/-- Calculates the change Ursula received after buying hot dogs and salads -/
theorem ursula_change : 
  let hot_dog_price : ℚ := 3/2  -- $1.50 as a rational number
  let salad_price : ℚ := 5/2    -- $2.50 as a rational number
  let hot_dog_count : ℕ := 5
  let salad_count : ℕ := 3
  let bill_value : ℕ := 10
  let bill_count : ℕ := 2
  
  let total_cost : ℚ := hot_dog_price * hot_dog_count + salad_price * salad_count
  let total_paid : ℕ := bill_value * bill_count
  
  (total_paid : ℚ) - total_cost = 5
  := by sorry

end NUMINAMATH_CALUDE_ursula_change_l1119_111974


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1119_111949

theorem trigonometric_equation_solution (x : Real) : 
  (8.456 * (Real.tan x)^2 * (Real.tan (3*x))^2 * Real.tan (4*x) = 
   (Real.tan x)^2 - (Real.tan (3*x))^2 + Real.tan (4*x)) ↔ 
  (∃ k : Int, x = k * Real.pi ∨ x = (Real.pi / 4) * (2 * k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1119_111949


namespace NUMINAMATH_CALUDE_polar_equation_C_max_area_OAB_l1119_111916

-- Define the curves C, C1, and C2
def C (x y : ℝ) : Prop := x^2 + y^2 = |x| + y ∧ y > 0

def C1 (x y t α : ℝ) : Prop := x = t * Real.cos α ∧ y = t * Real.sin α ∧ t > 0

def C2 (x y t α : ℝ) : Prop := x = -t * Real.sin α ∧ y = t * Real.cos α ∧ t > 0 ∧ 0 < α ∧ α < Real.pi / 2

-- Theorem for the polar coordinate equation of C
theorem polar_equation_C : 
  ∀ (ρ θ : ℝ), 0 < θ ∧ θ < Real.pi → 
  (C (ρ * Real.cos θ) (ρ * Real.sin θ) ↔ ρ = |Real.cos θ| + Real.sin θ) :=
sorry

-- Theorem for the maximum area of triangle OAB
theorem max_area_OAB :
  ∃ (x₁ y₁ x₂ y₂ t₁ t₂ α₁ α₂ : ℝ),
    C x₁ y₁ ∧ C x₂ y₂ ∧
    C1 x₁ y₁ t₁ α₁ ∧ C2 x₂ y₂ t₂ α₂ ∧
    (∀ (x₃ y₃ x₄ y₄ t₃ t₄ α₃ α₄ : ℝ),
      C x₃ y₃ ∧ C x₄ y₄ ∧ C1 x₃ y₃ t₃ α₃ ∧ C2 x₄ y₄ t₄ α₄ →
      (1 / 2 : ℝ) * |x₁ * y₂ - x₂ * y₁| ≥ (1 / 2 : ℝ) * |x₃ * y₄ - x₄ * y₃|) ∧
    (1 / 2 : ℝ) * |x₁ * y₂ - x₂ * y₁| = 1 :=
sorry

end NUMINAMATH_CALUDE_polar_equation_C_max_area_OAB_l1119_111916


namespace NUMINAMATH_CALUDE_four_digit_number_divisibility_l1119_111936

theorem four_digit_number_divisibility (a b c d : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let M := 1000 * a + 100 * b + 10 * c + d
  let N := 1000 * d + 100 * c + 10 * b + a
  (101 ∣ (M + N)) → a + d = b + c :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_divisibility_l1119_111936


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l1119_111969

/-- The value of m for which an ellipse and a hyperbola with given equations have the same foci -/
theorem ellipse_hyperbola_same_foci (m : ℝ) : 
  (∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1 → ∃ c : ℝ, c^2 = 4 - m^2 ∧ (x = c ∨ x = -c) ∧ y = 0) →
  (∀ x y : ℝ, x^2 / m - y^2 / 2 = 1 → ∃ c : ℝ, c^2 = m + 2 ∧ (x = c ∨ x = -c) ∧ y = 0) →
  (∃ c : ℝ, c^2 = 4 - m^2 ∧ c^2 = m + 2) →
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l1119_111969


namespace NUMINAMATH_CALUDE_max_value_theorem_l1119_111996

theorem max_value_theorem (a b : ℝ) 
  (h1 : 0 ≤ a - b ∧ a - b ≤ 1) 
  (h2 : 1 ≤ a + b ∧ a + b ≤ 4) 
  (h3 : ∀ x y : ℝ, 0 ≤ x - y ∧ x - y ≤ 1 → 1 ≤ x + y ∧ x + y ≤ 4 → x - 2*y ≤ a - 2*b) :
  8*a + 2002*b = 8 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1119_111996


namespace NUMINAMATH_CALUDE_inequality_proof_l1119_111915

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1119_111915


namespace NUMINAMATH_CALUDE_cakes_donated_proof_l1119_111956

/-- The number of slices per cake -/
def slices_per_cake : ℕ := 8

/-- The price of each slice in dollars -/
def price_per_slice : ℚ := 1

/-- The donation from the first business owner per slice in dollars -/
def donation1_per_slice : ℚ := 1/2

/-- The donation from the second business owner per slice in dollars -/
def donation2_per_slice : ℚ := 1/4

/-- The total amount raised in dollars -/
def total_raised : ℚ := 140

/-- The number of cakes donated -/
def num_cakes : ℕ := 10

theorem cakes_donated_proof :
  (num_cakes : ℚ) * slices_per_cake * (price_per_slice + donation1_per_slice + donation2_per_slice) = total_raised :=
by sorry

end NUMINAMATH_CALUDE_cakes_donated_proof_l1119_111956


namespace NUMINAMATH_CALUDE_product_of_distinct_solutions_l1119_111935

theorem product_of_distinct_solutions (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 2 / x = y + 2 / y) : x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_solutions_l1119_111935


namespace NUMINAMATH_CALUDE_m_range_l1119_111946

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3| ≤ 2

def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- Define the property that ¬p is a sufficient but not necessary condition for ¬q
def not_p_sufficient_not_necessary_for_not_q (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ (∃ x, ¬(q x m) ∧ p x)

-- Theorem statement
theorem m_range :
  ∀ m : ℝ, not_p_sufficient_not_necessary_for_not_q m ↔ (2 ≤ m ∧ m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_m_range_l1119_111946


namespace NUMINAMATH_CALUDE_mary_rose_garden_l1119_111943

/-- Represents the number of roses in Mary's flower garden and vase. -/
structure RoseGarden where
  R : ℕ  -- Initial number of roses in the garden
  B : ℕ  -- Number of roses left in the garden after cutting
  C : ℕ  -- Number of roses cut from the garden

/-- Theorem about Mary's rose garden -/
theorem mary_rose_garden (garden : RoseGarden) :
  garden.C = 16 - 6 ∧ 
  garden.R = garden.B + garden.C ∧ 
  garden.R - garden.C = garden.B := by
  sorry

#check mary_rose_garden

end NUMINAMATH_CALUDE_mary_rose_garden_l1119_111943


namespace NUMINAMATH_CALUDE_exactly_two_transformations_map_pattern_l1119_111987

/-- A pattern on a line consisting of alternating right-facing and left-facing triangles,
    followed by their vertically flipped versions, creating a symmetric, infinite, repeating pattern. -/
structure TrianglePattern where
  ℓ : Line

/-- Transformations that can be applied to the pattern -/
inductive Transformation
  | Rotate90 : Point → Transformation
  | TranslateParallel : Real → Transformation
  | Rotate120 : Point → Transformation
  | TranslatePerpendicular : Real → Transformation

/-- Predicate to check if a transformation maps the pattern onto itself -/
def maps_onto_self (t : Transformation) (p : TrianglePattern) : Prop :=
  sorry

theorem exactly_two_transformations_map_pattern (p : TrianglePattern) :
  ∃! (ts : Finset Transformation), ts.card = 2 ∧
    (∀ t ∈ ts, maps_onto_self t p) ∧
    (∀ t : Transformation, maps_onto_self t p → t ∈ ts) :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_transformations_map_pattern_l1119_111987


namespace NUMINAMATH_CALUDE_limit_fraction_powers_three_five_l1119_111921

/-- The limit of (3^n + 5^n) / (3^(n-1) + 5^(n-1)) as n approaches infinity is 5 -/
theorem limit_fraction_powers_three_five :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |((3 : ℝ)^n + 5^n) / ((3 : ℝ)^(n-1) + 5^(n-1)) - 5| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_fraction_powers_three_five_l1119_111921


namespace NUMINAMATH_CALUDE_computers_fixed_right_away_l1119_111929

theorem computers_fixed_right_away (total : ℕ) (unfixable_percent : ℚ) (spare_parts_percent : ℚ) :
  total = 20 →
  unfixable_percent = 20 / 100 →
  spare_parts_percent = 40 / 100 →
  (total : ℚ) * (1 - unfixable_percent - spare_parts_percent) = 8 := by
  sorry

end NUMINAMATH_CALUDE_computers_fixed_right_away_l1119_111929


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_of_items_l1119_111953

/-- The total amount Joan spent on toys and clothes -/
def total_spent_on_toys_and_clothes : ℚ := 60.10

/-- The cost of toy cars -/
def toy_cars_cost : ℚ := 14.88

/-- The cost of the skateboard -/
def skateboard_cost : ℚ := 4.88

/-- The cost of toy trucks -/
def toy_trucks_cost : ℚ := 5.86

/-- The cost of pants -/
def pants_cost : ℚ := 14.55

/-- The cost of the shirt -/
def shirt_cost : ℚ := 7.43

/-- The cost of the hat -/
def hat_cost : ℚ := 12.50

/-- Theorem stating that the sum of the costs of toys and clothes equals the total amount spent -/
theorem total_spent_equals_sum_of_items :
  toy_cars_cost + skateboard_cost + toy_trucks_cost + pants_cost + shirt_cost + hat_cost = total_spent_on_toys_and_clothes :=
by sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_of_items_l1119_111953


namespace NUMINAMATH_CALUDE_irrational_numbers_count_l1119_111902

theorem irrational_numbers_count : ∃! (s : Finset ℝ), 
  (∀ x ∈ s, Irrational x ∧ ∃ k : ℤ, (x + 1) / (x^2 - 3*x + 3) = k) ∧ 
  Finset.card s = 2 := by
sorry

end NUMINAMATH_CALUDE_irrational_numbers_count_l1119_111902


namespace NUMINAMATH_CALUDE_total_fruits_after_changes_l1119_111914

def initial_oranges : Nat := 40
def initial_apples : Nat := 25
def initial_bananas : Nat := 15

def removed_oranges : Nat := 37
def added_oranges : Nat := 7
def removed_apples : Nat := 10
def added_bananas : Nat := 12

def final_oranges : Nat := initial_oranges - removed_oranges + added_oranges
def final_apples : Nat := initial_apples - removed_apples
def final_bananas : Nat := initial_bananas + added_bananas

theorem total_fruits_after_changes :
  final_oranges + final_apples + final_bananas = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_after_changes_l1119_111914


namespace NUMINAMATH_CALUDE_problem_statement_l1119_111990

theorem problem_statement (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 7) : 
  x^2*y + x*y^2 = 245/36 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1119_111990


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l1119_111901

theorem greatest_common_divisor_under_60 : ∃ (n : ℕ), 
  n < 60 ∧ 
  n ∣ 546 ∧ 
  n ∣ 108 ∧ 
  (∀ m : ℕ, m < 60 → m ∣ 546 → m ∣ 108 → m ≤ n) ∧
  n = 42 := by
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l1119_111901


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1119_111907

theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1119_111907


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1119_111970

theorem smallest_x_absolute_value_equation : 
  (∀ x : ℝ, |5*x + 15| = 40 → x ≥ -11) ∧ 
  (|5*(-11) + 15| = 40) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1119_111970


namespace NUMINAMATH_CALUDE_existence_of_stabilization_l1119_111999

-- Define the function type
def PositiveIntegerFunction := ℕ+ → ℕ+

-- Define the conditions on the function
def SatisfiesConditions (f : PositiveIntegerFunction) : Prop :=
  (∀ m n : ℕ+, Nat.gcd (f m) (f n) ≤ (Nat.gcd m n) ^ 2014) ∧
  (∀ n : ℕ+, n ≤ f n ∧ f n ≤ n + 2014)

-- State the theorem
theorem existence_of_stabilization (f : PositiveIntegerFunction) 
  (h : SatisfiesConditions f) : 
  ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → f n = n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_stabilization_l1119_111999


namespace NUMINAMATH_CALUDE_expression_factorization_l1119_111930

theorem expression_factorization (x : ℝ) : 
  (7 * x^6 + 36 * x^4 - 8) - (3 * x^6 - 4 * x^4 + 6) = 2 * (2 * x^6 + 20 * x^4 - 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1119_111930


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_cubic_plus_exponential_negation_l1119_111939

theorem negation_of_universal_positive (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem cubic_plus_exponential_negation :
  (¬∀ x : ℝ, x^3 + 3^x > 0) ↔ (∃ x : ℝ, x^3 + 3^x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_cubic_plus_exponential_negation_l1119_111939


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1119_111961

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a := 3 / 4
  let r := 3 / 4
  let n := 15
  geometric_sum a r n = 3177884751 / 1073741824 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1119_111961


namespace NUMINAMATH_CALUDE_sin_150_degrees_l1119_111900

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l1119_111900


namespace NUMINAMATH_CALUDE_squares_theorem_l1119_111954

-- Define the points and lengths
variable (A B C O : ℝ × ℝ)
variable (a b c : ℝ)

-- Define the conditions
def squares_condition (A B C O : ℝ × ℝ) (a b c : ℝ) : Prop :=
  A = (a, a) ∧
  B = (b, 2*a + b) ∧
  C = (-c, c) ∧
  O = (0, 0) ∧
  c = a + b

-- Define the equality of line segments
def line_segments_equal (P Q R S : ℝ × ℝ) : Prop :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (R.1 - S.1)^2 + (R.2 - S.2)^2

-- Define perpendicularity of line segments
def perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (S.1 - R.1) + (Q.2 - P.2) * (S.2 - R.2) = 0

-- State the theorem
theorem squares_theorem (A B C O : ℝ × ℝ) (a b c : ℝ) 
  (h : squares_condition A B C O a b c) : 
  line_segments_equal O B A C ∧ perpendicular O B A C := by
  sorry

end NUMINAMATH_CALUDE_squares_theorem_l1119_111954


namespace NUMINAMATH_CALUDE_optimal_room_configuration_l1119_111994

/-- Represents a configuration of rooms --/
structure RoomConfiguration where
  large_rooms : ℕ
  small_rooms : ℕ

/-- Checks if a given room configuration is valid for the problem --/
def is_valid_configuration (config : RoomConfiguration) : Prop :=
  3 * config.large_rooms + 2 * config.small_rooms = 26

/-- Calculates the total number of rooms in a configuration --/
def total_rooms (config : RoomConfiguration) : ℕ :=
  config.large_rooms + config.small_rooms

/-- Theorem: The optimal room configuration includes exactly one small room --/
theorem optimal_room_configuration :
  ∃ (config : RoomConfiguration),
    is_valid_configuration config ∧
    (∀ (other : RoomConfiguration), is_valid_configuration other →
      total_rooms config ≤ total_rooms other) ∧
    config.small_rooms = 1 :=
sorry

end NUMINAMATH_CALUDE_optimal_room_configuration_l1119_111994


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l1119_111997

-- Define the polynomials p and q
def p (c d x : ℝ) : ℝ := x^3 + c*x + d
def q (c d x : ℝ) : ℝ := x^3 + c*x + d + 360

-- State the theorem
theorem cubic_roots_problem (c d r s : ℝ) : 
  (p c d r = 0 ∧ p c d s = 0 ∧ q c d (r+5) = 0 ∧ q c d (s-4) = 0) → 
  (d = 84 ∨ d = 1260) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l1119_111997


namespace NUMINAMATH_CALUDE_triangle_reconstruction_from_altitude_feet_l1119_111944

/-- A point in the Euclidean plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A triangle in the Euclidean plane -/
structure Triangle := (A B C : Point)

/-- Represents the feet of altitudes of a triangle -/
structure AltitudeFeet := (A₁ B₁ C₁ : Point)

/-- Predicate to check if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Predicate to check if points are the feet of altitudes of a triangle -/
def areFeetOfAltitudes (t : Triangle) (f : AltitudeFeet) : Prop := sorry

/-- Function to reconstruct a triangle from the feet of its altitudes -/
noncomputable def reconstructTriangle (f : AltitudeFeet) : Triangle := sorry

/-- Theorem stating that an acute-angled triangle can be uniquely reconstructed from its altitude feet -/
theorem triangle_reconstruction_from_altitude_feet 
  (t : Triangle) (f : AltitudeFeet) 
  (h1 : isAcuteAngled t) 
  (h2 : areFeetOfAltitudes t f) : 
  ∃! (t' : Triangle), isAcuteAngled t' ∧ areFeetOfAltitudes t' f ∧ t' = t := by
  sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_from_altitude_feet_l1119_111944


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_l1119_111945

theorem arithmetic_geometric_mean_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / 2 = 2 * Real.sqrt 3 ∧ Real.sqrt (x * y) = Real.sqrt 3 → |x - y| = 6 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_l1119_111945


namespace NUMINAMATH_CALUDE_theater_line_arrangements_l1119_111967

theorem theater_line_arrangements (n : ℕ) (h : n = 7) : 
  Nat.factorial n = 5040 := by
  sorry

end NUMINAMATH_CALUDE_theater_line_arrangements_l1119_111967


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l1119_111938

def numbers : List ℕ := [1871, 1997, 2020, 2028, 2113, 2125, 2140, 2222, 2300]

theorem mean_of_remaining_numbers :
  (∃ (subset : List ℕ), subset.length = 7 ∧ subset.sum / 7 = 2100 ∧ subset.toFinset ⊆ numbers.toFinset) →
  (numbers.sum - (2100 * 7)) / 2 = 1158 := by
sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l1119_111938


namespace NUMINAMATH_CALUDE_cloth_sale_meters_l1119_111973

/-- Proves that the number of meters of cloth sold is 75 given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 4950)
    (h2 : profit_per_meter = 15)
    (h3 : cost_price_per_meter = 51) :
    (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_meters_l1119_111973


namespace NUMINAMATH_CALUDE_f_properties_l1119_111931

/-- The function f(x) = -x³ + ax² + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

/-- The function g(x) = f(x) - ax² + 3 -/
def g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x - a*x^2 + 3

/-- The derivative of f(x) -/
def f_derivative (a b : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x + b

theorem f_properties (a b c : ℝ) :
  (f_derivative a b 1 = -3) ∧  -- Tangent line condition
  (f a b c 1 = -2) ∧          -- Point P(1, f(1)) condition
  (∀ x, g a b c x = -g a b c (-x)) →  -- g(x) is an odd function
  (∃ a' b' c', 
    (∀ x, f a' b' c' x = -x^3 - 2*x^2 + 4*x - 3) ∧
    (∀ x, f a' b' c' x ≥ -11) ∧
    (f a' b' c' (-2) = -11) ∧
    (∀ x, f a' b' c' x ≤ -41/27) ∧
    (f a' b' c' (2/3) = -41/27)) := by sorry

end NUMINAMATH_CALUDE_f_properties_l1119_111931


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1119_111989

/-- Given a line ax - by + 1 = 0 (where a > 0 and b > 0) passing through the center of the circle
    x^2 + y^2 + 2x - 4y + 1 = 0, the minimum value of 1/a + 1/b is 3 + 2√2. -/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : a * (-1) - b * 2 + 1 = 0) : 
    (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' * (-1) - b' * 2 + 1 = 0 → 1 / a + 1 / b ≤ 1 / a' + 1 / b') → 
    1 / a + 1 / b = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1119_111989


namespace NUMINAMATH_CALUDE_logarithm_sum_equation_l1119_111991

theorem logarithm_sum_equation (x : ℝ) (h : x > 0) :
  (1 / Real.log x / Real.log 3) + (1 / Real.log x / Real.log 4) + (1 / Real.log x / Real.log 5) = 1 →
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_equation_l1119_111991


namespace NUMINAMATH_CALUDE_total_wall_area_l1119_111903

/-- Represents the properties of tiles and the wall they cover -/
structure TileWall where
  regularTileArea : ℝ
  regularTileCount : ℝ
  jumboTileCount : ℝ
  jumboTileLengthRatio : ℝ

/-- The theorem stating the total wall area given the tile properties -/
theorem total_wall_area (w : TileWall)
  (h1 : w.regularTileArea * w.regularTileCount = 60)
  (h2 : w.jumboTileCount = w.regularTileCount / 3)
  (h3 : w.jumboTileLengthRatio = 3) :
  w.regularTileArea * w.regularTileCount + 
  (w.jumboTileLengthRatio * w.regularTileArea) * w.jumboTileCount = 120 := by
  sorry


end NUMINAMATH_CALUDE_total_wall_area_l1119_111903


namespace NUMINAMATH_CALUDE_james_barbell_cost_l1119_111927

/-- The final cost of James' new barbell purchase -/
def final_barbell_cost (old_barbell_cost : ℝ) (price_increase_rate : ℝ) 
  (sales_tax_rate : ℝ) (trade_in_value : ℝ) : ℝ :=
  let new_barbell_cost := old_barbell_cost * (1 + price_increase_rate)
  let total_cost_with_tax := new_barbell_cost * (1 + sales_tax_rate)
  total_cost_with_tax - trade_in_value

/-- Theorem stating the final cost of James' new barbell -/
theorem james_barbell_cost : 
  final_barbell_cost 250 0.30 0.10 100 = 257.50 := by
  sorry

end NUMINAMATH_CALUDE_james_barbell_cost_l1119_111927


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1119_111909

theorem radical_conjugate_sum_product (c d : ℝ) 
  (h1 : (c + Real.sqrt d) + (c - Real.sqrt d) = -6)
  (h2 : (c + Real.sqrt d) * (c - Real.sqrt d) = 1) :
  4 * c + d = -4 := by
  sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1119_111909


namespace NUMINAMATH_CALUDE_class_average_problem_l1119_111984

theorem class_average_problem (group1_percent : Real) (group1_score : Real)
                              (group2_percent : Real)
                              (group3_percent : Real) (group3_score : Real)
                              (total_average : Real) :
  group1_percent = 0.25 →
  group1_score = 0.8 →
  group2_percent = 0.5 →
  group3_percent = 0.25 →
  group3_score = 0.9 →
  total_average = 0.75 →
  group1_percent + group2_percent + group3_percent = 1 →
  group1_percent * group1_score + group2_percent * (65 / 100) + group3_percent * group3_score = total_average :=
by
  sorry


end NUMINAMATH_CALUDE_class_average_problem_l1119_111984


namespace NUMINAMATH_CALUDE_triangle_side_length_l1119_111924

theorem triangle_side_length 
  (AB : ℝ) 
  (angle_ADB : ℝ) 
  (sin_A : ℝ) 
  (sin_C : ℝ) 
  (h1 : AB = 30)
  (h2 : angle_ADB = Real.pi / 2)
  (h3 : sin_A = 2/3)
  (h4 : sin_C = 1/4) :
  ∃ (DC : ℝ), DC = 20 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1119_111924


namespace NUMINAMATH_CALUDE_work_together_duration_l1119_111981

/-- Given two workers A and B, where A can complete a job in 15 days and B in 20 days,
    this theorem proves that if they work together until 5/12 of the job is left,
    then they worked together for 5 days. -/
theorem work_together_duration (a_rate b_rate : ℚ) (work_left : ℚ) (days_worked : ℕ) :
  a_rate = 1 / 15 →
  b_rate = 1 / 20 →
  work_left = 5 / 12 →
  (a_rate + b_rate) * days_worked = 1 - work_left →
  days_worked = 5 :=
by sorry

end NUMINAMATH_CALUDE_work_together_duration_l1119_111981


namespace NUMINAMATH_CALUDE_town_average_age_l1119_111960

theorem town_average_age (k : ℕ) (h_k : k > 0) : 
  let num_children := 3 * k
  let num_adults := 2 * k
  let avg_age_children := 10
  let avg_age_adults := 40
  let total_population := num_children + num_adults
  let total_age := num_children * avg_age_children + num_adults * avg_age_adults
  (total_age : ℚ) / total_population = 22 :=
by sorry

end NUMINAMATH_CALUDE_town_average_age_l1119_111960


namespace NUMINAMATH_CALUDE_smallest_ab_value_l1119_111982

theorem smallest_ab_value (a b : ℤ) (h : (a : ℚ) / 2 + (b : ℚ) / 1009 = 1 / 2018) :
  ∃ (a₀ b₀ : ℤ), (a₀ : ℚ) / 2 + (b₀ : ℚ) / 1009 = 1 / 2018 ∧ |a₀ * b₀| = 504 ∧
    ∀ (a' b' : ℤ), (a' : ℚ) / 2 + (b' : ℚ) / 1009 = 1 / 2018 → |a' * b'| ≥ 504 :=
by sorry

end NUMINAMATH_CALUDE_smallest_ab_value_l1119_111982


namespace NUMINAMATH_CALUDE_relationship_between_sum_and_product_l1119_111988

theorem relationship_between_sum_and_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, a > 0 → b > 0 → (a * b > 1 → a + b > 1)) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b > 1 ∧ a * b ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_relationship_between_sum_and_product_l1119_111988


namespace NUMINAMATH_CALUDE_three_sequences_comparison_l1119_111951

theorem three_sequences_comparison 
  (a b c : ℕ → ℕ) : 
  ∃ m n : ℕ, m ≠ n ∧ 
    a m ≥ a n ∧ 
    b m ≥ b n ∧ 
    c m ≥ c n :=
by sorry

end NUMINAMATH_CALUDE_three_sequences_comparison_l1119_111951


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l1119_111905

/-- Proves that the fencing cost per meter is 60 cents for a rectangular park with given conditions -/
theorem fencing_cost_per_meter (length width : ℝ) (area perimeter total_cost : ℝ) : 
  length / width = 3 / 2 →
  area = 3750 →
  area = length * width →
  perimeter = 2 * (length + width) →
  total_cost = 150 →
  (total_cost / perimeter) * 100 = 60 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l1119_111905


namespace NUMINAMATH_CALUDE_nine_sided_figure_perimeter_l1119_111932

/-- The perimeter of a regular polygon with n sides of length s is n * s -/
def perimeter (n : ℕ) (s : ℝ) : ℝ := n * s

theorem nine_sided_figure_perimeter :
  let n : ℕ := 9
  let s : ℝ := 2
  perimeter n s = 18 := by sorry

end NUMINAMATH_CALUDE_nine_sided_figure_perimeter_l1119_111932


namespace NUMINAMATH_CALUDE_joe_cars_count_l1119_111979

theorem joe_cars_count (initial_cars new_cars : ℕ) 
  (h1 : initial_cars = 50) 
  (h2 : new_cars = 12) : 
  initial_cars + new_cars = 62 := by
  sorry

end NUMINAMATH_CALUDE_joe_cars_count_l1119_111979


namespace NUMINAMATH_CALUDE_today_is_thursday_l1119_111978

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define when A lies
def A_lies (d : Day) : Prop :=
  d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday

-- Define when B lies
def B_lies (d : Day) : Prop :=
  d = Day.Thursday ∨ d = Day.Friday ∨ d = Day.Saturday

-- Define the previous day
def prev_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday
  | Day.Sunday => Day.Saturday

-- Theorem statement
theorem today_is_thursday : 
  ∃ (d : Day), 
    (A_lies (prev_day d) ↔ ¬(A_lies d)) ∧ 
    (B_lies (prev_day d) ↔ ¬(B_lies d)) ∧ 
    d = Day.Thursday := by
  sorry

end NUMINAMATH_CALUDE_today_is_thursday_l1119_111978


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l1119_111919

theorem salary_reduction_percentage (S : ℝ) (P : ℝ) (h : S > 0) :
  2 * (S - (P / 100 * S)) = S → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l1119_111919


namespace NUMINAMATH_CALUDE_curve_single_intersection_l1119_111985

/-- The curve (x+2y+a)(x^2-y^2)=0 intersects at a single point if and only if a = 0 -/
theorem curve_single_intersection (a : ℝ) : 
  (∃! p : ℝ × ℝ, (p.1 + 2 * p.2 + a) * (p.1^2 - p.2^2) = 0) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_curve_single_intersection_l1119_111985


namespace NUMINAMATH_CALUDE_cuboid_inequality_l1119_111908

theorem cuboid_inequality (x y z : ℝ) (hxy : x < y) (hyz : y < z)
  (p : ℝ) (hp : p = 4 * (x + y + z))
  (s : ℝ) (hs : s = 2 * (x*y + y*z + z*x))
  (d : ℝ) (hd : d = Real.sqrt (x^2 + y^2 + z^2)) :
  x < (1/3) * ((1/4) * p - Real.sqrt (d^2 - (1/2) * s)) ∧
  z > (1/3) * ((1/4) * p + Real.sqrt (d^2 - (1/2) * s)) := by
sorry

end NUMINAMATH_CALUDE_cuboid_inequality_l1119_111908


namespace NUMINAMATH_CALUDE_range_of_m_l1119_111983

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = x * y) :
  (∃ m : ℝ, x + y / 4 < m^2 + 3 * m) ↔ ∃ m : ℝ, m < -4 ∨ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1119_111983


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l1119_111965

/-- Represents a 3x3x3 cube composed of unit cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  diagonal : Real

/-- Represents a plane that bisects the diagonal of the large cube -/
structure BisectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

/-- Represents the number of unit cubes intersected by the bisecting plane -/
def intersected_cubes (cube : LargeCube) (plane : BisectingPlane) : Nat :=
  sorry

/-- Main theorem: A plane perpendicular to and bisecting a space diagonal of a 3x3x3 cube
    intersects exactly 19 of the unit cubes -/
theorem intersected_cubes_count 
  (cube : LargeCube) 
  (plane : BisectingPlane) 
  (h1 : cube.size = 3)
  (h2 : cube.total_cubes = 27)
  (h3 : plane.perpendicular_to_diagonal)
  (h4 : plane.bisects_diagonal) :
  intersected_cubes cube plane = 19 := by
  sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l1119_111965


namespace NUMINAMATH_CALUDE_race_distance_l1119_111922

/-- Prove that the total distance of a race is 240 meters given the specified conditions -/
theorem race_distance (D : ℝ) 
  (h1 : D / 60 * 100 = D + 160) : D = 240 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l1119_111922


namespace NUMINAMATH_CALUDE_geometric_sequence_decreasing_l1119_111972

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a₁ * q^(n-1)

theorem geometric_sequence_decreasing (a₁ q : ℝ) :
  (∀ n : ℕ, geometric_sequence a₁ q (n+1) < geometric_sequence a₁ q n) ↔
  ((a₁ > 0 ∧ 0 < q ∧ q < 1) ∨ (a₁ < 0 ∧ q > 1)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_decreasing_l1119_111972


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l1119_111975

/-- Represents a number in the sequence -/
def SequenceNumber (n : ℕ) : ℕ := 20142015 + n * 10^6

/-- The sum of digits for any number in the sequence -/
def DigitSum : ℕ := 15

/-- A number is a candidate for being a perfect square if its digit sum is 0, 1, 4, 7, or 9 mod 9 -/
def IsPerfectSquareCandidate (n : ℕ) : Prop :=
  n % 9 = 0 ∨ n % 9 = 1 ∨ n % 9 = 4 ∨ n % 9 = 7 ∨ n % 9 = 9

theorem no_perfect_squares_in_sequence :
  ∀ n : ℕ, ¬ ∃ m : ℕ, (SequenceNumber n) = m^2 :=
sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l1119_111975


namespace NUMINAMATH_CALUDE_jellybean_problem_l1119_111992

theorem jellybean_problem :
  ∃ (n : ℕ), 
    n ≥ 150 ∧ 
    n % 17 = 9 ∧ 
    (∀ m : ℕ, m ≥ 150 ∧ m % 17 = 9 → m ≥ n) ∧
    n = 162 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l1119_111992


namespace NUMINAMATH_CALUDE_negative_x_is_directly_proportional_l1119_111968

/-- A function f : ℝ → ℝ is directly proportional if there exists a constant k such that f x = k * x for all x -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function f(x) = -x is directly proportional -/
theorem negative_x_is_directly_proportional :
  DirectlyProportional (fun x => -x) := by
  sorry

#check negative_x_is_directly_proportional

end NUMINAMATH_CALUDE_negative_x_is_directly_proportional_l1119_111968


namespace NUMINAMATH_CALUDE_cos_beta_minus_alpha_l1119_111910

theorem cos_beta_minus_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < 2 * π)
  (h4 : 5 * Real.sin (α - π / 6) = 1) (h5 : 5 * Real.sin (β - π / 6) = 1) :
  Real.cos (β - α) = -23 / 25 := by
sorry

end NUMINAMATH_CALUDE_cos_beta_minus_alpha_l1119_111910


namespace NUMINAMATH_CALUDE_intersections_of_related_functions_l1119_111923

/-- Given a quadratic function that intersects (0, 2) and (1, 1), 
    prove that the related linear function intersects the axes at (1/2, 0) and (0, -1) -/
theorem intersections_of_related_functions 
  (a c : ℝ) 
  (h1 : c = 2) 
  (h2 : a + c = 1) : 
  let f (x : ℝ) := c * x + a
  (f (1/2) = 0 ∧ f 0 = -1) := by
sorry

end NUMINAMATH_CALUDE_intersections_of_related_functions_l1119_111923


namespace NUMINAMATH_CALUDE_equality_holds_iff_l1119_111933

theorem equality_holds_iff (α : ℝ) : 
  Real.sqrt (1 + Real.sin (2 * α)) = Real.sin α + Real.cos α ↔ 
  -π/4 < α ∧ α < 3*π/4 :=
sorry

end NUMINAMATH_CALUDE_equality_holds_iff_l1119_111933


namespace NUMINAMATH_CALUDE_asian_games_mascot_sales_l1119_111962

/-- Asian Games Mascot Sales Problem -/
theorem asian_games_mascot_sales 
  (initial_price : ℝ) 
  (cost_price : ℝ) 
  (initial_sales : ℝ) 
  (price_reduction_factor : ℝ) :
  initial_price = 80 ∧ 
  cost_price = 50 ∧ 
  initial_sales = 200 ∧ 
  price_reduction_factor = 20 →
  ∃ (sales_function : ℝ → ℝ) 
    (profit_function : ℝ → ℝ) 
    (optimal_price : ℝ),
    (∀ x, sales_function x = -20 * x + 1800) ∧
    (profit_function 65 = 7500 ∧ profit_function 75 = 7500) ∧
    (optimal_price = 70 ∧ 
     ∀ x, profit_function x ≤ profit_function optimal_price) :=
by sorry

end NUMINAMATH_CALUDE_asian_games_mascot_sales_l1119_111962


namespace NUMINAMATH_CALUDE_mean_of_xyz_l1119_111977

theorem mean_of_xyz (original_mean : ℝ) (new_mean : ℝ) (x y z : ℝ) : 
  original_mean = 40 →
  new_mean = 50 →
  z = x + 10 →
  (12 * original_mean + x + y + z) / 15 = new_mean →
  (x + y + z) / 3 = 90 := by
sorry

end NUMINAMATH_CALUDE_mean_of_xyz_l1119_111977


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l1119_111955

/-- A prime number greater than 2 -/
def OddPrime (n : ℕ) : Prop := Nat.Prime n ∧ n > 2

theorem triangle_angle_problem :
  ∀ y z w : ℕ,
    OddPrime y →
    OddPrime z →
    OddPrime w →
    y + z + w = 90 →
    (∀ w' : ℕ, OddPrime w' → y + z + w' = 90 → w ≤ w') →
    w = 83 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l1119_111955


namespace NUMINAMATH_CALUDE_no_quadratic_factor_l1119_111952

def p (x : ℝ) : ℝ := x^4 - 6*x^2 + 25

def q₁ (x : ℝ) : ℝ := x^2 - 3*x + 4
def q₂ (x : ℝ) : ℝ := x^2 - 4
def q₃ (x : ℝ) : ℝ := x^2 + 3
def q₄ (x : ℝ) : ℝ := x^2 + 3*x - 4

theorem no_quadratic_factor :
  (∀ x, p x ≠ 0 → q₁ x ≠ 0) ∧
  (∀ x, p x ≠ 0 → q₂ x ≠ 0) ∧
  (∀ x, p x ≠ 0 → q₃ x ≠ 0) ∧
  (∀ x, p x ≠ 0 → q₄ x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_factor_l1119_111952


namespace NUMINAMATH_CALUDE_squared_lengths_sum_l1119_111966

/-- Two circles O and O₁, where O has equation x² + y² = 25 and O₁ has center (m, 0) -/
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 25}
def circle_O₁ (m : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - m)^2 + p.2^2 = (m - 3)^2 + 4^2}

/-- Point P where the circles intersect -/
def P : ℝ × ℝ := (3, 4)

/-- Line l with slope k passing through P -/
def line_l (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 - 4 = k * (p.1 - 3)}

/-- Line l₁ perpendicular to l passing through P -/
def line_l₁ (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 - 4 = (-1/k) * (p.1 - 3)}

/-- Points A and B where line l intersects circles O and O₁ -/
def A (k m : ℝ) : ℝ × ℝ := sorry
def B (k m : ℝ) : ℝ × ℝ := sorry

/-- Points C and D where line l₁ intersects circles O and O₁ -/
def C (k m : ℝ) : ℝ × ℝ := sorry
def D (k m : ℝ) : ℝ × ℝ := sorry

/-- The main theorem -/
theorem squared_lengths_sum (m : ℝ) (k : ℝ) (h : k ≠ 0) :
  ∀ (A B : ℝ × ℝ) (C D : ℝ × ℝ),
  A ∈ circle_O ∧ A ∈ line_l k ∧
  B ∈ circle_O₁ m ∧ B ∈ line_l k ∧
  C ∈ circle_O ∧ C ∈ line_l₁ k ∧
  D ∈ circle_O₁ m ∧ D ∈ line_l₁ k →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (C.1 - D.1)^2 + (C.2 - D.2)^2 = 4 * m^2 := by
  sorry

end NUMINAMATH_CALUDE_squared_lengths_sum_l1119_111966


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1119_111925

theorem arithmetic_equality : 45 * 52 + 28 * 45 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1119_111925


namespace NUMINAMATH_CALUDE_mathborough_rainfall_2006_l1119_111941

/-- The total rainfall in Mathborough for 2006 given the average monthly rainfall in 2005 and the increase in 2006 -/
theorem mathborough_rainfall_2006 
  (avg_2005 : ℝ) 
  (increase_2006 : ℝ) 
  (h1 : avg_2005 = 40) 
  (h2 : increase_2006 = 3) : 
  (avg_2005 + increase_2006) * 12 = 516 := by
  sorry

#check mathborough_rainfall_2006

end NUMINAMATH_CALUDE_mathborough_rainfall_2006_l1119_111941


namespace NUMINAMATH_CALUDE_amiths_age_l1119_111986

theorem amiths_age (a d : ℕ) : 
  (a - 5 = 3 * (d - 5)) → 
  (a + 10 = 2 * (d + 10)) → 
  a = 50 := by
sorry

end NUMINAMATH_CALUDE_amiths_age_l1119_111986


namespace NUMINAMATH_CALUDE_no_sin_4x_function_of_sin_x_l1119_111937

open Real

theorem no_sin_4x_function_of_sin_x : ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, sin (4 * x) = f (sin x) := by
  sorry

end NUMINAMATH_CALUDE_no_sin_4x_function_of_sin_x_l1119_111937


namespace NUMINAMATH_CALUDE_ratio_a_over_4_to_b_over_3_l1119_111940

theorem ratio_a_over_4_to_b_over_3 (a b c : ℝ) 
  (h1 : 3 * a^2 = 4 * b^2)
  (h2 : a * b * (c^2 + 2*c + 1) ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a = 2*c^2 + 3*c + c^(1/2))
  (h5 : b = c^2 + 5*c - c^(3/2)) :
  (a / 4) / (b / 3) = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_over_4_to_b_over_3_l1119_111940


namespace NUMINAMATH_CALUDE_fraction_simplification_l1119_111948

theorem fraction_simplification (m n : ℚ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (24 * m^3 * n^4) / (32 * m^4 * n^2) = (3 * n^2) / (4 * m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1119_111948


namespace NUMINAMATH_CALUDE_g_composition_equals_107_l1119_111964

/-- The function g defined as g(x) = 3x + 2 -/
def g (x : ℝ) : ℝ := 3 * x + 2

/-- Theorem stating that g(g(g(3))) = 107 -/
theorem g_composition_equals_107 : g (g (g 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_equals_107_l1119_111964


namespace NUMINAMATH_CALUDE_norine_retirement_age_l1119_111913

/-- Represents Norine's retirement conditions and calculates her retirement age -/
def norineRetirement (currentAge : ℕ) (yearsWorked : ℕ) (retirementSum : ℕ) : ℕ :=
  let currentSum := currentAge + yearsWorked
  let yearsToRetirement := (retirementSum - currentSum) / 2
  currentAge + yearsToRetirement

/-- Theorem stating that Norine will retire at age 58 given the problem conditions -/
theorem norine_retirement_age :
  norineRetirement 50 19 85 = 58 := by
  sorry

end NUMINAMATH_CALUDE_norine_retirement_age_l1119_111913


namespace NUMINAMATH_CALUDE_point_not_on_line_l1119_111906

theorem point_not_on_line (m b : ℝ) (h : m * b > 0) :
  ¬(0 = m * 1997 + b) :=
sorry

end NUMINAMATH_CALUDE_point_not_on_line_l1119_111906


namespace NUMINAMATH_CALUDE_triangle_configuration_l1119_111912

theorem triangle_configuration (AB : ℝ) (cosA sinC : ℝ) :
  AB = 30 →
  cosA = 4/5 →
  sinC = 4/5 →
  ∃ (DA DB BC DC : ℝ),
    DA = AB * cosA ∧
    DB ^ 2 = AB ^ 2 - DA ^ 2 ∧
    BC = DB / sinC ∧
    DC ^ 2 = BC ^ 2 - DB ^ 2 ∧
    DC = 13.5 := by sorry

end NUMINAMATH_CALUDE_triangle_configuration_l1119_111912


namespace NUMINAMATH_CALUDE_no_prime_solution_l1119_111950

def base_p_to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.foldl (fun acc d => acc * p + d) 0

theorem no_prime_solution :
  ¬∃ p : Nat, Prime p ∧
    (base_p_to_decimal [1, 0, 3, 2] p + 
     base_p_to_decimal [5, 0, 7] p + 
     base_p_to_decimal [2, 1, 4] p + 
     base_p_to_decimal [2, 0, 5] p + 
     base_p_to_decimal [1, 0] p = 
     base_p_to_decimal [4, 2, 3] p + 
     base_p_to_decimal [5, 4, 1] p + 
     base_p_to_decimal [6, 6, 0] p) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_solution_l1119_111950


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1119_111971

def probability_first_die : ℚ := 3 / 8
def probability_second_die : ℚ := 3 / 4

theorem dice_roll_probability :
  probability_first_die * probability_second_die = 9 / 32 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1119_111971


namespace NUMINAMATH_CALUDE_final_painting_height_l1119_111957

/-- Calculates the height of the final painting given the conditions -/
theorem final_painting_height :
  let total_paintings : ℕ := 5
  let total_area : ℝ := 200
  let small_painting_side : ℝ := 5
  let small_painting_count : ℕ := 3
  let large_painting_width : ℝ := 10
  let large_painting_height : ℝ := 8
  let final_painting_width : ℝ := 9
  
  let small_paintings_area : ℝ := small_painting_count * (small_painting_side * small_painting_side)
  let large_painting_area : ℝ := large_painting_width * large_painting_height
  let known_area : ℝ := small_paintings_area + large_painting_area
  let final_painting_area : ℝ := total_area - known_area
  
  final_painting_area / final_painting_width = 5 :=
by sorry

end NUMINAMATH_CALUDE_final_painting_height_l1119_111957


namespace NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l1119_111980

theorem cone_sphere_volume_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 * π * r^2 * h) = (1 / 3) * (4 / 3 * π * r^3) → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l1119_111980


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1119_111917

theorem arithmetic_calculation : 4 * (9 - 3)^2 - 8 = 136 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1119_111917


namespace NUMINAMATH_CALUDE_consecutive_integers_product_336_sum_21_l1119_111926

theorem consecutive_integers_product_336_sum_21 :
  ∃ (x : ℤ), (x * (x + 1) * (x + 2) = 336) ∧ (x + (x + 1) + (x + 2) = 21) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_336_sum_21_l1119_111926


namespace NUMINAMATH_CALUDE_rosie_apple_crisps_l1119_111958

/-- The number of apple crisps Rosie can make with a given number of apples -/
def apple_crisps (apples : ℕ) : ℕ :=
  (3 * apples) / 12

theorem rosie_apple_crisps :
  apple_crisps 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_apple_crisps_l1119_111958


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l1119_111959

theorem complex_equation_real_solution :
  ∀ x : ℝ, (x^2 + Complex.I * x + 6 : ℂ) = (2 * Complex.I + 5 * x : ℂ) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l1119_111959


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l1119_111928

theorem square_garden_perimeter (q p : ℝ) (h1 : q > 0) (h2 : p > 0) :
  q = p + 21 → p = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l1119_111928


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l1119_111947

/-- Given that 3/4 of 16 bananas are worth 12 oranges, 
    prove that 3/5 of 20 bananas are worth 12 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
    (3/4 : ℚ) * 16 * banana_value = 12 * orange_value →
    (3/5 : ℚ) * 20 * banana_value = 12 * orange_value :=
by sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l1119_111947


namespace NUMINAMATH_CALUDE_regression_line_intercept_l1119_111904

theorem regression_line_intercept (x_bar m : ℝ) (y_bar : ℝ) :
  x_bar = m → y_bar = 6 → y_bar = 2 * x_bar + m → m = 2 := by sorry

end NUMINAMATH_CALUDE_regression_line_intercept_l1119_111904


namespace NUMINAMATH_CALUDE_odd_function_implies_a_equals_one_l1119_111976

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = ax^3 + (a-1)x^2 + x -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + (a - 1) * x^2 + x

theorem odd_function_implies_a_equals_one :
  ∀ a : ℝ, IsOdd (f a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_equals_one_l1119_111976


namespace NUMINAMATH_CALUDE_comic_books_count_l1119_111920

theorem comic_books_count (total : ℕ) 
  (h1 : (30 : ℚ) / 100 * total = (total - (70 : ℚ) / 100 * total))
  (h2 : (70 : ℚ) / 100 * total ≥ 120)
  (h3 : ∀ n : ℕ, n < total → (70 : ℚ) / 100 * n < 120) : 
  total = 172 := by
sorry

end NUMINAMATH_CALUDE_comic_books_count_l1119_111920


namespace NUMINAMATH_CALUDE_games_within_division_is_48_l1119_111918

/-- Represents a basketball league with specific game scheduling rules -/
structure BasketballLeague where
  N : ℕ  -- Number of games against each team in own division
  M : ℕ  -- Number of games against each team in other division
  h1 : N > 3 * M
  h2 : M > 5
  h3 : 3 * N + 4 * M = 88

/-- The number of games a team plays within its own division -/
def gamesWithinDivision (league : BasketballLeague) : ℕ := 3 * league.N

/-- Theorem stating the number of games played within a team's own division -/
theorem games_within_division_is_48 (league : BasketballLeague) :
  gamesWithinDivision league = 48 := by
  sorry

#check games_within_division_is_48

end NUMINAMATH_CALUDE_games_within_division_is_48_l1119_111918


namespace NUMINAMATH_CALUDE_wendy_recycling_points_l1119_111995

/-- Calculates the points earned from recycling bags -/
def points_earned (total_bags : ℕ) (unrecycled_bags : ℕ) (points_per_bag : ℕ) : ℕ :=
  (total_bags - unrecycled_bags) * points_per_bag

/-- Proves that Wendy earns 210 points from recycling bags -/
theorem wendy_recycling_points :
  let total_bags : ℕ := 25
  let unrecycled_bags : ℕ := 4
  let points_per_bag : ℕ := 10
  points_earned total_bags unrecycled_bags points_per_bag = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_wendy_recycling_points_l1119_111995


namespace NUMINAMATH_CALUDE_sqrt_ten_plus_three_squared_times_sqrt_ten_minus_three_l1119_111911

theorem sqrt_ten_plus_three_squared_times_sqrt_ten_minus_three :
  (Real.sqrt 10 + 3)^2 * (Real.sqrt 10 - 3) = Real.sqrt 10 + 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ten_plus_three_squared_times_sqrt_ten_minus_three_l1119_111911


namespace NUMINAMATH_CALUDE_tangent_line_parallel_increasing_intervals_decreasing_interval_extreme_values_l1119_111993

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - (2*a + 3)*x + a^2

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - (2*a + 3)

-- Theorem for part 1
theorem tangent_line_parallel (a : ℝ) :
  f_derivative a (-1) = 2 → a = -1/2 := by sorry

-- Theorems for part 2
theorem increasing_intervals :
  let a := -2
  ∀ x, (x < 1/3 ∨ x > 1) → (f_derivative a x > 0) := by sorry

theorem decreasing_interval :
  let a := -2
  ∀ x, (1/3 < x ∧ x < 1) → (f_derivative a x < 0) := by sorry

theorem extreme_values :
  let a := -2
  (f a (1/3) = 112/27) ∧ (f a 1 = 4) := by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_increasing_intervals_decreasing_interval_extreme_values_l1119_111993


namespace NUMINAMATH_CALUDE_smallest_root_of_equation_l1119_111934

theorem smallest_root_of_equation (x : ℝ) : 
  (|x - 1| / x^2 = 6) → (x = -1/2 ∨ x = 1/3) ∧ (-1/2 < 1/3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_equation_l1119_111934


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l1119_111942

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  -- We don't need to define specific dimensions, as they don't affect the result

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- Theorem: The sum of edges, vertices, and faces of any rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) :
  num_edges rp + num_vertices rp + num_faces rp = 26 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_prism_sum_l1119_111942


namespace NUMINAMATH_CALUDE_binomial_divides_lcm_l1119_111963

theorem binomial_divides_lcm (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, k * Nat.choose (2 * n) n = Finset.lcm (Finset.range (2 * n + 1)) id :=
by sorry

end NUMINAMATH_CALUDE_binomial_divides_lcm_l1119_111963


namespace NUMINAMATH_CALUDE_max_sin_theta_is_one_l1119_111998

theorem max_sin_theta_is_one (a b : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0) :
  (∃ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≥ 0 ∧ a * Real.cos θ - b * Real.sin θ ≥ 0) →
  (∃ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≥ 0 ∧ a * Real.cos θ - b * Real.sin θ ≥ 0 ∧
    ∀ φ : ℝ, (a * Real.sin φ + b * Real.cos φ ≥ 0 ∧ a * Real.cos φ - b * Real.sin φ ≥ 0) →
      Real.sin θ ≥ Real.sin φ) →
  (∃ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≥ 0 ∧ a * Real.cos θ - b * Real.sin θ ≥ 0 ∧ Real.sin θ = 1) :=
sorry

end NUMINAMATH_CALUDE_max_sin_theta_is_one_l1119_111998
