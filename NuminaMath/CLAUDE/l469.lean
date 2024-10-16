import Mathlib

namespace NUMINAMATH_CALUDE_decimal_subtraction_equality_l469_46913

def repeating_decimal_789 : ℚ := 789 / 999
def repeating_decimal_456 : ℚ := 456 / 999
def repeating_decimal_123 : ℚ := 123 / 999

theorem decimal_subtraction_equality : 
  repeating_decimal_789 - repeating_decimal_456 - repeating_decimal_123 = 70 / 333 := by
  sorry

end NUMINAMATH_CALUDE_decimal_subtraction_equality_l469_46913


namespace NUMINAMATH_CALUDE_area_of_region_l469_46971

-- Define the lower bound function
def lower_bound (x : ℝ) : ℝ := |x - 4|

-- Define the upper bound function
def upper_bound (x : ℝ) : ℝ := 5 - |x - 2|

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | lower_bound p.1 ≤ p.2 ∧ p.2 ≤ upper_bound p.1}

-- Theorem statement
theorem area_of_region : MeasureTheory.volume region = 10 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l469_46971


namespace NUMINAMATH_CALUDE_inequality_solution_l469_46906

theorem inequality_solution (x : ℝ) : 
  (2*x - 1) / (x + 3) > (3*x - 2) / (x - 4) ↔ x ∈ Set.Ioo (-9 : ℝ) (-3) ∪ Set.Ioo 2 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l469_46906


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l469_46946

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2_a5 : a 2 + a 5 = 12)
  (h_an : ∃ n : ℕ, a n = 25) :
  ∃ n : ℕ, n = 13 ∧ a n = 25 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l469_46946


namespace NUMINAMATH_CALUDE_ten_points_chords_l469_46995

/-- The number of chords connecting n points on a circle -/
def num_chords (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else (n - 1) * n / 2

/-- The property that the number of chords follows the observed pattern -/
axiom chord_pattern : 
  num_chords 2 = 1 ∧ 
  num_chords 3 = 3 ∧ 
  num_chords 4 = 6 ∧ 
  num_chords 5 = 10 ∧ 
  num_chords 6 = 15

/-- Theorem: The number of chords connecting 10 points on a circle is 45 -/
theorem ten_points_chords : num_chords 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_points_chords_l469_46995


namespace NUMINAMATH_CALUDE_B_power_99_l469_46931

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, 1; 0, -1, 0]

theorem B_power_99 : B^99 = !![0, 0, 0; 0, 0, -1; 0, 1, 0] := by sorry

end NUMINAMATH_CALUDE_B_power_99_l469_46931


namespace NUMINAMATH_CALUDE_triangle_coverage_convex_polygon_coverage_l469_46996

-- Define a Circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a Triangle type
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

-- Define a ConvexPolygon type
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)

-- Function to check if a circle covers a point
def covers (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 ≤ c.radius^2

-- Function to check if a set of circles covers a triangle
def covers_triangle (circles : List Circle) (t : Triangle) : Prop :=
  ∀ p : ℝ × ℝ, (p = t.a ∨ p = t.b ∨ p = t.c) → ∃ c ∈ circles, covers c p

-- Function to check if a set of circles covers a convex polygon
def covers_polygon (circles : List Circle) (p : ConvexPolygon) : Prop :=
  ∀ v ∈ p.vertices, ∃ c ∈ circles, covers c v

-- Function to calculate the diameter of a convex polygon
def diameter (p : ConvexPolygon) : ℝ :=
  sorry

-- Theorem for triangle coverage
theorem triangle_coverage (t : Triangle) :
  ∃ circles : List Circle, circles.length ≤ 2 ∧ 
  (∀ c ∈ circles, c.radius = 0.5) ∧ 
  covers_triangle circles t :=
sorry

-- Theorem for convex polygon coverage
theorem convex_polygon_coverage (p : ConvexPolygon) :
  diameter p = 1 →
  ∃ circles : List Circle, circles.length ≤ 3 ∧ 
  (∀ c ∈ circles, c.radius = 0.5) ∧ 
  covers_polygon circles p :=
sorry

end NUMINAMATH_CALUDE_triangle_coverage_convex_polygon_coverage_l469_46996


namespace NUMINAMATH_CALUDE_ryans_initial_funds_l469_46957

/-- Proves that Ryan's initial funds equal the total cost minus crowdfunding amount -/
theorem ryans_initial_funds 
  (average_funding : ℕ) 
  (people_to_recruit : ℕ) 
  (total_cost : ℕ) 
  (h1 : average_funding = 10)
  (h2 : people_to_recruit = 80)
  (h3 : total_cost = 1000) :
  total_cost - (average_funding * people_to_recruit) = 200 := by
  sorry

#check ryans_initial_funds

end NUMINAMATH_CALUDE_ryans_initial_funds_l469_46957


namespace NUMINAMATH_CALUDE_oblique_prism_volume_l469_46917

/-- The volume of an oblique prism with a parallelogram base and inclined lateral edge -/
theorem oblique_prism_volume
  (base_side1 base_side2 lateral_edge : ℝ)
  (base_angle lateral_angle : ℝ)
  (h_base_side1 : base_side1 = 3)
  (h_base_side2 : base_side2 = 6)
  (h_lateral_edge : lateral_edge = 4)
  (h_base_angle : base_angle = Real.pi / 4)  -- 45°
  (h_lateral_angle : lateral_angle = Real.pi / 6)  -- 30°
  : Real.sqrt 6 * 18 = 
    base_side1 * base_side2 * Real.sin base_angle * 
    (lateral_edge * Real.cos lateral_angle) := by
  sorry


end NUMINAMATH_CALUDE_oblique_prism_volume_l469_46917


namespace NUMINAMATH_CALUDE_table_count_l469_46991

theorem table_count (num_books : ℕ) (h : num_books = 100000) :
  ∃ (num_tables : ℕ),
    (num_tables : ℚ) * (2 / 5 * num_tables) = num_books ∧
    num_tables = 500 := by
  sorry

end NUMINAMATH_CALUDE_table_count_l469_46991


namespace NUMINAMATH_CALUDE_arkansas_game_profit_calculation_l469_46901

/-- The amount of money made per t-shirt sold -/
def profit_per_shirt : ℕ := 98

/-- The total number of t-shirts sold during both games -/
def total_shirts_sold : ℕ := 163

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts_sold : ℕ := 89

/-- The money made from selling t-shirts during the Arkansas game -/
def arkansas_game_profit : ℕ := profit_per_shirt * arkansas_shirts_sold

theorem arkansas_game_profit_calculation :
  arkansas_game_profit = 8722 := by sorry

end NUMINAMATH_CALUDE_arkansas_game_profit_calculation_l469_46901


namespace NUMINAMATH_CALUDE_boutique_hats_count_l469_46982

/-- The total number of hats in the shipment -/
def total_hats : ℕ := 120

/-- The number of hats stored -/
def stored_hats : ℕ := 90

/-- The percentage of hats displayed -/
def displayed_percentage : ℚ := 25 / 100

theorem boutique_hats_count :
  total_hats = stored_hats / (1 - displayed_percentage) := by sorry

end NUMINAMATH_CALUDE_boutique_hats_count_l469_46982


namespace NUMINAMATH_CALUDE_range_of_a_l469_46954

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → a ∈ Set.Ioo (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l469_46954


namespace NUMINAMATH_CALUDE_zero_subset_A_l469_46967

def A : Set ℕ := {x | x < 4}

theorem zero_subset_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_zero_subset_A_l469_46967


namespace NUMINAMATH_CALUDE_second_number_value_l469_46966

theorem second_number_value (x y z : ℝ) : 
  z = 4.5 * y →
  y = 2.5 * x →
  (x + y + z) / 3 = 165 →
  y = 82.5 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l469_46966


namespace NUMINAMATH_CALUDE_exists_n_composite_power_of_two_plus_fifteen_l469_46979

theorem exists_n_composite_power_of_two_plus_fifteen :
  ∃ n : ℕ, ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2^n + 15 = a * b :=
by sorry

end NUMINAMATH_CALUDE_exists_n_composite_power_of_two_plus_fifteen_l469_46979


namespace NUMINAMATH_CALUDE_fundraising_contribution_l469_46926

theorem fundraising_contribution
  (total_goal : ℕ)
  (num_participants : ℕ)
  (admin_fee : ℕ)
  (h1 : total_goal = 2400)
  (h2 : num_participants = 8)
  (h3 : admin_fee = 20) :
  (total_goal / num_participants) + admin_fee = 320 := by
sorry

end NUMINAMATH_CALUDE_fundraising_contribution_l469_46926


namespace NUMINAMATH_CALUDE_senate_subcommittee_count_l469_46999

/-- The number of ways to form a subcommittee from a Senate committee -/
def subcommittee_ways (total_republicans : ℕ) (total_democrats : ℕ) 
  (subcommittee_republicans : ℕ) (min_subcommittee_democrats : ℕ) 
  (max_subcommittee_size : ℕ) : ℕ :=
  sorry

theorem senate_subcommittee_count : 
  subcommittee_ways 10 8 3 2 5 = 10080 := by sorry

end NUMINAMATH_CALUDE_senate_subcommittee_count_l469_46999


namespace NUMINAMATH_CALUDE_condition_for_squared_inequality_l469_46904

theorem condition_for_squared_inequality (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_for_squared_inequality_l469_46904


namespace NUMINAMATH_CALUDE_meaningful_expression_l469_46944

/-- The expression x + 1/(x-2) is meaningful for all real x except 2 -/
theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = x + 1 / (x - 2)) ↔ x ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l469_46944


namespace NUMINAMATH_CALUDE_polynomial_equality_l469_46959

/-- Given that 4x^5 + 3x^3 + 2x^2 + p(x) = 6x^3 - 5x^2 + 4x - 2 for all x,
    prove that p(x) = -4x^5 + 3x^3 - 7x^2 + 4x - 2 -/
theorem polynomial_equality (p : ℝ → ℝ) :
  (∀ x, 4 * x^5 + 3 * x^3 + 2 * x^2 + p x = 6 * x^3 - 5 * x^2 + 4 * x - 2) →
  (∀ x, p x = -4 * x^5 + 3 * x^3 - 7 * x^2 + 4 * x - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l469_46959


namespace NUMINAMATH_CALUDE_atomic_weight_X_is_13_l469_46925

/-- The atomic weight of element X in the compound H3XCOOH -/
def atomic_weight_X : ℝ :=
  let atomic_weight_H : ℝ := 1
  let atomic_weight_C : ℝ := 12
  let atomic_weight_O : ℝ := 16
  let molecular_weight : ℝ := 60
  molecular_weight - (3 * atomic_weight_H + atomic_weight_C + 3 * atomic_weight_O)

/-- Theorem stating that the atomic weight of X is 13 -/
theorem atomic_weight_X_is_13 : atomic_weight_X = 13 := by
  sorry

end NUMINAMATH_CALUDE_atomic_weight_X_is_13_l469_46925


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l469_46985

theorem complex_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z ^ 2 = 2 - 3*I) : 
  Complex.abs z ^ 2 = 13/4 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l469_46985


namespace NUMINAMATH_CALUDE_inequality_proof_l469_46911

theorem inequality_proof (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (x + z) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l469_46911


namespace NUMINAMATH_CALUDE_right_angled_iff_sum_radii_right_angled_iff_sum_squared_radii_l469_46976

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  r_a : ℝ
  r_b : ℝ
  r_c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < r ∧ 0 < r_a ∧ 0 < r_b ∧ 0 < r_c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

def is_right_angled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

theorem right_angled_iff_sum_radii (t : Triangle) :
  is_right_angled t ↔ t.r + t.r_a + t.r_b + t.r_c = t.a + t.b + t.c :=
sorry

theorem right_angled_iff_sum_squared_radii (t : Triangle) :
  is_right_angled t ↔ t.r^2 + t.r_a^2 + t.r_b^2 + t.r_c^2 = t.a^2 + t.b^2 + t.c^2 :=
sorry

end NUMINAMATH_CALUDE_right_angled_iff_sum_radii_right_angled_iff_sum_squared_radii_l469_46976


namespace NUMINAMATH_CALUDE_larger_number_of_product_56_sum_15_l469_46916

theorem larger_number_of_product_56_sum_15 (x y : ℕ) : 
  x * y = 56 → x + y = 15 → max x y = 8 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_product_56_sum_15_l469_46916


namespace NUMINAMATH_CALUDE_cyclic_inequality_with_powers_l469_46997

theorem cyclic_inequality_with_powers (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) (h₆ : x₆ > 0) :
  (x₂/x₁)^5 + (x₄/x₂)^5 + (x₆/x₃)^5 + (x₁/x₄)^5 + (x₃/x₅)^5 + (x₅/x₆)^5 ≥ 
  x₁/x₂ + x₂/x₄ + x₃/x₆ + x₄/x₁ + x₅/x₃ + x₆/x₅ := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_with_powers_l469_46997


namespace NUMINAMATH_CALUDE_fourth_root_fifth_power_eighth_l469_46918

theorem fourth_root_fifth_power_eighth : (((5 ^ (1/2)) ^ 5) ^ (1/4)) ^ 8 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_fifth_power_eighth_l469_46918


namespace NUMINAMATH_CALUDE_largest_six_digit_number_with_divisibility_l469_46963

theorem largest_six_digit_number_with_divisibility (A : ℕ) : 
  A ≤ 999999 ∧ 
  A ≥ 100000 ∧
  A % 19 = 0 ∧ 
  (A / 10) % 17 = 0 ∧ 
  (A / 100) % 13 = 0 →
  A ≤ 998412 :=
by sorry

end NUMINAMATH_CALUDE_largest_six_digit_number_with_divisibility_l469_46963


namespace NUMINAMATH_CALUDE_oliver_battle_gremlins_count_l469_46973

/-- Oliver's card collection -/
structure CardCollection where
  monster_club : ℕ
  alien_baseball : ℕ
  battle_gremlins : ℕ

/-- Oliver's card collection satisfies the given conditions -/
def oliver_collection : CardCollection where
  monster_club := 32
  alien_baseball := 16
  battle_gremlins := 48

/-- Theorem: Oliver has 48 Battle Gremlins cards given the conditions -/
theorem oliver_battle_gremlins_count : 
  oliver_collection.battle_gremlins = 48 ∧
  oliver_collection.monster_club = 2 * oliver_collection.alien_baseball ∧
  oliver_collection.battle_gremlins = 3 * oliver_collection.alien_baseball :=
by sorry

end NUMINAMATH_CALUDE_oliver_battle_gremlins_count_l469_46973


namespace NUMINAMATH_CALUDE_last_locker_exists_l469_46933

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents a corridor with a given number of lockers -/
def Corridor (n : Nat) := Fin n → LockerState

/-- Toggles the state of a locker -/
def toggleLocker (state : LockerState) : LockerState :=
  match state with
  | LockerState.Open => LockerState.Closed
  | LockerState.Closed => LockerState.Open

/-- Represents a single pass of toggling lockers with a given step size -/
def togglePass (c : Corridor 512) (step : Nat) : Corridor 512 :=
  sorry

/-- Represents the full toggling process until all lockers are open -/
def fullToggleProcess (c : Corridor 512) : Corridor 512 :=
  sorry

/-- Theorem stating that there exists a last locker to be opened -/
theorem last_locker_exists :
  ∃ (last : Fin 512), 
    ∀ (c : Corridor 512), 
      (fullToggleProcess c last = LockerState.Open) ∧ 
      (∀ (i : Fin 512), i.val > last.val → fullToggleProcess c i = LockerState.Open) :=
sorry

end NUMINAMATH_CALUDE_last_locker_exists_l469_46933


namespace NUMINAMATH_CALUDE_max_value_with_remainder_l469_46984

theorem max_value_with_remainder (A B : ℕ) : 
  A ≠ B → 
  A = 17 * 25 + B → 
  B < 17 → 
  (∀ C : ℕ, C < 17 → 17 * 25 + C ≤ 17 * 25 + B) → 
  A = 441 :=
by sorry

end NUMINAMATH_CALUDE_max_value_with_remainder_l469_46984


namespace NUMINAMATH_CALUDE_max_value_quadratic_l469_46927

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  x^2 + 2*x*y + 3*y^2 ≤ 20 + 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l469_46927


namespace NUMINAMATH_CALUDE_inequality_solution_l469_46936

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x + 2) ≥ 3 / (x - 2) + 7 / 4) ↔ (x ∈ Set.Ioo (-2) 2 ∪ Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l469_46936


namespace NUMINAMATH_CALUDE_total_mileage_scientific_notation_l469_46947

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The total mileage of national expressways -/
def totalMileage : ℕ := 108000

/-- Theorem: The scientific notation of the total mileage is 1.08 × 10^5 -/
theorem total_mileage_scientific_notation :
  ∃ (sn : ScientificNotation), sn.coefficient = 1.08 ∧ sn.exponent = 5 ∧ (sn.coefficient * (10 : ℝ) ^ sn.exponent = totalMileage) :=
sorry

end NUMINAMATH_CALUDE_total_mileage_scientific_notation_l469_46947


namespace NUMINAMATH_CALUDE_hotel_supplies_theorem_l469_46937

/-- The greatest number of bathrooms that can be stocked identically with given supplies -/
def max_bathrooms (toilet_paper soap towels shower_gel : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd (Nat.gcd toilet_paper soap) towels) shower_gel

/-- Theorem stating that the maximum number of bathrooms that can be stocked
    with the given supplies is 6 -/
theorem hotel_supplies_theorem :
  max_bathrooms 36 18 24 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hotel_supplies_theorem_l469_46937


namespace NUMINAMATH_CALUDE_system_solution_unique_l469_46980

theorem system_solution_unique :
  ∃! (x y : ℝ), x^2 + y * Real.sqrt (x * y) = 336 ∧ y^2 + x * Real.sqrt (x * y) = 112 ∧ x = 18 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l469_46980


namespace NUMINAMATH_CALUDE_not_sum_of_three_cubes_l469_46924

theorem not_sum_of_three_cubes : ¬ ∃ (x y z : ℤ), x^3 + y^3 + z^3 = 20042005 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_of_three_cubes_l469_46924


namespace NUMINAMATH_CALUDE_phone_contract_cost_l469_46950

/-- The total cost of buying a phone with a contract -/
def total_cost (phone_price : ℕ) (monthly_fee : ℕ) (contract_months : ℕ) : ℕ :=
  phone_price + monthly_fee * contract_months

/-- Theorem: The total cost of buying 1 phone with a 4-month contract is $30 -/
theorem phone_contract_cost :
  total_cost 2 7 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_phone_contract_cost_l469_46950


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l469_46977

theorem diophantine_equation_solution (x y : ℤ) :
  7 * x - 3 * y = 2 ↔ ∃ k : ℤ, x = 3 * k + 2 ∧ y = 7 * k + 4 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l469_46977


namespace NUMINAMATH_CALUDE_sixth_term_of_special_sequence_l469_46989

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sixth_term_of_special_sequence :
  ∀ (a : ℕ → ℝ),
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 = 2 →
  a 3 = 2 →
  a 4 = 2 →
  a 5 = 2 →
  a 6 = 2 :=
by sorry

end NUMINAMATH_CALUDE_sixth_term_of_special_sequence_l469_46989


namespace NUMINAMATH_CALUDE_cube_halving_l469_46902

theorem cube_halving (r : ℝ) :
  let a := (2 * r) ^ 3
  let a_half := (2 * (r / 2)) ^ 3
  a_half = (1 / 8) * a := by
  sorry

end NUMINAMATH_CALUDE_cube_halving_l469_46902


namespace NUMINAMATH_CALUDE_third_bounce_height_l469_46905

/-- Given an initial height and a bounce ratio, calculates the height of the nth bounce -/
def bounce_height (initial_height : ℝ) (bounce_ratio : ℝ) (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

/-- Converts meters to centimeters -/
def meters_to_cm (meters : ℝ) : ℝ :=
  meters * 100

theorem third_bounce_height :
  let initial_height : ℝ := 12.8
  let bounce_ratio : ℝ := 1/4
  let third_bounce_m := bounce_height initial_height bounce_ratio 3
  meters_to_cm third_bounce_m = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_bounce_height_l469_46905


namespace NUMINAMATH_CALUDE_carly_swimming_time_l469_46939

/-- Carly's swimming practice schedule and total time calculation -/
theorem carly_swimming_time :
  let butterfly_hours_per_day : ℕ := 3
  let butterfly_days_per_week : ℕ := 4
  let backstroke_hours_per_day : ℕ := 2
  let backstroke_days_per_week : ℕ := 6
  let weeks_per_month : ℕ := 4
  
  let butterfly_hours_per_week : ℕ := butterfly_hours_per_day * butterfly_days_per_week
  let backstroke_hours_per_week : ℕ := backstroke_hours_per_day * backstroke_days_per_week
  let total_hours_per_week : ℕ := butterfly_hours_per_week + backstroke_hours_per_week
  let total_hours_per_month : ℕ := total_hours_per_week * weeks_per_month
  
  total_hours_per_month = 96 :=
by
  sorry


end NUMINAMATH_CALUDE_carly_swimming_time_l469_46939


namespace NUMINAMATH_CALUDE_quadratic_properties_l469_46940

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) :
  f a b c (-2) = -11 →
  f a b c (-1) = 9 →
  f a b c 0 = 21 →
  f a b c 3 = 9 →
  (∃ (x_max : ℝ), x_max = 0 ∧ ∀ x, f a b c x ≤ f a b c x_max) ∧
  (∃ (x_sym : ℝ), x_sym = 1 ∧ ∀ x, f a b c (x_sym - x) = f a b c (x_sym + x)) ∧
  (∃ (x : ℝ), 3 < x ∧ x < 4 ∧ f a b c x = 0) ∧
  (∀ x, f a b c x > 21 ↔ 0 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l469_46940


namespace NUMINAMATH_CALUDE_transformation_correct_l469_46935

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def mirror_scale_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, -2]
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

theorem transformation_correct :
  mirror_scale_matrix * rotation_matrix = transformation_matrix :=
by sorry

end NUMINAMATH_CALUDE_transformation_correct_l469_46935


namespace NUMINAMATH_CALUDE_oplus_three_two_l469_46900

def oplus (a b : ℕ) : ℕ := a + b + a * b - 1

theorem oplus_three_two : oplus 3 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_oplus_three_two_l469_46900


namespace NUMINAMATH_CALUDE_x_over_y_value_l469_46930

theorem x_over_y_value (x y : ℝ) 
  (h1 : 3 < (2*x - y)/(x + 2*y)) 
  (h2 : (2*x - y)/(x + 2*y) < 7) 
  (h3 : ∃ (n : ℤ), x/y = n) : 
  x/y = -4 := by sorry

end NUMINAMATH_CALUDE_x_over_y_value_l469_46930


namespace NUMINAMATH_CALUDE_soccer_team_lineups_l469_46955

theorem soccer_team_lineups :
  let total_players : ℕ := 18
  let goalie : ℕ := 1
  let defenders : ℕ := 6
  let forwards : ℕ := 4
  (total_players.choose goalie) *
  ((total_players - goalie).choose defenders) *
  ((total_players - goalie - defenders).choose forwards) = 73457760 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_lineups_l469_46955


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l469_46956

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2)) ≥ 4 / 9 ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 5 ∧ 1 / (x + 2) + 1 / (y + 2) = 4 / 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l469_46956


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l469_46965

def f (x : ℝ) := x^3 + 3*x - 1

theorem root_exists_in_interval :
  Continuous f →
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 0.5 ∧ f x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l469_46965


namespace NUMINAMATH_CALUDE_min_value_expression_l469_46953

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((x^2 + y^2) * (3 * x^2 + y^2))) / (x * y) ≥ 1 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l469_46953


namespace NUMINAMATH_CALUDE_text_pages_count_l469_46998

theorem text_pages_count (total_pages : ℕ) (image_pages : ℕ) (intro_pages : ℕ) : 
  total_pages = 98 →
  image_pages = total_pages / 2 →
  intro_pages = 11 →
  (total_pages - image_pages - intro_pages) % 2 = 0 →
  (total_pages - image_pages - intro_pages) / 2 = 19 :=
by sorry

end NUMINAMATH_CALUDE_text_pages_count_l469_46998


namespace NUMINAMATH_CALUDE_weight_of_e_l469_46962

/-- Given three weights d, e, and f, prove that e equals 82 when their average is 42,
    the average of d and e is 35, and the average of e and f is 41. -/
theorem weight_of_e (d e f : ℝ) 
  (h1 : (d + e + f) / 3 = 42)
  (h2 : (d + e) / 2 = 35)
  (h3 : (e + f) / 2 = 41) : 
  e = 82 := by
  sorry

#check weight_of_e

end NUMINAMATH_CALUDE_weight_of_e_l469_46962


namespace NUMINAMATH_CALUDE_sum_of_squares_in_sequence_l469_46990

/-- A sequence with the property that a_{2n-1} = a_{n-1}^2 + a_n^2 for all n -/
def phi_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (2*n - 1) = (a (n-1))^2 + (a n)^2

theorem sum_of_squares_in_sequence (a : ℕ → ℝ) (h : phi_sequence a) :
  ∀ n : ℕ, ∃ m : ℕ, a m = (a (n-1))^2 + (a n)^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_in_sequence_l469_46990


namespace NUMINAMATH_CALUDE_sequence_matches_l469_46974

def a (n : ℕ) : ℤ := (-1)^n * (1 - 2*n)

theorem sequence_matches : 
  (a 1 = 1) ∧ (a 2 = -3) ∧ (a 3 = 5) ∧ (a 4 = -7) ∧ (a 5 = 9) := by
  sorry

end NUMINAMATH_CALUDE_sequence_matches_l469_46974


namespace NUMINAMATH_CALUDE_price_change_after_markup_and_markdown_l469_46919

theorem price_change_after_markup_and_markdown (original_price : ℝ) (markup_percent : ℝ) (markdown_percent : ℝ)
  (h_original_positive : original_price > 0)
  (h_markup : markup_percent = 10)
  (h_markdown : markdown_percent = 10) :
  original_price * (1 + markup_percent / 100) * (1 - markdown_percent / 100) < original_price :=
by sorry

end NUMINAMATH_CALUDE_price_change_after_markup_and_markdown_l469_46919


namespace NUMINAMATH_CALUDE_sum_of_extremes_l469_46948

def is_valid_number (n : ℕ) : Prop :=
  n > 100 ∧ n < 1000 ∧ ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [1, 2, 5]

def smallest_valid_number : ℕ := sorry

def largest_valid_number : ℕ := sorry

theorem sum_of_extremes :
  smallest_valid_number + largest_valid_number = 646 ∧
  is_valid_number smallest_valid_number ∧
  is_valid_number largest_valid_number ∧
  ∀ n : ℕ, is_valid_number n →
    smallest_valid_number ≤ n ∧ n ≤ largest_valid_number :=
sorry

end NUMINAMATH_CALUDE_sum_of_extremes_l469_46948


namespace NUMINAMATH_CALUDE_total_rods_for_fence_l469_46943

/-- Represents the types of metal used in the fence. -/
inductive Metal
| A  -- Aluminum
| B  -- Bronze
| C  -- Copper

/-- Represents the components of a fence panel. -/
inductive Component
| Sheet
| Beam

/-- The number of rods needed for each type of metal and component. -/
def rods_needed (m : Metal) (c : Component) : ℕ :=
  match m, c with
  | Metal.A, Component.Sheet => 10
  | Metal.B, Component.Sheet => 8
  | Metal.C, Component.Sheet => 12
  | Metal.A, Component.Beam => 6
  | Metal.B, Component.Beam => 4
  | Metal.C, Component.Beam => 5

/-- Represents a fence pattern. -/
structure Pattern :=
  (a_sheets : ℕ)
  (b_sheets : ℕ)
  (c_sheets : ℕ)
  (a_beams : ℕ)
  (b_beams : ℕ)
  (c_beams : ℕ)

/-- The composition of Pattern X. -/
def pattern_x : Pattern :=
  { a_sheets := 2
  , b_sheets := 1
  , c_sheets := 0
  , a_beams := 0
  , b_beams := 0
  , c_beams := 2 }

/-- The composition of Pattern Y. -/
def pattern_y : Pattern :=
  { a_sheets := 0
  , b_sheets := 2
  , c_sheets := 1
  , a_beams := 3
  , b_beams := 1
  , c_beams := 0 }

/-- Calculate the total number of rods needed for a given pattern and number of panels. -/
def total_rods (p : Pattern) (panels : ℕ) : ℕ :=
  (p.a_sheets * rods_needed Metal.A Component.Sheet +
   p.b_sheets * rods_needed Metal.B Component.Sheet +
   p.c_sheets * rods_needed Metal.C Component.Sheet +
   p.a_beams * rods_needed Metal.A Component.Beam +
   p.b_beams * rods_needed Metal.B Component.Beam +
   p.c_beams * rods_needed Metal.C Component.Beam) * panels

/-- The main theorem stating that the total number of rods needed is 416. -/
theorem total_rods_for_fence : 
  total_rods pattern_x 7 + total_rods pattern_y 3 = 416 := by
  sorry


end NUMINAMATH_CALUDE_total_rods_for_fence_l469_46943


namespace NUMINAMATH_CALUDE_unique_invalid_triangle_l469_46942

/-- Represents the ratio of altitudes of a triangle -/
structure AltitudeRatio where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a triangle with given side lengths satisfies the triangle inequality -/
def satisfiesTriangleInequality (x y z : ℚ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

/-- Converts altitude ratios to side length ratios -/
def toSideLengthRatio (ar : AltitudeRatio) : (ℚ × ℚ × ℚ) :=
  (1 / ar.a, 1 / ar.b, 1 / ar.c)

/-- Theorem stating that among the given altitude ratios, only 1:2:3 violates the triangle inequality -/
theorem unique_invalid_triangle (ar : AltitudeRatio) : 
  (ar = ⟨1, 1, 2⟩ ∨ ar = ⟨1, 2, 3⟩ ∨ ar = ⟨2, 3, 4⟩ ∨ ar = ⟨3, 4, 5⟩) →
  (¬satisfiesTriangleInequality (toSideLengthRatio ar).1 (toSideLengthRatio ar).2.1 (toSideLengthRatio ar).2.2 ↔ ar = ⟨1, 2, 3⟩) :=
sorry

end NUMINAMATH_CALUDE_unique_invalid_triangle_l469_46942


namespace NUMINAMATH_CALUDE_football_kick_distance_l469_46920

theorem football_kick_distance (longest_kick : ℝ) (average_kick : ℝ) (kick1 kick2 kick3 : ℝ) :
  longest_kick = 43 →
  average_kick = 37 →
  (kick1 + kick2 + kick3) / 3 = average_kick →
  kick1 = longest_kick →
  kick2 = kick3 →
  kick2 = 34 ∧ kick3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_football_kick_distance_l469_46920


namespace NUMINAMATH_CALUDE_fraction_simplification_l469_46958

theorem fraction_simplification : 
  (1/4 - 1/5) / (1/3 - 1/4) = 3/5 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l469_46958


namespace NUMINAMATH_CALUDE_four_bottles_left_l469_46945

/-- The number of bottles left after a given number of days for a person who drinks half a bottle per day -/
def bottles_left (initial_bottles : ℕ) (days : ℕ) : ℕ :=
  initial_bottles - (days / 2)

/-- Theorem stating that 4 bottles will be left after 28 days, starting with 18 bottles -/
theorem four_bottles_left : bottles_left 18 28 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_bottles_left_l469_46945


namespace NUMINAMATH_CALUDE_vector_operations_l469_46934

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- Vector addition and scalar multiplication -/
def vec_add (u v : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => u i + v i
def scalar_mul (r : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => r * v i

/-- Parallel vectors -/
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), v = scalar_mul k u

theorem vector_operations :
  (vec_add (vec_add (scalar_mul 3 a) b) (scalar_mul (-2) c) = ![0, 6]) ∧
  (∃! (m n : ℝ), a = vec_add (scalar_mul m b) (scalar_mul n c) ∧ m = 5/9 ∧ n = 8/9) ∧
  (∃! (k : ℝ), parallel (vec_add a (scalar_mul k c)) (vec_add (scalar_mul 2 b) (scalar_mul (-1) a)) ∧ k = -16/13) :=
by sorry

end NUMINAMATH_CALUDE_vector_operations_l469_46934


namespace NUMINAMATH_CALUDE_total_soak_time_l469_46964

def grass_soak_time : ℕ := 3
def marinara_soak_time : ℕ := 7
def ink_soak_time : ℕ := 5
def coffee_soak_time : ℕ := 10

def num_grass_stains : ℕ := 3
def num_marinara_stains : ℕ := 1
def num_ink_stains : ℕ := 2
def num_coffee_stains : ℕ := 1

theorem total_soak_time :
  grass_soak_time * num_grass_stains +
  marinara_soak_time * num_marinara_stains +
  ink_soak_time * num_ink_stains +
  coffee_soak_time * num_coffee_stains = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_soak_time_l469_46964


namespace NUMINAMATH_CALUDE_correct_savings_amount_l469_46912

/-- Represents a bank with its interest calculation method -/
structure Bank where
  name : String
  calculateInterest : (principal : ℝ) → ℝ

/-- Calculates the amount needed to save given a bank's interest calculation -/
def amountToSave (initialFunds : ℝ) (totalExpenses : ℝ) (bank : Bank) : ℝ :=
  totalExpenses - initialFunds - bank.calculateInterest initialFunds

/-- Theorem stating the correct amount to save for each bank -/
theorem correct_savings_amount 
  (initialFunds : ℝ) 
  (totalExpenses : ℝ) 
  (bettaBank gammaBank omegaBank epsilonBank : Bank) 
  (h1 : initialFunds = 150000)
  (h2 : totalExpenses = 182200)
  (h3 : bettaBank.calculateInterest initialFunds = 2720.33)
  (h4 : gammaBank.calculateInterest initialFunds = 3375)
  (h5 : omegaBank.calculateInterest initialFunds = 2349.13)
  (h6 : epsilonBank.calculateInterest initialFunds = 2264.11) :
  (amountToSave initialFunds totalExpenses bettaBank = 29479.67) ∧
  (amountToSave initialFunds totalExpenses gammaBank = 28825) ∧
  (amountToSave initialFunds totalExpenses omegaBank = 29850.87) ∧
  (amountToSave initialFunds totalExpenses epsilonBank = 29935.89) :=
by sorry


end NUMINAMATH_CALUDE_correct_savings_amount_l469_46912


namespace NUMINAMATH_CALUDE_purple_socks_probability_l469_46981

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer where
  green : ℕ
  purple : ℕ
  orange : ℕ

/-- Calculates the total number of socks in the drawer -/
def SockDrawer.total (d : SockDrawer) : ℕ :=
  d.green + d.purple + d.orange

/-- Calculates the probability of selecting a purple sock -/
def purpleProbability (d : SockDrawer) : ℚ :=
  d.purple / d.total

/-- The initial state of the sock drawer -/
def initialDrawer : SockDrawer :=
  { green := 6, purple := 18, orange := 12 }

/-- The number of purple socks added -/
def addedPurpleSocks : ℕ := 9

/-- The final state of the sock drawer after adding purple socks -/
def finalDrawer : SockDrawer :=
  { green := initialDrawer.green,
    purple := initialDrawer.purple + addedPurpleSocks,
    orange := initialDrawer.orange }

theorem purple_socks_probability :
  purpleProbability finalDrawer = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_purple_socks_probability_l469_46981


namespace NUMINAMATH_CALUDE_restaurant_pies_theorem_l469_46975

/-- The number of pies sold in a week by a restaurant that sells 8 pies per day -/
def pies_sold_in_week (pies_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  pies_per_day * days_in_week

/-- Proof that a restaurant selling 8 pies per day for a week sells 56 pies in total -/
theorem restaurant_pies_theorem :
  pies_sold_in_week 8 7 = 56 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_pies_theorem_l469_46975


namespace NUMINAMATH_CALUDE_kyle_fish_count_l469_46993

/-- Given that Carla, Kyle, and Tasha caught a total of 36 fish, 
    Carla caught 8 fish, and Kyle and Tasha caught the same number of fish,
    prove that Kyle caught 14 fish. -/
theorem kyle_fish_count (total : ℕ) (carla : ℕ) (kyle : ℕ) (tasha : ℕ)
  (h1 : total = 36)
  (h2 : carla = 8)
  (h3 : kyle = tasha)
  (h4 : total = carla + kyle + tasha) :
  kyle = 14 := by
  sorry

end NUMINAMATH_CALUDE_kyle_fish_count_l469_46993


namespace NUMINAMATH_CALUDE_symmetric_sine_graph_l469_46988

theorem symmetric_sine_graph (φ : Real) : 
  (-Real.pi / 2 < φ ∧ φ < Real.pi / 2) →
  (∀ x, Real.sin (2 * x + φ) = Real.sin (2 * (2 * Real.pi / 3 - x) + φ)) →
  φ = -Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_symmetric_sine_graph_l469_46988


namespace NUMINAMATH_CALUDE_solve_equation_l469_46961

/-- Given an equation 19(x + y) + 17 = 19(-x + y) - n where x = 1, prove that n = -55 -/
theorem solve_equation (y : ℝ) : 
  (∃ (n : ℝ), 19 * (1 + y) + 17 = 19 * (-1 + y) - n) → 
  (∃ (n : ℝ), 19 * (1 + y) + 17 = 19 * (-1 + y) - n ∧ n = -55) :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l469_46961


namespace NUMINAMATH_CALUDE_gcd_47_power_plus_one_l469_46908

theorem gcd_47_power_plus_one : Nat.gcd (47^11 + 1) (47^11 + 47^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_47_power_plus_one_l469_46908


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l469_46923

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 30
  let k : ℕ := 3
  let a : ℕ := 2
  (Nat.choose n k) * a^(n - k) = 4060 * 2^27 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l469_46923


namespace NUMINAMATH_CALUDE_lucy_money_theorem_l469_46907

def lucy_money_problem (initial_amount : ℚ) : ℚ :=
  let remaining_after_loss := initial_amount * (1 - 1/3)
  let spent := remaining_after_loss * (1/4)
  remaining_after_loss - spent

theorem lucy_money_theorem :
  lucy_money_problem 30 = 15 := by sorry

end NUMINAMATH_CALUDE_lucy_money_theorem_l469_46907


namespace NUMINAMATH_CALUDE_basswood_figurines_count_l469_46914

/-- The number of figurines that can be created from a block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be created from a block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be created from a block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 245

/-- The number of basswood blocks Adam has -/
def basswood_blocks : ℕ := 15

/-- The number of butternut wood blocks Adam has -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam has -/
def aspen_blocks : ℕ := 20

theorem basswood_figurines_count : 
  basswood_blocks * basswood_figurines + 
  butternut_blocks * butternut_figurines + 
  aspen_blocks * aspen_figurines = total_figurines :=
by sorry

end NUMINAMATH_CALUDE_basswood_figurines_count_l469_46914


namespace NUMINAMATH_CALUDE_functional_equation_zero_function_l469_46992

theorem functional_equation_zero_function 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = x * f x + y * f y) : 
  ∀ x : ℝ, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_zero_function_l469_46992


namespace NUMINAMATH_CALUDE_negative_two_inequality_l469_46903

theorem negative_two_inequality (a b : ℝ) (h : a < b) : -2 * a > -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_inequality_l469_46903


namespace NUMINAMATH_CALUDE_car_highway_efficiency_l469_46938

/-- The number of miles the car can travel on the highway with one gallon of gasoline. -/
def highway_miles_per_gallon : ℝ := 38

/-- The number of miles the car can travel in the city with one gallon of gasoline. -/
def city_miles_per_gallon : ℝ := 20

/-- Proves that the car can travel 38 miles on the highway with one gallon of gasoline,
    given the conditions stated in the problem. -/
theorem car_highway_efficiency :
  highway_miles_per_gallon = 38 ∧
  (4 / highway_miles_per_gallon + 4 / city_miles_per_gallon =
   8 / highway_miles_per_gallon * (1 + 0.45000000000000014)) :=
by sorry

end NUMINAMATH_CALUDE_car_highway_efficiency_l469_46938


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l469_46928

/-- Systematic sampling function that returns the next sample number -/
def nextSample (total : ℕ) (sampleSize : ℕ) (current : ℕ) : ℕ :=
  (current + total / sampleSize) % total

/-- Proposition: In a systematic sampling of 4 items from 56 items, 
    if items 7 and 35 are selected, then the other two selected items 
    are numbered 21 and 49 -/
theorem systematic_sampling_proof :
  let total := 56
  let sampleSize := 4
  let first := 7
  let second := 35
  nextSample total sampleSize first = 21 ∧
  nextSample total sampleSize second = 49 := by
  sorry

#eval nextSample 56 4 7  -- Should output 21
#eval nextSample 56 4 35 -- Should output 49

end NUMINAMATH_CALUDE_systematic_sampling_proof_l469_46928


namespace NUMINAMATH_CALUDE_parallelogram_contains_two_points_l469_46983

/-- The set L of points in the Cartesian plane -/
def L : Set (ℤ × ℤ) := {p | ∃ x y : ℤ, p = (41*x + 2*y, 59*x + 15*y)}

/-- Definition of a parallelogram centered at the origin -/
structure CenteredParallelogram where
  vertices : Fin 4 → ℝ × ℝ
  center_at_origin : (vertices 0) + (vertices 2) = (0, 0)
  parallel_sides : (vertices 1) - (vertices 0) = (vertices 3) - (vertices 2)

/-- The area of a parallelogram -/
def area (p : CenteredParallelogram) : ℝ :=
  let v1 := p.vertices 1 - p.vertices 0
  let v2 := p.vertices 3 - p.vertices 0
  |v1.1 * v2.2 - v1.2 * v2.1|

/-- Main theorem -/
theorem parallelogram_contains_two_points (p : CenteredParallelogram) (h : area p = 1990) :
  ∃ p1 p2 : ℤ × ℤ, p1 ∈ L ∧ p2 ∈ L ∧ p1 ≠ p2 ∧ 
  (∃ (i : Fin 4), ∃ (t : ℝ) (s : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧
    (↑p1.1, ↑p1.2) = t • p.vertices i + s • p.vertices ((i + 1) % 4)) ∧
  (∃ (j : Fin 4), ∃ (u : ℝ) (v : ℝ), 0 ≤ u ∧ u ≤ 1 ∧ 0 ≤ v ∧ v ≤ 1 ∧
    (↑p2.1, ↑p2.2) = u • p.vertices j + v • p.vertices ((j + 1) % 4)) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_contains_two_points_l469_46983


namespace NUMINAMATH_CALUDE_parabola_and_point_theorem_l469_46922

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

def on_parabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

def perpendicular (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

theorem parabola_and_point_theorem (C : Parabola) (A B O : Point) :
  on_parabola A C →
  on_parabola B C →
  A.x = 1 →
  A.y = 2 →
  O.x = 0 →
  O.y = 0 →
  B.x ≠ 0 →
  perpendicular A O B →
  (C.p = 2 ∧ B.x = 16 ∧ B.y = -8) := by sorry

end NUMINAMATH_CALUDE_parabola_and_point_theorem_l469_46922


namespace NUMINAMATH_CALUDE_leaves_collected_first_day_l469_46952

/-- Represents the number of leaves collected by Bronson -/
def total_leaves : ℕ := 25

/-- Represents the number of leaves collected on the second day -/
def second_day_leaves : ℕ := 13

/-- Represents the percentage of brown leaves -/
def brown_percent : ℚ := 1/5

/-- Represents the percentage of green leaves -/
def green_percent : ℚ := 1/5

/-- Represents the number of yellow leaves -/
def yellow_leaves : ℕ := 15

/-- Theorem stating the number of leaves collected on the first day -/
theorem leaves_collected_first_day : 
  total_leaves - second_day_leaves = 12 :=
sorry

end NUMINAMATH_CALUDE_leaves_collected_first_day_l469_46952


namespace NUMINAMATH_CALUDE_max_value_theorem_l469_46972

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 12) : 
  ∃ (z : ℝ), z = x^2 + 2*x*y + 3*y^2 ∧ z ≤ 132 + 48 * Real.sqrt 3 ∧
  ∃ (a b : ℝ), a^2 - 2*a*b + 3*b^2 = 12 ∧ a > 0 ∧ b > 0 ∧
  a^2 + 2*a*b + 3*b^2 = 132 + 48 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l469_46972


namespace NUMINAMATH_CALUDE_complex_conjugate_roots_imply_zero_coefficients_l469_46968

theorem complex_conjugate_roots_imply_zero_coefficients 
  (c d : ℝ) 
  (h : ∃ (u v : ℝ), (Complex.I * v + u)^2 + (15 + Complex.I * c) * (Complex.I * v + u) + (35 + Complex.I * d) = 0 ∧ 
                     (Complex.I * -v + u)^2 + (15 + Complex.I * c) * (Complex.I * -v + u) + (35 + Complex.I * d) = 0) : 
  c = 0 ∧ d = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_conjugate_roots_imply_zero_coefficients_l469_46968


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l469_46978

theorem cube_root_equation_solution (x : ℝ) : 
  (15 * x + (15 * x + 8) ^ (1/3)) ^ (1/3) = 8 → x = 168/5 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l469_46978


namespace NUMINAMATH_CALUDE_rent_increase_effect_rent_problem_l469_46941

theorem rent_increase_effect (num_friends : ℕ) (initial_avg_rent : ℚ) 
  (increased_rent : ℚ) (increase_percentage : ℚ) : ℚ :=
  let total_initial_rent := num_friends * initial_avg_rent
  let rent_increase := increased_rent * increase_percentage
  let new_total_rent := total_initial_rent + rent_increase
  let new_avg_rent := new_total_rent / num_friends
  new_avg_rent

theorem rent_problem :
  rent_increase_effect 4 800 1600 (1/5) = 880 := by sorry

end NUMINAMATH_CALUDE_rent_increase_effect_rent_problem_l469_46941


namespace NUMINAMATH_CALUDE_distance_between_points_l469_46915

theorem distance_between_points : ∃ d : ℝ, 
  let A : ℝ × ℝ := (13, 5)
  let B : ℝ × ℝ := (5, -10)
  d = ((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt ∧ d = 17 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l469_46915


namespace NUMINAMATH_CALUDE_line_parallel_perp_implies_planes_perp_l469_46960

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Parallel relation between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two planes -/
def planesPerpendicular (p1 : Plane3D) (p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is parallel to one plane and perpendicular to another,
    then the two planes are perpendicular -/
theorem line_parallel_perp_implies_planes_perp
  (c : Line3D) (α β : Plane3D)
  (h1 : parallel c α)
  (h2 : perpendicular c β) :
  planesPerpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perp_implies_planes_perp_l469_46960


namespace NUMINAMATH_CALUDE_triangle_formation_l469_46909

theorem triangle_formation (a : ℝ) : 
  (0 < a ∧ a + 3 > 5 ∧ a + 5 > 3 ∧ 3 + 5 > a) ↔ a = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l469_46909


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_find_S_5_l469_46969

/-- Given an arithmetic sequence {aₙ}, Sₙ represents the sum of its first n terms -/
def S (n : ℕ) : ℝ := sorry

/-- aₙ represents the nth term of the arithmetic sequence -/
def a (n : ℕ) : ℝ := sorry

/-- d represents the common difference of the arithmetic sequence -/
def d : ℝ := sorry

theorem arithmetic_sequence_sum (n : ℕ) :
  S n = n * a 1 + (n * (n - 1) / 2) * d := sorry

axiom sum_condition : S 3 + S 6 = 18

theorem find_S_5 : S 5 = 10 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_find_S_5_l469_46969


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l469_46921

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 5 -/
def binary_five : List Bool := [true, false, true]

/-- Theorem stating that the binary representation [1,0,1] is equal to 5 in decimal -/
theorem binary_101_equals_5 : binary_to_decimal binary_five = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l469_46921


namespace NUMINAMATH_CALUDE_range_of_a_l469_46910

/-- The range of real number a satisfying the given inequality -/
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (a * x + 1) * (Real.exp x - a * Real.exp 1 * x) ≥ 0) ↔ 
  (0 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l469_46910


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_400_l469_46929

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.length ≥ 2) ∧
  (∀ p ∈ powers, is_power_of_two p) ∧
  (powers.sum = n) ∧
  (powers.toFinset.card = powers.length)

def exponent_sum (powers : List ℕ) : ℕ :=
  (powers.map (λ p => (Nat.log p 2))).sum

theorem least_exponent_sum_for_400 :
  ∀ powers : List ℕ,
    sum_of_distinct_powers_of_two 400 powers →
    exponent_sum powers ≥ 19 :=
sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_400_l469_46929


namespace NUMINAMATH_CALUDE_bus_miss_time_l469_46994

theorem bus_miss_time (usual_time : ℝ) (h : usual_time = 12) :
  let slower_time := (5 / 4) * usual_time
  slower_time - usual_time = 3 :=
by sorry

end NUMINAMATH_CALUDE_bus_miss_time_l469_46994


namespace NUMINAMATH_CALUDE_function_value_l469_46932

/-- Given a function f where f(2x + 3) is defined and f(29) = 170,
    prove that f(2x + 3) = 170 for all x -/
theorem function_value (f : ℝ → ℝ) (h : f 29 = 170) : ∀ x, f (2 * x + 3) = 170 := by
  sorry

end NUMINAMATH_CALUDE_function_value_l469_46932


namespace NUMINAMATH_CALUDE_max_badminton_rackets_l469_46951

theorem max_badminton_rackets 
  (table_tennis_price badminton_price : ℕ)
  (total_rackets : ℕ)
  (max_expenditure : ℕ)
  (h1 : 2 * table_tennis_price + badminton_price = 220)
  (h2 : 3 * table_tennis_price + 2 * badminton_price = 380)
  (h3 : total_rackets = 30)
  (h4 : ∀ m : ℕ, m ≤ total_rackets → 
        (total_rackets - m) * table_tennis_price + m * badminton_price ≤ max_expenditure) :
  ∃ max_badminton : ℕ, 
    max_badminton ≤ total_rackets ∧
    (total_rackets - max_badminton) * table_tennis_price + max_badminton * badminton_price ≤ max_expenditure ∧
    ∀ n : ℕ, n > max_badminton → 
      (total_rackets - n) * table_tennis_price + n * badminton_price > max_expenditure :=
by
  sorry

end NUMINAMATH_CALUDE_max_badminton_rackets_l469_46951


namespace NUMINAMATH_CALUDE_constant_sum_l469_46949

theorem constant_sum (x y : ℝ) (h : x + y = 4) : 5 * x + 5 * y = 20 := by
  sorry

end NUMINAMATH_CALUDE_constant_sum_l469_46949


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l469_46987

theorem inequality_system_solution_range (m : ℝ) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ 
    (2 * ↑x₁ - 1 ≤ 5 ∧ ↑x₁ - 1 ≥ m) ∧ 
    (2 * ↑x₂ - 1 ≤ 5 ∧ ↑x₂ - 1 ≥ m) ∧
    (∀ x : ℤ, (2 * ↑x - 1 ≤ 5 ∧ ↑x - 1 ≥ m) → (x = x₁ ∨ x = x₂))) ↔
  (-1 < m ∧ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l469_46987


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l469_46986

theorem trigonometric_expression_equals_one :
  (Real.tan (45 * π / 180))^2 - (Real.sin (45 * π / 180))^2 = 
  (Real.tan (45 * π / 180))^2 * (Real.sin (45 * π / 180))^2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l469_46986


namespace NUMINAMATH_CALUDE_music_exam_songs_l469_46970

/-- Represents a girl participating in the music exam -/
inductive Girl
| Anna
| Bea
| Cili
| Dora

/-- The number of times each girl sang -/
def timesSang (g : Girl) : ℕ :=
  match g with
  | Girl.Anna => 8
  | Girl.Bea => 7  -- We assume 7 as it satisfies the conditions
  | Girl.Cili => 7 -- We assume 7 as it satisfies the conditions
  | Girl.Dora => 5

/-- The total number of individual singing assignments -/
def totalSingingAssignments : ℕ := 
  (timesSang Girl.Anna) + (timesSang Girl.Bea) + (timesSang Girl.Cili) + (timesSang Girl.Dora)

theorem music_exam_songs :
  (∀ g : Girl, timesSang g ≤ timesSang Girl.Anna) ∧ 
  (∀ g : Girl, g ≠ Girl.Anna → timesSang g < timesSang Girl.Anna) ∧
  (∀ g : Girl, g ≠ Girl.Dora → timesSang Girl.Dora < timesSang g) ∧
  (totalSingingAssignments % 3 = 0) →
  totalSingingAssignments / 3 = 9 := by
  sorry

#eval totalSingingAssignments / 3

end NUMINAMATH_CALUDE_music_exam_songs_l469_46970
