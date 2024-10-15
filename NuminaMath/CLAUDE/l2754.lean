import Mathlib

namespace NUMINAMATH_CALUDE_restaurant_budget_theorem_l2754_275474

theorem restaurant_budget_theorem (budget : ℝ) (budget_positive : budget > 0) :
  let rent := (1 / 4) * budget
  let remaining := budget - rent
  let food_and_beverages := (1 / 4) * remaining
  (food_and_beverages / budget) * 100 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_budget_theorem_l2754_275474


namespace NUMINAMATH_CALUDE_power_division_equality_l2754_275432

theorem power_division_equality : (3^3)^2 / 3^2 = 81 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l2754_275432


namespace NUMINAMATH_CALUDE_quadratic_point_m_value_l2754_275480

theorem quadratic_point_m_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 → 
  3 = -a * m^2 + 2 * a * m + 3 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_point_m_value_l2754_275480


namespace NUMINAMATH_CALUDE_tangent_circle_intersection_distance_l2754_275497

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the intersection of two circles
variable (intersect : Circle → Circle → Point)

-- Define the tangent line at a point on a circle
variable (tangent_at : Point → Circle → Point → Prop)

-- Define a circle passing through three points
variable (circle_through : Point → Point → Point → Circle)

-- Define the distance between two points
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem tangent_circle_intersection_distance
  (C₁ C₂ C₃ : Circle) (S A B P Q : Point) :
  intersect C₁ C₂ = S →
  tangent_at S C₁ A →
  tangent_at S C₂ B →
  C₃ = circle_through A B S →
  tangent_at S C₃ P →
  tangent_at S C₃ Q →
  A ≠ S →
  B ≠ S →
  P ≠ S →
  Q ≠ S →
  distance P S = distance Q S :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_intersection_distance_l2754_275497


namespace NUMINAMATH_CALUDE_max_carlson_jars_l2754_275491

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamJars :=
  (carlson_weight : ℕ)  -- Total weight of Carlson's jars
  (baby_weight : ℕ)     -- Total weight of Baby's jars
  (lightest_jar : ℕ)    -- Weight of Carlson's lightest jar

/-- The conditions of the problem -/
def valid_jam_state (j : JamJars) : Prop :=
  j.carlson_weight = 13 * j.baby_weight ∧
  j.carlson_weight - j.lightest_jar = 8 * (j.baby_weight + j.lightest_jar)

/-- The maximum number of jars Carlson could have initially -/
def max_jars (j : JamJars) : ℕ := j.carlson_weight / j.lightest_jar

/-- The theorem to prove -/
theorem max_carlson_jars :
  ∀ j : JamJars, valid_jam_state j → max_jars j ≤ 23 :=
by sorry

end NUMINAMATH_CALUDE_max_carlson_jars_l2754_275491


namespace NUMINAMATH_CALUDE_inverse_f_zero_l2754_275430

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 1 / (2 * a * x + 3 * b)

theorem inverse_f_zero (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ x, f a b x = 1 / (3 * b) ∧ (∀ y, f a b y = x → y = 0) :=
by sorry

end NUMINAMATH_CALUDE_inverse_f_zero_l2754_275430


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l2754_275483

/-- The perimeter of a hexagon with side length 4 inches is 24 inches. -/
theorem hexagon_perimeter (side_length : ℝ) : side_length = 4 → 6 * side_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l2754_275483


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2754_275469

noncomputable def f (x : ℝ) : ℝ := x^3 + 3*x - 1

theorem root_sum_reciprocal (a b c : ℝ) (m n : ℕ) :
  f a = 0 → f b = 0 → f c = 0 →
  (1 / (a^3 + b^3) + 1 / (b^3 + c^3) + 1 / (c^3 + a^3) : ℝ) = m / n →
  m > 0 → n > 0 →
  Nat.gcd m n = 1 →
  100 * m + n = 3989 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2754_275469


namespace NUMINAMATH_CALUDE_least_perimeter_l2754_275414

/-- Triangle DEF with given cosine values -/
structure TriangleDEF where
  d : ℕ
  e : ℕ
  f : ℕ
  cos_d : Real
  cos_e : Real
  cos_f : Real
  h_cos_d : cos_d = 8 / 17
  h_cos_e : cos_e = 15 / 17
  h_cos_f : cos_f = -5 / 13

/-- The perimeter of triangle DEF -/
def perimeter (t : TriangleDEF) : ℕ := t.d + t.e + t.f

/-- The least possible perimeter of triangle DEF is 503 -/
theorem least_perimeter (t : TriangleDEF) : 
  (∀ t' : TriangleDEF, perimeter t ≤ perimeter t') → perimeter t = 503 := by
  sorry

end NUMINAMATH_CALUDE_least_perimeter_l2754_275414


namespace NUMINAMATH_CALUDE_max_different_ages_l2754_275454

theorem max_different_ages 
  (average_age : ℝ) 
  (std_dev : ℝ) 
  (average_age_eq : average_age = 31) 
  (std_dev_eq : std_dev = 8) : 
  ∃ (max_ages : ℕ), 
    max_ages = 17 ∧ 
    ∀ (age : ℕ), 
      (↑age ≥ average_age - std_dev ∧ ↑age ≤ average_age + std_dev) ↔ 
      (age ≥ 23 ∧ age ≤ 39) :=
by sorry

end NUMINAMATH_CALUDE_max_different_ages_l2754_275454


namespace NUMINAMATH_CALUDE_circle_properties_l2754_275423

/-- Circle C in the Cartesian coordinate system -/
def circle_C (x y b : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + b = 0

/-- Point A -/
def point_A : ℝ × ℝ := (0, 3)

/-- Radius of circle C -/
def radius : ℝ := 1

theorem circle_properties :
  ∃ (b : ℝ), 
    (∀ x y, circle_C x y b → (x - 3)^2 + (y - 2)^2 = 1) ∧ 
    (b < 13) ∧
    (∃ (k : ℝ), k = -3/4 ∧ ∀ x y, 3*x + 4*y - 12 = 0 → circle_C x y b) ∧
    (∀ y, circle_C 0 3 b → y = 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2754_275423


namespace NUMINAMATH_CALUDE_smallest_value_x_squared_plus_8x_l2754_275452

theorem smallest_value_x_squared_plus_8x :
  (∀ x : ℝ, x^2 + 8*x ≥ -16) ∧ (∃ x : ℝ, x^2 + 8*x = -16) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_x_squared_plus_8x_l2754_275452


namespace NUMINAMATH_CALUDE_room_length_proof_l2754_275433

theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 →
  total_cost = 29925 →
  paving_rate = 900 →
  (total_cost / paving_rate) / width = 7 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l2754_275433


namespace NUMINAMATH_CALUDE_percentage_of_A_students_l2754_275426

theorem percentage_of_A_students (total_students : ℕ) (failed_students : ℕ) 
  (h1 : total_students = 32)
  (h2 : failed_students = 18)
  (h3 : ∃ (A : ℕ) (B_C : ℕ), 
    A + B_C + failed_students = total_students ∧ 
    B_C = (total_students - failed_students - A) / 4) :
  (((total_students - failed_students) : ℚ) / total_students) * 100 = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_A_students_l2754_275426


namespace NUMINAMATH_CALUDE_intersection_point_d_l2754_275498

/-- A function g(x) = 2x + c with c being an integer -/
def g (c : ℤ) : ℝ → ℝ := λ x ↦ 2 * x + c

/-- The inverse function of g -/
noncomputable def g_inv (c : ℤ) : ℝ → ℝ := λ x ↦ (x - c) / 2

theorem intersection_point_d (c : ℤ) (d : ℤ) :
  g c (-4) = d ∧ g_inv c (-4) = d → d = -4 := by sorry

end NUMINAMATH_CALUDE_intersection_point_d_l2754_275498


namespace NUMINAMATH_CALUDE_methane_moles_needed_l2754_275479

/-- Represents the chemical reaction C6H6 + CH4 → C7H8 + H2 -/
structure ChemicalReaction where
  benzene : ℝ
  methane : ℝ
  toluene : ℝ
  hydrogen : ℝ

/-- The molar mass of Benzene in g/mol -/
def benzene_molar_mass : ℝ := 78

/-- The total amount of Benzene required in grams -/
def total_benzene : ℝ := 156

/-- The number of moles of Toluene produced -/
def toluene_moles : ℝ := 2

/-- The number of moles of Hydrogen produced -/
def hydrogen_moles : ℝ := 2

theorem methane_moles_needed (reaction : ChemicalReaction) :
  reaction.benzene = total_benzene / benzene_molar_mass ∧
  reaction.toluene = toluene_moles ∧
  reaction.hydrogen = hydrogen_moles ∧
  reaction.benzene = reaction.methane →
  reaction.methane = 2 := by
  sorry

end NUMINAMATH_CALUDE_methane_moles_needed_l2754_275479


namespace NUMINAMATH_CALUDE_dance_studio_dancers_l2754_275481

/-- The number of performances -/
def num_performances : ℕ := 40

/-- The number of dancers in each performance -/
def dancers_per_performance : ℕ := 10

/-- The maximum number of times any pair of dancers can perform together -/
def max_pair_performances : ℕ := 1

/-- The minimum number of dancers required -/
def min_dancers : ℕ := 60

theorem dance_studio_dancers :
  ∀ (n : ℕ), n ≥ min_dancers →
  (n.choose 2) ≥ num_performances * (dancers_per_performance.choose 2) :=
by sorry

end NUMINAMATH_CALUDE_dance_studio_dancers_l2754_275481


namespace NUMINAMATH_CALUDE_max_value_of_function_l2754_275473

theorem max_value_of_function (x : ℝ) (h : x < 1/3) :
  3 * x + 1 / (3 * x - 1) ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2754_275473


namespace NUMINAMATH_CALUDE_solve_complex_equation_l2754_275435

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation (1+i)Z = 2
def equation (Z : ℂ) : Prop := (1 + i) * Z = 2

-- Theorem statement
theorem solve_complex_equation :
  ∀ Z : ℂ, equation Z → Z = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l2754_275435


namespace NUMINAMATH_CALUDE_exists_player_in_win_range_l2754_275453

/-- Represents a chess tournament with 2n+1 players -/
structure ChessTournament (n : ℕ) where
  /-- The number of games won by lower-rated players -/
  k : ℕ
  /-- Player ratings, assumed to be unique -/
  ratings : Fin (2*n+1) → ℕ
  ratings_unique : ∀ i j, i ≠ j → ratings i ≠ ratings j

/-- The number of wins for each player -/
def wins (t : ChessTournament n) : Fin (2*n+1) → ℕ :=
  sorry

theorem exists_player_in_win_range (n : ℕ) (t : ChessTournament n) :
  ∃ p : Fin (2*n+1), 
    (n : ℝ) - Real.sqrt (2 * t.k) ≤ wins t p ∧ 
    wins t p ≤ (n : ℝ) + Real.sqrt (2 * t.k) :=
  sorry

end NUMINAMATH_CALUDE_exists_player_in_win_range_l2754_275453


namespace NUMINAMATH_CALUDE_least_common_period_l2754_275405

-- Define the property for function f
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

-- Define the concept of a period for a function
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_common_period :
  ∃ p : ℝ, p > 0 ∧
    (∀ f : ℝ → ℝ, satisfies_condition f → is_period f p) ∧
    (∀ q : ℝ, q > 0 → (∀ f : ℝ → ℝ, satisfies_condition f → is_period f q) → p ≤ q) ∧
    p = 30 :=
  sorry

end NUMINAMATH_CALUDE_least_common_period_l2754_275405


namespace NUMINAMATH_CALUDE_student_distribution_proof_l2754_275445

def distribute_students (n : ℕ) (k : ℕ) : ℕ := sorry

theorem student_distribution_proof : 
  distribute_students 24 3 = 475 := by sorry

end NUMINAMATH_CALUDE_student_distribution_proof_l2754_275445


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l2754_275413

/-- A parallelogram in a 2D plane -/
structure Parallelogram where
  area : ℝ

/-- A quadrilateral formed by joining the midpoints of a parallelogram's sides -/
def midpoint_quadrilateral (p : Parallelogram) : Parallelogram :=
  { area := sorry }

/-- The area of the midpoint quadrilateral is 1/4 of the original parallelogram's area -/
theorem midpoint_quadrilateral_area (p : Parallelogram) :
  (midpoint_quadrilateral p).area = p.area / 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l2754_275413


namespace NUMINAMATH_CALUDE_tina_customers_l2754_275455

/-- Calculates the number of customers Tina sold books to -/
def number_of_customers (selling_price cost_price total_profit books_per_customer : ℚ) : ℚ :=
  (total_profit / (selling_price - cost_price)) / books_per_customer

/-- Theorem: Given the conditions, Tina sold books to 4 customers -/
theorem tina_customers :
  let selling_price : ℚ := 20
  let cost_price : ℚ := 5
  let total_profit : ℚ := 120
  let books_per_customer : ℚ := 2
  number_of_customers selling_price cost_price total_profit books_per_customer = 4 := by
  sorry

#eval number_of_customers 20 5 120 2

end NUMINAMATH_CALUDE_tina_customers_l2754_275455


namespace NUMINAMATH_CALUDE_hair_growth_calculation_l2754_275460

theorem hair_growth_calculation (initial_length : ℝ) (growth : ℝ) (final_length : ℝ) : 
  initial_length = 24 →
  final_length = 14 →
  final_length = initial_length / 2 + growth - 2 →
  growth = 4 := by
sorry

end NUMINAMATH_CALUDE_hair_growth_calculation_l2754_275460


namespace NUMINAMATH_CALUDE_fescue_percentage_in_y_l2754_275406

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The combined mixture of X and Y -/
def combinedMixture (x y : SeedMixture) (xProportion : ℝ) : SeedMixture :=
  { ryegrass := x.ryegrass * xProportion + y.ryegrass * (1 - xProportion)
  , bluegrass := x.bluegrass * xProportion + y.bluegrass * (1 - xProportion)
  , fescue := x.fescue * xProportion + y.fescue * (1 - xProportion) }

/-- The theorem stating the percentage of fescue in mixture Y -/
theorem fescue_percentage_in_y
  (x : SeedMixture)
  (y : SeedMixture)
  (h1 : x.ryegrass = 0.4)
  (h2 : x.bluegrass = 0.6)
  (h3 : x.fescue = 0)
  (h4 : y.ryegrass = 0.25)
  (h5 : x.ryegrass + x.bluegrass + x.fescue = 1)
  (h6 : y.ryegrass + y.bluegrass + y.fescue = 1)
  (h7 : (combinedMixture x y 0.4667).ryegrass = 0.32) :
  y.fescue = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_fescue_percentage_in_y_l2754_275406


namespace NUMINAMATH_CALUDE_sum_ab_over_2b_plus_1_geq_1_l2754_275443

variables (a b c : ℝ)

theorem sum_ab_over_2b_plus_1_geq_1
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 3) :
  (a * b) / (2 * b + 1) + (b * c) / (2 * c + 1) + (c * a) / (2 * a + 1) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_sum_ab_over_2b_plus_1_geq_1_l2754_275443


namespace NUMINAMATH_CALUDE_company_contracts_probability_l2754_275463

theorem company_contracts_probability
  (p_hardware : ℝ)
  (p_not_software : ℝ)
  (p_network : ℝ)
  (p_maintenance : ℝ)
  (p_at_least_one : ℝ)
  (h_hardware : p_hardware = 3/4)
  (h_not_software : p_not_software = 3/5)
  (h_network : p_network = 2/3)
  (h_maintenance : p_maintenance = 1/2)
  (h_at_least_one : p_at_least_one = 7/8) :
  p_hardware * (1 - p_not_software) * p_network * p_maintenance = 1/10 :=
sorry

end NUMINAMATH_CALUDE_company_contracts_probability_l2754_275463


namespace NUMINAMATH_CALUDE_decimal_places_theorem_l2754_275412

def first_1000_decimal_places (x : ℝ) : List ℕ :=
  sorry

theorem decimal_places_theorem :
  (∀ d ∈ first_1000_decimal_places ((6 + Real.sqrt 35) ^ 1999), d = 9) ∧
  (∀ d ∈ first_1000_decimal_places ((6 + Real.sqrt 37) ^ 1999), d = 0) ∧
  (∀ d ∈ first_1000_decimal_places ((6 + Real.sqrt 37) ^ 2000), d = 9) :=
by sorry

end NUMINAMATH_CALUDE_decimal_places_theorem_l2754_275412


namespace NUMINAMATH_CALUDE_inequalities_theorem_l2754_275446

theorem inequalities_theorem (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  (abs a < abs b) ∧ (a > b) ∧ (a + b > a * b) ∧ (a^3 > b^3) := by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l2754_275446


namespace NUMINAMATH_CALUDE_costco_mayo_price_l2754_275475

/-- The cost of a gallon of mayo at Costco -/
def costco_gallon_cost : ℚ := 8

/-- The volume of a gallon in ounces -/
def gallon_ounces : ℕ := 128

/-- The volume of a standard bottle in ounces -/
def bottle_ounces : ℕ := 16

/-- The cost of a standard bottle at a normal store -/
def normal_store_bottle_cost : ℚ := 3

/-- The savings when buying at Costco -/
def costco_savings : ℚ := 16

theorem costco_mayo_price :
  costco_gallon_cost = 
    (gallon_ounces / bottle_ounces : ℚ) * normal_store_bottle_cost - costco_savings :=
by sorry

end NUMINAMATH_CALUDE_costco_mayo_price_l2754_275475


namespace NUMINAMATH_CALUDE_like_terms_imply_m_minus_2n_equals_1_l2754_275458

/-- Two monomials are like terms if they have the same variables with the same exponents. -/
def are_like_terms (m n : ℕ) : Prop :=
  m = 3 ∧ n = 1

/-- The theorem states that if 3x^m*y and -5x^3*y^n are like terms, then m - 2n = 1. -/
theorem like_terms_imply_m_minus_2n_equals_1 (m n : ℕ) :
  are_like_terms m n → m - 2*n = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_m_minus_2n_equals_1_l2754_275458


namespace NUMINAMATH_CALUDE_button_probability_l2754_275468

def initial_red_c : ℕ := 6
def initial_green_c : ℕ := 12
def initial_total_c : ℕ := initial_red_c + initial_green_c

def remaining_fraction : ℚ := 3/4

theorem button_probability : 
  ∃ (removed_red removed_green : ℕ),
    removed_red = removed_green ∧
    initial_total_c - (removed_red + removed_green) = (remaining_fraction * initial_total_c).num ∧
    (initial_green_c - removed_green : ℚ) / (initial_total_c - (removed_red + removed_green) : ℚ) *
    (removed_green : ℚ) / ((removed_red + removed_green) : ℚ) = 5/14 :=
by sorry

end NUMINAMATH_CALUDE_button_probability_l2754_275468


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2754_275425

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (1 - Complex.I) / Complex.I
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2754_275425


namespace NUMINAMATH_CALUDE_knights_round_table_l2754_275410

theorem knights_round_table (n : ℕ) (h : n = 25) :
  let total_arrangements := n * (n + 1) * (n + 2) / 6
  let non_adjacent_arrangements := n * (n - 3) * (n - 4) / 2
  (total_arrangements - non_adjacent_arrangements : ℚ) / total_arrangements = 11 / 46 :=
by sorry

end NUMINAMATH_CALUDE_knights_round_table_l2754_275410


namespace NUMINAMATH_CALUDE_negative_abs_equals_opposite_l2754_275416

theorem negative_abs_equals_opposite (x : ℝ) : x < 0 → |x| = -x := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_equals_opposite_l2754_275416


namespace NUMINAMATH_CALUDE_max_gcd_15n_plus_4_8n_plus_1_l2754_275493

theorem max_gcd_15n_plus_4_8n_plus_1 :
  ∃ (k : ℕ), k > 0 ∧ Nat.gcd (15 * k + 4) (8 * k + 1) = 17 ∧
  ∀ (n : ℕ), n > 0 → Nat.gcd (15 * n + 4) (8 * n + 1) ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_15n_plus_4_8n_plus_1_l2754_275493


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l2754_275421

theorem weight_loss_challenge (initial_weight : ℝ) (h_initial_weight_pos : initial_weight > 0) :
  let weight_after_loss := initial_weight * (1 - 0.11)
  let measured_weight_loss_percentage := 0.0922
  ∃ (clothes_weight_percentage : ℝ),
    weight_after_loss * (1 + clothes_weight_percentage) = initial_weight * (1 - measured_weight_loss_percentage) ∧
    clothes_weight_percentage = 0.02 :=
by sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l2754_275421


namespace NUMINAMATH_CALUDE_quadratic_roots_l2754_275496

/-- Represents a quadratic equation of the form 2x^2 + (m+2)x + m = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  2 * x^2 + (m + 2) * x + m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (m + 2)^2 - 4 * 2 * m

theorem quadratic_roots (m : ℝ) :
  (∀ x, ∃ y z, quadratic_equation m x → x = y ∨ x = z) ∧
  (discriminant 2 = 0) ∧
  (quadratic_equation 2 (-1) ∧ ∀ x, quadratic_equation 2 x → x = -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2754_275496


namespace NUMINAMATH_CALUDE_least_positive_angle_phi_l2754_275422

theorem least_positive_angle_phi : 
  ∃ φ : Real, φ > 0 ∧ φ ≤ π/2 ∧ 
  Real.cos (15 * π/180) = Real.sin (45 * π/180) + Real.sin φ ∧
  ∀ ψ : Real, ψ > 0 ∧ ψ < φ → 
    Real.cos (15 * π/180) ≠ Real.sin (45 * π/180) + Real.sin ψ ∧
  φ = 15 * π/180 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_angle_phi_l2754_275422


namespace NUMINAMATH_CALUDE_prob_different_suits_enlarged_deck_l2754_275417

/-- A deck of cards with five suits -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h1 : total_cards = num_suits * cards_per_suit)
  (h2 : num_suits = 5)

/-- The probability of drawing two cards of different suits -/
def prob_different_suits (d : Deck) : ℚ :=
  (d.total_cards - d.cards_per_suit) / (d.total_cards - 1)

/-- The main theorem -/
theorem prob_different_suits_enlarged_deck :
  ∃ d : Deck, d.total_cards = 65 ∧ prob_different_suits d = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_enlarged_deck_l2754_275417


namespace NUMINAMATH_CALUDE_train_speed_l2754_275404

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 240) (h2 : time = 6) :
  length / time = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2754_275404


namespace NUMINAMATH_CALUDE_library_visitors_average_l2754_275471

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays : ℕ := 5
  let totalOtherDays : ℕ := 25
  let totalVisitors : ℕ := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  totalVisitors / 30

/-- Theorem stating that the average number of visitors per day is 285 -/
theorem library_visitors_average (sundayVisitors : ℕ) (otherDayVisitors : ℕ) 
    (h1 : sundayVisitors = 510) (h2 : otherDayVisitors = 240) : 
    averageVisitorsPerDay sundayVisitors otherDayVisitors = 285 := by
  sorry

#eval averageVisitorsPerDay 510 240

end NUMINAMATH_CALUDE_library_visitors_average_l2754_275471


namespace NUMINAMATH_CALUDE_evaluate_g_l2754_275490

/-- The function g(x) = 3x^2 - 5x + 8 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 8

/-- Theorem: 3g(2) + 2g(-2) = 90 -/
theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l2754_275490


namespace NUMINAMATH_CALUDE_max_area_right_triangle_l2754_275467

-- Define a right-angled triangle with integer side lengths
def RightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the perimeter constraint
def Perimeter (a b c : ℕ) : Prop :=
  a + b + c = 48

-- Define the area of a triangle
def Area (a b : ℕ) : ℕ :=
  a * b / 2

-- Theorem statement
theorem max_area_right_triangle :
  ∀ a b c : ℕ,
  RightTriangle a b c →
  Perimeter a b c →
  Area a b ≤ 288 :=
sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_l2754_275467


namespace NUMINAMATH_CALUDE_tree_space_for_given_conditions_l2754_275427

/-- Calculates the sidewalk space taken by each tree given the street length, number of trees, and space between trees. -/
def tree_space (street_length : ℕ) (num_trees : ℕ) (space_between : ℕ) : ℚ :=
  let total_gap_space := (num_trees - 1) * space_between
  let total_tree_space := street_length - total_gap_space
  (total_tree_space : ℚ) / num_trees

/-- Theorem stating that for a 151-foot street with 16 trees and 9 feet between each tree, each tree takes up 1 square foot of sidewalk space. -/
theorem tree_space_for_given_conditions :
  tree_space 151 16 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tree_space_for_given_conditions_l2754_275427


namespace NUMINAMATH_CALUDE_nails_to_buy_proof_l2754_275465

/-- Given the total number of nails needed, the number of nails already owned,
    and the number of nails found in the toolshed, calculate the number of nails
    that need to be bought. -/
def nails_to_buy (total_needed : ℕ) (already_owned : ℕ) (found_in_toolshed : ℕ) : ℕ :=
  total_needed - (already_owned + found_in_toolshed)

/-- Prove that the number of nails needed to buy is 109 given the specific quantities. -/
theorem nails_to_buy_proof :
  nails_to_buy 500 247 144 = 109 := by
  sorry

end NUMINAMATH_CALUDE_nails_to_buy_proof_l2754_275465


namespace NUMINAMATH_CALUDE_remainder_3056_div_78_l2754_275429

theorem remainder_3056_div_78 : 3056 % 78 = 14 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3056_div_78_l2754_275429


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2754_275499

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (9 - 5 * x) = 8 → x = -11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2754_275499


namespace NUMINAMATH_CALUDE_trees_that_died_haley_trees_died_l2754_275476

theorem trees_that_died (total : ℕ) (survived_more : ℕ) : ℕ :=
  let died := (total - survived_more) / 2
  died

theorem haley_trees_died : trees_that_died 11 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trees_that_died_haley_trees_died_l2754_275476


namespace NUMINAMATH_CALUDE_yellow_gumdrops_after_replacement_l2754_275485

/-- Represents the number of gumdrops of each color in a jar -/
structure GumdropsJar where
  blue : ℕ
  brown : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- The total number of gumdrops in the jar -/
def GumdropsJar.total (jar : GumdropsJar) : ℕ :=
  jar.blue + jar.brown + jar.red + jar.yellow + jar.green

/-- The percentage of gumdrops of a given color -/
def GumdropsJar.percentage (jar : GumdropsJar) (color : ℕ) : ℚ :=
  color / jar.total

theorem yellow_gumdrops_after_replacement (jar : GumdropsJar) :
  jar.blue = (jar.total * 2) / 5 →
  jar.brown = (jar.total * 3) / 20 →
  jar.red = jar.total / 10 →
  jar.yellow = jar.total / 5 →
  jar.green = 50 →
  (jar.yellow + jar.red / 3 : ℕ) = 78 := by
  sorry

end NUMINAMATH_CALUDE_yellow_gumdrops_after_replacement_l2754_275485


namespace NUMINAMATH_CALUDE_set_A_determination_l2754_275462

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}

theorem set_A_determination (A : Set ℕ) (h : (U \ A) = {2, 3}) : A = {1, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_A_determination_l2754_275462


namespace NUMINAMATH_CALUDE_triangle_double_angle_sine_sum_l2754_275450

/-- For angles α, β, and γ of a triangle, sin 2α + sin 2β + sin 2γ = 4 sin α sin β sin γ -/
theorem triangle_double_angle_sine_sum (α β γ : ℝ) 
  (h : α + β + γ = Real.pi) : 
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 
  4 * Real.sin α * Real.sin β * Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_double_angle_sine_sum_l2754_275450


namespace NUMINAMATH_CALUDE_roots_sum_theorem_l2754_275477

theorem roots_sum_theorem (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a + b + c = 6 →
  a*b + a*c + b*c = 11 →
  a*b*c = 6 →
  (a / (b*c + 2)) + (b / (a*c + 2)) + (c / (a*b + 2)) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_theorem_l2754_275477


namespace NUMINAMATH_CALUDE_handshake_arrangement_theorem_l2754_275451

-- Define the number of people in the group
def num_people : ℕ := 12

-- Define the number of handshakes per person
def handshakes_per_person : ℕ := 3

-- Define the function to calculate the number of distinct handshaking arrangements
def num_arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

-- Define the function to calculate the remainder when divided by 1000
def remainder_mod_1000 (x : ℕ) : ℕ := x % 1000

-- Theorem statement
theorem handshake_arrangement_theorem :
  num_arrangements num_people handshakes_per_person = 680680 ∧
  remainder_mod_1000 (num_arrangements num_people handshakes_per_person) = 680 := by
  sorry

end NUMINAMATH_CALUDE_handshake_arrangement_theorem_l2754_275451


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2754_275464

theorem triangle_angle_measure (A B C : ℝ) : 
  -- ABC is a triangle (sum of angles is 180°)
  A + B + C = 180 →
  -- Measure of angle C is 3/2 times the measure of angle B
  C = (3/2) * B →
  -- Angle B measures 30°
  B = 30 →
  -- Then the measure of angle A is 105°
  A = 105 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2754_275464


namespace NUMINAMATH_CALUDE_K_set_equals_target_set_l2754_275418

/-- The set of natural numbers K satisfying the given conditions for a fixed h = 2^r -/
def K_set (r : ℕ) : Set ℕ :=
  {K : ℕ | ∃ (m n : ℕ), m > 1 ∧ Odd m ∧
    K ∣ (m^(2^r) - 1) ∧
    K ∣ (n^((m^(2^r) - 1) / K) + 1)}

/-- The set of numbers of the form 2^(r+s) * t where t is odd -/
def target_set (r : ℕ) : Set ℕ :=
  {K : ℕ | ∃ (s t : ℕ), K = 2^(r+s) * t ∧ Odd t}

/-- The main theorem stating that K_set equals target_set for any non-negative integer r -/
theorem K_set_equals_target_set (r : ℕ) : K_set r = target_set r := by
  sorry

end NUMINAMATH_CALUDE_K_set_equals_target_set_l2754_275418


namespace NUMINAMATH_CALUDE_expression_evaluation_l2754_275400

theorem expression_evaluation : 2 * (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 94 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2754_275400


namespace NUMINAMATH_CALUDE_second_book_cost_l2754_275442

/-- Proves that the cost of the second book is $4 given the conditions of Shelby's book fair purchases. -/
theorem second_book_cost (initial_amount : ℕ) (first_book_cost : ℕ) (poster_cost : ℕ) (posters_bought : ℕ) :
  initial_amount = 20 →
  first_book_cost = 8 →
  poster_cost = 4 →
  posters_bought = 2 →
  ∃ (second_book_cost : ℕ),
    second_book_cost + first_book_cost + (poster_cost * posters_bought) = initial_amount ∧
    second_book_cost = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_second_book_cost_l2754_275442


namespace NUMINAMATH_CALUDE_product_of_constrained_integers_l2754_275470

theorem product_of_constrained_integers (a b : ℕ) 
  (h1 : 90 < a + b ∧ a + b < 99)
  (h2 : (9 : ℚ)/10 < (a : ℚ)/(b : ℚ) ∧ (a : ℚ)/(b : ℚ) < (91 : ℚ)/100) :
  a * b = 2346 := by
  sorry

end NUMINAMATH_CALUDE_product_of_constrained_integers_l2754_275470


namespace NUMINAMATH_CALUDE_radhika_games_count_l2754_275484

/-- The number of video games Radhika owns now -/
def total_games (christmas_games birthday_games family_games : ℕ) : ℕ :=
  let total_gifts := christmas_games + birthday_games + family_games
  let initial_games := (2 * total_gifts) / 3
  initial_games + total_gifts

/-- Theorem stating the total number of video games Radhika owns -/
theorem radhika_games_count :
  total_games 12 8 5 = 41 := by
  sorry

#eval total_games 12 8 5

end NUMINAMATH_CALUDE_radhika_games_count_l2754_275484


namespace NUMINAMATH_CALUDE_parallel_vectors_cos_2alpha_l2754_275472

theorem parallel_vectors_cos_2alpha (α : ℝ) :
  let a : ℝ × ℝ := (1/3, Real.tan α)
  let b : ℝ × ℝ := (Real.cos α, 1)
  (∃ (k : ℝ), a = k • b) → Real.cos (2 * α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_cos_2alpha_l2754_275472


namespace NUMINAMATH_CALUDE_money_division_l2754_275408

/-- Given a division of money among three people a, b, and c, where b's share is 65% of a's
    and c's share is 40% of a's, and c's share is 64 rupees, prove that the total sum is 328 rupees. -/
theorem money_division (a b c : ℝ) : 
  (b = 0.65 * a) →  -- b's share is 65% of a's
  (c = 0.40 * a) →  -- c's share is 40% of a's
  (c = 64) →        -- c's share is 64 rupees
  (a + b + c = 328) -- total sum is 328 rupees
:= by sorry

end NUMINAMATH_CALUDE_money_division_l2754_275408


namespace NUMINAMATH_CALUDE_seth_candy_bars_l2754_275424

theorem seth_candy_bars (max_candy_bars : ℕ) (seth_candy_bars : ℕ) : 
  max_candy_bars = 24 →
  seth_candy_bars = 3 * max_candy_bars + 6 →
  seth_candy_bars = 78 :=
by sorry

end NUMINAMATH_CALUDE_seth_candy_bars_l2754_275424


namespace NUMINAMATH_CALUDE_program_result_l2754_275444

/-- The smallest positive integer n for which n² + 4n ≥ 10000 -/
def smallest_n : ℕ := 99

/-- The function that computes x given n -/
def x (n : ℕ) : ℕ := 3 + 2 * n

/-- The function that computes S given n -/
def S (n : ℕ) : ℕ := n^2 + 4*n

theorem program_result :
  (∀ m : ℕ, m < smallest_n → S m < 10000) ∧
  S smallest_n ≥ 10000 ∧
  x smallest_n = 201 := by sorry

end NUMINAMATH_CALUDE_program_result_l2754_275444


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2754_275495

theorem regular_polygon_sides (D : ℕ) : D = 20 → ∃ n : ℕ, n = 8 ∧ D = n * (n - 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2754_275495


namespace NUMINAMATH_CALUDE_win_loss_opposite_win_loss_opposite_meanings_l2754_275486

/-- Represents the outcome of a game -/
inductive GameOutcome
| Win
| Loss

/-- Represents a team's or individual's record -/
structure Record where
  wins : ℕ
  losses : ℕ

/-- Updates the record based on a game outcome -/
def updateRecord (r : Record) (outcome : GameOutcome) : Record :=
  match outcome with
  | GameOutcome.Win => { wins := r.wins + 1, losses := r.losses }
  | GameOutcome.Loss => { wins := r.wins, losses := r.losses + 1 }

/-- Theorem stating that winning and losing have opposite effects on a record -/
theorem win_loss_opposite (r : Record) :
  updateRecord r GameOutcome.Win ≠ updateRecord r GameOutcome.Loss :=
by
  sorry

/-- Theorem stating that winning and losing are quantities with opposite meanings -/
theorem win_loss_opposite_meanings :
  ∃ (r : Record), updateRecord r GameOutcome.Win ≠ updateRecord r GameOutcome.Loss :=
by
  sorry

end NUMINAMATH_CALUDE_win_loss_opposite_win_loss_opposite_meanings_l2754_275486


namespace NUMINAMATH_CALUDE_inequality_proof_l2754_275447

theorem inequality_proof (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0) 
  (h_eq : (x^3 + y^3)^4 = z^3 + w^3) : 
  x^4*z + y^4*w ≥ z*w := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2754_275447


namespace NUMINAMATH_CALUDE_smallest_divisible_number_proof_l2754_275482

/-- The smallest 5-digit number divisible by 15, 32, 45, and a multiple of 9 and 6 -/
def smallest_divisible_number : ℕ := 11520

theorem smallest_divisible_number_proof :
  smallest_divisible_number ≥ 10000 ∧
  smallest_divisible_number < 100000 ∧
  smallest_divisible_number % 15 = 0 ∧
  smallest_divisible_number % 32 = 0 ∧
  smallest_divisible_number % 45 = 0 ∧
  smallest_divisible_number % 9 = 0 ∧
  smallest_divisible_number % 6 = 0 ∧
  ∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧
    n % 15 = 0 ∧ n % 32 = 0 ∧ n % 45 = 0 ∧ n % 9 = 0 ∧ n % 6 = 0 →
    n ≥ smallest_divisible_number :=
by sorry

#eval smallest_divisible_number

end NUMINAMATH_CALUDE_smallest_divisible_number_proof_l2754_275482


namespace NUMINAMATH_CALUDE_area_ratio_in_special_triangle_l2754_275434

-- Define the triangle ABC and point D
variable (A B C D : ℝ × ℝ)

-- Define the properties of the triangle and point D
def is_equilateral (A B C : ℝ × ℝ) : Prop := sorry

def on_side (D A C : ℝ × ℝ) : Prop := sorry

def angle_measure (B D C : ℝ × ℝ) : ℝ := sorry

def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_in_special_triangle 
  (h_equilateral : is_equilateral A B C)
  (h_on_side : on_side D A C)
  (h_angle : angle_measure B D C = 30) :
  triangle_area A D B / triangle_area C D B = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_in_special_triangle_l2754_275434


namespace NUMINAMATH_CALUDE_series_sum_equals_one_over_200_l2754_275438

/-- The nth term of the series -/
def seriesTerm (n : ℕ) : ℚ :=
  (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

/-- The sum of the series -/
noncomputable def seriesSum : ℚ := ∑' n, seriesTerm n

/-- Theorem stating that the sum of the series is 1/200 -/
theorem series_sum_equals_one_over_200 : seriesSum = 1 / 200 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_over_200_l2754_275438


namespace NUMINAMATH_CALUDE_expression_value_l2754_275420

theorem expression_value (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 384) 
  (h2 : 3*m*n + 2*n^2 = 560) : 
  2*m^2 + 13*m*n + 6*n^2 - 444 = 2004 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2754_275420


namespace NUMINAMATH_CALUDE_smallest_multiple_of_5_and_21_l2754_275441

theorem smallest_multiple_of_5_and_21 : ∃ b : ℕ+, 
  (∀ k : ℕ+, 5 ∣ k ∧ 21 ∣ k → b ≤ k) ∧ 5 ∣ b ∧ 21 ∣ b ∧ b = 105 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_5_and_21_l2754_275441


namespace NUMINAMATH_CALUDE_cake_after_four_trips_l2754_275419

/-- The fraction of cake remaining after a given number of trips to the pantry -/
def cakeRemaining (trips : ℕ) : ℚ :=
  (1 : ℚ) / 2^trips

/-- The theorem stating that after 4 trips, 1/16 of the cake remains -/
theorem cake_after_four_trips :
  cakeRemaining 4 = (1 : ℚ) / 16 := by
  sorry

#eval cakeRemaining 4

end NUMINAMATH_CALUDE_cake_after_four_trips_l2754_275419


namespace NUMINAMATH_CALUDE_cube_root_of_a_plus_one_l2754_275401

theorem cube_root_of_a_plus_one (a : ℕ) (x : ℝ) (h : x ^ 2 = a) :
  (a + 1 : ℝ) ^ (1/3) = (x ^ 2 + 1) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_a_plus_one_l2754_275401


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l2754_275488

/-- The function f(x) = (ax - 1)e^x is monotonically increasing on [0,1] if and only if a ≥ 1 -/
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, Monotone (fun x => (a * x - 1) * Real.exp x)) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l2754_275488


namespace NUMINAMATH_CALUDE_extremum_point_monotonicity_positive_when_m_leq_2_l2754_275448

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := Real.exp x - Real.log (x + m)

-- Theorem for the extremum point condition
theorem extremum_point (m : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f x m ≥ f 0 m ∨ f x m ≤ f 0 m) → 
  (deriv (f · m)) 0 = 0 := 
sorry

-- Theorem for monotonicity of f(x)
theorem monotonicity (m : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (f x₁ m < f x₂ m ∨ f x₁ m > f x₂ m) := 
sorry

-- Theorem for f(x) > 0 when m ≤ 2
theorem positive_when_m_leq_2 (x m : ℝ) : 
  m ≤ 2 → f x m > 0 := 
sorry

end NUMINAMATH_CALUDE_extremum_point_monotonicity_positive_when_m_leq_2_l2754_275448


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2754_275494

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : isArithmeticSequence a)
  (h_fifth : a 5 = 10)
  (h_sum : a 1 + a 2 + a 3 = 3) :
  a 1 = -2 ∧ ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
  sorry

#check arithmetic_sequence_property

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2754_275494


namespace NUMINAMATH_CALUDE_ellipse_intersection_slope_l2754_275428

/-- Given an ellipse mx^2 + ny^2 = 1 intersecting with y = 1 - x at A and B,
    if the slope of the line through origin and midpoint of AB is √2, then m/n = √2 -/
theorem ellipse_intersection_slope (m n : ℝ) (A B : ℝ × ℝ) :
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (m * x₁^2 + n * y₁^2 = 1) →
  (m * x₂^2 + n * y₂^2 = 1) →
  (y₁ = 1 - x₁) →
  (y₂ = 1 - x₂) →
  ((y₁ + y₂) / (x₁ + x₂) = Real.sqrt 2) →
  m / n = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_slope_l2754_275428


namespace NUMINAMATH_CALUDE_alice_fruit_consumption_impossible_l2754_275431

/-- Represents the number of each type of fruit in the basket -/
structure FruitBasket :=
  (apples : ℕ)
  (pears : ℕ)
  (oranges : ℕ)

/-- Represents Alice's fruit consumption for a day -/
inductive DailyConsumption
  | AP  -- Apple and Pear
  | AO  -- Apple and Orange
  | PO  -- Pear and Orange

def initial_basket : FruitBasket :=
  { apples := 5, pears := 8, oranges := 11 }

def consume_fruits (basket : FruitBasket) (consumption : DailyConsumption) : FruitBasket :=
  match consumption with
  | DailyConsumption.AP => { apples := basket.apples - 1, pears := basket.pears - 1, oranges := basket.oranges }
  | DailyConsumption.AO => { apples := basket.apples - 1, pears := basket.pears, oranges := basket.oranges - 1 }
  | DailyConsumption.PO => { apples := basket.apples, pears := basket.pears - 1, oranges := basket.oranges - 1 }

def fruits_equal (basket : FruitBasket) : Prop :=
  basket.apples = basket.pears ∧ basket.pears = basket.oranges

theorem alice_fruit_consumption_impossible :
  ∀ (days : ℕ) (consumptions : List DailyConsumption),
    days = consumptions.length →
    ¬(fruits_equal (consumptions.foldl consume_fruits initial_basket)) :=
  sorry


end NUMINAMATH_CALUDE_alice_fruit_consumption_impossible_l2754_275431


namespace NUMINAMATH_CALUDE_monthly_compounding_greater_than_annual_l2754_275415

theorem monthly_compounding_greater_than_annual : 
  (1 + 0.04 / 12) ^ 12 > 1 + 0.04 := by
  sorry

end NUMINAMATH_CALUDE_monthly_compounding_greater_than_annual_l2754_275415


namespace NUMINAMATH_CALUDE_distance_a_travels_is_60km_l2754_275437

/-- Represents the movement of two objects towards each other with doubling speed -/
structure DoubleSpeedMeeting where
  initial_distance : ℝ
  initial_speed_a : ℝ
  initial_speed_b : ℝ

/-- Calculates the distance traveled by object a until meeting object b -/
def distance_traveled_by_a (meeting : DoubleSpeedMeeting) : ℝ :=
  sorry

/-- Theorem stating that given the specific initial conditions, a travels 60 km until meeting b -/
theorem distance_a_travels_is_60km :
  let meeting := DoubleSpeedMeeting.mk 90 10 5
  distance_traveled_by_a meeting = 60 := by
  sorry

end NUMINAMATH_CALUDE_distance_a_travels_is_60km_l2754_275437


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2754_275436

theorem triangle_abc_properties (A B C : ℝ) (AB AC BC : ℝ) 
  (h_triangle : A + B + C = π)
  (h_AB : AB = 2)
  (h_AC : AC = 3)
  (h_BC : BC = Real.sqrt 7) : 
  A = π / 3 ∧ Real.cos (B - C) = 11 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2754_275436


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2754_275411

theorem solve_linear_equation (x : ℝ) : 2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2754_275411


namespace NUMINAMATH_CALUDE_rhombus_area_from_quadratic_roots_l2754_275459

theorem rhombus_area_from_quadratic_roots : ∀ (d₁ d₂ : ℝ),
  d₁^2 - 10*d₁ + 24 = 0 →
  d₂^2 - 10*d₂ + 24 = 0 →
  d₁ ≠ d₂ →
  (1/2) * d₁ * d₂ = 12 := by
sorry

end NUMINAMATH_CALUDE_rhombus_area_from_quadratic_roots_l2754_275459


namespace NUMINAMATH_CALUDE_six_at_three_equals_six_l2754_275439

/-- The @ operation for positive integers a and b where a > b -/
def at_op (a b : ℕ+) (h : a > b) : ℚ :=
  (a * b : ℚ) / (a - b)

/-- Theorem: 6 @ 3 = 6 -/
theorem six_at_three_equals_six :
  ∀ (h : (6 : ℕ+) > (3 : ℕ+)), at_op 6 3 h = 6 := by sorry

end NUMINAMATH_CALUDE_six_at_three_equals_six_l2754_275439


namespace NUMINAMATH_CALUDE_arctangent_inequalities_l2754_275457

theorem arctangent_inequalities (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.arctan x + Real.arctan y < π / 2 ↔ x * y < 1) ∧
  (Real.arctan x + Real.arctan y + Real.arctan z < π ↔ x * y * z < x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_arctangent_inequalities_l2754_275457


namespace NUMINAMATH_CALUDE_percent_of_number_l2754_275466

theorem percent_of_number : (25 : ℝ) / 100 * 280 = 70 := by sorry

end NUMINAMATH_CALUDE_percent_of_number_l2754_275466


namespace NUMINAMATH_CALUDE_simplify_expression_l2754_275440

theorem simplify_expression (x : ℝ) : 2 * (x - 3) - (-x + 4) = 3 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2754_275440


namespace NUMINAMATH_CALUDE_area_between_specific_lines_l2754_275489

/-- Line passing through two points -/
structure Line where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Calculate the area between two lines within a given x-range -/
noncomputable def areaBetweenLines (l1 l2 : Line) (x_start x_end : ℝ) : ℝ :=
  sorry

/-- The problem statement -/
theorem area_between_specific_lines :
  let line1 : Line := { x1 := 0, y1 := 5, x2 := 10, y2 := 2 }
  let line2 : Line := { x1 := 2, y1 := 6, x2 := 6, y2 := 0 }
  areaBetweenLines line1 line2 2 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_between_specific_lines_l2754_275489


namespace NUMINAMATH_CALUDE_average_annual_decrease_rate_optimal_price_reduction_l2754_275409

-- Part 1: Average annual percentage decrease
def initial_price : ℝ := 200
def final_price : ℝ := 162
def num_years : ℕ := 2

-- Part 2: Unit price reduction
def selling_price : ℝ := 200
def initial_daily_sales : ℕ := 20
def price_decrease_step : ℝ := 3
def sales_increase_step : ℕ := 6
def target_daily_profit : ℝ := 1150

-- Theorem for Part 1
theorem average_annual_decrease_rate (x : ℝ) :
  initial_price * (1 - x)^num_years = final_price →
  x = 0.1 := by sorry

-- Theorem for Part 2
theorem optimal_price_reduction (m : ℝ) :
  (selling_price - m - 162) * (initial_daily_sales + 2 * m) = target_daily_profit →
  m = 15 := by sorry

end NUMINAMATH_CALUDE_average_annual_decrease_rate_optimal_price_reduction_l2754_275409


namespace NUMINAMATH_CALUDE_not_perfect_square_l2754_275402

theorem not_perfect_square (n : ℕ) (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) : 
  ¬ ∃ k : ℕ, a * 10^(n+1) + 9 = k^2 :=
sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2754_275402


namespace NUMINAMATH_CALUDE_opposite_sides_equal_implies_parallelogram_condition_b_implies_parallelogram_l2754_275449

/-- A quadrilateral in a 2D plane --/
structure Quadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

/-- Definition of a parallelogram --/
def is_parallelogram {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) : Prop :=
  q.A - q.B = q.D - q.C ∧ q.A - q.D = q.B - q.C

/-- Theorem: If opposite sides of a quadrilateral are equal, it is a parallelogram --/
theorem opposite_sides_equal_implies_parallelogram 
  {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) :
  q.A - q.D = q.B - q.C → q.A - q.B = q.D - q.C → is_parallelogram q :=
by sorry

/-- Main theorem: If AD=BC and AB=DC, then ABCD is a parallelogram --/
theorem condition_b_implies_parallelogram 
  {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) :
  q.A - q.D = q.B - q.C → q.A - q.B = q.D - q.C → is_parallelogram q :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_equal_implies_parallelogram_condition_b_implies_parallelogram_l2754_275449


namespace NUMINAMATH_CALUDE_stadium_entry_count_l2754_275487

def basket_capacity : ℕ := 4634
def placards_per_person : ℕ := 2

theorem stadium_entry_count :
  let total_placards : ℕ := basket_capacity
  let people_entered : ℕ := total_placards / placards_per_person
  people_entered = 2317 := by sorry

end NUMINAMATH_CALUDE_stadium_entry_count_l2754_275487


namespace NUMINAMATH_CALUDE_weight_ratio_l2754_275492

theorem weight_ratio (sam_weight tyler_weight peter_weight : ℝ) : 
  tyler_weight = sam_weight + 25 →
  sam_weight = 105 →
  peter_weight = 65 →
  peter_weight / tyler_weight = 0.5 := by
sorry

end NUMINAMATH_CALUDE_weight_ratio_l2754_275492


namespace NUMINAMATH_CALUDE_triangle_quadratic_no_real_roots_l2754_275478

/-- Given a triangle with side lengths a, b, c, the quadratic equation 
    b^2 x^2 - (b^2 + c^2 - a^2)x + c^2 = 0 has no real roots. -/
theorem triangle_quadratic_no_real_roots (a b c : ℝ) 
    (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
    ∀ x : ℝ, b^2 * x^2 - (b^2 + c^2 - a^2) * x + c^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_quadratic_no_real_roots_l2754_275478


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2754_275461

/-- Two 2D vectors are perpendicular if their dot product is zero -/
def are_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

/-- The problem statement -/
theorem perpendicular_vectors (a : ℝ) (h1 : a ≠ 0) 
  (h2 : are_perpendicular (a, a+4) (-5, a)) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2754_275461


namespace NUMINAMATH_CALUDE_prove_not_p_or_not_q_l2754_275456

theorem prove_not_p_or_not_q (h1 : ¬(p ∧ q)) (h2 : p ∨ q) : ¬p ∨ ¬q := by
  sorry

end NUMINAMATH_CALUDE_prove_not_p_or_not_q_l2754_275456


namespace NUMINAMATH_CALUDE_toy_store_shelves_l2754_275407

def number_of_shelves (initial_stock new_shipment bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

theorem toy_store_shelves : 
  number_of_shelves 4 10 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l2754_275407


namespace NUMINAMATH_CALUDE_parabola_shift_l2754_275403

/-- Given a parabola y = x^2 + 2, shifting it 3 units left and 4 units down results in y = (x + 3)^2 - 2 -/
theorem parabola_shift (x y : ℝ) : 
  (y = x^2 + 2) → 
  (y = (x + 3)^2 - 2) ↔ 
  (y + 4 = ((x + 3) + 3)^2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l2754_275403
