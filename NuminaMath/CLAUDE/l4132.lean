import Mathlib

namespace NUMINAMATH_CALUDE_joan_bought_72_eggs_l4132_413287

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Joan bought -/
def dozens_bought : ℕ := 6

/-- The total number of eggs Joan bought -/
def total_eggs : ℕ := dozens_bought * eggs_per_dozen

theorem joan_bought_72_eggs : total_eggs = 72 := by
  sorry

end NUMINAMATH_CALUDE_joan_bought_72_eggs_l4132_413287


namespace NUMINAMATH_CALUDE_train_speed_l4132_413212

/-- The speed of a train given its length and time to cross an electric pole. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 3500) (h2 : time = 80) :
  length / time = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l4132_413212


namespace NUMINAMATH_CALUDE_max_a_is_maximum_l4132_413245

/-- The maximum value of a such that the line y = mx + 1 does not pass through
    any lattice points for 0 < x ≤ 200 and 1/2 < m < a -/
def max_a : ℚ := 101 / 201

/-- Predicate to check if a point (x, y) is a lattice point -/
def is_lattice_point (x y : ℚ) : Prop := ∃ (n m : ℤ), x = n ∧ y = m

/-- Predicate to check if the line y = mx + 1 passes through a lattice point -/
def line_passes_lattice_point (m : ℚ) (x : ℚ) : Prop :=
  ∃ (y : ℚ), is_lattice_point x y ∧ y = m * x + 1

theorem max_a_is_maximum :
  ∀ (a : ℚ), (∀ (m : ℚ), 1/2 < m → m < a →
    ∀ (x : ℚ), 0 < x → x ≤ 200 → ¬ line_passes_lattice_point m x) →
  a ≤ max_a :=
sorry

end NUMINAMATH_CALUDE_max_a_is_maximum_l4132_413245


namespace NUMINAMATH_CALUDE_number_of_girls_l4132_413221

theorem number_of_girls (total_children : Nat) (happy_children : Nat) (sad_children : Nat) 
  (neutral_children : Nat) (boys : Nat) (happy_boys : Nat) (sad_girls : Nat) (neutral_boys : Nat) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 16 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 4 →
  total_children = happy_children + sad_children + neutral_children →
  total_children - boys = 44 := by
  sorry

#check number_of_girls

end NUMINAMATH_CALUDE_number_of_girls_l4132_413221


namespace NUMINAMATH_CALUDE_angle_equation_solution_l4132_413298

theorem angle_equation_solution (A : Real) :
  (1/2 * Real.sin (A/2) + Real.cos (A/2) = 1) → A = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_angle_equation_solution_l4132_413298


namespace NUMINAMATH_CALUDE_fraction_simplification_l4132_413243

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hyz : y^3 - 1/x ≠ 0) : 
  (x^3 - 1/y) / (y^3 - 1/x) = x / y := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4132_413243


namespace NUMINAMATH_CALUDE_quadratic_diophantine_bound_l4132_413254

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The number of integer solutions to a quadratic Diophantine equation -/
def num_solutions (A B C D E : ℤ) : ℕ := sorry

theorem quadratic_diophantine_bound
  (A B C D E : ℤ)
  (hB : B ≠ 0)
  (hF : A * D^2 - B * C * D + B^2 * E ≠ 0) :
  num_solutions A B C D E ≤ 2 * num_divisors (Int.natAbs (A * D^2 - B * C * D + B^2 * E)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_diophantine_bound_l4132_413254


namespace NUMINAMATH_CALUDE_BF_length_is_10_8_l4132_413233

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  right_angle_A : True  -- Represents the right angle at A
  right_angle_C : True  -- Represents the right angle at C
  E_on_AC : True  -- Represents that E is on AC
  F_on_AC : True  -- Represents that F is on AC
  DE_perp_AC : True  -- Represents that DE is perpendicular to AC
  BF_perp_AC : True  -- Represents that BF is perpendicular to AC
  AE_length : Real
  DE_length : Real
  CE_length : Real
  h_AE : AE_length = 4
  h_DE : DE_length = 6
  h_CE : CE_length = 8

/-- Calculate the length of BF in the given quadrilateral -/
def calculate_BF_length (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that BF length is 10.8 -/
theorem BF_length_is_10_8 (q : Quadrilateral) : calculate_BF_length q = 10.8 := by sorry

end NUMINAMATH_CALUDE_BF_length_is_10_8_l4132_413233


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_l4132_413277

/-- Given a rectangle JKLM and a square NOPQ, if 30% of JKLM's area overlaps with NOPQ,
    and 40% of NOPQ's area overlaps with JKLM, then the ratio of JKLM's length to its width is 4/3. -/
theorem rectangle_square_overlap (j l m n : ℝ) :
  j > 0 → l > 0 → m > 0 → n > 0 →
  0.3 * (j * l) = 0.4 * (n * n) →
  j * l = m * n →
  j / m = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_l4132_413277


namespace NUMINAMATH_CALUDE_trace_bag_weight_l4132_413211

/-- Given:
  - Trace has 5 shopping bags
  - Trace's 5 bags weigh the same as Gordon's 2 bags
  - One of Gordon's bags weighs 3 pounds
  - The other of Gordon's bags weighs 7 pounds
  - All of Trace's bags weigh the same amount
Prove that one of Trace's bags weighs 2 pounds -/
theorem trace_bag_weight :
  ∀ (trace_bag_count : ℕ) 
    (gordon_bag1_weight gordon_bag2_weight : ℕ)
    (trace_total_weight : ℕ),
  trace_bag_count = 5 →
  gordon_bag1_weight = 3 →
  gordon_bag2_weight = 7 →
  trace_total_weight = gordon_bag1_weight + gordon_bag2_weight →
  ∃ (trace_single_bag_weight : ℕ),
    trace_single_bag_weight * trace_bag_count = trace_total_weight ∧
    trace_single_bag_weight = 2 :=
by sorry

end NUMINAMATH_CALUDE_trace_bag_weight_l4132_413211


namespace NUMINAMATH_CALUDE_fraction_irreducible_l4132_413260

theorem fraction_irreducible (a : ℤ) : 
  Nat.gcd (Int.natAbs (a^3 + 2*a)) (Int.natAbs (a^4 + 3*a^2 + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l4132_413260


namespace NUMINAMATH_CALUDE_part_one_part_two_l4132_413279

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < 3-a}

-- Part 1
theorem part_one :
  let a : ℝ := -2
  (B a ∩ A = {x | 1 ≤ x ∧ x < 4}) ∧
  (B a ∩ (Set.univ \ A) = {x | (-4 ≤ x ∧ x < 1) ∨ (4 ≤ x ∧ x < 5)}) :=
sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, (A ∪ B a = A) ↔ a ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4132_413279


namespace NUMINAMATH_CALUDE_golden_delicious_per_pint_l4132_413232

/-- The number of pink lady apples required to make one pint of cider -/
def pink_lady_per_pint : ℕ := 40

/-- The number of apples a farmhand can pick per hour -/
def apples_per_hour : ℕ := 240

/-- The number of farmhands -/
def num_farmhands : ℕ := 6

/-- The number of hours worked -/
def hours_worked : ℕ := 5

/-- The ratio of golden delicious to pink lady apples -/
def apple_ratio : ℚ := 1 / 3

/-- The number of pints of cider that can be made -/
def pints_of_cider : ℕ := 120

theorem golden_delicious_per_pint : ℕ := by
  sorry

end NUMINAMATH_CALUDE_golden_delicious_per_pint_l4132_413232


namespace NUMINAMATH_CALUDE_divisors_between_squares_l4132_413272

theorem divisors_between_squares (m a b d : ℕ) : 
  1 ≤ m → 
  m^2 < a → a < m^2 + m → 
  m^2 < b → b < m^2 + m → 
  a ≠ b → 
  m^2 < d → d < m^2 + m → 
  d ∣ (a * b) → 
  d = a ∨ d = b :=
by sorry

end NUMINAMATH_CALUDE_divisors_between_squares_l4132_413272


namespace NUMINAMATH_CALUDE_product_expansion_l4132_413249

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l4132_413249


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l4132_413273

theorem rectangle_perimeter (L W : ℝ) 
  (h1 : L - 4 = W + 3) 
  (h2 : (L - 4) * (W + 3) = L * W) : 
  2 * L + 2 * W = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l4132_413273


namespace NUMINAMATH_CALUDE_student_group_aging_l4132_413204

/-- Represents a group of students with their average age and age variance -/
structure StudentGroup where
  averageAge : ℝ
  ageVariance : ℝ

/-- Function to calculate the new state of a StudentGroup after a given time -/
def ageStudentGroup (group : StudentGroup) (years : ℝ) : StudentGroup :=
  { averageAge := group.averageAge + years
    ageVariance := group.ageVariance }

theorem student_group_aging :
  let initialGroup : StudentGroup := { averageAge := 13, ageVariance := 3 }
  let yearsLater : ℝ := 2
  let finalGroup := ageStudentGroup initialGroup yearsLater
  finalGroup.averageAge = 15 ∧ finalGroup.ageVariance = 3 := by
  sorry


end NUMINAMATH_CALUDE_student_group_aging_l4132_413204


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4132_413266

-- Define the conditions
def p (x : ℝ) : Prop := abs x > 1
def q (x : ℝ) : Prop := x < -2

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4132_413266


namespace NUMINAMATH_CALUDE_toothpicks_count_l4132_413253

/-- The number of small triangles in a row, starting from the base --/
def num_triangles_in_row (n : ℕ) : ℕ := 2500 - n + 1

/-- The total number of small triangles in the large triangle --/
def total_small_triangles : ℕ := (2500 * 2501) / 2

/-- The number of toothpicks needed for the interior and remaining exterior of the large triangle --/
def toothpicks_needed : ℕ := ((3 * total_small_triangles) / 2) + 2 * 2500

theorem toothpicks_count : toothpicks_needed = 4694375 := by sorry

end NUMINAMATH_CALUDE_toothpicks_count_l4132_413253


namespace NUMINAMATH_CALUDE_triangle_point_inequality_l4132_413283

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point P
def Point := ℝ × ℝ

-- Function to calculate distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Function to check if a point is inside a triangle
def isInside (t : Triangle) (p : Point) : Prop := sorry

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

-- Theorem statement
theorem triangle_point_inequality (t : Triangle) (P : Point) (s : ℝ) :
  perimeter t = 2 * s →
  isInside t P →
  s < distance t.A P + distance t.B P + distance t.C P ∧
  distance t.A P + distance t.B P + distance t.C P < 2 * s :=
by sorry

end NUMINAMATH_CALUDE_triangle_point_inequality_l4132_413283


namespace NUMINAMATH_CALUDE_max_stamps_for_50_dollars_l4132_413270

theorem max_stamps_for_50_dollars (stamp_price : ℕ) (available_amount : ℕ) : 
  stamp_price = 37 → available_amount = 5000 → 
  (∃ (n : ℕ), n * stamp_price ≤ available_amount ∧ 
  ∀ (m : ℕ), m * stamp_price ≤ available_amount → m ≤ n) → 
  (∃ (max_stamps : ℕ), max_stamps = 135) := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_for_50_dollars_l4132_413270


namespace NUMINAMATH_CALUDE_inverse_g_84_l4132_413281

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- State the theorem
theorem inverse_g_84 : g⁻¹ 84 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_84_l4132_413281


namespace NUMINAMATH_CALUDE_quadratic_system_solution_l4132_413236

theorem quadratic_system_solution (a b c : ℝ) :
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧
             b * x^2 + c * x + a = 0 ∧
             c * x^2 + a * x + b = 0) ↔
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_system_solution_l4132_413236


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l4132_413206

theorem quadratic_form_ratio (b c : ℝ) : 
  (∀ x, x^2 + 1500*x + 2400 = (x + b)^2 + c) → 
  c / b = -746.8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l4132_413206


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4132_413227

theorem polynomial_factorization (x : ℝ) :
  x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4132_413227


namespace NUMINAMATH_CALUDE_walters_coins_value_l4132_413238

/-- Represents the value of a coin in cents -/
def coin_value : String → Nat
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "half_dollar" => 50
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes half_dollars : Nat) : Nat :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime" +
  half_dollars * coin_value "half_dollar"

/-- Converts a number of cents to a percentage of a dollar -/
def cents_to_percentage (cents : Nat) : Nat :=
  cents

theorem walters_coins_value :
  total_value 2 1 2 1 = 77 ∧ cents_to_percentage (total_value 2 1 2 1) = 77 := by
  sorry

end NUMINAMATH_CALUDE_walters_coins_value_l4132_413238


namespace NUMINAMATH_CALUDE_correct_sunset_time_l4132_413205

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds a duration to a time -/
def addDuration (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes
  let additionalHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  let newHours := (t.hours + d.hours + additionalHours) % 24
  { hours := newHours, minutes := newMinutes }

theorem correct_sunset_time :
  let sunrise : Time := { hours := 6, minutes := 32 }
  let daylight : Duration := { hours := 11, minutes := 35 }
  let sunset := addDuration sunrise daylight
  sunset = { hours := 18, minutes := 7 } :=
by sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l4132_413205


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l4132_413239

def prob_boy_or_girl : ℚ := 1 / 2

def family_size : ℕ := 4

theorem prob_at_least_one_boy_and_girl :
  (1 : ℚ) - (prob_boy_or_girl ^ family_size + prob_boy_or_girl ^ family_size) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l4132_413239


namespace NUMINAMATH_CALUDE_repeated_root_implies_m_equals_two_l4132_413259

theorem repeated_root_implies_m_equals_two (x m : ℝ) : 
  (2 / (x - 1) + 3 = m / (x - 1)) →  -- Condition 1
  (x - 1 = 0) →                      -- Condition 2 (repeated root implies x - 1 = 0)
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_repeated_root_implies_m_equals_two_l4132_413259


namespace NUMINAMATH_CALUDE_two_car_garage_count_l4132_413293

theorem two_car_garage_count (total_houses : ℕ) (pool_houses : ℕ) (garage_and_pool : ℕ) (neither : ℕ) :
  total_houses = 65 →
  pool_houses = 40 →
  garage_and_pool = 35 →
  neither = 10 →
  ∃ (garage_houses : ℕ), garage_houses = 50 ∧ 
    total_houses = garage_houses + pool_houses - garage_and_pool + neither :=
by sorry

end NUMINAMATH_CALUDE_two_car_garage_count_l4132_413293


namespace NUMINAMATH_CALUDE_brown_paint_red_pigment_weight_l4132_413228

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue : Real
  red : Real
  yellow : Real

/-- Represents the weight of each paint in the mixture -/
structure MixtureWeights where
  maroon : Real
  green : Real

theorem brown_paint_red_pigment_weight
  (maroon : PaintMixture)
  (green : PaintMixture)
  (weights : MixtureWeights)
  (h_maroon_comp : maroon.blue = 0.5 ∧ maroon.red = 0.5 ∧ maroon.yellow = 0)
  (h_green_comp : green.blue = 0.3 ∧ green.red = 0 ∧ green.yellow = 0.7)
  (h_total_weight : weights.maroon + weights.green = 10)
  (h_brown_blue : weights.maroon * maroon.blue + weights.green * green.blue = 4) :
  weights.maroon * maroon.red = 2.5 := by
  sorry

#check brown_paint_red_pigment_weight

end NUMINAMATH_CALUDE_brown_paint_red_pigment_weight_l4132_413228


namespace NUMINAMATH_CALUDE_point_line_plane_relations_l4132_413278

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations
variable (lies_on : Point → Line → Prop)
variable (lies_in_line : Line → Plane → Prop)
variable (lies_in_point : Point → Plane → Prop)

-- State the theorem
theorem point_line_plane_relations 
  (A : Point) (a : Line) (α : Plane) (B : Point) :
  lies_on A a → lies_in_line a α → lies_in_point B α →
  (A ∈ {x : Point | lies_on x a}) ∧ 
  ({x : Point | lies_on x a} ⊆ {x : Point | lies_in_point x α}) ∧ 
  (B ∈ {x : Point | lies_in_point x α}) :=
sorry

end NUMINAMATH_CALUDE_point_line_plane_relations_l4132_413278


namespace NUMINAMATH_CALUDE_positive_cubic_interval_l4132_413241

theorem positive_cubic_interval (x : ℝ) :
  (x + 1) * (x - 1) * (x + 3) > 0 ↔ x ∈ Set.Ioo (-3 : ℝ) (-1 : ℝ) ∪ Set.Ioi (1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_positive_cubic_interval_l4132_413241


namespace NUMINAMATH_CALUDE_number_difference_l4132_413276

theorem number_difference (L S : ℕ) (h1 : L > S) (h2 : L = 6 * S + 15) (h3 : L = 1656) : L - S = 1383 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l4132_413276


namespace NUMINAMATH_CALUDE_marias_age_l4132_413229

theorem marias_age (cooper dante maria : ℕ) 
  (sum_ages : cooper + dante + maria = 31)
  (dante_twice_cooper : dante = 2 * cooper)
  (dante_younger_maria : dante = maria - 1) : 
  maria = 13 := by
  sorry

end NUMINAMATH_CALUDE_marias_age_l4132_413229


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l4132_413237

theorem hyperbola_focal_length (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 20 = 1 → y = 2*x) → 
  let c := Real.sqrt (a^2 + 20)
  2 * c = 10 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l4132_413237


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4132_413288

/-- Given a geometric sequence {a_n} with sum of first n terms S_n = 3^n + t, prove t + a_3 = 17 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = 3^n + t) →  -- Definition of S_n
  (∀ n, a (n+1) = S (n+1) - S n) →  -- Definition of a_n in terms of S_n
  (a 2)^2 = a 1 * a 3 →  -- Property of geometric sequence
  t + a 3 = 17 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4132_413288


namespace NUMINAMATH_CALUDE_unique_line_through_5_2_l4132_413214

/-- A line in the xy-plane is represented by its x and y intercepts -/
structure Line where
  x_intercept : ℕ
  y_intercept : ℕ

/-- Check if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Check if a number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- Check if a line passes through the point (5,2) -/
def passes_through_5_2 (l : Line) : Prop :=
  5 / l.x_intercept + 2 / l.y_intercept = 1

/-- The main theorem to be proved -/
theorem unique_line_through_5_2 : 
  ∃! l : Line, 
    is_prime l.x_intercept ∧ 
    is_power_of_two l.y_intercept ∧ 
    passes_through_5_2 l :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_5_2_l4132_413214


namespace NUMINAMATH_CALUDE_square_of_binomial_l4132_413257

theorem square_of_binomial (a : ℚ) :
  (∃ b : ℚ, ∀ x : ℚ, 9 * x^2 + 15 * x + a = (3 * x + b)^2) → a = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l4132_413257


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4132_413200

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 - a 9 + a 17 = 7 →
  a 3 + a 15 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4132_413200


namespace NUMINAMATH_CALUDE_steven_shirts_l4132_413285

/-- The number of shirts owned by Brian -/
def brian_shirts : ℕ := 3

/-- The number of shirts owned by Andrew relative to Brian -/
def andrew_multiplier : ℕ := 6

/-- The number of shirts owned by Steven relative to Andrew -/
def steven_multiplier : ℕ := 4

/-- Theorem: Given the conditions, Steven has 72 shirts -/
theorem steven_shirts : 
  steven_multiplier * (andrew_multiplier * brian_shirts) = 72 := by
sorry

end NUMINAMATH_CALUDE_steven_shirts_l4132_413285


namespace NUMINAMATH_CALUDE_leading_zeros_of_fraction_l4132_413292

/-- The number of leading zeros in the decimal representation of a fraction -/
def leadingZeros (n d : ℕ) : ℕ :=
  sorry

theorem leading_zeros_of_fraction :
  leadingZeros 1 (2^3 * 5^5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_leading_zeros_of_fraction_l4132_413292


namespace NUMINAMATH_CALUDE_sum_of_ratios_geq_two_l4132_413291

theorem sum_of_ratios_geq_two (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_geq_two_l4132_413291


namespace NUMINAMATH_CALUDE_frank_work_days_l4132_413280

/-- Calculates the number of days worked given total hours and hours per day -/
def days_worked (total_hours : Float) (hours_per_day : Float) : Float :=
  total_hours / hours_per_day

/-- Theorem: Frank worked 4 days given the conditions -/
theorem frank_work_days :
  let total_hours : Float := 8.0
  let hours_per_day : Float := 2.0
  days_worked total_hours hours_per_day = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_frank_work_days_l4132_413280


namespace NUMINAMATH_CALUDE_network_connections_l4132_413240

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n * k) / 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l4132_413240


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l4132_413295

theorem sqrt_two_minus_one_power (n : ℤ) :
  ∃ k : ℤ, (Real.sqrt 2 - 1) ^ n = Real.sqrt (k + 1) - Real.sqrt k := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l4132_413295


namespace NUMINAMATH_CALUDE_ab_equals_seventeen_l4132_413218

theorem ab_equals_seventeen
  (h1 : a - b = 5)
  (h2 : a^2 + b^2 = 34)
  (h3 : a^3 - b^3 = 30)
  (h4 : a^2 + b^2 - c^2 = 50)
  (h5 : c = 2*a - b)
  : a * b = 17 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_seventeen_l4132_413218


namespace NUMINAMATH_CALUDE_abc_sum_bound_l4132_413263

theorem abc_sum_bound (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (M : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (a' b' c' : ℝ),
    a' + b' + c' = 1 ∧ a' * b' + a' * c' + b' * c' > M ∧
    a * b + a * c + b * c ≤ 1/2 ∧
    a * b + a * c + b * c < 1/2 + ε :=
sorry

end NUMINAMATH_CALUDE_abc_sum_bound_l4132_413263


namespace NUMINAMATH_CALUDE_number_difference_l4132_413265

theorem number_difference (L S : ℕ) (h1 : L = 1575) (h2 : L = 7 * S + 15) : L - S = 1353 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l4132_413265


namespace NUMINAMATH_CALUDE_necessary_condition_example_l4132_413255

theorem necessary_condition_example : 
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧ 
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_example_l4132_413255


namespace NUMINAMATH_CALUDE_point_P_on_circle_M_and_line_L_l4132_413286

/-- Circle M with center (3,2) and radius √2 -/
def circle_M (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 2

/-- Line L with equation x + y - 3 = 0 -/
def line_L (x y : ℝ) : Prop := x + y - 3 = 0

/-- Point P with coordinates (2,1) -/
def point_P : ℝ × ℝ := (2, 1)

theorem point_P_on_circle_M_and_line_L :
  circle_M point_P.1 point_P.2 ∧ line_L point_P.1 point_P.2 := by
  sorry

end NUMINAMATH_CALUDE_point_P_on_circle_M_and_line_L_l4132_413286


namespace NUMINAMATH_CALUDE_number_of_schnauzers_l4132_413294

/-- Given the number of Doberman puppies and an equation relating it to the number of Schnauzers,
    this theorem proves the number of Schnauzers. -/
theorem number_of_schnauzers (D S : ℤ) (h1 : 3*D - 5 + (D - S) = 90) (h2 : D = 20) : S = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_of_schnauzers_l4132_413294


namespace NUMINAMATH_CALUDE_baker_shopping_cost_l4132_413296

theorem baker_shopping_cost :
  let flour_boxes : ℕ := 3
  let flour_price : ℕ := 3
  let egg_trays : ℕ := 3
  let egg_price : ℕ := 10
  let milk_liters : ℕ := 7
  let milk_price : ℕ := 5
  let soda_boxes : ℕ := 2
  let soda_price : ℕ := 3
  let total_cost : ℕ := flour_boxes * flour_price + egg_trays * egg_price + 
                        milk_liters * milk_price + soda_boxes * soda_price
  total_cost = 80 := by
sorry


end NUMINAMATH_CALUDE_baker_shopping_cost_l4132_413296


namespace NUMINAMATH_CALUDE_grid_sum_theorem_l4132_413264

/-- Represents a 3x3 grid where each cell contains a number from 1 to 3 --/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Checks if a row in the grid contains 1, 2, and 3 --/
def valid_row (g : Grid) (r : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ c : Fin 3, g r c = n.succ

/-- Checks if a column in the grid contains 1, 2, and 3 --/
def valid_column (g : Grid) (c : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ r : Fin 3, g r c = n.succ

/-- Checks if the entire grid is valid --/
def valid_grid (g : Grid) : Prop :=
  (∀ r : Fin 3, valid_row g r) ∧ (∀ c : Fin 3, valid_column g c)

theorem grid_sum_theorem (g : Grid) :
  valid_grid g →
  g 0 0 = 2 →
  g 1 1 = 3 →
  g 1 2 + g 2 2 + 4 = 8 := by sorry

end NUMINAMATH_CALUDE_grid_sum_theorem_l4132_413264


namespace NUMINAMATH_CALUDE_square_area_error_l4132_413271

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l4132_413271


namespace NUMINAMATH_CALUDE_AC_length_l4132_413247

-- Define the isosceles trapezoid
structure IsoscelesTrapezoid :=
  (AB : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (BC : ℝ)
  (isIsosceles : AD = BC)

-- Define our specific trapezoid
def specificTrapezoid : IsoscelesTrapezoid :=
  { AB := 30
  , CD := 12
  , AD := 15
  , BC := 15
  , isIsosceles := rfl }

-- Theorem statement
theorem AC_length (t : IsoscelesTrapezoid) (h : t = specificTrapezoid) :
  ∃ (AC : ℝ), AC = Real.sqrt (12^2 + 20^2) :=
sorry

end NUMINAMATH_CALUDE_AC_length_l4132_413247


namespace NUMINAMATH_CALUDE_total_spent_is_124_l4132_413208

/-- The total amount spent on entertainment and additional expenses -/
def total_spent (computer_game_cost movie_ticket_cost num_tickets snack_cost transportation_cost num_trips : ℕ) : ℕ :=
  computer_game_cost + 
  movie_ticket_cost * num_tickets + 
  snack_cost + 
  transportation_cost * num_trips

/-- Theorem stating that the total amount spent is $124 given the specific costs -/
theorem total_spent_is_124 :
  total_spent 66 12 3 7 5 3 = 124 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_124_l4132_413208


namespace NUMINAMATH_CALUDE_train_length_l4132_413209

/-- Calculate the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (time : ℝ) : 
  speed = 72 → platform_length = 230 → time = 26 → 
  (speed * 1000 / 3600) * time - platform_length = 290 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l4132_413209


namespace NUMINAMATH_CALUDE_arithmetic_computation_l4132_413213

theorem arithmetic_computation : 12 + 4 * (5 - 9)^2 / 2 = 44 := by sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l4132_413213


namespace NUMINAMATH_CALUDE_train_speed_l4132_413250

/-- Calculates the speed of a train given its composition and time to cross a bridge -/
theorem train_speed (num_carriages : ℕ) (carriage_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  num_carriages = 24 →
  carriage_length = 60 →
  bridge_length = 1500 →
  crossing_time = 3 →
  (((num_carriages + 1) * carriage_length + bridge_length) / 1000) / (crossing_time / 60) = 60 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l4132_413250


namespace NUMINAMATH_CALUDE_max_daily_profit_l4132_413261

/-- Represents the daily profit function for a store selling a product -/
def daily_profit (x : ℝ) : ℝ :=
  (2 + 0.5 * x) * (200 - 10 * x)

/-- Theorem stating the maximum daily profit and the corresponding selling price -/
theorem max_daily_profit :
  ∃ (x : ℝ), daily_profit x = 720 ∧ 
  (∀ (y : ℝ), daily_profit y ≤ daily_profit x) ∧
  x = 8 :=
sorry

end NUMINAMATH_CALUDE_max_daily_profit_l4132_413261


namespace NUMINAMATH_CALUDE_exponential_function_passes_through_point_zero_one_l4132_413216

theorem exponential_function_passes_through_point_zero_one
  (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x
  f 0 = 1 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_passes_through_point_zero_one_l4132_413216


namespace NUMINAMATH_CALUDE_bus_speed_with_stoppages_l4132_413248

/-- Given a bus that travels at 90 km/hr excluding stoppages and stops for 4 minutes per hour,
    its speed including stoppages is 84 km/hr. -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages = 90 →
  stoppage_time = 4 →
  total_time = 60 →
  (speed_without_stoppages * (total_time - stoppage_time)) / total_time = 84 := by
  sorry

#check bus_speed_with_stoppages

end NUMINAMATH_CALUDE_bus_speed_with_stoppages_l4132_413248


namespace NUMINAMATH_CALUDE_SR_equals_15_l4132_413275

/-- Triangle PQR with point S on PR --/
structure TrianglePQRWithS where
  /-- Length of PQ --/
  PQ : ℝ
  /-- Length of QR --/
  QR : ℝ
  /-- Length of PS --/
  PS : ℝ
  /-- Length of QS --/
  QS : ℝ
  /-- PQ equals QR --/
  eq_PQ_QR : PQ = QR
  /-- PQ equals 10 --/
  eq_PQ_10 : PQ = 10
  /-- PS equals 6 --/
  eq_PS_6 : PS = 6
  /-- QS equals 5 --/
  eq_QS_5 : QS = 5

/-- The length of SR in the given triangle configuration --/
def SR (t : TrianglePQRWithS) : ℝ := 15

/-- Theorem: The length of SR is 15 in the given triangle configuration --/
theorem SR_equals_15 (t : TrianglePQRWithS) : SR t = 15 := by
  sorry

end NUMINAMATH_CALUDE_SR_equals_15_l4132_413275


namespace NUMINAMATH_CALUDE_max_gcd_17n_plus_4_10n_plus_3_l4132_413251

theorem max_gcd_17n_plus_4_10n_plus_3 :
  ∃ (k : ℕ), k > 0 ∧
  Nat.gcd (17 * k + 4) (10 * k + 3) = 11 ∧
  ∀ (n : ℕ), n > 0 → Nat.gcd (17 * n + 4) (10 * n + 3) ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_17n_plus_4_10n_plus_3_l4132_413251


namespace NUMINAMATH_CALUDE_least_possible_n_l4132_413231

/-- The type of rational coefficients for the polynomial terms -/
structure Coefficient where
  a : ℚ
  b : ℚ

/-- 
Checks if a list of coefficients satisfies the equation
x^2 + x + 4 = ∑(i=1 to n) (a_i * x + b_i)^2 for all real x
-/
def satisfies_equation (coeffs : List Coefficient) : Prop :=
  ∀ (x : ℝ), x^2 + x + 4 = (coeffs.map (fun c => (c.a * x + c.b)^2)).sum

/-- The main theorem stating that 5 is the least possible value of n -/
theorem least_possible_n :
  (∃ (coeffs : List Coefficient), coeffs.length = 5 ∧ satisfies_equation coeffs) ∧
  (∀ (n : ℕ) (coeffs : List Coefficient), n < 5 → coeffs.length = n → ¬satisfies_equation coeffs) :=
sorry

end NUMINAMATH_CALUDE_least_possible_n_l4132_413231


namespace NUMINAMATH_CALUDE_area_ratio_is_one_l4132_413230

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the right triangle PQR with given side lengths
def rightTrianglePQR : Triangle :=
  { P := (0, 15),
    Q := (0, 0),
    R := (20, 0) }

-- Define midpoints S and T
def S : ℝ × ℝ := (0, 7.5)
def T : ℝ × ℝ := (12.5, 12.5)

-- Define point Y as the intersection of RT and QS
def Y : ℝ × ℝ := sorry

-- Define the areas of quadrilateral PSYT and triangle QYR
def areaPSYT : ℝ := sorry
def areaQYR : ℝ := sorry

-- Theorem statement
theorem area_ratio_is_one :
  areaPSYT = areaQYR :=
sorry

end NUMINAMATH_CALUDE_area_ratio_is_one_l4132_413230


namespace NUMINAMATH_CALUDE_solve_linear_equation_l4132_413235

theorem solve_linear_equation (x : ℚ) (h : 3 * x - 4 = -2 * x + 11) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l4132_413235


namespace NUMINAMATH_CALUDE_non_congruent_squares_count_l4132_413217

/-- Represents a square on a lattice grid -/
structure LatticeSquare where
  -- We'll represent a square by its side length and orientation
  side_length : ℕ
  is_rotated : Bool

/-- The size of the grid -/
def grid_size : ℕ := 6

/-- Counts the number of squares of a given side length on the grid -/
def count_squares (side_length : ℕ) : ℕ :=
  (grid_size - side_length) * (grid_size - side_length)

/-- Counts all non-congruent squares on the 6x6 grid -/
def count_all_squares : ℕ :=
  -- Count regular squares
  (count_squares 1) + (count_squares 2) + (count_squares 3) + 
  (count_squares 4) + (count_squares 5) +
  -- Count rotated squares (same formula as regular squares)
  (count_squares 1) + (count_squares 2) + (count_squares 3) + 
  (count_squares 4) + (count_squares 5)

/-- Theorem: The number of non-congruent squares on a 6x6 grid is 110 -/
theorem non_congruent_squares_count : count_all_squares = 110 := by
  sorry

end NUMINAMATH_CALUDE_non_congruent_squares_count_l4132_413217


namespace NUMINAMATH_CALUDE_segment_length_l4132_413207

/-- Given 5 points on a line, prove that PQ = 11 -/
theorem segment_length (P Q R S T : ℝ) : 
  P < Q ∧ Q < R ∧ R < S ∧ S < T →
  (Q - P) + (R - P) + (S - P) + (T - P) = 67 →
  (Q - P) + (R - Q) + (S - Q) + (T - Q) = 34 →
  Q - P = 11 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_l4132_413207


namespace NUMINAMATH_CALUDE_odd_function_a_value_l4132_413234

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_a_value
  (f : ℝ → ℝ)
  (h_odd : isOddFunction f)
  (h_neg : ∀ x, x < 0 → f x = x^2 + a*x)
  (h_f2 : f 2 = 6)
  (a : ℝ) :
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_odd_function_a_value_l4132_413234


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l4132_413252

theorem cubic_equation_solution :
  ∃! x : ℝ, (x^3 - x^2) / (x^2 + 3*x + 2) + x = -3 ∧ x ≠ -1 ∧ x ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l4132_413252


namespace NUMINAMATH_CALUDE_min_a_for_increasing_f_l4132_413246

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

-- State the theorem
theorem min_a_for_increasing_f :
  (∀ a : ℝ, ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x < y → f a x < f a y) →
  (∃ a_min : ℝ, a_min = -3 ∧ 
    (∀ a : ℝ, (∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x < y → f a x < f a y) → a ≥ a_min)) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_increasing_f_l4132_413246


namespace NUMINAMATH_CALUDE_range_of_a_l4132_413289

open Set

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc 0 a, f x ∈ Icc (-4) 0) ∧ 
  (Icc (-4) 0 ⊆ f '' Icc 0 a) ↔ 
  a ∈ Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4132_413289


namespace NUMINAMATH_CALUDE_correct_multiplication_l4132_413244

theorem correct_multiplication (x : ℕ) (h : 63 + x = 70) : 36 * x = 252 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplication_l4132_413244


namespace NUMINAMATH_CALUDE_unique_distance_l4132_413202

def is_valid_distance (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  (n / 10) = n - ((n % 10) * 10 + (n / 10))

theorem unique_distance : ∃! n : ℕ, is_valid_distance n ∧ n = 98 :=
sorry

end NUMINAMATH_CALUDE_unique_distance_l4132_413202


namespace NUMINAMATH_CALUDE_power_sixteen_div_sixteen_squared_l4132_413226

theorem power_sixteen_div_sixteen_squared : 2^16 / 16^2 = 256 := by
  sorry

end NUMINAMATH_CALUDE_power_sixteen_div_sixteen_squared_l4132_413226


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l4132_413224

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + k = 0) ↔ k ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l4132_413224


namespace NUMINAMATH_CALUDE_range_of_a_l4132_413269

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(p x) ∧ (q x a))) →
  (∀ a : ℝ, a ≥ 1) ∧ (∀ b : ℝ, b ≥ 1 → ∃ a : ℝ, a = b) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4132_413269


namespace NUMINAMATH_CALUDE_position_change_l4132_413290

/-- The position of a person from the back in a line of descending height order -/
def position_from_back_descending (total : ℕ) (position : ℕ) : Prop :=
  position > 0 ∧ position ≤ total

/-- The position of a person from the back in a line of ascending height order -/
def position_from_back_ascending (total : ℕ) (position : ℕ) : Prop :=
  position > 0 ∧ position ≤ total

/-- Theorem stating the relationship between a person's position in descending and ascending order lines -/
theorem position_change 
  (total : ℕ) 
  (position_desc : ℕ) 
  (position_asc : ℕ) 
  (h1 : total = 22)
  (h2 : position_desc = 13)
  (h3 : position_from_back_descending total position_desc)
  (h4 : position_from_back_ascending total position_asc)
  : position_asc = 10 := by
  sorry

#check position_change

end NUMINAMATH_CALUDE_position_change_l4132_413290


namespace NUMINAMATH_CALUDE_uranium_conductivity_is_deductive_reasoning_l4132_413223

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Metal : U → Prop)
variable (ConductsElectricity : U → Prop)

-- Define uranium as a constant in our universe
variable (uranium : U)

-- Define what deductive reasoning is
def is_deductive_reasoning (premise1 premise2 conclusion : Prop) : Prop :=
  (premise1 ∧ premise2) → conclusion

-- State the theorem
theorem uranium_conductivity_is_deductive_reasoning :
  is_deductive_reasoning
    (∀ x : U, Metal x → ConductsElectricity x)
    (Metal uranium)
    (ConductsElectricity uranium) :=
by
  sorry


end NUMINAMATH_CALUDE_uranium_conductivity_is_deductive_reasoning_l4132_413223


namespace NUMINAMATH_CALUDE_square_distance_equivalence_l4132_413299

theorem square_distance_equivalence :
  ∀ (s : Real), s = 1 →
  (5 : Real) / Real.sqrt 2 = (5 : Real) / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_square_distance_equivalence_l4132_413299


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l4132_413274

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define event A
def A : Set Ω := {ω | ω.1 = 0}

-- Define event B
def B : Set Ω := {ω | ω.1.val + ω.2.val + 2 = 6}

-- State the theorem
theorem conditional_probability_B_given_A :
  P B / P A = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l4132_413274


namespace NUMINAMATH_CALUDE_smallest_number_proof_l4132_413256

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  b = 28 →                 -- Median is 28
  c = b + 6 →              -- Largest number is 6 more than median
  a ≤ b ∧ b ≤ c →          -- b is the median
  a = 28 :=                -- Smallest number is 28
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l4132_413256


namespace NUMINAMATH_CALUDE_field_trip_adults_l4132_413258

theorem field_trip_adults (van_capacity : ℕ) (num_vans : ℕ) (num_students : ℕ) :
  van_capacity = 4 →
  num_vans = 2 →
  num_students = 2 →
  ∃ (num_adults : ℕ), num_adults + num_students = num_vans * van_capacity ∧ num_adults = 6 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_adults_l4132_413258


namespace NUMINAMATH_CALUDE_min_value_sum_fourth_and_square_l4132_413268

theorem min_value_sum_fourth_and_square (t : ℝ) :
  let f := fun (a : ℝ) => a^4 + (t - a)^2
  ∃ (min_val : ℝ), (∀ (a : ℝ), f a ≥ min_val) ∧ (min_val = t^4 / 16 + t^2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_fourth_and_square_l4132_413268


namespace NUMINAMATH_CALUDE_ceiling_squared_fraction_l4132_413242

theorem ceiling_squared_fraction : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_squared_fraction_l4132_413242


namespace NUMINAMATH_CALUDE_large_cube_volume_l4132_413203

theorem large_cube_volume (die_surface_area : ℝ) (h : die_surface_area = 96) :
  let die_face_area := die_surface_area / 6
  let large_cube_face_area := 4 * die_face_area
  let large_cube_side_length := Real.sqrt large_cube_face_area
  large_cube_side_length ^ 3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_volume_l4132_413203


namespace NUMINAMATH_CALUDE_cylinder_not_triangular_front_view_l4132_413219

-- Define a spatial geometric body
structure SpatialBody where
  name : String

-- Define the front view of a spatial body
inductive FrontView
  | Triangle
  | Rectangle
  | Other

-- Define a function that returns the front view of a spatial body
def frontViewOf (body : SpatialBody) : FrontView :=
  sorry

-- Define a cylinder
def cylinder : SpatialBody :=
  { name := "Cylinder" }

-- Theorem: A cylinder cannot have a triangular front view
theorem cylinder_not_triangular_front_view :
  frontViewOf cylinder ≠ FrontView.Triangle :=
sorry

end NUMINAMATH_CALUDE_cylinder_not_triangular_front_view_l4132_413219


namespace NUMINAMATH_CALUDE_two_non_congruent_triangles_l4132_413282

/-- A triangle with integer side lengths. -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The perimeter of a triangle. -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- Check if a triangle satisfies the triangle inequality. -/
def is_valid (t : IntTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Check if two triangles are congruent. -/
def is_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all valid triangles with perimeter 11. -/
def valid_triangles : Set IntTriangle :=
  {t : IntTriangle | perimeter t = 11 ∧ is_valid t}

/-- The theorem to be proved. -/
theorem two_non_congruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ valid_triangles ∧
    t2 ∈ valid_triangles ∧
    ¬is_congruent t1 t2 ∧
    ∀ (t : IntTriangle),
      t ∈ valid_triangles →
      is_congruent t t1 ∨ is_congruent t t2 :=
sorry

end NUMINAMATH_CALUDE_two_non_congruent_triangles_l4132_413282


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l4132_413215

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be converted -/
def original_number : ℕ := 2270000

/-- Converts a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation := sorry

theorem scientific_notation_correct :
  let sn := to_scientific_notation original_number
  sn.coefficient = 2.27 ∧ sn.exponent = 6 ∧ original_number = sn.coefficient * (10 ^ sn.exponent) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l4132_413215


namespace NUMINAMATH_CALUDE_expression_evaluation_l4132_413284

theorem expression_evaluation : 60 + 5 * 12 / (180 / 3) = 61 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4132_413284


namespace NUMINAMATH_CALUDE_gbp_share_change_l4132_413262

/-- The change in the share of British pounds in the National Wealth Fund -/
theorem gbp_share_change (
  total : ℝ)
  (initial_share : ℝ)
  (other_amounts : List ℝ)
  (h_total : total = 794.26)
  (h_initial : initial_share = 8.2)
  (h_other : other_amounts = [39.84, 34.72, 600.3, 110.54, 0.31]) :
  ∃ (δ : ℝ), abs (δ + 7) < 0.5 ∧ 
  δ = (total - (other_amounts.sum)) / total * 100 - initial_share :=
sorry

end NUMINAMATH_CALUDE_gbp_share_change_l4132_413262


namespace NUMINAMATH_CALUDE_polynomial_not_equal_33_l4132_413297

theorem polynomial_not_equal_33 (x y : ℤ) :
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_equal_33_l4132_413297


namespace NUMINAMATH_CALUDE_component_unqualified_l4132_413225

/-- A component is qualified if its diameter is within the specified range. -/
def IsQualified (measured : ℝ) (nominal : ℝ) (tolerance : ℝ) : Prop :=
  nominal - tolerance ≤ measured ∧ measured ≤ nominal + tolerance

/-- The component is unqualified if it's not qualified. -/
def IsUnqualified (measured : ℝ) (nominal : ℝ) (tolerance : ℝ) : Prop :=
  ¬(IsQualified measured nominal tolerance)

theorem component_unqualified (measured : ℝ) (h : measured = 19.9) :
  IsUnqualified measured 20 0.02 := by
  sorry

#check component_unqualified

end NUMINAMATH_CALUDE_component_unqualified_l4132_413225


namespace NUMINAMATH_CALUDE_positive_expression_l4132_413201

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 1 < z ∧ z < 2) : 
  y + z^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l4132_413201


namespace NUMINAMATH_CALUDE_number_difference_l4132_413267

theorem number_difference (L S : ℕ) (h1 : L = 1637) (h2 : L = 6 * S + 5) : L - S = 1365 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l4132_413267


namespace NUMINAMATH_CALUDE_triangle_median_similarity_exists_l4132_413220

/-- 
Given a triangle with sides a, b, c (where a < b < c), we define the following:
1) The triangle formed by the medians is similar to the original triangle.
2) The relationship between sides and medians is given by:
   4sa² = -a² + 2b² + 2c²
   4sb² = 2a² - b² + 2c²
   4sc² = 2a² + 2b² - c²
   where sa, sb, sc are the medians opposite to sides a, b, c respectively.
3) The sides satisfy the equation: b² = (a² + c²) / 2

This theorem states that there exists a triplet of natural numbers (a, b, c) 
that satisfies all these conditions, with a < b < c.
-/
theorem triangle_median_similarity_exists : 
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ 
  (b * b : ℚ) = (a * a + c * c) / 2 ∧
  (∃ (sa sb sc : ℚ), 
    4 * sa * sa = -a * a + 2 * b * b + 2 * c * c ∧
    4 * sb * sb = 2 * a * a - b * b + 2 * c * c ∧
    4 * sc * sc = 2 * a * a + 2 * b * b - c * c ∧
    (a : ℚ) / sc = (b : ℚ) / sb ∧ (b : ℚ) / sb = (c : ℚ) / sa) :=
by sorry

end NUMINAMATH_CALUDE_triangle_median_similarity_exists_l4132_413220


namespace NUMINAMATH_CALUDE_student_number_problem_l4132_413222

theorem student_number_problem (x : ℝ) : 5 * x - 138 = 102 → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l4132_413222


namespace NUMINAMATH_CALUDE_specific_tile_arrangement_l4132_413210

/-- The number of distinguishable arrangements for a row of tiles -/
def tileArrangements (brown purple green yellow blue : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow + blue) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green *
   Nat.factorial yellow * Nat.factorial blue)

/-- Theorem: The number of distinguishable arrangements for a row consisting of
    1 brown tile, 1 purple tile, 3 green tiles, 3 yellow tiles, and 2 blue tiles
    is equal to 50400. -/
theorem specific_tile_arrangement :
  tileArrangements 1 1 3 3 2 = 50400 := by
  sorry

end NUMINAMATH_CALUDE_specific_tile_arrangement_l4132_413210
