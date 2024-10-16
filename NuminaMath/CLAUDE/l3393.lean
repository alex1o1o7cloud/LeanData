import Mathlib

namespace NUMINAMATH_CALUDE_number_satisfying_condition_l3393_339382

theorem number_satisfying_condition : ∃ x : ℝ, 0.05 * x = 0.20 * 650 + 190 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l3393_339382


namespace NUMINAMATH_CALUDE_expression_simplification_l3393_339341

theorem expression_simplification (a : ℝ) 
  (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -3) :
  (a^2 - 9) / (a^2 + 6*a + 9) / ((a - 3) / (a^2 + 3*a)) - (a - a^2) / (a - 1) = 2*a :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3393_339341


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3393_339304

theorem quadratic_always_positive (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x + 1 > 0) → -2 < b ∧ b < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3393_339304


namespace NUMINAMATH_CALUDE_sqrt_3_squared_4_fourth_l3393_339372

theorem sqrt_3_squared_4_fourth : Real.sqrt (3^2 * 4^4) = 48 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_squared_4_fourth_l3393_339372


namespace NUMINAMATH_CALUDE_jane_max_tickets_l3393_339344

/-- Calculates the maximum number of tickets that can be bought given a budget and pricing structure -/
def maxTickets (budget : ℕ) (normalPrice discountPrice : ℕ) (discountThreshold : ℕ) : ℕ :=
  let fullPriceTickets := min discountThreshold (budget / normalPrice)
  let remainingBudget := budget - fullPriceTickets * normalPrice
  fullPriceTickets + remainingBudget / discountPrice

/-- The maximum number of tickets Jane can buy is 11 -/
theorem jane_max_tickets :
  maxTickets 150 15 12 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_jane_max_tickets_l3393_339344


namespace NUMINAMATH_CALUDE_inequality_proof_l3393_339336

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.log (1 + Real.exp 1))
  (hb : b = Real.sqrt (Real.exp 1))
  (hc : c = 2 * Real.exp 1 / 3) :
  c > b ∧ b > a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3393_339336


namespace NUMINAMATH_CALUDE_rain_hours_calculation_l3393_339346

theorem rain_hours_calculation (total_hours rain_hours : ℕ) 
  (h1 : total_hours = 9)
  (h2 : rain_hours = 4) : 
  total_hours - rain_hours = 5 := by
sorry

end NUMINAMATH_CALUDE_rain_hours_calculation_l3393_339346


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l3393_339326

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l3393_339326


namespace NUMINAMATH_CALUDE_double_side_halves_energy_l3393_339311

/-- Represents the energy stored between two point charges -/
structure EnergyBetweenCharges where
  distance : ℝ
  charge1 : ℝ
  charge2 : ℝ
  energy : ℝ

/-- Represents a configuration of three point charges in an equilateral triangle -/
structure TriangleConfiguration where
  sideLength : ℝ
  charge : ℝ
  totalEnergy : ℝ

/-- The relation between energy, distance, and charges -/
axiom energy_proportionality 
  (e1 e2 : EnergyBetweenCharges) : 
  e1.charge1 = e2.charge1 → e1.charge2 = e2.charge2 → 
  e1.energy * e1.distance = e2.energy * e2.distance

/-- The total energy in a triangle configuration is the sum of energies between pairs -/
axiom triangle_energy 
  (tc : TriangleConfiguration) (e : EnergyBetweenCharges) :
  e.distance = tc.sideLength → e.charge1 = tc.charge → e.charge2 = tc.charge →
  tc.totalEnergy = 3 * e.energy

/-- Theorem: Doubling the side length of the triangle halves the total energy -/
theorem double_side_halves_energy 
  (tc1 tc2 : TriangleConfiguration) :
  tc1.charge = tc2.charge →
  tc2.sideLength = 2 * tc1.sideLength →
  tc2.totalEnergy = tc1.totalEnergy / 2 := by
  sorry

end NUMINAMATH_CALUDE_double_side_halves_energy_l3393_339311


namespace NUMINAMATH_CALUDE_unique_promotion_solution_l3393_339392

/-- Represents the promotional offer for pencils -/
structure PencilPromotion where
  base : ℕ  -- The number of pencils Pete's mom gave money for
  bonus : ℕ -- The additional pencils Pete could buy due to the promotion

/-- Defines the specific promotion where Pete buys 12 more pencils -/
def specificPromotion : PencilPromotion := { base := 49, bonus := 12 }

/-- Theorem stating that the specific promotion is the only one satisfying the conditions -/
theorem unique_promotion_solution : 
  ∀ (p : PencilPromotion), p.bonus = 12 → p.base = 49 := by
  sorry

#check unique_promotion_solution

end NUMINAMATH_CALUDE_unique_promotion_solution_l3393_339392


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3393_339331

theorem max_value_trig_expression :
  ∀ x y z : ℝ, 
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) * 
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 9 / 2 ∧
  ∃ x y z : ℝ, 
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) * 
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3393_339331


namespace NUMINAMATH_CALUDE_expression_equality_l3393_339332

theorem expression_equality : 
  (5 + 8) * (5^2 + 8^2) * (5^4 + 8^4) * (5^8 + 8^8) * (5^16 + 8^16) * (5^32 + 8^32) = 8^32 - 5^32 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3393_339332


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l3393_339306

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l3393_339306


namespace NUMINAMATH_CALUDE_complement_N_subset_complement_M_l3393_339340

/-- The set of real numbers -/
def R : Set ℝ := Set.univ

/-- The set M defined as {x | 0 < x < 2} -/
def M : Set ℝ := {x | 0 < x ∧ x < 2}

/-- The set N defined as {x | x^2 + x - 6 ≤ 0} -/
def N : Set ℝ := {x | x^2 + x - 6 ≤ 0}

/-- Theorem stating that the complement of N is a subset of the complement of M -/
theorem complement_N_subset_complement_M : (R \ N) ⊆ (R \ M) := by
  sorry

end NUMINAMATH_CALUDE_complement_N_subset_complement_M_l3393_339340


namespace NUMINAMATH_CALUDE_fractions_not_both_integers_l3393_339390

theorem fractions_not_both_integers (n : ℤ) : 
  ¬(∃ (x y : ℤ), (n - 6 : ℤ) = 15 * x ∧ (n - 5 : ℤ) = 24 * y) := by
  sorry

end NUMINAMATH_CALUDE_fractions_not_both_integers_l3393_339390


namespace NUMINAMATH_CALUDE_angle_sum_B_plus_D_l3393_339325

-- Define the triangle AFG and external angle BFD
structure Triangle :=
  (A B D F G : Real)

-- State the theorem
theorem angle_sum_B_plus_D (t : Triangle) 
  (h1 : t.A = 30) -- Given: Angle A is 30 degrees
  (h2 : t.F = t.G) -- Given: Angle AFG equals Angle AGF
  : t.B + t.D = 75 := by
  sorry


end NUMINAMATH_CALUDE_angle_sum_B_plus_D_l3393_339325


namespace NUMINAMATH_CALUDE_cos_sin_75_deg_l3393_339355

theorem cos_sin_75_deg : Real.cos (75 * π / 180) ^ 4 - Real.sin (75 * π / 180) ^ 4 = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_75_deg_l3393_339355


namespace NUMINAMATH_CALUDE_malar_completion_time_l3393_339359

/-- The number of days Malar takes to complete the task alone -/
def M : ℝ := 60

/-- The number of days Roja takes to complete the task alone -/
def R : ℝ := 84

/-- The number of days Malar and Roja take to complete the task together -/
def T : ℝ := 35

theorem malar_completion_time :
  (1 / M + 1 / R = 1 / T) → M = 60 := by
  sorry

end NUMINAMATH_CALUDE_malar_completion_time_l3393_339359


namespace NUMINAMATH_CALUDE_biathlon_distance_l3393_339385

/-- Biathlon problem -/
theorem biathlon_distance (total_distance : ℝ) (run_velocity : ℝ) (bike_velocity : ℝ) (total_time : ℝ)
  (h1 : total_distance = 155)
  (h2 : run_velocity = 10)
  (h3 : bike_velocity = 29)
  (h4 : total_time = 6) :
  ∃ (bike_distance : ℝ), 
    bike_distance + (total_distance - bike_distance) = total_distance ∧
    bike_distance / bike_velocity + (total_distance - bike_distance) / run_velocity = total_time ∧
    bike_distance = 145 := by
  sorry

end NUMINAMATH_CALUDE_biathlon_distance_l3393_339385


namespace NUMINAMATH_CALUDE_sector_central_angle_l3393_339323

/-- Theorem: Given a circular sector with arc length 3 and radius 2, the central angle is 3/2 radians. -/
theorem sector_central_angle (l : ℝ) (r : ℝ) (θ : ℝ) 
  (hl : l = 3) (hr : r = 2) (hθ : l = r * θ) : θ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3393_339323


namespace NUMINAMATH_CALUDE_xiaotong_message_forwarding_l3393_339389

theorem xiaotong_message_forwarding :
  ∃ (x : ℕ), x > 0 ∧ 1 + x + x^2 = 91 :=
by sorry

end NUMINAMATH_CALUDE_xiaotong_message_forwarding_l3393_339389


namespace NUMINAMATH_CALUDE_tourist_count_scientific_notation_l3393_339391

theorem tourist_count_scientific_notation :
  ∀ (n : ℝ), n = 15.276 * 1000000 → 
  ∃ (a : ℝ) (b : ℤ), n = a * (10 : ℝ) ^ b ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.5276 ∧ b = 7 :=
by sorry

end NUMINAMATH_CALUDE_tourist_count_scientific_notation_l3393_339391


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3393_339357

/-- Given a point P (x_P, y_P) on the curve 2x^2 - y = 0 and a fixed point A (0, -1),
    the midpoint M (x, y) of AP satisfies the equation 8x^2 - 2y - 1 = 0 -/
theorem midpoint_trajectory (x_P y_P x y : ℝ) : 
  (2 * x_P^2 = y_P) →  -- P is on the curve 2x^2 - y = 0
  (x = x_P / 2) →      -- x-coordinate of midpoint
  (y = (y_P - 1) / 2)  -- y-coordinate of midpoint
  → 8 * x^2 - 2 * y - 1 = 0 := by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3393_339357


namespace NUMINAMATH_CALUDE_total_books_l3393_339378

/-- The total number of books Sandy, Benny, and Tim have together is 67. -/
theorem total_books (sandy_books benny_books tim_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : benny_books = 24)
  (h3 : tim_books = 33) :
  sandy_books + benny_books + tim_books = 67 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l3393_339378


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3393_339321

theorem sphere_surface_area (V : ℝ) (r : ℝ) (S : ℝ) :
  V = 48 * Real.pi →
  V = (4 / 3) * Real.pi * r^3 →
  S = 4 * Real.pi * r^2 →
  S = 144 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3393_339321


namespace NUMINAMATH_CALUDE_total_weight_lifted_l3393_339338

def weight_per_hand : ℕ := 8
def number_of_hands : ℕ := 2

theorem total_weight_lifted : weight_per_hand * number_of_hands = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_lifted_l3393_339338


namespace NUMINAMATH_CALUDE_bridge_anchor_ratio_l3393_339301

/-- Proves that the ratio of concrete needed for each bridge anchor is 1:1 --/
theorem bridge_anchor_ratio
  (total_concrete : ℕ)
  (roadway_concrete : ℕ)
  (one_anchor_concrete : ℕ)
  (pillar_concrete : ℕ)
  (h1 : total_concrete = 4800)
  (h2 : roadway_concrete = 1600)
  (h3 : one_anchor_concrete = 700)
  (h4 : pillar_concrete = 1800)
  : (one_anchor_concrete : ℚ) / one_anchor_concrete = 1 := by
  sorry

#check bridge_anchor_ratio

end NUMINAMATH_CALUDE_bridge_anchor_ratio_l3393_339301


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3393_339376

theorem units_digit_of_product (n : ℕ) : n = 10 * 11 * 12 * 13 * 14 * 15 * 16 / 800 → n % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3393_339376


namespace NUMINAMATH_CALUDE_cos_roots_of_quadratic_l3393_339375

theorem cos_roots_of_quadratic (α β : ℂ) : 
  (2 * α^2 + 3 * α + 5 = 0) → 
  (2 * β^2 + 3 * β + 5 = 0) → 
  (Complex.cos α)^2 + (Complex.cos α) + 1 = 0 ∧ 
  (Complex.cos β)^2 + (Complex.cos β) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_roots_of_quadratic_l3393_339375


namespace NUMINAMATH_CALUDE_function_properties_l3393_339337

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) : ℝ := x + a * x^2 + b * Real.log x

theorem function_properties :
  (f a b 1 = 0 ∧ (deriv (f a b)) 1 = 2) →
  (a = -1 ∧ b = 3 ∧ ∀ x > 0, f a b x ≤ 2 * x - 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3393_339337


namespace NUMINAMATH_CALUDE_quadratic_b_value_l3393_339312

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + c

-- Define the theorem
theorem quadratic_b_value :
  ∀ (b c y₁ y₂ : ℝ),
  (f b c 2 = y₁) →
  (f b c (-2) = y₂) →
  (y₁ - y₂ = 12) →
  b = 3 := by
sorry


end NUMINAMATH_CALUDE_quadratic_b_value_l3393_339312


namespace NUMINAMATH_CALUDE_daisy_count_divisible_by_four_l3393_339374

theorem daisy_count_divisible_by_four (D : ℕ) : 
  (∃ k : ℕ, 8 + D + 48 = 4 * k) → 
  (∃ m : ℕ, D = 4 * m) := by
sorry

end NUMINAMATH_CALUDE_daisy_count_divisible_by_four_l3393_339374


namespace NUMINAMATH_CALUDE_cos_2013pi_l3393_339314

theorem cos_2013pi : Real.cos (2013 * Real.pi) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_2013pi_l3393_339314


namespace NUMINAMATH_CALUDE_hiking_speeds_l3393_339387

-- Define the hiking speeds and relationships
def lucas_speed : ℚ := 5
def mia_speed_ratio : ℚ := 3/4
def grace_speed_ratio : ℚ := 6/7
def liam_speed_ratio : ℚ := 4/3

-- Define the hiking speeds of Mia, Grace, and Liam
def mia_speed : ℚ := lucas_speed * mia_speed_ratio
def grace_speed : ℚ := mia_speed * grace_speed_ratio
def liam_speed : ℚ := grace_speed * liam_speed_ratio

-- Theorem to prove Grace's and Liam's hiking speeds
theorem hiking_speeds :
  grace_speed = 45/14 ∧ liam_speed = 30/7 := by
  sorry

end NUMINAMATH_CALUDE_hiking_speeds_l3393_339387


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_l3393_339302

/-- The set A is defined as a circle centered at (2, 1) with radius 1 -/
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 ≤ 1}

/-- The set B is defined as a diamond shape centered at (1, 1) -/
def B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2*|p.1 - 1| + |p.2 - 1| ≤ a}

/-- Theorem stating the range of a for which A is a subset of B -/
theorem range_of_a_for_subset (a : ℝ) : A ⊆ B a ↔ a ≥ 2 + Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_l3393_339302


namespace NUMINAMATH_CALUDE_bag_draw_comparison_l3393_339351

/-- A bag containing red and black balls -/
structure Bag where
  red : ℕ
  black : ℕ

/-- Random variable for drawing with replacement -/
def xi₁ (b : Bag) : ℕ → ℝ := sorry

/-- Random variable for drawing without replacement -/
def xi₂ (b : Bag) : ℕ → ℝ := sorry

/-- Expected value of a random variable -/
def expectation (X : ℕ → ℝ) : ℝ := sorry

/-- Variance of a random variable -/
def variance (X : ℕ → ℝ) : ℝ := sorry

/-- Theorem about expected values and variances of xi₁ and xi₂ -/
theorem bag_draw_comparison (b : Bag) (h : b.red = 1 ∧ b.black = 2) : 
  expectation (xi₁ b) = expectation (xi₂ b) ∧ 
  variance (xi₁ b) > variance (xi₂ b) := by sorry

end NUMINAMATH_CALUDE_bag_draw_comparison_l3393_339351


namespace NUMINAMATH_CALUDE_family_weight_problem_l3393_339316

theorem family_weight_problem (total_weight daughter_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : daughter_weight = 42) :
  ∃ (grandmother_weight child_weight : ℝ),
    grandmother_weight + daughter_weight + child_weight = total_weight ∧
    child_weight = (1 / 5) * grandmother_weight ∧
    daughter_weight + child_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_family_weight_problem_l3393_339316


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l3393_339319

def quadratic_equation (b c : ℝ) := fun x : ℝ => x^2 + b*x + c

def roots (f : ℝ → ℝ) (r₁ r₂ : ℝ) : Prop :=
  f r₁ = 0 ∧ f r₂ = 0

theorem correct_quadratic_equation :
  ∃ (b₁ c₁ b₂ c₂ : ℝ),
    roots (quadratic_equation b₁ c₁) 5 3 ∧
    roots (quadratic_equation b₂ c₂) (-7) (-2) ∧
    b₁ = -8 ∧
    c₂ = 14 →
    quadratic_equation (-8) 14 = quadratic_equation b₁ c₂ :=
sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l3393_339319


namespace NUMINAMATH_CALUDE_polynomial_property_implies_P0_values_l3393_339322

/-- A polynomial P with real coefficients satisfying the given property -/
def SatisfiesProperty (P : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (|y^2 - P x| ≤ 2 * |x|) ↔ (|x^2 - P y| ≤ 2 * |y|)

/-- The theorem stating the possible values of P(0) -/
theorem polynomial_property_implies_P0_values (P : ℝ → ℝ) (h : SatisfiesProperty P) :
  P 0 < 0 ∨ P 0 = 1 :=
sorry

end NUMINAMATH_CALUDE_polynomial_property_implies_P0_values_l3393_339322


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l3393_339356

theorem abs_sum_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ (-6.5 < x ∧ x < 3.5) :=
sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l3393_339356


namespace NUMINAMATH_CALUDE_journalism_club_arrangement_l3393_339343

/-- The number of students in the arrangement -/
def num_students : ℕ := 5

/-- The number of teachers in the arrangement -/
def num_teachers : ℕ := 2

/-- The number of possible positions for the teacher pair -/
def teacher_pair_positions : ℕ := num_students - 1

/-- The total number of arrangements -/
def total_arrangements : ℕ := num_students.factorial * (teacher_pair_positions * num_teachers.factorial)

/-- Theorem stating that the total number of arrangements is 960 -/
theorem journalism_club_arrangement :
  total_arrangements = 960 := by sorry

end NUMINAMATH_CALUDE_journalism_club_arrangement_l3393_339343


namespace NUMINAMATH_CALUDE_detergent_volume_in_new_solution_l3393_339307

/-- Represents the components of a cleaning solution -/
inductive Component
| Bleach
| Detergent
| Water

/-- Represents the ratio of components in a solution -/
def Ratio := Component → ℚ

def original_ratio : Ratio :=
  fun c => match c with
  | Component.Bleach => 4
  | Component.Detergent => 40
  | Component.Water => 100

def new_ratio : Ratio :=
  fun c => match c with
  | Component.Bleach => 3 * (original_ratio Component.Bleach)
  | Component.Detergent => (1/2) * (original_ratio Component.Detergent)
  | Component.Water => original_ratio Component.Water

def water_volume : ℚ := 300

theorem detergent_volume_in_new_solution :
  (new_ratio Component.Detergent / new_ratio Component.Water) * water_volume = 60 := by
  sorry

end NUMINAMATH_CALUDE_detergent_volume_in_new_solution_l3393_339307


namespace NUMINAMATH_CALUDE_chatterbox_jokes_l3393_339300

def n : ℕ := 10  -- number of chatterboxes

-- Sum of natural numbers from 1 to m
def sum_to(m : ℕ) : ℕ := m * (m + 1) / 2

-- Total number of jokes told
def total_jokes : ℕ := sum_to 100 + sum_to 99

theorem chatterbox_jokes :
  total_jokes / n = 1000 :=
sorry

end NUMINAMATH_CALUDE_chatterbox_jokes_l3393_339300


namespace NUMINAMATH_CALUDE_no_three_digit_special_couples_l3393_339328

/-- Definition of a special couple for three-digit numbers -/
def is_special_couple (abc cba : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧  -- Digits are single-digit natural numbers
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧     -- Digits are distinct
    abc = 100 * a + 10 * b + c ∧
    cba = 100 * c + 10 * b + a ∧
    a + b + c = 9               -- Sum of digits is 9

/-- Theorem: There are no special couples with three-digit numbers -/
theorem no_three_digit_special_couples :
  ¬ ∃ (abc cba : ℕ), is_special_couple abc cba :=
sorry

end NUMINAMATH_CALUDE_no_three_digit_special_couples_l3393_339328


namespace NUMINAMATH_CALUDE_no_triangle_satisfies_equation_l3393_339349

-- Define a structure for a triangle
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z
  triangle_ineq1 : x + y > z
  triangle_ineq2 : y + z > x
  triangle_ineq3 : z + x > y

-- Theorem statement
theorem no_triangle_satisfies_equation :
  ¬∃ t : Triangle, t.x^3 + t.y^3 + t.z^3 = (t.x + t.y) * (t.y + t.z) * (t.z + t.x) :=
sorry

end NUMINAMATH_CALUDE_no_triangle_satisfies_equation_l3393_339349


namespace NUMINAMATH_CALUDE_cross_product_result_l3393_339327

def vector1 : ℝ × ℝ × ℝ := (3, -4, 7)
def vector2 : ℝ × ℝ × ℝ := (2, 5, -1)

def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a1, a2, a3) := v1
  let (b1, b2, b3) := v2
  (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)

theorem cross_product_result :
  cross_product vector1 vector2 = (-31, 17, 23) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_result_l3393_339327


namespace NUMINAMATH_CALUDE_mollys_current_age_l3393_339362

/-- Represents the ages of Sandy and Molly -/
structure Ages where
  sandy : ℕ
  molly : ℕ

/-- The ratio of Sandy's age to Molly's age is 4:3 -/
def age_ratio (ages : Ages) : Prop :=
  4 * ages.molly = 3 * ages.sandy

/-- Sandy will be 42 years old after 6 years -/
def sandy_future_age (ages : Ages) : Prop :=
  ages.sandy + 6 = 42

theorem mollys_current_age (ages : Ages) 
  (h1 : age_ratio ages) 
  (h2 : sandy_future_age ages) : 
  ages.molly = 27 := by
  sorry

end NUMINAMATH_CALUDE_mollys_current_age_l3393_339362


namespace NUMINAMATH_CALUDE_ab_value_l3393_339315

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the main theorem
theorem ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ (w x y z : ℕ), w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
    (w : ℝ) = (log10 a) ^ (1/3 : ℝ) ∧
    (x : ℝ) = (log10 b) ^ (1/3 : ℝ) ∧
    (y : ℝ) = log10 (a ^ (1/3 : ℝ)) ∧
    (z : ℝ) = log10 (b ^ (1/3 : ℝ)) ∧
    w + x + y + z = 12) →
  a * b = 10^9 :=
by sorry

end NUMINAMATH_CALUDE_ab_value_l3393_339315


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3393_339366

def S : Finset Int := {-7, -5, -3, -2, 2, 4, 6, 13}

theorem min_sum_of_squares (a b c d e f g h : Int) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
                c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
                d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
                e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
                f ≠ g ∧ f ≠ h ∧
                g ≠ h)
  (h_in_S : a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 34 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : Int), 
    (a' ∈ S ∧ b' ∈ S ∧ c' ∈ S ∧ d' ∈ S ∧ e' ∈ S ∧ f' ∈ S ∧ g' ∈ S ∧ h' ∈ S) ∧
    (a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ a' ≠ e' ∧ a' ≠ f' ∧ a' ≠ g' ∧ a' ≠ h' ∧
     b' ≠ c' ∧ b' ≠ d' ∧ b' ≠ e' ∧ b' ≠ f' ∧ b' ≠ g' ∧ b' ≠ h' ∧
     c' ≠ d' ∧ c' ≠ e' ∧ c' ≠ f' ∧ c' ≠ g' ∧ c' ≠ h' ∧
     d' ≠ e' ∧ d' ≠ f' ∧ d' ≠ g' ∧ d' ≠ h' ∧
     e' ≠ f' ∧ e' ≠ g' ∧ e' ≠ h' ∧
     f' ≠ g' ∧ f' ≠ h' ∧
     g' ≠ h') ∧
    (a' + b' + c' + d')^2 + (e' + f' + g' + h')^2 = 34 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3393_339366


namespace NUMINAMATH_CALUDE_always_winnable_l3393_339305

/-- Represents a move in the card game -/
def move (deck : List ℕ) : List ℕ :=
  match deck with
  | [] => []
  | x :: xs => (xs.take x).reverse ++ [x] ++ xs.drop x

/-- Predicate to check if 1 is at the top of the deck -/
def hasOneOnTop (deck : List ℕ) : Prop :=
  match deck with
  | 1 :: _ => True
  | _ => False

/-- Theorem stating that the game is always winnable -/
theorem always_winnable (n : ℕ) (deck : List ℕ) :
  (deck.length = n) →
  (∀ i, i ∈ deck ↔ 1 ≤ i ∧ i ≤ n) →
  ∃ k, hasOneOnTop ((move^[k]) deck) :=
sorry


end NUMINAMATH_CALUDE_always_winnable_l3393_339305


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3393_339397

/-- Given a quadratic equation x² + px + q = 0 with roots x₁ and x₂,
    this theorem proves the form of the quadratic equation whose roots are
    y₁ = (x₁ + x₁²) / (1 - x₂) and y₂ = (x₂ + x₂²) / (1 - x₁) -/
theorem quadratic_transformation (p q : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 + p*x₁ + q = 0 →
  x₂^2 + p*x₂ + q = 0 →
  x₁ ≠ x₂ →
  x₁ ≠ 1 →
  x₂ ≠ 1 →
  let y₁ := (x₁ + x₁^2) / (1 - x₂)
  let y₂ := (x₂ + x₂^2) / (1 - x₁)
  ∃ (y : ℝ), y^2 + (p*(1 + 3*q - p^2) / (1 + p + q))*y + (q*(1 - p + q) / (1 + p + q)) = 0 ↔
             (y = y₁ ∨ y = y₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3393_339397


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3393_339334

theorem trigonometric_identities (φ : Real) 
  (h1 : φ ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.tan (φ + Real.pi / 4) = -1 / 3) : 
  Real.tan (2 * φ) = 4 / 3 ∧ 
  (Real.sin φ + Real.cos φ) / (2 * Real.cos φ - Real.sin φ) = -1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3393_339334


namespace NUMINAMATH_CALUDE_claire_pets_l3393_339361

theorem claire_pets (total_pets : ℕ) (total_males : ℕ) : 
  total_pets = 92 →
  total_males = 25 →
  ∃ (gerbils hamsters : ℕ),
    gerbils + hamsters = total_pets ∧
    (gerbils / 4 : ℚ) + (hamsters / 3 : ℚ) = total_males ∧
    gerbils = 68 := by
  sorry

end NUMINAMATH_CALUDE_claire_pets_l3393_339361


namespace NUMINAMATH_CALUDE_max_total_profit_l3393_339383

/-- The fixed cost in million yuan -/
def fixed_cost : ℝ := 20

/-- The variable cost per unit in million yuan -/
def variable_cost_per_unit : ℝ := 10

/-- The total revenue function k(Q) in million yuan -/
def total_revenue (Q : ℝ) : ℝ := 40 * Q - Q^2

/-- The total cost function C(Q) in million yuan -/
def total_cost (Q : ℝ) : ℝ := fixed_cost + variable_cost_per_unit * Q

/-- The total profit function L(Q) in million yuan -/
def total_profit (Q : ℝ) : ℝ := total_revenue Q - total_cost Q

/-- The theorem stating that the maximum total profit is 205 million yuan -/
theorem max_total_profit : ∃ Q : ℝ, ∀ x : ℝ, total_profit Q ≥ total_profit x ∧ total_profit Q = 205 :=
sorry

end NUMINAMATH_CALUDE_max_total_profit_l3393_339383


namespace NUMINAMATH_CALUDE_statement3_is_analogous_reasoning_l3393_339353

-- Define the concept of a geometric figure
structure GeometricFigure where
  name : String

-- Define the concept of a property for geometric figures
structure Property where
  description : String

-- Define the concept of reasoning
inductive Reasoning
| Analogous
| Inductive
| Deductive

-- Define the statement about equilateral triangles
def equilateralTriangleProperty : Property :=
  { description := "The sum of distances from a point inside to its sides is constant" }

-- Define the statement about regular tetrahedrons
def regularTetrahedronProperty : Property :=
  { description := "The sum of distances from a point inside to its faces is constant" }

-- Define the reasoning process in statement ③
def statement3 (equilateralTriangle regularTetrahedron : GeometricFigure)
               (equilateralProp tetrahedronProp : Property) : Prop :=
  (equilateralProp = equilateralTriangleProperty) →
  (tetrahedronProp = regularTetrahedronProperty) →
  ∃ (r : Reasoning), r = Reasoning.Analogous

-- Theorem statement
theorem statement3_is_analogous_reasoning 
  (equilateralTriangle regularTetrahedron : GeometricFigure)
  (equilateralProp tetrahedronProp : Property) :
  statement3 equilateralTriangle regularTetrahedron equilateralProp tetrahedronProp :=
by
  sorry

end NUMINAMATH_CALUDE_statement3_is_analogous_reasoning_l3393_339353


namespace NUMINAMATH_CALUDE_lcm_of_15_25_35_l3393_339333

theorem lcm_of_15_25_35 : Nat.lcm 15 (Nat.lcm 25 35) = 525 := by sorry

end NUMINAMATH_CALUDE_lcm_of_15_25_35_l3393_339333


namespace NUMINAMATH_CALUDE_smallest_number_divisible_when_increased_l3393_339354

def is_divisible_by_all (n : ℕ) (divisors : List ℕ) : Prop :=
  ∀ d ∈ divisors, (n % d = 0)

theorem smallest_number_divisible_when_increased : ∃! n : ℕ, 
  (is_divisible_by_all (n + 9) [8, 11, 24]) ∧ 
  (∀ m : ℕ, m < n → ¬ is_divisible_by_all (m + 9) [8, 11, 24]) ∧
  n = 255 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_when_increased_l3393_339354


namespace NUMINAMATH_CALUDE_function_equality_l3393_339365

theorem function_equality (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = x + f (f y)) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_equality_l3393_339365


namespace NUMINAMATH_CALUDE_prime_sum_gcd_ratio_l3393_339377

theorem prime_sum_gcd_ratio (n : ℕ) (p : ℕ) (hp : Prime p) (h_p : p = 2 * n - 1) 
  (a : Fin n → ℕ+) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ i j : Fin n, i ≠ j ∧ (a i + a j : ℕ) / Nat.gcd (a i) (a j) ≥ p := by
sorry

end NUMINAMATH_CALUDE_prime_sum_gcd_ratio_l3393_339377


namespace NUMINAMATH_CALUDE_percentage_difference_l3393_339367

theorem percentage_difference (A B : ℝ) (h1 : A > 0) (h2 : B > A) :
  let x := 100 * (B - A) / A
  B = A * (1 + x / 100) := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l3393_339367


namespace NUMINAMATH_CALUDE_otimes_self_otimes_self_l3393_339324

/-- Definition of the ⊗ operation -/
def otimes (x y : ℝ) : ℝ := x^3 + x^2 - y

/-- Theorem: For any real number a, a ⊗ (a ⊗ a) = a -/
theorem otimes_self_otimes_self (a : ℝ) : otimes a (otimes a a) = a := by
  sorry

end NUMINAMATH_CALUDE_otimes_self_otimes_self_l3393_339324


namespace NUMINAMATH_CALUDE_solve_for_h_l3393_339317

/-- The y-intercept of the first equation -/
def y_intercept1 : ℝ := 2025

/-- The y-intercept of the second equation -/
def y_intercept2 : ℝ := 2026

/-- The first equation -/
def equation1 (h j x y : ℝ) : Prop := y = 4 * (x - h)^2 + j

/-- The second equation -/
def equation2 (h k x y : ℝ) : Prop := y = x^3 - 3 * (x - h)^2 + k

/-- Positive integer x-intercepts for the first equation -/
def positive_integer_roots1 (h j : ℝ) : Prop :=
  ∃ (x1 x2 : ℕ), x1 ≠ 0 ∧ x2 ≠ 0 ∧ equation1 h j x1 0 ∧ equation1 h j x2 0

/-- Positive integer x-intercepts for the second equation -/
def positive_integer_roots2 (h k : ℝ) : Prop :=
  ∃ (x1 x2 : ℕ), x1 ≠ 0 ∧ x2 ≠ 0 ∧ equation2 h k x1 0 ∧ equation2 h k x2 0

/-- The main theorem -/
theorem solve_for_h :
  ∃ (h j k : ℝ),
    equation1 h j 0 y_intercept1 ∧
    equation2 h k 0 y_intercept2 ∧
    positive_integer_roots1 h j ∧
    positive_integer_roots2 h k ∧
    h = 45 := by sorry

end NUMINAMATH_CALUDE_solve_for_h_l3393_339317


namespace NUMINAMATH_CALUDE_binomial_sum_problem_l3393_339310

theorem binomial_sum_problem (a : ℚ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℚ) :
  (∀ x, (a*x + 1)^5 * (x + 2)^4 = a₀*(x + 2)^9 + a₁*(x + 2)^8 + a₂*(x + 2)^7 + 
                                   a₃*(x + 2)^6 + a₄*(x + 2)^5 + a₅*(x + 2)^4 + 
                                   a₆*(x + 2)^3 + a₇*(x + 2)^2 + a₈*(x + 2) + a₉) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 1024 →
  a₀ + a₂ + a₄ + a₆ + a₈ = (2^10 - 14^5) / 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_problem_l3393_339310


namespace NUMINAMATH_CALUDE_n_to_b_equals_eight_l3393_339345

theorem n_to_b_equals_eight :
  let n : ℝ := 2 ^ (1/4)
  let b : ℝ := 12.000000000000002
  n ^ b = 8 := by
sorry

end NUMINAMATH_CALUDE_n_to_b_equals_eight_l3393_339345


namespace NUMINAMATH_CALUDE_triangle_sides_relation_l3393_339369

theorem triangle_sides_relation (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : Real.cos (2 * Real.pi / 3) = -1/2) :
  a^2 + a*c + c^2 - b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_relation_l3393_339369


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l3393_339370

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem derivative_f_at_one : 
  deriv f 1 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l3393_339370


namespace NUMINAMATH_CALUDE_find_number_l3393_339335

theorem find_number : ∃ x : ℝ, 2.12 + 0.345 + x = 2.4690000000000003 ∧ x = 0.0040000000000003 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3393_339335


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3393_339379

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 3, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3393_339379


namespace NUMINAMATH_CALUDE_third_discount_percentage_l3393_339373

/-- Given a car with an initial price and three successive discounts, 
    calculate the third discount percentage. -/
theorem third_discount_percentage 
  (initial_price : ℝ) 
  (first_discount second_discount : ℝ)
  (final_price : ℝ) :
  initial_price = 12000 →
  first_discount = 0.20 →
  second_discount = 0.15 →
  final_price = 7752 →
  ∃ (third_discount : ℝ),
    final_price = initial_price * 
      (1 - first_discount) * 
      (1 - second_discount) * 
      (1 - third_discount) ∧
    third_discount = 0.05 := by
  sorry


end NUMINAMATH_CALUDE_third_discount_percentage_l3393_339373


namespace NUMINAMATH_CALUDE_power_of_seven_roots_l3393_339320

theorem power_of_seven_roots (x : ℝ) (h : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_roots_l3393_339320


namespace NUMINAMATH_CALUDE_marcus_has_more_cards_l3393_339329

/-- The number of baseball cards Marcus has -/
def marcus_cards : ℕ := 210

/-- The number of baseball cards Carter has -/
def carter_cards : ℕ := 152

/-- The difference in baseball cards between Marcus and Carter -/
def card_difference : ℕ := marcus_cards - carter_cards

theorem marcus_has_more_cards : card_difference = 58 := by
  sorry

end NUMINAMATH_CALUDE_marcus_has_more_cards_l3393_339329


namespace NUMINAMATH_CALUDE_halfway_fraction_l3393_339342

theorem halfway_fraction (a b c d : ℤ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 7) :
  (a / b + c / d) / 2 = 41 / 56 :=
sorry

end NUMINAMATH_CALUDE_halfway_fraction_l3393_339342


namespace NUMINAMATH_CALUDE_divisibility_by_three_l3393_339380

theorem divisibility_by_three (u v : ℤ) : 
  (9 ∣ u^2 + u*v + v^2) → (3 ∣ u) ∧ (3 ∣ v) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l3393_339380


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_35_l3393_339381

def complement (α : ℝ) : ℝ := 90 - α

def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35 : 
  supplement (complement 35) = 125 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_35_l3393_339381


namespace NUMINAMATH_CALUDE_complex_cubic_sum_ratio_l3393_339388

theorem complex_cubic_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_eq : 2 * ((x - y)^2 + (x - z)^2 + (y - z)^2) = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_cubic_sum_ratio_l3393_339388


namespace NUMINAMATH_CALUDE_fish_population_estimate_l3393_339363

/-- Approximate number of fish in a pond given tagging and recapture data -/
theorem fish_population_estimate (tagged_initial : ℕ) (second_catch : ℕ) (tagged_second : ℕ) :
  tagged_initial = 70 →
  second_catch = 50 →
  tagged_second = 2 →
  (tagged_second : ℚ) / second_catch = tagged_initial / (tagged_initial + 1680) :=
by
  sorry

#check fish_population_estimate

end NUMINAMATH_CALUDE_fish_population_estimate_l3393_339363


namespace NUMINAMATH_CALUDE_ice_cream_flavor_ratio_l3393_339395

def total_flavors : ℕ := 100
def flavors_two_years_ago : ℕ := total_flavors / 4
def flavors_remaining : ℕ := 25
def flavors_tried_total : ℕ := total_flavors - flavors_remaining
def flavors_last_year : ℕ := flavors_tried_total - flavors_two_years_ago

theorem ice_cream_flavor_ratio :
  (flavors_last_year : ℚ) / flavors_two_years_ago = 2 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavor_ratio_l3393_339395


namespace NUMINAMATH_CALUDE_tricycle_count_l3393_339386

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) :
  total_children = 7 →
  total_wheels = 19 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  ∃ (bicycles tricycles : ℕ),
    bicycles + tricycles = total_children ∧
    bicycles * bicycle_wheels + tricycles * tricycle_wheels = total_wheels ∧
    tricycles = 5 :=
by sorry

end NUMINAMATH_CALUDE_tricycle_count_l3393_339386


namespace NUMINAMATH_CALUDE_negation_of_positive_quadratic_inequality_l3393_339368

theorem negation_of_positive_quadratic_inequality :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_quadratic_inequality_l3393_339368


namespace NUMINAMATH_CALUDE_toast_costs_one_pound_l3393_339358

/-- The cost of a slice of toast -/
def toast_cost : ℝ := sorry

/-- The cost of an egg -/
def egg_cost : ℝ := 3

/-- Dale's breakfast cost -/
def dale_breakfast : ℝ := 2 * toast_cost + 2 * egg_cost

/-- Andrew's breakfast cost -/
def andrew_breakfast : ℝ := toast_cost + 2 * egg_cost

/-- The total cost of both breakfasts -/
def total_cost : ℝ := 15

theorem toast_costs_one_pound :
  dale_breakfast + andrew_breakfast = total_cost →
  toast_cost = 1 := by sorry

end NUMINAMATH_CALUDE_toast_costs_one_pound_l3393_339358


namespace NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l3393_339352

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |x - 2| - m

-- Part I
theorem solution_set_part_i :
  ∀ x : ℝ, f x 5 > 0 ↔ x > 3 ∨ x < -2 :=
sorry

-- Part II
theorem solution_set_part_ii :
  ∀ m : ℝ, (∀ x : ℝ, f x m ≥ 2) ↔ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l3393_339352


namespace NUMINAMATH_CALUDE_function_behavior_l3393_339394

def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_prop : ∀ x, f x = -f (2 - x))
  (h_decr : is_decreasing_on f 1 2) :
  is_increasing_on f (-2) (-1) ∧ is_increasing_on f 3 4 := by
  sorry

end NUMINAMATH_CALUDE_function_behavior_l3393_339394


namespace NUMINAMATH_CALUDE_johnny_guitar_picks_l3393_339339

theorem johnny_guitar_picks (total red blue yellow : ℕ) : 
  total > 0 → 
  2 * red = total → 
  3 * blue = total → 
  yellow = total - red - blue → 
  blue = 12 → 
  yellow = 6 := by
sorry

end NUMINAMATH_CALUDE_johnny_guitar_picks_l3393_339339


namespace NUMINAMATH_CALUDE_rose_jasmine_distance_l3393_339318

/-- Represents the positions of trees and flowers on a straight line -/
structure ForestLine where
  -- Positions of trees
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  -- Ensure trees are in order
  ab_pos : a < b
  bc_pos : b < c
  cd_pos : c < d
  de_pos : d < e
  -- Total distance between A and E is 28
  ae_dist : e - a = 28
  -- Positions of flowers
  daisy : ℝ
  rose : ℝ
  jasmine : ℝ
  carnation : ℝ
  -- Flowers at midpoints
  daisy_mid : daisy = (a + b) / 2
  rose_mid : rose = (b + c) / 2
  jasmine_mid : jasmine = (c + d) / 2
  carnation_mid : carnation = (d + e) / 2
  -- Distance between daisy and carnation is 20
  daisy_carnation_dist : carnation - daisy = 20

/-- The distance between the rose bush and the jasmine is 6 meters -/
theorem rose_jasmine_distance (f : ForestLine) : f.jasmine - f.rose = 6 := by
  sorry

end NUMINAMATH_CALUDE_rose_jasmine_distance_l3393_339318


namespace NUMINAMATH_CALUDE_rational_times_sqrt_two_rational_implies_zero_l3393_339308

theorem rational_times_sqrt_two_rational_implies_zero (x : ℚ) :
  (∃ (y : ℚ), y = x * Real.sqrt 2) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_times_sqrt_two_rational_implies_zero_l3393_339308


namespace NUMINAMATH_CALUDE_gym_cost_is_650_l3393_339330

/-- Calculates the total cost of two gym memberships for a year -/
def total_gym_cost (cheap_monthly_fee : ℕ) (cheap_signup_fee : ℕ) : ℕ :=
  let expensive_monthly_fee := 3 * cheap_monthly_fee
  let expensive_signup_fee := 4 * expensive_monthly_fee
  let cheap_yearly_cost := 12 * cheap_monthly_fee + cheap_signup_fee
  let expensive_yearly_cost := 12 * expensive_monthly_fee + expensive_signup_fee
  cheap_yearly_cost + expensive_yearly_cost

/-- Proves that the total cost of two gym memberships for a year is $650 -/
theorem gym_cost_is_650 : total_gym_cost 10 50 = 650 := by
  sorry

end NUMINAMATH_CALUDE_gym_cost_is_650_l3393_339330


namespace NUMINAMATH_CALUDE_enrique_commission_l3393_339399

/-- Calculates the total commission for a salesperson given their sales and commission rate. -/
def calculate_commission (suit_price : ℚ) (suit_count : ℕ) 
                         (shirt_price : ℚ) (shirt_count : ℕ) 
                         (loafer_price : ℚ) (loafer_count : ℕ) 
                         (commission_rate : ℚ) : ℚ :=
  let total_sales := suit_price * suit_count + 
                     shirt_price * shirt_count + 
                     loafer_price * loafer_count
  total_sales * commission_rate

/-- Theorem stating that Enrique's commission is $300.00 given his sales and commission rate. -/
theorem enrique_commission :
  calculate_commission 700 2 50 6 150 2 (15/100) = 300 := by
  sorry

end NUMINAMATH_CALUDE_enrique_commission_l3393_339399


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l3393_339396

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3)
  : (total_families * total_average) / (total_families - childless_families) = 45 / 12 := by
  sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l3393_339396


namespace NUMINAMATH_CALUDE_path_length_for_73_l3393_339313

/-- The length of a path along squares constructed on subdivisions of a segment --/
def path_length (segment_length : ℝ) : ℝ :=
  3 * segment_length

theorem path_length_for_73 :
  path_length 73 = 219 := by
  sorry

end NUMINAMATH_CALUDE_path_length_for_73_l3393_339313


namespace NUMINAMATH_CALUDE_equation_solutions_l3393_339398

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = -11 ∧ x₂ = 1 ∧ 
  (∀ x : ℝ, 4 * (2 * x + 1)^2 = 9 * (x - 3)^2 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3393_339398


namespace NUMINAMATH_CALUDE_martha_black_butterflies_l3393_339360

def butterfly_collection (total blue yellow black : ℕ) : Prop :=
  total = blue + yellow + black ∧ blue = 2 * yellow

theorem martha_black_butterflies :
  ∀ total blue yellow black : ℕ,
  butterfly_collection total blue yellow black →
  total = 19 →
  blue = 6 →
  black = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_martha_black_butterflies_l3393_339360


namespace NUMINAMATH_CALUDE_virginia_friends_l3393_339347

/-- The number of friends Virginia gave Sweettarts to -/
def num_friends (total : ℕ) (per_person : ℕ) : ℕ :=
  (total / per_person) - 1

/-- Proof that Virginia gave Sweettarts to 3 friends -/
theorem virginia_friends :
  num_friends 13 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_virginia_friends_l3393_339347


namespace NUMINAMATH_CALUDE_triangle_side_length_l3393_339348

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 : ℝ) * a * c * Real.sin B = Real.sqrt 3 →
  B = π / 3 →
  a^2 + c^2 = 3 * a * c →
  b = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l3393_339348


namespace NUMINAMATH_CALUDE_complex_quadratic_roots_l3393_339393

theorem complex_quadratic_roots (z : ℂ) :
  z ^ 2 = -91 + 104 * I ∧ (7 + 10 * I) ^ 2 = -91 + 104 * I →
  z = 7 + 10 * I ∨ z = -7 - 10 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_quadratic_roots_l3393_339393


namespace NUMINAMATH_CALUDE_fraction_simplification_l3393_339350

theorem fraction_simplification :
  (1722^2 - 1715^2) / (1731^2 - 1708^2) = (7 * 3437) / (23 * 3439) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3393_339350


namespace NUMINAMATH_CALUDE_charcoal_drawings_count_l3393_339384

/-- The total number of drawings Thomas has -/
def total_drawings : ℕ := 120

/-- The number of drawings made with colored pencils -/
def colored_pencil_drawings : ℕ := 35

/-- The number of drawings made with blending markers -/
def blending_marker_drawings : ℕ := 22

/-- The number of drawings made with pastels -/
def pastel_drawings : ℕ := 15

/-- The number of drawings made with watercolors -/
def watercolor_drawings : ℕ := 12

/-- The number of charcoal drawings -/
def charcoal_drawings : ℕ := total_drawings - (colored_pencil_drawings + blending_marker_drawings + pastel_drawings + watercolor_drawings)

theorem charcoal_drawings_count : charcoal_drawings = 36 := by
  sorry

end NUMINAMATH_CALUDE_charcoal_drawings_count_l3393_339384


namespace NUMINAMATH_CALUDE_parabola_and_tangent_lines_l3393_339364

-- Define the parabola
structure Parabola where
  -- Standard form equation: x² = 2py
  p : ℝ
  -- Vertex at origin
  vertex : (ℝ × ℝ) := (0, 0)
  -- Focus on y-axis
  focus : (ℝ × ℝ) := (0, p)

-- Define a point on the parabola
def point_on_parabola (par : Parabola) (x y : ℝ) : Prop :=
  x^2 = 2 * par.p * y

-- Define a line
structure Line where
  m : ℝ
  b : ℝ

-- Define a point on a line
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.b

-- Define when a line intersects a parabola at a single point
def single_intersection (par : Parabola) (l : Line) : Prop :=
  ∃! (x y : ℝ), point_on_parabola par x y ∧ point_on_line l x y

-- Theorem statement
theorem parabola_and_tangent_lines :
  ∃ (par : Parabola),
    -- Parabola passes through (2, 1)
    point_on_parabola par 2 1 ∧
    -- Standard equation is x² = 4y
    par.p = 2 ∧
    -- Lines x = 2 and x - y - 1 = 0 are the only lines through (2, 1)
    -- that intersect the parabola at a single point
    (∀ (l : Line),
      point_on_line l 2 1 →
      single_intersection par l ↔ (l.m = 0 ∧ l.b = 2) ∨ (l.m = 1 ∧ l.b = -1)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_tangent_lines_l3393_339364


namespace NUMINAMATH_CALUDE_range_of_linear_function_l3393_339303

-- Define the function f on the closed interval [0, 1]
def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

-- State the theorem
theorem range_of_linear_function
  (a b : ℝ)
  (h_a_neg : a < 0)
  : Set.range (fun x => f x a b) = Set.Icc (a + b) b := by
  sorry

end NUMINAMATH_CALUDE_range_of_linear_function_l3393_339303


namespace NUMINAMATH_CALUDE_shopkeeper_loss_l3393_339309

theorem shopkeeper_loss (X : ℝ) (h : X > 0) : 
  let intended_sale_price := 1.1 * X
  let remaining_goods_value := 0.4 * X
  let actual_sale_price := 1.1 * remaining_goods_value
  let loss := X - actual_sale_price
  let percentage_loss := (loss / X) * 100
  percentage_loss = 56 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_l3393_339309


namespace NUMINAMATH_CALUDE_function_identity_l3393_339371

theorem function_identity (f : ℕ → ℕ) (h : ∀ n, f (n + 1) > f (f n)) : ∀ n, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l3393_339371
