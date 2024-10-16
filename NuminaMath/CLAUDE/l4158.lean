import Mathlib

namespace NUMINAMATH_CALUDE_empty_plane_speed_theorem_l4158_415858

/-- The speed of an empty plane given the conditions of the problem -/
def empty_plane_speed (p1 p2 p3 : ℕ) (speed_reduction : ℕ) (avg_speed : ℕ) : ℕ :=
  3 * avg_speed + p1 * speed_reduction + p2 * speed_reduction + p3 * speed_reduction

/-- Theorem stating the speed of an empty plane under the given conditions -/
theorem empty_plane_speed_theorem :
  empty_plane_speed 50 60 40 2 500 = 600 := by
  sorry

#eval empty_plane_speed 50 60 40 2 500

end NUMINAMATH_CALUDE_empty_plane_speed_theorem_l4158_415858


namespace NUMINAMATH_CALUDE_carpenter_woodblocks_l4158_415813

theorem carpenter_woodblocks (total_needed : ℕ) (current_logs : ℕ) (additional_logs : ℕ) 
  (h1 : total_needed = 80)
  (h2 : current_logs = 8)
  (h3 : additional_logs = 8) :
  (total_needed / (current_logs + additional_logs) : ℕ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_carpenter_woodblocks_l4158_415813


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4158_415851

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + 2*i) / (3 - 4*i)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4158_415851


namespace NUMINAMATH_CALUDE_range_of_t_l4158_415889

theorem range_of_t (t α β : ℝ) 
  (h1 : t = Real.cos β ^ 3 + (α / 2) * Real.cos β)
  (h2 : α ≤ t)
  (h3 : t ≤ α - 5 * Real.cos β) :
  -2/3 ≤ t ∧ t ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l4158_415889


namespace NUMINAMATH_CALUDE_stop_time_is_sixty_l4158_415884

/-- Calculates the stop time in minutes given the total journey time and driving time in hours. -/
def stop_time_minutes (total_journey_hours driving_hours : ℕ) : ℕ :=
  (total_journey_hours - driving_hours) * 60

/-- Theorem stating that the stop time is 60 minutes given the specific journey details. -/
theorem stop_time_is_sixty : stop_time_minutes 13 12 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stop_time_is_sixty_l4158_415884


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4158_415868

theorem negation_of_universal_proposition (P Q : Prop) :
  (P ↔ ∀ x : ℤ, x < 1) →
  (Q ↔ ∃ x : ℤ, x ≥ 1) →
  (¬P ↔ Q) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4158_415868


namespace NUMINAMATH_CALUDE_intersection_sum_l4158_415844

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The cubic equation y = x³ - 3x - 4 -/
def cubic (p : Point) : Prop :=
  p.y = p.x^3 - 3*p.x - 4

/-- The linear equation x + 3y = 3 -/
def linear (p : Point) : Prop :=
  p.x + 3*p.y = 3

theorem intersection_sum :
  ∃ (p₁ p₂ p₃ : Point),
    (cubic p₁ ∧ linear p₁) ∧
    (cubic p₂ ∧ linear p₂) ∧
    (cubic p₃ ∧ linear p₃) ∧
    (p₁.x + p₂.x + p₃.x = 8/3) ∧
    (p₁.y + p₂.y + p₃.y = 19/9) := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l4158_415844


namespace NUMINAMATH_CALUDE_arrangements_eq_24_l4158_415886

/-- The number of letter cards -/
def n : ℕ := 6

/-- The number of cards that can be freely arranged -/
def k : ℕ := n - 2

/-- The number of different arrangements of n letter cards where two cards are fixed at the ends -/
def num_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 2)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_eq_24 : num_arrangements n = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_eq_24_l4158_415886


namespace NUMINAMATH_CALUDE_parallelogram_area_18_16_l4158_415803

/-- The area of a parallelogram given its base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 16 cm is 288 square centimeters -/
theorem parallelogram_area_18_16 : 
  parallelogram_area 18 16 = 288 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_18_16_l4158_415803


namespace NUMINAMATH_CALUDE_decagon_diagonals_l4158_415806

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l4158_415806


namespace NUMINAMATH_CALUDE_parallel_perpendicular_plane_perpendicular_plane_line_parallel_l4158_415899

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)

-- Statement 2
theorem parallel_perpendicular_plane 
  (a b : Line) (α : Plane) :
  parallel a b → perpendicularToPlane a α → perpendicularToPlane b α :=
sorry

-- Statement 4
theorem perpendicular_plane_line_parallel 
  (a b : Line) (α : Plane) :
  perpendicularToPlane a α → perpendicular b a → parallel a b :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_plane_perpendicular_plane_line_parallel_l4158_415899


namespace NUMINAMATH_CALUDE_class_composition_l4158_415821

theorem class_composition (n : ℕ) (m : ℕ) : 
  n > 0 ∧ m > 0 ∧ m ≤ n ∧ 
  (⌊(m : ℚ) / n * 100 + 0.5⌋ : ℚ) = 51 →
  Odd n ∧ n ≥ 35 :=
by sorry

end NUMINAMATH_CALUDE_class_composition_l4158_415821


namespace NUMINAMATH_CALUDE_money_division_l4158_415848

/-- The problem of dividing money among three children -/
theorem money_division (anusha babu esha : ℕ) : 
  12 * anusha = 8 * babu ∧ 
  8 * babu = 6 * esha ∧ 
  anusha = 84 → 
  anusha + babu + esha = 378 := by
  sorry


end NUMINAMATH_CALUDE_money_division_l4158_415848


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4158_415812

def triangle_abc (a b c A B C : ℝ) : Prop :=
  b = c * (2 * Real.sin A + Real.cos A) ∧ 
  a = Real.sqrt 2 ∧ 
  B = 3 * Real.pi / 4

theorem triangle_abc_properties (a b c A B C : ℝ) 
  (h : triangle_abc a b c A B C) :
  Real.sin C = Real.sqrt 5 / 5 ∧ 
  (1/2) * a * c * Real.sin B = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l4158_415812


namespace NUMINAMATH_CALUDE_basketball_substitutions_remainder_l4158_415873

/-- Number of ways to make substitutions in a basketball game -/
def substitution_ways (total_players starters max_substitutions : ℕ) : ℕ :=
  let substitutes := total_players - starters
  let a0 := 1  -- No substitutions
  let a1 := starters * substitutes  -- One substitution
  let a2 := a1 * (starters - 1) * (substitutes - 1)  -- Two substitutions
  let a3 := a2 * (starters - 2) * (substitutes - 2)  -- Three substitutions
  let a4 := a3 * (starters - 3) * (substitutes - 3)  -- Four substitutions
  a0 + a1 + a2 + a3 + a4

/-- Theorem stating the remainder when the number of substitution ways is divided by 1000 -/
theorem basketball_substitutions_remainder :
  substitution_ways 14 5 4 % 1000 = 606 := by
  sorry

end NUMINAMATH_CALUDE_basketball_substitutions_remainder_l4158_415873


namespace NUMINAMATH_CALUDE_max_value_of_min_expression_l4158_415830

theorem max_value_of_min_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (⨅ x ∈ ({1/a, 2/b, 4/c, (a*b*c)^(1/3)} : Set ℝ), x) ≤ Real.sqrt 2 ∧ 
  ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    (⨅ x ∈ ({1/a', 2/b', 4/c', (a'*b'*c')^(1/3)} : Set ℝ), x) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_min_expression_l4158_415830


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l4158_415831

/-- The line y = mx + 2 is tangent to the ellipse x² + y²/4 = 1 if and only if m² = 0 -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! p : ℝ × ℝ, p.1^2 + (p.2^2 / 4) = 1 ∧ p.2 = m * p.1 + 2) ↔ m^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l4158_415831


namespace NUMINAMATH_CALUDE_joined_right_triangles_fourth_square_l4158_415847

theorem joined_right_triangles_fourth_square 
  (PQ QR RS : ℝ) 
  (h1 : PQ^2 = 25) 
  (h2 : QR^2 = 49) 
  (h3 : RS^2 = 64) 
  (h4 : PQ > 0 ∧ QR > 0 ∧ RS > 0) : 
  (PQ^2 + QR^2) + RS^2 = 138 := by
  sorry

end NUMINAMATH_CALUDE_joined_right_triangles_fourth_square_l4158_415847


namespace NUMINAMATH_CALUDE_reciprocal_of_hcf_24_182_l4158_415879

theorem reciprocal_of_hcf_24_182 : 
  let a : ℕ := 24
  let b : ℕ := 182
  let hcf := Nat.gcd a b
  1 / (hcf : ℚ) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_hcf_24_182_l4158_415879


namespace NUMINAMATH_CALUDE_circle_diameter_theorem_l4158_415880

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a point inside or on a circle
def PointInOrOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 ≤ c.radius^2

-- Define a diameter of a circle
def Diameter (c : Circle) (d : ℝ × ℝ → ℝ × ℝ → Prop) : Prop :=
  ∀ p q, d p q → PointOnCircle c p ∧ PointOnCircle c q ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 4 * c.radius^2

-- Define a point being on one side of a diameter
def OnOneSideOfDiameter (c : Circle) (d : ℝ × ℝ → ℝ × ℝ → Prop) (p : ℝ × ℝ) : Prop :=
  ∃ q r, Diameter c d ∧ d q r ∧
    ((p.1 - q.1) * (r.2 - q.2) - (p.2 - q.2) * (r.1 - q.1) ≥ 0 ∨
     (p.1 - q.1) * (r.2 - q.2) - (p.2 - q.2) * (r.1 - q.1) ≤ 0)

theorem circle_diameter_theorem (ω : Circle) (inner_circle : Circle) (points : Finset (ℝ × ℝ)) 
    (h1 : ∀ p ∈ points, PointOnCircle ω p)
    (h2 : inner_circle.radius < ω.radius)
    (h3 : ∀ p ∈ points, PointInOrOnCircle inner_circle p) :
  ∃ d : ℝ × ℝ → ℝ × ℝ → Prop, Diameter ω d ∧ 
    (∀ p ∈ points, OnOneSideOfDiameter ω d p) ∧
    (∀ p q, d p q → p ∉ points ∧ q ∉ points) :=
  sorry

end NUMINAMATH_CALUDE_circle_diameter_theorem_l4158_415880


namespace NUMINAMATH_CALUDE_math_club_team_selection_l4158_415815

theorem math_club_team_selection (total_boys : ℕ) (total_girls : ℕ) 
  (team_boys : ℕ) (team_girls : ℕ) : 
  total_boys = 7 → 
  total_girls = 9 → 
  team_boys = 4 → 
  team_girls = 3 → 
  (team_boys + team_girls : ℕ) = 7 → 
  (Nat.choose total_boys team_boys) * (Nat.choose total_girls team_girls) = 2940 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l4158_415815


namespace NUMINAMATH_CALUDE_sum_of_digits_511_base2_l4158_415832

/-- The sum of the digits in the base-2 representation of 511₁₀ is 9. -/
theorem sum_of_digits_511_base2 : 
  (List.range 9).sum = (Nat.digits 2 511).sum := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_511_base2_l4158_415832


namespace NUMINAMATH_CALUDE_negation_equivalence_l4158_415818

theorem negation_equivalence :
  (¬ ∀ (a b : ℝ), ab > 0 → a > 0) ↔ (∀ (a b : ℝ), ab ≤ 0 → a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4158_415818


namespace NUMINAMATH_CALUDE_simon_age_is_10_l4158_415827

def alvin_age : ℕ := 30

def simon_age : ℕ := alvin_age / 2 - 5

theorem simon_age_is_10 : simon_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_simon_age_is_10_l4158_415827


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l4158_415887

theorem bobby_candy_problem (initial_candy : ℕ) : 
  initial_candy + 4 + 14 = 51 → initial_candy = 33 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l4158_415887


namespace NUMINAMATH_CALUDE_oil_price_reduction_l4158_415857

/-- Given a 20% reduction in the price of oil, if a housewife can obtain 5 kg more for Rs. 800 after the reduction, then the reduced price per kg is Rs. 32. -/
theorem oil_price_reduction (P : ℝ) (h1 : P > 0) :
  let R := 0.8 * P
  800 / R - 800 / P = 5 →
  R = 32 := by sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l4158_415857


namespace NUMINAMATH_CALUDE_quadratic_equation_from_roots_l4158_415860

theorem quadratic_equation_from_roots (x₁ x₂ : ℝ) (hx₁ : x₁ = 1) (hx₂ : x₂ = 2) :
  ∃ a b c : ℝ, a ≠ 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧
  a * x^2 + b * x + c = x^2 - 3*x + 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_roots_l4158_415860


namespace NUMINAMATH_CALUDE_sixth_root_equation_l4158_415816

theorem sixth_root_equation (x : ℝ) : 
  (x * (x^4)^(1/3))^(1/6) = 2 → x = 2^(18/7) := by
sorry

end NUMINAMATH_CALUDE_sixth_root_equation_l4158_415816


namespace NUMINAMATH_CALUDE_volume_of_solid_l4158_415876

/-- Volume of a solid with specific dimensions -/
theorem volume_of_solid (a : ℝ) (h1 : a = 3 * Real.sqrt 2) : 
  2 * a^3 = 108 * Real.sqrt 2 := by
  sorry

#check volume_of_solid

end NUMINAMATH_CALUDE_volume_of_solid_l4158_415876


namespace NUMINAMATH_CALUDE_smallest_valid_number_l4158_415888

def is_valid_number (x : ℕ) : Prop :=
  x > 0 ∧ 
  ∃ (multiples : Finset ℕ), 
    multiples.card = 10 ∧ 
    ∀ m ∈ multiples, 
      m < 100 ∧ 
      m % 2 = 1 ∧ 
      ∃ k : ℕ, k % 2 = 1 ∧ m = k * x

theorem smallest_valid_number : 
  ∀ y : ℕ, y < 3 → ¬(is_valid_number y) ∧ is_valid_number 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l4158_415888


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l4158_415890

theorem consecutive_integers_square_sum (n : ℕ) : 
  (n > 0) → 
  (n^2 + (n + 1)^2 = n * (n + 1) + 91) → 
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l4158_415890


namespace NUMINAMATH_CALUDE_unique_rational_root_l4158_415850

def f (x : ℚ) : ℚ := 6 * x^5 - 4 * x^4 - 16 * x^3 + 8 * x^2 + 4 * x - 3

theorem unique_rational_root :
  ∀ x : ℚ, f x = 0 ↔ x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_unique_rational_root_l4158_415850


namespace NUMINAMATH_CALUDE_train_distance_l4158_415838

/-- Represents the efficiency of a coal-powered train in miles per pound of coal -/
def train_efficiency : ℚ := 5 / 2

/-- Represents the amount of coal remaining in pounds -/
def coal_remaining : ℕ := 160

/-- Calculates the distance a train can travel given its efficiency and remaining coal -/
def distance_traveled (efficiency : ℚ) (coal : ℕ) : ℚ :=
  efficiency * coal

/-- Theorem stating that the train can travel 400 miles before running out of fuel -/
theorem train_distance : distance_traveled train_efficiency coal_remaining = 400 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l4158_415838


namespace NUMINAMATH_CALUDE_intersection_when_m_neg_three_subset_condition_l4158_415854

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Theorem 1: A ∩ B when m = -3
theorem intersection_when_m_neg_three :
  A ∩ B (-3) = {x | -3 ≤ x ∧ x ≤ -2} := by sorry

-- Theorem 2: B ⊆ A iff m ≥ -1
theorem subset_condition :
  ∀ m : ℝ, B m ⊆ A ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_neg_three_subset_condition_l4158_415854


namespace NUMINAMATH_CALUDE_greg_earnings_l4158_415804

def base_charge : ℕ := 20
def per_minute_charge : ℕ := 1

def earnings (num_dogs : ℕ) (minutes : ℕ) : ℕ :=
  num_dogs * (base_charge + per_minute_charge * minutes)

theorem greg_earnings : 
  earnings 1 10 + earnings 2 7 + earnings 3 9 = 171 := by
  sorry

end NUMINAMATH_CALUDE_greg_earnings_l4158_415804


namespace NUMINAMATH_CALUDE_expression_value_l4158_415846

/-- Custom operation for real numbers -/
def custom_op (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating the value of the expression when x^2 - 3x + 1 = 0 -/
theorem expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  custom_op (x + 1) (x - 2) (3*x) (x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4158_415846


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4158_415824

/-- Given vectors a, b, and c in ℝ², prove that if a + b is parallel to c, then x = -5 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2*x, x]
  let c : Fin 2 → ℝ := ![3, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • c) →
  x = -5 := by
sorry


end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4158_415824


namespace NUMINAMATH_CALUDE_equation_solvability_l4158_415866

theorem equation_solvability (a : ℝ) : 
  (∃ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x = 2 * a - 1) ↔ 
  (-1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_equation_solvability_l4158_415866


namespace NUMINAMATH_CALUDE_not_all_sqrt5_periodic_all_sqrt3_periodic_l4158_415826

-- Define the function types
def RealFunction := ℝ → ℝ

-- Define the functional equations
def SatisfiesSqrt5Equation (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (x - 1) + f (x + 1) = Real.sqrt 5 * f x

def SatisfiesSqrt3Equation (g : RealFunction) : Prop :=
  ∀ x : ℝ, g (x - 1) + g (x + 1) = Real.sqrt 3 * g x

-- Define periodicity
def IsPeriodic (f : RealFunction) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = f x

-- Theorem statements
theorem not_all_sqrt5_periodic :
  ∃ f : RealFunction, SatisfiesSqrt5Equation f ∧ ¬IsPeriodic f :=
sorry

theorem all_sqrt3_periodic :
  ∀ g : RealFunction, SatisfiesSqrt3Equation g → IsPeriodic g :=
sorry

end NUMINAMATH_CALUDE_not_all_sqrt5_periodic_all_sqrt3_periodic_l4158_415826


namespace NUMINAMATH_CALUDE_total_length_climbed_result_l4158_415875

/-- The total length of ladders climbed by two workers in inches -/
def total_length_climbed (keaton_ladder_height : ℕ) (keaton_climbs : ℕ) 
  (reece_ladder_diff : ℕ) (reece_climbs : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - reece_ladder_diff
  let keaton_total := keaton_ladder_height * keaton_climbs
  let reece_total := reece_ladder_height * reece_climbs
  (keaton_total + reece_total) * 12

/-- Theorem stating the total length climbed by both workers -/
theorem total_length_climbed_result : 
  total_length_climbed 30 20 4 15 = 11880 := by
  sorry

end NUMINAMATH_CALUDE_total_length_climbed_result_l4158_415875


namespace NUMINAMATH_CALUDE_third_angle_relationship_l4158_415835

theorem third_angle_relationship (a b c : ℝ) : 
  a = b → a = 36 → a + b + c = 180 → c = 3 * a := by sorry

end NUMINAMATH_CALUDE_third_angle_relationship_l4158_415835


namespace NUMINAMATH_CALUDE_jeremy_purchase_l4158_415825

theorem jeremy_purchase (computer_price : ℝ) (accessory_percentage : ℝ) (initial_money_factor : ℝ) : 
  computer_price = 3000 →
  accessory_percentage = 0.1 →
  initial_money_factor = 2 →
  let accessory_price := computer_price * accessory_percentage
  let initial_money := computer_price * initial_money_factor
  let total_spent := computer_price + accessory_price
  initial_money - total_spent = 2700 := by
sorry

end NUMINAMATH_CALUDE_jeremy_purchase_l4158_415825


namespace NUMINAMATH_CALUDE_f_always_positive_l4158_415817

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Triangle inequality conditions
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define the function f(x)
def f (t : Triangle) (x : ℝ) : ℝ :=
  t.b^2 * x^2 + (t.b^2 + t.c^2 - t.a^2) * x + t.c^2

-- Theorem statement
theorem f_always_positive (t : Triangle) : ∀ x : ℝ, f t x > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_always_positive_l4158_415817


namespace NUMINAMATH_CALUDE_probability_at_least_one_unqualified_l4158_415898

/-- The number of products -/
def total_products : ℕ := 5

/-- The number of qualified products -/
def qualified_products : ℕ := 3

/-- The number of unqualified products -/
def unqualified_products : ℕ := 2

/-- The number of products inspected -/
def inspected_products : ℕ := 2

/-- The probability of selecting at least one unqualified product -/
def prob_at_least_one_unqualified : ℚ := 7/10

/-- Theorem stating the probability of selecting at least one unqualified product -/
theorem probability_at_least_one_unqualified :
  let total_ways := Nat.choose total_products inspected_products
  let qualified_ways := Nat.choose qualified_products inspected_products
  1 - (qualified_ways : ℚ) / (total_ways : ℚ) = prob_at_least_one_unqualified :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_unqualified_l4158_415898


namespace NUMINAMATH_CALUDE_inequality_solution_l4158_415828

theorem inequality_solution (x : ℝ) :
  x + 1 > 0 →
  x + 1 - Real.sqrt (x + 1) ≠ 0 →
  (x^2 / ((x + 1 - Real.sqrt (x + 1))^2) < (x^2 + 3*x + 18) / (x + 1)^2) ↔ 
  (-1 < x ∧ x < 3) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l4158_415828


namespace NUMINAMATH_CALUDE_improved_running_distance_l4158_415869

/-- Proves that if a person can run 40 yards in 5 seconds and improves their speed by 40%, 
    they can run 112 yards in 10 seconds. -/
theorem improved_running_distance 
  (initial_distance : ℝ) 
  (initial_time : ℝ) 
  (improvement_percentage : ℝ) 
  (new_time : ℝ) :
  initial_distance = 40 ∧ 
  initial_time = 5 ∧ 
  improvement_percentage = 40 ∧ 
  new_time = 10 →
  (initial_distance + initial_distance * (improvement_percentage / 100)) * (new_time / initial_time) = 112 :=
by sorry

end NUMINAMATH_CALUDE_improved_running_distance_l4158_415869


namespace NUMINAMATH_CALUDE_volume_of_specific_solid_l4158_415885

/-- 
A solid with a square base and extended top edges.
s: side length of the square base
-/
structure ExtendedSolid where
  s : ℝ
  base_square : s > 0
  top_extended : ℝ × ℝ
  vertical_edge : ℝ

/-- The volume of the ExtendedSolid -/
noncomputable def volume (solid : ExtendedSolid) : ℝ :=
  solid.s^2 * solid.s

/-- Theorem: The volume of the specific ExtendedSolid is 128√2 -/
theorem volume_of_specific_solid :
  ∃ (solid : ExtendedSolid),
    solid.s = 4 * Real.sqrt 2 ∧
    solid.top_extended = (3 * solid.s, solid.s) ∧
    solid.vertical_edge = solid.s ∧
    volume solid = 128 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_solid_l4158_415885


namespace NUMINAMATH_CALUDE_num_cubes_5_peaks_num_cubes_2014_peaks_painted_area_2014_peaks_l4158_415845

/-- Represents a wall made of unit cubes with a given number of peaks -/
structure Wall where
  peaks : ℕ

/-- The number of cubes needed to construct a wall with n peaks -/
def num_cubes (w : Wall) : ℕ := 3 * w.peaks - 1

/-- The painted surface area of a wall with n peaks, excluding the base -/
def painted_area (w : Wall) : ℕ := 10 * w.peaks + 9

/-- Theorem stating the number of cubes for a wall with 5 peaks -/
theorem num_cubes_5_peaks : num_cubes { peaks := 5 } = 14 := by sorry

/-- Theorem stating the number of cubes for a wall with 2014 peaks -/
theorem num_cubes_2014_peaks : num_cubes { peaks := 2014 } = 6041 := by sorry

/-- Theorem stating the painted area for a wall with 2014 peaks -/
theorem painted_area_2014_peaks : painted_area { peaks := 2014 } = 20139 := by sorry

end NUMINAMATH_CALUDE_num_cubes_5_peaks_num_cubes_2014_peaks_painted_area_2014_peaks_l4158_415845


namespace NUMINAMATH_CALUDE_sum_greater_than_2e_squared_l4158_415836

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem sum_greater_than_2e_squared (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) 
  (hf₁ : f a x₁ = 1) (hf₂ : f a x₂ = 1) : 
  x₁ + x₂ > 2 * (Real.exp 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_2e_squared_l4158_415836


namespace NUMINAMATH_CALUDE_first_class_product_rate_l4158_415814

/-- Given a product with a pass rate and a rate of first-class products among qualified products,
    prove that the overall rate of first-class products is their product. -/
theorem first_class_product_rate
  (pass_rate : ℝ)
  (first_class_rate_among_qualified : ℝ)
  (h1 : 0 ≤ pass_rate ∧ pass_rate ≤ 1)
  (h2 : 0 ≤ first_class_rate_among_qualified ∧ first_class_rate_among_qualified ≤ 1) :
  pass_rate * first_class_rate_among_qualified =
  pass_rate * first_class_rate_among_qualified :=
by sorry

end NUMINAMATH_CALUDE_first_class_product_rate_l4158_415814


namespace NUMINAMATH_CALUDE_meal_cost_is_45_l4158_415852

/-- The cost of a meal consisting of one pizza and three burgers -/
def meal_cost (burger_price : ℝ) : ℝ :=
  let pizza_price := 2 * burger_price
  pizza_price + 3 * burger_price

/-- Theorem: The cost of one pizza and three burgers is $45 -/
theorem meal_cost_is_45 :
  meal_cost 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_is_45_l4158_415852


namespace NUMINAMATH_CALUDE_cindy_envelopes_left_l4158_415862

def envelopes_left (initial_envelopes : ℕ) (num_friends : ℕ) (envelopes_per_friend : ℕ) : ℕ :=
  initial_envelopes - (num_friends * envelopes_per_friend)

theorem cindy_envelopes_left :
  let initial_envelopes : ℕ := 74
  let num_friends : ℕ := 11
  let envelopes_per_friend : ℕ := 6
  envelopes_left initial_envelopes num_friends envelopes_per_friend = 8 := by
  sorry

end NUMINAMATH_CALUDE_cindy_envelopes_left_l4158_415862


namespace NUMINAMATH_CALUDE_license_plate_combinations_l4158_415872

/-- The number of consonants available for the license plate. -/
def num_consonants : ℕ := 21

/-- The number of vowels available for the license plate. -/
def num_vowels : ℕ := 5

/-- The number of digits available for the license plate. -/
def num_digits : ℕ := 10

/-- The total number of possible license plate combinations. -/
def total_combinations : ℕ := num_consonants^2 * num_vowels^2 * num_digits

/-- Theorem stating that the total number of license plate combinations is 110,250. -/
theorem license_plate_combinations : total_combinations = 110250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l4158_415872


namespace NUMINAMATH_CALUDE_smallest_k_cube_sum_multiple_360_k_38_cube_sum_multiple_360_smallest_k_is_38_l4158_415893

theorem smallest_k_cube_sum_multiple_360 : 
  ∀ k : ℕ, k > 0 → (k * (k + 1) / 2)^2 % 360 = 0 → k ≥ 38 :=
by sorry

theorem k_38_cube_sum_multiple_360 : 
  (38 * (38 + 1) / 2)^2 % 360 = 0 :=
by sorry

theorem smallest_k_is_38 :
  ∀ k : ℕ, k > 0 → (k * (k + 1) / 2)^2 % 360 = 0 → k ≥ 38 ∧ 
  (38 * (38 + 1) / 2)^2 % 360 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_cube_sum_multiple_360_k_38_cube_sum_multiple_360_smallest_k_is_38_l4158_415893


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4158_415808

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_term : a 1 = 3
  arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Main theorem -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
    (h : S seq 8 = seq.a 8) : seq.a 19 = -15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4158_415808


namespace NUMINAMATH_CALUDE_dave_tickets_l4158_415895

theorem dave_tickets (initial_tickets : ℕ) : 
  (initial_tickets - 2 - 10 = 2) → initial_tickets = 14 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l4158_415895


namespace NUMINAMATH_CALUDE_f_derivative_inequality_implies_a_range_existence_of_intersecting_tangents_l4158_415820

/-- The function f(x) = x³ - ax² + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem f_derivative_inequality_implies_a_range :
  (∀ x : ℝ, f' 0 x ≥ |x| - 3/4) → -1 ≤ 0 ∧ 0 ≤ 1 := by sorry

theorem existence_of_intersecting_tangents :
  ∃ x₁ x₂ t : ℝ, x₁ ≠ x₂ ∧ 
  (f 0 x₁ + f' 0 x₁ * (2 - x₁) = t) ∧ 
  (f 0 x₂ + f' 0 x₂ * (2 - x₂) = t) ∧
  t ≤ 10 ∧
  (∀ s : ℝ, (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ 
    (f 0 y₁ + f' 0 y₁ * (2 - y₁) = s) ∧ 
    (f 0 y₂ + f' 0 y₂ * (2 - y₂) = s)) → 
  s ≤ 10) := by sorry

end NUMINAMATH_CALUDE_f_derivative_inequality_implies_a_range_existence_of_intersecting_tangents_l4158_415820


namespace NUMINAMATH_CALUDE_circle_logarithm_l4158_415853

theorem circle_logarithm (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : 
  (2 * Real.log a = Real.log (a^2)) →
  (4 * Real.log b = Real.log (b^4)) →
  (2 * π * Real.log (a^2) = Real.log (b^4)) →
  Real.log b / Real.log a = π :=
by sorry

end NUMINAMATH_CALUDE_circle_logarithm_l4158_415853


namespace NUMINAMATH_CALUDE_senior_junior_ratio_l4158_415856

theorem senior_junior_ratio (S J : ℕ) (k : ℕ+) :
  S = k * J →
  (1 / 8 : ℚ) * S + (3 / 4 : ℚ) * J = (1 / 3 : ℚ) * (S + J) →
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_senior_junior_ratio_l4158_415856


namespace NUMINAMATH_CALUDE_min_value_system_min_value_exact_l4158_415829

open Real

theorem min_value_system (x y z : ℝ) 
  (eq1 : 2 * cos x = 1 / tan y)
  (eq2 : 2 * sin y = tan z)
  (eq3 : cos z = 1 / tan x) :
  ∀ (a b c : ℝ), 
    (2 * cos a = 1 / tan b) → 
    (2 * sin b = tan c) → 
    (cos c = 1 / tan a) → 
    sin x + cos z ≤ sin a + cos c :=
by sorry

theorem min_value_exact (x y z : ℝ) 
  (eq1 : 2 * cos x = 1 / tan y)
  (eq2 : 2 * sin y = tan z)
  (eq3 : cos z = 1 / tan x) :
  ∃ (a b c : ℝ), 
    (2 * cos a = 1 / tan b) ∧ 
    (2 * sin b = tan c) ∧ 
    (cos c = 1 / tan a) ∧ 
    sin a + cos c = -5 * Real.sqrt 3 / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_system_min_value_exact_l4158_415829


namespace NUMINAMATH_CALUDE_optimal_avocado_buying_strategy_l4158_415801

/-- Represents the optimal buying strategy for avocados -/
theorem optimal_avocado_buying_strategy 
  (recipe_need : ℕ) 
  (initial_count : ℕ) 
  (price_less_than_10 : ℝ) 
  (price_10_or_more : ℝ) 
  (h1 : recipe_need = 3) 
  (h2 : initial_count = 5) 
  (h3 : price_10_or_more < price_less_than_10) : 
  let additional_buy := 5
  let total_count := initial_count + additional_buy
  let total_cost := additional_buy * price_10_or_more
  (∀ n : ℕ, 
    let alt_total_count := initial_count + n
    let alt_total_cost := if alt_total_count < 10 then n * price_less_than_10 else n * price_10_or_more
    (alt_total_count ≥ recipe_need → total_cost ≤ alt_total_cost) ∧ 
    (alt_total_cost = total_cost → total_count ≥ alt_total_count)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_avocado_buying_strategy_l4158_415801


namespace NUMINAMATH_CALUDE_max_value_expression_l4158_415855

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({1, 2, 3, 4} : Set ℕ) →
  b ∈ ({1, 2, 3, 4} : Set ℕ) →
  c ∈ ({1, 2, 3, 4} : Set ℕ) →
  d ∈ ({1, 2, 3, 4} : Set ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  c * a^b - d ≤ 127 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l4158_415855


namespace NUMINAMATH_CALUDE_test_questions_count_l4158_415811

theorem test_questions_count : ∀ (total_questions : ℕ),
  (total_questions % 4 = 0) →  -- Test consists of 4 equal sections
  (20 : ℝ) / total_questions > 0.60 →  -- Percentage correct > 60%
  (20 : ℝ) / total_questions < 0.70 →  -- Percentage correct < 70%
  total_questions = 32 := by
sorry

end NUMINAMATH_CALUDE_test_questions_count_l4158_415811


namespace NUMINAMATH_CALUDE_existence_condition_l4158_415878

theorem existence_condition (a : ℝ) : 
  (∃ x y : ℝ, Real.sqrt (2 * x * y + a) = x + y + 17) ↔ a ≥ -289/2 := by
sorry

end NUMINAMATH_CALUDE_existence_condition_l4158_415878


namespace NUMINAMATH_CALUDE_andrews_grapes_l4158_415891

theorem andrews_grapes (price_grapes : ℕ) (quantity_mangoes : ℕ) (price_mangoes : ℕ) (total_paid : ℕ) :
  price_grapes = 74 →
  quantity_mangoes = 9 →
  price_mangoes = 59 →
  total_paid = 975 →
  ∃ (quantity_grapes : ℕ), 
    quantity_grapes * price_grapes + quantity_mangoes * price_mangoes = total_paid ∧
    quantity_grapes = 6 := by
  sorry

end NUMINAMATH_CALUDE_andrews_grapes_l4158_415891


namespace NUMINAMATH_CALUDE_carpet_fits_both_rooms_l4158_415897

-- Define the carpet and room dimensions
def carpet_width : ℝ := 25
def carpet_length : ℝ := 50
def room1_width : ℝ := 38
def room1_length : ℝ := 55
def room2_width : ℝ := 50
def room2_length : ℝ := 55

-- Define a function to check if the carpet fits in a room
def carpet_fits_room (carpet_w carpet_l room_w room_l : ℝ) : Prop :=
  carpet_w^2 + carpet_l^2 = room_w^2 + room_l^2

-- Theorem statement
theorem carpet_fits_both_rooms :
  carpet_fits_room carpet_width carpet_length room1_width room1_length ∧
  carpet_fits_room carpet_width carpet_length room2_width room2_length :=
by sorry

end NUMINAMATH_CALUDE_carpet_fits_both_rooms_l4158_415897


namespace NUMINAMATH_CALUDE_fish_in_tank_l4158_415871

theorem fish_in_tank (total : ℕ) (blue : ℕ) (spotted : ℕ) : 
  (3 * blue = total) →  -- One third of the fish are blue
  (2 * spotted = blue) →  -- Half of the blue fish have spots
  (spotted = 10) →  -- There are 10 blue, spotted fish
  total = 60 := by
  sorry

end NUMINAMATH_CALUDE_fish_in_tank_l4158_415871


namespace NUMINAMATH_CALUDE_largest_c_for_f_range_containing_2_l4158_415837

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

-- State the theorem
theorem largest_c_for_f_range_containing_2 :
  (∃ (c : ℝ), ∀ (d : ℝ), 
    (∃ (x : ℝ), f d x = 2) → d ≤ c) ∧
  (∃ (x : ℝ), f 11 x = 2) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_f_range_containing_2_l4158_415837


namespace NUMINAMATH_CALUDE_mary_sheep_count_l4158_415834

theorem mary_sheep_count : ∃ (m : ℕ), 
  (∀ (b : ℕ), b = 2 * m + 35 →
    m + 266 = b - 69) → m = 300 := by
  sorry

end NUMINAMATH_CALUDE_mary_sheep_count_l4158_415834


namespace NUMINAMATH_CALUDE_total_cinnamon_swirls_l4158_415892

/-- The number of people eating cinnamon swirls -/
def num_people : ℕ := 3

/-- The number of pieces Jane ate -/
def janes_pieces : ℕ := 4

/-- Theorem: The total number of cinnamon swirl pieces prepared is 12 -/
theorem total_cinnamon_swirls :
  ∀ (pieces_per_person : ℕ),
  (pieces_per_person = janes_pieces) →
  (num_people * pieces_per_person = 12) :=
by sorry

end NUMINAMATH_CALUDE_total_cinnamon_swirls_l4158_415892


namespace NUMINAMATH_CALUDE_distance_to_focus_l4158_415805

/-- A point on a parabola with a specific distance property -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_to_line : |x + 3| = 5

/-- The theorem stating the distance from the point to the focus -/
theorem distance_to_focus (P : ParabolaPoint) :
  Real.sqrt ((P.x - 1)^2 + P.y^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_l4158_415805


namespace NUMINAMATH_CALUDE_right_triangle_GHI_side_GH_l4158_415841

/-- Represents a right triangle GHI with angle G = 30°, angle H = 90°, and HI = 10 -/
structure RightTriangleGHI where
  G : Real
  H : Real
  I : Real
  angleG : G = 30 * π / 180
  angleH : H = π / 2
  rightAngle : H = π / 2
  sideHI : I = 10

/-- Theorem stating that in the given right triangle GHI, GH = 10√3 -/
theorem right_triangle_GHI_side_GH (t : RightTriangleGHI) : 
  Real.sqrt ((10 * Real.sqrt 3) ^ 2) = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_GHI_side_GH_l4158_415841


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l4158_415877

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- Statement 1: Solution set of f(x) + x^2 - 1 > 0
theorem solution_set_f (x : ℝ) : f x + x^2 - 1 > 0 ↔ x > 1 ∨ x < 0 := by sorry

-- Statement 2: Range of m when solution set of f(x) < g(x) is non-empty
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, f x < g x m) → m > 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l4158_415877


namespace NUMINAMATH_CALUDE_decagon_triangles_l4158_415819

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def num_triangles : ℕ := Nat.choose n k

theorem decagon_triangles : num_triangles = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l4158_415819


namespace NUMINAMATH_CALUDE_problem_statement_l4158_415800

theorem problem_statement (x y : ℝ) (hx : x = 7) (hy : y = -2) :
  (x - 2*y)^y = 1/121 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l4158_415800


namespace NUMINAMATH_CALUDE_circular_saw_blade_distance_l4158_415881

/-- Given a circle with center (2, 3) and radius 8, and points A, B, and C on the circle
    such that ∠ABC is a right angle, AB = 8, and BC = 3, 
    prove that the square of the distance from B to the center of the circle is 41. -/
theorem circular_saw_blade_distance (A B C : ℝ × ℝ) : 
  let O : ℝ × ℝ := (2, 3)
  let r : ℝ := 8
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2 →  -- A is on the circle
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 →  -- B is on the circle
  (C.1 - O.1)^2 + (C.2 - O.2)^2 = r^2 →  -- C is on the circle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8^2 →  -- AB = 8
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 3^2 →  -- BC = 3
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →  -- ∠ABC is a right angle
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = 41 := by
sorry

end NUMINAMATH_CALUDE_circular_saw_blade_distance_l4158_415881


namespace NUMINAMATH_CALUDE_power_sum_equality_l4158_415864

theorem power_sum_equality : (3^2)^3 + (2^3)^2 = 793 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l4158_415864


namespace NUMINAMATH_CALUDE_sculpture_exposed_area_l4158_415822

/-- Represents a layer in the sculpture -/
structure Layer where
  cubes : ℕ
  exposed_top : ℕ
  exposed_side : ℕ

/-- The sculpture configuration -/
def sculpture : List Layer := [
  ⟨9, 9, 16⟩,
  ⟨6, 6, 10⟩,
  ⟨4, 4, 8⟩,
  ⟨1, 1, 4⟩
]

/-- The total number of cubes in the sculpture -/
def total_cubes : ℕ := (sculpture.map Layer.cubes).sum

/-- Calculates the exposed surface area of a layer -/
def exposed_area (layer : Layer) : ℕ := layer.exposed_top + layer.exposed_side

/-- Calculates the total exposed surface area of the sculpture -/
def total_exposed_area : ℕ := (sculpture.map exposed_area).sum

/-- Theorem: The total exposed surface area of the sculpture is 58 square meters -/
theorem sculpture_exposed_area :
  total_cubes = 20 ∧ total_exposed_area = 58 := by sorry

end NUMINAMATH_CALUDE_sculpture_exposed_area_l4158_415822


namespace NUMINAMATH_CALUDE_correct_sales_growth_equation_l4158_415833

/-- Represents the growth of new energy vehicle sales over two months -/
def sales_growth (initial_sales : ℝ) (final_sales : ℝ) (growth_rate : ℝ) : Prop :=
  initial_sales * (1 + growth_rate)^2 = final_sales

/-- Theorem stating that the given equation correctly represents the sales growth -/
theorem correct_sales_growth_equation :
  ∃ x : ℝ, sales_growth 33.2 54.6 x :=
sorry

end NUMINAMATH_CALUDE_correct_sales_growth_equation_l4158_415833


namespace NUMINAMATH_CALUDE_inequality_proof_l4158_415809

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4158_415809


namespace NUMINAMATH_CALUDE_fishermen_catch_l4158_415883

theorem fishermen_catch (total_fish : ℕ) (carp_ratio : ℚ) (perch_ratio : ℚ) 
  (h_total : total_fish = 80)
  (h_carp : carp_ratio = 5 / 9)
  (h_perch : perch_ratio = 7 / 11) :
  ∃ (first_catch second_catch : ℕ),
    first_catch + second_catch = total_fish ∧
    first_catch = 36 ∧
    second_catch = 44 := by
  sorry

end NUMINAMATH_CALUDE_fishermen_catch_l4158_415883


namespace NUMINAMATH_CALUDE_remaining_balance_calculation_l4158_415861

def cost_per_charge : ℚ := 3.5
def number_of_charges : ℕ := 4
def initial_budget : ℚ := 20

theorem remaining_balance_calculation :
  initial_budget - (cost_per_charge * number_of_charges) = 6 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balance_calculation_l4158_415861


namespace NUMINAMATH_CALUDE_equation_and_inequalities_solution_l4158_415894

theorem equation_and_inequalities_solution :
  (∃! x : ℝ, (3 / (x - 1) = 1 / (2 * x + 3))) ∧
  (∀ x : ℝ, (3 * x - 1 ≥ x + 1 ∧ x + 3 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 5 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_and_inequalities_solution_l4158_415894


namespace NUMINAMATH_CALUDE_last_ball_is_white_l4158_415849

/-- Represents the color of a ball -/
inductive BallColor
| White
| Black

/-- Represents the state of the box -/
structure BoxState where
  white : Nat
  black : Nat

/-- The process of drawing and replacing balls -/
def process (state : BoxState) : BoxState :=
  sorry

/-- Predicate to check if the process has terminated (only one ball left) -/
def isTerminated (state : BoxState) : Prop :=
  state.white + state.black = 1

/-- Theorem stating that the last ball is white given an odd initial number of white balls -/
theorem last_ball_is_white 
  (initial_white : Nat) 
  (initial_black : Nat) 
  (h_odd : Odd initial_white) :
  ∃ (final_state : BoxState), 
    (∃ (n : Nat), final_state = (process^[n] ⟨initial_white, initial_black⟩)) ∧ 
    isTerminated final_state ∧ 
    final_state.white = 1 :=
  sorry

end NUMINAMATH_CALUDE_last_ball_is_white_l4158_415849


namespace NUMINAMATH_CALUDE_power_division_seventeen_l4158_415870

theorem power_division_seventeen : (17 : ℕ)^9 / (17 : ℕ)^7 = 289 := by sorry

end NUMINAMATH_CALUDE_power_division_seventeen_l4158_415870


namespace NUMINAMATH_CALUDE_slope_of_line_l4158_415839

/-- The slope of a line given by the equation 4y = -6x + 12 is -3/2 -/
theorem slope_of_line (x y : ℝ) : 4 * y = -6 * x + 12 → (y - 3) / x = -3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l4158_415839


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l4158_415882

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x - 1 = 0) ↔ (m ≥ -3 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l4158_415882


namespace NUMINAMATH_CALUDE_f_increasing_implies_k_nonpositive_l4158_415810

def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x - 8

theorem f_increasing_implies_k_nonpositive :
  ∀ k : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 14 → f k x < f k y) → k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_implies_k_nonpositive_l4158_415810


namespace NUMINAMATH_CALUDE_two_numbers_difference_l4158_415859

theorem two_numbers_difference (x y : ℚ) 
  (sum_eq : x + y = 40)
  (triple_minus_quadruple : 3 * y - 4 * x = 20) :
  |y - x| = 80 / 7 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l4158_415859


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l4158_415843

def anna_lap_time : ℕ := 5
def stephanie_lap_time : ℕ := 8
def james_lap_time : ℕ := 10

theorem earliest_meeting_time :
  let lap_times := [anna_lap_time, stephanie_lap_time, james_lap_time]
  Nat.lcm (Nat.lcm anna_lap_time stephanie_lap_time) james_lap_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l4158_415843


namespace NUMINAMATH_CALUDE_cos_210_degrees_l4158_415840

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l4158_415840


namespace NUMINAMATH_CALUDE_average_of_abc_is_three_l4158_415865

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 1501 * C - 3003 * A = 6006)
  (eq2 : 1501 * B + 4504 * A = 7507)
  (eq3 : A + B = 1) :
  (A + B + C) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_abc_is_three_l4158_415865


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l4158_415807

/-- A parallelogram with opposite vertices at (2, -3) and (10, 9) has its diagonals intersecting at (6, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (10, 9)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (6, 3) := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l4158_415807


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l4158_415823

/-- Given two parallel vectors a and b, prove that the magnitude of b is 2√10 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, 3) → 
  b.1 = -2 → 
  (∃ k : ℝ, b = k • a) → 
  ‖b‖ = 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l4158_415823


namespace NUMINAMATH_CALUDE_n_times_n_plus_one_divisible_by_two_l4158_415874

theorem n_times_n_plus_one_divisible_by_two (n : ℕ) (h : 1 ≤ n ∧ n ≤ 99) : 
  2 ∣ (n * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_n_times_n_plus_one_divisible_by_two_l4158_415874


namespace NUMINAMATH_CALUDE_nickels_count_l4158_415842

/-- Given 30 coins (nickels and dimes) with a total value of 240 cents, prove that the number of nickels is 12. -/
theorem nickels_count (n d : ℕ) : 
  n + d = 30 →  -- Total number of coins
  5 * n + 10 * d = 240 →  -- Total value in cents
  n = 12 :=  -- Number of nickels
by sorry

end NUMINAMATH_CALUDE_nickels_count_l4158_415842


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_set_l4158_415867

def A : Set ℝ := {-2, -1, 0, 1}

def B : Set ℝ := {y | ∃ x, y = 1 / (2^x - 2)}

theorem A_intersect_B_eq_set : A ∩ B = {-2, -1, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_set_l4158_415867


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l4158_415863

/-- The volume of a rectangular box given the areas of its faces -/
theorem rectangular_box_volume (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (area1 : a * b = 36) (area2 : b * c = 12) (area3 : a * c = 9) : 
  a * b * c = 144 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l4158_415863


namespace NUMINAMATH_CALUDE_quotient_negative_one_sum_zero_l4158_415802

theorem quotient_negative_one_sum_zero (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a / b = -1 → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quotient_negative_one_sum_zero_l4158_415802


namespace NUMINAMATH_CALUDE_car_rental_cost_l4158_415896

/-- Calculates the car rental cost for a vacation given the number of people,
    Airbnb rental cost, and each person's share. -/
theorem car_rental_cost (num_people : ℕ) (airbnb_cost : ℕ) (person_share : ℕ) : 
  num_people = 8 → airbnb_cost = 3200 → person_share = 500 →
  num_people * person_share - airbnb_cost = 800 := by
sorry

end NUMINAMATH_CALUDE_car_rental_cost_l4158_415896
