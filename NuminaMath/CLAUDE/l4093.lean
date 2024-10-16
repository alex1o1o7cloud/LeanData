import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_proposition_l4093_409393

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l4093_409393


namespace NUMINAMATH_CALUDE_large_bucket_capacity_l4093_409324

theorem large_bucket_capacity (small_bucket : ℝ) (large_bucket : ℝ) : 
  (large_bucket = 2 * small_bucket + 3) →
  (2 * small_bucket + 5 * large_bucket = 63) →
  large_bucket = 11 := by
sorry

end NUMINAMATH_CALUDE_large_bucket_capacity_l4093_409324


namespace NUMINAMATH_CALUDE_largest_n_for_triangle_inequality_l4093_409305

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, 
    such that ∠A + ∠C = 2∠B, the largest positive integer n 
    for which a^n + c^n ≤ 2b^n holds is 4. -/
theorem largest_n_for_triangle_inequality (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  A > 0 → B > 0 → C > 0 → 
  A + B + C = π → 
  A + C = 2 * B → 
  ∃ (n : ℕ), n > 0 ∧ a^n + c^n ≤ 2*b^n ∧ 
  ∀ (m : ℕ), m > n → ¬(a^m + c^m ≤ 2*b^m) → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_largest_n_for_triangle_inequality_l4093_409305


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l4093_409364

theorem alcohol_mixture_problem (x : ℝ) :
  (x * 50 + 30 * 150) / (50 + 150) = 25 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l4093_409364


namespace NUMINAMATH_CALUDE_wind_velocity_problem_l4093_409326

/-- Represents the relationship between pressure, area, and velocity -/
def pressure_relation (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^2

theorem wind_velocity_problem (k : ℝ) :
  pressure_relation k 2 8 = 4 →
  pressure_relation k 4.5 (40/3) = 25 :=
by sorry

end NUMINAMATH_CALUDE_wind_velocity_problem_l4093_409326


namespace NUMINAMATH_CALUDE_combustible_ice_reserves_scientific_notation_l4093_409306

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem combustible_ice_reserves_scientific_notation :
  scientific_notation 150000000000 = (1.5, 11) :=
sorry

end NUMINAMATH_CALUDE_combustible_ice_reserves_scientific_notation_l4093_409306


namespace NUMINAMATH_CALUDE_largest_number_is_4968_l4093_409329

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total_students : ℕ
  first_number : ℕ
  second_number : ℕ
  hTotal : total_students = 5000
  hRange : first_number ≥ 1 ∧ second_number ≤ total_students
  hOrder : first_number < second_number

/-- The largest number in the systematic sample -/
def largest_number (s : SystematicSample) : ℕ :=
  s.first_number + (s.second_number - s.first_number) * ((s.total_students - s.first_number) / (s.second_number - s.first_number))

/-- Theorem stating the largest number in the systematic sample -/
theorem largest_number_is_4968 (s : SystematicSample) 
  (h1 : s.first_number = 18) 
  (h2 : s.second_number = 68) : 
  largest_number s = 4968 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_is_4968_l4093_409329


namespace NUMINAMATH_CALUDE_circle_radius_problem_l4093_409348

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if two circles are congruent -/
def are_congruent (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

/-- Checks if a point is on a circle -/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

theorem circle_radius_problem (A B C D : Circle) : 
  are_externally_tangent A B ∧ 
  are_externally_tangent B C ∧ 
  are_externally_tangent A C ∧
  is_internally_tangent A D ∧
  is_internally_tangent B D ∧
  is_internally_tangent C D ∧
  are_congruent B C ∧
  A.radius = 2 ∧
  point_on_circle D.center A →
  B.radius = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l4093_409348


namespace NUMINAMATH_CALUDE_inequality_domain_l4093_409332

theorem inequality_domain (x : ℝ) : 
  (4 * x^2 / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9) ↔ 
  (x ≥ -1/2 ∧ x < 0) ∨ (x > 0 ∧ x < 45/8) :=
sorry

end NUMINAMATH_CALUDE_inequality_domain_l4093_409332


namespace NUMINAMATH_CALUDE_evaluate_expression_l4093_409338

theorem evaluate_expression (a b c : ℚ) 
  (ha : a = 1/2) (hb : b = 1/4) (hc : c = 5) : 
  a^2 * b^3 * c = 5/256 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4093_409338


namespace NUMINAMATH_CALUDE_division_remainder_zero_l4093_409322

theorem division_remainder_zero (dividend : ℝ) (divisor : ℝ) (quotient : ℝ) 
  (h1 : dividend = 57843.67)
  (h2 : divisor = 1242.51)
  (h3 : quotient = 46.53) :
  dividend - divisor * quotient = 0 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_zero_l4093_409322


namespace NUMINAMATH_CALUDE_soup_donation_per_person_l4093_409307

theorem soup_donation_per_person
  (num_shelters : ℕ)
  (people_per_shelter : ℕ)
  (total_cans : ℕ)
  (h1 : num_shelters = 6)
  (h2 : people_per_shelter = 30)
  (h3 : total_cans = 1800) :
  total_cans / (num_shelters * people_per_shelter) = 10 := by
sorry

end NUMINAMATH_CALUDE_soup_donation_per_person_l4093_409307


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l4093_409341

theorem cube_root_equation_solution :
  ∃ x : ℝ, (x ≠ 0 ∧ (5 - 2/x)^(1/3) = -3) → x = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l4093_409341


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_l4093_409397

/-- The surface area of a regular tetrahedron inscribed in a sphere, 
    which is itself inscribed in a cube with a surface area of 54 square meters. -/
theorem tetrahedron_surface_area : ℝ := by
  -- Define the surface area of the cube
  let cube_surface_area : ℝ := 54

  -- Define the relationship between the cube and the inscribed sphere
  let sphere_inscribed_in_cube : Prop := sorry

  -- Define the relationship between the sphere and the inscribed tetrahedron
  let tetrahedron_inscribed_in_sphere : Prop := sorry

  -- State that the surface area of the inscribed regular tetrahedron is 12√3
  have h : ∃ (area : ℝ), area = 12 * Real.sqrt 3 := sorry

  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_tetrahedron_surface_area_l4093_409397


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l4093_409314

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 16 * y = x * y) :
  x + y ≥ 25 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 16 * y₀ = x₀ * y₀ ∧ x₀ + y₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l4093_409314


namespace NUMINAMATH_CALUDE_sum_of_distance_and_reciprocal_l4093_409382

theorem sum_of_distance_and_reciprocal (a b : ℝ) : 
  (|a| = 5 ∧ b = (-1/3)⁻¹) → (a + b = 2 ∨ a + b = -8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_distance_and_reciprocal_l4093_409382


namespace NUMINAMATH_CALUDE_hexagon_area_2016_l4093_409320

/-- The area of the hexagon formed by constructing squares on the sides of a right triangle -/
def hexagon_area (a b : ℕ) : ℕ := 2 * (a^2 + b^2 + a*b)

/-- The proposition to be proved -/
theorem hexagon_area_2016 :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ hexagon_area a b = 2016 ∧
  (∀ (x y : ℕ), x > 0 → y > 0 → hexagon_area x y = 2016 → (x = 12 ∧ y = 24) ∨ (x = 24 ∧ y = 12)) :=
sorry

end NUMINAMATH_CALUDE_hexagon_area_2016_l4093_409320


namespace NUMINAMATH_CALUDE_driver_speed_ratio_l4093_409371

/-- Two drivers meet halfway between cities A and B. The first driver left earlier
    than the second driver by an amount of time equal to half the time it would have
    taken them to meet if they had left simultaneously. This theorem proves the ratio
    of their speeds. -/
theorem driver_speed_ratio
  (x : ℝ)  -- Distance between cities A and B
  (v₁ v₂ : ℝ)  -- Speeds of the first and second driver respectively
  (h₁ : v₁ > 0)  -- First driver's speed is positive
  (h₂ : v₂ > 0)  -- Second driver's speed is positive
  (h₃ : x > 0)  -- Distance between cities is positive
  (h₄ : x / (2 * v₁) = x / (2 * v₂) + x / (2 * (v₁ + v₂)))  -- Meeting condition
  : v₂ / v₁ = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_driver_speed_ratio_l4093_409371


namespace NUMINAMATH_CALUDE_ellipse_equation_l4093_409358

/-- An ellipse with given properties has the equation x²/4 + y²/2 = 1 -/
theorem ellipse_equation (e : Set (ℝ × ℝ)) : 
  (∀ p ∈ e, p.1^2 / 4 + p.2^2 / 2 = 1) ↔
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 = b^2 + 2 ∧
    (∀ x y : ℝ, (x, y) ∈ e ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    (∃ x₁ x₂ y₁ y₂ : ℝ, 
      (x₁, y₁) ∈ e ∧ (x₂, y₂) ∈ e ∧
      y₁ = x₁ + 1 ∧ y₂ = x₂ + 1 ∧
      (x₁ + x₂) / 2 = -2/3)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4093_409358


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l4093_409319

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l4093_409319


namespace NUMINAMATH_CALUDE_tangent_slope_at_pi_over_four_l4093_409302

theorem tangent_slope_at_pi_over_four :
  let f (x : ℝ) := Real.tan x
  (deriv f) (π / 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_pi_over_four_l4093_409302


namespace NUMINAMATH_CALUDE_data_transformation_theorem_l4093_409350

/-- Represents a set of numerical data -/
structure DataSet where
  values : List ℝ

/-- Calculates the average of a DataSet -/
def average (d : DataSet) : ℝ := sorry

/-- Calculates the variance of a DataSet -/
def variance (d : DataSet) : ℝ := sorry

/-- Transforms a DataSet by subtracting a constant from each value -/
def transform (d : DataSet) (c : ℝ) : DataSet := sorry

theorem data_transformation_theorem (original : DataSet) :
  let transformed := transform original 80
  average transformed = 1.2 →
  variance transformed = 4.4 →
  average original = 81.2 ∧ variance original = 4.4 := by sorry

end NUMINAMATH_CALUDE_data_transformation_theorem_l4093_409350


namespace NUMINAMATH_CALUDE_import_tax_threshold_l4093_409310

/-- The amount in excess of which the import tax was applied -/
def X : ℝ := 1000

/-- The total value of the item -/
def total_value : ℝ := 2580

/-- The import tax rate -/
def tax_rate : ℝ := 0.07

/-- The amount of import tax paid -/
def tax_paid : ℝ := 110.60

/-- Theorem stating that X is the correct amount in excess of which the import tax was applied -/
theorem import_tax_threshold (X total_value tax_rate tax_paid : ℝ) 
  (h1 : total_value = 2580)
  (h2 : tax_rate = 0.07)
  (h3 : tax_paid = 110.60) :
  X = 1000 ∧ tax_rate * (total_value - X) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_import_tax_threshold_l4093_409310


namespace NUMINAMATH_CALUDE_prob_log_inequality_l4093_409321

open Real MeasureTheory ProbabilityTheory

/-- The probability of selecting a number x from [0,3] such that -1 ≤ log_(1/2)(x + 1/2) ≤ 1 is 1/2 -/
theorem prob_log_inequality (μ : Measure ℝ) [IsProbabilityMeasure μ] : 
  μ {x ∈ Set.Icc 0 3 | -1 ≤ log (x + 1/2) / log (1/2) ∧ log (x + 1/2) / log (1/2) ≤ 1} = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_prob_log_inequality_l4093_409321


namespace NUMINAMATH_CALUDE_desiree_age_proof_l4093_409392

/-- Desiree's current age -/
def desiree_age : ℕ := 6

/-- Desiree's cousin's current age -/
def cousin_age : ℕ := 3

/-- Proves that Desiree's current age is 6 years old -/
theorem desiree_age_proof :
  (desiree_age = 2 * cousin_age) ∧
  (desiree_age + 30 = (2/3 : ℚ) * (cousin_age + 30) + 14) →
  desiree_age = 6 := by
sorry

end NUMINAMATH_CALUDE_desiree_age_proof_l4093_409392


namespace NUMINAMATH_CALUDE_cone_height_increase_l4093_409396

theorem cone_height_increase (h r : ℝ) (h' : ℝ) : 
  h > 0 → r > 0 → 
  ((1/3) * Real.pi * r^2 * h') = 2.9 * ((1/3) * Real.pi * r^2 * h) → 
  (h' - h) / h = 1.9 := by
sorry

end NUMINAMATH_CALUDE_cone_height_increase_l4093_409396


namespace NUMINAMATH_CALUDE_bullying_instances_count_l4093_409345

-- Define the given constants
def suspension_days_per_instance : ℕ := 3
def typical_person_digits : ℕ := 20
def suspension_multiplier : ℕ := 3

-- Define the total suspension days
def total_suspension_days : ℕ := suspension_multiplier * typical_person_digits

-- Define the number of bullying instances
def bullying_instances : ℕ := total_suspension_days / suspension_days_per_instance

-- Theorem statement
theorem bullying_instances_count : bullying_instances = 20 := by
  sorry

end NUMINAMATH_CALUDE_bullying_instances_count_l4093_409345


namespace NUMINAMATH_CALUDE_complex_power_equality_l4093_409378

theorem complex_power_equality : (1 - Complex.I) ^ (2 * Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_equality_l4093_409378


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l4093_409351

theorem cubic_roots_sum_of_cubes (p q s r₁ r₂ : ℝ) : 
  (∀ x, x^3 - p*x^2 + q*x - s = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = 0) →
  r₁^3 + r₂^3 = p^3 - 3*q*p :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l4093_409351


namespace NUMINAMATH_CALUDE_complex_equation_result_l4093_409363

theorem complex_equation_result (m n : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : m / (1 + i) = 1 - n * i) : 
  m - n = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_result_l4093_409363


namespace NUMINAMATH_CALUDE_complex_square_simplification_l4093_409344

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 + 3 * i)^2 = 7 + 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l4093_409344


namespace NUMINAMATH_CALUDE_rogers_crayons_l4093_409389

theorem rogers_crayons (new_crayons used_crayons broken_crayons : ℕ) 
  (h1 : new_crayons = 2)
  (h2 : used_crayons = 4)
  (h3 : broken_crayons = 8) :
  new_crayons + used_crayons + broken_crayons = 14 := by
  sorry

end NUMINAMATH_CALUDE_rogers_crayons_l4093_409389


namespace NUMINAMATH_CALUDE_abs_m_minus_one_geq_abs_m_minus_one_l4093_409330

theorem abs_m_minus_one_geq_abs_m_minus_one (m : ℝ) : |m - 1| ≥ |m| - 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_m_minus_one_geq_abs_m_minus_one_l4093_409330


namespace NUMINAMATH_CALUDE_perfect_square_fraction_l4093_409388

theorem perfect_square_fraction (a b : ℕ+) (k : ℕ) 
  (h : k = (a.val^2 + b.val^2) / (a.val * b.val + 1)) : 
  ∃ (n : ℕ), k = n^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_fraction_l4093_409388


namespace NUMINAMATH_CALUDE_nancy_math_problems_l4093_409374

/-- The number of math problems Nancy had to solve -/
def math_problems : ℝ := 17.0

/-- The number of spelling problems Nancy had to solve -/
def spelling_problems : ℝ := 15.0

/-- The number of problems Nancy can finish in an hour -/
def problems_per_hour : ℝ := 8.0

/-- The number of hours it took Nancy to finish all problems -/
def total_hours : ℝ := 4.0

/-- Theorem stating that the number of math problems Nancy had is 17.0 -/
theorem nancy_math_problems : 
  math_problems = 
    problems_per_hour * total_hours - spelling_problems :=
by sorry

end NUMINAMATH_CALUDE_nancy_math_problems_l4093_409374


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l4093_409304

theorem min_value_sum_of_reciprocals (a b c d e f g : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) 
  (pos_e : 0 < e) (pos_f : 0 < f) (pos_g : 0 < g)
  (sum_eq_8 : a + b + c + d + e + f + g = 8) :
  1/a + 4/b + 9/c + 16/d + 25/e + 36/f + 49/g ≥ 98 ∧ 
  ∃ (a' b' c' d' e' f' g' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧ 0 < g' ∧
    a' + b' + c' + d' + e' + f' + g' = 8 ∧
    1/a' + 4/b' + 9/c' + 16/d' + 25/e' + 36/f' + 49/g' = 98 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l4093_409304


namespace NUMINAMATH_CALUDE_min_value_theorem_l4093_409362

open Real

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 1/x + 4/y ≤ 1/a + 4/b) ∧ (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1/a + 4/b = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4093_409362


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l4093_409337

/-- Proves that the cost of an adult ticket is $9, given the conditions of the problem -/
theorem adult_ticket_cost (child_ticket_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (children_tickets : ℕ) :
  child_ticket_cost = 6 →
  total_tickets = 225 →
  total_revenue = 1875 →
  children_tickets = 50 →
  (total_revenue - child_ticket_cost * children_tickets) / (total_tickets - children_tickets) = 9 := by
sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l4093_409337


namespace NUMINAMATH_CALUDE_negation_equivalence_l4093_409375

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (student : U → Prop)
variable (shares_truth : U → Prop)

-- Define the original statement
def every_student_shares_truth : Prop :=
  ∀ x, student x → shares_truth x

-- Define the negation
def negation_statement : Prop :=
  ∃ x, student x ∧ ¬(shares_truth x)

-- Theorem to prove
theorem negation_equivalence :
  ¬(every_student_shares_truth U student shares_truth) ↔ negation_statement U student shares_truth :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4093_409375


namespace NUMINAMATH_CALUDE_test_questions_missed_l4093_409343

theorem test_questions_missed (friend_missed : ℕ) (your_missed : ℕ) : 
  your_missed = 5 * friend_missed →
  your_missed + friend_missed = 216 →
  your_missed = 180 := by
sorry

end NUMINAMATH_CALUDE_test_questions_missed_l4093_409343


namespace NUMINAMATH_CALUDE_walkers_commute_l4093_409311

/-- Ms. Walker's commute problem -/
theorem walkers_commute
  (speed_to_work : ℝ)
  (speed_from_work : ℝ)
  (total_time : ℝ)
  (h1 : speed_to_work = 60)
  (h2 : speed_from_work = 40)
  (h3 : total_time = 1) :
  ∃ (distance : ℝ), 
    distance / speed_to_work + distance / speed_from_work = total_time ∧ 
    distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_walkers_commute_l4093_409311


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_l4093_409359

/-- The number of tiles in the linear arrangement -/
def num_tiles : ℕ := 6

/-- The number of blue tiles -/
def num_blue : ℕ := 4

/-- The number of red tiles -/
def num_red : ℕ := 1

/-- The number of green tiles -/
def num_green : ℕ := 1

/-- A function that calculates the number of distinct linear arrangements
    of tiles, considering end-to-end reflection as identical arrangements -/
def count_distinct_arrangements (n : ℕ) (b r g : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of distinct arrangements is 15 -/
theorem distinct_arrangements_count :
  count_distinct_arrangements num_tiles num_blue num_red num_green = 15 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_l4093_409359


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l4093_409387

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0 ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ x₁ x₂ : ℝ, (x₁+1)^2 - 144 = 0 ∧ (x₂+1)^2 - 144 = 0 ∧ x₁ = 11 ∧ x₂ = -13) ∧
  (∃ x₁ x₂ : ℝ, 3*(x₁-2)^2 = x₁*(x₁-2) ∧ 3*(x₂-2)^2 = x₂*(x₂-2) ∧ x₁ = 2 ∧ x₂ = 3) ∧
  (∃ x₁ x₂ : ℝ, x₁^2 + 5*x₁ - 1 = 0 ∧ x₂^2 + 5*x₂ - 1 = 0 ∧ 
    x₁ = (-5 + Real.sqrt 29) / 2 ∧ x₂ = (-5 - Real.sqrt 29) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l4093_409387


namespace NUMINAMATH_CALUDE_f_inequality_solution_set_f_inequality_a_range_l4093_409327

def f (x : ℝ) : ℝ := |2*x - 2| - |x + 1|

theorem f_inequality_solution_set :
  {x : ℝ | f x ≤ 3} = {x : ℝ | -2/3 ≤ x ∧ x ≤ 6} := by sorry

theorem f_inequality_a_range :
  {a : ℝ | ∀ x, f x ≤ |x + 1| + a^2} = {a : ℝ | a ≤ -2 ∨ 2 ≤ a} := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_set_f_inequality_a_range_l4093_409327


namespace NUMINAMATH_CALUDE_negative_product_cube_squared_l4093_409383

theorem negative_product_cube_squared (a b : ℝ) : (-a * b^3)^2 = a^2 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_cube_squared_l4093_409383


namespace NUMINAMATH_CALUDE_min_stamps_for_47_cents_l4093_409377

/-- Represents the number of stamps of each denomination -/
structure StampCombination where
  threes : ℕ
  fours : ℕ
  fives : ℕ

/-- Calculates the total value of stamps in cents -/
def total_value (sc : StampCombination) : ℕ :=
  3 * sc.threes + 4 * sc.fours + 5 * sc.fives

/-- Calculates the total number of stamps -/
def total_stamps (sc : StampCombination) : ℕ :=
  sc.threes + sc.fours + sc.fives

/-- Checks if at least two types of stamps are used -/
def uses_at_least_two_types (sc : StampCombination) : Prop :=
  (sc.threes > 0 && sc.fours > 0) || (sc.threes > 0 && sc.fives > 0) || (sc.fours > 0 && sc.fives > 0)

/-- States that 10 is the minimum number of stamps needed to make 47 cents -/
theorem min_stamps_for_47_cents :
  ∃ (sc : StampCombination),
    total_value sc = 47 ∧
    uses_at_least_two_types sc ∧
    total_stamps sc = 10 ∧
    (∀ (sc' : StampCombination),
      total_value sc' = 47 →
      uses_at_least_two_types sc' →
      total_stamps sc' ≥ 10) :=
  sorry

end NUMINAMATH_CALUDE_min_stamps_for_47_cents_l4093_409377


namespace NUMINAMATH_CALUDE_exists_divisible_figure_l4093_409380

/-- A non-rectangular grid figure composed of cells -/
structure GridFigure where
  cells : ℕ
  nonRectangular : Bool

/-- Represents the ability to divide a figure into equal parts -/
def isDivisible (f : GridFigure) (n : ℕ) : Prop :=
  ∃ (k : ℕ), f.cells = n * k

/-- The main theorem stating the existence of a figure divisible by 2 to 7 -/
theorem exists_divisible_figure :
  ∃ (f : GridFigure), f.nonRectangular ∧
    (∀ n : ℕ, 2 ≤ n ∧ n ≤ 7 → isDivisible f n) :=
  sorry

end NUMINAMATH_CALUDE_exists_divisible_figure_l4093_409380


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l4093_409366

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l4093_409366


namespace NUMINAMATH_CALUDE_complement_of_B_in_A_l4093_409399

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 3, 5}

theorem complement_of_B_in_A : (A \ B) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_in_A_l4093_409399


namespace NUMINAMATH_CALUDE_root_property_l4093_409328

theorem root_property (a : ℝ) (h : a^2 + a - 2009 = 0) : a^2 + a - 1 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l4093_409328


namespace NUMINAMATH_CALUDE_cubic_inequality_l4093_409347

theorem cubic_inequality (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l4093_409347


namespace NUMINAMATH_CALUDE_eve_walking_distance_l4093_409379

theorem eve_walking_distance (ran_distance : Real) (extra_distance : Real) :
  ran_distance = 0.7 ∧ extra_distance = 0.1 →
  ∃ walked_distance : Real, walked_distance = ran_distance - extra_distance ∧ walked_distance = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_eve_walking_distance_l4093_409379


namespace NUMINAMATH_CALUDE_min_value_implies_a_f_less_than_x_squared_l4093_409340

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) :=
sorry

theorem f_less_than_x_squared (a : ℝ) :
  (∀ x > 1, f a x < x^2) →
  a ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_f_less_than_x_squared_l4093_409340


namespace NUMINAMATH_CALUDE_gas_pressure_change_l4093_409313

/-- Represents the pressure-volume relationship for a gas at constant temperature -/
def pressure_volume_relation (p1 p2 v1 v2 : ℝ) : Prop :=
  p1 * v1 = p2 * v2

/-- Theorem: Given inverse proportionality of pressure and volume,
    if a gas initially at 8 kPa in a 3.5-liter container is transferred to a 7-liter container,
    its new pressure will be 4 kPa -/
theorem gas_pressure_change (p2 : ℝ) :
  pressure_volume_relation 8 p2 3.5 7 → p2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gas_pressure_change_l4093_409313


namespace NUMINAMATH_CALUDE_price_change_l4093_409301

theorem price_change (P : ℝ) (h : P > 0) :
  let price_2012 := P * 1.25
  let price_2013 := price_2012 * 0.88
  (price_2013 - P) / P * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_price_change_l4093_409301


namespace NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l4093_409303

theorem belt_and_road_population_scientific_notation :
  (4600000000 : ℝ) = 4.6 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l4093_409303


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_two_l4093_409372

theorem logarithm_expression_equals_two :
  (Real.log 243 / Real.log 3) / (Real.log 3 / Real.log 81) -
  (Real.log 729 / Real.log 3) / (Real.log 3 / Real.log 27) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_two_l4093_409372


namespace NUMINAMATH_CALUDE_clock_hands_at_30_degrees_48_times_daily_l4093_409390

/-- Represents a clock with an hour hand and a minute hand -/
structure Clock where
  hour_hand : ℝ
  minute_hand : ℝ

/-- The speed of the minute hand relative to the hour hand -/
def minute_hand_speed : ℝ := 12

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The angle between clock hands we're interested in -/
def target_angle : ℝ := 30

/-- Function to count the number of times the clock hands form the target angle in a day -/
def count_target_angle_occurrences (c : Clock) : ℕ :=
  sorry

theorem clock_hands_at_30_degrees_48_times_daily :
  ∀ c : Clock, count_target_angle_occurrences c = 48 :=
sorry

end NUMINAMATH_CALUDE_clock_hands_at_30_degrees_48_times_daily_l4093_409390


namespace NUMINAMATH_CALUDE_floor_times_self_eq_90_l4093_409309

theorem floor_times_self_eq_90 (x : ℝ) (h1 : x > 0) (h2 : ⌊x⌋ * x = 90) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_90_l4093_409309


namespace NUMINAMATH_CALUDE_no_rain_probability_l4093_409317

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^4 = 1/81 := by sorry

end NUMINAMATH_CALUDE_no_rain_probability_l4093_409317


namespace NUMINAMATH_CALUDE_class_selection_ways_l4093_409323

def total_classes : ℕ := 10
def advanced_classes : ℕ := 3
def classes_to_select : ℕ := 5
def min_advanced : ℕ := 2

theorem class_selection_ways : 
  (Nat.choose advanced_classes min_advanced) * 
  (Nat.choose (total_classes - advanced_classes) (classes_to_select - min_advanced)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_class_selection_ways_l4093_409323


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l4093_409384

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- The function f(x) = x^2 - |x + a| -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 - |x + a|

theorem even_function_implies_a_zero :
  ∀ a : ℝ, IsEven (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l4093_409384


namespace NUMINAMATH_CALUDE_roots_sum_of_reciprocal_cubes_l4093_409370

theorem roots_sum_of_reciprocal_cubes (a b c : ℝ) (r s : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : c ≠ 0) 
  (h3 : a * r^2 + b * r + c = 0) 
  (h4 : a * s^2 + b * s + c = 0) 
  (h5 : r ≠ 0) 
  (h6 : s ≠ 0) : 
  1 / r^3 + 1 / s^3 = (-b^3 + 3*a*b*c) / c^3 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_reciprocal_cubes_l4093_409370


namespace NUMINAMATH_CALUDE_parabola_f_value_l4093_409391

/-- Represents a parabola of the form x = dy² + ey + f -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.d * y^2 + p.e * y + p.f

theorem parabola_f_value (p : Parabola) :
  (p.x_coord 1 = -3) →  -- vertex at (-3, 1)
  (p.x_coord 3 = -1) →  -- passes through (-1, 3)
  (p.x_coord 0 = -2.5) →  -- passes through (-2.5, 0)
  p.f = -2.5 := by
  sorry

#check parabola_f_value

end NUMINAMATH_CALUDE_parabola_f_value_l4093_409391


namespace NUMINAMATH_CALUDE_carrot_juice_distribution_l4093_409342

-- Define the set of glass volumes
def glassVolumes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define a type for a distribution of glasses
def Distribution := List (List ℕ)

-- Define the property of a valid distribution
def isValidDistribution (d : Distribution) : Prop :=
  d.length = 3 ∧
  d.all (fun l => l.length = 3) ∧
  d.all (fun l => l.sum = 15) ∧
  d.join.toFinset = glassVolumes.toFinset

-- State the theorem
theorem carrot_juice_distribution :
  ∃ (d1 d2 : Distribution),
    isValidDistribution d1 ∧
    isValidDistribution d2 ∧
    d1 ≠ d2 ∧
    ∀ (d : Distribution), isValidDistribution d → (d = d1 ∨ d = d2) :=
  sorry

end NUMINAMATH_CALUDE_carrot_juice_distribution_l4093_409342


namespace NUMINAMATH_CALUDE_tangent_slope_minimum_tangent_slope_minimum_achieved_l4093_409333

theorem tangent_slope_minimum (b : ℝ) (h : b > 0) : 
  (2 / b + b) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem tangent_slope_minimum_achieved (b : ℝ) (h : b > 0) : 
  (2 / b + b = 2 * Real.sqrt 2) ↔ (2 / b = b) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_minimum_tangent_slope_minimum_achieved_l4093_409333


namespace NUMINAMATH_CALUDE_tangent_lines_theorem_intersection_theorem_l4093_409398

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the point M
def point_M : ℝ × ℝ := (3, 1)

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 3
def tangent_line_2 (x y : ℝ) : Prop := 3*x - 4*y - 5 = 0

-- Define the family of lines
def line_family (a x y : ℝ) : Prop := a*x - y + 3 = 0

-- Theorem for tangent lines
theorem tangent_lines_theorem :
  (∀ x y : ℝ, tangent_line_1 x → circle_C x y → x = 3 ∧ y = 1 ∨ x = 3 ∧ y = 3) ∧
  (∀ x y : ℝ, tangent_line_2 x y → circle_C x y → x = 3 ∧ y = 1 ∨ x = 0 ∧ y = 5/4) :=
sorry

-- Theorem for intersection
theorem intersection_theorem :
  ∀ a : ℝ, ∃ x y : ℝ, line_family a x y ∧ circle_C x y :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_theorem_intersection_theorem_l4093_409398


namespace NUMINAMATH_CALUDE_equation_proof_l4093_409357

theorem equation_proof : 484 + 2 * 22 * 3 + 9 = 625 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l4093_409357


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l4093_409300

theorem solve_exponential_equation :
  ∃ y : ℝ, (40 : ℝ)^3 = 8^y ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l4093_409300


namespace NUMINAMATH_CALUDE_a_equals_three_iff_parallel_l4093_409356

def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y + 6 * a = 0

def parallel (a : ℝ) : Prop := ∀ (x y : ℝ), line1 a x y ↔ ∃ (k : ℝ), line2 a (x + k) (y + k)

theorem a_equals_three_iff_parallel :
  ∀ (a : ℝ), a = 3 ↔ parallel a :=
sorry

end NUMINAMATH_CALUDE_a_equals_three_iff_parallel_l4093_409356


namespace NUMINAMATH_CALUDE_find_b_value_l4093_409346

theorem find_b_value (circle_sum : ℕ) (total_sum : ℕ) (d : ℕ) :
  circle_sum = 21 * 5 ∧
  total_sum = 69 ∧
  d + 5 + 9 = 21 →
  ∃ b : ℕ, b = 10 ∧ circle_sum - (2 + 8 + 9 + b + d) = total_sum :=
by sorry

end NUMINAMATH_CALUDE_find_b_value_l4093_409346


namespace NUMINAMATH_CALUDE_third_root_unity_sum_l4093_409308

theorem third_root_unity_sum (z : ℂ) (h1 : z^3 - 1 = 0) (h2 : z ≠ 1) :
  z^100 + z^101 + z^102 + z^103 + z^104 = 0 := by
  sorry

end NUMINAMATH_CALUDE_third_root_unity_sum_l4093_409308


namespace NUMINAMATH_CALUDE_complex_division_result_l4093_409354

theorem complex_division_result : (10 * Complex.I) / (1 - 2 * Complex.I) = -4 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l4093_409354


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l4093_409368

theorem fraction_subtraction_equality : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l4093_409368


namespace NUMINAMATH_CALUDE_probability_factor_less_than_seven_l4093_409315

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem probability_factor_less_than_seven :
  let all_factors := factors 60
  let factors_less_than_seven := all_factors.filter (· < 7)
  (factors_less_than_seven.card : ℚ) / all_factors.card = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_seven_l4093_409315


namespace NUMINAMATH_CALUDE_quadratic_root_l4093_409335

/-- A quadratic polynomial with coefficients a, b, and c. -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Predicate to check if a quadratic polynomial has exactly one root. -/
def has_one_root (a b c : ℝ) : Prop := b^2 = 4 * a * c

theorem quadratic_root (a b c : ℝ) (ha : a ≠ 0) :
  has_one_root a b c →
  has_one_root (-a) (b - 30*a) (17*a - 7*b + c) →
  ∃! x : ℝ, quadratic a b c x = 0 ∧ x = -11 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_l4093_409335


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l4093_409355

theorem absolute_value_equation_solutions (z : ℝ) :
  ∃ (x y : ℝ), (|x - y^2| = z*x + y^2 ∧ z*x + y^2 ≥ 0) ↔
  ((x = 0 ∧ y = 0) ∨
   (∃ (y : ℝ), x = 2*y^2/(1-z) ∧ z ≠ 1 ∧ z > -1)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l4093_409355


namespace NUMINAMATH_CALUDE_no_real_solutions_l4093_409339

theorem no_real_solutions :
  ∀ x : ℝ, x ≠ 2 → (4 * x^3 + 3 * x^2 + x + 2) / (x - 2) ≠ 4 * x^2 + 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l4093_409339


namespace NUMINAMATH_CALUDE_sum_of_extremes_3point5_l4093_409334

/-- A number that rounds to 3.5 when rounded to one decimal place -/
def RoundsTo3Point5 (x : ℝ) : Prop :=
  (x ≥ 3.45) ∧ (x < 3.55)

/-- The theorem stating the sum of the largest and smallest 3-digit decimals
    that round to 3.5 is 6.99 -/
theorem sum_of_extremes_3point5 :
  ∃ (min max : ℝ),
    (∀ x, RoundsTo3Point5 x → x ≥ min) ∧
    (∀ x, RoundsTo3Point5 x → x ≤ max) ∧
    (RoundsTo3Point5 min) ∧
    (RoundsTo3Point5 max) ∧
    (min + max = 6.99) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extremes_3point5_l4093_409334


namespace NUMINAMATH_CALUDE_slowest_pump_time_l4093_409352

/-- Three pumps with rates in ratio 2:3:4 fill a pool in 6 hours. The slowest pump fills it in 27 hours. -/
theorem slowest_pump_time (pool_volume : ℝ) (h : pool_volume > 0) : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧  -- Pump rates are positive
    r₂ = (3/2) * r₁ ∧           -- Ratio of rates
    r₃ = 2 * r₁ ∧               -- Ratio of rates
    (r₁ + r₂ + r₃) * 6 = pool_volume ∧  -- All pumps fill the pool in 6 hours
    r₁ * 27 = pool_volume       -- Slowest pump fills the pool in 27 hours
  := by sorry

end NUMINAMATH_CALUDE_slowest_pump_time_l4093_409352


namespace NUMINAMATH_CALUDE_area_BCD_equals_135_l4093_409367

-- Define the triangle ABC
def triangle_ABC : Set (ℝ × ℝ) := sorry

-- Define the triangle BCD
def triangle_BCD : Set (ℝ × ℝ) := sorry

-- Define the area function
def area : Set (ℝ × ℝ) → ℝ := sorry

-- Define the length function
def length : ℝ × ℝ → ℝ × ℝ → ℝ := sorry

-- Define point A
def A : ℝ × ℝ := sorry

-- Define point C
def C : ℝ × ℝ := sorry

-- Define point D
def D : ℝ × ℝ := sorry

-- Theorem statement
theorem area_BCD_equals_135 :
  area triangle_ABC = 36 →
  length A C = 8 →
  length C D = 30 →
  area triangle_BCD = 135 := by
  sorry

end NUMINAMATH_CALUDE_area_BCD_equals_135_l4093_409367


namespace NUMINAMATH_CALUDE_bert_stamps_correct_l4093_409373

/-- The number of stamps Bert bought -/
def stamps_bought : ℕ := 300

/-- The number of stamps Bert had before the purchase -/
def stamps_before : ℕ := stamps_bought / 2

/-- The total number of stamps Bert has after the purchase -/
def total_stamps : ℕ := 450

/-- Theorem stating that the number of stamps Bert bought is correct -/
theorem bert_stamps_correct :
  stamps_bought = 300 ∧
  stamps_before = stamps_bought / 2 ∧
  total_stamps = stamps_before + stamps_bought :=
by sorry

end NUMINAMATH_CALUDE_bert_stamps_correct_l4093_409373


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4093_409361

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) :
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4093_409361


namespace NUMINAMATH_CALUDE_nathan_nickels_l4093_409312

theorem nathan_nickels (n : ℕ) : 
  20 < n ∧ n < 200 ∧ 
  n % 4 = 2 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 2 → 
  n = 142 := by sorry

end NUMINAMATH_CALUDE_nathan_nickels_l4093_409312


namespace NUMINAMATH_CALUDE_remainder_2519_div_8_l4093_409369

theorem remainder_2519_div_8 : 2519 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_8_l4093_409369


namespace NUMINAMATH_CALUDE_min_value_polynomial_l4093_409336

theorem min_value_polynomial (x y : ℝ) : 
  ∀ a b : ℝ, 5 * a^2 - 4 * a * b + 4 * b^2 + 12 * a + 25 ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_polynomial_l4093_409336


namespace NUMINAMATH_CALUDE_BC_time_proof_l4093_409360

-- Define work rates for A, B, and C
def A_rate : ℚ := 1 / 4
def B_rate : ℚ := 1 / 12
def AC_rate : ℚ := 1 / 2

-- Define the time taken by B and C together
def BC_time : ℚ := 3

-- Theorem statement
theorem BC_time_proof :
  let C_rate : ℚ := AC_rate - A_rate
  let BC_rate : ℚ := B_rate + C_rate
  BC_time = 1 / BC_rate :=
by sorry

end NUMINAMATH_CALUDE_BC_time_proof_l4093_409360


namespace NUMINAMATH_CALUDE_remaining_time_indeterminate_l4093_409353

/-- Represents the state of a math test -/
structure MathTest where
  totalProblems : ℕ
  firstInterval : ℕ
  secondInterval : ℕ
  problemsCompletedFirst : ℕ
  problemsCompletedSecond : ℕ
  problemsLeft : ℕ

/-- Theorem stating that the remaining time cannot be determined -/
theorem remaining_time_indeterminate (test : MathTest) 
  (h1 : test.totalProblems = 75)
  (h2 : test.firstInterval = 20)
  (h3 : test.secondInterval = 20)
  (h4 : test.problemsCompletedFirst = 10)
  (h5 : test.problemsCompletedSecond = 2 * test.problemsCompletedFirst)
  (h6 : test.problemsLeft = 45)
  (h7 : test.totalProblems = test.problemsCompletedFirst + test.problemsCompletedSecond + test.problemsLeft) :
  ¬∃ (remainingTime : ℕ), True := by
  sorry

#check remaining_time_indeterminate

end NUMINAMATH_CALUDE_remaining_time_indeterminate_l4093_409353


namespace NUMINAMATH_CALUDE_investment_distribution_l4093_409381

def total_investment : ℝ := 1500
def final_amount : ℝ := 1800
def years : ℕ := 3

def interest_rate_trusty : ℝ := 0.04
def interest_rate_solid : ℝ := 0.06
def interest_rate_quick : ℝ := 0.07

def compound_factor (rate : ℝ) (years : ℕ) : ℝ :=
  (1 + rate) ^ years

theorem investment_distribution (x y : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ total_investment →
  x * compound_factor interest_rate_trusty years +
  y * compound_factor interest_rate_solid years +
  (total_investment - x - y) * compound_factor interest_rate_quick years = final_amount →
  x = 375 := by sorry

end NUMINAMATH_CALUDE_investment_distribution_l4093_409381


namespace NUMINAMATH_CALUDE_chord_length_line_circle_specific_chord_length_l4093_409325

/-- The length of the chord cut off by a line on a circle -/
theorem chord_length_line_circle (a b c d e f : ℝ) (h1 : a ≠ 0 ∨ b ≠ 0) :
  let line := {(x, y) : ℝ × ℝ | a * x + b * y = c}
  let circle := {(x, y) : ℝ × ℝ | (x - d)^2 + (y - e)^2 = f^2}
  let center := (d, e)
  let radius := f
  let dist_center_to_line := |a * d + b * e - c| / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (radius^2 - dist_center_to_line^2) = 6 * Real.sqrt 11 / 5 :=
by sorry

/-- The specific case for the given problem -/
theorem specific_chord_length :
  let line := {(x, y) : ℝ × ℝ | 3 * x + 4 * y = 7}
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4}
  let center := (2, 0)
  let radius := 2
  let dist_center_to_line := |3 * 2 + 4 * 0 - 7| / Real.sqrt (3^2 + 4^2)
  2 * Real.sqrt (radius^2 - dist_center_to_line^2) = 6 * Real.sqrt 11 / 5 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_line_circle_specific_chord_length_l4093_409325


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l4093_409395

/-- An arithmetic sequence with given second and fifth terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  second_term : a 2 = 2
  fifth_term : a 5 = 5

/-- The general term of the arithmetic sequence is n -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l4093_409395


namespace NUMINAMATH_CALUDE_price_increase_percentage_l4093_409365

theorem price_increase_percentage (old_price new_price : ℝ) (h1 : old_price = 300) (h2 : new_price = 330) :
  (new_price - old_price) / old_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l4093_409365


namespace NUMINAMATH_CALUDE_intersection_A_B_l4093_409318

-- Define set A
def A : Set ℝ := {x | 1 < x - 1 ∧ x - 1 ≤ 3}

-- Define set B
def B : Set ℝ := {2, 3, 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l4093_409318


namespace NUMINAMATH_CALUDE_tree_height_problem_l4093_409394

theorem tree_height_problem (h1 h2 : ℝ) : 
  h1 = h2 + 20 →  -- One tree is 20 feet taller than the other
  h2 / h1 = 5 / 7 →  -- The heights are in the ratio 5:7
  h1 = 70 := by  -- The height of the taller tree is 70 feet
sorry

end NUMINAMATH_CALUDE_tree_height_problem_l4093_409394


namespace NUMINAMATH_CALUDE_remainder_h_x10_div_h_x_l4093_409385

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Define the theorem
theorem remainder_h_x10_div_h_x :
  ∃ (q : ℝ → ℝ), h (x^10) = h x * q x + 6 :=
sorry

end NUMINAMATH_CALUDE_remainder_h_x10_div_h_x_l4093_409385


namespace NUMINAMATH_CALUDE_compute_expression_l4093_409349

theorem compute_expression : 10 + 4 * (5 + 3)^3 = 2058 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l4093_409349


namespace NUMINAMATH_CALUDE_cards_distribution_l4093_409386

/-- 
Given 52 cards dealt to 8 people as evenly as possible, 
this theorem proves that 4 people will have fewer than 7 cards.
-/
theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 52)
  (h2 : num_people = 8) :
  (num_people - (total_cards % num_people)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l4093_409386


namespace NUMINAMATH_CALUDE_boys_can_be_truthful_l4093_409316

/-- Represents the possible grades a student can receive -/
inductive Grade
  | Three
  | Four
  | Five

/-- Compares two grades -/
def Grade.gt (a b : Grade) : Prop :=
  match a, b with
  | Five, Three => True
  | Five, Four => True
  | Four, Three => True
  | _, _ => False

/-- Represents the grades of a student for three tests -/
structure StudentGrades :=
  (test1 : Grade)
  (test2 : Grade)
  (test3 : Grade)

/-- Checks if one student has higher grades than another for at least two tests -/
def higherGradesInTwoTests (a b : StudentGrades) : Prop :=
  (a.test1.gt b.test1 ∧ a.test2.gt b.test2) ∨
  (a.test1.gt b.test1 ∧ a.test3.gt b.test3) ∨
  (a.test2.gt b.test2 ∧ a.test3.gt b.test3)

/-- The main theorem stating that there exists a set of grades satisfying all conditions -/
theorem boys_can_be_truthful :
  ∃ (valera seryozha dima : StudentGrades),
    higherGradesInTwoTests valera seryozha ∧
    higherGradesInTwoTests seryozha dima ∧
    higherGradesInTwoTests dima valera :=
  sorry

end NUMINAMATH_CALUDE_boys_can_be_truthful_l4093_409316


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4093_409331

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, Real.sqrt (x + 2) - Real.sqrt (1 - 2*x) > 0 → (x + 1) / (x - 1) ≤ 0) ∧
  (∃ x, (x + 1) / (x - 1) ≤ 0 ∧ Real.sqrt (x + 2) - Real.sqrt (1 - 2*x) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4093_409331


namespace NUMINAMATH_CALUDE_sum_three_digit_integers_mod_1000_l4093_409376

def sum_three_digit_integers : ℕ :=
  (45 * 100 * 100) + (45 * 100 * 10) + (45 * 100)

theorem sum_three_digit_integers_mod_1000 :
  sum_three_digit_integers % 1000 = 500 := by sorry

end NUMINAMATH_CALUDE_sum_three_digit_integers_mod_1000_l4093_409376
