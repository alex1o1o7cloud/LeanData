import Mathlib

namespace NUMINAMATH_CALUDE_modular_arithmetic_proof_l2903_290387

theorem modular_arithmetic_proof :
  ∃ (a b : ℤ), (a * 7 ≡ 1 [ZMOD 63]) ∧ 
               (b * 13 ≡ 1 [ZMOD 63]) ∧ 
               ((3 * a + 5 * b) % 63 = 13) := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_proof_l2903_290387


namespace NUMINAMATH_CALUDE_least_trees_required_l2903_290312

theorem least_trees_required (n : ℕ) : 
  (n > 0 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n) → 
  (∀ m : ℕ, m > 0 ∧ 4 ∣ m ∧ 5 ∣ m ∧ 6 ∣ m → n ≤ m) → 
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_least_trees_required_l2903_290312


namespace NUMINAMATH_CALUDE_sqrt_sum_geq_product_sum_l2903_290302

theorem sqrt_sum_geq_product_sum {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : Real.sqrt x + Real.sqrt y + Real.sqrt z ≥ x * y + y * z + z * x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_geq_product_sum_l2903_290302


namespace NUMINAMATH_CALUDE_carbonate_weight_in_al2co3_l2903_290310

/-- The molecular weight of the carbonate part in Al2(CO3)3 -/
def carbonate_weight (total_weight : ℝ) (al_weight : ℝ) : ℝ :=
  total_weight - 2 * al_weight

/-- Theorem: The molecular weight of the carbonate part in Al2(CO3)3 is 180.04 g/mol -/
theorem carbonate_weight_in_al2co3 :
  carbonate_weight 234 26.98 = 180.04 := by
  sorry

end NUMINAMATH_CALUDE_carbonate_weight_in_al2co3_l2903_290310


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_l2903_290349

theorem consecutive_odd_sum (n : ℤ) : 
  (n + 2 = 9) → (n + (n + 2) + (n + 4) = n + 20) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_l2903_290349


namespace NUMINAMATH_CALUDE_mean_median_difference_l2903_290383

/-- Represents the frequency distribution of missed school days -/
def frequency_distribution : List (Nat × Nat) := [
  (0, 2), (1, 5), (2, 1), (3, 3), (4, 2), (5, 4), (6, 1), (7, 2)
]

/-- Total number of students -/
def total_students : Nat := 20

/-- Calculates the median number of days missed -/
def median (dist : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- Calculates the mean number of days missed -/
def mean (dist : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem mean_median_difference :
  mean frequency_distribution total_students - median frequency_distribution total_students = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2903_290383


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l2903_290391

theorem purely_imaginary_condition (θ : ℝ) : 
  (∃ (y : ℝ), Complex.mk (Real.sin (2 * θ) - 1) (Real.sqrt 2 * Real.cos θ + 1) = Complex.I * y) ↔ 
  (∃ (k : ℤ), θ = 2 * k * Real.pi + Real.pi / 4) :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l2903_290391


namespace NUMINAMATH_CALUDE_apartment_building_occupancy_l2903_290306

theorem apartment_building_occupancy :
  let total_floors : ℕ := 12
  let full_floors : ℕ := total_floors / 2
  let half_capacity_floors : ℕ := total_floors - full_floors
  let apartments_per_floor : ℕ := 10
  let people_per_apartment : ℕ := 4
  let people_per_full_floor : ℕ := apartments_per_floor * people_per_apartment
  let people_per_half_floor : ℕ := people_per_full_floor / 2
  let total_people : ℕ := full_floors * people_per_full_floor + half_capacity_floors * people_per_half_floor
  total_people = 360 := by
  sorry

end NUMINAMATH_CALUDE_apartment_building_occupancy_l2903_290306


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l2903_290394

/-- The probability of getting exactly k successes in n independent Bernoulli trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of flipping exactly 3 heads in 8 flips of an unfair coin -/
theorem unfair_coin_probability : 
  binomial_probability 8 3 (1/3) = 1792/6561 := by
  sorry


end NUMINAMATH_CALUDE_unfair_coin_probability_l2903_290394


namespace NUMINAMATH_CALUDE_cooper_pies_per_day_l2903_290341

/-- The number of days Cooper makes pies -/
def days : ℕ := 12

/-- The number of pies Ashley eats -/
def pies_eaten : ℕ := 50

/-- The number of pies remaining -/
def pies_remaining : ℕ := 34

/-- The number of pies Cooper makes per day -/
def pies_per_day : ℕ := 7

theorem cooper_pies_per_day :
  days * pies_per_day - pies_eaten = pies_remaining :=
by sorry

end NUMINAMATH_CALUDE_cooper_pies_per_day_l2903_290341


namespace NUMINAMATH_CALUDE_distance_calculation_l2903_290307

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 24

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time difference between Maxwell's and Brad's start times in hours -/
def time_difference : ℝ := 1

/-- Total time Maxwell walks before meeting Brad in hours -/
def total_time : ℝ := 3

theorem distance_calculation :
  distance_between_homes = maxwell_speed * total_time + brad_speed * (total_time - time_difference) :=
by sorry

end NUMINAMATH_CALUDE_distance_calculation_l2903_290307


namespace NUMINAMATH_CALUDE_distinct_exponentiation_values_l2903_290345

-- Define a function to represent different parenthesizations of 3^3^3^3
def exponentiation_order (n : Nat) : Nat :=
  match n with
  | 0 => 3^(3^(3^3))  -- standard order
  | 1 => 3^((3^3)^3)
  | 2 => (3^3)^(3^3)
  | 3 => (3^(3^3))^3
  | _ => ((3^3)^3)^3

-- Theorem statement
theorem distinct_exponentiation_values :
  ∃ (S : Finset Nat), (Finset.card S = 5) ∧ 
  (∀ (i : Nat), i < 5 → exponentiation_order i ∈ S) ∧
  (∀ (x : Nat), x ∈ S → ∃ (i : Nat), i < 5 ∧ exponentiation_order i = x) :=
sorry

end NUMINAMATH_CALUDE_distinct_exponentiation_values_l2903_290345


namespace NUMINAMATH_CALUDE_equation_solution_range_l2903_290342

theorem equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (x + m) / (x - 3) + (3 * m) / (3 - x) = 3) →
  m < 9 / 2 ∧ m ≠ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2903_290342


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2903_290377

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_3, and a_4 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  (a 3 / a 1) ^ 2 = a 4 / a 1

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) (h_geom : geometric_subseq a) : 
  a 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2903_290377


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l2903_290322

-- Define the types for our variables
variables {a b c : ℝ}

-- Define the solution set of the first inequality
def solution_set_1 : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := {x | x ≤ -1 ∨ x ≥ -1/2}

-- State the theorem
theorem inequality_solution_sets :
  (∀ x, ax^2 - b*x + c ≥ 0 ↔ x ∈ solution_set_1) →
  (∀ x, c*x^2 + b*x + a ≤ 0 ↔ x ∈ solution_set_2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l2903_290322


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l2903_290327

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y - 1 = k * (x + 3)

-- Define the point on the parabola
def point_on_parabola (a : ℝ) : Prop := parabola 3 a ∧ (3 - 2)^2 + a^2 = 5^2

-- Theorem statement
theorem parabola_intersection_theorem (k : ℝ) :
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2) ↔ k = 0 ∨ k = -1 ∨ k = 2/3 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l2903_290327


namespace NUMINAMATH_CALUDE_evaluate_expression_l2903_290368

theorem evaluate_expression : -(18 / 3 * 8 - 48 + 4 * 6) = -24 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2903_290368


namespace NUMINAMATH_CALUDE_expression_simplification_l2903_290347

theorem expression_simplification :
  ((2 + 3 + 4 + 5) / 2) + ((2 * 5 + 8) / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2903_290347


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l2903_290323

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def N (m : ℝ) : Set ℝ := {x : ℝ | x - m ≥ 0}

-- Theorem for the first part
theorem union_condition (m : ℝ) : M ∪ N m = N m ↔ m ≤ -2 := by sorry

-- Theorem for the second part
theorem intersection_condition (m : ℝ) : M ∩ N m = ∅ ↔ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l2903_290323


namespace NUMINAMATH_CALUDE_youth_palace_participants_l2903_290359

theorem youth_palace_participants (last_year this_year : ℕ) :
  this_year = last_year + 41 →
  this_year = 3 * last_year - 35 →
  this_year = 79 ∧ last_year = 38 := by
  sorry

end NUMINAMATH_CALUDE_youth_palace_participants_l2903_290359


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2903_290331

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^4 - 3*x^3 - 9*x^2 + 27*x - 8 = (65 + 81*Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2903_290331


namespace NUMINAMATH_CALUDE_problem_statement_l2903_290329

theorem problem_statement (a b x y : ℕ+) (P : ℕ) 
  (h1 : ∃ k : ℕ, a * x + b * y = k * (a^2 + b^2))
  (h2 : P = x^2 + y^2)
  (h3 : Nat.Prime P) :
  (P ∣ (a^2 + b^2)) ∧ (a = x ∧ b = y) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2903_290329


namespace NUMINAMATH_CALUDE_total_limes_is_195_l2903_290373

/-- The number of limes picked by each person -/
def fred_limes : ℕ := 36
def alyssa_limes : ℕ := 32
def nancy_limes : ℕ := 35
def david_limes : ℕ := 42
def eileen_limes : ℕ := 50

/-- The total number of limes picked -/
def total_limes : ℕ := fred_limes + alyssa_limes + nancy_limes + david_limes + eileen_limes

/-- Theorem stating that the total number of limes picked is 195 -/
theorem total_limes_is_195 : total_limes = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_limes_is_195_l2903_290373


namespace NUMINAMATH_CALUDE_sum_x_coordinates_preserved_l2903_290357

/-- A polygon in the Cartesian plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- Create a new polygon from the midpoints of the sides of a given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

theorem sum_x_coordinates_preserved (n : ℕ) (Q1 : Polygon) 
  (h1 : Q1.vertices.length = n)
  (Q2 := midpointPolygon Q1)
  (Q3 := midpointPolygon Q2) :
  sumXCoordinates Q3 = sumXCoordinates Q1 :=
sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_preserved_l2903_290357


namespace NUMINAMATH_CALUDE_product_a_b_equals_27_over_8_l2903_290358

theorem product_a_b_equals_27_over_8 
  (a b c : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : c = 3 → a = b^2) 
  (h3 : b + c = 2*a) 
  (h4 : c = 3) 
  (h5 : b + c = b * c) : 
  a * b = 27/8 := by
sorry

end NUMINAMATH_CALUDE_product_a_b_equals_27_over_8_l2903_290358


namespace NUMINAMATH_CALUDE_snug_fit_circles_l2903_290367

/-- Given a circle of diameter 3 inches containing two circles of diameters 2 inches and 1 inch,
    the diameter of two additional identical circles that fit snugly within the larger circle
    is 12/7 inches. -/
theorem snug_fit_circles (R : ℝ) (r₁ : ℝ) (r₂ : ℝ) (d : ℝ) :
  R = 3/2 ∧ r₁ = 1 ∧ r₂ = 1/2 →
  d > 0 →
  (R - d)^2 + (R - d)^2 = (2*d)^2 →
  d = 6/7 :=
by sorry

end NUMINAMATH_CALUDE_snug_fit_circles_l2903_290367


namespace NUMINAMATH_CALUDE_real_number_inequality_l2903_290301

theorem real_number_inequality (x : Fin 8 → ℝ) (h : ∀ i j, i ≠ j → x i ≠ x j) :
  ∃ i j, i ≠ j ∧ 0 < (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) < Real.tan (π / 7) := by
  sorry

end NUMINAMATH_CALUDE_real_number_inequality_l2903_290301


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l2903_290380

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = r1 + r2

theorem tangent_circles_radius (r : ℝ) :
  r > 0 →
  externally_tangent (0, 0) (3, 0) 1 r →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l2903_290380


namespace NUMINAMATH_CALUDE_volume_for_weight_less_than_112_l2903_290374

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℝ
  /-- Assumption that k is positive -/
  k_pos : k > 0

/-- The volume of the substance given its weight -/
def volume (s : Substance) (weight : ℝ) : ℝ := s.k * weight

theorem volume_for_weight_less_than_112 (s : Substance) (weight : ℝ) 
  (h1 : volume s 112 = 48) (h2 : 0 < weight) (h3 : weight < 112) :
  volume s weight = (48 / 112) * weight := by
sorry

end NUMINAMATH_CALUDE_volume_for_weight_less_than_112_l2903_290374


namespace NUMINAMATH_CALUDE_expression_equality_l2903_290366

theorem expression_equality : |1 - Real.sqrt 2| - Real.sqrt 8 + (Real.sqrt 2 - 1)^0 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2903_290366


namespace NUMINAMATH_CALUDE_vector_projection_on_x_axis_l2903_290363

theorem vector_projection_on_x_axis (a : ℝ) (φ : ℝ) :
  a = 5 →
  φ = Real.pi / 3 →
  a * Real.cos φ = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_on_x_axis_l2903_290363


namespace NUMINAMATH_CALUDE_vector_problem_l2903_290314

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (-1, 2)
def c (m : ℝ) : ℝ × ℝ := (2, m)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_problem (m : ℝ) :
  (dot_product a (c m) < m^2 → m > 4 ∨ m < -2) ∧
  (parallel (a.1 + (c m).1, a.2 + (c m).2) b → m = -14) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l2903_290314


namespace NUMINAMATH_CALUDE_smallest_dimension_is_eight_l2903_290320

/-- Represents a rectangular crate with dimensions a, b, and c. -/
structure Crate where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a right circular cylinder. -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate. -/
def cylinderFitsInCrate (cyl : Cylinder) (cr : Crate) : Prop :=
  2 * cyl.radius ≤ cr.a ∧ 2 * cyl.radius ≤ cr.b ∧ cyl.height ≤ cr.c ∨
  2 * cyl.radius ≤ cr.a ∧ 2 * cyl.radius ≤ cr.c ∧ cyl.height ≤ cr.b ∨
  2 * cyl.radius ≤ cr.b ∧ 2 * cyl.radius ≤ cr.c ∧ cyl.height ≤ cr.a

/-- The main theorem stating that the smallest dimension of the crate is 8 feet. -/
theorem smallest_dimension_is_eight
  (cr : Crate)
  (h1 : cr.b = 8)
  (h2 : cr.c = 12)
  (h3 : ∃ (cyl : Cylinder), cyl.radius = 7 ∧ cylinderFitsInCrate cyl cr) :
  min cr.a (min cr.b cr.c) = 8 := by
  sorry


end NUMINAMATH_CALUDE_smallest_dimension_is_eight_l2903_290320


namespace NUMINAMATH_CALUDE_product_of_solutions_l2903_290321

theorem product_of_solutions (x : ℝ) : 
  (|18 / x + 4| = 3) → 
  (∃ y : ℝ, (|18 / y + 4| = 3) ∧ x * y = 324 / 7) :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2903_290321


namespace NUMINAMATH_CALUDE_sin_alpha_plus_seven_pi_sixth_l2903_290339

theorem sin_alpha_plus_seven_pi_sixth (α : ℝ) 
  (h : Real.sin α + Real.cos (α - π / 6) = Real.sqrt 3 / 3) : 
  Real.sin (α + 7 * π / 6) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_seven_pi_sixth_l2903_290339


namespace NUMINAMATH_CALUDE_ashley_family_movie_cost_l2903_290397

/-- Calculates the total cost of a movie outing for Ashley's family --/
def movie_outing_cost (
  child_ticket_price : ℝ)
  (adult_ticket_price_diff : ℝ)
  (senior_ticket_price_diff : ℝ)
  (morning_discount : ℝ)
  (voucher_discount : ℝ)
  (popcorn_price : ℝ)
  (soda_price : ℝ)
  (candy_price : ℝ)
  (concession_discount : ℝ) : ℝ :=
  let adult_ticket_price := child_ticket_price + adult_ticket_price_diff
  let senior_ticket_price := adult_ticket_price - senior_ticket_price_diff
  let ticket_cost := 2 * adult_ticket_price + 4 * child_ticket_price + senior_ticket_price
  let discounted_ticket_cost := ticket_cost * (1 - morning_discount) - child_ticket_price - voucher_discount
  let concession_cost := 3 * popcorn_price + 2 * soda_price + candy_price
  let discounted_concession_cost := concession_cost * (1 - concession_discount)
  discounted_ticket_cost + discounted_concession_cost

/-- Theorem stating the total cost of Ashley's family's movie outing --/
theorem ashley_family_movie_cost :
  movie_outing_cost 4.25 3.50 1.75 0.10 4.00 5.25 3.50 4.00 0.10 = 50.47 := by
  sorry

end NUMINAMATH_CALUDE_ashley_family_movie_cost_l2903_290397


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_circle_angles_l2903_290351

theorem right_triangle_inscribed_circle_angles (k : ℝ) (k_pos : k > 0) :
  ∃ (α β : ℝ),
    α + β = π / 2 ∧
    (α = π / 4 - Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1))) ∨
     α = π / 4 + Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1)))) ∧
    (β = π / 4 - Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1))) ∨
     β = π / 4 + Real.arcsin (Real.sqrt 2 * (k - 1) / (2 * (k + 1)))) :=
by sorry


end NUMINAMATH_CALUDE_right_triangle_inscribed_circle_angles_l2903_290351


namespace NUMINAMATH_CALUDE_red_tetrahedron_volume_l2903_290365

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem red_tetrahedron_volume (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume := cube_side_length ^ 3
  let green_tetrahedron_volume := (1 / 3) * (1 / 2 * cube_side_length ^ 2) * cube_side_length
  let red_tetrahedron_volume := cube_volume - 4 * green_tetrahedron_volume
  red_tetrahedron_volume = 512 / 3 := by
  sorry

end NUMINAMATH_CALUDE_red_tetrahedron_volume_l2903_290365


namespace NUMINAMATH_CALUDE_problem_statement_l2903_290354

theorem problem_statement (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 3) :
  (a + 1) * (b - 1) = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2903_290354


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2903_290315

theorem minimum_value_theorem (a b : ℝ) (h : a - 3*b + 6 = 0) :
  ∃ (m : ℝ), m = (1/4 : ℝ) ∧ ∀ x y : ℝ, x - 3*y + 6 = 0 → 2^x + (1/8)^y ≥ m :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2903_290315


namespace NUMINAMATH_CALUDE_remaining_soup_feeds_six_adults_l2903_290338

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Proves that given 5 cans of soup, where each can feeds 3 adults or 5 children,
    if 15 children are fed, the remaining soup will feed 6 adults -/
theorem remaining_soup_feeds_six_adults 
  (can : SoupCan) 
  (h1 : can.adults = 3) 
  (h2 : can.children = 5) 
  (total_cans : ℕ) 
  (h3 : total_cans = 5) 
  (children_fed : ℕ) 
  (h4 : children_fed = 15) : 
  (total_cans - (children_fed / can.children)) * can.adults = 6 := by
sorry

end NUMINAMATH_CALUDE_remaining_soup_feeds_six_adults_l2903_290338


namespace NUMINAMATH_CALUDE_largest_prime_factor_l2903_290370

def expression : ℤ := 16^4 + 3 * 16^2 + 2 - 15^4

theorem largest_prime_factor (p : ℕ) : 
  Nat.Prime p ∧ p ∣ expression.natAbs ∧ 
  ∀ q : ℕ, Nat.Prime q ∧ q ∣ expression.natAbs → q ≤ p ↔ p = 241 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l2903_290370


namespace NUMINAMATH_CALUDE_oranges_sold_count_l2903_290316

/-- Given information about oranges on a truck -/
structure OrangeTruck where
  bags : Nat
  oranges_per_bag : Nat
  rotten : Nat
  for_juice : Nat

/-- Calculate the number of oranges to be sold -/
def oranges_to_sell (truck : OrangeTruck) : Nat :=
  truck.bags * truck.oranges_per_bag - (truck.rotten + truck.for_juice)

/-- Theorem stating the number of oranges to be sold -/
theorem oranges_sold_count (truck : OrangeTruck) 
  (h1 : truck.bags = 10)
  (h2 : truck.oranges_per_bag = 30)
  (h3 : truck.rotten = 50)
  (h4 : truck.for_juice = 30) :
  oranges_to_sell truck = 220 := by
  sorry

#eval oranges_to_sell { bags := 10, oranges_per_bag := 30, rotten := 50, for_juice := 30 }

end NUMINAMATH_CALUDE_oranges_sold_count_l2903_290316


namespace NUMINAMATH_CALUDE_square_pyramid_volume_l2903_290393

/-- The volume of a square pyramid inscribed in a cube -/
theorem square_pyramid_volume (cube_side_length : ℝ) (pyramid_volume : ℝ) :
  cube_side_length = 3 →
  pyramid_volume = (1 / 3) * (cube_side_length ^ 3) →
  pyramid_volume = 9 := by
sorry

end NUMINAMATH_CALUDE_square_pyramid_volume_l2903_290393


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_squared_l2903_290337

theorem sqrt_x_minus_one_squared (x : ℝ) (h : |1 - x| = 1 + |x|) : 
  Real.sqrt ((x - 1)^2) = 1 - x :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_squared_l2903_290337


namespace NUMINAMATH_CALUDE_line_parameterization_l2903_290325

def is_valid_parameterization (x₀ y₀ dx dy : ℝ) : Prop :=
  y₀ = 3 * x₀ + 5 ∧ ∃ (k : ℝ), dx = k * 1 ∧ dy = k * 3

theorem line_parameterization 
  (x₀ y₀ dx dy t : ℝ) :
  is_valid_parameterization x₀ y₀ dx dy ↔ 
  ∀ t, (3 * (x₀ + t * dx) + 5 = y₀ + t * dy) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l2903_290325


namespace NUMINAMATH_CALUDE_tarantulas_needed_l2903_290356

/-- The number of legs for each animal type --/
def legs_per_chimp : ℕ := 4
def legs_per_lion : ℕ := 4
def legs_per_lizard : ℕ := 4
def legs_per_tarantula : ℕ := 8

/-- The number of animals already seen --/
def chimps_seen : ℕ := 12
def lions_seen : ℕ := 8
def lizards_seen : ℕ := 5

/-- The total number of legs Borgnine wants to see --/
def total_legs_goal : ℕ := 1100

/-- Theorem: The number of tarantulas needed to reach the total legs goal --/
theorem tarantulas_needed : 
  (chimps_seen * legs_per_chimp + 
   lions_seen * legs_per_lion + 
   lizards_seen * legs_per_lizard + 
   125 * legs_per_tarantula) = total_legs_goal :=
by sorry

end NUMINAMATH_CALUDE_tarantulas_needed_l2903_290356


namespace NUMINAMATH_CALUDE_jess_gallery_distance_l2903_290348

/-- The distance Jess walks to the gallery -/
def distance_to_gallery (total_distance : ℕ) (distance_to_store : ℕ) (distance_gallery_to_work : ℕ) : ℕ :=
  total_distance - distance_to_store - distance_gallery_to_work

/-- Proof that Jess walks 6 blocks to the gallery -/
theorem jess_gallery_distance :
  distance_to_gallery 25 11 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jess_gallery_distance_l2903_290348


namespace NUMINAMATH_CALUDE_bee_multiplier_l2903_290328

/-- Given the number of bees seen on two consecutive days, 
    prove that the ratio of bees on the second day to the first day is 3 -/
theorem bee_multiplier (bees_day1 bees_day2 : ℕ) 
  (h1 : bees_day1 = 144) 
  (h2 : bees_day2 = 432) : 
  (bees_day2 : ℚ) / bees_day1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bee_multiplier_l2903_290328


namespace NUMINAMATH_CALUDE_dice_product_probability_composite_probability_l2903_290308

/-- A function that determines if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of possible outcomes when rolling a 6-sided die -/
def dieOutcomes : Finset ℕ := sorry

/-- The set of all possible outcomes when rolling 4 dice -/
def allOutcomes : Finset (ℕ × ℕ × ℕ × ℕ) := sorry

/-- The product of the numbers in a 4-tuple -/
def product (t : ℕ × ℕ × ℕ × ℕ) : ℕ := sorry

/-- The set of outcomes that result in a non-composite product -/
def nonCompositeOutcomes : Finset (ℕ × ℕ × ℕ × ℕ) := sorry

theorem dice_product_probability :
  (Finset.card nonCompositeOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 13 / 1296 :=
sorry

theorem composite_probability :
  1 - (Finset.card nonCompositeOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 1283 / 1296 :=
sorry

end NUMINAMATH_CALUDE_dice_product_probability_composite_probability_l2903_290308


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2903_290352

-- Problem 1
theorem problem_1 : Real.sqrt 9 + |3 - Real.pi| - Real.sqrt ((-3)^2) = Real.pi - 3 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x : ℝ, 3 * (x - 1)^3 = 81 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2903_290352


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2903_290350

theorem complex_expression_equality (y : ℂ) (h : y = Complex.exp (2 * π * I / 9)) :
  (3 * y + y^3) * (3 * y^3 + y^9) * (3 * y^6 + y^18) = 121 + 48 * (y + y^6) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2903_290350


namespace NUMINAMATH_CALUDE_min_shapes_for_square_l2903_290319

/-- The area of one shape in square units -/
def shape_area : ℕ := 3

/-- The side length of the square formed by the shapes -/
def square_side : ℕ := 6

/-- The area of the square formed by the shapes -/
def square_area : ℕ := square_side * square_side

/-- The number of shapes required to form the square -/
def num_shapes : ℕ := square_area / shape_area

theorem min_shapes_for_square : 
  ∀ n : ℕ, n < num_shapes → 
  ¬∃ s : ℕ, s * s = n * shape_area ∧ s % shape_area = 0 := by
  sorry

#eval num_shapes  -- Should output 12

end NUMINAMATH_CALUDE_min_shapes_for_square_l2903_290319


namespace NUMINAMATH_CALUDE_sqrt_minus_one_mod_prime_l2903_290386

theorem sqrt_minus_one_mod_prime (p : Nat) (h_prime : Prime p) (h_gt_two : p > 2) :
  (∃ x : Nat, x^2 ≡ -1 [ZMOD p]) ↔ ∃ k : Nat, p = 4*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_minus_one_mod_prime_l2903_290386


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l2903_290340

/-- The surface area of a cuboid -/
def cuboidSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a cuboid with length 8, width 10, and height 12 is 592 -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 8 10 12 = 592 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l2903_290340


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l2903_290313

/-- If a circle with equation (x-1)^2 + (y-2)^2 = 1 is symmetric about the line y = x + b, then b = 1 -/
theorem circle_symmetry_line (b : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 1 ↔ (x - 1)^2 + ((x + b) - 2)^2 = 1) → 
  b = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l2903_290313


namespace NUMINAMATH_CALUDE_roots_cube_equality_l2903_290372

/-- Given a polynomial P(x) = 3x² + 3mx + m² - 1 where m is a real number,
    and x₁, x₂ are the roots of P(x), prove that P(x₁³) = P(x₂³) -/
theorem roots_cube_equality (m : ℝ) (x₁ x₂ : ℝ) : 
  let P := fun x : ℝ => 3 * x^2 + 3 * m * x + m^2 - 1
  (P x₁ = 0 ∧ P x₂ = 0) → P (x₁^3) = P (x₂^3) := by
  sorry

end NUMINAMATH_CALUDE_roots_cube_equality_l2903_290372


namespace NUMINAMATH_CALUDE_unique_prime_triple_l2903_290388

theorem unique_prime_triple : ∃! (I M C : ℕ),
  (Nat.Prime I ∧ Nat.Prime M ∧ Nat.Prime C) ∧
  (I ≤ M ∧ M ≤ C) ∧
  (I * M * C = I + M + C + 1007) ∧
  I = 2 ∧ M = 2 ∧ C = 337 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l2903_290388


namespace NUMINAMATH_CALUDE_largest_undefined_x_l2903_290336

theorem largest_undefined_x : 
  let f (x : ℝ) := 10 * x^2 - 30 * x + 20
  ∃ (max : ℝ), f max = 0 ∧ ∀ x, f x = 0 → x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_largest_undefined_x_l2903_290336


namespace NUMINAMATH_CALUDE_min_value_theorem_l2903_290369

theorem min_value_theorem (a b : ℝ) (h1 : a + b = 2) (h2 : b > 0) :
  (∀ x y : ℝ, x + y = 2 → y > 0 → (1 / (2 * |x|) + |x| / y) ≥ 3/4) ∧
  (∃ x y : ℝ, x + y = 2 ∧ y > 0 ∧ 1 / (2 * |x|) + |x| / y = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2903_290369


namespace NUMINAMATH_CALUDE_study_session_duration_in_minutes_l2903_290335

-- Define the duration of the study session
def study_session_hours : ℕ := 8
def study_session_minutes : ℕ := 45

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem study_session_duration_in_minutes :
  study_session_hours * minutes_per_hour + study_session_minutes = 525 :=
by sorry

end NUMINAMATH_CALUDE_study_session_duration_in_minutes_l2903_290335


namespace NUMINAMATH_CALUDE_tangent_addition_formula_l2903_290355

theorem tangent_addition_formula : 
  (Real.tan (12 * π / 180) + Real.tan (18 * π / 180)) / 
  (1 - Real.tan (12 * π / 180) * Real.tan (18 * π / 180)) = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_addition_formula_l2903_290355


namespace NUMINAMATH_CALUDE_fourth_square_dots_l2903_290398

/-- The side length of the nth square in the sequence -/
def side_length (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- The number of dots in the nth square -/
def num_dots (n : ℕ) : ℕ := (side_length n) ^ 2

theorem fourth_square_dots :
  num_dots 4 = 49 := by sorry

end NUMINAMATH_CALUDE_fourth_square_dots_l2903_290398


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2903_290375

/-- The quadratic function f(x) = x^2 + ax + 6 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 6

theorem quadratic_inequality_solutions (a : ℝ) :
  (a = 5 → {x : ℝ | f 5 x < 0} = {x : ℝ | -3 < x ∧ x < -2}) ∧
  ({x : ℝ | f a x > 0} = Set.univ → a ∈ Set.Ioo (-2*Real.sqrt 6) (2*Real.sqrt 6)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2903_290375


namespace NUMINAMATH_CALUDE_intersection_point_product_l2903_290317

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := y^2 / 16 + x^2 / 9 = 1

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 / 5 = 1

-- Define the common foci
def common_foci (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ (a b c d : ℝ), 
    a^2 / 16 + b^2 / 9 = 1 ∧ 
    c^2 / 4 - d^2 / 5 = 1 ∧
    F1 = (b, a) ∧ F2 = (-b, -a)

-- Define the point of intersection
def is_intersection_point (P : ℝ × ℝ) : Prop :=
  is_on_ellipse P.1 P.2 ∧ is_on_hyperbola P.1 P.2

-- The theorem
theorem intersection_point_product (F1 F2 P : ℝ × ℝ) :
  common_foci F1 F2 → is_intersection_point P →
  (P.1 - F1.1)^2 + (P.2 - F1.2)^2 * ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 144 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_product_l2903_290317


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2903_290346

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (m : ℝ), m = b / a ∧ m = Real.sqrt 3) →
  (∃ (d : ℝ), d = 2 * Real.sqrt 3 ∧ d = b) →
  (∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2903_290346


namespace NUMINAMATH_CALUDE_nancy_scholarship_amount_l2903_290364

/-- Proves that Nancy's scholarship amount is $3,000 given the tuition costs and other conditions --/
theorem nancy_scholarship_amount : 
  ∀ (tuition : ℕ) 
    (parent_contribution : ℕ) 
    (work_hours : ℕ) 
    (hourly_rate : ℕ) 
    (scholarship : ℕ),
  tuition = 22000 →
  parent_contribution = tuition / 2 →
  work_hours = 200 →
  hourly_rate = 10 →
  scholarship + 2 * scholarship + parent_contribution + work_hours * hourly_rate = tuition →
  scholarship = 3000 := by
sorry


end NUMINAMATH_CALUDE_nancy_scholarship_amount_l2903_290364


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l2903_290379

/-- Given that 50 cows eat 50 bags of husk in 50 days, prove that one cow will eat one bag of husk in 50 days -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 50 ∧ bags = 50 ∧ days = 50) :
  (1 : ℕ) * bags * days = cows * (1 : ℕ) * days :=
by sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l2903_290379


namespace NUMINAMATH_CALUDE_equation_solution_l2903_290389

theorem equation_solution :
  let f (x : ℂ) := (3 * x^2 - 1) / (4 * x - 4)
  ∀ x : ℂ, f x = 2/3 ↔ x = 8/18 + (Complex.I * Real.sqrt 116)/18 ∨ x = 8/18 - (Complex.I * Real.sqrt 116)/18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2903_290389


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2903_290378

/-- A hyperbola with given asymptotes and passing through a specific point -/
theorem hyperbola_equation (x y : ℝ) : 
  (∀ (k : ℝ), (2*x = 3*y ∨ 2*x = -3*y) → k*(2*x) = k*(3*y)) →  -- Asymptotes condition
  (4*(1:ℝ)^2 - 9*(2:ℝ)^2 = -32) →                              -- Point (1,2) satisfies the equation
  (4*x^2 - 9*y^2 = -32)                                         -- Resulting hyperbola equation
  := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2903_290378


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2903_290376

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 8 * Real.pi →
  Real.cos (a 3 + a 7) = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2903_290376


namespace NUMINAMATH_CALUDE_determinant_solution_l2903_290305

theorem determinant_solution (a : ℝ) (h : a ≠ 0) :
  ∃ x : ℝ, Matrix.det 
    ![![x + a, x, x],
      ![x, x + a, x],
      ![x, x, x + a]] = 0 ↔ x = -a / 3 := by
  sorry

end NUMINAMATH_CALUDE_determinant_solution_l2903_290305


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l2903_290371

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem minimum_value_implies_a (a : ℝ) :
  a > 0 →
  (∀ x, x > 0 → x ≤ Real.exp 1 → f a x ≥ 3/2) →
  (∃ x, 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l2903_290371


namespace NUMINAMATH_CALUDE_water_for_bathing_is_two_l2903_290344

/-- Calculates the water needed for bathing per horse per day -/
def water_for_bathing (initial_horses : ℕ) (added_horses : ℕ) (drinking_water_per_horse : ℕ) (total_days : ℕ) (total_water : ℕ) : ℚ :=
  let total_horses := initial_horses + added_horses
  let total_drinking_water := total_horses * drinking_water_per_horse * total_days
  let total_bathing_water := total_water - total_drinking_water
  (total_bathing_water : ℚ) / (total_horses * total_days : ℚ)

/-- Theorem: Given the conditions, each horse needs 2 liters of water for bathing per day -/
theorem water_for_bathing_is_two :
  water_for_bathing 3 5 5 28 1568 = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_for_bathing_is_two_l2903_290344


namespace NUMINAMATH_CALUDE_plot_length_is_100_l2903_290343

/-- Proves that the length of a rectangular plot is 100 meters given specific conditions. -/
theorem plot_length_is_100 (width : ℝ) (path_width : ℝ) (gravel_cost_per_sqm : ℝ) (total_gravel_cost : ℝ) :
  width = 65 →
  path_width = 2.5 →
  gravel_cost_per_sqm = 0.4 →
  total_gravel_cost = 340 →
  ∃ (length : ℝ),
    ((length + 2 * path_width) * (width + 2 * path_width) - length * width) * gravel_cost_per_sqm = total_gravel_cost ∧
    length = 100 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_100_l2903_290343


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2903_290309

theorem absolute_value_inequality (x : ℝ) : 
  |8 - 3*x| > 0 ↔ x ≠ 8/3 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2903_290309


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l2903_290333

theorem square_garden_perimeter (a p : ℝ) (h1 : a > 0) (h2 : p > 0) (h3 : a = 2 * p + 14.25) : p = 38 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l2903_290333


namespace NUMINAMATH_CALUDE_rectangular_prism_ratio_l2903_290300

/-- In a rectangular prism with edges a ≤ b ≤ c, if a:b = b:c = c:√(a² + b²), 
    then (a/b)² = (√5 - 1)/2 -/
theorem rectangular_prism_ratio (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) 
  (h_ratio : a / b = b / c ∧ b / c = c / Real.sqrt (a^2 + b^2)) :
  (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_ratio_l2903_290300


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2903_290392

theorem quadratic_roots_property (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + a*x₁ + 4 = 0 ∧ 
   x₂^2 + a*x₂ + 4 = 0 ∧ 
   x₁^2 - 20/(3*x₂^3) = x₂^2 - 20/(3*x₁^3)) → 
  a = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2903_290392


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_5_l2903_290326

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + a

theorem tangent_parallel_implies_a_equals_5 (a : ℝ) :
  f' a 1 = 7 → a = 5 := by
  sorry

#check tangent_parallel_implies_a_equals_5

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_5_l2903_290326


namespace NUMINAMATH_CALUDE_min_c_value_l2903_290385

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c)
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1002 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 1002 ∧
    ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a'| + |x - b'| + |x - 1002| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l2903_290385


namespace NUMINAMATH_CALUDE_bd_range_l2903_290332

/-- Represents a quadrilateral ABCD with side lengths and diagonal BD --/
structure Quadrilateral :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)
  (BD : ℤ)

/-- The specific quadrilateral from the problem --/
def problem_quadrilateral : Quadrilateral :=
  { AB := 7
  , BC := 15
  , CD := 7
  , DA := 11
  , BD := 0 }  -- BD is initially set to 0, but will be constrained later

theorem bd_range (q : Quadrilateral) (h : q = problem_quadrilateral) :
  9 ≤ q.BD ∧ q.BD ≤ 17 := by
  sorry

#check bd_range

end NUMINAMATH_CALUDE_bd_range_l2903_290332


namespace NUMINAMATH_CALUDE_power_of_product_with_negative_l2903_290304

theorem power_of_product_with_negative (m n : ℝ) : (-2 * m^3 * n^2)^2 = 4 * m^6 * n^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_with_negative_l2903_290304


namespace NUMINAMATH_CALUDE_round_trip_ticket_percentage_l2903_290362

theorem round_trip_ticket_percentage (total_passengers : ℝ) 
  (h1 : (0.2 : ℝ) * total_passengers = (passengers_with_roundtrip_and_car : ℝ))
  (h2 : (0.5 : ℝ) * (passengers_with_roundtrip : ℝ) = passengers_with_roundtrip - passengers_with_roundtrip_and_car) :
  (passengers_with_roundtrip : ℝ) / total_passengers = (0.4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_round_trip_ticket_percentage_l2903_290362


namespace NUMINAMATH_CALUDE_each_score_is_individual_l2903_290395

/-- Represents a candidate's math score -/
structure MathScore where
  score : ℝ

/-- Represents the population of candidates -/
structure Population where
  candidates : Finset MathScore
  size_gt_100000 : candidates.card > 100000

/-- Represents a sample of candidates -/
structure Sample where
  scores : Finset MathScore
  size_eq_1000 : scores.card = 1000

/-- Theorem stating that each math score in the sample is an individual data point -/
theorem each_score_is_individual (pop : Population) (sample : Sample) 
  (h_sample : ∀ s ∈ sample.scores, s ∈ pop.candidates) :
  ∀ s ∈ sample.scores, ∃! i : MathScore, i = s :=
sorry

end NUMINAMATH_CALUDE_each_score_is_individual_l2903_290395


namespace NUMINAMATH_CALUDE_vector_perpendicular_parallel_l2903_290384

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v = (t * w.1, t * w.2)

theorem vector_perpendicular_parallel :
  (∃ k : ℝ, perpendicular (k * a.1 + b.1, k * a.2 + b.2) (a.1 - 3 * b.1, a.2 - 3 * b.2) ∧ k = 19) ∧
  (∃ k : ℝ, parallel (k * a.1 + b.1, k * a.2 + b.2) (a.1 - 3 * b.1, a.2 - 3 * b.2) ∧ k = -1/3) :=
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_parallel_l2903_290384


namespace NUMINAMATH_CALUDE_pelicans_remaining_theorem_l2903_290396

/-- The number of Pelicans remaining in Shark Bite Cove after some moved to Pelican Bay -/
def pelicansRemaining (sharksInPelicanBay : ℕ) : ℕ :=
  let originalPelicans := sharksInPelicanBay / 2
  let pelicansMoved := originalPelicans / 3
  originalPelicans - pelicansMoved

/-- Theorem stating that given 60 sharks in Pelican Bay, 20 Pelicans remain in Shark Bite Cove -/
theorem pelicans_remaining_theorem :
  pelicansRemaining 60 = 20 := by
  sorry

#eval pelicansRemaining 60

end NUMINAMATH_CALUDE_pelicans_remaining_theorem_l2903_290396


namespace NUMINAMATH_CALUDE_matches_for_128_teams_l2903_290303

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  num_teams : ℕ
  num_teams_positive : 0 < num_teams

/-- Calculates the number of matches required to determine a champion. -/
def matches_required (tournament : SingleEliminationTournament) : ℕ :=
  tournament.num_teams - 1

/-- Theorem: In a single-elimination tournament with 128 teams, 127 matches are required. -/
theorem matches_for_128_teams :
  let tournament := SingleEliminationTournament.mk 128 (by norm_num)
  matches_required tournament = 127 := by
  sorry

end NUMINAMATH_CALUDE_matches_for_128_teams_l2903_290303


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l2903_290330

theorem smallest_integer_with_remainder_two : ∃! m : ℕ,
  m > 1 ∧
  m % 13 = 2 ∧
  m % 5 = 2 ∧
  m % 3 = 2 ∧
  ∀ n : ℕ, n > 1 ∧ n % 13 = 2 ∧ n % 5 = 2 ∧ n % 3 = 2 → m ≤ n :=
by
  use 197
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l2903_290330


namespace NUMINAMATH_CALUDE_four_people_five_chairs_middle_empty_l2903_290324

/-- The number of ways to arrange people in chairs. -/
def seating_arrangements (total_chairs : ℕ) (people : ℕ) (empty_chair : ℕ) : ℕ :=
  (total_chairs - 1).factorial / ((total_chairs - 1 - people).factorial)

/-- Theorem: There are 24 ways to arrange 4 people in 5 chairs with the middle chair empty. -/
theorem four_people_five_chairs_middle_empty :
  seating_arrangements 5 4 3 = 24 := by sorry

end NUMINAMATH_CALUDE_four_people_five_chairs_middle_empty_l2903_290324


namespace NUMINAMATH_CALUDE_lily_shopping_theorem_l2903_290361

/-- Calculates the remaining amount for coffee after Lily's shopping trip --/
def remaining_for_coffee (initial_amount : ℚ) (celery_cost : ℚ) (cereal_cost : ℚ) (cereal_discount : ℚ)
  (bread_cost : ℚ) (milk_cost : ℚ) (milk_discount : ℚ) (potato_cost : ℚ) (potato_quantity : ℕ) : ℚ :=
  initial_amount - (celery_cost + cereal_cost * (1 - cereal_discount) + bread_cost + 
  milk_cost * (1 - milk_discount) + potato_cost * potato_quantity)

theorem lily_shopping_theorem (initial_amount : ℚ) (celery_cost : ℚ) (cereal_cost : ℚ) (cereal_discount : ℚ)
  (bread_cost : ℚ) (milk_cost : ℚ) (milk_discount : ℚ) (potato_cost : ℚ) (potato_quantity : ℕ) :
  initial_amount = 60 ∧ 
  celery_cost = 5 ∧ 
  cereal_cost = 12 ∧ 
  cereal_discount = 0.5 ∧ 
  bread_cost = 8 ∧ 
  milk_cost = 10 ∧ 
  milk_discount = 0.1 ∧ 
  potato_cost = 1 ∧ 
  potato_quantity = 6 →
  remaining_for_coffee initial_amount celery_cost cereal_cost cereal_discount bread_cost milk_cost milk_discount potato_cost potato_quantity = 26 := by
  sorry

#eval remaining_for_coffee 60 5 12 0.5 8 10 0.1 1 6

end NUMINAMATH_CALUDE_lily_shopping_theorem_l2903_290361


namespace NUMINAMATH_CALUDE_cube_sum_problem_l2903_290360

theorem cube_sum_problem (x y : ℝ) (h1 : x^3 + y^3 = 7) (h2 : x^6 + y^6 = 49) : 
  x^9 + y^9 = 343 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l2903_290360


namespace NUMINAMATH_CALUDE_system_solution_l2903_290381

theorem system_solution (x y z u : ℚ) : 
  x + y = 12 ∧ 
  x / z = 3 / 2 ∧ 
  z + u = 10 ∧ 
  y * u = 36 →
  x = 6 ∧ y = 6 ∧ z = 4 ∧ u = 6 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2903_290381


namespace NUMINAMATH_CALUDE_value_of_a_l2903_290334

theorem value_of_a (a b : ℚ) (h1 : b / a = 3) (h2 : b = 12 - 5 * a) : a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2903_290334


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l2903_290353

-- Problem 1
theorem problem_1 : 13 + (-24) - (-40) = 29 := by sorry

-- Problem 2
theorem problem_2 : 3 * (-2) + (-36) / 4 = -15 := by sorry

-- Problem 3
theorem problem_3 : (1 + 3/4 - 7/8 - 7/16) / (-7/8) = -1/2 := by sorry

-- Problem 4
theorem problem_4 : (-2)^3 / 4 - (10 - (-1)^10 * 2) = -10 := by sorry

-- Problem 5
theorem problem_5 (x y : ℝ) : 7*x*y + 2 - 3*x*y - 5 = 4*x*y - 3 := by sorry

-- Problem 6
theorem problem_6 (x : ℝ) : 4*x^2 - (5*x + x^2) + 6*x - 2*x^2 = x^2 + x := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l2903_290353


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2903_290399

theorem circle_area_ratio (s r : ℝ) (hs : s > 0) (hr : r > 0) (h : r = 0.4 * s) :
  (π * (r / 2)^2) / (π * (s / 2)^2) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2903_290399


namespace NUMINAMATH_CALUDE_arithmetic_fraction_subtraction_l2903_290318

theorem arithmetic_fraction_subtraction :
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_fraction_subtraction_l2903_290318


namespace NUMINAMATH_CALUDE_solution_sets_equivalent_l2903_290390

theorem solution_sets_equivalent : 
  {x : ℝ | |8*x + 9| < 7} = {x : ℝ | -4*x^2 - 9*x - 2 > 0} := by sorry

end NUMINAMATH_CALUDE_solution_sets_equivalent_l2903_290390


namespace NUMINAMATH_CALUDE_polynomial_equality_l2903_290311

theorem polynomial_equality (x : ℝ) : 
  (x - 2/3) * (x + 1/2) = x^2 - (1/6)*x - 1/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2903_290311


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2903_290382

theorem quadratic_coefficient (a b c y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = -16 →
  b = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2903_290382
