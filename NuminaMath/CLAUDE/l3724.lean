import Mathlib

namespace NUMINAMATH_CALUDE_probability_two_defective_shipment_l3724_372407

/-- The probability of selecting two defective smartphones from a shipment -/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / (total : ℚ) * ((defective - 1) : ℚ) / ((total - 1) : ℚ)

/-- Theorem stating the probability of selecting two defective smartphones -/
theorem probability_two_defective_shipment :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ 
  |probability_two_defective 240 84 - 1216/10000| < ε :=
sorry

end NUMINAMATH_CALUDE_probability_two_defective_shipment_l3724_372407


namespace NUMINAMATH_CALUDE_complement_to_set_l3724_372438

def U : Finset Nat := {1,2,3,4,5,6,7,8}

theorem complement_to_set (B : Finset Nat) :
  (U \ B = {1,3}) → (B = {2,4,5,6,7,8}) := by
  sorry

end NUMINAMATH_CALUDE_complement_to_set_l3724_372438


namespace NUMINAMATH_CALUDE_garden_length_l3724_372497

/-- Given a rectangular garden with area 120 m², if reducing its length by 2m results in a square,
    then the original length of the garden is 12 meters. -/
theorem garden_length (length width : ℝ) : 
  length * width = 120 →
  (length - 2) * (length - 2) = width * (length - 2) →
  length = 12 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l3724_372497


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l3724_372487

theorem polynomial_identity_sum_of_squares : 
  ∀ (p q r s t u : ℤ), 
  (∀ y, 729 * y^3 + 64 = (p * y^2 + q * y + r) * (s * y^2 + t * y + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 543106 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l3724_372487


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l3724_372419

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

theorem symmetry_coordinates :
  let A : Point := ⟨3, -2⟩
  let A' : Point := ⟨-3, 2⟩
  symmetricToOrigin A A' → A'.x = -3 ∧ A'.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l3724_372419


namespace NUMINAMATH_CALUDE_square_of_ten_n_plus_five_l3724_372476

theorem square_of_ten_n_plus_five (n : ℕ) : (10 * n + 5)^2 = 100 * n * (n + 1) + 25 := by
  sorry

#eval (10 * 199 + 5)^2  -- Should output 3980025

end NUMINAMATH_CALUDE_square_of_ten_n_plus_five_l3724_372476


namespace NUMINAMATH_CALUDE_cubic_local_min_implies_a_range_l3724_372455

/-- A function f has a local minimum in the interval (1, 2) -/
def has_local_min_in_interval (f : ℝ → ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 2 ∧ ∀ y, 1 < y ∧ y < 2 → f x ≤ f y

/-- The cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + a

theorem cubic_local_min_implies_a_range :
  ∀ a : ℝ, has_local_min_in_interval (f a) → 1 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_cubic_local_min_implies_a_range_l3724_372455


namespace NUMINAMATH_CALUDE_science_club_committee_probability_l3724_372443

theorem science_club_committee_probability : 
  let total_members : ℕ := 24
  let boys : ℕ := 12
  let girls : ℕ := 12
  let committee_size : ℕ := 5
  let total_combinations := Nat.choose total_members committee_size
  let all_boys_combinations := Nat.choose boys committee_size
  let all_girls_combinations := Nat.choose girls committee_size
  let favorable_combinations := total_combinations - (all_boys_combinations + all_girls_combinations)
  favorable_combinations / total_combinations = 1705 / 1771 :=
by
  sorry

end NUMINAMATH_CALUDE_science_club_committee_probability_l3724_372443


namespace NUMINAMATH_CALUDE_negation_equivalence_l3724_372450

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3724_372450


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l3724_372448

/-- For any positive real number a not equal to 1, 
    the function f(x) = a^(x-3) + 1 passes through the point (3, 2) -/
theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) + 1
  f 3 = 2 := by sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l3724_372448


namespace NUMINAMATH_CALUDE_modulus_of_z_l3724_372471

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = Complex.abs (2 + Complex.I) + 2 * Complex.I) : 
  Complex.abs z = 3 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3724_372471


namespace NUMINAMATH_CALUDE_N2O5_molecular_weight_l3724_372480

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in N2O5 -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in N2O5 -/
def num_O : ℕ := 5

/-- The molecular weight of N2O5 in g/mol -/
def molecular_weight_N2O5 : ℝ :=
  (num_N : ℝ) * atomic_weight_N + (num_O : ℝ) * atomic_weight_O

theorem N2O5_molecular_weight :
  molecular_weight_N2O5 = 108.02 := by
  sorry

end NUMINAMATH_CALUDE_N2O5_molecular_weight_l3724_372480


namespace NUMINAMATH_CALUDE_soda_cans_with_tax_l3724_372420

/-- Given:
  S : number of cans bought for Q quarters
  Q : number of quarters for S cans
  t : tax rate as a fraction of 1
  D : number of dollars available
-/
theorem soda_cans_with_tax (S Q : ℕ) (t : ℚ) (D : ℕ) :
  let cans_purchasable := (4 * D * S * (1 + t)) / Q
  cans_purchasable = (4 * D * S * (1 + t)) / Q :=
by sorry

end NUMINAMATH_CALUDE_soda_cans_with_tax_l3724_372420


namespace NUMINAMATH_CALUDE_train_crossing_time_l3724_372454

/-- The time taken for a train to cross a platform of equal length -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 900 →
  train_speed_kmh = 108 →
  (2 * train_length) / (train_speed_kmh * 1000 / 3600) = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3724_372454


namespace NUMINAMATH_CALUDE_triangle_arctangent_sum_l3724_372425

/-- In a triangle ABC with sides a, b, c, arbitrary angle C, and positive real number k,
    under certain conditions, the sum of two specific arctangents equals π/4. -/
theorem triangle_arctangent_sum (a b c k : ℝ) (h1 : k > 0) : 
  ∃ (h : Set ℝ), h.Nonempty ∧ ∀ (x : ℝ), x ∈ h → 
    Real.arctan (a / (b + c + k)) + Real.arctan (b / (a + c + k)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_arctangent_sum_l3724_372425


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3724_372418

theorem min_sum_of_squares (x y : ℝ) (h : (x + 3) * (y - 3) = 0) :
  ∃ (m : ℝ), (∀ a b : ℝ, (a + 3) * (b - 3) = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 18 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3724_372418


namespace NUMINAMATH_CALUDE_constant_molecular_weight_l3724_372485

/-- The molecular weight of an acid in g/mol -/
def molecular_weight : ℝ := 792

/-- The number of moles of the acid -/
def num_moles : ℝ := 9

/-- Theorem stating that the molecular weight remains constant regardless of the number of moles -/
theorem constant_molecular_weight : 
  ∀ n : ℝ, n > 0 → molecular_weight = molecular_weight := by sorry

end NUMINAMATH_CALUDE_constant_molecular_weight_l3724_372485


namespace NUMINAMATH_CALUDE_subtracted_number_l3724_372491

theorem subtracted_number (x : ℝ) : x = 7 → 4 * 5.0 - x = 13 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l3724_372491


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3724_372401

def A : Set ℝ := {-1, 0, 1}
def B (a : ℝ) : Set ℝ := {a + 1, 2 * a}

theorem intersection_implies_a_value (a : ℝ) :
  A ∩ B a = {0} → a = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3724_372401


namespace NUMINAMATH_CALUDE_angle_of_inclination_negative_slope_one_l3724_372490

/-- The angle of inclination of a line given by the equation x + y + 3 = 0 is 3π/4 -/
theorem angle_of_inclination_negative_slope_one (x y : ℝ) :
  x + y + 3 = 0 → Real.arctan (-1) = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_negative_slope_one_l3724_372490


namespace NUMINAMATH_CALUDE_min_value_theorem_l3724_372403

theorem min_value_theorem (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 4*m + n = 1) :
  (4/m + 1/n) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3724_372403


namespace NUMINAMATH_CALUDE_segment_length_given_ratio_points_l3724_372428

/-- Represents a point on a line segment -/
structure PointOnSegment (A B : ℝ) where
  position : ℝ
  h1 : A ≤ position
  h2 : position ≤ B

/-- The length of a line segment -/
def segmentLength (A B : ℝ) : ℝ := B - A

theorem segment_length_given_ratio_points 
  (A B : ℝ) 
  (P Q : PointOnSegment A B)
  (h_order : A < P.position ∧ P.position < Q.position ∧ Q.position < B)
  (h_P_ratio : P.position - A = 3/8 * (B - A))
  (h_Q_ratio : Q.position - A = 2/5 * (B - A))
  (h_PQ_length : Q.position - P.position = 3)
  : segmentLength A B = 120 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_given_ratio_points_l3724_372428


namespace NUMINAMATH_CALUDE_five_topping_pizzas_l3724_372435

theorem five_topping_pizzas (n k : ℕ) : n = 8 → k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_topping_pizzas_l3724_372435


namespace NUMINAMATH_CALUDE_ratio_from_mean_ratio_l3724_372434

theorem ratio_from_mean_ratio {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a + b) / (2 * Real.sqrt (a * b)) = 25 / 24 →
  a / b = 16 / 9 ∨ a / b = 9 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_from_mean_ratio_l3724_372434


namespace NUMINAMATH_CALUDE_triangle_coloring_ways_l3724_372469

-- Define the number of colors
def num_colors : ℕ := 4

-- Define a triangle as having 3 vertices
def triangle_vertices : ℕ := 3

-- Function to calculate the number of ways to color the triangle
def color_triangle_ways : ℕ :=
  num_colors * (num_colors - 1) * (num_colors - 2)

-- Theorem statement
theorem triangle_coloring_ways :
  color_triangle_ways = 24 :=
sorry

end NUMINAMATH_CALUDE_triangle_coloring_ways_l3724_372469


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3724_372423

/-- A circle with center (a, a) and radius 1 -/
def Circle (a : ℝ) := {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - a)^2 = 1}

/-- The line y = 2x -/
def Line := {p : ℝ × ℝ | p.2 = 2 * p.1}

/-- The intersection points of the circle and the line -/
def Intersection (a : ℝ) := Circle a ∩ Line

/-- The area of the triangle formed by the center of the circle and two points -/
def TriangleArea (center : ℝ × ℝ) (p q : ℝ × ℝ) : ℝ := sorry

theorem circle_line_intersection (a : ℝ) :
  a > 0 →
  ∃ (p q : ℝ × ℝ), p ∈ Intersection a ∧ q ∈ Intersection a ∧ p ≠ q ∧
  TriangleArea (a, a) p q = 1/2 →
  a = Real.sqrt 10 / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3724_372423


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l3724_372466

/-- A taxi fare system with a fixed starting fee and a proportional amount per mile -/
structure TaxiFare where
  startingFee : ℝ
  costPerMile : ℝ

/-- Calculate the total fare for a given distance -/
def calculateFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.startingFee + tf.costPerMile * distance

theorem taxi_fare_calculation (tf : TaxiFare) 
  (h1 : tf.startingFee = 20)
  (h2 : calculateFare tf 60 = 150)
  : calculateFare tf 80 = 193.33 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l3724_372466


namespace NUMINAMATH_CALUDE_quadratic_roots_d_value_l3724_372400

theorem quadratic_roots_d_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) →
  d = 9.8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_d_value_l3724_372400


namespace NUMINAMATH_CALUDE_sams_book_count_l3724_372424

/-- The total number of books Sam bought --/
def total_books (a m c f s : ℝ) : ℝ := a + m + c + f + s

/-- Theorem stating the total number of books Sam bought --/
theorem sams_book_count :
  ∀ (a m c f s : ℝ),
    a = 13.0 →
    m = 17.0 →
    c = 15.0 →
    f = 10.0 →
    s = 2 * a →
    total_books a m c f s = 81.0 := by
  sorry

end NUMINAMATH_CALUDE_sams_book_count_l3724_372424


namespace NUMINAMATH_CALUDE_matts_age_l3724_372464

theorem matts_age (john_age matt_age : ℕ) 
  (h1 : matt_age = 4 * john_age - 3)
  (h2 : matt_age + john_age = 52) :
  matt_age = 41 := by
  sorry

end NUMINAMATH_CALUDE_matts_age_l3724_372464


namespace NUMINAMATH_CALUDE_tom_family_invitation_l3724_372461

theorem tom_family_invitation :
  ∀ (total_plates : ℕ) (days : ℕ) (siblings : ℕ) (meals_per_day : ℕ) (plates_per_meal : ℕ),
    total_plates = 144 →
    days = 4 →
    siblings = 3 →
    meals_per_day = 3 →
    plates_per_meal = 2 →
    ∃ (family_members : ℕ),
      family_members = (total_plates / (days * meals_per_day * plates_per_meal)) - (siblings + 1) ∧
      family_members = 2 :=
by sorry

end NUMINAMATH_CALUDE_tom_family_invitation_l3724_372461


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l3724_372433

theorem gcd_of_polynomial_and_multiple (x : ℤ) (h : ∃ k : ℤ, x = 46200 * k) :
  let f := fun (x : ℤ) => (3*x + 5) * (5*x + 3) * (11*x + 6) * (x + 11)
  Int.gcd (f x) x = 990 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l3724_372433


namespace NUMINAMATH_CALUDE_well_volume_approximation_l3724_372416

/-- The volume of a cylinder with diameter 6 meters and height 24 meters is approximately 678.58464 cubic meters. -/
theorem well_volume_approximation :
  let diameter : ℝ := 6
  let height : ℝ := 24
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * height
  ∃ ε > 0, |volume - 678.58464| < ε :=
by sorry

end NUMINAMATH_CALUDE_well_volume_approximation_l3724_372416


namespace NUMINAMATH_CALUDE_profit_share_b_is_1600_l3724_372482

/-- Represents the profit share calculation for a business venture --/
structure ProfitShare where
  investment_a : ℕ
  investment_b : ℕ
  investment_c : ℕ
  difference_ac : ℕ

/-- Calculates the profit share of partner b given the investments and difference between a and c's shares --/
def calculate_share_b (ps : ProfitShare) : ℕ :=
  sorry

/-- Theorem stating that given the specific investments and difference, b's share is 1600 --/
theorem profit_share_b_is_1600 :
  ∀ (ps : ProfitShare),
    ps.investment_a = 8000 ∧
    ps.investment_b = 10000 ∧
    ps.investment_c = 12000 ∧
    ps.difference_ac = 640 →
    calculate_share_b ps = 1600 :=
  sorry

end NUMINAMATH_CALUDE_profit_share_b_is_1600_l3724_372482


namespace NUMINAMATH_CALUDE_opposite_roots_implies_n_eq_neg_two_l3724_372463

/-- The equation has roots that are numerically equal but of opposite signs -/
def has_opposite_roots (k b d e n : ℝ) : Prop :=
  ∃ x : ℝ, (k * x^2 - b * x) / (d * x - e) = (n - 2) / (n + 2) ∧
            ∃ y : ℝ, y = -x ∧ (k * y^2 - b * y) / (d * y - e) = (n - 2) / (n + 2)

/-- Theorem stating that if the equation has roots that are numerically equal but of opposite signs, then n = -2 -/
theorem opposite_roots_implies_n_eq_neg_two (k b d e : ℝ) :
  ∀ n : ℝ, has_opposite_roots k b d e n → n = -2 :=
by sorry

end NUMINAMATH_CALUDE_opposite_roots_implies_n_eq_neg_two_l3724_372463


namespace NUMINAMATH_CALUDE_problem_solution_l3724_372459

/-- The probability that student A solves the problem -/
def prob_A : ℚ := 1/5

/-- The probability that student B solves the problem -/
def prob_B : ℚ := 1/4

/-- The probability that student C solves the problem -/
def prob_C : ℚ := 1/3

/-- The probability that exactly two students solve the problem -/
def prob_two_solve : ℚ := 
  prob_A * prob_B * (1 - prob_C) + 
  prob_A * prob_C * (1 - prob_B) + 
  (1 - prob_A) * prob_B * prob_C

/-- The probability that the problem is not solved -/
def prob_not_solved : ℚ := (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- The probability that the problem is solved -/
def prob_solved : ℚ := 1 - prob_not_solved

theorem problem_solution : 
  prob_two_solve = 3/20 ∧ prob_solved = 3/5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3724_372459


namespace NUMINAMATH_CALUDE_x_eight_plus_x_four_plus_one_eq_zero_l3724_372429

theorem x_eight_plus_x_four_plus_one_eq_zero 
  (x : ℂ) (h : x^2 + x + 1 = 0) : x^8 + x^4 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_eight_plus_x_four_plus_one_eq_zero_l3724_372429


namespace NUMINAMATH_CALUDE_sum_of_ages_l3724_372442

-- Define variables for current ages
def divya_age : ℕ := 5
def nacho_age : ℕ := 25
def samantha_age : ℕ := 50

-- Theorem statement
theorem sum_of_ages : 
  -- Conditions
  (divya_age + 5 = (nacho_age + 5) / 3) ∧ 
  (samantha_age - nacho_age = 8) ∧
  (samantha_age = 2 * nacho_age) ∧
  (divya_age = 5) →
  -- Conclusion
  divya_age + nacho_age + samantha_age = 80 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3724_372442


namespace NUMINAMATH_CALUDE_problem_solution_l3724_372408

theorem problem_solution (x : ℝ) :
  x - Real.sqrt (x^2 + 1) + 1 / (x + Real.sqrt (x^2 + 1)) = 28 →
  x^2 - Real.sqrt (x^4 + 1) + 1 / (x^2 - Real.sqrt (x^4 + 1)) = -2 * Real.sqrt 38026 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3724_372408


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3724_372494

theorem ellipse_foci_distance (a b : ℝ) (ha : a = 8) (hb : b = 3) :
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 55 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3724_372494


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l3724_372465

theorem right_triangle_side_length 
  (A B C : ℝ) (AB BC AC : ℝ) :
  -- Triangle ABC is right-angled at A
  A + B + C = π / 2 →
  -- BC = 10
  BC = 10 →
  -- tan C = 3cos B
  Real.tan C = 3 * Real.cos B →
  -- AB² + AC² = BC²
  AB^2 + AC^2 = BC^2 →
  -- AB = (20√2)/3
  AB = 20 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l3724_372465


namespace NUMINAMATH_CALUDE_seating_theorem_l3724_372496

/-- The number of ways to arrange 5 boys and 4 girls in a row of 9 chairs such that at least 2 boys are next to each other -/
def seating_arrangements (num_boys num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating that the number of seating arrangements for 5 boys and 4 girls with at least 2 boys next to each other is 359000 -/
theorem seating_theorem :
  seating_arrangements 5 4 = 359000 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l3724_372496


namespace NUMINAMATH_CALUDE_f_of_g_of_3_equals_29_l3724_372449

def f (x : ℝ) : ℝ := 3 * x - 4

def g (x : ℝ) : ℝ := 2 * x + 3

theorem f_of_g_of_3_equals_29 : f (2 + g 3) = 29 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_of_3_equals_29_l3724_372449


namespace NUMINAMATH_CALUDE_smallest_factor_of_32_with_sum_3_l3724_372439

theorem smallest_factor_of_32_with_sum_3 (a b c : Int) : 
  a * b * c = 32 → a + b + c = 3 → min a (min b c) = -4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_of_32_with_sum_3_l3724_372439


namespace NUMINAMATH_CALUDE_triangle_angle_inequalities_l3724_372412

-- Define a structure for a triangle's angles
structure TriangleAngles where
  a : Real
  b : Real
  c : Real
  sum_is_pi : a + b + c = Real.pi

-- Theorem statement
theorem triangle_angle_inequalities (t : TriangleAngles) :
  (Real.sin t.a + Real.sin t.b + Real.sin t.c ≤ 3 * Real.sqrt 3 / 2) ∧
  (Real.cos (t.a / 2) + Real.cos (t.b / 2) + Real.cos (t.c / 2) ≤ 3 * Real.sqrt 3 / 2) ∧
  (Real.cos t.a * Real.cos t.b * Real.cos t.c ≤ 1 / 8) ∧
  (Real.sin (2 * t.a) + Real.sin (2 * t.b) + Real.sin (2 * t.c) ≤ Real.sin t.a + Real.sin t.b + Real.sin t.c) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_inequalities_l3724_372412


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3724_372458

theorem quadratic_equation_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (4 * x₁^2 - 2 * x₁ = (1/4 : ℝ)) ∧ (4 * x₂^2 - 2 * x₂ = (1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3724_372458


namespace NUMINAMATH_CALUDE_f_property_f_expression_l3724_372432

-- Define the function f
def f : ℝ → ℝ := λ x ↦ x^2 - 4

-- State the theorem
theorem f_property : ∀ x : ℝ, f (1 + x) = x^2 + 2*x - 1 := by
  sorry

-- Prove that f(x) = x^2 - 4
theorem f_expression : ∀ x : ℝ, f x = x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_f_property_f_expression_l3724_372432


namespace NUMINAMATH_CALUDE_polynomial_conclusions_l3724_372472

theorem polynomial_conclusions (x a : ℝ) : 
  let M : ℝ → ℝ := λ x => 2 * x^2 - 3 * x - 2
  let N : ℝ → ℝ := λ x => x^2 - a * x + 3
  (∃! i : Fin 3, 
    (i = 0 → (M x = 0 → (13 * x) / (x^2 - 3 * x - 1) = 26 / 3)) ∧
    (i = 1 → (a = -3 → (∀ y ≥ 4, M y - N y ≥ -14) → (∃ z ≥ 4, M z - N z = -14))) ∧
    (i = 2 → (a = 0 → (M x * N x = 0 → ∃ r s : ℝ, r ≠ s ∧ M r = 0 ∧ M s = 0))))
  := by sorry

end NUMINAMATH_CALUDE_polynomial_conclusions_l3724_372472


namespace NUMINAMATH_CALUDE_b_investment_is_4200_l3724_372405

/-- Represents the investment and profit details of a partnership business -/
structure BusinessPartnership where
  a_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Calculates B's investment given the partnership details -/
def calculate_b_investment (bp : BusinessPartnership) : ℕ :=
  bp.total_profit * bp.a_investment / bp.a_profit_share - bp.a_investment - bp.c_investment

/-- Theorem stating that B's investment is 4200 given the specified conditions -/
theorem b_investment_is_4200 (bp : BusinessPartnership) 
  (h1 : bp.a_investment = 6300)
  (h2 : bp.c_investment = 10500)
  (h3 : bp.total_profit = 13000)
  (h4 : bp.a_profit_share = 3900) :
  calculate_b_investment bp = 4200 := by
  sorry

#eval calculate_b_investment ⟨6300, 10500, 13000, 3900⟩

end NUMINAMATH_CALUDE_b_investment_is_4200_l3724_372405


namespace NUMINAMATH_CALUDE_bounce_count_is_seven_l3724_372427

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The number of bounces a ball makes before returning to a vertex -/
def num_bounces (t : Triangle) (Y : ℝ × ℝ) : ℕ := sorry

/-- The theorem stating that the number of bounces is 7 for the given triangle and point -/
theorem bounce_count_is_seven :
  let t : Triangle := { A := (0, 0), B := (7, 0), C := (7/2, 3*Real.sqrt 3/2) }
  let Y : ℝ × ℝ := (7/2, 3*Real.sqrt 3/2)
  num_bounces t Y = 7 := by sorry

end NUMINAMATH_CALUDE_bounce_count_is_seven_l3724_372427


namespace NUMINAMATH_CALUDE_survey_analysis_l3724_372488

structure SurveyData where
  total : Nat
  aged_50_below_not_return : Nat
  aged_50_above_return : Nat
  aged_50_above_total : Nat

def chi_square (a b c d : Nat) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem survey_analysis (data : SurveyData) 
  (h1 : data.total = 100)
  (h2 : data.aged_50_below_not_return = 55)
  (h3 : data.aged_50_above_return = 15)
  (h4 : data.aged_50_above_total = 40) :
  let a := data.total - data.aged_50_above_total - data.aged_50_below_not_return
  let b := data.aged_50_below_not_return
  let c := data.aged_50_above_return
  let d := data.aged_50_above_total - data.aged_50_above_return
  (c : ℚ) / data.aged_50_above_total = 3 / 8 ∧ 
  chi_square a b c d > 10828 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_survey_analysis_l3724_372488


namespace NUMINAMATH_CALUDE_a_10_ends_with_many_nines_l3724_372470

/-- Definition of the sequence aₙ -/
def a : ℕ → ℕ
  | 0 => 9
  | n + 1 => 3 * (a n)^4 + 4 * (a n)^3

/-- Theorem: a₁₀ ends with more than 1000 nines -/
theorem a_10_ends_with_many_nines :
  a 10 % (10^1001) = 10^1001 - 1 := by
  sorry

end NUMINAMATH_CALUDE_a_10_ends_with_many_nines_l3724_372470


namespace NUMINAMATH_CALUDE_consecutive_sum_not_power_of_two_l3724_372481

theorem consecutive_sum_not_power_of_two (n k x : ℕ) (h : n > 1) :
  (n * (n + 2 * k - 1)) / 2 ≠ 2^x :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_not_power_of_two_l3724_372481


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l3724_372406

theorem angle_sum_in_circle (x : ℝ) : 
  3 * x + 6 * x + 2 * x + x = 360 → x = 30 := by
  sorry

#check angle_sum_in_circle

end NUMINAMATH_CALUDE_angle_sum_in_circle_l3724_372406


namespace NUMINAMATH_CALUDE_max_sum_given_quadratic_l3724_372426

theorem max_sum_given_quadratic (a b : ℝ) (h : a^2 - a*b + b^2 = 1) : a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_quadratic_l3724_372426


namespace NUMINAMATH_CALUDE_second_account_interest_rate_l3724_372445

/-- Proves that the interest rate of the second account is 0.1 given the problem conditions -/
theorem second_account_interest_rate 
  (total_investment : ℝ)
  (first_account_rate : ℝ)
  (first_account_investment : ℝ)
  (h_total : total_investment = 7200)
  (h_first_rate : first_account_rate = 0.08)
  (h_first_inv : first_account_investment = 4000)
  (h_equal_interest : first_account_rate * first_account_investment = 
    (total_investment - first_account_investment) * (10 / 100)) :
  (10 : ℝ) / 100 = (first_account_rate * first_account_investment) / (total_investment - first_account_investment) :=
by sorry

end NUMINAMATH_CALUDE_second_account_interest_rate_l3724_372445


namespace NUMINAMATH_CALUDE_flower_seedling_problem_l3724_372475

/-- Represents the unit price of flower seedlings --/
structure FlowerPrice where
  typeA : ℝ
  typeB : ℝ

/-- Represents the cost function for purchasing flower seedlings --/
def cost_function (p : FlowerPrice) (a : ℝ) : ℝ :=
  p.typeA * (12 - a) + (p.typeB - a) * a

/-- The theorem statement for the flower seedling problem --/
theorem flower_seedling_problem (p : FlowerPrice) :
  (3 * p.typeA + 5 * p.typeB = 210) →
  (4 * p.typeA + 10 * p.typeB = 380) →
  p.typeA = 20 ∧ p.typeB = 30 ∧
  ∃ (a_min a_max : ℝ), 0 < a_min ∧ a_min < 12 ∧ 0 < a_max ∧ a_max < 12 ∧
    ∀ (a : ℝ), 0 < a ∧ a < 12 →
      229 ≤ cost_function p a ∧ cost_function p a ≤ 265 ∧
      cost_function p a_min = 229 ∧ cost_function p a_max = 265 := by
  sorry

end NUMINAMATH_CALUDE_flower_seedling_problem_l3724_372475


namespace NUMINAMATH_CALUDE_complex_quadrant_l3724_372447

theorem complex_quadrant (z : ℂ) (h : (1 - I) / (z - 2) = 1 + I) : 
  0 < z.re ∧ z.im < 0 := by
sorry

end NUMINAMATH_CALUDE_complex_quadrant_l3724_372447


namespace NUMINAMATH_CALUDE_simplify_expression_l3724_372498

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3724_372498


namespace NUMINAMATH_CALUDE_average_of_DEF_l3724_372467

theorem average_of_DEF (D E F : ℚ) 
  (eq1 : 2003 * F - 4006 * D = 8012)
  (eq2 : 2003 * E + 6009 * D = 10010) :
  (D + E + F) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_DEF_l3724_372467


namespace NUMINAMATH_CALUDE_quadratic_equation_with_root_three_l3724_372495

theorem quadratic_equation_with_root_three :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 3*x = 0) ∧ (3 : ℝ) ∈ {x : ℝ | a * x^2 + b * x + c = 0} :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_root_three_l3724_372495


namespace NUMINAMATH_CALUDE_ipod_problem_l3724_372409

def problem (emmy_initial : ℕ) (emmy_lost : ℕ) (rosa_given_away : ℕ) : Prop :=
  let emmy_current := emmy_initial - emmy_lost
  let rosa_current := emmy_current / 3
  let rosa_initial := rosa_current + rosa_given_away
  emmy_current + rosa_current = 21

theorem ipod_problem : problem 25 9 4 := by
  sorry

end NUMINAMATH_CALUDE_ipod_problem_l3724_372409


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3724_372492

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 0 → x + 1/x ≥ 2)) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3724_372492


namespace NUMINAMATH_CALUDE_chord_length_exists_point_P_l3724_372453

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 16

-- Define point F
def F : ℝ × ℝ := (-2, 0)

-- Define the line x = -4
def line_x_eq_neg_4 (x y : ℝ) : Prop := x = -4

-- Theorem 1: Length of the chord
theorem chord_length :
  ∃ (G : ℝ × ℝ), C₁ G.1 G.2 →
  ∃ (T : ℝ × ℝ), line_x_eq_neg_4 T.1 T.2 →
  (G.1 - F.1 = T.1 - G.1 ∧ G.2 - F.2 = T.2 - G.2) →
  ∃ (chord_length : ℝ), chord_length = 7 :=
sorry

-- Theorem 2: Existence of point P
theorem exists_point_P :
  ∃ (P : ℝ × ℝ), P = (4, 0) ∧
  ∀ (G : ℝ × ℝ), C₁ G.1 G.2 →
  (G.1 - P.1)^2 + (G.2 - P.2)^2 = 4 * ((G.1 - F.1)^2 + (G.2 - F.2)^2) :=
sorry

end NUMINAMATH_CALUDE_chord_length_exists_point_P_l3724_372453


namespace NUMINAMATH_CALUDE_computer_price_decrease_l3724_372422

/-- The price of a computer after a certain number of years, given an initial price and a constant rate of decrease every two years. -/
def price_after_years (initial_price : ℝ) (decrease_rate : ℝ) (years : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ (years / 2)

/-- Theorem stating that a computer with an initial price of 8100 yuan, decreasing by one-third every two years, will cost 2400 yuan after 6 years. -/
theorem computer_price_decrease (initial_price : ℝ) (years : ℕ) :
  initial_price = 8100 →
  years = 6 →
  price_after_years initial_price (1/3) years = 2400 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_decrease_l3724_372422


namespace NUMINAMATH_CALUDE_construction_time_correct_l3724_372444

/-- Represents the construction project with given parameters -/
structure ConstructionProject where
  initialWorkers : ℕ
  additionalWorkers : ℕ
  additionalWorkersStartDay : ℕ
  delayWithoutAdditionalWorkers : ℕ

/-- The planned construction time in days -/
def plannedConstructionTime (project : ConstructionProject) : ℕ := 110

theorem construction_time_correct (project : ConstructionProject) 
  (h1 : project.initialWorkers = 100)
  (h2 : project.additionalWorkers = 100)
  (h3 : project.additionalWorkersStartDay = 10)
  (h4 : project.delayWithoutAdditionalWorkers = 90) :
  plannedConstructionTime project = 110 := by
  sorry

#check construction_time_correct

end NUMINAMATH_CALUDE_construction_time_correct_l3724_372444


namespace NUMINAMATH_CALUDE_brick_length_is_8_l3724_372404

-- Define the surface area function for a rectangular prism
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

-- Theorem statement
theorem brick_length_is_8 :
  ∃ (l : ℝ), l > 0 ∧ surface_area l 6 2 = 152 ∧ l = 8 :=
by sorry

end NUMINAMATH_CALUDE_brick_length_is_8_l3724_372404


namespace NUMINAMATH_CALUDE_series_sum_equals_half_l3724_372478

/-- The sum of the series Σ(3^(2^k) / (9^(2^k) - 1)) for k from 0 to infinity is equal to 1/2 -/
theorem series_sum_equals_half :
  (∑' k : ℕ, (3 ^ (2 ^ k)) / ((9 ^ (2 ^ k)) - 1)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_half_l3724_372478


namespace NUMINAMATH_CALUDE_sum_A_and_B_l3724_372474

theorem sum_A_and_B : 
  let B := 278 + 365 * 3
  let A := 20 * 100 + 87 * 10
  A + B = 4243 := by
sorry

end NUMINAMATH_CALUDE_sum_A_and_B_l3724_372474


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l3724_372414

theorem same_terminal_side_angle : ∃ (θ : ℝ), 
  θ ∈ Set.Icc (-2 * Real.pi) 0 ∧ 
  ∃ (k : ℤ), θ = (52 / 7 : ℝ) * Real.pi + 2 * k * Real.pi ∧
  θ = -(4 / 7 : ℝ) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l3724_372414


namespace NUMINAMATH_CALUDE_bob_shucking_rate_is_two_bob_shucking_rate_consistent_with_two_hours_l3724_372477

/-- Represents the rate at which Bob shucks oysters in oysters per minute -/
def bob_shucking_rate : ℚ :=
  10 / 5

theorem bob_shucking_rate_is_two :
  bob_shucking_rate = 2 :=
by
  -- Proof goes here
  sorry

theorem bob_shucking_rate_consistent_with_two_hours :
  bob_shucking_rate * 120 = 240 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_bob_shucking_rate_is_two_bob_shucking_rate_consistent_with_two_hours_l3724_372477


namespace NUMINAMATH_CALUDE_total_lives_calculation_l3724_372410

/-- Given 6 initial players, 9 additional players, and 5 lives per player,
    the total number of lives is 75. -/
theorem total_lives_calculation (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ)
    (h1 : initial_players = 6)
    (h2 : additional_players = 9)
    (h3 : lives_per_player = 5) :
    (initial_players + additional_players) * lives_per_player = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_calculation_l3724_372410


namespace NUMINAMATH_CALUDE_marie_glue_sticks_l3724_372446

theorem marie_glue_sticks :
  ∀ (allison_glue allison_paper marie_glue marie_paper : ℕ),
    allison_glue = marie_glue + 8 →
    marie_paper = 6 * allison_paper →
    marie_paper = 30 →
    allison_glue + allison_paper = 28 →
    marie_glue = 15 := by
  sorry

end NUMINAMATH_CALUDE_marie_glue_sticks_l3724_372446


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3724_372415

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I * z) = 1) : 
  Complex.abs (2 * z - 3) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3724_372415


namespace NUMINAMATH_CALUDE_largest_even_n_inequality_l3724_372421

theorem largest_even_n_inequality (n : ℕ) : 
  (n = 8 ∧ n % 2 = 0) ↔ 
  (∀ x : ℝ, (Real.sin x)^(2*n) + (Real.cos x)^(2*n) + (Real.tan x)^2 ≥ 1/n) ∧
  (∀ m : ℕ, m > n → m % 2 = 0 → 
    ∃ x : ℝ, (Real.sin x)^(2*m) + (Real.cos x)^(2*m) + (Real.tan x)^2 < 1/m) :=
by sorry

end NUMINAMATH_CALUDE_largest_even_n_inequality_l3724_372421


namespace NUMINAMATH_CALUDE_factorization_existence_l3724_372431

theorem factorization_existence : ∃ (a b c : ℤ), 
  (∀ x, (x - a) * (x - 10) + 1 = (x + b) * (x + c)) ∧ (a = 8 ∨ a = 12) := by
  sorry

end NUMINAMATH_CALUDE_factorization_existence_l3724_372431


namespace NUMINAMATH_CALUDE_a_travel_time_l3724_372441

-- Define the speed ratio of A to B
def speed_ratio : ℚ := 3 / 4

-- Define the time difference between A and B in hours
def time_difference : ℚ := 1 / 2

-- Theorem statement
theorem a_travel_time (t : ℚ) : 
  (t + time_difference) / t = 1 / speed_ratio → t + time_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_travel_time_l3724_372441


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3724_372451

/-- The complex number i(2-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Complex.I * (2 - Complex.I) = Complex.mk x y := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3724_372451


namespace NUMINAMATH_CALUDE_water_left_in_cooler_l3724_372457

/-- Proves that the amount of water left in the cooler after filling all cups is 84 ounces -/
theorem water_left_in_cooler : 
  let initial_gallons : ℕ := 3
  let ounces_per_cup : ℕ := 6
  let rows : ℕ := 5
  let chairs_per_row : ℕ := 10
  let ounces_per_gallon : ℕ := 128
  
  let total_chairs : ℕ := rows * chairs_per_row
  let initial_ounces : ℕ := initial_gallons * ounces_per_gallon
  let ounces_used : ℕ := total_chairs * ounces_per_cup
  let ounces_left : ℕ := initial_ounces - ounces_used

  ounces_left = 84 := by sorry

end NUMINAMATH_CALUDE_water_left_in_cooler_l3724_372457


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3724_372440

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 6
  let φ : ℝ := π / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (2 * Real.sqrt 6, Real.sqrt 2, 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3724_372440


namespace NUMINAMATH_CALUDE_isosceles_triangle_50_largest_angle_l3724_372493

/-- An isosceles triangle with one angle opposite an equal side measuring 50 degrees -/
structure IsoscelesTriangle50 where
  /-- The measure of one of the angles opposite an equal side -/
  angle_opposite_equal_side : ℝ
  /-- The measure of the largest angle in the triangle -/
  largest_angle : ℝ
  /-- Assertion that the angle opposite an equal side is 50 degrees -/
  h_angle_50 : angle_opposite_equal_side = 50

/-- 
Theorem: In an isosceles triangle where one of the angles opposite an equal side 
measures 50°, the largest angle measures 80°.
-/
theorem isosceles_triangle_50_largest_angle 
  (t : IsoscelesTriangle50) : t.largest_angle = 80 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_50_largest_angle_l3724_372493


namespace NUMINAMATH_CALUDE_sarahs_test_score_l3724_372437

theorem sarahs_test_score 
  (hunter_score : ℕ) 
  (john_score : ℕ) 
  (grant_score : ℕ) 
  (sarah_score : ℕ) 
  (hunter_score_val : hunter_score = 45)
  (john_score_def : john_score = 2 * hunter_score)
  (grant_score_def : grant_score = john_score + 10)
  (sarah_score_def : sarah_score = grant_score - 5) :
  sarah_score = 95 := by
sorry

end NUMINAMATH_CALUDE_sarahs_test_score_l3724_372437


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l3724_372430

theorem unique_solution_quadratic_system (y : ℚ) 
  (eq1 : 10 * y^2 + 9 * y - 2 = 0)
  (eq2 : 30 * y^2 + 77 * y - 14 = 0) : 
  y = 1/5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l3724_372430


namespace NUMINAMATH_CALUDE_polynomial_common_factor_l3724_372456

-- Define variables
variable (x y m n : ℝ)

-- Define the polynomial
def polynomial (x y m n : ℝ) : ℝ := 4*x*(m-n) + 2*y*(m-n)^2

-- Define the common factor
def common_factor (m n : ℝ) : ℝ := 2*(m-n)

-- Theorem statement
theorem polynomial_common_factor :
  ∃ (a b : ℝ), polynomial x y m n = common_factor m n * (a*x + b*y*(m-n)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_common_factor_l3724_372456


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3724_372413

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_3_eq_9 : a 3 = 9
  S_3_eq_33 : (a 1 + a 2 + a 3) = 33

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  ∃ (d : ℝ),
    (∀ n, seq.a n = seq.a 1 + (n - 1) * d) ∧
    d = -2 ∧
    (∀ n, seq.a n = 15 - 2 * n) ∧
    (∃ n_max : ℕ, ∀ n : ℕ, 
      (n * (seq.a 1 + seq.a n) / 2) ≤ (n_max * (seq.a 1 + seq.a n_max) / 2) ∧
      n_max * (seq.a 1 + seq.a n_max) / 2 = 49) :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3724_372413


namespace NUMINAMATH_CALUDE_total_money_is_36000_l3724_372483

/-- The number of phones Vivienne has -/
def vivienne_phones : ℕ := 40

/-- The number of additional phones Aliyah has compared to Vivienne -/
def aliyah_extra_phones : ℕ := 10

/-- The price at which each phone is sold -/
def price_per_phone : ℕ := 400

/-- The total amount of money Aliyah and Vivienne have together after selling their phones -/
def total_money : ℕ := (vivienne_phones + (vivienne_phones + aliyah_extra_phones)) * price_per_phone

/-- Theorem stating that the total amount of money Aliyah and Vivienne have together is $36,000 -/
theorem total_money_is_36000 : total_money = 36000 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_36000_l3724_372483


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3724_372452

/-- Given a geometric sequence {a_n} where a₁ = 1 and a₅ = 81, prove that a₃ = 9 -/
theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_a1 : a 1 = 1) 
  (h_a5 : a 5 = 81) : 
  a 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3724_372452


namespace NUMINAMATH_CALUDE_largest_number_l3724_372489

theorem largest_number (a b c d e : ℚ) 
  (sum1 : a + b + c + d = 210)
  (sum2 : a + b + c + e = 230)
  (sum3 : a + b + d + e = 250)
  (sum4 : a + c + d + e = 270)
  (sum5 : b + c + d + e = 290) :
  max a (max b (max c (max d e))) = 102.5 := by
sorry

end NUMINAMATH_CALUDE_largest_number_l3724_372489


namespace NUMINAMATH_CALUDE_square_root_equality_implies_zero_product_l3724_372460

theorem square_root_equality_implies_zero_product (x y z : ℝ) 
  (h : Real.sqrt (x - y + z) = Real.sqrt x - Real.sqrt y + Real.sqrt z) : 
  (x - y) * (y - z) * (z - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_implies_zero_product_l3724_372460


namespace NUMINAMATH_CALUDE_notebook_marker_cost_l3724_372402

/-- Given the cost of notebooks and markers, prove the cost of a specific combination -/
theorem notebook_marker_cost (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7.30)
  (h2 : 5 * x + 3 * y = 11.65) : 
  2 * x + y = 4.35 := by
  sorry

end NUMINAMATH_CALUDE_notebook_marker_cost_l3724_372402


namespace NUMINAMATH_CALUDE_multiplication_division_equality_l3724_372479

theorem multiplication_division_equality : (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 800 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_equality_l3724_372479


namespace NUMINAMATH_CALUDE_distributeBallsWithRedBox_eq_1808_l3724_372473

/-- Number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := Nat.choose n r

/-- Number of ways to distribute 7 distinguishable balls into 3 distinguishable boxes,
    where one box (red) can contain at most 3 balls -/
def distributeBallsWithRedBox : ℕ :=
  choose 7 3 * distribute 4 2 +
  choose 7 2 * distribute 5 2 +
  choose 7 1 * distribute 6 2 +
  distribute 7 2

theorem distributeBallsWithRedBox_eq_1808 :
  distributeBallsWithRedBox = 1808 := by sorry

end NUMINAMATH_CALUDE_distributeBallsWithRedBox_eq_1808_l3724_372473


namespace NUMINAMATH_CALUDE_happy_children_count_l3724_372499

theorem happy_children_count (total : ℕ) (sad : ℕ) (neither : ℕ) (boys : ℕ) (girls : ℕ) 
  (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ) : ℕ :=
  by
  -- Define the given conditions
  have h1 : total = 60 := by sorry
  have h2 : sad = 10 := by sorry
  have h3 : neither = 20 := by sorry
  have h4 : boys = 16 := by sorry
  have h5 : girls = 44 := by sorry
  have h6 : happy_boys = 6 := by sorry
  have h7 : sad_girls = 4 := by sorry
  have h8 : neither_boys = 4 := by sorry

  -- Prove that the number of happy children is 30
  have happy_children : ℕ := total - (sad + neither)
  exact happy_children

end NUMINAMATH_CALUDE_happy_children_count_l3724_372499


namespace NUMINAMATH_CALUDE_ball_collection_theorem_l3724_372484

theorem ball_collection_theorem (r b y : ℕ) : 
  b + y = 9 →
  r + y = 5 →
  r + b = 6 →
  r + b + y = 10 := by
sorry

end NUMINAMATH_CALUDE_ball_collection_theorem_l3724_372484


namespace NUMINAMATH_CALUDE_distance_on_line_l3724_372486

/-- Given two points (a, b) and (c, d) on the line x + y = px + q,
    prove that the distance between them is |a-c|√(1 + (p-1)²) -/
theorem distance_on_line (p q a b c d : ℝ) :
  (a + b = p * a + q) →
  (c + d = p * c + q) →
  Real.sqrt ((c - a)^2 + (d - b)^2) = |a - c| * Real.sqrt (1 + (p - 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l3724_372486


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3724_372411

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 3*x + 10 > 0} = Set.Ioo (-2) 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3724_372411


namespace NUMINAMATH_CALUDE_two_integers_sum_l3724_372468

theorem two_integers_sum (x y : ℕ+) : 
  x - y = 4 → x * y = 192 → x + y = 28 := by sorry

end NUMINAMATH_CALUDE_two_integers_sum_l3724_372468


namespace NUMINAMATH_CALUDE_union_A_complement_B_l3724_372436

def I : Set ℤ := {x | |x| < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_A_complement_B : A ∪ (I \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_l3724_372436


namespace NUMINAMATH_CALUDE_min_value_theorem_l3724_372462

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (4 * x / (x + 3 * y)) + (3 * y / x) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3724_372462


namespace NUMINAMATH_CALUDE_unique_valid_sequence_l3724_372417

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ i j k, a i + a j ≠ a k) ∧
  (∀ m, ∃ k > m, a k = 2 * k - 1)

theorem unique_valid_sequence :
  ∀ a : ℕ → ℕ, is_valid_sequence a ↔ (∀ n, a n = 2 * n - 1) :=
sorry

end NUMINAMATH_CALUDE_unique_valid_sequence_l3724_372417
