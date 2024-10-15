import Mathlib

namespace NUMINAMATH_CALUDE_unpainted_cubes_not_multiple_of_painted_cubes_l3690_369028

theorem unpainted_cubes_not_multiple_of_painted_cubes (n : ℕ) (h : n ≥ 1) :
  ¬(6 * n^2 + 12 * n + 8 ∣ n^3) := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_not_multiple_of_painted_cubes_l3690_369028


namespace NUMINAMATH_CALUDE_curler_ratio_l3690_369013

theorem curler_ratio (total : ℕ) (pink : ℕ) (green : ℕ) (blue : ℕ) : 
  total = 16 → 
  pink = total / 4 → 
  green = 4 → 
  blue = total - pink - green →
  blue / pink = 2 := by
sorry

end NUMINAMATH_CALUDE_curler_ratio_l3690_369013


namespace NUMINAMATH_CALUDE_triangles_containing_center_l3690_369037

/-- Given a regular polygon with 2n+1 sides, this theorem states the number of triangles
    formed by the vertices of the polygon and containing the center of the polygon. -/
theorem triangles_containing_center (n : ℕ) :
  let sides := 2 * n + 1
  (sides.choose 3 : ℚ) - (sides : ℚ) * (n.choose 2 : ℚ) = n * (n + 1) * (2 * n + 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangles_containing_center_l3690_369037


namespace NUMINAMATH_CALUDE_triangle_isosceles_l3690_369073

/-- A triangle with sides a, b, and c satisfying the given equation is isosceles with c as the base -/
theorem triangle_isosceles (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : 1/a - 1/b + 1/c = 1/(a-b+c)) : a = c :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l3690_369073


namespace NUMINAMATH_CALUDE_initial_height_proof_l3690_369094

/-- Calculates the initial height of a person before a growth spurt -/
def initial_height (growth_rate : ℕ) (growth_period : ℕ) (final_height_feet : ℕ) : ℕ :=
  let final_height_inches := final_height_feet * 12
  let total_growth := growth_rate * growth_period
  final_height_inches - total_growth

/-- Theorem stating that given the specific growth conditions, 
    the initial height was 66 inches -/
theorem initial_height_proof : 
  initial_height 2 3 6 = 66 := by
  sorry

end NUMINAMATH_CALUDE_initial_height_proof_l3690_369094


namespace NUMINAMATH_CALUDE_curve_equation_l3690_369039

/-- Given a curve of the form ax^2 + by^2 = 2 passing through the points (0, 5/3) and (1, 1),
    prove that its equation is 16/25 * x^2 + 9/25 * y^2 = 1. -/
theorem curve_equation (a b : ℝ) (h1 : a * 0^2 + b * (5/3)^2 = 2) (h2 : a * 1^2 + b * 1^2 = 2) :
  ∃ (x y : ℝ), 16/25 * x^2 + 9/25 * y^2 = 1 ↔ a * x^2 + b * y^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_curve_equation_l3690_369039


namespace NUMINAMATH_CALUDE_weak_coffee_amount_is_one_l3690_369004

/-- The amount of coffee used per cup of water for weak coffee -/
def weak_coffee_amount : ℝ := 1

/-- The number of cups of each type of coffee made -/
def cups_per_type : ℕ := 12

/-- The total amount of coffee used in tablespoons -/
def total_coffee : ℕ := 36

/-- Theorem stating that the amount of coffee used per cup of water for weak coffee is 1 tablespoon -/
theorem weak_coffee_amount_is_one :
  weak_coffee_amount = 1 ∧
  cups_per_type * weak_coffee_amount + cups_per_type * (2 * weak_coffee_amount) = total_coffee :=
by sorry

end NUMINAMATH_CALUDE_weak_coffee_amount_is_one_l3690_369004


namespace NUMINAMATH_CALUDE_rice_problem_l3690_369043

theorem rice_problem (total : ℚ) : 
  (21 : ℚ) / 50 * total = 210 → total = 500 := by
  sorry

end NUMINAMATH_CALUDE_rice_problem_l3690_369043


namespace NUMINAMATH_CALUDE_flags_left_proof_l3690_369018

/-- Calculates the number of flags left after installation -/
def flags_left (circumference : ℕ) (interval : ℕ) (available_flags : ℕ) : ℕ :=
  available_flags - (circumference / interval)

/-- Theorem: Given the specific conditions, the number of flags left is 2 -/
theorem flags_left_proof :
  flags_left 200 20 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_flags_left_proof_l3690_369018


namespace NUMINAMATH_CALUDE_number_difference_l3690_369027

theorem number_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_number_difference_l3690_369027


namespace NUMINAMATH_CALUDE_triangle_properties_l3690_369091

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = t.a * Real.cos t.C + (Real.sqrt 3 / 3) * t.a * Real.sin t.C)
  (h2 : t.a = Real.sqrt 7)
  (h3 : t.b * t.c = 6) : 
  t.A = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3690_369091


namespace NUMINAMATH_CALUDE_propositions_truthfulness_l3690_369069

-- Define the properties
def isPositiveEven (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

-- Theorem statement
theorem propositions_truthfulness :
  (∃ n : ℕ, isPositiveEven n ∧ isPrime n) ∧
  (∃ n : ℕ, ¬isPrime n ∧ ¬isPositiveEven n) ∧
  (∃ n : ℕ, ¬isPositiveEven n ∧ ¬isPrime n) ∧
  (∀ n : ℕ, isPrime n → ¬isPositiveEven n) :=
sorry

end NUMINAMATH_CALUDE_propositions_truthfulness_l3690_369069


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3690_369095

theorem matrix_equation_solution : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 1, 3]
  N^3 - 3 • N^2 + 2 • N = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3690_369095


namespace NUMINAMATH_CALUDE_integer_divisibility_problem_l3690_369093

theorem integer_divisibility_problem (n : ℤ) :
  (5 ∣ (3 * n - 2)) ∧ (7 ∣ (2 * n + 1)) ↔ ∃ m : ℤ, n = 35 * m + 24 := by
  sorry

end NUMINAMATH_CALUDE_integer_divisibility_problem_l3690_369093


namespace NUMINAMATH_CALUDE_car_initial_speed_l3690_369068

/-- Represents a point on the road --/
inductive Point
| A
| B
| C
| D

/-- Represents the speed of the car at different segments --/
structure Speed where
  initial : ℝ
  fromBtoC : ℝ
  fromCtoD : ℝ

/-- Represents the distance between points --/
structure Distance where
  total : ℝ
  AtoB : ℝ
  BtoC : ℝ
  CtoD : ℝ

/-- Represents the travel time between points --/
structure TravelTime where
  BtoC : ℝ
  CtoD : ℝ

/-- The main theorem stating the conditions and the result to be proved --/
theorem car_initial_speed 
  (d : Distance)
  (s : Speed)
  (t : TravelTime)
  (h1 : d.total = 100)
  (h2 : d.total - d.AtoB = 0.5 * s.initial)
  (h3 : s.fromBtoC = s.initial - 10)
  (h4 : s.fromCtoD = s.initial - 20)
  (h5 : d.CtoD = 20)
  (h6 : t.BtoC = t.CtoD + 1/12)
  (h7 : d.BtoC / s.fromBtoC = t.BtoC)
  (h8 : d.CtoD / s.fromCtoD = t.CtoD)
  : s.initial = 100 := by
  sorry


end NUMINAMATH_CALUDE_car_initial_speed_l3690_369068


namespace NUMINAMATH_CALUDE_quadratic_equation_m_l3690_369034

/-- Given that (m+3)x^(m^2-7) + mx - 2 = 0 is a quadratic equation in x, prove that m = 3 -/
theorem quadratic_equation_m (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 3) * x^(m^2 - 7) + m * x - 2 = a * x^2 + b * x + c) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_l3690_369034


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l3690_369007

def first_four_primes : List Nat := [2, 3, 5, 7]

theorem arithmetic_mean_reciprocals_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length : ℚ) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l3690_369007


namespace NUMINAMATH_CALUDE_free_younger_son_time_l3690_369096

/-- Given a total number of tape strands and cutting rates for Hannah and her son,
    calculate the time needed to cut all strands. -/
def time_to_cut (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℚ :=
  (total_strands : ℚ) / ((hannah_rate + son_rate) : ℚ)

/-- Theorem stating that it takes 5 minutes to cut 45 strands of tape
    when Hannah cuts 7 strands per minute and her son cuts 2 strands per minute. -/
theorem free_younger_son_time :
  time_to_cut 45 7 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_free_younger_son_time_l3690_369096


namespace NUMINAMATH_CALUDE_probability_one_defective_six_two_l3690_369058

/-- The probability of selecting exactly one defective product from a set of products -/
def probability_one_defective (total : ℕ) (defective : ℕ) : ℚ :=
  let qualified := total - defective
  (defective.choose 1 * qualified.choose 1 : ℚ) / total.choose 2

/-- Given 6 products with 2 defective ones, the probability of selecting exactly one defective product is 8/15 -/
theorem probability_one_defective_six_two :
  probability_one_defective 6 2 = 8 / 15 := by
  sorry

#eval probability_one_defective 6 2

end NUMINAMATH_CALUDE_probability_one_defective_six_two_l3690_369058


namespace NUMINAMATH_CALUDE_linear_programming_problem_l3690_369052

theorem linear_programming_problem (x y a b : ℝ) :
  3 * x - y - 6 ≤ 0 →
  x - y + 2 ≥ 0 →
  x ≥ 0 →
  y ≥ 0 →
  a > 0 →
  b > 0 →
  (∀ x' y', 3 * x' - y' - 6 ≤ 0 → x' - y' + 2 ≥ 0 → x' ≥ 0 → y' ≥ 0 → a * x' + b * y' ≤ 12) →
  a * x + b * y = 12 →
  (2 / a + 3 / b) ≥ 25 / 6 :=
by sorry

end NUMINAMATH_CALUDE_linear_programming_problem_l3690_369052


namespace NUMINAMATH_CALUDE_geometric_series_relation_l3690_369024

/-- Given real numbers c and d satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 3/4. -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (c / d) / (1 - 1 / d) = 6) :
    (c / (c + 2 * d)) / (1 - 1 / (c + 2 * d)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l3690_369024


namespace NUMINAMATH_CALUDE_flowers_left_l3690_369065

theorem flowers_left (total : ℕ) (min_young : ℕ) (yoo_jeong : ℕ) 
  (h1 : total = 18) 
  (h2 : min_young = 5) 
  (h3 : yoo_jeong = 6) : 
  total - (min_young + yoo_jeong) = 7 := by
sorry

end NUMINAMATH_CALUDE_flowers_left_l3690_369065


namespace NUMINAMATH_CALUDE_red_crayons_count_l3690_369029

/-- Proves that the number of red crayons is 11 given the specified conditions. -/
theorem red_crayons_count (orange_boxes : Nat) (orange_per_box : Nat)
  (blue_boxes : Nat) (blue_per_box : Nat) (total_crayons : Nat) :
  orange_boxes = 6 → orange_per_box = 8 →
  blue_boxes = 7 → blue_per_box = 5 →
  total_crayons = 94 →
  total_crayons - (orange_boxes * orange_per_box + blue_boxes * blue_per_box) = 11 := by
  sorry

end NUMINAMATH_CALUDE_red_crayons_count_l3690_369029


namespace NUMINAMATH_CALUDE_admissible_set_characterization_l3690_369082

def IsAdmissible (A : Set ℤ) : Prop :=
  ∀ x y k : ℤ, x ∈ A → y ∈ A → (x^2 + k*x*y + y^2) ∈ A

theorem admissible_set_characterization (m n : ℤ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (∀ A : Set ℤ, IsAdmissible A → m ∈ A → n ∈ A → A = Set.univ) ↔ Int.gcd m n = 1 :=
sorry

end NUMINAMATH_CALUDE_admissible_set_characterization_l3690_369082


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l3690_369051

/-- Given 6 people in an elevator, if a 7th person weighing 133 lbs enters
    and the new average weight becomes 151 lbs, then the initial average
    weight was 154 lbs. -/
theorem elevator_weight_problem :
  ∀ (initial_average : ℝ),
  (6 * initial_average + 133) / 7 = 151 →
  initial_average = 154 :=
by
  sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l3690_369051


namespace NUMINAMATH_CALUDE_isosceles_triangles_angle_l3690_369077

/-- Isosceles triangle -/
structure IsoscelesTriangle (P Q R : ℝ × ℝ) :=
  (isosceles : dist P Q = dist Q R)

/-- Similar triangles -/
def SimilarTriangles (P Q R P' Q' R' : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    dist P Q = k * dist P' Q' ∧
    dist Q R = k * dist Q' R' ∧
    dist R P = k * dist R' P'

/-- Point lies on line segment -/
def OnSegment (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

/-- Point lies on line extension -/
def OnExtension (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ P = (1 - t) • A + t • B

/-- Perpendicular line segments -/
def Perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (R.1 - P.1) * (S.1 - Q.1) + (R.2 - P.2) * (S.2 - Q.2) = 0

/-- Angle measure -/
def AngleMeasure (P Q R : ℝ × ℝ) : ℝ :=
  sorry

theorem isosceles_triangles_angle (A B C A₁ B₁ C₁ : ℝ × ℝ) :
  IsoscelesTriangle A B C →
  IsoscelesTriangle A₁ B₁ C₁ →
  SimilarTriangles A B C A₁ B₁ C₁ →
  dist A C / dist A₁ C₁ = 5 / Real.sqrt 3 →
  OnSegment A₁ A C →
  OnSegment B₁ B C →
  OnExtension C₁ A B →
  Perpendicular A₁ B₁ B C →
  AngleMeasure A B C = 120 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_angle_l3690_369077


namespace NUMINAMATH_CALUDE_convention_handshakes_l3690_369020

/-- The number of companies at the convention -/
def num_companies : ℕ := 3

/-- The number of representatives from each company -/
def reps_per_company : ℕ := 5

/-- The total number of people at the convention -/
def total_people : ℕ := num_companies * reps_per_company

/-- The number of people each person shakes hands with -/
def handshakes_per_person : ℕ := total_people - 1 - (reps_per_company - 1)

/-- The total number of handshakes at the convention -/
def total_handshakes : ℕ := (total_people * handshakes_per_person) / 2

theorem convention_handshakes :
  total_handshakes = 75 :=
sorry

end NUMINAMATH_CALUDE_convention_handshakes_l3690_369020


namespace NUMINAMATH_CALUDE_union_with_empty_set_l3690_369012

theorem union_with_empty_set (A B : Set ℕ) : 
  A = {1, 2} → B = ∅ → A ∪ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_union_with_empty_set_l3690_369012


namespace NUMINAMATH_CALUDE_farm_field_solution_l3690_369045

/-- Represents the farm field ploughing problem -/
structure FarmField where
  planned_hectares_per_day : ℕ
  actual_hectares_per_day : ℕ
  extra_days : ℕ
  hectares_left : ℕ

/-- Calculates the total area and initial planned days for the farm field -/
def calculate_farm_area_and_days (f : FarmField) : ℕ × ℕ :=
  let initial_days := (f.actual_hectares_per_day * (f.extra_days + 1) + f.hectares_left) / f.planned_hectares_per_day
  let total_area := f.planned_hectares_per_day * initial_days + f.hectares_left
  (total_area, initial_days)

/-- Theorem stating the solution to the farm field problem -/
theorem farm_field_solution (f : FarmField) 
  (h1 : f.planned_hectares_per_day = 160)
  (h2 : f.actual_hectares_per_day = 85)
  (h3 : f.extra_days = 2)
  (h4 : f.hectares_left = 40) :
  calculate_farm_area_and_days f = (520, 3) := by
  sorry

#eval calculate_farm_area_and_days { planned_hectares_per_day := 160, actual_hectares_per_day := 85, extra_days := 2, hectares_left := 40 }

end NUMINAMATH_CALUDE_farm_field_solution_l3690_369045


namespace NUMINAMATH_CALUDE_students_playing_basketball_l3690_369035

/-- The number of students who play basketball in a college, given the total number of students,
    the number of students who play cricket, and the number of students who play both sports. -/
theorem students_playing_basketball
  (total : ℕ)
  (cricket : ℕ)
  (both : ℕ)
  (h1 : total = 880)
  (h2 : cricket = 500)
  (h3 : both = 220) :
  total = cricket + (cricket + both - total) - both :=
by sorry

end NUMINAMATH_CALUDE_students_playing_basketball_l3690_369035


namespace NUMINAMATH_CALUDE_existence_of_three_quadratics_l3690_369097

theorem existence_of_three_quadratics : ∃ (f₁ f₂ f₃ : ℝ → ℝ),
  (∃ x₁, f₁ x₁ = 0) ∧
  (∃ x₂, f₂ x₂ = 0) ∧
  (∃ x₃, f₃ x₃ = 0) ∧
  (∀ x, (f₁ x + f₂ x) ≠ 0) ∧
  (∀ x, (f₁ x + f₃ x) ≠ 0) ∧
  (∀ x, (f₂ x + f₃ x) ≠ 0) ∧
  (∀ x, f₁ x = (x - 1)^2) ∧
  (∀ x, f₂ x = x^2) ∧
  (∀ x, f₃ x = (x - 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_three_quadratics_l3690_369097


namespace NUMINAMATH_CALUDE_intersection_equality_l3690_369005

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x - 5 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x - 1 = 0}

-- State the theorem
theorem intersection_equality (a : ℝ) : A ∩ B a = B a → a = 0 ∨ a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l3690_369005


namespace NUMINAMATH_CALUDE_negation_of_implication_l3690_369066

theorem negation_of_implication (p q : Prop) : 
  ¬(p → q) ↔ p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3690_369066


namespace NUMINAMATH_CALUDE_wallet_problem_l3690_369023

/-- The number of quarters in the wallet -/
def num_quarters : ℕ := 15

/-- The number of dimes in the wallet -/
def num_dimes : ℕ := 25

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of dimes that equal the value of the quarters -/
def n : ℕ := 38

theorem wallet_problem :
  (num_quarters * quarter_value : ℕ) = n * dime_value :=
by sorry

end NUMINAMATH_CALUDE_wallet_problem_l3690_369023


namespace NUMINAMATH_CALUDE_flower_bed_properties_l3690_369046

/-- Represents a rectangular flower bed with specific properties -/
structure FlowerBed where
  length : ℝ
  width : ℝ
  area : ℝ
  theta : ℝ

/-- Theorem about the properties of a specific flower bed -/
theorem flower_bed_properties :
  ∃ (fb : FlowerBed),
    fb.area = 50 ∧
    fb.width = (2/3) * fb.length ∧
    Real.tan fb.theta = (fb.length - fb.width) / (fb.length + fb.width) ∧
    fb.length = 75 ∧
    fb.width = 50 ∧
    Real.tan fb.theta = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_flower_bed_properties_l3690_369046


namespace NUMINAMATH_CALUDE_total_amount_calculation_l3690_369070

theorem total_amount_calculation (r p q : ℝ) 
  (h1 : r = 2000.0000000000002) 
  (h2 : r = (2/3) * (p + q + r)) : 
  p + q + r = 3000.0000000000003 := by
sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l3690_369070


namespace NUMINAMATH_CALUDE_parallel_vector_sum_diff_l3690_369033

/-- Given two 2D vectors a and b, if a + b is parallel to a - b, then the first component of a is -4/3. -/
theorem parallel_vector_sum_diff (a b : ℝ × ℝ) :
  a.1 = m ∧ a.2 = 2 ∧ b = (2, -3) →
  (∃ k : ℝ, k ≠ 0 ∧ (a + b) = k • (a - b)) →
  m = -4/3 := by sorry

end NUMINAMATH_CALUDE_parallel_vector_sum_diff_l3690_369033


namespace NUMINAMATH_CALUDE_solution_properties_l3690_369011

theorem solution_properties (a b : ℝ) (h : a^2 - 5*b^2 = 1) :
  (0 < a + b * Real.sqrt 5 → a ≥ 0) ∧
  (1 < a + b * Real.sqrt 5 → a ≥ 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_solution_properties_l3690_369011


namespace NUMINAMATH_CALUDE_final_jasmine_concentration_l3690_369060

/-- Calculates the final jasmine concentration after adding pure jasmine and water to a solution -/
theorem final_jasmine_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 80)
  (h2 : initial_concentration = 0.1)
  (h3 : added_jasmine = 5)
  (h4 : added_water = 15) :
  let initial_jasmine := initial_volume * initial_concentration
  let final_jasmine := initial_jasmine + added_jasmine
  let final_volume := initial_volume + added_jasmine + added_water
  final_jasmine / final_volume = 0.13 := by
sorry


end NUMINAMATH_CALUDE_final_jasmine_concentration_l3690_369060


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3690_369017

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*m*x + m^2 - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ * (x₂ + x₁) + x₂^2 = 5*m →
  m = (5 - Real.sqrt 13) / 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3690_369017


namespace NUMINAMATH_CALUDE_remainder_problem_l3690_369063

theorem remainder_problem (x : ℕ+) : (6 * x.val) % 9 = 3 → x.val % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3690_369063


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3690_369089

theorem logarithm_expression_equality : 
  9^(Real.log 2 / Real.log 3) - 4 * (Real.log 3 / Real.log 4) * (Real.log 8 / Real.log 27) + 
  (1/3) * (Real.log 8 / Real.log 6) - 2 * (Real.log (Real.sqrt 3) / Real.log (1/6)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3690_369089


namespace NUMINAMATH_CALUDE_inequality_proof_l3690_369009

theorem inequality_proof (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  1 ≤ (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ∧
  (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3690_369009


namespace NUMINAMATH_CALUDE_bounds_per_meter_proof_l3690_369002

/-- Represents the number of bounds in one meter -/
def bounds_per_meter : ℚ :=
  21 / 100

/-- The number of leaps that equal 3 bounds -/
def leaps_to_bounds : ℕ := 4

/-- The number of bounds that equal 4 leaps -/
def bounds_to_leaps : ℕ := 3

/-- The number of strides that equal 2 leaps -/
def strides_to_leaps : ℕ := 5

/-- The number of leaps that equal 5 strides -/
def leaps_to_strides : ℕ := 2

/-- The number of strides that equal 10 meters -/
def strides_to_meters : ℕ := 7

/-- The number of meters that equal 7 strides -/
def meters_to_strides : ℕ := 10

theorem bounds_per_meter_proof :
  bounds_per_meter = 21 / 100 :=
by sorry

end NUMINAMATH_CALUDE_bounds_per_meter_proof_l3690_369002


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3690_369072

theorem quadratic_roots_relation (A B C : ℝ) (r s p q : ℝ) : 
  (A * r^2 + B * r + C = 0) →
  (A * s^2 + B * s + C = 0) →
  (r^2)^2 + p * r^2 + q = 0 →
  (s^2)^2 + p * s^2 + q = 0 →
  p = (2 * A * C - B^2) / A^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3690_369072


namespace NUMINAMATH_CALUDE_exam_score_problem_l3690_369079

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) :
  total_questions = 60 →
  correct_score = 4 →
  total_score = 120 →
  correct_answers = 36 →
  (total_questions - correct_answers) * (correct_score - (total_score - correct_answers * correct_score) / (total_questions - correct_answers)) = total_questions - correct_answers :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3690_369079


namespace NUMINAMATH_CALUDE_age_difference_proof_l3690_369080

def elder_age : ℕ := 30

theorem age_difference_proof (younger_age : ℕ) 
  (h : elder_age - 6 = 3 * (younger_age - 6)) : 
  elder_age - younger_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3690_369080


namespace NUMINAMATH_CALUDE_remaining_quarters_l3690_369055

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.50
def jeans_cost : ℚ := 11.50
def quarter_value : ℚ := 0.25

theorem remaining_quarters : 
  (initial_amount - (pizza_cost + soda_cost + jeans_cost)) / quarter_value = 97 := by
  sorry

end NUMINAMATH_CALUDE_remaining_quarters_l3690_369055


namespace NUMINAMATH_CALUDE_average_height_is_12_l3690_369071

def plant_heights (h1 h2 h3 h4 : ℝ) : Prop :=
  h1 = 27 ∧ h3 = 9 ∧
  ((h2 = h1 / 3 ∨ h2 = h1 * 3) ∧
   (h3 = h2 / 3 ∨ h3 = h2 * 3) ∧
   (h4 = h3 / 3 ∨ h4 = h3 * 3))

theorem average_height_is_12 (h1 h2 h3 h4 : ℝ) :
  plant_heights h1 h2 h3 h4 → (h1 + h2 + h3 + h4) / 4 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_average_height_is_12_l3690_369071


namespace NUMINAMATH_CALUDE_derivative_f_minus_f4x_l3690_369006

/-- Given a function f where the derivative of f(x) - f(2x) at x = 1 is 5 and at x = 2 is 7,
    the derivative of f(x) - f(4x) at x = 1 is 19. -/
theorem derivative_f_minus_f4x (f : ℝ → ℝ) 
  (h1 : deriv (fun x ↦ f x - f (2 * x)) 1 = 5)
  (h2 : deriv (fun x ↦ f x - f (2 * x)) 2 = 7) :
  deriv (fun x ↦ f x - f (4 * x)) 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_minus_f4x_l3690_369006


namespace NUMINAMATH_CALUDE_line_intersection_l3690_369026

theorem line_intersection (a b c : ℝ) : 
  (3 = a * 1 + b) ∧ 
  (3 = b * 1 + c) ∧ 
  (3 = c * 1 + a) → 
  a = (3/2 : ℝ) ∧ b = (3/2 : ℝ) ∧ c = (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l3690_369026


namespace NUMINAMATH_CALUDE_grade_change_impossibility_l3690_369044

theorem grade_change_impossibility : ¬ ∃ (n₁ n₂ n₃ n₄ : ℤ),
  2 * n₁ + n₂ - 2 * n₃ - n₄ = 27 ∧
  -n₁ + 2 * n₂ + n₃ - 2 * n₄ = -27 :=
sorry

end NUMINAMATH_CALUDE_grade_change_impossibility_l3690_369044


namespace NUMINAMATH_CALUDE_tangent_circles_radii_l3690_369062

/-- Two externally tangent circles with specific properties -/
structure TangentCircles where
  r₁ : ℝ  -- radius of the smaller circle
  r₂ : ℝ  -- radius of the larger circle
  h₁ : r₂ = r₁ + 5  -- difference between radii is 5
  h₂ : ∃ (d : ℝ), d = 2.4 * r₁ ∧ d^2 + r₁^2 = (r₂ - r₁)^2  -- distance property

/-- The radii of the two circles are 4 and 9 -/
theorem tangent_circles_radii (c : TangentCircles) : c.r₁ = 4 ∧ c.r₂ = 9 :=
  sorry

end NUMINAMATH_CALUDE_tangent_circles_radii_l3690_369062


namespace NUMINAMATH_CALUDE_rectangular_plot_minus_circular_garden_l3690_369049

/-- The area of a rectangular plot minus a circular garden --/
theorem rectangular_plot_minus_circular_garden :
  let rectangle_length : ℝ := 8
  let rectangle_width : ℝ := 12
  let circle_radius : ℝ := 3
  let rectangle_area := rectangle_length * rectangle_width
  let circle_area := π * circle_radius ^ 2
  rectangle_area - circle_area = 96 - 9 * π := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_minus_circular_garden_l3690_369049


namespace NUMINAMATH_CALUDE_derivative_cos_at_pi_12_l3690_369092

/-- Given a function f(x) = cos(2x + π/3), prove that its derivative at x = π/12 is -2. -/
theorem derivative_cos_at_pi_12 (f : ℝ → ℝ) (h : ∀ x, f x = Real.cos (2 * x + π / 3)) :
  deriv f (π / 12) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_cos_at_pi_12_l3690_369092


namespace NUMINAMATH_CALUDE_intersection_implies_outside_circle_l3690_369057

theorem intersection_implies_outside_circle (a b : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  a^2 + b^2 > 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_outside_circle_l3690_369057


namespace NUMINAMATH_CALUDE_inequality_solution_l3690_369021

theorem inequality_solution : ∃! (x y z : ℤ),
  (1 / Real.sqrt (x - 2*y + z + 1 : ℝ) +
   2 / Real.sqrt (2*x - y + 3*z - 1 : ℝ) +
   3 / Real.sqrt (3*y - 3*x - 4*z + 3 : ℝ) >
   x^2 - 4*x + 3) ∧
  (x = 3 ∧ y = 1 ∧ z = -1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3690_369021


namespace NUMINAMATH_CALUDE_negative_three_a_cubed_squared_l3690_369074

theorem negative_three_a_cubed_squared (a : ℝ) : (-3 * a^3)^2 = 9 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_a_cubed_squared_l3690_369074


namespace NUMINAMATH_CALUDE_john_needs_additional_money_l3690_369088

/-- The amount of money John needs -/
def money_needed : ℚ := 2.50

/-- The amount of money John has -/
def money_has : ℚ := 0.75

/-- The additional money John needs -/
def additional_money : ℚ := money_needed - money_has

theorem john_needs_additional_money : additional_money = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_additional_money_l3690_369088


namespace NUMINAMATH_CALUDE_parallelepiped_surface_area_l3690_369025

theorem parallelepiped_surface_area (a b c : ℝ) (h_sphere : a^2 + b^2 + c^2 = 12) 
  (h_volume : a * b * c = 8) : 2 * (a * b + b * c + c * a) = 24 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_surface_area_l3690_369025


namespace NUMINAMATH_CALUDE_no_complex_numbers_satisfying_condition_l3690_369098

theorem no_complex_numbers_satisfying_condition : ¬∃ (a b c : ℂ) (h : ℕ), 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ 
  (∀ (k l m : ℤ), (abs k + abs l + abs m ≥ 1996) → 
    Complex.abs (k • a + l • b + m • c) > 1 / h) :=
by sorry

end NUMINAMATH_CALUDE_no_complex_numbers_satisfying_condition_l3690_369098


namespace NUMINAMATH_CALUDE_wall_width_l3690_369056

theorem wall_width (area : ℝ) (height : ℝ) (width : ℝ) :
  area = 8 ∧ height = 4 ∧ area = width * height → width = 2 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_l3690_369056


namespace NUMINAMATH_CALUDE_complex_fraction_ratio_l3690_369087

theorem complex_fraction_ratio : 
  let z : ℂ := (2 + I) / I
  ∃ a b : ℝ, z = a + b * I ∧ b / a = -2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_ratio_l3690_369087


namespace NUMINAMATH_CALUDE_cork_price_calculation_l3690_369085

/-- The price of a bottle of wine with a cork -/
def bottle_with_cork : ℚ := 2.10

/-- The additional cost of a bottle without a cork compared to the cork price -/
def additional_cost : ℚ := 2.00

/-- The discount rate for the cork when buying in large quantities -/
def cork_discount : ℚ := 0.12

/-- The price of the cork before discount -/
def cork_price : ℚ := bottle_with_cork - (bottle_with_cork - additional_cost) / 2

/-- The discounted price of the cork -/
def discounted_cork_price : ℚ := cork_price * (1 - cork_discount)

theorem cork_price_calculation :
  discounted_cork_price = 0.044 := by sorry

end NUMINAMATH_CALUDE_cork_price_calculation_l3690_369085


namespace NUMINAMATH_CALUDE_pineapple_purchase_l3690_369019

/-- The number of pineapples bought by Steve and Georgia -/
def num_pineapples : ℕ := 12

/-- The cost of each pineapple in dollars -/
def cost_per_pineapple : ℚ := 5/4

/-- The shipping cost in dollars -/
def shipping_cost : ℚ := 21

/-- The total cost per pineapple (including shipping) in dollars -/
def total_cost_per_pineapple : ℚ := 3

theorem pineapple_purchase :
  (↑num_pineapples * cost_per_pineapple + shipping_cost) / ↑num_pineapples = total_cost_per_pineapple :=
sorry

end NUMINAMATH_CALUDE_pineapple_purchase_l3690_369019


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l3690_369000

theorem quadratic_equation_range (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x - m - 1 = 0) ↔ m ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l3690_369000


namespace NUMINAMATH_CALUDE_arcsin_neg_half_equals_neg_pi_sixth_l3690_369048

theorem arcsin_neg_half_equals_neg_pi_sixth : 
  Real.arcsin (-0.5) = -π/6 := by sorry

end NUMINAMATH_CALUDE_arcsin_neg_half_equals_neg_pi_sixth_l3690_369048


namespace NUMINAMATH_CALUDE_equation_is_pair_of_straight_lines_l3690_369075

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := 3 * x^2 - 12 * y^2 = 0

/-- Definition of a pair of straight lines -/
def is_pair_of_straight_lines (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∧ c ≠ 0 ∧
    ∀ x y, f x y ↔ (a * x + b * y = 0) ∨ (c * x + d * y = 0)

/-- Theorem stating that the equation represents a pair of straight lines -/
theorem equation_is_pair_of_straight_lines :
  is_pair_of_straight_lines equation :=
sorry

end NUMINAMATH_CALUDE_equation_is_pair_of_straight_lines_l3690_369075


namespace NUMINAMATH_CALUDE_polynomial_expansion_properties_l3690_369030

theorem polynomial_expansion_properties (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x : ℝ, (1 + 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  a₂ = 24 ∧ a + a₁ + a₂ + a₃ + a₄ = 81 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_properties_l3690_369030


namespace NUMINAMATH_CALUDE_number_equality_l3690_369040

theorem number_equality (x : ℝ) : (30 / 100 : ℝ) * x = (25 / 100 : ℝ) * 45 → x = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l3690_369040


namespace NUMINAMATH_CALUDE_library_books_before_grant_l3690_369016

theorem library_books_before_grant (books_purchased : ℕ) (total_books_now : ℕ) 
  (h1 : books_purchased = 2647)
  (h2 : total_books_now = 8582) :
  total_books_now - books_purchased = 5935 :=
by sorry

end NUMINAMATH_CALUDE_library_books_before_grant_l3690_369016


namespace NUMINAMATH_CALUDE_parabola_c_value_l3690_369054

theorem parabola_c_value (a b c : ℚ) :
  (∀ y : ℚ, -3 = a * 1^2 + b * 1 + c) →
  (∀ y : ℚ, -6 = a * 3^2 + b * 3 + c) →
  c = -15/4 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3690_369054


namespace NUMINAMATH_CALUDE_computation_proof_l3690_369090

theorem computation_proof : 55 * 1212 - 15 * 1212 = 48480 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l3690_369090


namespace NUMINAMATH_CALUDE_wooden_strip_triangle_l3690_369001

theorem wooden_strip_triangle (x : ℝ) : 
  (0 < x ∧ x < 5 ∧ 
   x + x > 10 - 2*x ∧
   10 - 2*x > 0) ↔ 
  (2.5 < x ∧ x < 5) :=
sorry

end NUMINAMATH_CALUDE_wooden_strip_triangle_l3690_369001


namespace NUMINAMATH_CALUDE_subset_sum_exists_l3690_369086

theorem subset_sum_exists (A : ℕ) (h1 : ∀ i ∈ Finset.range 9, A % (i + 1) = 0)
  (h2 : ∃ (S : Finset ℕ), (∀ x ∈ S, x ∈ Finset.range 9) ∧ S.sum id = 2 * A) :
  ∃ (T : Finset ℕ), T ⊆ S ∧ T.sum id = A :=
sorry

end NUMINAMATH_CALUDE_subset_sum_exists_l3690_369086


namespace NUMINAMATH_CALUDE_vehicle_passing_condition_min_speed_for_passing_l3690_369067

-- Define the speeds and distances
def VB : ℝ := 40  -- mph
def VC : ℝ := 65  -- mph
def dist_AB : ℝ := 100  -- ft
def dist_BC : ℝ := 250  -- ft

-- Define the theorem
theorem vehicle_passing_condition (VA : ℝ) :
  VA > 2 →
  (dist_AB / (VA + VB)) < (dist_BC / (VB + VC)) :=
by
  sorry

-- Define the main theorem that answers the original question
theorem min_speed_for_passing :
  ∃ (VA : ℝ), VA > 2 ∧
  ∀ (VA' : ℝ), VA' > VA →
  (dist_AB / (VA' + VB)) < (dist_BC / (VB + VC)) :=
by
  sorry

end NUMINAMATH_CALUDE_vehicle_passing_condition_min_speed_for_passing_l3690_369067


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l3690_369038

def coin_flips (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem unfair_coin_probability :
  let n : ℕ := 8
  let p_tails : ℚ := 3/4
  let k : ℕ := 3
  coin_flips n p_tails k = 189/128 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l3690_369038


namespace NUMINAMATH_CALUDE_steiner_ellipses_equations_l3690_369076

/-- Barycentric coordinates in a triangle -/
structure BarycentricCoord where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Circumscribed Steiner Ellipse equation -/
def circumscribedSteinerEllipse (p : BarycentricCoord) : Prop :=
  p.β * p.γ + p.α * p.γ + p.α * p.β = 0

/-- Inscribed Steiner Ellipse equation -/
def inscribedSteinerEllipse (p : BarycentricCoord) : Prop :=
  2 * p.β * p.γ + 2 * p.α * p.γ + 2 * p.α * p.β = p.α^2 + p.β^2 + p.γ^2

/-- Theorem stating the equations of Steiner ellipses in barycentric coordinates -/
theorem steiner_ellipses_equations (p : BarycentricCoord) :
  (circumscribedSteinerEllipse p ↔ p.β * p.γ + p.α * p.γ + p.α * p.β = 0) ∧
  (inscribedSteinerEllipse p ↔ 2 * p.β * p.γ + 2 * p.α * p.γ + 2 * p.α * p.β = p.α^2 + p.β^2 + p.γ^2) :=
by sorry

end NUMINAMATH_CALUDE_steiner_ellipses_equations_l3690_369076


namespace NUMINAMATH_CALUDE_smallest_a_value_l3690_369081

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℝ, Real.sin (a * x + b) = Real.sin (15 * x)) : 
  a ≥ 15 ∧ ∃ (a₀ b₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ (∀ x : ℝ, Real.sin (a₀ * x + b₀) = Real.sin (15 * x)) ∧ a₀ = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3690_369081


namespace NUMINAMATH_CALUDE_area_of_region_l3690_369099

/-- The region defined by the inequality |4x - 20| + |3y - 6| ≤ 4 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |4 * p.1 - 20| + |3 * p.2 - 6| ≤ 4}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem stating that the area of the region is 8/3 -/
theorem area_of_region : area Region = 8/3 := by sorry

end NUMINAMATH_CALUDE_area_of_region_l3690_369099


namespace NUMINAMATH_CALUDE_test_results_l3690_369015

/-- Given a class with the following properties:
  * 30 students enrolled
  * 25 students answered question 1 correctly
  * 22 students answered question 2 correctly
  * 18 students answered question 3 correctly
  * 5 students did not take the test
Prove that 18 students answered all three questions correctly. -/
theorem test_results (total_students : ℕ) (q1_correct : ℕ) (q2_correct : ℕ) (q3_correct : ℕ) (absent : ℕ)
  (h1 : total_students = 30)
  (h2 : q1_correct = 25)
  (h3 : q2_correct = 22)
  (h4 : q3_correct = 18)
  (h5 : absent = 5) :
  q3_correct = 18 ∧ q3_correct = (total_students - absent - (total_students - absent - q1_correct) - (total_students - absent - q2_correct)) :=
by sorry

end NUMINAMATH_CALUDE_test_results_l3690_369015


namespace NUMINAMATH_CALUDE_min_value_expression_l3690_369064

theorem min_value_expression (x y : ℝ) : (3*x*y - 1)^2 + (x - y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3690_369064


namespace NUMINAMATH_CALUDE_equal_volume_cans_l3690_369078

/-- Represents a cylindrical can with radius and height -/
structure Can where
  radius : ℝ
  height : ℝ

/-- Theorem stating the relation between two cans with equal volume -/
theorem equal_volume_cans (can1 can2 : Can) 
  (h_volume : can1.radius ^ 2 * can1.height = can2.radius ^ 2 * can2.height)
  (h_height : can2.height = 4 * can1.height)
  (h_narrow_radius : can1.radius = 10) :
  can2.radius = 20 := by
  sorry

end NUMINAMATH_CALUDE_equal_volume_cans_l3690_369078


namespace NUMINAMATH_CALUDE_a_lt_one_necessary_not_sufficient_for_a_squared_lt_one_l3690_369053

theorem a_lt_one_necessary_not_sufficient_for_a_squared_lt_one :
  (∀ a : ℝ, a^2 < 1 → a < 1) ∧
  (∃ a : ℝ, a < 1 ∧ a^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_a_lt_one_necessary_not_sufficient_for_a_squared_lt_one_l3690_369053


namespace NUMINAMATH_CALUDE_total_cases_california_l3690_369041

/-- Calculates the total number of positive Coronavirus cases after three days,
    given the initial number of cases and daily changes. -/
def totalCasesAfterThreeDays (initialCases : ℕ) (newCasesDay2 : ℕ) (recoveriesDay2 : ℕ)
                              (newCasesDay3 : ℕ) (recoveriesDay3 : ℕ) : ℕ :=
  initialCases + (newCasesDay2 - recoveriesDay2) + (newCasesDay3 - recoveriesDay3)

/-- Theorem stating that given the specific numbers from the problem,
    the total number of positive cases after the third day is 3750. -/
theorem total_cases_california : totalCasesAfterThreeDays 2000 500 50 1500 200 = 3750 := by
  sorry

end NUMINAMATH_CALUDE_total_cases_california_l3690_369041


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3690_369042

theorem quadratic_equation_solutions : 
  ∀ x : ℝ, x^2 = 6*x ↔ x = 0 ∨ x = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3690_369042


namespace NUMINAMATH_CALUDE_queen_center_probability_queen_center_probability_2004_l3690_369047

/-- Probability of queen being in center after n moves -/
def prob_queen_center (n : ℕ) : ℚ :=
  1/3 + 2/3 * (-1/2)^n

/-- Initial configuration with queen in center -/
def initial_config : List Char := ['R', 'Q', 'R']

/-- Theorem stating the probability of queen being in center after n moves -/
theorem queen_center_probability (n : ℕ) : 
  prob_queen_center n = 1/3 + 2/3 * (-1/2)^n :=
sorry

/-- Corollary for the specific case of 2004 moves -/
theorem queen_center_probability_2004 : 
  prob_queen_center 2004 = 1/3 + 1/(3 * 2^2003) :=
sorry

end NUMINAMATH_CALUDE_queen_center_probability_queen_center_probability_2004_l3690_369047


namespace NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l3690_369022

/-- Represents the number of tickets Tom has -/
structure Tickets :=
  (yellow : ℕ)
  (red : ℕ)
  (blue : ℕ)

/-- Calculates the total number of blue tickets equivalent to a given number of tickets -/
def blueEquivalent (t : Tickets) : ℕ :=
  t.yellow * 100 + t.red * 10 + t.blue

/-- The number of blue tickets needed to win a Bible -/
def bibleRequirement : ℕ := 1000

/-- Tom's current tickets -/
def tomsTickets : Tickets := ⟨8, 3, 7⟩

/-- Theorem stating how many more blue tickets Tom needs -/
theorem tom_needs_163_blue_tickets :
  bibleRequirement - blueEquivalent tomsTickets = 163 := by
  sorry

end NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l3690_369022


namespace NUMINAMATH_CALUDE_system_solution_l3690_369036

theorem system_solution (u v w : ℚ) 
  (eq1 : 3 * u - 4 * v + w = 26)
  (eq2 : 6 * u + 5 * v - 2 * w = -17) :
  u + v + w = 101 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3690_369036


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3690_369003

def P : Set ℝ := {x | x^2 - 16 < 0}
def Q : Set ℝ := {x | ∃ n : ℤ, x = 2 * n}

theorem intersection_P_Q : P ∩ Q = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3690_369003


namespace NUMINAMATH_CALUDE_quadratic_point_relationship_l3690_369031

/-- The quadratic function f(x) = x² + 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

theorem quadratic_point_relationship (c : ℝ) :
  let y₁ := f c (-4)
  let y₂ := f c (-3)
  let y₃ := f c 1
  y₂ < y₁ ∧ y₁ < y₃ := by sorry

end NUMINAMATH_CALUDE_quadratic_point_relationship_l3690_369031


namespace NUMINAMATH_CALUDE_chocolate_cost_l3690_369008

theorem chocolate_cost (total_cost candy_price_difference : ℚ)
  (h1 : total_cost = 7)
  (h2 : candy_price_difference = 4) : 
  ∃ (chocolate_cost : ℚ), 
    chocolate_cost + (chocolate_cost + candy_price_difference) = total_cost ∧ 
    chocolate_cost = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_l3690_369008


namespace NUMINAMATH_CALUDE_problem_l3690_369050

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 2 * Real.cos x + 1

theorem problem (a b c : ℝ) (h : ∀ x : ℝ, a * f x + b * f (x - c) = 1) :
  b * Real.cos c / a = -1 := by sorry

end NUMINAMATH_CALUDE_problem_l3690_369050


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_difference_l3690_369061

-- Define the polynomial
def f (x : ℝ) : ℝ := 20 * x^3 - 40 * x^2 + 18 * x - 1

-- Define the roots
variable (a b c : ℝ)

-- State the theorem
theorem root_sum_reciprocal_difference (ha : f a = 0) (hb : f b = 0) (hc : f c = 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (ha_bounds : 0 < a ∧ a < 1) (hb_bounds : 0 < b ∧ b < 1) (hc_bounds : 0 < c ∧ c < 1) :
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_difference_l3690_369061


namespace NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l3690_369010

theorem unique_solution_for_rational_equation :
  ∃! k : ℚ, ∀ x : ℚ, (x + 3) / (k * x + x - 3) = x ∧ k * x + x - 3 ≠ 0 → k = -7/3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l3690_369010


namespace NUMINAMATH_CALUDE_base_three_digit_difference_l3690_369059

/-- The number of digits in the base-b representation of a positive integer n -/
def numDigits (n : ℕ+) (b : ℕ) : ℕ :=
  Nat.log b n + 1

/-- Theorem: The number of digits in the base-3 representation of 1500
    is exactly 1 more than the number of digits in the base-3 representation of 300 -/
theorem base_three_digit_difference :
  numDigits 1500 3 = numDigits 300 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_base_three_digit_difference_l3690_369059


namespace NUMINAMATH_CALUDE_ellipse_condition_l3690_369083

-- Define the equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / (3 - m) = 1

-- Define the condition for an ellipse with foci on the y-axis
def is_ellipse_with_y_foci (m : ℝ) : Prop :=
  m > 1 ∧ m < 3 ∧ (3 - m > m - 1)

-- State the theorem
theorem ellipse_condition (m : ℝ) :
  (1 < m ∧ m < 2) ↔ is_ellipse_with_y_foci m :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3690_369083


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l3690_369032

/-- The type of positive natural numbers -/
def PositiveNat := {n : ℕ // n > 0}

/-- n-th iterate of a function -/
def iterate (f : PositiveNat → PositiveNat) : ℕ → (PositiveNat → PositiveNat)
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

/-- The main theorem stating that no function satisfies the given condition -/
theorem no_function_satisfies_condition :
  ¬ ∃ (f : PositiveNat → PositiveNat),
    ∀ (n : ℕ), (iterate f n) ⟨n + 1, Nat.succ_pos n⟩ = ⟨n + 2, Nat.succ_pos (n + 1)⟩ :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l3690_369032


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l3690_369084

/-- The length of the path traveled by point F when rolling a quarter-circle -/
theorem quarter_circle_roll_path_length 
  (radius : ℝ) 
  (h_radius : radius = 3 / Real.pi) : 
  let path_length := 3 * (Real.pi * radius / 2)
  path_length = 4.5 := by sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l3690_369084


namespace NUMINAMATH_CALUDE_wendy_full_face_time_l3690_369014

/-- Calculates the total time for Wendy's "full face" routine -/
def full_face_time (num_products : ℕ) (wait_time : ℕ) (makeup_time : ℕ) : ℕ :=
  (num_products - 1) * wait_time + makeup_time

/-- Proves that Wendy's "full face" routine takes 50 minutes -/
theorem wendy_full_face_time :
  full_face_time 5 5 30 = 50 := by
  sorry

end NUMINAMATH_CALUDE_wendy_full_face_time_l3690_369014
