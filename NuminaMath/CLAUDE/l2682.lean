import Mathlib

namespace NUMINAMATH_CALUDE_tetrahedron_distance_sum_l2682_268264

/-- Tetrahedron with face areas, distances, and volume -/
structure Tetrahedron where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  S₄ : ℝ
  H₁ : ℝ
  H₂ : ℝ
  H₃ : ℝ
  H₄ : ℝ
  V : ℝ

/-- The theorem about the sum of weighted distances in a tetrahedron -/
theorem tetrahedron_distance_sum (t : Tetrahedron) (k : ℝ) 
    (h₁ : t.S₁ / 1 = k)
    (h₂ : t.S₂ / 2 = k)
    (h₃ : t.S₃ / 3 = k)
    (h₄ : t.S₄ / 4 = k) :
  1 * t.H₁ + 2 * t.H₂ + 3 * t.H₃ + 4 * t.H₄ = 3 * t.V / k := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_distance_sum_l2682_268264


namespace NUMINAMATH_CALUDE_fifteen_times_thirtysix_plus_fifteen_times_three_cubed_l2682_268260

theorem fifteen_times_thirtysix_plus_fifteen_times_three_cubed : 15 * 36 + 15 * 3^3 = 945 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_times_thirtysix_plus_fifteen_times_three_cubed_l2682_268260


namespace NUMINAMATH_CALUDE_shortest_minor_arc_line_l2682_268284

/-- The point M -/
def M : ℝ × ℝ := (1, -2)

/-- The circle C -/
def C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

/-- A line passing through a point -/
def LineThrough (m : ℝ × ℝ) (a b c : ℝ) : Prop :=
  a * m.1 + b * m.2 + c = 0

/-- The theorem stating the equation of the line that divides the circle into two arcs with the shortest minor arc -/
theorem shortest_minor_arc_line :
  ∃ (a b c : ℝ), LineThrough M a b c ∧
  (∀ (x y : ℝ), C x y → (a * x + b * y + c = 0 → 
    ∀ (a' b' c' : ℝ), LineThrough M a' b' c' → 
      (∃ (x' y' : ℝ), C x' y' ∧ a' * x' + b' * y' + c' = 0) → 
        (∃ (x'' y'' : ℝ), C x'' y'' ∧ a * x'' + b * y'' + c = 0 ∧ 
          ∀ (x''' y''' : ℝ), C x''' y''' ∧ a' * x''' + b' * y''' + c' = 0 → 
            (x'' - M.1)^2 + (y'' - M.2)^2 ≤ (x''' - M.1)^2 + (y''' - M.2)^2))) ∧
  a = 1 ∧ b = 2 ∧ c = 3 :=
sorry

end NUMINAMATH_CALUDE_shortest_minor_arc_line_l2682_268284


namespace NUMINAMATH_CALUDE_opera_house_seats_l2682_268247

theorem opera_house_seats (rows : ℕ) (revenue : ℕ) (ticket_price : ℕ) (occupancy_rate : ℚ) :
  rows = 150 →
  revenue = 12000 →
  ticket_price = 10 →
  occupancy_rate = 4/5 →
  ∃ (seats_per_row : ℕ), seats_per_row = 10 ∧ 
    (revenue / ticket_price : ℚ) = (occupancy_rate * (rows * seats_per_row : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_opera_house_seats_l2682_268247


namespace NUMINAMATH_CALUDE_composite_sum_of_prime_powers_l2682_268263

theorem composite_sum_of_prime_powers (p q t : Nat) : 
  Prime p → Prime q → Prime t → p ≠ q → p ≠ t → q ≠ t →
  ∃ n : Nat, n > 1 ∧ n ∣ (2016^p + 2017^q + 2018^t) :=
by sorry

end NUMINAMATH_CALUDE_composite_sum_of_prime_powers_l2682_268263


namespace NUMINAMATH_CALUDE_demographic_prediction_basis_l2682_268226

/-- Represents the possible bases for demographic predictions -/
inductive DemographicBasis
  | PopulationQuantityAndDensity
  | AgeComposition
  | GenderRatio
  | BirthAndDeathRates

/-- Represents different countries -/
inductive Country
  | Mexico
  | UnitedStates
  | Sweden
  | Germany

/-- Represents the prediction for population growth -/
inductive PopulationPrediction
  | Increase
  | Stable
  | Decrease

/-- Function that assigns a population prediction to each country -/
def countryPrediction : Country → PopulationPrediction
  | Country.Mexico => PopulationPrediction.Increase
  | Country.UnitedStates => PopulationPrediction.Increase
  | Country.Sweden => PopulationPrediction.Stable
  | Country.Germany => PopulationPrediction.Decrease

/-- The main basis used by demographers for their predictions -/
def mainBasis : DemographicBasis := DemographicBasis.AgeComposition

theorem demographic_prediction_basis :
  (∀ c : Country, ∃ p : PopulationPrediction, countryPrediction c = p) →
  mainBasis = DemographicBasis.AgeComposition :=
by sorry

end NUMINAMATH_CALUDE_demographic_prediction_basis_l2682_268226


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_under_120_l2682_268246

theorem largest_multiple_of_9_under_120 : 
  ∃ n : ℕ, n * 9 = 117 ∧ 117 < 120 ∧ ∀ m : ℕ, m * 9 < 120 → m * 9 ≤ 117 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_under_120_l2682_268246


namespace NUMINAMATH_CALUDE_chord_probability_chord_probability_proof_l2682_268245

/-- The probability that a randomly chosen point on a circle's circumference,
    when connected to a fixed point on the circumference, forms a chord with
    length between R and √3R, where R is the radius of the circle. -/
theorem chord_probability (R : ℝ) (R_pos : R > 0) : ℝ :=
  1 / 3

/-- Proof of the chord probability theorem -/
theorem chord_probability_proof (R : ℝ) (R_pos : R > 0) :
  chord_probability R R_pos = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_probability_chord_probability_proof_l2682_268245


namespace NUMINAMATH_CALUDE_total_balloons_count_l2682_268256

/-- The number of violet balloons Dan, Tim, and Molly have in total -/
def total_balloons (dan_balloons : ℕ) : ℕ :=
  dan_balloons + (7 * dan_balloons) + (5 * dan_balloons)

/-- Theorem stating that the total number of violet balloons is 377 -/
theorem total_balloons_count : total_balloons 29 = 377 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_count_l2682_268256


namespace NUMINAMATH_CALUDE_xy_equals_one_l2682_268233

theorem xy_equals_one (x y : ℝ) (h : x + y = 1/x + 1/y ∧ x + y ≠ 0) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_one_l2682_268233


namespace NUMINAMATH_CALUDE_georges_socks_l2682_268238

/-- The number of socks George's dad gave him -/
def socks_from_dad (initial_socks bought_socks total_socks : ℝ) : ℝ :=
  total_socks - (initial_socks + bought_socks)

/-- Proof that George's dad gave him 4 socks -/
theorem georges_socks : socks_from_dad 28 36 68 = 4 := by
  sorry

end NUMINAMATH_CALUDE_georges_socks_l2682_268238


namespace NUMINAMATH_CALUDE_unique_solution_iff_p_zero_l2682_268229

/-- The system of equations has exactly one solution if and only if p = 0 -/
theorem unique_solution_iff_p_zero (p : ℝ) :
  (∃! x y : ℝ, x^2 - y^2 = 0 ∧ x*y + p*x - p*y = p^2) ↔ p = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_p_zero_l2682_268229


namespace NUMINAMATH_CALUDE_max_pairs_correct_max_pairs_achievable_l2682_268231

/-- The maximum number of pairs that can be chosen from the set {1, 2, ..., 3009}
    such that no two pairs have a common element and all sums of pairs are distinct
    and less than or equal to 3009 -/
def max_pairs : ℕ := 1003

theorem max_pairs_correct : 
  ∀ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ Finset.range 3009 ∧ p.2 ∈ Finset.range 3009) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 3009) →
    pairs.card ≤ max_pairs :=
by sorry

theorem max_pairs_achievable :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ Finset.range 3009 ∧ p.2 ∈ Finset.range 3009) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 3009) ∧
    pairs.card = max_pairs :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_correct_max_pairs_achievable_l2682_268231


namespace NUMINAMATH_CALUDE_bus_ride_difference_l2682_268220

-- Define the lengths of the bus rides
def oscar_ride : ℝ := 0.75
def charlie_ride : ℝ := 0.25

-- Theorem statement
theorem bus_ride_difference : oscar_ride - charlie_ride = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l2682_268220


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2682_268214

theorem mean_equality_implies_z_value :
  let mean1 := (8 + 15 + 24) / 3
  let mean2 := (18 + z) / 2
  mean1 = mean2 → z = 40 / 3 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2682_268214


namespace NUMINAMATH_CALUDE_original_earnings_before_raise_l2682_268272

/-- If a person's weekly earnings increased by 50% to $75, prove that their original weekly earnings were $50. -/
theorem original_earnings_before_raise (new_earnings : ℝ) (increase_percentage : ℝ) :
  new_earnings = 75 →
  increase_percentage = 50 →
  ∃ original_earnings : ℝ,
    original_earnings * (1 + increase_percentage / 100) = new_earnings ∧
    original_earnings = 50 :=
by sorry

end NUMINAMATH_CALUDE_original_earnings_before_raise_l2682_268272


namespace NUMINAMATH_CALUDE_prod_mod_seven_l2682_268286

theorem prod_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_prod_mod_seven_l2682_268286


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_l2682_268221

def line1 (t k : ℝ) : ℝ × ℝ × ℝ := (3 + 2*t, 2 + 3*t, 2 - k*t)
def line2 (u k : ℝ) : ℝ × ℝ × ℝ := (1 + k*u, 5 - u, 6 + 2*u)

def are_coplanar (k : ℝ) : Prop :=
  ∃ (a b c d : ℝ), ∀ (t u : ℝ),
    a * (line1 t k).1 + b * (line1 t k).2.1 + c * (line1 t k).2.2 + d = 0 ∧
    a * (line2 u k).1 + b * (line2 u k).2.1 + c * (line2 u k).2.2 + d = 0

theorem lines_coplanar_iff_k (k : ℝ) :
  are_coplanar k ↔ (k = -5 - 3 * Real.sqrt 3 ∨ k = -5 + 3 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_l2682_268221


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l2682_268239

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l2682_268239


namespace NUMINAMATH_CALUDE_marks_of_a_l2682_268203

theorem marks_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 48 →
  (a + b + c + d) / 4 = 47 →
  e = d + 3 →
  (b + c + d + e) / 4 = 48 →
  a = 43 := by
sorry

end NUMINAMATH_CALUDE_marks_of_a_l2682_268203


namespace NUMINAMATH_CALUDE_bull_work_hours_equality_l2682_268290

/-- Represents the work rate of bulls ploughing fields -/
structure BullWork where
  bulls : ℕ
  fields : ℕ
  days : ℕ
  hours_per_day : ℝ

/-- Calculates the total bull-hours for a given BullWork -/
def total_bull_hours (work : BullWork) : ℝ :=
  work.bulls * work.fields * work.days * work.hours_per_day

theorem bull_work_hours_equality (work1 work2 : BullWork) 
  (h1 : work1.bulls = 10)
  (h2 : work1.fields = 20)
  (h3 : work1.days = 3)
  (h4 : work2.bulls = 30)
  (h5 : work2.fields = 32)
  (h6 : work2.days = 2)
  (h7 : work2.hours_per_day = 8)
  (h8 : total_bull_hours work1 = total_bull_hours work2) :
  work1.hours_per_day = 12.8 := by
  sorry

end NUMINAMATH_CALUDE_bull_work_hours_equality_l2682_268290


namespace NUMINAMATH_CALUDE_trihedral_angle_sum_l2682_268234

/-- Represents a trihedral angle with plane angles α, β, and γ. -/
structure TrihedralAngle where
  α : ℝ
  β : ℝ
  γ : ℝ
  α_pos : 0 < α
  β_pos : 0 < β
  γ_pos : 0 < γ

/-- The sum of any two plane angles of a trihedral angle is greater than the third. -/
theorem trihedral_angle_sum (t : TrihedralAngle) : t.α + t.β > t.γ ∧ t.β + t.γ > t.α ∧ t.α + t.γ > t.β := by
  sorry

end NUMINAMATH_CALUDE_trihedral_angle_sum_l2682_268234


namespace NUMINAMATH_CALUDE_irregular_polygon_rotation_implies_composite_l2682_268248

/-- An n-gon inscribed in a circle -/
structure InscribedPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Rotation of a point about a center by an angle -/
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- A polygon is irregular if not all its sides are equal -/
def isIrregular (p : InscribedPolygon n) : Prop := sorry

/-- A polygon coincides with itself under rotation -/
def coincidesSelfUnderRotation (p : InscribedPolygon n) (angle : ℝ) : Prop := sorry

/-- A number is composite if it's not prime and greater than 1 -/
def isComposite (n : ℕ) : Prop := ¬(Nat.Prime n) ∧ n > 1

theorem irregular_polygon_rotation_implies_composite 
  (n : ℕ) (p : InscribedPolygon n) (α : ℝ) :
  isIrregular p →
  α ≠ 2 * Real.pi →
  coincidesSelfUnderRotation p α →
  isComposite n := by
  sorry

end NUMINAMATH_CALUDE_irregular_polygon_rotation_implies_composite_l2682_268248


namespace NUMINAMATH_CALUDE_probability_multiple_of_three_l2682_268201

def is_multiple_of_three (n : ℕ) : Bool :=
  n % 3 = 0

def count_multiples_of_three (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_three |>.length

theorem probability_multiple_of_three : 
  (count_multiples_of_three 24 : ℚ) / 24 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_of_three_l2682_268201


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2682_268271

/-- Proves that for a rectangle with length 40 cm and breadth 20 cm,
    if the length is decreased by 5 cm and the breadth is increased by 5 cm,
    the area of the new rectangle is 75 sq. cm larger than the original rectangle. -/
theorem rectangle_area_increase :
  let original_length : ℝ := 40
  let original_breadth : ℝ := 20
  let new_length : ℝ := original_length - 5
  let new_breadth : ℝ := original_breadth + 5
  let original_area : ℝ := original_length * original_breadth
  let new_area : ℝ := new_length * new_breadth
  new_area - original_area = 75 :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_increase_l2682_268271


namespace NUMINAMATH_CALUDE_equation_solution_l2682_268283

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -1 ∧ x₂ = 5 ∧ 
  (∀ x : ℝ, (x + 1)^2 = 6*x + 6 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2682_268283


namespace NUMINAMATH_CALUDE_largest_divisor_of_fifth_power_minus_self_l2682_268287

/-- A number is composite if it has a proper divisor -/
def IsComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The largest integer that always divides n^5 - n for all composite n -/
def LargestCommonDivisor : ℕ := 6

theorem largest_divisor_of_fifth_power_minus_self :
  ∀ n : ℕ, IsComposite n → (n^5 - n) % LargestCommonDivisor = 0 ∧
  ∀ k : ℕ, k > LargestCommonDivisor → ∃ m : ℕ, IsComposite m ∧ (m^5 - m) % k ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_fifth_power_minus_self_l2682_268287


namespace NUMINAMATH_CALUDE_item_distribution_l2682_268250

theorem item_distribution (n₁ n₂ n₃ k : ℕ) (h₁ : n₁ = 5) (h₂ : n₂ = 3) (h₃ : n₃ = 2) (h₄ : k = 3) :
  (Nat.choose (n₁ + k - 1) (k - 1)) * (Nat.choose (n₂ + k - 1) (k - 1)) * (Nat.choose (n₃ + k - 1) (k - 1)) = 1260 :=
by sorry

end NUMINAMATH_CALUDE_item_distribution_l2682_268250


namespace NUMINAMATH_CALUDE_W_min_value_l2682_268217

/-- The function W defined on real numbers x and y -/
def W (x y : ℝ) : ℝ := 5 * x^2 - 4 * x * y + y^2 - 2 * y + 8 * x + 3

/-- Theorem stating that W has a minimum value of -2 -/
theorem W_min_value :
  (∀ x y : ℝ, W x y ≥ -2) ∧ (∃ x y : ℝ, W x y = -2) := by
  sorry

end NUMINAMATH_CALUDE_W_min_value_l2682_268217


namespace NUMINAMATH_CALUDE_sport_corn_syrup_amount_l2682_268209

/-- Represents the ratios in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio (r : DrinkRatio) : DrinkRatio :=
  { flavoring := r.flavoring,
    corn_syrup := r.corn_syrup / 3,
    water := r.water * 2 }

/-- Theorem stating the amount of corn syrup in the sport formulation -/
theorem sport_corn_syrup_amount (water_amount : ℚ) :
  let sr := sport_ratio standard_ratio
  let flavoring := water_amount / sr.water
  flavoring * sr.corn_syrup = 7 :=
sorry

end NUMINAMATH_CALUDE_sport_corn_syrup_amount_l2682_268209


namespace NUMINAMATH_CALUDE_team_B_eligible_l2682_268261

-- Define the height limit for submarine service
def submarine_height_limit : ℝ := 168

-- Define the teams and their height characteristics
structure Team where
  name : String
  height_characteristic : ℝ

-- Define the four teams
def team_A : Team := ⟨"A", 166⟩
def team_B : Team := ⟨"B", 167⟩
def team_C : Team := ⟨"C", 169⟩
def team_D : Team := ⟨"D", 167⟩

-- Define a function to check if a team is eligible
def is_eligible (t : Team) : Prop :=
  t.name = "B" ∧ t.height_characteristic ≤ submarine_height_limit

-- Theorem statement
theorem team_B_eligible :
  ∀ t : Team, is_eligible t ↔ t = team_B :=
sorry

end NUMINAMATH_CALUDE_team_B_eligible_l2682_268261


namespace NUMINAMATH_CALUDE_problem_solution_l2682_268216

theorem problem_solution (m n : ℝ) : 
  (∃ k : ℝ, k^2 = m + 3 ∧ (k = 1 ∨ k = -1)) →
  (2*n - 12)^(1/3) = 4 →
  m = -2 ∧ n = 38 ∧ Real.sqrt (m + n) = 6 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2682_268216


namespace NUMINAMATH_CALUDE_boat_distance_theorem_l2682_268274

/-- Calculates the distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: A boat with speed 40 km/hr in still water, traveling in a stream 
    with speed 5 km/hr for 1 hour downstream, travels 45 km -/
theorem boat_distance_theorem :
  distance_downstream 40 5 1 = 45 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_theorem_l2682_268274


namespace NUMINAMATH_CALUDE_perpendicular_vector_k_value_l2682_268255

theorem perpendicular_vector_k_value :
  let a : Fin 2 → ℝ := ![1, 1]
  let b : Fin 2 → ℝ := ![2, -3]
  ∀ k : ℝ, (k • a - 2 • b) • a = 0 → k = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_k_value_l2682_268255


namespace NUMINAMATH_CALUDE_product_of_zero_functions_is_zero_function_l2682_268225

-- Define the concept of a zero function on a domain D
def is_zero_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, f x = 0

-- State the theorem
theorem product_of_zero_functions_is_zero_function 
  (f g : ℝ → ℝ) (D : Set ℝ) 
  (hf : is_zero_function f D) (hg : is_zero_function g D) : 
  is_zero_function (fun x ↦ f x * g x) D :=
sorry

end NUMINAMATH_CALUDE_product_of_zero_functions_is_zero_function_l2682_268225


namespace NUMINAMATH_CALUDE_max_xy_value_l2682_268270

theorem max_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : Real.sqrt 3 = Real.sqrt (9^x * 3^y)) : 
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ Real.sqrt 3 = Real.sqrt (9^a * 3^b) → x * y ≥ a * b) ∧ 
  x * y = 1/8 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l2682_268270


namespace NUMINAMATH_CALUDE_mryak_bryak_price_difference_l2682_268266

/-- The price of one "mryak" in rubles -/
def mryak_price : ℝ := sorry

/-- The price of one "bryak" in rubles -/
def bryak_price : ℝ := sorry

/-- Three "mryak" are 10 rubles more expensive than five "bryak" -/
axiom price_relation1 : 3 * mryak_price = 5 * bryak_price + 10

/-- Six "mryak" are 31 rubles more expensive than eight "bryak" -/
axiom price_relation2 : 6 * mryak_price = 8 * bryak_price + 31

/-- The price difference between seven "mryak" and nine "bryak" is 38 rubles -/
theorem mryak_bryak_price_difference : 7 * mryak_price - 9 * bryak_price = 38 := by
  sorry

end NUMINAMATH_CALUDE_mryak_bryak_price_difference_l2682_268266


namespace NUMINAMATH_CALUDE_parallelogram_area_l2682_268295

theorem parallelogram_area (base height area : ℝ) : 
  base = 10 →
  height = 2 * base →
  area = base * height →
  area = 200 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2682_268295


namespace NUMINAMATH_CALUDE_cubic_function_theorem_l2682_268240

/-- A cubic function with a parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

/-- The derivative of f -/
def f' (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3

theorem cubic_function_theorem (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f c x₁ = 0 ∧ f c x₂ = 0) →  -- two distinct zeros
  (∃ x₀ : ℝ, f c x₀ = 0 ∧ ∀ x : ℝ, f c x ≤ f c x₀) →  -- one zero is the maximum point
  c = -2 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_theorem_l2682_268240


namespace NUMINAMATH_CALUDE_number_of_friends_who_received_pebbles_l2682_268298

-- Define the given quantities
def total_weight_kg : ℕ := 36
def pebble_weight_g : ℕ := 250
def pebbles_per_friend : ℕ := 4

-- Define the conversion factor from kg to g
def kg_to_g : ℕ := 1000

-- Theorem to prove
theorem number_of_friends_who_received_pebbles :
  (total_weight_kg * kg_to_g) / (pebble_weight_g * pebbles_per_friend) = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_of_friends_who_received_pebbles_l2682_268298


namespace NUMINAMATH_CALUDE_bricklayer_problem_l2682_268205

theorem bricklayer_problem (x : ℝ) 
  (h1 : (x / 12 + x / 15 - 15) * 6 = x) : x = 900 := by
  sorry

#check bricklayer_problem

end NUMINAMATH_CALUDE_bricklayer_problem_l2682_268205


namespace NUMINAMATH_CALUDE_x_value_proof_l2682_268276

theorem x_value_proof (x : ℝ) : 
  (⌊x⌋ + ⌈x⌉ = 7) ∧ (10 ≤ 3*x - 5 ∧ 3*x - 5 ≤ 13) → 3 < x ∧ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_x_value_proof_l2682_268276


namespace NUMINAMATH_CALUDE_box_of_balls_l2682_268265

theorem box_of_balls (N : ℕ) : N - 44 = 70 - N → N = 57 := by
  sorry

end NUMINAMATH_CALUDE_box_of_balls_l2682_268265


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2682_268249

theorem solution_set_inequality (x : ℝ) : (x - 1) * (3 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2682_268249


namespace NUMINAMATH_CALUDE_square_symbol_function_l2682_268252

/-- Represents the possible functions of symbols in a program flowchart -/
inductive FlowchartSymbolFunction
  | Output
  | Assignment
  | Decision
  | EndOfAlgorithm
  | Calculation

/-- Represents a symbol in a program flowchart -/
structure FlowchartSymbol where
  shape : String
  function : FlowchartSymbolFunction

/-- The square symbol in a program flowchart -/
def squareSymbol : FlowchartSymbol :=
  { shape := "□", function := FlowchartSymbolFunction.Assignment }

/-- Theorem stating the function of the square symbol in a program flowchart -/
theorem square_symbol_function :
  (squareSymbol.function = FlowchartSymbolFunction.Assignment) ∨
  (squareSymbol.function = FlowchartSymbolFunction.Calculation) :=
by sorry

end NUMINAMATH_CALUDE_square_symbol_function_l2682_268252


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2682_268237

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 5*x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2682_268237


namespace NUMINAMATH_CALUDE_consecutive_numbers_percentage_l2682_268269

theorem consecutive_numbers_percentage (a b c d e f g : ℤ) : 
  (a + b + c + d + e + f + g) / 7 = 9 ∧ 
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧ f = e + 1 ∧ g = f + 1 →
  a * 100 / g = 50 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_percentage_l2682_268269


namespace NUMINAMATH_CALUDE_no_three_digit_even_with_digit_sum_27_l2682_268242

/-- A function that returns the digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is a 3-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A theorem stating that there are no 3-digit even numbers with a digit sum of 27 -/
theorem no_three_digit_even_with_digit_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ Even n ∧ digit_sum n = 27 := by sorry

end NUMINAMATH_CALUDE_no_three_digit_even_with_digit_sum_27_l2682_268242


namespace NUMINAMATH_CALUDE_ticket_distribution_l2682_268213

/-- The number of ways to distribute 5 consecutive tickets to 5 people. -/
def distribute_tickets : ℕ := 5 * 4 * 3 * 2 * 1

/-- The number of ways for A and B to receive consecutive tickets. -/
def consecutive_for_ab : ℕ := 4 * 2

/-- The number of ways to distribute the remaining tickets to 3 people. -/
def distribute_remaining : ℕ := 3 * 2 * 1

/-- 
Theorem: The number of ways to distribute 5 consecutive movie tickets to 5 people, 
including A and B, such that A and B receive consecutive tickets, is equal to 48.
-/
theorem ticket_distribution : 
  consecutive_for_ab * distribute_remaining = 48 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_l2682_268213


namespace NUMINAMATH_CALUDE_total_money_l2682_268215

def jack_money : ℕ := 26
def ben_money : ℕ := jack_money - 9
def eric_money : ℕ := ben_money - 10

theorem total_money : eric_money + ben_money + jack_money = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2682_268215


namespace NUMINAMATH_CALUDE_fraction_calculation_l2682_268200

theorem fraction_calculation : 
  (7/6) / ((1/6) - (1/3)) * (3/14) / (3/5) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2682_268200


namespace NUMINAMATH_CALUDE_average_minutes_theorem_l2682_268243

/-- Represents the distribution of attendees and their listening durations --/
structure LectureAttendance where
  total_attendees : ℕ
  full_listeners : ℕ
  sleepers : ℕ
  half_listeners : ℕ
  quarter_listeners : ℕ
  lecture_duration : ℕ

/-- Calculates the average minutes heard by attendees --/
def average_minutes_heard (attendance : LectureAttendance) : ℚ :=
  let full_minutes := attendance.full_listeners * attendance.lecture_duration
  let half_minutes := attendance.half_listeners * (attendance.lecture_duration / 2)
  let quarter_minutes := attendance.quarter_listeners * (attendance.lecture_duration / 4)
  let total_minutes := full_minutes + half_minutes + quarter_minutes
  (total_minutes : ℚ) / attendance.total_attendees

/-- The theorem stating the average minutes heard is 59.1 --/
theorem average_minutes_theorem (attendance : LectureAttendance) 
  (h1 : attendance.lecture_duration = 120)
  (h2 : attendance.full_listeners = attendance.total_attendees * 30 / 100)
  (h3 : attendance.sleepers = attendance.total_attendees * 15 / 100)
  (h4 : attendance.half_listeners = (attendance.total_attendees - attendance.full_listeners - attendance.sleepers) * 40 / 100)
  (h5 : attendance.quarter_listeners = attendance.total_attendees - attendance.full_listeners - attendance.sleepers - attendance.half_listeners) :
  average_minutes_heard attendance = 591/10 := by
  sorry

end NUMINAMATH_CALUDE_average_minutes_theorem_l2682_268243


namespace NUMINAMATH_CALUDE_inverse_113_mod_114_l2682_268204

theorem inverse_113_mod_114 : ∃ x : ℕ, x ∈ Finset.range 114 ∧ (113 * x) % 114 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_inverse_113_mod_114_l2682_268204


namespace NUMINAMATH_CALUDE_permutations_of_middle_digits_l2682_268280

/-- The number of permutations of four digits with two pairs of repeated digits -/
def permutations_with_repetition : ℕ := 6

/-- The set of digits to be permuted -/
def digits : Finset ℕ := {2, 2, 3, 3}

/-- The theorem stating that the number of permutations of the given digits is 6 -/
theorem permutations_of_middle_digits :
  Finset.card (Finset.powersetCard 4 digits) = permutations_with_repetition :=
sorry

end NUMINAMATH_CALUDE_permutations_of_middle_digits_l2682_268280


namespace NUMINAMATH_CALUDE_min_framing_for_picture_l2682_268223

/-- Calculates the minimum number of linear feet of framing needed for an enlarged picture with a border -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (framed_width + framed_height)
  (perimeter_inches + 11) / 12  -- Round up to the nearest foot

/-- The minimum number of linear feet of framing needed for the given picture specifications -/
theorem min_framing_for_picture : min_framing_feet 4 6 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_framing_for_picture_l2682_268223


namespace NUMINAMATH_CALUDE_sum_reciprocal_l2682_268224

theorem sum_reciprocal (x : ℝ) (w : ℝ) (h1 : x ≠ 0) (h2 : w = x^2 + (1/x)^2) (h3 : w = 23) :
  x + (1/x) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_l2682_268224


namespace NUMINAMATH_CALUDE_marie_stamps_giveaway_l2682_268273

theorem marie_stamps_giveaway (notebooks : Nat) (stamps_per_notebook : Nat)
  (binders : Nat) (stamps_per_binder : Nat) (keep_percentage : Rat) :
  notebooks = 30 →
  stamps_per_notebook = 120 →
  binders = 7 →
  stamps_per_binder = 210 →
  keep_percentage = 35 / 100 →
  (notebooks * stamps_per_notebook + binders * stamps_per_binder : Nat) -
    (((notebooks * stamps_per_notebook + binders * stamps_per_binder : Nat) : Rat) *
      keep_percentage).floor.toNat = 3296 := by
  sorry

end NUMINAMATH_CALUDE_marie_stamps_giveaway_l2682_268273


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2682_268218

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-3 : ℝ) 2 = {x : ℝ | a * x^2 + 5 * x + b < 0}) : 
  Set.Ioo (-1/3 : ℝ) (1/2 : ℝ) = {x : ℝ | b * x^2 + 5 * x + a > 0} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2682_268218


namespace NUMINAMATH_CALUDE_same_color_probability_l2682_268267

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of red balls in the bag -/
def red_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 2

/-- Calculates the number of combinations of n items taken r at a time -/
def combinations (n r : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

/-- The probability of drawing two balls of the same color -/
theorem same_color_probability : 
  (combinations white_balls drawn_balls + combinations red_balls drawn_balls) / 
  combinations total_balls drawn_balls = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2682_268267


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2682_268241

/-- Given two supplementary angles C and D, where the measure of angle C is 5 times
    the measure of angle D, prove that the measure of angle C is 150°. -/
theorem angle_measure_proof (C D : ℝ) : 
  C + D = 180 →  -- Angles C and D are supplementary
  C = 5 * D →    -- Measure of angle C is 5 times angle D
  C = 150 := by  -- Measure of angle C is 150°
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2682_268241


namespace NUMINAMATH_CALUDE_household_survey_total_l2682_268289

theorem household_survey_total (total : ℕ) (neither : ℕ) (only_a : ℕ) (both : ℕ) : 
  total = 180 ∧ 
  neither = 80 ∧ 
  only_a = 60 ∧ 
  both = 10 ∧ 
  (∃ (only_b : ℕ), only_b = 3 * both) →
  total = neither + only_a + both + (3 * both) :=
by sorry

end NUMINAMATH_CALUDE_household_survey_total_l2682_268289


namespace NUMINAMATH_CALUDE_exam_total_questions_l2682_268244

/-- Represents an exam with given parameters -/
structure Exam where
  totalTime : ℕ
  answeredQuestions : ℕ
  timeUsed : ℕ
  timeLeftWhenFinished : ℕ

/-- Calculates the total number of questions on the exam -/
def totalQuestions (e : Exam) : ℕ :=
  let remainingTime := e.totalTime - e.timeUsed
  let questionRate := e.answeredQuestions / e.timeUsed
  e.answeredQuestions + questionRate * remainingTime

/-- Theorem stating that the total number of questions on the given exam is 80 -/
theorem exam_total_questions :
  let e : Exam := {
    totalTime := 60,
    answeredQuestions := 16,
    timeUsed := 12,
    timeLeftWhenFinished := 0
  }
  totalQuestions e = 80 := by
  sorry


end NUMINAMATH_CALUDE_exam_total_questions_l2682_268244


namespace NUMINAMATH_CALUDE_roots_relation_l2682_268232

/-- The polynomial h(x) -/
def h (x : ℝ) : ℝ := x^3 - 2*x^2 + 4*x - 1

/-- The polynomial j(x) -/
def j (x p q r : ℝ) : ℝ := x^3 + p*x^2 + q*x + r

/-- Theorem stating the relationship between h(x) and j(x) and the values of p, q, and r -/
theorem roots_relation (p q r : ℝ) : 
  (∀ s, h s = 0 → ∃ t, j t p q r = 0 ∧ s = t + 2) → 
  p = 4 ∧ q = 8 ∧ r = 7 := by
  sorry

end NUMINAMATH_CALUDE_roots_relation_l2682_268232


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2682_268212

theorem inequality_equivalence (x : ℝ) :
  (x - 4) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2682_268212


namespace NUMINAMATH_CALUDE_count_integers_in_range_l2682_268211

theorem count_integers_in_range : ∃ (S : Finset Int), 
  (∀ n : Int, n ∈ S ↔ 15 < n^2 ∧ n^2 < 120) ∧ Finset.card S = 14 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l2682_268211


namespace NUMINAMATH_CALUDE_student_rank_from_first_l2682_268257

/-- Given a class of students, proves that a student with a certain rank from the last
    has a corresponding rank from the first. -/
theorem student_rank_from_first (total_students : ℕ) (rank_from_last : ℕ) :
  total_students = 58 → rank_from_last = 34 → total_students - rank_from_last + 1 = 25 := by
  sorry

#check student_rank_from_first

end NUMINAMATH_CALUDE_student_rank_from_first_l2682_268257


namespace NUMINAMATH_CALUDE_expression_simplification_l2682_268281

theorem expression_simplification (x y : ℝ) : 
  4 * x + 8 * x^2 + 6 * y - (3 - 5 * x - 8 * x^2 - 2 * y) = 16 * x^2 + 9 * x + 8 * y - 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2682_268281


namespace NUMINAMATH_CALUDE_least_k_for_convergence_l2682_268227

def u : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 3 * u n - 5 * (u n)^2

def L : ℚ := 1/5

theorem least_k_for_convergence :
  ∀ k < 7, |u k - L| > 1/2^100 ∧ |u 7 - L| ≤ 1/2^100 := by sorry

end NUMINAMATH_CALUDE_least_k_for_convergence_l2682_268227


namespace NUMINAMATH_CALUDE_cube_root_five_to_seven_sum_l2682_268254

theorem cube_root_five_to_seven_sum : 
  (5^7 + 5^7 + 5^7 + 5^7 + 5^7 : ℝ)^(1/3) = 25 * (5^2)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_five_to_seven_sum_l2682_268254


namespace NUMINAMATH_CALUDE_blanket_price_problem_l2682_268297

/-- Given the conditions of blanket purchases and average price, prove the unknown rate --/
theorem blanket_price_problem (price1 price2 avg_price : ℚ) (count1 count2 count_unknown : ℕ) :
  price1 = 100 →
  price2 = 150 →
  avg_price = 150 →
  count1 = 3 →
  count2 = 3 →
  count_unknown = 2 →
  let total_count := count1 + count2 + count_unknown
  let total_price := count1 * price1 + count2 * price2 + count_unknown * avg_price
  let unknown_rate := (total_price - count1 * price1 - count2 * price2) / count_unknown
  unknown_rate = 225 := by
  sorry

end NUMINAMATH_CALUDE_blanket_price_problem_l2682_268297


namespace NUMINAMATH_CALUDE_smallest_k_with_odd_solutions_l2682_268268

/-- The number of positive integral solutions to the equation 2xy - 3x - 5y = k -/
def num_solutions (k : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 * p.2 - 3 * p.1 - 5 * p.2 = k) (Finset.product (Finset.range 1000) (Finset.range 1000))).card

/-- Predicate to check if a number is odd -/
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem smallest_k_with_odd_solutions :
  (∀ k < 5, ¬(is_odd (num_solutions k))) ∧ 
  (is_odd (num_solutions 5)) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_with_odd_solutions_l2682_268268


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_planes_perp_to_line_are_parallel_l2682_268207

/-- A 3D space -/
structure Space3D where
  -- Add necessary structure here

/-- A line in 3D space -/
structure Line3D (S : Space3D) where
  -- Add necessary structure here

/-- A plane in 3D space -/
structure Plane3D (S : Space3D) where
  -- Add necessary structure here

/-- Perpendicularity between a line and a plane -/
def perpendicular_line_plane (S : Space3D) (l : Line3D S) (p : Plane3D S) : Prop :=
  sorry

/-- Perpendicularity between a plane and a line -/
def perpendicular_plane_line (S : Space3D) (p : Plane3D S) (l : Line3D S) : Prop :=
  sorry

/-- Parallelism between two lines -/
def parallel_lines (S : Space3D) (l1 l2 : Line3D S) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel_planes (S : Space3D) (p1 p2 : Plane3D S) : Prop :=
  sorry

/-- Theorem: Two lines perpendicular to the same plane are parallel to each other -/
theorem lines_perp_to_plane_are_parallel (S : Space3D) (l1 l2 : Line3D S) (p : Plane3D S)
  (h1 : perpendicular_line_plane S l1 p) (h2 : perpendicular_line_plane S l2 p) :
  parallel_lines S l1 l2 :=
sorry

/-- Theorem: Two planes perpendicular to the same line are parallel to each other -/
theorem planes_perp_to_line_are_parallel (S : Space3D) (p1 p2 : Plane3D S) (l : Line3D S)
  (h1 : perpendicular_plane_line S p1 l) (h2 : perpendicular_plane_line S p2 l) :
  parallel_planes S p1 p2 :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_planes_perp_to_line_are_parallel_l2682_268207


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l2682_268251

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Define the universal set U
def U (m : ℝ) : Set ℝ := A ∪ B m

-- Theorem for part (1)
theorem part_one (m : ℝ) (h : m = 3) : 
  A ∩ (U m \ B m) = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem for part (2)
theorem part_two : 
  {m : ℝ | A ∩ B m = ∅} = {m : ℝ | m ≤ -2} := by sorry

-- Theorem for part (3)
theorem part_three : 
  {m : ℝ | A ∩ B m = A} = {m : ℝ | m ≥ 4} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l2682_268251


namespace NUMINAMATH_CALUDE_apple_orange_ratio_l2682_268210

/-- Represents the number of fruits each child received -/
structure FruitDistribution where
  mike_oranges : ℕ
  matt_apples : ℕ
  mark_bananas : ℕ

/-- The fruit distribution satisfies the problem conditions -/
def valid_distribution (d : FruitDistribution) : Prop :=
  d.mike_oranges = 3 ∧
  d.mark_bananas = d.mike_oranges + d.matt_apples ∧
  d.mike_oranges + d.matt_apples + d.mark_bananas = 18

theorem apple_orange_ratio (d : FruitDistribution) 
  (h : valid_distribution d) : 
  d.matt_apples / d.mike_oranges = 2 := by
  sorry

#check apple_orange_ratio

end NUMINAMATH_CALUDE_apple_orange_ratio_l2682_268210


namespace NUMINAMATH_CALUDE_country_club_cost_l2682_268293

/-- Calculates the amount one person pays for the first year of a country club membership,
    given they pay half the total cost for a group. -/
theorem country_club_cost
  (num_people : ℕ)
  (joining_fee : ℕ)
  (monthly_cost : ℕ)
  (months_in_year : ℕ)
  (h_num_people : num_people = 4)
  (h_joining_fee : joining_fee = 4000)
  (h_monthly_cost : monthly_cost = 1000)
  (h_months_in_year : months_in_year = 12) :
  (num_people * joining_fee + num_people * monthly_cost * months_in_year) / 2 = 32000 := by
  sorry

#check country_club_cost

end NUMINAMATH_CALUDE_country_club_cost_l2682_268293


namespace NUMINAMATH_CALUDE_complex_equality_modulus_l2682_268219

theorem complex_equality_modulus (x y : ℝ) (i : ℂ) : 
  i * i = -1 →
  (2 + i) * (3 - x * i) = 3 + (y + 5) * i →
  Complex.abs (x + y * i) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_modulus_l2682_268219


namespace NUMINAMATH_CALUDE_quadratic_root_reciprocal_l2682_268206

/-- If m is a root of ax² + bx + 1 = 0, then 1/m is a root of x² + bx + a = 0 -/
theorem quadratic_root_reciprocal (a b m : ℝ) (hm : m ≠ 0) 
  (h : a * m^2 + b * m + 1 = 0) : 
  (1/m)^2 + b * (1/m) + a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_reciprocal_l2682_268206


namespace NUMINAMATH_CALUDE_tan_sum_of_roots_l2682_268235

theorem tan_sum_of_roots (α β : Real) : 
  (∃ (x : Real), x^2 - 3 * Real.sqrt 3 * x + 4 = 0 ∧ x = Real.tan α) ∧
  (∃ (y : Real), y^2 - 3 * Real.sqrt 3 * y + 4 = 0 ∧ y = Real.tan β) ∧
  α ∈ Set.Ioo (-π/2) (π/2) ∧
  β ∈ Set.Ioo (-π/2) (π/2) →
  Real.tan (α + β) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_of_roots_l2682_268235


namespace NUMINAMATH_CALUDE_wilted_flower_ratio_l2682_268279

theorem wilted_flower_ratio (initial_roses : ℕ) (remaining_flowers : ℕ) :
  initial_roses = 36 →
  remaining_flowers = 12 →
  (initial_roses / 2 - remaining_flowers) / (initial_roses / 2) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_wilted_flower_ratio_l2682_268279


namespace NUMINAMATH_CALUDE_mia_study_time_l2682_268278

theorem mia_study_time (total_minutes : ℕ) (tv_fraction : ℚ) (study_minutes : ℕ) : 
  total_minutes = 1440 →
  tv_fraction = 1 / 5 →
  study_minutes = 288 →
  (study_minutes : ℚ) / (total_minutes - (tv_fraction * total_minutes : ℚ)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_mia_study_time_l2682_268278


namespace NUMINAMATH_CALUDE_polynomial_value_l2682_268288

theorem polynomial_value (x y : ℝ) (h : x - 2*y + 3 = 8) : x - 2*y = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l2682_268288


namespace NUMINAMATH_CALUDE_unique_consecutive_set_sum_20_l2682_268236

/-- A set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum : ℕ
  h1 : start ≥ 2
  h2 : length ≥ 2
  h3 : sum = (length * (2 * start + length - 1)) / 2

/-- The theorem stating that there is exactly one set of consecutive positive integers
    starting from 2 or higher, with at least two numbers, whose sum is 20 -/
theorem unique_consecutive_set_sum_20 :
  ∃! (s : ConsecutiveSet), s.sum = 20 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_set_sum_20_l2682_268236


namespace NUMINAMATH_CALUDE_bread_cost_calculation_l2682_268292

/-- Calculates the total cost of bread for a committee luncheon --/
def calculate_bread_cost (committee_size : ℕ) (sandwiches_per_person : ℕ) 
  (bread_types : ℕ) (croissant_pack_size : ℕ) (croissant_pack_price : ℚ)
  (ciabatta_pack_size : ℕ) (ciabatta_pack_price : ℚ)
  (multigrain_pack_size : ℕ) (multigrain_pack_price : ℚ)
  (discount_threshold : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  sorry

/-- The total cost of bread for the committee luncheon is $51.36 --/
theorem bread_cost_calculation :
  calculate_bread_cost 24 2 3 12 8 10 9 20 7 50 0.1 0.07 = 51.36 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_calculation_l2682_268292


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_two_plus_sqrt_three_l2682_268259

theorem sqrt_sum_equals_sqrt_of_two_plus_sqrt_three (a b : ℚ) :
  Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3) →
  ((a = 1/2 ∧ b = 3/2) ∨ (a = 3/2 ∧ b = 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_two_plus_sqrt_three_l2682_268259


namespace NUMINAMATH_CALUDE_college_students_count_l2682_268291

theorem college_students_count : ℕ :=
  let students_to_professors_ratio : ℕ := 15
  let total_people : ℕ := 40000
  let students : ℕ := 37500

  have h1 : students = students_to_professors_ratio * (total_people - students) := by sorry
  have h2 : students + (total_people - students) = total_people := by sorry

  students

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_college_students_count_l2682_268291


namespace NUMINAMATH_CALUDE_refrigerator_deposit_l2682_268277

/-- Proves the deposit amount for a refrigerator purchase with installments -/
theorem refrigerator_deposit (cash_price : ℕ) (num_installments : ℕ) (installment_amount : ℕ) (savings : ℕ) : 
  cash_price = 8000 →
  num_installments = 30 →
  installment_amount = 300 →
  savings = 4000 →
  cash_price + savings = num_installments * installment_amount + (cash_price + savings - num_installments * installment_amount) :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_deposit_l2682_268277


namespace NUMINAMATH_CALUDE_min_pieces_for_horizontal_four_l2682_268202

/-- Represents a chessboard as a list of 8 rows, each containing 8 cells --/
def Chessboard := List (List Bool)

/-- Checks if a row contains 4 consecutive true values --/
def hasFourConsecutive (row : List Bool) : Bool :=
  sorry

/-- Checks if any row in the chessboard has 4 consecutive pieces --/
def hasHorizontalFour (board : Chessboard) : Bool :=
  sorry

/-- Generates all possible arrangements of n pieces on a chessboard --/
def allArrangements (n : Nat) : List Chessboard :=
  sorry

theorem min_pieces_for_horizontal_four :
  ∀ n : Nat, (n ≥ 49 ↔ ∀ board ∈ allArrangements n, hasHorizontalFour board) :=
by sorry

end NUMINAMATH_CALUDE_min_pieces_for_horizontal_four_l2682_268202


namespace NUMINAMATH_CALUDE_square_diff_over_square_sum_l2682_268282

theorem square_diff_over_square_sum (a b : ℝ) (h : a * b / (a^2 + b^2) = 1/4) :
  |a^2 - b^2| / (a^2 + b^2) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_over_square_sum_l2682_268282


namespace NUMINAMATH_CALUDE_purple_probability_ten_sided_die_l2682_268294

/-- Represents a die with a specific number of sides and purple faces -/
structure Die :=
  (sides : ℕ)
  (purpleFaces : ℕ)
  (hPurple : purpleFaces ≤ sides)

/-- Calculates the probability of rolling a purple face on a given die -/
def probabilityPurple (d : Die) : ℚ :=
  d.purpleFaces / d.sides

/-- Theorem stating that for a 10-sided die with 2 purple faces, 
    the probability of rolling a purple face is 1/5 -/
theorem purple_probability_ten_sided_die :
  ∀ d : Die, d.sides = 10 → d.purpleFaces = 2 → probabilityPurple d = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_purple_probability_ten_sided_die_l2682_268294


namespace NUMINAMATH_CALUDE_triangle_theorem_l2682_268228

/-- Triangle ABC with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sqrt 2 * t.b * t.c = t.b^2 + t.c^2 - t.a^2) :
  t.A = π / 4 ∧ 
  (t.a = 2 * Real.sqrt 2 ∧ t.B = π / 3 → t.b = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2682_268228


namespace NUMINAMATH_CALUDE_parabola_line_intersection_theorem_l2682_268299

-- Define the parabola C: y^2 = 3x
def C (x y : ℝ) : Prop := y^2 = 3*x

-- Define the line l: y = (3/2)x + b
def l (x y b : ℝ) : Prop := y = (3/2)*x + b

-- Define the intersection points E and F
def E (x y : ℝ) : Prop := C x y ∧ ∃ b, l x y b
def F (x y : ℝ) : Prop := C x y ∧ ∃ b, l x y b

-- Define point H on x-axis
def H (x : ℝ) : Prop := ∃ b, l x 0 b

-- Define the vector relationship
def vector_relationship (e_x e_y f_x f_y h_x k : ℝ) : Prop :=
  (h_x - e_x, -e_y) = k • (f_x - h_x, f_y)

-- Theorem statement
theorem parabola_line_intersection_theorem 
  (e_x e_y f_x f_y h_x : ℝ) (k : ℝ) :
  C e_x e_y → C f_x f_y →
  (∃ b, l e_x e_y b ∧ l f_x f_y b) →
  H h_x →
  vector_relationship e_x e_y f_x f_y h_x k →
  k > 1 →
  (f_x - e_x)^2 + (f_y - e_y)^2 = (4*Real.sqrt 13 / 3)^2 →
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_theorem_l2682_268299


namespace NUMINAMATH_CALUDE_complex_sum_equals_i_l2682_268208

theorem complex_sum_equals_i : Complex.I ^ 2 = -1 → (1 : ℂ) + Complex.I + Complex.I ^ 2 = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_i_l2682_268208


namespace NUMINAMATH_CALUDE_lottery_probabilities_l2682_268230

-- Define the lottery setup
def total_balls : ℕ := 10
def balls_with_2 : ℕ := 8
def balls_with_5 : ℕ := 2
def drawn_balls : ℕ := 3

-- Define the possible prize amounts
def prize_amounts : List ℕ := [6, 9, 12]

-- Define the corresponding probabilities
def probabilities : List ℚ := [7/15, 7/15, 1/15]

-- Theorem statement
theorem lottery_probabilities :
  let possible_outcomes := List.zip prize_amounts probabilities
  ∀ (outcome : ℕ × ℚ), outcome ∈ possible_outcomes →
    (∃ (n2 n5 : ℕ), n2 + n5 = drawn_balls ∧
      n2 * 2 + n5 * 5 = outcome.1 ∧
      (n2.choose balls_with_2 * n5.choose balls_with_5) / drawn_balls.choose total_balls = outcome.2) :=
by sorry

end NUMINAMATH_CALUDE_lottery_probabilities_l2682_268230


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2682_268258

/-- Represents the stratified sampling scenario -/
structure SamplingScenario where
  total_members : ℕ
  boys : ℕ
  girls : ℕ
  sample_size : ℕ

/-- The specific scenario from the problem -/
def track_team : SamplingScenario :=
  { total_members := 42
  , boys := 28
  , girls := 14
  , sample_size := 6 }

/-- The probability of an individual being selected -/
def selection_probability (s : SamplingScenario) : ℚ :=
  s.sample_size / s.total_members

/-- The number of boys selected in stratified sampling -/
def boys_selected (s : SamplingScenario) : ℕ :=
  (s.sample_size * s.boys) / s.total_members

/-- The number of girls selected in stratified sampling -/
def girls_selected (s : SamplingScenario) : ℕ :=
  (s.sample_size * s.girls) / s.total_members

theorem stratified_sampling_theorem (s : SamplingScenario) :
  s = track_team →
  selection_probability s = 1/7 ∧
  boys_selected s = 4 ∧
  girls_selected s = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2682_268258


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2682_268275

/-- The equation (x+5)(x+2) = k + 3x has exactly one real solution if and only if k = 6 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2682_268275


namespace NUMINAMATH_CALUDE_acid_mixture_water_volume_l2682_268222

/-- Represents the composition of a mixture --/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- Calculates the total volume of a mixture --/
def totalVolume (m : Mixture) : ℝ := m.acid + m.water

/-- Represents the problem setup --/
structure AcidMixtureProblem where
  initialMixture : Mixture
  pureAcidVolume : ℝ
  finalWaterPercentage : ℝ

/-- Calculates the final mixture composition --/
def finalMixture (problem : AcidMixtureProblem) (addedVolume : ℝ) : Mixture :=
  { acid := problem.pureAcidVolume + addedVolume * problem.initialMixture.acid,
    water := addedVolume * problem.initialMixture.water }

/-- The main theorem to prove --/
theorem acid_mixture_water_volume
  (problem : AcidMixtureProblem)
  (h1 : problem.initialMixture.acid = 0.1)
  (h2 : problem.initialMixture.water = 0.9)
  (h3 : problem.pureAcidVolume = 5)
  (h4 : problem.finalWaterPercentage = 0.4) :
  ∃ (addedVolume : ℝ),
    let finalMix := finalMixture problem addedVolume
    finalMix.water / totalVolume finalMix = problem.finalWaterPercentage ∧
    finalMix.water = 3.6 := by
  sorry


end NUMINAMATH_CALUDE_acid_mixture_water_volume_l2682_268222


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2682_268296

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y + f (x + y) = x * y

/-- The theorem stating that functions satisfying the equation are of the form x - 1 or x + 1. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    (∀ x, f x = x - 1) ∨ (∀ x, f x = x + 1) := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l2682_268296


namespace NUMINAMATH_CALUDE_one_tetrahedron_possible_l2682_268262

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the set of available triangles -/
def AvailableTriangles : Multiset Triangle := sorry

/-- The number of tetrahedra that can be formed from the available triangles -/
def NumTetrahedra : ℕ := sorry

/-- Theorem stating that only one tetrahedron can be formed -/
theorem one_tetrahedron_possible : NumTetrahedra = 1 := by sorry

end NUMINAMATH_CALUDE_one_tetrahedron_possible_l2682_268262


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2682_268285

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (Complex.I : ℂ) / (1 + 2 * Complex.I) → Complex.im z = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2682_268285


namespace NUMINAMATH_CALUDE_stamp_problem_l2682_268253

theorem stamp_problem (x y : ℕ) : 
  (x + y > 400) →
  (∃ k : ℕ, x - k = (13 : ℚ) / 19 * (y + k)) →
  (∃ k : ℕ, y - k = (11 : ℚ) / 17 * (x + k)) →
  x = 227 ∧ y = 221 :=
by sorry

end NUMINAMATH_CALUDE_stamp_problem_l2682_268253
