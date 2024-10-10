import Mathlib

namespace dance_workshop_avg_age_children_l2928_292873

theorem dance_workshop_avg_age_children (total_participants : ℕ) 
  (overall_avg_age : ℚ) (num_women : ℕ) (num_men : ℕ) (num_children : ℕ) 
  (avg_age_women : ℚ) (avg_age_men : ℚ) 
  (h1 : total_participants = 50)
  (h2 : overall_avg_age = 20)
  (h3 : num_women = 30)
  (h4 : num_men = 10)
  (h5 : num_children = 10)
  (h6 : avg_age_women = 22)
  (h7 : avg_age_men = 25)
  (h8 : total_participants = num_women + num_men + num_children) :
  (total_participants * overall_avg_age - num_women * avg_age_women - num_men * avg_age_men) / num_children = 9 := by
  sorry

end dance_workshop_avg_age_children_l2928_292873


namespace trisection_point_intersection_l2928_292824

noncomputable section

def f (x : ℝ) := Real.log x / Real.log 2

theorem trisection_point_intersection
  (x₁ x₂ : ℝ)
  (h_order : 0 < x₁ ∧ x₁ < x₂)
  (h_x₁ : x₁ = 4)
  (h_x₂ : x₂ = 16) :
  ∃ x₄ : ℝ, f x₄ = (2 * f x₁ + f x₂) / 3 ∧ x₄ = 2^(8/3) :=
sorry


end trisection_point_intersection_l2928_292824


namespace area_regular_octagon_in_circle_l2928_292840

/-- The area of a regular octagon inscribed in a circle -/
theorem area_regular_octagon_in_circle (r : ℝ) (h : r^2 * Real.pi = 256 * Real.pi) :
  8 * ((2 * r * Real.sin (Real.pi / 8))^2 * Real.sqrt 2 / 4) = 
    8 * (2 * 16 * Real.sin (Real.pi / 8))^2 * Real.sqrt 2 / 4 := by
  sorry

#check area_regular_octagon_in_circle

end area_regular_octagon_in_circle_l2928_292840


namespace hyperbola_eccentricity_l2928_292835

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  eq : (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1
  a_pos : a > 0
  b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point (x y : ℝ)

/-- The left focus of the hyperbola -/
def leftFocus (h : Hyperbola a b) : Point c 0 := sorry

/-- The right vertex of the hyperbola -/
def rightVertex (h : Hyperbola a b) : Point a 0 := sorry

/-- The upper endpoint of the imaginary axis -/
def upperImaginaryEndpoint (h : Hyperbola a b) : Point 0 b := sorry

/-- The point where AB intersects the asymptote -/
def intersectionPoint (h : Hyperbola a b) : Point (a/2) (b/2) := sorry

/-- FM bisects ∠BFA -/
def fmBisectsAngle (h : Hyperbola a b) : Prop := sorry

/-- Eccentricity of the hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Main theorem: The eccentricity of the hyperbola is 1 + √3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (bisect : fmBisectsAngle h) : eccentricity h = 1 + Real.sqrt 3 := by
  sorry

end hyperbola_eccentricity_l2928_292835


namespace f_increasing_on_interval_l2928_292850

noncomputable def f (x : ℝ) : ℝ := Real.exp (abs x) * Real.sin x

theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Ioo (-π/4) (3*π/4)) :=
sorry

end f_increasing_on_interval_l2928_292850


namespace sprint_medal_theorem_l2928_292843

/-- Represents the number of ways to award medals in a specific sprinting competition scenario. -/
def medalAwardingWays (totalSprinters : ℕ) (americanSprinters : ℕ) (canadianSprinters : ℕ) : ℕ :=
  -- The actual computation is not provided here
  sorry

/-- Theorem stating the number of ways to award medals in the given scenario. -/
theorem sprint_medal_theorem :
  medalAwardingWays 10 4 3 = 552 := by
  sorry

end sprint_medal_theorem_l2928_292843


namespace probability_multiple_3_or_4_in_30_l2928_292826

def is_multiple_of_3_or_4 (n : ℕ) : Bool :=
  n % 3 = 0 || n % 4 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_3_or_4 |>.length

theorem probability_multiple_3_or_4_in_30 :
  count_multiples 30 / 30 = 1 / 2 := by
  sorry

end probability_multiple_3_or_4_in_30_l2928_292826


namespace target_line_is_correct_l2928_292810

/-- The line we want to prove is correct -/
def target_line (x y : ℝ) : Prop := y = -3 * x - 2

/-- The line perpendicular to our target line -/
def perpendicular_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

/-- The curve to which our target line is tangent -/
def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 - 1

/-- Theorem stating that our target line is perpendicular to the given line
    and tangent to the given curve -/
theorem target_line_is_correct :
  (∀ x y : ℝ, perpendicular_line x y → 
    ∃ k : ℝ, k ≠ 0 ∧ (∀ x' y' : ℝ, target_line x' y' → 
      y' - y = k * (x' - x))) ∧ 
  (∃ x y : ℝ, target_line x y ∧ y = curve x ∧ 
    ∀ h : ℝ, h ≠ 0 → (curve (x + h) - curve x) / h ≠ -3) :=
sorry

end target_line_is_correct_l2928_292810


namespace taxi_fare_calculation_l2928_292858

/-- Taxi fare calculation -/
theorem taxi_fare_calculation 
  (base_distance : ℝ) 
  (rate_multiplier : ℝ) 
  (total_distance_1 : ℝ) 
  (total_fare_1 : ℝ) 
  (total_distance_2 : ℝ) 
  (h1 : base_distance = 60) 
  (h2 : rate_multiplier = 1.25) 
  (h3 : total_distance_1 = 80) 
  (h4 : total_fare_1 = 180) 
  (h5 : total_distance_2 = 100) :
  let base_rate := total_fare_1 / (base_distance + rate_multiplier * (total_distance_1 - base_distance))
  let total_fare_2 := base_rate * (base_distance + rate_multiplier * (total_distance_2 - base_distance))
  total_fare_2 = 3960 / 17 := by
sorry

end taxi_fare_calculation_l2928_292858


namespace hyperbola_equation_final_hyperbola_equation_l2928_292867

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (C₁ : ℝ → ℝ → Prop) (C₂ : ℝ → ℝ → Prop),
    (∀ x y, C₁ x y ↔ x^2 = 2*y) ∧ 
    (∀ x y, C₂ x y ↔ x^2/a^2 - y^2/b^2 = 1) ∧
    (∃ A : ℝ × ℝ, A.1 = a ∧ A.2 = 0 ∧ C₂ A.1 A.2) ∧
    (a^2 + b^2 = 5*a^2) ∧
    (∃ l : ℝ → ℝ, (∀ x, l x = b/a*(x - a)) ∧
      (∀ x, C₁ x (l x) → (∃! y, C₁ x y ∧ y = l x)))) →
  a = 1 ∧ b = 2 :=
by sorry

/-- The final form of the hyperbola equation -/
theorem final_hyperbola_equation :
  ∃ (C : ℝ → ℝ → Prop), ∀ x y, C x y ↔ x^2 - y^2/4 = 1 :=
by sorry

end hyperbola_equation_final_hyperbola_equation_l2928_292867


namespace quadratic_root_two_l2928_292864

theorem quadratic_root_two (c : ℝ) : (2 : ℝ)^2 = c → c = 4 := by
  sorry

end quadratic_root_two_l2928_292864


namespace divisibility_of_expression_l2928_292846

theorem divisibility_of_expression (p : ℕ) (h_prime : Nat.Prime p) (h_gt_two : p > 2) :
  ∃ k : ℤ, (⌊(2 + Real.sqrt 5)^p⌋ : ℤ) - 2^(p + 1) = k * p :=
sorry

end divisibility_of_expression_l2928_292846


namespace craig_dave_bench_press_ratio_l2928_292841

/-- Proves that Craig's bench press is 20% of Dave's bench press -/
theorem craig_dave_bench_press_ratio :
  let dave_weight : ℝ := 175
  let dave_bench_press : ℝ := 3 * dave_weight
  let mark_bench_press : ℝ := 55
  let craig_bench_press : ℝ := mark_bench_press + 50
  (craig_bench_press / dave_bench_press) * 100 = 20 := by
  sorry

end craig_dave_bench_press_ratio_l2928_292841


namespace average_rounds_is_three_l2928_292880

/-- Represents the number of golfers who played a certain number of rounds -/
def GolferDistribution := List (ℕ × ℕ)

/-- Calculates the total number of rounds played by all golfers -/
def totalRounds (dist : GolferDistribution) : ℕ :=
  dist.foldl (fun acc (rounds, golfers) => acc + rounds * golfers) 0

/-- Calculates the total number of golfers -/
def totalGolfers (dist : GolferDistribution) : ℕ :=
  dist.foldl (fun acc (_, golfers) => acc + golfers) 0

/-- Rounds a rational number to the nearest integer -/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem average_rounds_is_three (golfData : GolferDistribution) 
  (h : golfData = [(1, 4), (2, 3), (3, 6), (4, 2), (5, 4), (6, 1)]) : 
  roundToNearest (totalRounds golfData / totalGolfers golfData) = 3 := by
  sorry

end average_rounds_is_three_l2928_292880


namespace additional_chicken_wings_l2928_292818

theorem additional_chicken_wings (num_friends : ℕ) (initial_wings : ℕ) (wings_per_person : ℕ) : 
  num_friends = 4 → initial_wings = 9 → wings_per_person = 4 →
  num_friends * wings_per_person - initial_wings = 7 := by
  sorry

end additional_chicken_wings_l2928_292818


namespace power_multiplication_l2928_292857

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end power_multiplication_l2928_292857


namespace difference_of_squares_l2928_292809

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l2928_292809


namespace mean_squares_sum_l2928_292855

theorem mean_squares_sum (a b c : ℝ) : 
  (a + b + c) / 3 = 12 →
  (a * b * c) ^ (1/3 : ℝ) = 5 →
  3 / (1/a + 1/b + 1/c) = 4 →
  a^2 + b^2 + c^2 = 1108.5 := by
  sorry

end mean_squares_sum_l2928_292855


namespace prob_at_least_one_japanese_events_independent_iff_l2928_292862

-- Define the Little Green Lotus structure
structure LittleGreenLotus where
  isBoy : Bool
  speaksJapanese : Bool
  speaksKorean : Bool

-- Define the total number of Little Green Lotus
def totalLotus : ℕ := 36

-- Define the number of boys and girls
def numBoys : ℕ := 12
def numGirls : ℕ := 24

-- Define the number of boys and girls who can speak Japanese
def numBoysJapanese : ℕ := 8
def numGirlsJapanese : ℕ := 12

-- Define the number of boys and girls who can speak Korean as variables
variable (m n : ℕ)

-- Define the constraints on m
axiom m_bounds : 6 ≤ m ∧ m ≤ 8

-- Define the events A and B
def eventA (lotus : LittleGreenLotus) : Prop := lotus.isBoy
def eventB (lotus : LittleGreenLotus) : Prop := lotus.speaksKorean

-- Theorem 1: Probability of at least one of two randomly selected Little Green Lotus can speak Japanese
theorem prob_at_least_one_japanese :
  (totalLotus.choose 2 - (totalLotus - numBoysJapanese - numGirlsJapanese).choose 2) / totalLotus.choose 2 = 17 / 21 := by
  sorry

-- Theorem 2: Events A and B are independent if and only if n = 2m
theorem events_independent_iff (m n : ℕ) (h : 6 ≤ m ∧ m ≤ 8) :
  (numBoys * (m + n) = m * totalLotus) ↔ n = 2 * m := by
  sorry

end prob_at_least_one_japanese_events_independent_iff_l2928_292862


namespace cell_phone_production_ambiguity_l2928_292890

/-- Represents the production of cell phones in a factory --/
structure CellPhoneProduction where
  machines_count : ℕ
  phones_per_machine : ℕ
  total_production : ℕ

/-- The production scenario described in the problem --/
def factory_scenario : CellPhoneProduction :=
  { machines_count := 10
  , phones_per_machine := 5
  , total_production := 50 }

/-- The production rate for some machines described in the problem --/
def some_machines_rate : ℕ := 10

/-- Theorem stating the ambiguity in the production calculation --/
theorem cell_phone_production_ambiguity :
  (factory_scenario.machines_count * factory_scenario.phones_per_machine = factory_scenario.total_production) ∧
  (factory_scenario.phones_per_machine ≠ some_machines_rate) :=
by sorry

end cell_phone_production_ambiguity_l2928_292890


namespace stating_min_rows_for_150_cans_l2928_292829

/-- 
Represents the number of cans in a row given its position
-/
def cans_in_row (n : ℕ) : ℕ := 3 * n

/-- 
Calculates the total number of cans for a given number of rows
-/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

/-- 
Theorem stating that 10 is the minimum number of rows needed to have at least 150 cans
-/
theorem min_rows_for_150_cans : 
  (∀ k < 10, total_cans k < 150) ∧ total_cans 10 ≥ 150 := by
  sorry

end stating_min_rows_for_150_cans_l2928_292829


namespace lcm_from_hcf_and_product_l2928_292834

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 12 → a * b = 2460 → Nat.lcm a b = 205 := by
  sorry

end lcm_from_hcf_and_product_l2928_292834


namespace max_rectangles_is_k_times_l_l2928_292820

/-- A partition of a square into rectangles -/
structure SquarePartition where
  k : ℕ  -- number of rectangles intersected by a vertical line
  l : ℕ  -- number of rectangles intersected by a horizontal line
  no_interior_intersections : Bool  -- no two segments intersect at an interior point
  no_collinear_segments : Bool  -- no two segments lie on the same line

/-- The number of rectangles in a square partition -/
def num_rectangles (p : SquarePartition) : ℕ := sorry

/-- The maximum number of rectangles in any valid square partition -/
def max_rectangles (p : SquarePartition) : ℕ := p.k * p.l

/-- Theorem: The maximum number of rectangles in a valid square partition is k * l -/
theorem max_rectangles_is_k_times_l (p : SquarePartition) 
  (h1 : p.no_interior_intersections = true) 
  (h2 : p.no_collinear_segments = true) : 
  num_rectangles p ≤ max_rectangles p := by sorry

end max_rectangles_is_k_times_l_l2928_292820


namespace vector_2016_coordinates_l2928_292838

def matrix_transformation (x_n y_n : ℝ) : ℝ × ℝ :=
  (x_n, x_n + y_n)

def vector_sequence (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => (2, 0)
  | n + 1 => matrix_transformation (vector_sequence n).1 (vector_sequence n).2

theorem vector_2016_coordinates :
  vector_sequence 2015 = (2, 4030) := by
  sorry

end vector_2016_coordinates_l2928_292838


namespace ascending_order_l2928_292865

theorem ascending_order (a b c : ℝ) 
  (ha : a = Real.rpow 0.8 0.7)
  (hb : b = Real.rpow 0.8 0.9)
  (hc : c = Real.rpow 1.2 0.8) :
  b < a ∧ a < c := by
  sorry

end ascending_order_l2928_292865


namespace pure_imaginary_condition_l2928_292823

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (Complex.I * b : ℂ) = (1 + a * Complex.I) / (1 - Complex.I)) → a = 1 := by
  sorry

end pure_imaginary_condition_l2928_292823


namespace bridge_length_l2928_292856

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 145 →
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  (train_speed * crossing_time) - train_length = 230 :=
by sorry

end bridge_length_l2928_292856


namespace square_field_side_length_l2928_292817

theorem square_field_side_length (area : ℝ) (side : ℝ) :
  area = 225 →
  side * side = area →
  side = 15 := by
  sorry

end square_field_side_length_l2928_292817


namespace checkerboard_area_equality_l2928_292845

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Condition for convexity

-- Define the division points on the sides
def division_points (q : ConvexQuadrilateral) : Fin 4 → Fin 8 → ℝ × ℝ :=
  sorry -- Function that returns the division points on each side

-- Define the cells formed by connecting corresponding division points
def cells (q : ConvexQuadrilateral) : List (List (ℝ × ℝ)) :=
  sorry -- List of cells, each cell represented by its vertices

-- Define the area of a cell
def cell_area (cell : List (ℝ × ℝ)) : ℝ :=
  sorry -- Function to calculate the area of a cell

-- Define the sum of areas of alternating cells (checkerboard pattern)
def alternating_sum (cells : List (List (ℝ × ℝ))) : ℝ :=
  sorry -- Sum of areas of alternating cells

-- The theorem to be proved
theorem checkerboard_area_equality (q : ConvexQuadrilateral) :
  let c := cells q
  alternating_sum c = alternating_sum (List.drop 1 c) :=
sorry

end checkerboard_area_equality_l2928_292845


namespace angle_sum_in_circle_l2928_292807

theorem angle_sum_in_circle (x : ℝ) : 
  (6*x + 7*x + 3*x + 2*x + 4*x = 360) → x = 180/11 := by
  sorry

end angle_sum_in_circle_l2928_292807


namespace fraction_transformation_l2928_292868

theorem fraction_transformation (p q r s x y : ℝ) 
  (h1 : p ≠ q) 
  (h2 : q ≠ 0) 
  (h3 : y ≠ 0) 
  (h4 : s ≠ y * r) 
  (h5 : (p + x) / (q + y * x) = r / s) : 
  x = (q * r - p * s) / (s - y * r) := by
sorry

end fraction_transformation_l2928_292868


namespace sin_300_degrees_l2928_292808

theorem sin_300_degrees : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l2928_292808


namespace soap_brand_survey_l2928_292816

theorem soap_brand_survey (total : ℕ) (neither : ℕ) (only_a : ℕ) (both_to_only_b_ratio : ℕ) 
  (h1 : total = 180)
  (h2 : neither = 80)
  (h3 : only_a = 60)
  (h4 : both_to_only_b_ratio = 3) :
  ∃ (both : ℕ), 
    neither + only_a + both + both_to_only_b_ratio * both = total ∧ 
    both = 10 := by
  sorry

end soap_brand_survey_l2928_292816


namespace total_owed_after_borrowing_l2928_292815

/-- The total amount owed when borrowing additional money -/
theorem total_owed_after_borrowing (initial_debt additional_borrowed : ℕ) :
  initial_debt = 20 →
  additional_borrowed = 8 →
  initial_debt + additional_borrowed = 28 := by
  sorry

end total_owed_after_borrowing_l2928_292815


namespace negation_of_universal_proposition_l2928_292819

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l2928_292819


namespace plane_perpendicularity_condition_l2928_292899

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity_condition 
  (α β : Plane) (l : Line) 
  (h1 : α ≠ β) 
  (h2 : subset l α) :
  (∀ l, subset l α → perpendicular l β → plane_perpendicular α β) ∧ 
  (∃ l, subset l α ∧ plane_perpendicular α β ∧ ¬perpendicular l β) :=
sorry

end plane_perpendicularity_condition_l2928_292899


namespace siblings_age_ratio_l2928_292876

theorem siblings_age_ratio : 
  ∀ (aaron_age henry_age sister_age : ℕ),
  aaron_age = 15 →
  sister_age = 3 * aaron_age →
  aaron_age + henry_age + sister_age = 240 →
  henry_age / sister_age = 4 := by
sorry

end siblings_age_ratio_l2928_292876


namespace sqrt_product_equality_l2928_292882

theorem sqrt_product_equality (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (16 * x) * Real.sqrt (5 * x) * Real.sqrt (6 * x) * Real.sqrt (30 * x) = 30) : 
  x = 1 / 2 := by
sorry

end sqrt_product_equality_l2928_292882


namespace cookie_cost_cookie_cost_is_65_l2928_292839

/-- The cost of a package of cookies, given the amount Diane has and the additional amount she needs. -/
theorem cookie_cost (diane_has : ℕ) (diane_needs : ℕ) : ℕ :=
  diane_has + diane_needs

/-- Proof that the cost of the cookies is 65 cents. -/
theorem cookie_cost_is_65 : cookie_cost 27 38 = 65 := by
  sorry

end cookie_cost_cookie_cost_is_65_l2928_292839


namespace apartments_on_more_floors_proof_l2928_292804

/-- Represents the number of apartments on a floor with more apartments -/
def apartments_on_more_floors : ℕ := 6

/-- Represents the total number of floors in the building -/
def total_floors : ℕ := 12

/-- Represents the number of apartments on floors with fewer apartments -/
def apartments_on_fewer_floors : ℕ := 5

/-- Represents the maximum number of residents per apartment -/
def max_residents_per_apartment : ℕ := 4

/-- Represents the maximum total number of residents in the building -/
def max_total_residents : ℕ := 264

theorem apartments_on_more_floors_proof :
  let floors_with_more := total_floors / 2
  let floors_with_fewer := total_floors / 2
  let total_apartments_fewer := floors_with_fewer * apartments_on_fewer_floors
  let total_apartments := max_total_residents / max_residents_per_apartment
  let apartments_on_more_total := total_apartments - total_apartments_fewer
  apartments_on_more_floors = apartments_on_more_total / floors_with_more :=
by
  sorry

#check apartments_on_more_floors_proof

end apartments_on_more_floors_proof_l2928_292804


namespace largest_number_l2928_292852

theorem largest_number (a b c d e f : ℝ) 
  (ha : a = 0.986) 
  (hb : b = 0.9859) 
  (hc : c = 0.98609) 
  (hd : d = 0.896) 
  (he : e = 0.8979) 
  (hf : f = 0.987) : 
  f = max a (max b (max c (max d (max e f)))) :=
sorry

end largest_number_l2928_292852


namespace quadratic_vertex_form_l2928_292814

theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a h k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3/2 := by
  sorry

end quadratic_vertex_form_l2928_292814


namespace area_of_grid_with_cutouts_l2928_292832

/-- The area of a square grid with triangular cutouts -/
theorem area_of_grid_with_cutouts (grid_side : ℕ) (cell_side : ℝ) 
  (dark_grey_area : ℝ) (light_grey_area : ℝ) : 
  grid_side = 6 → 
  cell_side = 1 → 
  dark_grey_area = 3 → 
  light_grey_area = 6 → 
  (grid_side : ℝ) * (grid_side : ℝ) * cell_side * cell_side - dark_grey_area - light_grey_area = 27 := by
sorry

end area_of_grid_with_cutouts_l2928_292832


namespace coterminal_angle_correct_l2928_292860

/-- The angle in degrees that is coterminal with 1000° and lies between 0° and 360° -/
def coterminal_angle : ℝ := 280

/-- Proof that the coterminal angle is correct -/
theorem coterminal_angle_correct :
  0 ≤ coterminal_angle ∧ 
  coterminal_angle < 360 ∧
  ∃ (k : ℤ), coterminal_angle = 1000 - 360 * k :=
by sorry

end coterminal_angle_correct_l2928_292860


namespace bob_water_usage_percentage_l2928_292859

-- Define the farmers
inductive Farmer
| Bob
| Brenda
| Bernie

-- Define the crop types
inductive Crop
| Corn
| Cotton
| Beans

-- Define the acreage for each farmer and crop
def acreage : Farmer → Crop → ℕ
  | Farmer.Bob, Crop.Corn => 3
  | Farmer.Bob, Crop.Cotton => 9
  | Farmer.Bob, Crop.Beans => 12
  | Farmer.Brenda, Crop.Corn => 6
  | Farmer.Brenda, Crop.Cotton => 7
  | Farmer.Brenda, Crop.Beans => 14
  | Farmer.Bernie, Crop.Corn => 2
  | Farmer.Bernie, Crop.Cotton => 12
  | Farmer.Bernie, Crop.Beans => 0

-- Define water requirements for each crop (in gallons per acre)
def waterPerAcre : Crop → ℕ
  | Crop.Corn => 20
  | Crop.Cotton => 80
  | Crop.Beans => 40  -- Twice as much as corn

-- Calculate total water used by a farmer
def farmerWaterUsage (f : Farmer) : ℕ :=
  (acreage f Crop.Corn * waterPerAcre Crop.Corn) +
  (acreage f Crop.Cotton * waterPerAcre Crop.Cotton) +
  (acreage f Crop.Beans * waterPerAcre Crop.Beans)

-- Calculate total water used by all farmers
def totalWaterUsage : ℕ :=
  farmerWaterUsage Farmer.Bob +
  farmerWaterUsage Farmer.Brenda +
  farmerWaterUsage Farmer.Bernie

-- Theorem: Bob's water usage is 36% of total water usage
theorem bob_water_usage_percentage :
  (farmerWaterUsage Farmer.Bob : ℚ) / totalWaterUsage * 100 = 36 := by
  sorry

end bob_water_usage_percentage_l2928_292859


namespace sum_of_digits_of_factorials_of_fib_l2928_292869

-- Define the first 10 Fibonacci numbers
def fib : List Nat := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

-- Function to calculate factorial
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Function to sum the digits of a number
def sumDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumDigits (n / 10)

-- Theorem statement
theorem sum_of_digits_of_factorials_of_fib : 
  (fib.map (λ x => sumDigits (factorial x))).sum = 240 := by
  sorry


end sum_of_digits_of_factorials_of_fib_l2928_292869


namespace largest_multiple_12_negation_gt_neg_150_l2928_292836

theorem largest_multiple_12_negation_gt_neg_150 :
  ∀ n : ℤ, n ≥ 0 → 12 ∣ n → -n > -150 → n ≤ 144 :=
by
  sorry

end largest_multiple_12_negation_gt_neg_150_l2928_292836


namespace sum_difference_even_odd_100_l2928_292812

/-- Sum of first n positive even integers -/
def sumEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of first n positive odd integers -/
def sumOddIntegers (n : ℕ) : ℕ := n^2

theorem sum_difference_even_odd_100 :
  sumEvenIntegers 100 - sumOddIntegers 100 = 100 := by
  sorry

end sum_difference_even_odd_100_l2928_292812


namespace rectangular_cross_section_shapes_l2928_292806

/-- Enumeration of the geometric shapes in question -/
inductive GeometricShape
  | RectangularPrism
  | Cylinder
  | Cone
  | Cube

/-- Predicate to determine if a shape can have a rectangular cross-section -/
def has_rectangular_cross_section (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.RectangularPrism => true
  | GeometricShape.Cylinder => true
  | GeometricShape.Cone => false
  | GeometricShape.Cube => true

/-- The set of shapes that can have a rectangular cross-section -/
def shapes_with_rectangular_cross_section : Set GeometricShape :=
  {shape | has_rectangular_cross_section shape}

/-- Theorem stating which shapes can have a rectangular cross-section -/
theorem rectangular_cross_section_shapes :
  shapes_with_rectangular_cross_section =
    {GeometricShape.RectangularPrism, GeometricShape.Cylinder, GeometricShape.Cube} :=
by sorry


end rectangular_cross_section_shapes_l2928_292806


namespace investment_rate_calculation_l2928_292894

theorem investment_rate_calculation 
  (total_investment : ℝ) 
  (first_rate : ℝ) 
  (first_amount : ℝ) 
  (total_interest : ℝ) :
  total_investment = 10000 →
  first_rate = 0.06 →
  first_amount = 7200 →
  total_interest = 684 →
  let second_amount := total_investment - first_amount
  let first_interest := first_amount * first_rate
  let second_interest := total_interest - first_interest
  let second_rate := second_interest / second_amount
  second_rate = 0.09 := by sorry

end investment_rate_calculation_l2928_292894


namespace symmetry_about_y_equals_x_l2928_292866

/-- The set of points (x, y) satisfying the given conditions is symmetric about y = x -/
theorem symmetry_about_y_equals_x (r : ℝ) :
  ∀ (x y : ℝ), x^2 + y^2 ≤ r^2 ∧ x + y > 0 →
  ∃ (x' y' : ℝ), x'^2 + y'^2 ≤ r^2 ∧ x' + y' > 0 ∧ x' = y ∧ y' = x :=
by sorry

end symmetry_about_y_equals_x_l2928_292866


namespace monica_students_l2928_292888

/-- Represents the number of students in each class and the overlaps between classes -/
structure ClassData where
  class1 : ℕ
  class2 : ℕ
  class3 : ℕ
  class4 : ℕ
  class5 : ℕ
  class6 : ℕ
  overlap12 : ℕ
  overlap45 : ℕ
  overlap236 : ℕ
  overlap56 : ℕ

/-- Calculates the number of individual students Monica sees each day -/
def individualStudents (data : ClassData) : ℕ :=
  data.class1 + data.class2 + data.class3 + data.class4 + data.class5 + data.class6 -
  (data.overlap12 + data.overlap45 + data.overlap236 + data.overlap56)

/-- Theorem stating that Monica sees 114 individual students each day -/
theorem monica_students :
  ∀ (data : ClassData),
    data.class1 = 20 ∧
    data.class2 = 25 ∧
    data.class3 = 25 ∧
    data.class4 = 10 ∧
    data.class5 = 28 ∧
    data.class6 = 28 ∧
    data.overlap12 = 5 ∧
    data.overlap45 = 3 ∧
    data.overlap236 = 6 ∧
    data.overlap56 = 8 →
    individualStudents data = 114 :=
by
  sorry


end monica_students_l2928_292888


namespace fraction_power_product_l2928_292884

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end fraction_power_product_l2928_292884


namespace base8_addition_example_l2928_292898

/-- Addition in base 8 -/
def base8_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 8 -/
def base10_to_base8 (n : ℕ) : ℕ := sorry

theorem base8_addition_example : 
  base8_add (base10_to_base8 83) (base10_to_base8 46) = base10_to_base8 130 := by sorry

end base8_addition_example_l2928_292898


namespace sqrt_meaningful_range_l2928_292847

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 := by
  sorry

end sqrt_meaningful_range_l2928_292847


namespace discriminant_nonnegativity_l2928_292801

theorem discriminant_nonnegativity (x : ℤ) :
  x^2 * (81 - 56 * x^2) ≥ 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  sorry

end discriminant_nonnegativity_l2928_292801


namespace top_square_after_folds_l2928_292833

/-- Represents a 6x6 grid of numbers -/
def Grid := Fin 6 → Fin 6 → Nat

/-- Initial grid configuration -/
def initial_grid : Grid :=
  fun i j => 6 * i.val + j.val + 1

/-- Fold operation types -/
inductive FoldType
  | TopOver
  | BottomOver
  | RightOver
  | LeftOver

/-- Apply a single fold operation to the grid -/
def apply_fold (g : Grid) (ft : FoldType) : Grid :=
  sorry  -- Implementation of folding logic

/-- Sequence of folds as described in the problem -/
def fold_sequence : List FoldType :=
  [FoldType.TopOver, FoldType.BottomOver, FoldType.RightOver, 
   FoldType.LeftOver, FoldType.TopOver, FoldType.RightOver]

/-- Apply a sequence of folds to the grid -/
def apply_fold_sequence (g : Grid) (folds : List FoldType) : Grid :=
  sorry  -- Implementation of applying multiple folds

theorem top_square_after_folds (g : Grid) :
  g = initial_grid →
  (apply_fold_sequence g fold_sequence) 0 0 = 22 :=
sorry

end top_square_after_folds_l2928_292833


namespace perimeter_ratio_of_similar_triangles_l2928_292821

/-- Two triangles are similar with a given ratio -/
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) (r : ℝ) : Prop := sorry

/-- The perimeter of a triangle -/
def perimeter (t : Set (ℝ × ℝ)) : ℝ := sorry

theorem perimeter_ratio_of_similar_triangles 
  (abc a1b1c1 : Set (ℝ × ℝ)) : 
  similar_triangles abc a1b1c1 (1/2) → 
  perimeter abc / perimeter a1b1c1 = 1/2 := by
  sorry

end perimeter_ratio_of_similar_triangles_l2928_292821


namespace pure_imaginary_complex_fraction_l2928_292879

theorem pure_imaginary_complex_fraction (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
sorry

end pure_imaginary_complex_fraction_l2928_292879


namespace at_least_one_triangle_inside_l2928_292837

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a pentagon -/
structure Pentagon :=
  (vertices : Fin 5 → Point)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (vertices : Fin 3 → Point)

/-- Checks if a pentagon is convex and equilateral -/
def isConvexEquilateralPentagon (p : Pentagon) : Prop :=
  sorry

/-- Constructs equilateral triangles on the sides of a pentagon -/
def constructTriangles (p : Pentagon) : Fin 5 → EquilateralTriangle :=
  sorry

/-- Checks if a triangle is entirely contained within a pentagon -/
def isTriangleContained (t : EquilateralTriangle) (p : Pentagon) : Prop :=
  sorry

/-- The main theorem -/
theorem at_least_one_triangle_inside (p : Pentagon) 
  (h : isConvexEquilateralPentagon p) :
  ∃ (i : Fin 5), isTriangleContained (constructTriangles p i) p :=
sorry

end at_least_one_triangle_inside_l2928_292837


namespace minimum_jumps_circle_l2928_292889

/-- Represents a jump on the circle of points -/
inductive Jump
| Two  : Jump  -- Jump of 2 points
| Three : Jump  -- Jump of 3 points

/-- Represents a sequence of jumps -/
def JumpSequence := List Jump

/-- Function to check if a sequence of jumps visits all points and returns to start -/
def validSequence (n : Nat) (seq : JumpSequence) : Prop :=
  -- Implementation details omitted
  sorry

theorem minimum_jumps_circle :
  ∀ (seq : JumpSequence),
    validSequence 2016 seq →
    seq.length ≥ 2017 :=
by sorry

end minimum_jumps_circle_l2928_292889


namespace three_digit_sum_product_l2928_292861

theorem three_digit_sum_product (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) :
  let y : ℕ := 9
  let z : ℕ := 9
  100 * x + 10 * y + z = x + y + z + x * y + y * z + z * x + x * y * z :=
by sorry

end three_digit_sum_product_l2928_292861


namespace bride_groom_age_difference_oldest_bride_problem_l2928_292891

theorem bride_groom_age_difference : ℕ → ℕ → ℕ → Prop :=
  fun total_age groom_age age_difference =>
    let bride_age := total_age - groom_age
    bride_age - groom_age = age_difference

theorem oldest_bride_problem (total_age groom_age : ℕ) 
  (h1 : total_age = 185) 
  (h2 : groom_age = 83) : 
  bride_groom_age_difference total_age groom_age 19 := by
  sorry

end bride_groom_age_difference_oldest_bride_problem_l2928_292891


namespace average_weight_B_and_C_l2928_292871

theorem average_weight_B_and_C (A B C : ℝ) : 
  (A + B + C) / 3 = 45 →
  (A + B) / 2 = 40 →
  B = 31 →
  (B + C) / 2 = 43 := by
sorry

end average_weight_B_and_C_l2928_292871


namespace school_students_count_prove_school_students_count_l2928_292870

theorem school_students_count : ℕ → Prop :=
  fun total_students =>
    let chess_students := (total_students : ℚ) * (1 / 10)
    let swimming_students := chess_students * (1 / 2)
    swimming_students = 100 →
    total_students = 2000

-- The proof is omitted
theorem prove_school_students_count :
  ∃ (n : ℕ), school_students_count n :=
sorry

end school_students_count_prove_school_students_count_l2928_292870


namespace shaded_area_is_24_l2928_292887

structure Rectangle where
  width : ℝ
  height : ℝ

structure Triangle where
  base : ℝ
  height : ℝ

def shaded_area (rect : Rectangle) (tri : Triangle) : ℝ :=
  sorry

theorem shaded_area_is_24 (rect : Rectangle) (tri : Triangle) :
  rect.width = 8 ∧ rect.height = 12 ∧ tri.base = 8 ∧ tri.height = rect.height →
  shaded_area rect tri = 24 := by
  sorry

end shaded_area_is_24_l2928_292887


namespace union_of_A_and_B_l2928_292805

def A : Set ℕ := {0, 1, 2}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2^a}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 4} := by
  sorry

end union_of_A_and_B_l2928_292805


namespace perfect_square_quadratic_l2928_292827

theorem perfect_square_quadratic (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - a*x + 16 = (x - b)^2) → (a = 8 ∨ a = -8) := by
  sorry

end perfect_square_quadratic_l2928_292827


namespace curve_is_parabola_l2928_292886

theorem curve_is_parabola (θ : Real) (r : Real → Real) (x y : Real) :
  (r θ = 1 / (1 - Real.sin θ)) →
  (x^2 + y^2 = r θ^2) →
  (y = r θ * Real.sin θ) →
  (x^2 = 2*y + 1) :=
by
  sorry

#check curve_is_parabola

end curve_is_parabola_l2928_292886


namespace matrix_determinant_from_eigenvectors_l2928_292853

/-- Given a 2x2 matrix A with specific eigenvectors and eigenvalues, prove that its determinant is -4 -/
theorem matrix_determinant_from_eigenvectors (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A.mulVec ![1, -1] = (-1 : ℝ) • ![1, -1]) → 
  (A.mulVec ![3, 2] = 4 • ![3, 2]) → 
  a * d - b * c = -4 := by
  sorry


end matrix_determinant_from_eigenvectors_l2928_292853


namespace odd_iff_a_eq_zero_l2928_292875

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  3 * Real.log (x + Real.sqrt (x^2 + 1)) + a * (7^x + 7^(-x))

def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_iff_a_eq_zero (a : ℝ) :
  isOdd (f a) ↔ a = 0 :=
sorry

end odd_iff_a_eq_zero_l2928_292875


namespace nested_f_evaluation_l2928_292892

/-- The function f(x) = x^2 - 3x + 1 -/
def f (x : ℤ) : ℤ := x^2 - 3*x + 1

/-- Theorem stating that f(f(f(f(f(f(-1)))))) = 3432163846882600 -/
theorem nested_f_evaluation : f (f (f (f (f (f (-1)))))) = 3432163846882600 := by
  sorry

end nested_f_evaluation_l2928_292892


namespace necklaces_given_to_friends_l2928_292885

theorem necklaces_given_to_friends (initial : ℕ) (sold : ℕ) (remaining : ℕ) :
  initial = 60 →
  sold = 16 →
  remaining = 26 →
  initial - sold - remaining = 18 :=
by
  sorry

end necklaces_given_to_friends_l2928_292885


namespace rectangular_plot_width_l2928_292825

theorem rectangular_plot_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 432 →
  width = 12 := by
sorry

end rectangular_plot_width_l2928_292825


namespace functional_equation_solution_l2928_292854

/-- Given a function g: ℝ → ℝ satisfying the functional equation
    (g x * g y - g (x * y)) / 4 = x + y + 3 for all x, y ∈ ℝ,
    prove that g x = x + 4 for all x ∈ ℝ. -/
theorem functional_equation_solution (g : ℝ → ℝ)
    (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 3) :
  ∀ x : ℝ, g x = x + 4 := by
  sorry

end functional_equation_solution_l2928_292854


namespace sum_of_interior_angles_octagon_l2928_292842

theorem sum_of_interior_angles_octagon (a : ℝ) : a = 1080 := by
  sorry

end sum_of_interior_angles_octagon_l2928_292842


namespace existsNonSymmetricalEqualTriangles_l2928_292863

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents an ellipse -/
structure Ellipse :=
  (center : Point)
  (semiMajorAxis : ℝ)
  (semiMinorAxis : ℝ)

/-- Checks if a point is inside or on the ellipse -/
def isPointInEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.semiMajorAxis^2 + (p.y - e.center.y)^2 / e.semiMinorAxis^2 ≤ 1

/-- Checks if a triangle is inscribed in an ellipse -/
def isTriangleInscribed (t : Triangle) (e : Ellipse) : Prop :=
  isPointInEllipse t.a e ∧ isPointInEllipse t.b e ∧ isPointInEllipse t.c e

/-- Checks if two triangles are equal -/
def areTrianglesEqual (t1 t2 : Triangle) : Prop :=
  -- Definition of triangle equality (e.g., same side lengths)
  sorry

/-- Checks if two triangles are symmetrical with respect to the x-axis -/
def areTrianglesSymmetricalXAxis (t1 t2 : Triangle) : Prop :=
  -- Definition of symmetry with respect to x-axis
  sorry

/-- Checks if two triangles are symmetrical with respect to the y-axis -/
def areTrianglesSymmetricalYAxis (t1 t2 : Triangle) : Prop :=
  -- Definition of symmetry with respect to y-axis
  sorry

/-- Checks if two triangles are symmetrical with respect to the center -/
def areTrianglesSymmetricalCenter (t1 t2 : Triangle) (e : Ellipse) : Prop :=
  -- Definition of symmetry with respect to center
  sorry

/-- Main theorem: There exist two equal triangles inscribed in an ellipse that are not symmetrical -/
theorem existsNonSymmetricalEqualTriangles :
  ∃ (e : Ellipse) (t1 t2 : Triangle),
    isTriangleInscribed t1 e ∧
    isTriangleInscribed t2 e ∧
    areTrianglesEqual t1 t2 ∧
    ¬(areTrianglesSymmetricalXAxis t1 t2 ∨
      areTrianglesSymmetricalYAxis t1 t2 ∨
      areTrianglesSymmetricalCenter t1 t2 e) :=
by
  sorry

end existsNonSymmetricalEqualTriangles_l2928_292863


namespace min_moves_correct_l2928_292883

/-- The minimum number of moves in Bethan's grid game -/
def min_moves (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 2 + n
  else
    (n^2 + 1) / 2

/-- Theorem stating the minimum number of moves in Bethan's grid game -/
theorem min_moves_correct (n : ℕ) (h : n > 0) :
  min_moves n = if n % 2 = 0 then n^2 / 2 + n else (n^2 + 1) / 2 :=
by sorry

end min_moves_correct_l2928_292883


namespace OPRQ_shapes_l2928_292893

-- Define the points
def O : ℝ × ℝ := (0, 0)
def P (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, y₁)
def Q (x₂ y₂ : ℝ) : ℝ × ℝ := (x₂, y₂)
def R (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := (x₁ - x₂, y₁ - y₂)

-- Define the quadrilateral OPRQ
def OPRQ (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {O, P x₁ y₁, Q x₂ y₂, R x₁ y₁ x₂ y₂}

-- Define conditions for parallelogram, straight line, and trapezoid
def isParallelogram (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  P x₁ y₁ + Q x₂ y₂ = R x₁ y₁ x₂ y₂

def isStraightLine (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * y₂ = x₂ * y₁

def isTrapezoid (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (k : ℝ), x₂ = k * (x₁ - x₂) ∧ y₂ = k * (y₁ - y₂)

-- Theorem statement
theorem OPRQ_shapes (x₁ y₁ x₂ y₂ : ℝ) (h : P x₁ y₁ ≠ Q x₂ y₂) :
  (isParallelogram x₁ y₁ x₂ y₂) ∧
  (isStraightLine x₁ y₁ x₂ y₂ → OPRQ x₁ y₁ x₂ y₂ = {O, P x₁ y₁, Q x₂ y₂, R x₁ y₁ x₂ y₂}) ∧
  (∃ x₁' y₁' x₂' y₂', isTrapezoid x₁' y₁' x₂' y₂') :=
sorry

end OPRQ_shapes_l2928_292893


namespace hens_count_l2928_292811

/-- Represents the number of hens and cows in a farm -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of heads in the farm -/
def totalHeads (f : Farm) : ℕ := f.hens + f.cows

/-- The total number of feet in the farm -/
def totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- A farm satisfying the given conditions -/
def satisfiesConditions (f : Farm) : Prop :=
  totalHeads f = 50 ∧ totalFeet f = 144

theorem hens_count (f : Farm) (h : satisfiesConditions f) : f.hens = 28 := by
  sorry

end hens_count_l2928_292811


namespace root_intersection_l2928_292896

-- Define the original equation
def original_equation (x : ℝ) : Prop := x^2 - 2*x = 0

-- Define the roots of the original equation
def is_root (x : ℝ) : Prop := original_equation x

-- Define the pairs of equations
def pair_A (x y : ℝ) : Prop := (y = x^2 ∧ y = 2*x)
def pair_B (x y : ℝ) : Prop := (y = x^2 - 2*x ∧ y = 0)
def pair_C (x y : ℝ) : Prop := (y = x ∧ y = x - 2)
def pair_D (x y : ℝ) : Prop := (y = x^2 - 2*x + 1 ∧ y = 1)
def pair_E (x y : ℝ) : Prop := (y = x^2 - 1 ∧ y = 2*x - 1)

-- Theorem stating that pair C does not yield the roots while others do
theorem root_intersection :
  (∃ x y : ℝ, pair_C x y ∧ is_root x) = false ∧
  (∃ x y : ℝ, pair_A x y ∧ is_root x) = true ∧
  (∃ x y : ℝ, pair_B x y ∧ is_root x) = true ∧
  (∃ x y : ℝ, pair_D x y ∧ is_root x) = true ∧
  (∃ x y : ℝ, pair_E x y ∧ is_root x) = true :=
by sorry

end root_intersection_l2928_292896


namespace wednesday_fraction_is_one_fourth_l2928_292802

/-- Represents the daily fabric delivery and earnings for a textile company. -/
structure TextileDelivery where
  monday_yards : ℕ
  tuesday_multiplier : ℕ
  fabric_cost : ℕ
  total_earnings : ℕ

/-- Calculates the fraction of fabric delivered on Wednesday compared to Tuesday. -/
def wednesday_fraction (d : TextileDelivery) : ℚ :=
  let monday_earnings := d.monday_yards * d.fabric_cost
  let tuesday_yards := d.monday_yards * d.tuesday_multiplier
  let tuesday_earnings := tuesday_yards * d.fabric_cost
  let wednesday_earnings := d.total_earnings - monday_earnings - tuesday_earnings
  let wednesday_yards := wednesday_earnings / d.fabric_cost
  wednesday_yards / tuesday_yards

/-- Theorem stating that the fraction of fabric delivered on Wednesday compared to Tuesday is 1/4. -/
theorem wednesday_fraction_is_one_fourth (d : TextileDelivery) 
    (h1 : d.monday_yards = 20)
    (h2 : d.tuesday_multiplier = 2)
    (h3 : d.fabric_cost = 2)
    (h4 : d.total_earnings = 140) : 
  wednesday_fraction d = 1/4 := by
  sorry

end wednesday_fraction_is_one_fourth_l2928_292802


namespace consecutive_integers_count_l2928_292830

def list_K : List ℤ := sorry

theorem consecutive_integers_count :
  (list_K.head? = some (-3)) ∧ 
  (∀ i j, i ∈ list_K → j ∈ list_K → i < j → ∀ k, i < k ∧ k < j → k ∈ list_K) ∧
  (∃ max_pos ∈ list_K, max_pos > 0 ∧ ∀ x ∈ list_K, x > 0 → x ≤ max_pos) ∧
  (∃ min_pos ∈ list_K, min_pos > 0 ∧ ∀ x ∈ list_K, x > 0 → x ≥ min_pos) ∧
  (∃ max_pos min_pos, max_pos ∈ list_K ∧ min_pos ∈ list_K ∧ 
    max_pos > 0 ∧ min_pos > 0 ∧ max_pos - min_pos = 4) →
  list_K.length = 9 := by
sorry

end consecutive_integers_count_l2928_292830


namespace sine_function_period_l2928_292831

theorem sine_function_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x ∈ Set.Icc (-π) (5*π), ∃ y, y = a * Real.sin (b * x + c) + d) →
  (∃ n : ℕ, n = 5 ∧ (6*π) / n = (2*π) / b) →
  b = 5/3 := by
sorry

end sine_function_period_l2928_292831


namespace removed_triangles_area_l2928_292803

/-- Given a square with side length x and isosceles right triangles removed from each corner to form a rectangle with perimeter 32, the combined area of the four removed triangles is x²/2. -/
theorem removed_triangles_area (x : ℝ) (r s : ℝ) : 
  x > 0 → 
  2 * (r + s) + 2 * |r - s| = 32 → 
  (r + s)^2 + (r - s)^2 = x^2 → 
  2 * r * s = x^2 / 2 :=
by sorry

#check removed_triangles_area

end removed_triangles_area_l2928_292803


namespace garment_fraction_l2928_292895

theorem garment_fraction (bikini_fraction trunks_fraction : ℝ) 
  (h1 : bikini_fraction = 0.38) 
  (h2 : trunks_fraction = 0.25) : 
  bikini_fraction + trunks_fraction = 0.63 := by
  sorry

end garment_fraction_l2928_292895


namespace cannot_determine_package_size_l2928_292874

/-- Represents the number of candies in a package -/
def CandiesPerPackage := ℕ

/-- Represents the state of candies on the desk -/
structure CandyPile :=
  (initial : ℕ)
  (added : ℕ)
  (final : ℕ)

/-- Given a candy pile state, it's not possible to determine the number of candies per package -/
theorem cannot_determine_package_size (pile : CandyPile) : 
  pile.initial = 6 → pile.added = 4 → pile.final = 10 → 
  ¬∃ (package_size : CandiesPerPackage), ∀ (other_size : CandiesPerPackage), package_size = other_size :=
by sorry

end cannot_determine_package_size_l2928_292874


namespace exists_alpha_for_sequence_l2928_292849

/-- A sequence of non-zero real numbers satisfying the given condition -/
def SequenceA (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n ≠ 0 ∧ a n ^ 2 - a (n - 1) * a (n + 1) = 1

/-- The theorem to be proved -/
theorem exists_alpha_for_sequence (a : ℕ → ℝ) (h : SequenceA a) :
  ∃ α : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = α * a n - a (n - 1) := by
  sorry

end exists_alpha_for_sequence_l2928_292849


namespace eleven_divides_six_digit_repeating_l2928_292872

/-- A 6-digit positive integer where the first three digits are the same as its last three digits -/
def SixDigitRepeating (z : ℕ) : Prop :=
  ∃ (a b c : ℕ), 
    0 < a ∧ a ≤ 9 ∧ 
    b ≤ 9 ∧ 
    c ≤ 9 ∧ 
    z = 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c

theorem eleven_divides_six_digit_repeating (z : ℕ) (h : SixDigitRepeating z) : 
  11 ∣ z := by
  sorry

end eleven_divides_six_digit_repeating_l2928_292872


namespace transportation_cost_calculation_l2928_292848

def transportation_cost (initial_amount dress_cost pants_cost jacket_cost dress_count pants_count jacket_count remaining_amount : ℕ) : ℕ :=
  let clothes_cost := dress_cost * dress_count + pants_cost * pants_count + jacket_cost * jacket_count
  let total_spent := initial_amount - remaining_amount
  total_spent - clothes_cost

theorem transportation_cost_calculation :
  transportation_cost 400 20 12 30 5 3 4 139 = 5 :=
by sorry

end transportation_cost_calculation_l2928_292848


namespace unique_solution_l2928_292822

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := x + y = 3
def equation2 (x y : ℝ) : Prop := x - y = 1

-- Theorem statement
theorem unique_solution :
  ∃! (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = 2 ∧ y = 1 := by
  sorry

end unique_solution_l2928_292822


namespace sum_of_squares_of_roots_l2928_292828

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 4 * p^2 + 7 * p - 9 = 0) →
  (3 * q^3 - 4 * q^2 + 7 * q - 9 = 0) →
  (3 * r^3 - 4 * r^2 + 7 * r - 9 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end sum_of_squares_of_roots_l2928_292828


namespace election_win_percentage_l2928_292897

theorem election_win_percentage 
  (total_voters : ℕ) 
  (republican_ratio : ℚ) 
  (democrat_ratio : ℚ) 
  (republican_for_x : ℚ) 
  (democrat_for_x : ℚ) :
  republican_ratio + democrat_ratio = 1 →
  republican_ratio / democrat_ratio = 3 / 2 →
  republican_for_x = 3 / 4 →
  democrat_for_x = 3 / 20 →
  let total_for_x := republican_ratio * republican_for_x + democrat_ratio * democrat_for_x
  let total_for_y := 1 - total_for_x
  (total_for_x - total_for_y) / (total_for_x + total_for_y) = 1 / 50 :=
by sorry

end election_win_percentage_l2928_292897


namespace swap_correct_specific_swap_l2928_292844

def swap_values (a b : ℕ) : ℕ × ℕ := 
  let c := a
  let a' := b
  let b' := c
  (a', b')

theorem swap_correct (a b : ℕ) : 
  let (a', b') := swap_values a b
  a' = b ∧ b' = a := by
sorry

theorem specific_swap : 
  let (a', b') := swap_values 10 20
  a' = 20 ∧ b' = 10 := by
sorry

end swap_correct_specific_swap_l2928_292844


namespace intersection_complement_problem_l2928_292878

open Set

theorem intersection_complement_problem :
  let U : Set ℝ := Set.univ
  let A : Set ℝ := {x | x > 0}
  let B : Set ℝ := {x | x > 1}
  A ∩ (U \ B) = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_complement_problem_l2928_292878


namespace minimum_speed_x_l2928_292877

/-- Minimum speed problem for vehicle X --/
theorem minimum_speed_x (distance_xy distance_xz speed_y speed_z : ℝ) 
  (h1 : distance_xy = 500)
  (h2 : distance_xz = 300)
  (h3 : speed_y = 40)
  (h4 : speed_z = 30)
  (h5 : speed_y > speed_z)
  (speed_x : ℝ) :
  speed_x > 135 ↔ distance_xz / (speed_x - speed_z) < distance_xy / (speed_x + speed_y) :=
by sorry

end minimum_speed_x_l2928_292877


namespace inequality_proof_l2928_292800

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (z^2 - x^2) / (x + y) + (x^2 - y^2) / (y + z) + (y^2 - z^2) / (z + x) ≥ 0 := by
  sorry

end inequality_proof_l2928_292800


namespace beef_for_community_event_l2928_292851

/-- The amount of beef needed for a given number of hamburgers -/
def beef_needed (hamburgers : ℕ) : ℚ :=
  (4 : ℚ) / 10 * hamburgers

theorem beef_for_community_event : beef_needed 35 = 14 := by
  sorry

end beef_for_community_event_l2928_292851


namespace intersection_of_A_and_B_l2928_292813

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end intersection_of_A_and_B_l2928_292813


namespace stacy_heather_walking_problem_l2928_292881

/-- The problem of Stacy and Heather walking towards each other -/
theorem stacy_heather_walking_problem 
  (total_distance : ℝ) 
  (heather_speed : ℝ) 
  (stacy_speed : ℝ) 
  (heather_distance : ℝ) :
  total_distance = 15 →
  heather_speed = 5 →
  stacy_speed = heather_speed + 1 →
  heather_distance = 5.7272727272727275 →
  ∃ (time_difference : ℝ), 
    time_difference = 24 / 60 ∧ 
    time_difference * stacy_speed = total_distance - (heather_distance + stacy_speed * (heather_distance / heather_speed)) :=
by sorry

end stacy_heather_walking_problem_l2928_292881
