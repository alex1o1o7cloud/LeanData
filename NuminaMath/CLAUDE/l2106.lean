import Mathlib

namespace NUMINAMATH_CALUDE_yolandas_walking_rate_l2106_210662

/-- Proves that Yolanda's walking rate is 3 miles per hour given the problem conditions -/
theorem yolandas_walking_rate
  (total_distance : ℝ)
  (bob_start_delay : ℝ)
  (bob_rate : ℝ)
  (bob_distance : ℝ)
  (h1 : total_distance = 52)
  (h2 : bob_start_delay = 1)
  (h3 : bob_rate = 4)
  (h4 : bob_distance = 28) :
  ∃ (yolanda_rate : ℝ),
    yolanda_rate = 3 ∧
    yolanda_rate * (bob_distance / bob_rate + bob_start_delay) + bob_distance = total_distance :=
by sorry

end NUMINAMATH_CALUDE_yolandas_walking_rate_l2106_210662


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2106_210600

/-- Given a right triangle with acute angles in the ratio 5:4 and hypotenuse 10 cm,
    the length of the side opposite the smaller angle is 10 * sin(40°) -/
theorem right_triangle_side_length (a b c : ℝ) (θ₁ θ₂ : Real) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem (right triangle condition)
  c = 10 →  -- hypotenuse length
  θ₁ / θ₂ = 5 / 4 →  -- ratio of acute angles
  θ₁ + θ₂ = π / 2 →  -- sum of acute angles in a right triangle
  θ₂ < θ₁ →  -- θ₂ is the smaller angle
  b = 10 * Real.sin (40 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2106_210600


namespace NUMINAMATH_CALUDE_stan_pays_magician_l2106_210643

/-- The total amount Stan pays the magician -/
def total_payment (hourly_rate : ℕ) (hours_per_day : ℕ) (weeks : ℕ) : ℕ :=
  hourly_rate * hours_per_day * (weeks * 7)

/-- Proof that Stan pays the magician $2520 -/
theorem stan_pays_magician :
  total_payment 60 3 2 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_stan_pays_magician_l2106_210643


namespace NUMINAMATH_CALUDE_no_a_exists_for_union_range_of_a_for_intersection_l2106_210625

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x = 0}

-- Define set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | a*x^2 - 2*x + 8 = 0}

-- Theorem 1: There does not exist a real number 'a' such that A ∪ B = {0, 2, 4}
theorem no_a_exists_for_union : ¬ ∃ a : ℝ, A ∪ B a = {0, 2, 4} := by
  sorry

-- Theorem 2: The range of 'a' when A ∩ B = B is {0} ∪ (1/8, +∞)
theorem range_of_a_for_intersection (a : ℝ) : 
  (A ∩ B a = B a) ↔ (a = 0 ∨ a > 1/8) := by
  sorry

end NUMINAMATH_CALUDE_no_a_exists_for_union_range_of_a_for_intersection_l2106_210625


namespace NUMINAMATH_CALUDE_store_coupon_distribution_l2106_210647

/-- Calculates the number of coupons per remaining coloring book -/
def coupons_per_book (initial_stock : ℚ) (books_sold : ℚ) (total_coupons : ℕ) : ℚ :=
  total_coupons / (initial_stock - books_sold)

/-- Proves that given the problem conditions, the number of coupons per remaining book is 4 -/
theorem store_coupon_distribution :
  coupons_per_book 40 20 80 = 4 := by
  sorry

#eval coupons_per_book 40 20 80

end NUMINAMATH_CALUDE_store_coupon_distribution_l2106_210647


namespace NUMINAMATH_CALUDE_rachel_essay_time_l2106_210636

/-- Calculates the total time spent on an essay given the writing speed, number of pages, research time, and editing time. -/
def total_essay_time (writing_speed : ℝ) (pages : ℕ) (research_time : ℝ) (editing_time : ℝ) : ℝ :=
  (pages : ℝ) * writing_speed + research_time + editing_time

/-- Proves that Rachel spent 5 hours on her essay given the conditions. -/
theorem rachel_essay_time : 
  let writing_speed : ℝ := 30  -- minutes per page
  let pages : ℕ := 6
  let research_time : ℝ := 45  -- minutes
  let editing_time : ℝ := 75   -- minutes
  total_essay_time writing_speed pages research_time editing_time / 60 = 5 := by
sorry


end NUMINAMATH_CALUDE_rachel_essay_time_l2106_210636


namespace NUMINAMATH_CALUDE_flood_monitoring_technologies_l2106_210628

-- Define the set of available technologies
inductive GeoTechnology
| RemoteSensing
| GPS
| GIS
| DigitalEarth

-- Define the capabilities of technologies
def canMonitorDisaster (tech : GeoTechnology) : Prop :=
  match tech with
  | GeoTechnology.RemoteSensing => true
  | _ => false

def canManageInfo (tech : GeoTechnology) : Prop :=
  match tech with
  | GeoTechnology.GIS => true
  | _ => false

def isEffectiveForFloodMonitoring (tech : GeoTechnology) : Prop :=
  canMonitorDisaster tech ∨ canManageInfo tech

-- Define the set of effective technologies
def effectiveTechnologies : Set GeoTechnology :=
  {tech | isEffectiveForFloodMonitoring tech}

-- Theorem statement
theorem flood_monitoring_technologies :
  effectiveTechnologies = {GeoTechnology.RemoteSensing, GeoTechnology.GIS} :=
sorry

end NUMINAMATH_CALUDE_flood_monitoring_technologies_l2106_210628


namespace NUMINAMATH_CALUDE_total_fish_count_l2106_210657

/-- The total number of fish caught by Brendan and his dad -/
def total_fish (morning_catch : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) : ℕ :=
  morning_catch + afternoon_catch - thrown_back + dad_catch

/-- Theorem stating the total number of fish caught by Brendan and his dad -/
theorem total_fish_count :
  total_fish 8 3 5 13 = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l2106_210657


namespace NUMINAMATH_CALUDE_zeros_in_repeated_nines_square_l2106_210683

/-- The number of zeros in the decimal representation of n^2 -/
def zeros_in_square (n : ℕ) : ℕ := sorry

/-- The number of nines in the decimal representation of n -/
def count_nines (n : ℕ) : ℕ := sorry

theorem zeros_in_repeated_nines_square (n : ℕ) :
  (∀ k ≤ 3, zeros_in_square (10^k - 1) = k - 1) →
  count_nines 999999 = 6 →
  zeros_in_square 999999 = 5 := by sorry

end NUMINAMATH_CALUDE_zeros_in_repeated_nines_square_l2106_210683


namespace NUMINAMATH_CALUDE_complex_not_purely_imaginary_range_l2106_210639

theorem complex_not_purely_imaginary_range (a : ℝ) : 
  ¬(∃ (y : ℝ), (a^2 - a - 2) + (|a-1| - 1)*I = y*I) → a ≠ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_not_purely_imaginary_range_l2106_210639


namespace NUMINAMATH_CALUDE_count_valid_B_l2106_210646

def is_divisible_by_33 (n : ℕ) : Prop := n % 33 = 0

def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def number_3A3B3 (A B : ℕ) : ℕ := 30303 + 1000 * A + 10 * B

theorem count_valid_B :
  ∃ (S : Finset ℕ),
    (∀ B ∈ S, digit B) ∧
    (∀ A, digit A → (is_divisible_by_33 (number_3A3B3 A B) ↔ B ∈ S)) ∧
    Finset.card S = 10 :=
sorry

end NUMINAMATH_CALUDE_count_valid_B_l2106_210646


namespace NUMINAMATH_CALUDE_prove_max_value_l2106_210609

def max_value_theorem (a b c : ℝ × ℝ) : Prop :=
  let norm_squared := λ v : ℝ × ℝ => v.1^2 + v.2^2
  norm_squared a = 9 ∧ 
  norm_squared b = 4 ∧ 
  norm_squared c = 16 →
  norm_squared (a.1 - 3*b.1, a.2 - 3*b.2) + 
  norm_squared (b.1 - 3*c.1, b.2 - 3*c.2) + 
  norm_squared (c.1 - 3*a.1, c.2 - 3*a.2) ≤ 428

theorem prove_max_value : ∀ a b c : ℝ × ℝ, max_value_theorem a b c := by
  sorry

end NUMINAMATH_CALUDE_prove_max_value_l2106_210609


namespace NUMINAMATH_CALUDE_textbook_recycling_savings_scientific_notation_l2106_210608

theorem textbook_recycling_savings_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    (31680000000 : ℝ) = a * (10 : ℝ) ^ n ∧
    a = 3.168 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_textbook_recycling_savings_scientific_notation_l2106_210608


namespace NUMINAMATH_CALUDE_triangle_side_length_l2106_210698

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  a : Real
  b : Real

-- Define the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : t.A = π / 3)  -- 60 degrees in radians
  (h2 : t.a = Real.sqrt 3)
  (h3 : t.B = π / 6)  -- 30 degrees in radians
  : t.b = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2106_210698


namespace NUMINAMATH_CALUDE_cereal_box_theorem_l2106_210672

/-- The number of clusters of oats in each spoonful -/
def clusters_per_spoonful : ℕ := 4

/-- The number of spoonfuls of cereal in each bowl -/
def spoonfuls_per_bowl : ℕ := 25

/-- The number of clusters of oats in each box -/
def clusters_per_box : ℕ := 500

/-- The number of bowlfuls of cereal in each box -/
def bowls_per_box : ℕ := 5

theorem cereal_box_theorem : 
  clusters_per_box / (clusters_per_spoonful * spoonfuls_per_bowl) = bowls_per_box := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_theorem_l2106_210672


namespace NUMINAMATH_CALUDE_auction_bid_relationship_l2106_210627

/-- Joe's bid at the auction -/
def joes_bid : ℝ := 160000

/-- Nelly's winning bid at the auction -/
def nellys_bid : ℝ := 482000

/-- Theorem stating the relationship between Joe's and Nelly's bids -/
theorem auction_bid_relationship : 
  nellys_bid = 3 * joes_bid + 2000 ∧ joes_bid = 160000 := by
  sorry

end NUMINAMATH_CALUDE_auction_bid_relationship_l2106_210627


namespace NUMINAMATH_CALUDE_no_infinite_sequence_exists_l2106_210651

theorem no_infinite_sequence_exists : ¬ ∃ (k : ℕ → ℝ), 
  (∀ n, k n ≠ 0) ∧ 
  (∀ n, k (n + 1) = k n - 1 / k n) ∧ 
  (∀ n, k n * k (n + 1) ≥ 0) := by
sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_exists_l2106_210651


namespace NUMINAMATH_CALUDE_neznaika_contradiction_l2106_210634

theorem neznaika_contradiction (S T : ℝ) 
  (h1 : S ≤ 50 * T) 
  (h2 : 60 * T ≤ S) 
  (h3 : T > 0) : 
  False :=
by sorry

end NUMINAMATH_CALUDE_neznaika_contradiction_l2106_210634


namespace NUMINAMATH_CALUDE_polygon_angles_l2106_210620

theorem polygon_angles (n : ℕ) : 
  (n - 2) * 180 = 5 * 360 → n = 12 := by
sorry

end NUMINAMATH_CALUDE_polygon_angles_l2106_210620


namespace NUMINAMATH_CALUDE_sum_of_roots_of_quadratic_l2106_210664

theorem sum_of_roots_of_quadratic : ∃ (x₁ x₂ : ℝ), 
  x₁^2 - 6*x₁ + 8 = 0 ∧ 
  x₂^2 - 6*x₂ + 8 = 0 ∧ 
  x₁ ≠ x₂ ∧
  x₁ + x₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_quadratic_l2106_210664


namespace NUMINAMATH_CALUDE_team_a_score_l2106_210616

theorem team_a_score (total_points team_b_points team_c_points : ℕ) 
  (h1 : team_b_points = 9)
  (h2 : team_c_points = 4)
  (h3 : total_points = 15)
  : total_points - (team_b_points + team_c_points) = 2 := by
  sorry

end NUMINAMATH_CALUDE_team_a_score_l2106_210616


namespace NUMINAMATH_CALUDE_polynomial_no_ab_term_l2106_210649

theorem polynomial_no_ab_term (m : ℤ) : 
  (∀ a b : ℤ, 2 * (a^2 - 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2) = a^2 - 4*b^2) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_no_ab_term_l2106_210649


namespace NUMINAMATH_CALUDE_similar_right_triangles_perimeter_l2106_210622

theorem similar_right_triangles_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * a + a * a = b * b →  -- First triangle is right-angled with equal legs
  (c / b) * (c / b) = 2 →  -- Ratio of hypotenuses squared
  2 * ((c / b) * a) + c = 30 * Real.sqrt 2 + 30 := by
sorry

end NUMINAMATH_CALUDE_similar_right_triangles_perimeter_l2106_210622


namespace NUMINAMATH_CALUDE_dot_product_specific_vectors_l2106_210670

theorem dot_product_specific_vectors :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (1, -1)
  (a.1 * b.1 + a.2 * b.2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_specific_vectors_l2106_210670


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l2106_210611

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a b : Line) (α : Plane) 
  (h1 : parallel a α) 
  (h2 : perpendicular b α) : 
  perpendicularLines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l2106_210611


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2106_210640

/-- The quadratic equation x^2 + 4x - 4 = 0 has two distinct real roots -/
theorem quadratic_two_distinct_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ - 4 = 0 ∧ x₂^2 + 4*x₂ - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2106_210640


namespace NUMINAMATH_CALUDE_cylinder_no_triangular_cross_section_l2106_210652

-- Define the types of geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | TriangularPrism
  | Cube

-- Define a function to check if a solid can have a triangular cross-section
def canHaveTriangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => False
  | _ => True

-- Theorem statement
theorem cylinder_no_triangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveTriangularCrossSection solid) ↔ solid = GeometricSolid.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_cylinder_no_triangular_cross_section_l2106_210652


namespace NUMINAMATH_CALUDE_jane_ice_cream_purchase_l2106_210669

/-- The number of ice cream cones Jane purchased -/
def num_ice_cream_cones : ℕ := 15

/-- The number of pudding cups Jane purchased -/
def num_pudding_cups : ℕ := 5

/-- The cost of one ice cream cone in dollars -/
def ice_cream_cost : ℕ := 5

/-- The cost of one pudding cup in dollars -/
def pudding_cost : ℕ := 2

/-- The difference in dollars between ice cream and pudding expenses -/
def expense_difference : ℕ := 65

theorem jane_ice_cream_purchase :
  num_ice_cream_cones * ice_cream_cost = num_pudding_cups * pudding_cost + expense_difference :=
by sorry

end NUMINAMATH_CALUDE_jane_ice_cream_purchase_l2106_210669


namespace NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_195_l2106_210642

theorem cos_75_cos_15_minus_sin_75_sin_195 : 
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - 
  Real.sin (75 * π / 180) * Real.sin (195 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_195_l2106_210642


namespace NUMINAMATH_CALUDE_banana_distribution_l2106_210674

theorem banana_distribution (total_children : ℕ) 
  (original_bananas_per_child : ℕ) 
  (extra_bananas_per_child : ℕ) : 
  total_children = 720 →
  original_bananas_per_child = 2 →
  extra_bananas_per_child = 2 →
  (total_children - (total_children * original_bananas_per_child) / 
   (original_bananas_per_child + extra_bananas_per_child)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l2106_210674


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l2106_210666

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l2106_210666


namespace NUMINAMATH_CALUDE_notebooks_in_scenario3_l2106_210633

/-- Represents the production scenario in a factory --/
structure ProductionScenario where
  workers : ℕ
  hours : ℕ
  tablets : ℕ
  notebooks : ℕ

/-- The production rate for tablets (time to produce one tablet) --/
def tablet_rate : ℝ := 1

/-- The production rate for notebooks (time to produce one notebook) --/
def notebook_rate : ℝ := 2

/-- The given production scenarios --/
def scenario1 : ProductionScenario := ⟨120, 1, 360, 240⟩
def scenario2 : ProductionScenario := ⟨100, 2, 400, 500⟩
def scenario3 (n : ℕ) : ProductionScenario := ⟨80, 3, 480, n⟩

/-- Theorem stating that the number of notebooks produced in scenario3 is 120 --/
theorem notebooks_in_scenario3 : ∃ n : ℕ, scenario3 n = ⟨80, 3, 480, 120⟩ := by
  sorry


end NUMINAMATH_CALUDE_notebooks_in_scenario3_l2106_210633


namespace NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l2106_210668

theorem reciprocal_roots_quadratic (k : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ r₁ * r₂ = 1 ∧ 
    (∀ x : ℝ, 5.2 * x * x + 14.3 * x + k = 0 ↔ (x = r₁ ∨ x = r₂))) → 
  k = 5.2 := by
sorry


end NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l2106_210668


namespace NUMINAMATH_CALUDE_rounding_estimate_greater_l2106_210655

theorem rounding_estimate_greater (x y z x' y' z' : ℤ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hx' : x' ≥ x) (hy' : y' ≤ y) (hz' : z' ≤ z) :
  2 * ((x' : ℚ) / y' - z') > 2 * ((x : ℚ) / y - z) :=
sorry

end NUMINAMATH_CALUDE_rounding_estimate_greater_l2106_210655


namespace NUMINAMATH_CALUDE_f_derivative_l2106_210641

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - (x + 1)

-- State the theorem
theorem f_derivative : 
  ∀ x : ℝ, deriv f x = 4 * x + 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_l2106_210641


namespace NUMINAMATH_CALUDE_solution_set_f_geq_5_range_of_a_l2106_210673

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Part I: Solution set of f(x) ≥ 5
theorem solution_set_f_geq_5 :
  {x : ℝ | f x ≥ 5} = Set.Iic (-3) ∪ Set.Ici 2 :=
sorry

-- Part II: Range of a for which f(x) > a^2 - 2a holds for all x
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x > a^2 - 2*a) ↔ a ∈ Set.Ioo (-1) 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_5_range_of_a_l2106_210673


namespace NUMINAMATH_CALUDE_age_ratio_becomes_three_to_one_l2106_210695

/-- Represents the ages of Ted and Alex -/
structure Ages where
  ted : ℕ
  alex : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.ted - 3 = 4 * (ages.alex - 3)) ∧
  (ages.ted - 5 = 5 * (ages.alex - 5))

/-- The theorem to prove -/
theorem age_ratio_becomes_three_to_one (ages : Ages) :
  problem_conditions ages →
  ∃ (x : ℕ), x = 1 ∧ (ages.ted + x) / (ages.alex + x) = 3 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_becomes_three_to_one_l2106_210695


namespace NUMINAMATH_CALUDE_trip_time_at_new_speed_l2106_210601

-- Define the original speed, time, and new speed
def original_speed : ℝ := 80
def original_time : ℝ := 3
def new_speed : ℝ := 50

-- Define the constant distance
def distance : ℝ := original_speed * original_time

-- Theorem to prove
theorem trip_time_at_new_speed :
  distance / new_speed = 4.8 := by sorry

end NUMINAMATH_CALUDE_trip_time_at_new_speed_l2106_210601


namespace NUMINAMATH_CALUDE_correct_substitution_l2106_210650

theorem correct_substitution (x y : ℝ) : 
  (x = 3*y - 1 ∧ x - 2*y = 4) → (3*y - 1 - 2*y = 4) := by
  sorry

end NUMINAMATH_CALUDE_correct_substitution_l2106_210650


namespace NUMINAMATH_CALUDE_andrew_bought_65_planks_l2106_210684

/-- The number of wooden planks Andrew bought initially -/
def total_planks : ℕ :=
  let andrew_bedroom := 8
  let living_room := 20
  let kitchen := 11
  let guest_bedroom := andrew_bedroom - 2
  let hallway := 4
  let num_hallways := 2
  let ruined_per_bedroom := 3
  let num_bedrooms := 2
  let leftover := 6
  andrew_bedroom + living_room + kitchen + guest_bedroom + 
  (hallway * num_hallways) + (ruined_per_bedroom * num_bedrooms) + leftover

/-- Theorem stating that Andrew bought 65 wooden planks initially -/
theorem andrew_bought_65_planks : total_planks = 65 := by
  sorry

end NUMINAMATH_CALUDE_andrew_bought_65_planks_l2106_210684


namespace NUMINAMATH_CALUDE_system_no_solution_l2106_210659

theorem system_no_solution (n : ℝ) : 
  (∃ (x y z : ℝ), nx + y = 1 ∧ ny + z = 1 ∧ x + nz = 1) ↔ n ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_system_no_solution_l2106_210659


namespace NUMINAMATH_CALUDE_clothespin_count_total_clothespins_l2106_210644

theorem clothespin_count (handkerchiefs : ℕ) (ropes : ℕ) : ℕ :=
  let ends_per_handkerchief := 2
  let pins_for_handkerchiefs := handkerchiefs * ends_per_handkerchief
  let pins_for_ropes := ropes
  pins_for_handkerchiefs + pins_for_ropes

theorem total_clothespins : clothespin_count 40 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_clothespin_count_total_clothespins_l2106_210644


namespace NUMINAMATH_CALUDE_download_speed_calculation_l2106_210691

theorem download_speed_calculation (total_size : ℕ) (downloaded : ℕ) (remaining_time : ℕ) : 
  total_size = 880 ∧ downloaded = 310 ∧ remaining_time = 190 →
  (total_size - downloaded) / remaining_time = 3 := by
sorry

end NUMINAMATH_CALUDE_download_speed_calculation_l2106_210691


namespace NUMINAMATH_CALUDE_cost_of_horse_l2106_210613

/-- Proves that the cost of a horse is 2000 given the problem conditions --/
theorem cost_of_horse (total_cost : ℝ) (num_horses : ℕ) (num_cows : ℕ) 
  (horse_profit_rate : ℝ) (cow_profit_rate : ℝ) (total_profit : ℝ) :
  total_cost = 13400 ∧ 
  num_horses = 4 ∧ 
  num_cows = 9 ∧ 
  horse_profit_rate = 0.1 ∧ 
  cow_profit_rate = 0.2 ∧ 
  total_profit = 1880 →
  ∃ (horse_cost cow_cost : ℝ),
    num_horses * horse_cost + num_cows * cow_cost = total_cost ∧
    num_horses * horse_cost * horse_profit_rate + num_cows * cow_cost * cow_profit_rate = total_profit ∧
    horse_cost = 2000 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_horse_l2106_210613


namespace NUMINAMATH_CALUDE_meaningful_fraction_l2106_210667

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l2106_210667


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_min_value_sum_of_reciprocals_achieved_l2106_210689

theorem min_value_sum_of_reciprocals (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : x + y = 4) :
  (1 / (x - 1) + 1 / (y - 1)) ≥ 2 :=
sorry

theorem min_value_sum_of_reciprocals_achieved (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : x + y = 4) :
  (1 / (x - 1) + 1 / (y - 1) = 2) ↔ (x = 2 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_min_value_sum_of_reciprocals_achieved_l2106_210689


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_l2106_210648

/-- The intersection point of two lines is in the first quadrant iff a is in the range (-1, 2) -/
theorem intersection_in_first_quadrant (a : ℝ) :
  (∃ x y : ℝ, ax + y - 4 = 0 ∧ x - y - 2 = 0 ∧ x > 0 ∧ y > 0) ↔ -1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_in_first_quadrant_l2106_210648


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2106_210602

theorem mod_equivalence_unique_solution : 
  ∃! n : ℕ, n ≤ 6 ∧ n ≡ -4752 [ZMOD 7] := by sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2106_210602


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l2106_210656

theorem no_perfect_square_in_range : 
  ¬ ∃ (n : ℕ), 5 ≤ n ∧ n ≤ 12 ∧ ∃ (m : ℕ), 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l2106_210656


namespace NUMINAMATH_CALUDE_computer_task_time_l2106_210696

theorem computer_task_time (t_m : ℝ) (n : ℕ) (t_n : ℝ) : 
  t_m = 36 → 
  n = 12 → 
  n * (1 / t_m) + n * (1 / t_n) = 1 → 
  t_n = 18 := by
sorry

end NUMINAMATH_CALUDE_computer_task_time_l2106_210696


namespace NUMINAMATH_CALUDE_expression_evaluation_l2106_210688

theorem expression_evaluation (x : ℝ) (h : x < 0) :
  Real.sqrt (x^2 / (1 + (x + 1) / x)) = Real.sqrt (x^3 / (2*x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2106_210688


namespace NUMINAMATH_CALUDE_first_train_speed_l2106_210630

/-- Proves that the speed of the first train is 45 kmph given the problem conditions --/
theorem first_train_speed (v : ℝ) : 
  v > 0 → -- The speed of the first train is positive
  (∃ t : ℝ, t > 0 ∧ v * (1 + t) = 90 ∧ 90 * t = 90) → -- Equations from the problem
  v = 45 := by
  sorry


end NUMINAMATH_CALUDE_first_train_speed_l2106_210630


namespace NUMINAMATH_CALUDE_jose_investment_is_45000_l2106_210607

/-- Represents the investment scenario of Tom and Jose -/
structure InvestmentScenario where
  tom_investment : ℕ
  tom_months : ℕ
  jose_months : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given scenario -/
def calculate_jose_investment (scenario : InvestmentScenario) : ℕ :=
  (scenario.jose_profit * scenario.tom_investment * scenario.tom_months) /
  (scenario.tom_months * (scenario.total_profit - scenario.jose_profit))

/-- Theorem stating that Jose's investment is 45000 given the specific scenario -/
theorem jose_investment_is_45000 (scenario : InvestmentScenario)
  (h1 : scenario.tom_investment = 30000)
  (h2 : scenario.tom_months = 12)
  (h3 : scenario.jose_months = 10)
  (h4 : scenario.total_profit = 45000)
  (h5 : scenario.jose_profit = 25000) :
  calculate_jose_investment scenario = 45000 := by
  sorry


end NUMINAMATH_CALUDE_jose_investment_is_45000_l2106_210607


namespace NUMINAMATH_CALUDE_largest_three_digit_with_seven_hundreds_l2106_210678

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_seven_in_hundreds_place (n : ℕ) : Prop := (n / 100) % 10 = 7

theorem largest_three_digit_with_seven_hundreds : 
  ∀ n : ℕ, is_three_digit n → has_seven_in_hundreds_place n → n ≤ 799 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_with_seven_hundreds_l2106_210678


namespace NUMINAMATH_CALUDE_pencils_used_l2106_210604

theorem pencils_used (initial : ℕ) (current : ℕ) (h1 : initial = 94) (h2 : current = 91) :
  initial - current = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_used_l2106_210604


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l2106_210612

theorem square_perimeter_problem (area_A : ℝ) (prob_not_in_B : ℝ) : 
  area_A = 30 →
  prob_not_in_B = 0.4666666666666667 →
  let area_B := area_A * (1 - prob_not_in_B)
  let side_B := Real.sqrt area_B
  let perimeter_B := 4 * side_B
  perimeter_B = 16 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l2106_210612


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_specific_remainders_l2106_210635

theorem four_digit_numbers_with_specific_remainders :
  ∀ N : ℕ,
  (1000 ≤ N ∧ N ≤ 9999) →
  (N % 2 = 0 ∧ N % 3 = 1 ∧ N % 5 = 3 ∧ N % 7 = 5 ∧ N % 11 = 9) →
  (N = 2308 ∨ N = 4618 ∨ N = 6928 ∨ N = 9238) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_specific_remainders_l2106_210635


namespace NUMINAMATH_CALUDE_paperclips_exceed_500_l2106_210665

def paperclips (n : ℕ) : ℕ := 5 * 4^n

theorem paperclips_exceed_500 : 
  (∃ k, paperclips k > 500) ∧ 
  (∀ j, j < 3 → paperclips j ≤ 500) ∧
  (paperclips 3 > 500) := by
  sorry

end NUMINAMATH_CALUDE_paperclips_exceed_500_l2106_210665


namespace NUMINAMATH_CALUDE_marions_bike_cost_l2106_210618

theorem marions_bike_cost (marion_cost stephanie_cost total_cost : ℕ) : 
  stephanie_cost = 2 * marion_cost →
  total_cost = marion_cost + stephanie_cost →
  total_cost = 1068 →
  marion_cost = 356 := by
sorry

end NUMINAMATH_CALUDE_marions_bike_cost_l2106_210618


namespace NUMINAMATH_CALUDE_greatest_possible_area_l2106_210663

/-- A convex equilateral pentagon with side length 2 and two right angles -/
structure ConvexEquilateralPentagon where
  side_length : ℝ
  has_two_right_angles : Prop
  is_convex : Prop
  is_equilateral : Prop
  side_length_eq_two : side_length = 2

/-- The area of a ConvexEquilateralPentagon -/
def area (p : ConvexEquilateralPentagon) : ℝ := sorry

theorem greatest_possible_area (p : ConvexEquilateralPentagon) :
  area p ≤ 4 + Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_greatest_possible_area_l2106_210663


namespace NUMINAMATH_CALUDE_maria_paint_cans_l2106_210685

/-- Represents the paint situation for Maria's room painting problem -/
structure PaintSituation where
  initialRooms : ℕ
  finalRooms : ℕ
  lostCans : ℕ

/-- Calculates the number of cans used for the final number of rooms -/
def cansUsed (s : PaintSituation) : ℕ :=
  s.finalRooms / ((s.initialRooms - s.finalRooms) / s.lostCans)

/-- Theorem stating that for Maria's specific situation, 16 cans were used -/
theorem maria_paint_cans :
  let s : PaintSituation := { initialRooms := 40, finalRooms := 32, lostCans := 4 }
  cansUsed s = 16 := by sorry

end NUMINAMATH_CALUDE_maria_paint_cans_l2106_210685


namespace NUMINAMATH_CALUDE_combined_instruments_count_l2106_210606

/-- Represents the number of instruments owned by a person -/
structure InstrumentCount where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- Calculates the total number of instruments -/
def totalInstruments (ic : InstrumentCount) : ℕ :=
  ic.flutes + ic.horns + ic.harps

/-- Charlie's instrument count -/
def charlie : InstrumentCount :=
  { flutes := 1, horns := 2, harps := 1 }

/-- Carli's instrument count -/
def carli : InstrumentCount :=
  { flutes := 2 * charlie.flutes,
    horns := charlie.horns / 2,
    harps := 0 }

/-- Theorem: The combined total number of musical instruments owned by Charlie and Carli is 7 -/
theorem combined_instruments_count :
  totalInstruments charlie + totalInstruments carli = 7 := by
  sorry

end NUMINAMATH_CALUDE_combined_instruments_count_l2106_210606


namespace NUMINAMATH_CALUDE_parallelogram_height_l2106_210653

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) (h1 : area = 33.3) (h2 : base = 9) 
    (h3 : area = base * height) : height = 3.7 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2106_210653


namespace NUMINAMATH_CALUDE_science_fair_participants_l2106_210680

theorem science_fair_participants (total_girls : ℕ) (total_boys : ℕ)
  (girls_participation_rate : ℚ) (boys_participation_rate : ℚ)
  (h1 : total_girls = 150)
  (h2 : total_boys = 100)
  (h3 : girls_participation_rate = 4 / 5)
  (h4 : boys_participation_rate = 3 / 4) :
  let participating_girls : ℚ := girls_participation_rate * total_girls
  let participating_boys : ℚ := boys_participation_rate * total_boys
  let total_participants : ℚ := participating_girls + participating_boys
  participating_girls / total_participants = 8 / 13 := by
sorry

end NUMINAMATH_CALUDE_science_fair_participants_l2106_210680


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l2106_210637

def f (x : ℝ) : ℝ := -x + 1

theorem linear_function_decreasing (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f x₂ = y₂) 
  (h3 : x₁ < x₂) : 
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l2106_210637


namespace NUMINAMATH_CALUDE_A_power_2023_l2106_210624

def A : Matrix (Fin 3) (Fin 3) ℚ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_2023 : A^2023 = A := by sorry

end NUMINAMATH_CALUDE_A_power_2023_l2106_210624


namespace NUMINAMATH_CALUDE_deduced_card_final_card_l2106_210605

-- Define the suits and ranks
inductive Suit
| Hearts | Spades | Clubs | Diamonds

inductive Rank
| A | K | Q | J | Ten | Nine | Eight | Seven | Six | Five | Four | Three | Two

-- Define a card as a pair of suit and rank
structure Card where
  suit : Suit
  rank : Rank

-- Define the set of cards in the drawer
def drawer : List Card := [
  ⟨Suit.Hearts, Rank.A⟩, ⟨Suit.Hearts, Rank.Q⟩, ⟨Suit.Hearts, Rank.Four⟩,
  ⟨Suit.Spades, Rank.J⟩, ⟨Suit.Spades, Rank.Eight⟩, ⟨Suit.Spades, Rank.Four⟩,
  ⟨Suit.Spades, Rank.Two⟩, ⟨Suit.Spades, Rank.Seven⟩, ⟨Suit.Spades, Rank.Three⟩,
  ⟨Suit.Clubs, Rank.K⟩, ⟨Suit.Clubs, Rank.Q⟩, ⟨Suit.Clubs, Rank.Five⟩,
  ⟨Suit.Clubs, Rank.Four⟩, ⟨Suit.Clubs, Rank.Six⟩,
  ⟨Suit.Diamonds, Rank.A⟩, ⟨Suit.Diamonds, Rank.Five⟩
]

-- Define the conditions based on the conversation
def qian_first_statement (c : Card) : Prop :=
  c.rank = Rank.A ∨ c.rank = Rank.Q ∨ c.rank = Rank.Five ∨ c.rank = Rank.Four

def sun_first_statement (c : Card) : Prop :=
  c.suit = Suit.Hearts ∨ c.suit = Suit.Diamonds

def qian_second_statement (c : Card) : Prop :=
  c.rank ≠ Rank.A

-- The main theorem
theorem deduced_card :
  ∃! c : Card, c ∈ drawer ∧
    qian_first_statement c ∧
    sun_first_statement c ∧
    qian_second_statement c :=
  sorry

-- The final conclusion
theorem final_card :
  ∃! c : Card, c ∈ drawer ∧
    qian_first_statement c ∧
    sun_first_statement c ∧
    qian_second_statement c ∧
    c = ⟨Suit.Diamonds, Rank.Five⟩ :=
  sorry

end NUMINAMATH_CALUDE_deduced_card_final_card_l2106_210605


namespace NUMINAMATH_CALUDE_complement_N_star_in_N_l2106_210661

def N : Set ℕ := {n : ℕ | True}
def N_star : Set ℕ := {n : ℕ | n > 0}

theorem complement_N_star_in_N : N \ N_star = {0} := by sorry

end NUMINAMATH_CALUDE_complement_N_star_in_N_l2106_210661


namespace NUMINAMATH_CALUDE_triangle_perimeter_inside_polygon_l2106_210610

-- Define a polygon as a set of points in 2D space
def Polygon : Type := Set (ℝ × ℝ)

-- Define a triangle as a set of three points in 2D space
def Triangle : Type := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Function to check if a triangle is inside a polygon
def isInside (t : Triangle) (p : Polygon) : Prop := sorry

-- Function to calculate the perimeter of a polygon
def perimeterPolygon (p : Polygon) : ℝ := sorry

-- Function to calculate the perimeter of a triangle
def perimeterTriangle (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_perimeter_inside_polygon (t : Triangle) (p : Polygon) :
  isInside t p → perimeterTriangle t ≤ perimeterPolygon p := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_inside_polygon_l2106_210610


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2106_210690

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_properties (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (is_geometric_sequence (fun n => (a n)^2)) ∧
  (is_geometric_sequence (fun n => a (2*n))) ∧
  (is_geometric_sequence (fun n => 1 / (a n))) ∧
  (is_geometric_sequence (fun n => |a n|)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2106_210690


namespace NUMINAMATH_CALUDE_race_start_distances_l2106_210603

-- Define the start distances
def start_A_B : ℝ := 50
def start_B_C : ℝ := 157.89473684210532

-- Theorem statement
theorem race_start_distances :
  let start_A_C := start_A_B + start_B_C
  start_A_C = 207.89473684210532 := by sorry

end NUMINAMATH_CALUDE_race_start_distances_l2106_210603


namespace NUMINAMATH_CALUDE_valid_perm_count_l2106_210682

/-- 
Given a permutation π of n distinct elements, we define:
inv_count(π, i) = number of elements to the left of π(i) that are greater than π(i) +
                  number of elements to the right of π(i) that are less than π(i)
-/
def inv_count (π : Fin n → Fin n) (i : Fin n) : ℕ := sorry

/-- A permutation is valid if inv_count is even for all elements -/
def is_valid_perm (π : Fin n → Fin n) : Prop :=
  ∀ i, Even (inv_count π i)

/-- The number of valid permutations -/
def count_valid_perms (n : ℕ) : ℕ := sorry

theorem valid_perm_count (n : ℕ) : 
  count_valid_perms n = (Nat.factorial (n / 2)) * (Nat.factorial ((n + 1) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_valid_perm_count_l2106_210682


namespace NUMINAMATH_CALUDE_range_of_b_l2106_210626

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem range_of_b (f : ℝ → ℝ) (b : ℝ) :
  is_odd_function f →
  has_period f 4 →
  (∀ x ∈ Set.Ioo 0 2, f x = Real.log (x^2 - x + b)) →
  (∃ (zs : Finset ℝ), zs.card = 5 ∧ ∀ z ∈ zs, z ∈ Set.Icc (-2) 2 ∧ f z = 0) →
  b ∈ Set.Ioo (1/4) 1 ∪ {5/4} :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l2106_210626


namespace NUMINAMATH_CALUDE_remainder_problem_l2106_210692

theorem remainder_problem (x : ℕ) (h1 : x > 1) (h2 : ¬ Nat.Prime x) 
  (h3 : 5000 % x = 25) : 9995 % x = 25 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2106_210692


namespace NUMINAMATH_CALUDE_nils_geese_count_l2106_210632

/-- Represents the number of geese Nils initially has. -/
def initial_geese : ℕ := sorry

/-- Represents the number of days the feed lasts with the initial number of geese. -/
def initial_days : ℕ := sorry

/-- Represents the amount of feed one goose consumes per day. -/
def feed_per_goose_per_day : ℝ := sorry

/-- Represents the total amount of feed available. -/
def total_feed : ℝ := sorry

/-- The feed lasts 20 days longer when 50 geese are sold. -/
axiom sell_condition : total_feed = feed_per_goose_per_day * (initial_days + 20) * (initial_geese - 50)

/-- The feed lasts 10 days less when 100 geese are bought. -/
axiom buy_condition : total_feed = feed_per_goose_per_day * (initial_days - 10) * (initial_geese + 100)

/-- The initial amount of feed equals the product of initial days, initial geese, and feed per goose per day. -/
axiom initial_condition : total_feed = feed_per_goose_per_day * initial_days * initial_geese

/-- Theorem stating that Nils initially has 300 geese. -/
theorem nils_geese_count : initial_geese = 300 := by sorry

end NUMINAMATH_CALUDE_nils_geese_count_l2106_210632


namespace NUMINAMATH_CALUDE_system_solution_l2106_210621

theorem system_solution (a b x y z : ℝ) (ha : a ≠ 0) (hb : b ≠ 1) 
  (hyz : y ≠ z) (h2y3z : 2*y ≠ 3*z) (h3a2x2ay : 3*a^2*x ≠ 2*a*y) (hb_neq : b ≠ -19/15) :
  (a * x + z) / (y - z) = (1 + b) / (1 - b) ∧
  (2 * a * x - 3 * b) / (2 * y - 3 * z) = 1 ∧
  (5 * z - 4 * b) / (3 * a^2 * x - 2 * a * y) = b / a →
  x = 1/a ∧ y = 1 ∧ z = b := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2106_210621


namespace NUMINAMATH_CALUDE_residue_of_11_power_1234_mod_19_l2106_210675

theorem residue_of_11_power_1234_mod_19 :
  (11 : ℤ)^1234 ≡ 16 [ZMOD 19] := by sorry

end NUMINAMATH_CALUDE_residue_of_11_power_1234_mod_19_l2106_210675


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2106_210694

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ
  repeatingLength : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + (x.repeatingPart : ℚ) / ((10 ^ x.repeatingLength - 1) : ℚ)

/-- The repeating decimal 7.036036036... -/
def number : RepeatingDecimal :=
  { integerPart := 7
    repeatingPart := 36
    repeatingLength := 3 }

theorem repeating_decimal_equals_fraction :
  toRational number = 781 / 111 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2106_210694


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l2106_210617

/-- Triangle ABC with vertices A and B on the y-axis, and perimeter 10 -/
structure Triangle :=
  (C : ℝ × ℝ)
  (perimeter : ℝ)
  (h_A : A = (0, 2))
  (h_B : B = (0, -2))
  (h_perimeter : perimeter = 10)

/-- The equation of an ellipse -/
def is_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The trajectory of vertex C forms an ellipse -/
theorem trajectory_is_ellipse (t : Triangle) : 
  ∃ (x y : ℝ), is_ellipse x y 5 9 ∧ x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l2106_210617


namespace NUMINAMATH_CALUDE_parade_function_correct_l2106_210677

/-- Represents the function relationship between row number and number of people in a trapezoidal parade. -/
def parade_function (x : ℤ) : ℤ := x + 39

/-- Theorem stating the correctness of the parade function for a specific trapezoidal parade configuration. -/
theorem parade_function_correct :
  ∀ x : ℤ, 1 ≤ x → x ≤ 60 →
  (parade_function x = 40 + (x - 1)) ∧
  (parade_function 1 = 40) ∧
  (∀ i : ℤ, 1 ≤ i → i < 60 → parade_function (i + 1) = parade_function i + 1) :=
by sorry

end NUMINAMATH_CALUDE_parade_function_correct_l2106_210677


namespace NUMINAMATH_CALUDE_cd_ratio_l2106_210631

/-- Represents the number of CDs Tyler has at different stages --/
structure CDCount where
  initial : ℕ
  given_away : ℕ
  bought : ℕ
  final : ℕ

/-- Theorem stating the ratio of CDs given away to initial CDs --/
theorem cd_ratio (c : CDCount) 
  (h1 : c.initial = 21)
  (h2 : c.bought = 8)
  (h3 : c.final = 22)
  (h4 : c.initial - c.given_away + c.bought = c.final) :
  (c.given_away : ℚ) / c.initial = 1 / 3 := by
  sorry

#check cd_ratio

end NUMINAMATH_CALUDE_cd_ratio_l2106_210631


namespace NUMINAMATH_CALUDE_sequence_periodicity_l2106_210660

def units_digit (n : ℕ) : ℕ := n % 10

def a (n : ℕ) : ℕ := units_digit (n^n)

theorem sequence_periodicity : ∀ n : ℕ, a (n + 20) = a n := by
  sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l2106_210660


namespace NUMINAMATH_CALUDE_N_prime_iff_k_eq_two_l2106_210679

def N (k : ℕ) : ℕ := (10^(2*k) - 1) / 99

theorem N_prime_iff_k_eq_two :
  ∀ k : ℕ, k > 0 → (Nat.Prime (N k) ↔ k = 2) := by sorry

end NUMINAMATH_CALUDE_N_prime_iff_k_eq_two_l2106_210679


namespace NUMINAMATH_CALUDE_age_sum_proof_l2106_210654

/-- Tom's age in years -/
def tom_age : ℕ := 9

/-- Tom's sister's age in years -/
def sister_age : ℕ := tom_age / 2 + 1

/-- The sum of Tom's and his sister's ages -/
def sum_ages : ℕ := tom_age + sister_age

theorem age_sum_proof : sum_ages = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_proof_l2106_210654


namespace NUMINAMATH_CALUDE_divisibility_proof_l2106_210629

def is_valid_number (r b c : Nat) : Prop :=
  r < 10 ∧ b < 10 ∧ c < 10

def number_value (r b c : Nat) : Nat :=
  523000 + r * 100 + b * 10 + c

theorem divisibility_proof (r b c : Nat) 
  (h1 : is_valid_number r b c) 
  (h2 : r * b * c = 180) 
  (h3 : (number_value r b c) % 89 = 0) : 
  (number_value r b c) % 5886 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_proof_l2106_210629


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2106_210615

theorem arithmetic_sequence_sum : ∀ (a₁ aₙ d n : ℕ),
  a₁ = 1 →
  aₙ = 21 →
  d = 2 →
  n * (a₁ + aₙ) = (aₙ - a₁ + d) * (aₙ - a₁ + d) →
  n * (a₁ + aₙ) / 2 = 121 :=
by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2106_210615


namespace NUMINAMATH_CALUDE_triangle_inequality_with_powers_l2106_210658

theorem triangle_inequality_with_powers (n : ℕ) (a b c : ℝ) 
  (hn : n > 1) 
  (hab : a > 0) (hbc : b > 0) (hca : c > 0)
  (hsum : a + b + c = 1)
  (htriangle : a < b + c ∧ b < a + c ∧ c < a + b) :
  (a^n + b^n)^(1/n : ℝ) + (b^n + c^n)^(1/n : ℝ) + (c^n + a^n)^(1/n : ℝ) < 1 + 2^(1/n : ℝ)/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_powers_l2106_210658


namespace NUMINAMATH_CALUDE_min_side_length_square_l2106_210681

theorem min_side_length_square (s : ℝ) : s ≥ 0 → s ^ 2 ≥ 900 → s ≥ 30 := by
  sorry

end NUMINAMATH_CALUDE_min_side_length_square_l2106_210681


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2106_210699

theorem polynomial_multiplication (a b : ℝ) : (2*a + 3*b) * (2*a - b) = 4*a^2 + 4*a*b - 3*b^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2106_210699


namespace NUMINAMATH_CALUDE_apple_production_theorem_l2106_210687

/-- The apple production problem -/
theorem apple_production_theorem :
  let first_year : ℕ := 40
  let second_year : ℕ := 2 * first_year + 8
  let third_year : ℕ := (3 * second_year) / 4
  first_year + second_year + third_year = 194 := by
sorry

end NUMINAMATH_CALUDE_apple_production_theorem_l2106_210687


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2106_210623

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+,
  (5 : ℕ)^(x.val) - (3 : ℕ)^(y.val) = (z.val)^2 →
  x = 2 ∧ y = 2 ∧ z = 4 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2106_210623


namespace NUMINAMATH_CALUDE_quadratic_solution_and_gcd_sum_l2106_210614

theorem quadratic_solution_and_gcd_sum : ∃ m n p : ℕ,
  (∀ x : ℝ, x * (4 * x - 5) = 7 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) ∧
  Nat.gcd m (Nat.gcd n p) = 1 ∧
  m + n + p = 150 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_and_gcd_sum_l2106_210614


namespace NUMINAMATH_CALUDE_system_solution_l2106_210619

theorem system_solution :
  ∀ x y z : ℝ,
  (x^2 + y^2 + 25*z^2 = 6*x*z + 8*y*z) ∧
  (3*x^2 + 2*y^2 + z^2 = 240) →
  ((x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2106_210619


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2106_210676

theorem quadratic_inequality_solution_set (x : ℝ) :
  x^2 - 3*x + 2 > 0 ↔ x < 1 ∨ x > 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2106_210676


namespace NUMINAMATH_CALUDE_regression_satisfies_negative_correlation_l2106_210697

/-- Represents the regression equation for sales volume based on selling price -/
def regression_equation (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the correlation between sales volume and selling price -/
def negative_correlation (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂

theorem regression_satisfies_negative_correlation :
  negative_correlation regression_equation :=
sorry

end NUMINAMATH_CALUDE_regression_satisfies_negative_correlation_l2106_210697


namespace NUMINAMATH_CALUDE_math_problem_proof_l2106_210671

theorem math_problem_proof (first_answer : ℕ) (second_answer : ℕ) (third_answer : ℕ) : 
  first_answer = 600 →
  second_answer = 2 * first_answer →
  first_answer + second_answer + third_answer = 3200 →
  first_answer + second_answer - third_answer = 400 := by
  sorry

end NUMINAMATH_CALUDE_math_problem_proof_l2106_210671


namespace NUMINAMATH_CALUDE_infinite_solutions_abs_value_equation_l2106_210638

theorem infinite_solutions_abs_value_equation (a : ℝ) :
  (∀ x : ℝ, |x - 2| = a * x - 2) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_abs_value_equation_l2106_210638


namespace NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l2106_210686

def total_students : ℕ := 5
def students_to_select : ℕ := 3

theorem probability_of_selecting_A_and_B :
  (Nat.choose (total_students - 2) (students_to_select - 2)) / 
  (Nat.choose total_students students_to_select) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l2106_210686


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_square_digits_l2106_210693

/-- A function that checks if a number has all different digits -/
def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- A function that checks if a number is divisible by the square of each of its digits -/
def divisible_by_square_of_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % (d * d) = 0

theorem smallest_four_digit_divisible_by_square_digits :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 →
    has_different_digits n →
    divisible_by_square_of_digits n →
    2268 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_square_digits_l2106_210693


namespace NUMINAMATH_CALUDE_total_informed_is_258_l2106_210645

/-- Represents the number of people in the initial group -/
def initial_group : ℕ := 6

/-- Represents the number of people each person calls -/
def calls_per_person : ℕ := 6

/-- Calculates the total number of people informed after two rounds of calls -/
def total_informed : ℕ := 
  initial_group + 
  (initial_group * calls_per_person) + 
  (initial_group * calls_per_person * calls_per_person)

/-- Theorem stating that the total number of people informed is 258 -/
theorem total_informed_is_258 : total_informed = 258 := by
  sorry

end NUMINAMATH_CALUDE_total_informed_is_258_l2106_210645
