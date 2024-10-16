import Mathlib

namespace NUMINAMATH_CALUDE_surface_area_of_modified_structure_l743_74368

/-- Represents the dimensions of the large cube -/
def large_cube_dim : ℕ := 12

/-- Represents the dimensions of the small cubes -/
def small_cube_dim : ℕ := 2

/-- The total number of small cubes in the original structure -/
def total_small_cubes : ℕ := 64

/-- The number of small cubes removed from the structure -/
def removed_cubes : ℕ := 7

/-- The surface area of a single 2x2x2 cube before modification -/
def small_cube_surface_area : ℕ := 24

/-- The additional surface area exposed on each small cube after modification -/
def additional_exposed_area : ℕ := 6

/-- The surface area of a modified small cube -/
def modified_small_cube_area : ℕ := small_cube_surface_area + additional_exposed_area

/-- The theorem to be proved -/
theorem surface_area_of_modified_structure :
  (total_small_cubes - removed_cubes) * modified_small_cube_area = 1710 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_modified_structure_l743_74368


namespace NUMINAMATH_CALUDE_inequality_of_squares_existence_of_positive_l743_74399

theorem inequality_of_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

theorem existence_of_positive (x y z : ℝ) :
  let a := x^2 - 2*y + Real.pi/2
  let b := y^2 - 2*z + Real.pi/3
  let c := z^2 - 2*x + Real.pi/6
  (a > 0) ∨ (b > 0) ∨ (c > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_squares_existence_of_positive_l743_74399


namespace NUMINAMATH_CALUDE_graduation_ceremony_arrangements_l743_74311

/-- The number of events in the graduation ceremony program -/
def total_events : ℕ := 6

/-- The number of positions event A can be placed in -/
def a_positions : ℕ := 3

/-- The number of events that must be scheduled together -/
def together_events : ℕ := 2

/-- The number of possible arrangements for the graduation ceremony program -/
def possible_arrangements : ℕ := 120

/-- Theorem stating that the number of possible arrangements is correct -/
theorem graduation_ceremony_arrangements :
  (total_events = 6) →
  (a_positions = 3) →
  (together_events = 2) →
  (possible_arrangements = 120) := by
  sorry

end NUMINAMATH_CALUDE_graduation_ceremony_arrangements_l743_74311


namespace NUMINAMATH_CALUDE_wedding_guests_fraction_l743_74312

theorem wedding_guests_fraction (total_guests : ℚ) : 
  let children_fraction : ℚ := 1/8
  let adult_fraction : ℚ := 1 - children_fraction
  let men_fraction_of_adults : ℚ := 3/7
  let women_fraction_of_adults : ℚ := 1 - men_fraction_of_adults
  let adult_women_fraction : ℚ := adult_fraction * women_fraction_of_adults
  adult_women_fraction = 1/2 := by
sorry

end NUMINAMATH_CALUDE_wedding_guests_fraction_l743_74312


namespace NUMINAMATH_CALUDE_paco_cookies_left_l743_74354

/-- The number of cookies Paco has left -/
def cookies_left (initial : ℕ) (given_away : ℕ) (eaten : ℕ) : ℕ :=
  initial - given_away - eaten

/-- Theorem stating that Paco has 12 cookies left -/
theorem paco_cookies_left :
  cookies_left 36 14 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_left_l743_74354


namespace NUMINAMATH_CALUDE_total_profit_calculation_total_profit_is_630_l743_74348

/-- Calculates the total profit given investment conditions and A's share of profit -/
theorem total_profit_calculation (a_initial : ℕ) (b_initial : ℕ) (a_withdrawal : ℕ) (b_addition : ℕ) (months : ℕ) (a_share : ℕ) : ℕ :=
  let a_investment_months := a_initial * 8 + (a_initial - a_withdrawal) * 4
  let b_investment_months := b_initial * 8 + (b_initial + b_addition) * 4
  let total_ratio_parts := a_investment_months + b_investment_months
  let total_profit := a_share * total_ratio_parts / a_investment_months
  total_profit

/-- The total profit at the end of the year is 630 Rs -/
theorem total_profit_is_630 :
  total_profit_calculation 3000 4000 1000 1000 12 240 = 630 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_calculation_total_profit_is_630_l743_74348


namespace NUMINAMATH_CALUDE_euler_disproof_l743_74384

theorem euler_disproof : 133^4 + 110^4 + 56^4 = 143^4 := by
  sorry

end NUMINAMATH_CALUDE_euler_disproof_l743_74384


namespace NUMINAMATH_CALUDE_factorization_proof_l743_74363

theorem factorization_proof (x : ℝ) : 2*x^3 - 8*x^2 + 8*x = 2*x*(x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l743_74363


namespace NUMINAMATH_CALUDE_smallest_b_for_even_polynomial_l743_74389

theorem smallest_b_for_even_polynomial : ∃ (b : ℕ+), 
  (∀ (x : ℤ), ∃ (k : ℤ), x^4 + (b : ℤ)^3 + (b : ℤ)^2 = 2 * k) ∧ 
  (∀ (b' : ℕ+), b' < b → ∃ (x : ℤ), ∀ (k : ℤ), x^4 + (b' : ℤ)^3 + (b' : ℤ)^2 ≠ 2 * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_even_polynomial_l743_74389


namespace NUMINAMATH_CALUDE_millet_majority_on_tuesday_l743_74340

/-- Represents the proportion of millet seeds remaining after birds eat -/
def milletRemaining : ℝ := 0.7

/-- Calculates the amount of millet seeds in the feeder after n days -/
def milletAmount (n : ℕ) : ℝ := 1 - milletRemaining ^ n

/-- The day when more than half the seeds are millet -/
def milletMajorityDay : ℕ := 2

theorem millet_majority_on_tuesday :
  milletAmount milletMajorityDay > 0.5 ∧
  ∀ k : ℕ, k < milletMajorityDay → milletAmount k ≤ 0.5 :=
sorry

end NUMINAMATH_CALUDE_millet_majority_on_tuesday_l743_74340


namespace NUMINAMATH_CALUDE_dividend_calculation_l743_74371

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (hq : quotient = 120) 
  (hd : divisor = 456) 
  (hr : remainder = 333) : 
  divisor * quotient + remainder = 55053 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l743_74371


namespace NUMINAMATH_CALUDE_employee_payment_proof_l743_74346

/-- The weekly payment for employee B -/
def payment_B : ℝ := 250

/-- The weekly payment for employee A -/
def payment_A : ℝ := 1.2 * payment_B

/-- The total weekly payment for both employees -/
def total_payment : ℝ := 550

theorem employee_payment_proof :
  payment_B + payment_A = total_payment ∧ payment_A = 1.2 * payment_B → payment_B = 250 :=
by sorry

end NUMINAMATH_CALUDE_employee_payment_proof_l743_74346


namespace NUMINAMATH_CALUDE_toms_vaccines_l743_74323

theorem toms_vaccines (total_payment : ℕ) (trip_cost : ℕ) (vaccine_cost : ℕ) (doctor_visit : ℕ) 
  (insurance_coverage : ℚ) :
  total_payment = 1340 →
  trip_cost = 1200 →
  vaccine_cost = 45 →
  doctor_visit = 250 →
  insurance_coverage = 4/5 →
  ∃ (num_vaccines : ℕ), 
    (total_payment : ℚ) = trip_cost + (1 - insurance_coverage) * (doctor_visit + num_vaccines * vaccine_cost) ∧
    num_vaccines = 10 :=
by sorry

end NUMINAMATH_CALUDE_toms_vaccines_l743_74323


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l743_74396

/-- Proves that the conversion from spherical coordinates (10, 3π/4, π/4) to 
    rectangular coordinates results in (-5, 5, 5√2) -/
theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 10
  let θ : ℝ := 3 * Real.pi / 4
  let φ : ℝ := Real.pi / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x = -5) ∧ (y = 5) ∧ (z = 5 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l743_74396


namespace NUMINAMATH_CALUDE_group_size_is_correct_l743_74337

/-- The number of people in a group where:
  1. The average weight increases by 2.5 kg when a new person joins.
  2. The person being replaced weighs 45 kg.
  3. The new person weighs 65 kg.
-/
def group_size : ℕ := 8

/-- The weight of the person being replaced -/
def original_weight : ℝ := 45

/-- The weight of the new person joining the group -/
def new_weight : ℝ := 65

/-- The increase in average weight when the new person joins -/
def average_increase : ℝ := 2.5

theorem group_size_is_correct : 
  (new_weight - original_weight) = (average_increase * group_size) :=
sorry

end NUMINAMATH_CALUDE_group_size_is_correct_l743_74337


namespace NUMINAMATH_CALUDE_intersection_when_a_is_4_range_of_a_for_subset_l743_74355

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 3) * (x - 3 * a - 5) < 0}
def B : Set ℝ := {x | -x^2 + 5*x + 14 > 0}

-- Part 1
theorem intersection_when_a_is_4 :
  A 4 ∩ B = {x | 3 < x ∧ x < 7} :=
sorry

-- Part 2
theorem range_of_a_for_subset :
  {a : ℝ | A a ⊆ B} = {a : ℝ | -7/3 ≤ a ∧ a ≤ 2/3} :=
sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_4_range_of_a_for_subset_l743_74355


namespace NUMINAMATH_CALUDE_circle_line_intersection_l743_74357

/-- Given a circle C: (x-a)² + (y-2)² = 4 (a > 0) and a line l: x - y + 3 = 0,
    if the length of the chord cut by line l on circle C is 2√3,
    then a = √2 - 1 -/
theorem circle_line_intersection (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, (x - a)^2 + (y - 2)^2 = 4 → x - y + 3 = 0 →
    ∃ x' y' : ℝ, (x' - a)^2 + (y' - 2)^2 = 4 ∧ x' - y' + 3 = 0 ∧
    ((x - x')^2 + (y - y')^2)^(1/2) = 2 * Real.sqrt 3) →
  a = Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l743_74357


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l743_74315

theorem circle_diameter_from_area (A : Real) (d : Real) :
  A = 225 * Real.pi → d = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l743_74315


namespace NUMINAMATH_CALUDE_ann_boxes_sold_l743_74320

theorem ann_boxes_sold (n : ℕ) (mark_sold ann_sold : ℕ) : 
  n = 12 →
  mark_sold = n - 11 →
  ann_sold < n →
  mark_sold ≥ 1 →
  ann_sold ≥ 1 →
  mark_sold + ann_sold < n →
  ann_sold = n - 2 :=
by sorry

end NUMINAMATH_CALUDE_ann_boxes_sold_l743_74320


namespace NUMINAMATH_CALUDE_yellow_shirt_pairs_l743_74359

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) : 
  blue_students = 57 →
  yellow_students = 75 →
  total_students = 132 →
  total_pairs = 66 →
  blue_blue_pairs = 23 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_shirt_pairs_l743_74359


namespace NUMINAMATH_CALUDE_b_97_mod_81_l743_74362

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b_97_mod_81 : b 97 ≡ 52 [MOD 81] := by sorry

end NUMINAMATH_CALUDE_b_97_mod_81_l743_74362


namespace NUMINAMATH_CALUDE_max_coins_identifiable_l743_74392

/-- The maximum number of coins that can be tested to identify one counterfeit (lighter) coin -/
def max_coins (n : ℕ) : ℕ := 2 * n^2 + 1

/-- A balance scale used for weighing coins -/
structure BalanceScale :=
  (weigh : ℕ → ℕ → Bool)

/-- Represents the process of identifying a counterfeit coin -/
structure CoinIdentification :=
  (n : ℕ)  -- Number of weighings allowed
  (coins : ℕ)  -- Total number of coins
  (scale : BalanceScale)
  (max_weighings_per_coin : ℕ := 2)  -- Maximum number of times each coin can be weighed

/-- Theorem stating the maximum number of coins that can be tested -/
theorem max_coins_identifiable (ci : CoinIdentification) :
  ci.coins ≤ max_coins ci.n ↔
  ∃ (strategy : Unit), true  -- Placeholder for the existence of a valid identification strategy
:= by sorry

end NUMINAMATH_CALUDE_max_coins_identifiable_l743_74392


namespace NUMINAMATH_CALUDE_tray_height_l743_74361

theorem tray_height (side_length : ℝ) (corner_distance : ℝ) (cut_angle : ℝ) 
  (h1 : side_length = 150)
  (h2 : corner_distance = 5)
  (h3 : cut_angle = 45) : 
  let tray_height := corner_distance * Real.sqrt 2 * Real.sin (cut_angle * π / 180)
  tray_height = 5 := by sorry

end NUMINAMATH_CALUDE_tray_height_l743_74361


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l743_74365

theorem fraction_sum_equation (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 5 * b) / (b + 5 * a) = 3) : a / b = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l743_74365


namespace NUMINAMATH_CALUDE_subset_union_equality_l743_74341

theorem subset_union_equality (n : ℕ) (A : Fin (n + 1) → Set (Fin n)) 
  (h : ∀ i, (A i).Nonempty) :
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧
  (⋃ (i ∈ I), A i) = (⋃ (j ∈ J), A j) := by
sorry

end NUMINAMATH_CALUDE_subset_union_equality_l743_74341


namespace NUMINAMATH_CALUDE_jasons_music_store_spending_l743_74309

/-- The problem of calculating Jason's total spending at the music store -/
theorem jasons_music_store_spending
  (flute_cost : ℝ)
  (music_stand_cost : ℝ)
  (song_book_cost : ℝ)
  (h1 : flute_cost = 142.46)
  (h2 : music_stand_cost = 8.89)
  (h3 : song_book_cost = 7.00) :
  flute_cost + music_stand_cost + song_book_cost = 158.35 := by
  sorry

end NUMINAMATH_CALUDE_jasons_music_store_spending_l743_74309


namespace NUMINAMATH_CALUDE_freds_baseball_cards_l743_74388

/-- Fred's baseball card problem -/
theorem freds_baseball_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 40 → cards_bought = 22 → remaining_cards = initial_cards - cards_bought → 
  remaining_cards = 18 := by
  sorry

end NUMINAMATH_CALUDE_freds_baseball_cards_l743_74388


namespace NUMINAMATH_CALUDE_no_valid_assignment_l743_74335

/-- Represents a vertex of the hexagon or its center -/
inductive Vertex
| A | B | C | D | E | F | G

/-- Represents a triangle formed by the center and two adjacent vertices -/
structure Triangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

/-- The set of all triangles in the hexagon -/
def hexagonTriangles : List Triangle := [
  ⟨Vertex.A, Vertex.B, Vertex.G⟩,
  ⟨Vertex.B, Vertex.C, Vertex.G⟩,
  ⟨Vertex.C, Vertex.D, Vertex.G⟩,
  ⟨Vertex.D, Vertex.E, Vertex.G⟩,
  ⟨Vertex.E, Vertex.F, Vertex.G⟩,
  ⟨Vertex.F, Vertex.A, Vertex.G⟩
]

/-- A function that assigns an integer to each vertex -/
def VertexAssignment := Vertex → Int

/-- Checks if the integers assigned to a triangle are in ascending order clockwise -/
def isAscendingClockwise (assignment : VertexAssignment) (t : Triangle) : Prop :=
  assignment t.v1 < assignment t.v2 ∧ assignment t.v2 < assignment t.v3

/-- The main theorem stating that no valid assignment exists -/
theorem no_valid_assignment :
  ¬∃ (assignment : VertexAssignment),
    (∀ v1 v2 : Vertex, v1 ≠ v2 → assignment v1 ≠ assignment v2) ∧
    (∀ t ∈ hexagonTriangles, isAscendingClockwise assignment t) :=
sorry


end NUMINAMATH_CALUDE_no_valid_assignment_l743_74335


namespace NUMINAMATH_CALUDE_third_test_score_l743_74391

def test1 : ℝ := 85
def test2 : ℝ := 79
def test4 : ℝ := 84
def test5 : ℝ := 85
def targetAverage : ℝ := 85
def numTests : ℕ := 5

theorem third_test_score (test3 : ℝ) : 
  (test1 + test2 + test3 + test4 + test5) / numTests = targetAverage → 
  test3 = 92 := by
sorry

end NUMINAMATH_CALUDE_third_test_score_l743_74391


namespace NUMINAMATH_CALUDE_min_distance_curve_line_l743_74382

/-- The minimum distance between a point on y = 2ln(x) and a point on y = 2x + 3 is √5 -/
theorem min_distance_curve_line : 
  let curve := (fun x : ℝ => 2 * Real.log x)
  let line := (fun x : ℝ => 2 * x + 3)
  ∃ (M N : ℝ × ℝ), 
    (M.2 = curve M.1) ∧ 
    (N.2 = line N.1) ∧
    (∀ (P Q : ℝ × ℝ), P.2 = curve P.1 → Q.2 = line Q.1 → 
      Real.sqrt 5 ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_min_distance_curve_line_l743_74382


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l743_74329

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 1600 10 = 160 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l743_74329


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l743_74331

theorem negation_of_existence_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l743_74331


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l743_74356

theorem max_value_on_ellipse :
  ∃ (max : ℝ),
    (∀ x y : ℝ, (y^2 / 4 + x^2 / 3 = 1) → 2*x + y ≤ max) ∧
    (∃ x y : ℝ, (y^2 / 4 + x^2 / 3 = 1) ∧ 2*x + y = max) ∧
    max = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l743_74356


namespace NUMINAMATH_CALUDE_two_true_propositions_l743_74367

theorem two_true_propositions (p q : Prop) (h : p ∧ q) :
  (p ∨ q) ∧ p ∧ ¬(¬q) ∧ ¬((¬p) ∨ (¬q)) :=
by sorry

end NUMINAMATH_CALUDE_two_true_propositions_l743_74367


namespace NUMINAMATH_CALUDE_fiscal_revenue_scientific_notation_l743_74381

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

/-- Rounds a ScientificNotation to a specified number of significant figures -/
def roundToSignificantFigures (sn : ScientificNotation) (figures : ℕ) : ScientificNotation := sorry

/-- The fiscal revenue in yuan -/
def fiscalRevenue : ℝ := 1073 * 10^9

theorem fiscal_revenue_scientific_notation :
  roundToSignificantFigures (toScientificNotation fiscalRevenue) 2 =
  ScientificNotation.mk 1.07 11 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_fiscal_revenue_scientific_notation_l743_74381


namespace NUMINAMATH_CALUDE_first_hole_depth_l743_74302

/-- Represents the depth of a hole dug by workers. -/
def hole_depth (workers : ℕ) (hours : ℕ) (rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours : ℚ) * rate

/-- The work rate is constant for both holes. -/
def work_rate : ℚ := 1 / 12

theorem first_hole_depth :
  let first_hole := hole_depth 45 8 work_rate
  let second_hole := hole_depth 90 6 work_rate
  second_hole = 45 →
  first_hole = 30 := by sorry

end NUMINAMATH_CALUDE_first_hole_depth_l743_74302


namespace NUMINAMATH_CALUDE_complex_equation_solution_l743_74386

theorem complex_equation_solution (z : ℂ) :
  z * (1 + 3 * Complex.I) = 4 + Complex.I →
  z = 7/10 - 11/10 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l743_74386


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l743_74328

theorem max_value_sqrt_sum (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  Real.sqrt ((2 - a) * (2 - b) * (2 - c)) + Real.sqrt (a * b * c) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l743_74328


namespace NUMINAMATH_CALUDE_sum_of_yellow_and_blue_is_red_l743_74377

theorem sum_of_yellow_and_blue_is_red (a b : ℕ) :
  ∃ m : ℕ, (4 * a + 3) + (4 * b + 2) = 4 * m + 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_yellow_and_blue_is_red_l743_74377


namespace NUMINAMATH_CALUDE_smallest_cube_ending_in_388_l743_74387

def is_cube_ending_in_388 (n : ℕ) : Prop := n^3 % 1000 = 388

theorem smallest_cube_ending_in_388 : 
  (∃ (n : ℕ), is_cube_ending_in_388 n) ∧ 
  (∀ (m : ℕ), m < 16 → ¬is_cube_ending_in_388 m) ∧ 
  is_cube_ending_in_388 16 :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_in_388_l743_74387


namespace NUMINAMATH_CALUDE_douglas_vote_percentage_l743_74397

theorem douglas_vote_percentage (total_voters : ℕ) (x_voters y_voters : ℕ) 
  (douglas_total_votes douglas_x_votes douglas_y_votes : ℕ) :
  x_voters = 2 * y_voters →
  douglas_total_votes = (66 * (x_voters + y_voters)) / 100 →
  douglas_x_votes = (74 * x_voters) / 100 →
  douglas_total_votes = douglas_x_votes + douglas_y_votes →
  (douglas_y_votes * 100) / y_voters = 50 :=
by sorry

end NUMINAMATH_CALUDE_douglas_vote_percentage_l743_74397


namespace NUMINAMATH_CALUDE_problem_solution_l743_74364

theorem problem_solution (a b c d e : ℝ) 
  (eq1 : a - b - c + d = 18)
  (eq2 : a + b - c - d = 6)
  (eq3 : c + d - e = 5) :
  (2 * b - d + e) ^ 3 = 13824 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l743_74364


namespace NUMINAMATH_CALUDE_parallel_line_x_coordinate_l743_74353

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines when two points form a line segment parallel to the y-axis. -/
def parallelToYAxis (p q : Point) : Prop :=
  p.x = q.x

/-- The problem statement -/
theorem parallel_line_x_coordinate 
  (M N : Point)
  (h_parallel : parallelToYAxis M N)
  (h_M : M = ⟨3, -5⟩)
  (h_N : N = ⟨N.x, 2⟩) :
  N.x = 3 := by
  sorry

#check parallel_line_x_coordinate

end NUMINAMATH_CALUDE_parallel_line_x_coordinate_l743_74353


namespace NUMINAMATH_CALUDE_vector_proof_l743_74322

theorem vector_proof (a b : ℝ × ℝ) : 
  b = (1, -2) → 
  (a.1 * b.1 + a.2 * b.2 = -Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) → 
  Real.sqrt (a.1^2 + a.2^2) = 3 * Real.sqrt 5 → 
  a = (-3, 6) := by sorry

end NUMINAMATH_CALUDE_vector_proof_l743_74322


namespace NUMINAMATH_CALUDE_final_surface_area_l743_74352

/-- Represents the cube structure after modifications --/
structure ModifiedCube where
  initialSize : Nat
  smallCubeSize : Nat
  removedCornerCubes : Nat
  removedCentralCube : Nat
  removedCenterUnits : Bool

/-- Calculates the surface area of the modified cube structure --/
def surfaceArea (cube : ModifiedCube) : Nat :=
  let totalSmallCubes := (cube.initialSize / cube.smallCubeSize) ^ 3
  let remainingCubes := totalSmallCubes - cube.removedCornerCubes - cube.removedCentralCube
  let initialSurfaceArea := remainingCubes * (6 * cube.smallCubeSize ^ 2)
  let additionalInternalSurface := if cube.removedCenterUnits then remainingCubes * 6 else 0
  initialSurfaceArea + additionalInternalSurface

/-- The main theorem stating the surface area of the final structure --/
theorem final_surface_area :
  let cube : ModifiedCube := {
    initialSize := 12,
    smallCubeSize := 3,
    removedCornerCubes := 8,
    removedCentralCube := 1,
    removedCenterUnits := true
  }
  surfaceArea cube = 3300 := by
  sorry

end NUMINAMATH_CALUDE_final_surface_area_l743_74352


namespace NUMINAMATH_CALUDE_expression_equality_l743_74305

theorem expression_equality : (2^1501 + 5^1502)^2 - (2^1501 - 5^1502)^2 = 20 * 10^1501 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l743_74305


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l743_74379

theorem absolute_value_inequality (x : ℝ) : 
  |((x + 1) / x)| > ((x + 1) / x) ↔ -1 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l743_74379


namespace NUMINAMATH_CALUDE_jogging_distance_l743_74370

theorem jogging_distance (x t : ℝ) 
  (h1 : (x + 3/4) * (3*t/4) = x * t)
  (h2 : (x - 3/4) * (t + 3) = x * t) :
  x * t = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_jogging_distance_l743_74370


namespace NUMINAMATH_CALUDE_ellipse_existence_in_acute_triangle_l743_74321

/-- Represents an acute triangle -/
structure AcuteTriangle where
  -- Add necessary fields for an acute triangle
  is_acute : Bool

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Orthocenter of a triangle -/
def orthocenter (t : AcuteTriangle) : Point :=
  sorry

/-- Circumcenter of a triangle -/
def circumcenter (t : AcuteTriangle) : Point :=
  sorry

/-- Theorem: For any acute triangle, there exists an ellipse with one focus
    at the orthocenter and the other at the circumcenter of the triangle -/
theorem ellipse_existence_in_acute_triangle (t : AcuteTriangle) :
  ∃ e : Ellipse, e.focus1 = orthocenter t ∧ e.focus2 = circumcenter t :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_existence_in_acute_triangle_l743_74321


namespace NUMINAMATH_CALUDE_percentage_increase_sum_l743_74343

theorem percentage_increase_sum (A B C x y : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A = 120 → B = 110 → C = 100 →
  A = C * (1 + x / 100) →
  B = C * (1 + y / 100) →
  x + y = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_sum_l743_74343


namespace NUMINAMATH_CALUDE_vector_subtraction_magnitude_l743_74347

/-- Given two 2D vectors a and b, where the angle between them is 45°,
    a = (-1, 1), and |b| = 1, prove that |a - 2b| = √2 -/
theorem vector_subtraction_magnitude (a b : ℝ × ℝ) :
  let angle := Real.pi / 4
  a.1 = -1 ∧ a.2 = 1 →
  Real.sqrt (b.1^2 + b.2^2) = 1 →
  a.1 * b.1 + a.2 * b.2 = Real.sqrt 2 / 2 →
  Real.sqrt ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_magnitude_l743_74347


namespace NUMINAMATH_CALUDE_slope_at_one_l743_74313

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem slope_at_one (α : ℝ) :
  (deriv f 1 = α) →
  (Real.cos α / (Real.sin α - 4 * Real.cos α) = -1) :=
by sorry

end NUMINAMATH_CALUDE_slope_at_one_l743_74313


namespace NUMINAMATH_CALUDE_inheritance_calculation_l743_74339

theorem inheritance_calculation (x : ℝ) : 
  let after_charity := 0.95 * x
  let federal_tax := 0.25 * after_charity
  let after_federal := after_charity - federal_tax
  let state_tax := 0.12 * after_federal
  federal_tax + state_tax = 15000 → x = 46400 := by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l743_74339


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_six_l743_74349

/-- The rational equation (2x + m) / (x - 3) = 1 has no solution with respect to x if and only if m = -6 -/
theorem no_solution_iff_m_eq_neg_six (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (2 * x + m) / (x - 3) ≠ 1) ↔ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_six_l743_74349


namespace NUMINAMATH_CALUDE_average_weight_problem_l743_74318

theorem average_weight_problem (num_group1 : ℕ) (num_group2 : ℕ) (avg_weight_group2 : ℝ) (avg_weight_total : ℝ) :
  num_group1 = 24 →
  num_group2 = 8 →
  avg_weight_group2 = 45.15 →
  avg_weight_total = 48.975 →
  let total_num := num_group1 + num_group2
  let total_weight := total_num * avg_weight_total
  let weight_group2 := num_group2 * avg_weight_group2
  let weight_group1 := total_weight - weight_group2
  let avg_weight_group1 := weight_group1 / num_group1
  avg_weight_group1 = 50.25 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l743_74318


namespace NUMINAMATH_CALUDE_spring_excursion_participants_l743_74398

theorem spring_excursion_participants :
  let water_students : ℕ := 80
  let fruit_students : ℕ := 70
  let neither_students : ℕ := 6
  let both_students : ℕ := (water_students + fruit_students - neither_students) / 2
  let total_participants : ℕ := both_students * 2
  total_participants = 104 :=
by sorry

end NUMINAMATH_CALUDE_spring_excursion_participants_l743_74398


namespace NUMINAMATH_CALUDE_horner_operations_for_f_l743_74325

/-- Represents a polynomial of degree 4 -/
structure Polynomial4 where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Counts the number of operations in Horner's method for a polynomial of degree 4 -/
def hornerOperations (p : Polynomial4) : ℕ × ℕ :=
  sorry

/-- The specific polynomial f(x) = 2x^4 + 3x^3 - 2x^2 + 4x - 6 -/
def f : Polynomial4 := {
  a := 2
  b := 3
  c := -2
  d := 4
  e := -6
}

theorem horner_operations_for_f :
  hornerOperations f = (4, 4) := by sorry

end NUMINAMATH_CALUDE_horner_operations_for_f_l743_74325


namespace NUMINAMATH_CALUDE_tangent_circles_radii_l743_74358

/-- Two externally touching circles with a line passing through their point of tangency -/
structure TangentCircles where
  /-- Center of the first circle -/
  O₁ : ℝ × ℝ
  /-- Center of the second circle -/
  O₂ : ℝ × ℝ
  /-- Point of tangency -/
  K : ℝ × ℝ
  /-- Radius of the first circle -/
  r₁ : ℝ
  /-- Radius of the second circle -/
  r₂ : ℝ
  /-- Distance between centers is 36 -/
  centers_distance : dist O₁ O₂ = 36
  /-- Circles touch externally -/
  touch_externally : dist O₁ K + dist K O₂ = dist O₁ O₂
  /-- K is on both circles -/
  K_on_circles : dist O₁ K = r₁ ∧ dist K O₂ = r₂

/-- Chords formed by the line passing through the point of tangency -/
structure Chords (tc : TangentCircles) where
  /-- Point A on the first circle -/
  A : ℝ × ℝ
  /-- Point B on the second circle -/
  B : ℝ × ℝ
  /-- A and B are on their respective circles -/
  on_circles : dist tc.O₁ A = tc.r₁ ∧ dist tc.O₂ B = tc.r₂
  /-- K is between A and B -/
  K_between : dist A tc.K + dist tc.K B = dist A B
  /-- Ratio of chord lengths is 13/5 -/
  chord_ratio : dist tc.K B = (13/5) * dist A tc.K

/-- Theorem: The radii of the circles are 10 and 26 -/
theorem tangent_circles_radii (tc : TangentCircles) (c : Chords tc) :
  tc.r₁ = 10 ∧ tc.r₂ = 26 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_radii_l743_74358


namespace NUMINAMATH_CALUDE_child_workers_count_l743_74375

/-- Represents the number of workers of each type and their daily wages --/
structure WorkforceData where
  male_workers : ℕ
  female_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the number of child workers given the workforce data --/
def calculate_child_workers (data : WorkforceData) : ℕ :=
  let total_workers := data.male_workers + data.female_workers
  let total_wage := data.male_workers * data.male_wage + data.female_workers * data.female_wage
  let x := (data.average_wage * total_workers - total_wage) / (data.average_wage - data.child_wage)
  x

/-- Theorem stating that the number of child workers is 5 given the specific workforce data --/
theorem child_workers_count (data : WorkforceData) 
  (h1 : data.male_workers = 20)
  (h2 : data.female_workers = 15)
  (h3 : data.male_wage = 35)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 26) :
  calculate_child_workers data = 5 := by
  sorry

end NUMINAMATH_CALUDE_child_workers_count_l743_74375


namespace NUMINAMATH_CALUDE_parabola_shift_l743_74308

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 3

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := original_parabola (x + 2) + 2

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = 3 * x^2 + 6 * x - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l743_74308


namespace NUMINAMATH_CALUDE_smallest_x_value_l743_74327

theorem smallest_x_value (x : ℝ) : 
  (3 * x^2 + 36 * x - 72 = x * (x + 20) + 8) → x ≥ -10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l743_74327


namespace NUMINAMATH_CALUDE_physics_class_size_l743_74326

/-- Proves that the number of students in the physics class is 42 --/
theorem physics_class_size :
  ∀ (total_students : ℕ) 
    (math_only : ℕ) 
    (physics_only : ℕ) 
    (both : ℕ),
  total_students = 53 →
  math_only + physics_only + both = total_students →
  physics_only + both = 2 * (math_only + both) →
  both = 10 →
  physics_only + both = 42 :=
by
  sorry

#check physics_class_size

end NUMINAMATH_CALUDE_physics_class_size_l743_74326


namespace NUMINAMATH_CALUDE_pencil_color_fractions_l743_74307

theorem pencil_color_fractions (L : ℝ) (h1 : L = 9.333333333333332) : 
  let black_fraction : ℝ := 1/8
  let remaining_after_black : ℝ := L - black_fraction * L
  let blue_fraction_of_remaining : ℝ := 7/12
  let white_fraction_of_remaining : ℝ := 1 - blue_fraction_of_remaining
  white_fraction_of_remaining = 5/12 := by
sorry

end NUMINAMATH_CALUDE_pencil_color_fractions_l743_74307


namespace NUMINAMATH_CALUDE_nathan_tokens_used_l743_74395

/-- The number of times Nathan played air hockey -/
def air_hockey_games : ℕ := 2

/-- The number of times Nathan played basketball -/
def basketball_games : ℕ := 4

/-- The cost in tokens for each game -/
def tokens_per_game : ℕ := 3

/-- The total number of tokens Nathan used -/
def total_tokens : ℕ := air_hockey_games * tokens_per_game + basketball_games * tokens_per_game

theorem nathan_tokens_used : total_tokens = 18 := by
  sorry

end NUMINAMATH_CALUDE_nathan_tokens_used_l743_74395


namespace NUMINAMATH_CALUDE_room_population_change_l743_74373

theorem room_population_change (initial_men initial_women : ℕ) : 
  initial_men / initial_women = 4 / 5 →
  ∃ (current_women : ℕ),
    initial_men + 2 = 14 ∧
    current_women = 2 * (initial_women - 3) ∧
    current_women = 24 :=
by sorry

end NUMINAMATH_CALUDE_room_population_change_l743_74373


namespace NUMINAMATH_CALUDE_polynomial_division_l743_74372

def P (x : ℝ) : ℝ := x^6 - 6*x^4 - 4*x^3 + 9*x^2 + 12*x + 4

def Q (x : ℝ) : ℝ := x^4 + x^3 - 3*x^2 - 5*x - 2

def R (x : ℝ) : ℝ := x^2 - x - 2

theorem polynomial_division :
  ∀ x : ℝ, P x = Q x * R x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_l743_74372


namespace NUMINAMATH_CALUDE_system_solution_exists_l743_74301

theorem system_solution_exists : ∃ (x y z : ℝ),
  (2 * x + 3 * y + z = 13) ∧
  (4 * x^2 + 9 * y^2 + z^2 - 2 * x + 15 * y + 3 * z = 82) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_l743_74301


namespace NUMINAMATH_CALUDE_multiply_by_six_l743_74374

theorem multiply_by_six (x : ℚ) (h : x / 11 = 2) : 6 * x = 132 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_six_l743_74374


namespace NUMINAMATH_CALUDE_negation_equivalence_l743_74324

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l743_74324


namespace NUMINAMATH_CALUDE_shawn_score_shawn_score_is_six_l743_74338

theorem shawn_score (points_per_basket : ℕ) (matthew_points : ℕ) (total_baskets : ℕ) : ℕ :=
  let matthew_baskets := matthew_points / points_per_basket
  let shawn_baskets := total_baskets - matthew_baskets
  let shawn_points := shawn_baskets * points_per_basket
  shawn_points

theorem shawn_score_is_six :
  shawn_score 3 9 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_shawn_score_shawn_score_is_six_l743_74338


namespace NUMINAMATH_CALUDE_min_value_theorem_l743_74345

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 3) :
  (1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x)) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l743_74345


namespace NUMINAMATH_CALUDE_angle_counterexample_l743_74378

theorem angle_counterexample : ∃ (angle1 angle2 : ℝ), 
  angle1 + angle2 = 90 ∧ angle1 = angle2 := by
  sorry

end NUMINAMATH_CALUDE_angle_counterexample_l743_74378


namespace NUMINAMATH_CALUDE_train_crossing_time_l743_74393

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_pass_man : ℝ) 
  (h1 : train_length = 186)
  (h2 : platform_length = 279)
  (h3 : time_pass_man = 8) :
  (train_length + platform_length) / (train_length / time_pass_man) = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l743_74393


namespace NUMINAMATH_CALUDE_range_of_a_l743_74336

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x + 2| ≤ 3) → -5 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l743_74336


namespace NUMINAMATH_CALUDE_point_satisfies_inequality_l743_74314

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The inequality function -/
def inequality (p : Point) : ℝ :=
  (p.x + 2*p.y - 1) * (p.x - p.y + 3)

/-- Theorem stating that the point (0,2) satisfies the inequality -/
theorem point_satisfies_inequality : 
  let p : Point := ⟨0, 2⟩
  inequality p > 0 := by
  sorry


end NUMINAMATH_CALUDE_point_satisfies_inequality_l743_74314


namespace NUMINAMATH_CALUDE_replaced_crew_weight_l743_74316

/-- Proves that the replaced crew member weighs 40 kg given the conditions of the problem -/
theorem replaced_crew_weight (n : ℕ) (old_avg new_avg new_weight : ℝ) :
  n = 20 ∧
  new_avg = old_avg + 2 ∧
  new_weight = 80 →
  n * new_avg - (n - 1) * old_avg = 40 :=
by sorry

end NUMINAMATH_CALUDE_replaced_crew_weight_l743_74316


namespace NUMINAMATH_CALUDE_algorithm_design_properties_algorithm_not_endless_algorithm_not_unique_correct_statement_about_algorithms_l743_74366

/-- Represents the properties of an algorithm -/
structure Algorithm where
  finite : Bool
  clearlyDefined : Bool
  nonUnique : Bool
  simple : Bool
  convenient : Bool
  operable : Bool

/-- Defines the correct properties of an algorithm according to computer science -/
def correctAlgorithmProperties : Algorithm :=
  { finite := true
  , clearlyDefined := true
  , nonUnique := true
  , simple := true
  , convenient := true
  , operable := true }

/-- Theorem stating that algorithms should be designed to be simple, convenient, and operable -/
theorem algorithm_design_properties :
  (a : Algorithm) → a = correctAlgorithmProperties → a.simple ∧ a.convenient ∧ a.operable :=
by sorry

/-- Theorem stating that an algorithm cannot run endlessly -/
theorem algorithm_not_endless :
  (a : Algorithm) → a = correctAlgorithmProperties → a.finite :=
by sorry

/-- Theorem stating that there can be multiple algorithms for a task -/
theorem algorithm_not_unique :
  (a : Algorithm) → a = correctAlgorithmProperties → a.nonUnique :=
by sorry

/-- Main theorem proving that the statement about algorithm design properties is correct -/
theorem correct_statement_about_algorithms :
  ∃ (a : Algorithm), a = correctAlgorithmProperties ∧
    (a.simple ∧ a.convenient ∧ a.operable) ∧
    a.finite ∧
    a.nonUnique :=
by sorry

end NUMINAMATH_CALUDE_algorithm_design_properties_algorithm_not_endless_algorithm_not_unique_correct_statement_about_algorithms_l743_74366


namespace NUMINAMATH_CALUDE_parallel_transitive_perpendicular_to_plane_parallel_l743_74380

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicularity relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Axiom for transitivity of parallelism
axiom parallel_trans {a b c : Line} : parallel a b → parallel b c → parallel a c

-- Axiom for perpendicular lines to a plane being parallel
axiom perpendicular_parallel {a b : Line} {γ : Plane} : 
  perpendicular a γ → perpendicular b γ → parallel a b

-- Theorem 1: If a∥b and b∥c, then a∥c
theorem parallel_transitive {a b c : Line} : 
  parallel a b → parallel b c → parallel a c :=
by sorry

-- Theorem 2: If a⊥γ and b⊥γ, then a∥b
theorem perpendicular_to_plane_parallel {a b : Line} {γ : Plane} : 
  perpendicular a γ → perpendicular b γ → parallel a b :=
by sorry

end NUMINAMATH_CALUDE_parallel_transitive_perpendicular_to_plane_parallel_l743_74380


namespace NUMINAMATH_CALUDE_triangle_properties_l743_74360

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  median_CM : ℝ → ℝ → ℝ
  altitude_BH : ℝ → ℝ → ℝ

/-- The given triangle satisfies the problem conditions -/
def given_triangle : Triangle where
  A := (5, 1)
  B := sorry
  C := sorry
  median_CM := λ x y ↦ 2*x - y - 5
  altitude_BH := λ x y ↦ x - 2*y - 5

theorem triangle_properties (t : Triangle) (h : t = given_triangle) : 
  t.C = (4, 3) ∧ 
  (λ x y ↦ 6*x - 5*y - 9) = (λ x y ↦ 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l743_74360


namespace NUMINAMATH_CALUDE_buratino_coins_impossibility_l743_74319

theorem buratino_coins_impossibility : ¬ ∃ (n : ℕ), 303 + 6 * n = 456 := by sorry

end NUMINAMATH_CALUDE_buratino_coins_impossibility_l743_74319


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l743_74300

/-- Represents the repeating decimal 0.37246̄ as a rational number -/
def repeating_decimal : ℚ := 37245 / 99900

/-- Theorem stating that the repeating decimal 0.37246̄ is equal to 37245/99900 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 37245 / 99900 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l743_74300


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l743_74344

/-- Given polynomial functions f, g, h, and j, prove their sum equals 3x^2 + x - 2 -/
theorem sum_of_polynomials (x : ℝ) : 
  let f := fun (x : ℝ) => 2 * x^2 - 4 * x + 3
  let g := fun (x : ℝ) => -3 * x^2 + 7 * x - 6
  let h := fun (x : ℝ) => 3 * x^2 - 3 * x + 2
  let j := fun (x : ℝ) => x^2 + x - 1
  f x + g x + h x + j x = 3 * x^2 + x - 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l743_74344


namespace NUMINAMATH_CALUDE_nth_roots_of_unity_real_roots_l743_74317

theorem nth_roots_of_unity_real_roots (n : ℕ) (h : n > 0) :
  ¬ (∀ z : ℂ, z^n = 1 → (z.re = 1 ∧ z.im = 0)) :=
sorry

end NUMINAMATH_CALUDE_nth_roots_of_unity_real_roots_l743_74317


namespace NUMINAMATH_CALUDE_exponential_function_property_l743_74306

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_property (a : ℝ) (b : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (∀ x ∈ Set.Icc (-2) 1, f a x ≤ 4) →
  (∀ x ∈ Set.Icc (-2) 1, f a x ≥ b) →
  (f a (-2) = 4 ∨ f a 1 = 4) →
  (f a (-2) = b ∨ f a 1 = b) →
  (∀ x y : ℝ, x < y → (2 - 7*b)*x > (2 - 7*b)*y) →
  a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l743_74306


namespace NUMINAMATH_CALUDE_reflection_line_sum_l743_74350

/-- Given a line y = mx + b, if the reflection of point (1, -2) across this line is (-3, 6), then m + b = 3 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    (x + 1) / 2 = -1 ∧ 
    (y - (-2)) / 2 = 2 ∧ 
    y = m * x + b ∧ 
    (1 - x) ^ 2 + (-2 - y) ^ 2 = (x - (-3)) ^ 2 + (y - 6) ^ 2) →
  m + b = 3 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l743_74350


namespace NUMINAMATH_CALUDE_delta_value_l743_74376

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ - 3 → Δ = -9 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l743_74376


namespace NUMINAMATH_CALUDE_largest_square_tile_size_l743_74351

theorem largest_square_tile_size 
  (length width : ℕ) 
  (h_length : length = 378) 
  (h_width : width = 595) : 
  ∃ (tile_size : ℕ), 
    tile_size = Nat.gcd length width ∧ 
    tile_size = 7 ∧
    length % tile_size = 0 ∧ 
    width % tile_size = 0 ∧
    ∀ (larger_size : ℕ), larger_size > tile_size → 
      length % larger_size ≠ 0 ∨ width % larger_size ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_square_tile_size_l743_74351


namespace NUMINAMATH_CALUDE_four_digit_number_remainder_l743_74310

theorem four_digit_number_remainder (a b c d : Nat) 
  (h1 : a ≠ 0) 
  (h2 : d ≠ 0) 
  (h3 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) 
  (h4 : ∃ k : Int, (1000 * a + 100 * b + 10 * c + d) + (1000 * a + 100 * c + 10 * b + d) = 900 * k) : 
  (1000 * a + 100 * b + 10 * c + d) % 90 = 45 := by
sorry

end NUMINAMATH_CALUDE_four_digit_number_remainder_l743_74310


namespace NUMINAMATH_CALUDE_second_graders_borrowed_books_l743_74369

theorem second_graders_borrowed_books (initial_books borrowed_books : ℕ) : 
  initial_books = 75 → 
  initial_books - borrowed_books = 57 → 
  borrowed_books = 18 := by
sorry

end NUMINAMATH_CALUDE_second_graders_borrowed_books_l743_74369


namespace NUMINAMATH_CALUDE_book_sale_profit_l743_74334

/-- Calculates the percent profit for a book sale given the cost, markup percentage, and discount percentage. -/
theorem book_sale_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) :
  cost = 50 ∧ markup_percent = 30 ∧ discount_percent = 10 →
  (((cost * (1 + markup_percent / 100)) * (1 - discount_percent / 100) - cost) / cost) * 100 = 17 := by
sorry

end NUMINAMATH_CALUDE_book_sale_profit_l743_74334


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l743_74385

theorem students_taking_one_subject (both : ℕ) (geometry : ℕ) (history_only : ℕ)
  (h1 : both = 15)
  (h2 : geometry = 30)
  (h3 : history_only = 18) :
  geometry - both + history_only = 33 := by
sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l743_74385


namespace NUMINAMATH_CALUDE_self_centered_max_solutions_l743_74390

/-- A polynomial is self-centered if it has integer coefficients and p(200) = 200 -/
def SelfCentered (p : ℤ → ℤ) : Prop :=
  (∀ x, ∃ n : ℕ, p x = (x : ℤ) ^ n) ∧ p 200 = 200

/-- The main theorem: any self-centered polynomial has at most 10 integer solutions to p(k) = k^4 -/
theorem self_centered_max_solutions (p : ℤ → ℤ) (h : SelfCentered p) :
  ∃ s : Finset ℤ, s.card ≤ 10 ∧ ∀ k : ℤ, p k = k^4 → k ∈ s := by
  sorry

end NUMINAMATH_CALUDE_self_centered_max_solutions_l743_74390


namespace NUMINAMATH_CALUDE_tangent_circles_t_value_l743_74332

-- Define the circles
def circle1 (t : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = t^2
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y + 24 = 0

-- Define tangency
def tangent (t : ℝ) : Prop := ∃ x y : ℝ, circle1 t x y ∧ circle2 x y

-- Theorem statement
theorem tangent_circles_t_value :
  ∀ t : ℝ, t > 0 → tangent t → t = 4 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_t_value_l743_74332


namespace NUMINAMATH_CALUDE_sue_votes_l743_74333

theorem sue_votes (total_votes : ℕ) (candidate1_percent : ℚ) (candidate2_percent : ℚ)
  (h_total : total_votes = 1000)
  (h_cand1 : candidate1_percent = 20 / 100)
  (h_cand2 : candidate2_percent = 45 / 100) :
  (1 - (candidate1_percent + candidate2_percent)) * total_votes = 350 :=
by sorry

end NUMINAMATH_CALUDE_sue_votes_l743_74333


namespace NUMINAMATH_CALUDE_gilbert_judah_ratio_l743_74304

/-- The number of crayons in each person's box -/
structure CrayonBoxes where
  karen : ℕ
  beatrice : ℕ
  gilbert : ℕ
  judah : ℕ

/-- The conditions of the crayon box problem -/
def crayon_box_conditions (boxes : CrayonBoxes) : Prop :=
  boxes.karen = 2 * boxes.beatrice ∧
  boxes.beatrice = 2 * boxes.gilbert ∧
  boxes.gilbert = boxes.judah ∧
  boxes.karen = 128 ∧
  boxes.judah = 8

/-- The theorem stating the ratio of crayons in Gilbert's box to Judah's box -/
theorem gilbert_judah_ratio (boxes : CrayonBoxes) 
  (h : crayon_box_conditions boxes) : 
  boxes.gilbert / boxes.judah = 4 := by
  sorry


end NUMINAMATH_CALUDE_gilbert_judah_ratio_l743_74304


namespace NUMINAMATH_CALUDE_square_sum_value_l743_74303

theorem square_sum_value (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -9) : x^2 + 9*y^2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l743_74303


namespace NUMINAMATH_CALUDE_no_real_solutions_l743_74394

theorem no_real_solutions : ¬ ∃ (x : ℝ), (x^(1/4) : ℝ) = 20 / (9 - 2 * (x^(1/4) : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l743_74394


namespace NUMINAMATH_CALUDE_inequality_addition_l743_74342

theorem inequality_addition (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l743_74342


namespace NUMINAMATH_CALUDE_reflect_distance_C_l743_74383

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflect_distance (p : ℝ × ℝ) : ℝ :=
  2 * |p.2|

theorem reflect_distance_C : reflect_distance (-3, 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_reflect_distance_C_l743_74383


namespace NUMINAMATH_CALUDE_greatest_good_and_smallest_bad_l743_74330

/-- Definition of a GOOD number -/
def isGood (M : ℕ) : Prop :=
  ∃ a b c d : ℕ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

/-- Definition of a BAD number -/
def isBad (M : ℕ) : Prop := ¬(isGood M)

/-- The greatest GOOD number -/
def greatestGood : ℕ := 576

/-- The smallest BAD number -/
def smallestBad : ℕ := 443

/-- Theorem stating that 576 is the greatest GOOD number and 443 is the smallest BAD number -/
theorem greatest_good_and_smallest_bad :
  (∀ M : ℕ, M > greatestGood → isBad M) ∧
  (∀ M : ℕ, M < smallestBad → isGood M) ∧
  isGood greatestGood ∧
  isBad smallestBad :=
sorry

end NUMINAMATH_CALUDE_greatest_good_and_smallest_bad_l743_74330
