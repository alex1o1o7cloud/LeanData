import Mathlib

namespace NUMINAMATH_CALUDE_x_fourth_minus_inverse_fourth_l1158_115832

theorem x_fourth_minus_inverse_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 527 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_inverse_fourth_l1158_115832


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l1158_115864

/-- The number of dice being thrown -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 4

/-- The minimum possible sum when throwing the dice -/
def min_sum : ℕ := num_dice

/-- The maximum possible sum when throwing the dice -/
def max_sum : ℕ := num_dice * sides_per_die

/-- The number of possible unique sums -/
def unique_sums : ℕ := max_sum - min_sum + 1

/-- The minimum number of throws required to ensure a repeated sum -/
def min_throws : ℕ := unique_sums + 1

theorem min_throws_for_repeated_sum :
  min_throws = 14 :=
sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l1158_115864


namespace NUMINAMATH_CALUDE_intersection_line_circle_l1158_115838

/-- Given a line x + y = a intersecting a circle x² + y² = 1 at points A and B,
    if |OA + OB| = |OA - OB|, then a = ±1 -/
theorem intersection_line_circle (a : ℝ) (A B : ℝ × ℝ) : 
  (∀ (x y : ℝ), x + y = a → x^2 + y^2 = 1 → (x, y) = A ∨ (x, y) = B) → 
  ‖(A.1, A.2)‖ = 1 →
  ‖(B.1, B.2)‖ = 1 →
  ‖(A.1 + B.1, A.2 + B.2)‖ = ‖(A.1 - B.1, A.2 - B.2)‖ →
  a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l1158_115838


namespace NUMINAMATH_CALUDE_binomial_inequality_l1158_115831

theorem binomial_inequality (n : ℤ) (x : ℝ) (h : x > 0) : (1 + x)^n ≥ 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l1158_115831


namespace NUMINAMATH_CALUDE_boat_speed_is_twelve_l1158_115816

/-- Represents the speed of a boat and current in a river --/
structure RiverJourney where
  boat_speed : ℝ
  current_speed : ℝ

/-- Represents the time taken for upstream and downstream journeys --/
structure JourneyTimes where
  upstream_time : ℝ
  downstream_time : ℝ

/-- Checks if the given boat speed is consistent with the journey times --/
def is_consistent_speed (journey : RiverJourney) (times : JourneyTimes) : Prop :=
  (journey.boat_speed - journey.current_speed) * times.upstream_time =
  (journey.boat_speed + journey.current_speed) * times.downstream_time

/-- The main theorem to prove --/
theorem boat_speed_is_twelve (times : JourneyTimes) 
    (h1 : times.upstream_time = 5)
    (h2 : times.downstream_time = 3) :
    ∃ (journey : RiverJourney), 
      journey.boat_speed = 12 ∧ 
      is_consistent_speed journey times := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_is_twelve_l1158_115816


namespace NUMINAMATH_CALUDE_internal_tangent_segment_bounded_l1158_115874

/-- Two equal circles with a common internal tangent and external tangents -/
structure TwoCirclesWithTangents where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of both circles (they are equal) -/
  radius : ℝ
  /-- Point where the common internal tangent intersects the external tangent of the first circle -/
  P : ℝ × ℝ
  /-- Point where the common internal tangent intersects the external tangent of the second circle -/
  Q : ℝ × ℝ
  /-- The circles are equal -/
  equal_circles : radius > 0
  /-- P is on the external tangent of the first circle -/
  P_on_external_tangent1 : (P.1 - center1.1) * (P.1 - center1.1) + (P.2 - center1.2) * (P.2 - center1.2) = radius * radius
  /-- Q is on the external tangent of the second circle -/
  Q_on_external_tangent2 : (Q.1 - center2.1) * (Q.1 - center2.1) + (Q.2 - center2.2) * (Q.2 - center2.2) = radius * radius
  /-- PQ is perpendicular to the radii at P and Q -/
  tangent_perpendicular : 
    (P.1 - center1.1) * (Q.1 - P.1) + (P.2 - center1.2) * (Q.2 - P.2) = 0 ∧
    (Q.1 - center2.1) * (P.1 - Q.1) + (Q.2 - center2.2) * (P.2 - Q.2) = 0

/-- The theorem statement -/
theorem internal_tangent_segment_bounded (c : TwoCirclesWithTangents) :
  (c.P.1 - c.Q.1) * (c.P.1 - c.Q.1) + (c.P.2 - c.Q.2) * (c.P.2 - c.Q.2) ≤
  (c.center1.1 - c.center2.1) * (c.center1.1 - c.center2.1) + (c.center1.2 - c.center2.2) * (c.center1.2 - c.center2.2) :=
sorry

end NUMINAMATH_CALUDE_internal_tangent_segment_bounded_l1158_115874


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1158_115820

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_value 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a)
  (h_a5 : a 5 = 1/2)
  (h_a4a6 : 4 * a 4 + a 6 = 2)
  (h_mn : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) :
  ∃ m n : ℕ, (1 : ℝ) / m + 4 / n ≥ 3/2 ∧
  (∀ k l : ℕ, (1 : ℝ) / k + 4 / l ≥ 3/2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1158_115820


namespace NUMINAMATH_CALUDE_amount_subtracted_l1158_115801

theorem amount_subtracted (number : ℝ) (subtracted_amount : ℝ) : 
  number = 70 →
  (number / 2) - subtracted_amount = 25 →
  subtracted_amount = 10 := by
sorry

end NUMINAMATH_CALUDE_amount_subtracted_l1158_115801


namespace NUMINAMATH_CALUDE_element_type_determined_by_protons_nuclide_type_determined_by_protons_and_neutrons_chemical_properties_determined_by_outermost_electrons_highest_valence_equals_main_group_number_l1158_115880

-- Define basic types
structure Element where
  protons : ℕ

structure Nuclide where
  protons : ℕ
  neutrons : ℕ

structure MainGroupElement where
  protons : ℕ
  outermostElectrons : ℕ

-- Define properties
def elementType (e : Element) : ℕ := e.protons

def nuclideType (n : Nuclide) : ℕ × ℕ := (n.protons, n.neutrons)

def mainChemicalProperties (e : MainGroupElement) : ℕ := e.outermostElectrons

def highestPositiveValence (e : MainGroupElement) : ℕ := e.outermostElectrons

-- Theorem statements
theorem element_type_determined_by_protons (e1 e2 : Element) :
  elementType e1 = elementType e2 ↔ e1.protons = e2.protons :=
sorry

theorem nuclide_type_determined_by_protons_and_neutrons (n1 n2 : Nuclide) :
  nuclideType n1 = nuclideType n2 ↔ n1.protons = n2.protons ∧ n1.neutrons = n2.neutrons :=
sorry

theorem chemical_properties_determined_by_outermost_electrons (e : MainGroupElement) :
  mainChemicalProperties e = e.outermostElectrons :=
sorry

theorem highest_valence_equals_main_group_number (e : MainGroupElement) :
  highestPositiveValence e = e.outermostElectrons :=
sorry

end NUMINAMATH_CALUDE_element_type_determined_by_protons_nuclide_type_determined_by_protons_and_neutrons_chemical_properties_determined_by_outermost_electrons_highest_valence_equals_main_group_number_l1158_115880


namespace NUMINAMATH_CALUDE_total_dress_cost_l1158_115836

theorem total_dress_cost (pauline_dress : ℕ) (h1 : pauline_dress = 30)
  (jean_dress : ℕ) (h2 : jean_dress = pauline_dress - 10)
  (ida_dress : ℕ) (h3 : ida_dress = jean_dress + 30)
  (patty_dress : ℕ) (h4 : patty_dress = ida_dress + 10) :
  pauline_dress + jean_dress + ida_dress + patty_dress = 160 := by
sorry

end NUMINAMATH_CALUDE_total_dress_cost_l1158_115836


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l1158_115871

theorem sum_of_five_consecutive_even_integers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l1158_115871


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1158_115886

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 6) 
  (h2 : a^2 + b^2 = 50) : 
  a * b = 7 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1158_115886


namespace NUMINAMATH_CALUDE_xy_nonneg_iff_abs_sum_eq_sum_abs_l1158_115833

theorem xy_nonneg_iff_abs_sum_eq_sum_abs (x y : ℝ) : x * y ≥ 0 ↔ |x + y| = |x| + |y| := by
  sorry

end NUMINAMATH_CALUDE_xy_nonneg_iff_abs_sum_eq_sum_abs_l1158_115833


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1158_115867

open Set

def A : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 3}

theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -3 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1158_115867


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1158_115839

/-- The curve C is defined by the equation x^2 / (k - 5) + y^2 / (3 - k) = -1 -/
def curve_C (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (k - 5) + p.2^2 / (3 - k) = -1}

/-- Predicate to check if a curve represents an ellipse with foci on the y-axis -/
def is_ellipse_y_foci (C : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ C = {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}

/-- The main theorem stating that 4 ≤ k < 5 is a necessary but not sufficient condition -/
theorem necessary_not_sufficient_condition (k : ℝ) :
  (is_ellipse_y_foci (curve_C k) → 4 ≤ k ∧ k < 5) ∧
  ¬(4 ≤ k ∧ k < 5 → is_ellipse_y_foci (curve_C k)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1158_115839


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1158_115803

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 4, 5}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ N) ∩ M = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1158_115803


namespace NUMINAMATH_CALUDE_simplify_fraction_l1158_115824

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (1 - x / (x - 1)) / (1 / (x^2 - x)) = -x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1158_115824


namespace NUMINAMATH_CALUDE_total_red_balloons_l1158_115830

/-- The number of red balloons Fred has -/
def fred_balloons : ℕ := 10

/-- The number of red balloons Sam has -/
def sam_balloons : ℕ := 46

/-- The number of red balloons Dan has -/
def dan_balloons : ℕ := 16

/-- The total number of red balloons -/
def total_balloons : ℕ := fred_balloons + sam_balloons + dan_balloons

theorem total_red_balloons : total_balloons = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_red_balloons_l1158_115830


namespace NUMINAMATH_CALUDE_nth_inequality_l1158_115875

theorem nth_inequality (x : ℝ) (n : ℕ) (h : x > 0) :
  x + (n^n : ℝ) / x^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_nth_inequality_l1158_115875


namespace NUMINAMATH_CALUDE_door_opening_probability_l1158_115837

theorem door_opening_probability (total_keys : ℕ) (opening_keys : ℕ) : 
  total_keys = 4 → 
  opening_keys = 2 → 
  (opening_keys : ℚ) * (total_keys - opening_keys) / (total_keys * (total_keys - 1)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_door_opening_probability_l1158_115837


namespace NUMINAMATH_CALUDE_perpendicular_vectors_condition_l1158_115860

/-- Given two vectors in R², prove that if they satisfy certain conditions, then a specific component of one vector equals -1. -/
theorem perpendicular_vectors_condition (m : ℝ) : 
  let a : Fin 2 → ℝ := ![m, 3]
  let b : Fin 2 → ℝ := ![-2, 2]
  (∀ i, (a - b) i * b i = 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_condition_l1158_115860


namespace NUMINAMATH_CALUDE_no_real_solutions_l1158_115821

theorem no_real_solutions :
  ¬∃ x : ℝ, x^4 + (x+1)^4 + (x+2)^4 = (x+3)^4 + 10 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1158_115821


namespace NUMINAMATH_CALUDE_bridget_apples_l1158_115883

/-- The number of apples Bridget bought -/
def total_apples : ℕ := 26

/-- The number of apples Bridget gave to Cassie -/
def apples_to_cassie : ℕ := 5

/-- The number of apples Bridget gave to Dan -/
def apples_to_dan : ℕ := 2

/-- The number of apples Bridget kept for herself -/
def apples_kept : ℕ := 6

theorem bridget_apples : 
  total_apples / 2 - apples_to_cassie - apples_to_dan = apples_kept :=
by sorry

end NUMINAMATH_CALUDE_bridget_apples_l1158_115883


namespace NUMINAMATH_CALUDE_peanut_butter_sandwich_days_l1158_115845

/-- Given:
  - There are 5 school days in a week
  - Karen packs ham sandwiches on 3 school days
  - Karen packs cake on one randomly chosen day
  - The probability of packing a ham sandwich and cake on the same day is 12%
  Prove that Karen packs peanut butter sandwiches on 2 days. -/
theorem peanut_butter_sandwich_days :
  ∀ (total_days ham_days cake_days : ℕ) 
    (prob_ham_and_cake : ℚ),
  total_days = 5 →
  ham_days = 3 →
  cake_days = 1 →
  prob_ham_and_cake = 12 / 100 →
  (ham_days : ℚ) / total_days * (cake_days : ℚ) / total_days = prob_ham_and_cake →
  total_days - ham_days = 2 :=
by sorry

end NUMINAMATH_CALUDE_peanut_butter_sandwich_days_l1158_115845


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l1158_115848

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.sin (10 * π / 180) =
  1 / (2 * Real.sin (10 * π / 180) ^ 2 * Real.cos (20 * π / 180)) + 4 / (Real.sqrt 3 * Real.sin (10 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l1158_115848


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1158_115888

theorem decimal_to_fraction (x : ℚ) : x = 224/100 → x = 56/25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1158_115888


namespace NUMINAMATH_CALUDE_regular_polygon_angle_relation_l1158_115889

theorem regular_polygon_angle_relation (n : ℕ) : n ≥ 3 →
  (120 : ℝ) = 5 * (360 / n) → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_angle_relation_l1158_115889


namespace NUMINAMATH_CALUDE_max_value_x2_l1158_115834

theorem max_value_x2 (x₁ x₂ x₃ : ℝ) 
  (h : x₁^2 + x₂^2 + x₃^2 + x₁*x₂ + x₂*x₃ = 2) : 
  |x₂| ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x2_l1158_115834


namespace NUMINAMATH_CALUDE_calculate_expression_factorize_polynomial_l1158_115806

-- Part 1
theorem calculate_expression : (1 / 3)⁻¹ - Real.sqrt 16 + (-2016)^0 = 0 := by sorry

-- Part 2
theorem factorize_polynomial (x : ℝ) : 3 * x^2 - 6 * x + 3 = 3 * (x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_factorize_polynomial_l1158_115806


namespace NUMINAMATH_CALUDE_negative_square_opposite_l1158_115841

-- Define opposite numbers
def opposite (a b : ℤ) : Prop := a = -b

-- Theorem statement
theorem negative_square_opposite : opposite (-2^2) ((-2)^2) := by
  sorry

end NUMINAMATH_CALUDE_negative_square_opposite_l1158_115841


namespace NUMINAMATH_CALUDE_ratio_nine_to_five_percent_l1158_115869

/-- The ratio 9 : 5 expressed as a percentage -/
def ratio_to_percent : ℚ := 9 / 5 * 100

/-- Theorem: The ratio 9 : 5 expressed as a percentage is equal to 180% -/
theorem ratio_nine_to_five_percent : ratio_to_percent = 180 := by
  sorry

end NUMINAMATH_CALUDE_ratio_nine_to_five_percent_l1158_115869


namespace NUMINAMATH_CALUDE_six_million_three_hundred_ninety_thousand_scientific_notation_l1158_115847

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem six_million_three_hundred_ninety_thousand_scientific_notation :
  toScientificNotation 6390000 = ScientificNotation.mk 6.39 6 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_six_million_three_hundred_ninety_thousand_scientific_notation_l1158_115847


namespace NUMINAMATH_CALUDE_mike_picked_52_peaches_l1158_115865

/-- The number of peaches Mike initially had -/
def initial_peaches : ℕ := 34

/-- The number of peaches Mike has now -/
def current_peaches : ℕ := 86

/-- The number of peaches Mike picked -/
def picked_peaches : ℕ := current_peaches - initial_peaches

theorem mike_picked_52_peaches : picked_peaches = 52 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_52_peaches_l1158_115865


namespace NUMINAMATH_CALUDE_balls_in_boxes_count_l1158_115861

/-- The number of ways to place 4 different balls into 4 numbered boxes with exactly one empty box -/
def placeBallsInBoxes : ℕ :=
  -- We define this as a natural number, but don't provide the implementation
  sorry

/-- The theorem stating that there are 144 ways to place 4 different balls into 4 numbered boxes
    such that exactly one box is empty -/
theorem balls_in_boxes_count : placeBallsInBoxes = 144 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_count_l1158_115861


namespace NUMINAMATH_CALUDE_banknote_probability_l1158_115843

/-- Represents a bag of banknotes -/
structure Bag :=
  (ten : ℕ)    -- Number of ten-yuan banknotes
  (five : ℕ)   -- Number of five-yuan banknotes
  (one : ℕ)    -- Number of one-yuan banknotes

/-- Calculate the total value of banknotes in a bag -/
def bagValue (b : Bag) : ℕ :=
  10 * b.ten + 5 * b.five + b.one

/-- Calculate the number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The probability of drawing at least one 5-yuan note from bag B -/
def probAtLeastOne5 (b : Bag) : ℚ :=
  (choose2 b.five.succ + b.five * b.one) / choose2 (b.five + b.one)

theorem banknote_probability :
  let bagA : Bag := ⟨2, 0, 3⟩
  let bagB : Bag := ⟨0, 4, 3⟩
  let totalDraws := choose2 (bagValue bagA) * choose2 (bagValue bagB)
  let favorableDraws := choose2 bagA.one * (choose2 bagB.five.succ + bagB.five * bagB.one)
  (favorableDraws : ℚ) / totalDraws = 9 / 35 := by
  sorry

end NUMINAMATH_CALUDE_banknote_probability_l1158_115843


namespace NUMINAMATH_CALUDE_hexagon_to_rhombus_l1158_115842

/-- A regular hexagon -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A rhombus -/
structure Rhombus where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A part of the hexagon after cutting -/
structure HexagonPart where
  area : ℝ
  area_pos : area > 0

/-- Function to cut a regular hexagon into three parts -/
def cut_hexagon (h : RegularHexagon) : (HexagonPart × HexagonPart × HexagonPart) :=
  sorry

/-- Function to form a rhombus from three hexagon parts -/
def form_rhombus (p1 p2 p3 : HexagonPart) : Rhombus :=
  sorry

/-- Theorem stating that a regular hexagon can be cut into three parts that form a rhombus -/
theorem hexagon_to_rhombus (h : RegularHexagon) :
  ∃ (p1 p2 p3 : HexagonPart), 
    let (p1', p2', p3') := cut_hexagon h
    p1 = p1' ∧ p2 = p2' ∧ p3 = p3' ∧
    ∃ (r : Rhombus), r = form_rhombus p1 p2 p3 :=
  sorry

end NUMINAMATH_CALUDE_hexagon_to_rhombus_l1158_115842


namespace NUMINAMATH_CALUDE_hotel_assignment_problem_l1158_115807

/-- The number of ways to assign friends to rooms -/
def assignFriendsToRooms (numFriends numRooms maxPerRoom : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of assignments for the given problem -/
theorem hotel_assignment_problem :
  assignFriendsToRooms 6 5 2 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_hotel_assignment_problem_l1158_115807


namespace NUMINAMATH_CALUDE_billy_age_l1158_115856

theorem billy_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 64) : 
  billy = 48 := by
sorry

end NUMINAMATH_CALUDE_billy_age_l1158_115856


namespace NUMINAMATH_CALUDE_fifth_term_value_l1158_115813

/-- An arithmetic sequence satisfying the given recursive relation -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = -a n + n

/-- The fifth term of the sequence is 9/4 -/
theorem fifth_term_value (a : ℕ → ℚ) (h : ArithmeticSequence a) : a 5 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l1158_115813


namespace NUMINAMATH_CALUDE_total_pages_read_l1158_115826

theorem total_pages_read (pages_yesterday pages_today : ℕ) 
  (h1 : pages_yesterday = 21) 
  (h2 : pages_today = 17) : 
  pages_yesterday + pages_today = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_read_l1158_115826


namespace NUMINAMATH_CALUDE_circle_chord_distance_l1158_115878

theorem circle_chord_distance (r : ℝ) (AB AC BC : ℝ) : 
  r = 10 →
  AB = 2 * r →
  AC = 12 →
  AB^2 = AC^2 + BC^2 →
  BC = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_chord_distance_l1158_115878


namespace NUMINAMATH_CALUDE_jerry_shelf_problem_l1158_115808

/-- The number of action figures Jerry added to the shelf -/
def action_figures_added : ℕ := 7

/-- The initial number of action figures on the shelf -/
def initial_action_figures : ℕ := 5

/-- The number of books on the shelf (constant) -/
def books : ℕ := 9

/-- Theorem stating that the number of action figures added satisfies the problem conditions -/
theorem jerry_shelf_problem :
  initial_action_figures + action_figures_added = books + 3 :=
by sorry

end NUMINAMATH_CALUDE_jerry_shelf_problem_l1158_115808


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1158_115881

theorem perfect_square_condition (a b c d : ℕ+) : 
  (↑a + Real.rpow 2 (1/3 : ℝ) * ↑b + Real.rpow 2 (2/3 : ℝ) * ↑c)^2 = ↑d → 
  ∃ (n : ℕ), d = n^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1158_115881


namespace NUMINAMATH_CALUDE_tan_half_alpha_eq_two_implies_ratio_l1158_115885

theorem tan_half_alpha_eq_two_implies_ratio (α : Real) 
  (h : Real.tan (α / 2) = 2) : 
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_alpha_eq_two_implies_ratio_l1158_115885


namespace NUMINAMATH_CALUDE_jason_fire_frequency_l1158_115844

/-- Given the conditions of Jason's gameplay in Duty for Ashes, prove that he fires his weapon every 15 seconds on average. -/
theorem jason_fire_frequency
  (flame_duration : ℕ)
  (total_flame_time : ℕ)
  (seconds_per_minute : ℕ)
  (h1 : flame_duration = 5)
  (h2 : total_flame_time = 20)
  (h3 : seconds_per_minute = 60) :
  (seconds_per_minute : ℚ) / ((total_flame_time : ℚ) / (flame_duration : ℚ)) = 15 := by
  sorry

#check jason_fire_frequency

end NUMINAMATH_CALUDE_jason_fire_frequency_l1158_115844


namespace NUMINAMATH_CALUDE_total_chewing_gums_l1158_115800

theorem total_chewing_gums (mary sam sue : ℕ) : 
  mary = 5 → sam = 10 → sue = 15 → mary + sam + sue = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_chewing_gums_l1158_115800


namespace NUMINAMATH_CALUDE_cookie_problem_l1158_115827

theorem cookie_problem : ∃! C : ℕ, 0 < C ∧ C < 80 ∧ C % 6 = 5 ∧ C % 9 = 7 ∧ C = 29 := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l1158_115827


namespace NUMINAMATH_CALUDE_gcd_840_1764_l1158_115882

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l1158_115882


namespace NUMINAMATH_CALUDE_least_integer_square_98_more_than_double_l1158_115896

theorem least_integer_square_98_more_than_double : 
  ∃ x : ℤ, x^2 = 2*x + 98 ∧ ∀ y : ℤ, y^2 = 2*y + 98 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_integer_square_98_more_than_double_l1158_115896


namespace NUMINAMATH_CALUDE_crosswalk_wait_probability_l1158_115855

/-- Represents the duration of the red light in seconds -/
def red_light_duration : ℕ := 40

/-- Represents the minimum waiting time in seconds for which we calculate the probability -/
def min_wait_time : ℕ := 15

/-- The probability of waiting at least 'min_wait_time' seconds for a green light when encountering a red light -/
def wait_probability : ℚ := 5/8

/-- Theorem stating that the probability of waiting at least 'min_wait_time' seconds for a green light
    when encountering a red light of duration 'red_light_duration' is equal to 'wait_probability' -/
theorem crosswalk_wait_probability :
  (red_light_duration - min_wait_time : ℚ) / red_light_duration = wait_probability := by
  sorry

end NUMINAMATH_CALUDE_crosswalk_wait_probability_l1158_115855


namespace NUMINAMATH_CALUDE_line_through_two_points_l1158_115862

/-- 
Given a line with equation x = 8y + 5 that passes through points (m, n) and (m + 2, n + p),
prove that p = 1/4.
-/
theorem line_through_two_points (m n p : ℝ) : 
  (m = 8 * n + 5) ∧ (m + 2 = 8 * (n + p) + 5) → p = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_line_through_two_points_l1158_115862


namespace NUMINAMATH_CALUDE_cubic_curve_rational_points_l1158_115805

-- Define a cubic curve with rational coefficients
def CubicCurve (f : ℚ → ℚ → ℚ) : Prop :=
  ∀ x y, ∃ a b c d e g h k l : ℚ, 
    f x y = a*x^3 + b*x^2*y + c*x*y^2 + d*y^3 + e*x^2 + g*x*y + h*y^2 + k*x + l*y

-- Define a point on the curve
def PointOnCurve (f : ℚ → ℚ → ℚ) (x y : ℚ) : Prop :=
  f x y = 0

-- Theorem statement
theorem cubic_curve_rational_points 
  (f : ℚ → ℚ → ℚ) 
  (hf : CubicCurve f) 
  (x₀ y₀ : ℚ) 
  (h₀ : PointOnCurve f x₀ y₀) :
  ∃ x' y' : ℚ, x' ≠ x₀ ∧ y' ≠ y₀ ∧ PointOnCurve f x' y' :=
sorry

end NUMINAMATH_CALUDE_cubic_curve_rational_points_l1158_115805


namespace NUMINAMATH_CALUDE_vertex_below_x_axis_iff_k_less_than_4_l1158_115804

/-- A quadratic function of the form y = x^2 - 4x + k -/
def quadratic_function (x k : ℝ) : ℝ := x^2 - 4*x + k

/-- The x-coordinate of the vertex of the quadratic function -/
def vertex_x : ℝ := 2

/-- The y-coordinate of the vertex of the quadratic function -/
def vertex_y (k : ℝ) : ℝ := quadratic_function vertex_x k

/-- The vertex is below the x-axis if its y-coordinate is negative -/
def vertex_below_x_axis (k : ℝ) : Prop := vertex_y k < 0

theorem vertex_below_x_axis_iff_k_less_than_4 :
  ∀ k : ℝ, vertex_below_x_axis k ↔ k < 4 := by sorry

end NUMINAMATH_CALUDE_vertex_below_x_axis_iff_k_less_than_4_l1158_115804


namespace NUMINAMATH_CALUDE_even_function_iff_b_zero_l1158_115872

/-- For real numbers a and b, and function f(x) = a*cos(x) + b*sin(x),
    f(x) is an even function if and only if b = 0 -/
theorem even_function_iff_b_zero (a b : ℝ) :
  (∀ x, a * Real.cos x + b * Real.sin x = a * Real.cos (-x) + b * Real.sin (-x)) ↔ b = 0 :=
by sorry

end NUMINAMATH_CALUDE_even_function_iff_b_zero_l1158_115872


namespace NUMINAMATH_CALUDE_substance_volume_weight_relation_l1158_115892

/-- Given a substance where volume is directly proportional to weight,
    prove that if 48 cubic inches weigh 112 ounces,
    then 63 ounces will have a volume of 27 cubic inches. -/
theorem substance_volume_weight_relation 
  (k : ℚ) -- Constant of proportionality
  (h1 : 48 = k * 112) -- 48 cubic inches weigh 112 ounces
  : k * 63 = 27 := by
  sorry

end NUMINAMATH_CALUDE_substance_volume_weight_relation_l1158_115892


namespace NUMINAMATH_CALUDE_hiking_trip_solution_l1158_115866

/-- Represents the hiking trip scenario -/
structure HikingTrip where
  men_count : ℕ
  women_count : ℕ
  total_weight : ℝ
  men_backpack_weight : ℝ
  women_backpack_weight : ℝ

/-- Checks if the hiking trip satisfies the given conditions -/
def is_valid_hiking_trip (trip : HikingTrip) : Prop :=
  trip.men_count = 2 ∧
  trip.women_count = 3 ∧
  trip.total_weight = 44 ∧
  trip.men_count * trip.men_backpack_weight + trip.women_count * trip.women_backpack_weight = trip.total_weight ∧
  trip.men_backpack_weight + trip.women_backpack_weight + trip.women_backpack_weight / 2 = 
    trip.women_backpack_weight + trip.men_backpack_weight / 2

theorem hiking_trip_solution (trip : HikingTrip) :
  is_valid_hiking_trip trip → trip.women_backpack_weight = 8 ∧ trip.men_backpack_weight = 10 := by
  sorry

end NUMINAMATH_CALUDE_hiking_trip_solution_l1158_115866


namespace NUMINAMATH_CALUDE_work_completion_time_l1158_115818

theorem work_completion_time 
  (people : ℕ) 
  (original_time : ℕ) 
  (original_work : ℝ) 
  (h1 : original_time = 16) 
  (h2 : people * original_time = original_work) :
  (2 * people) * 8 = original_work / 2 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1158_115818


namespace NUMINAMATH_CALUDE_value_of_a_l1158_115873

-- Define set A
def A : Set ℝ := {x | x^2 ≠ 1}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x = a}

-- Theorem statement
theorem value_of_a (a : ℝ) (h : B a ⊆ A) : a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1158_115873


namespace NUMINAMATH_CALUDE_tangent_line_at_two_condition_equivalent_to_range_l1158_115812

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + (2 * a - 1) / x + 1 - 3 * a

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y - 4 = 0

-- Theorem for part I
theorem tangent_line_at_two (a : ℝ) (h : a = 1) :
  ∃ y, f a 2 = y ∧ tangent_line 2 y :=
sorry

-- Theorem for part II
theorem condition_equivalent_to_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ (1 - a) * Real.log x) ↔ a ≥ 1/3 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_two_condition_equivalent_to_range_l1158_115812


namespace NUMINAMATH_CALUDE_incorrect_calculation_l1158_115891

theorem incorrect_calculation (x : ℝ) : (-3 * x)^2 ≠ 6 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l1158_115891


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1158_115863

theorem sum_of_numbers (A B C : ℚ) : 
  (A / B = 2 / 5) → 
  (B / C = 4 / 7) → 
  (A = 16) → 
  (A + B + C = 126) := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1158_115863


namespace NUMINAMATH_CALUDE_tourist_base_cottages_l1158_115890

theorem tourist_base_cottages :
  ∀ (x : ℕ) (n : ℕ+),
    (2 * x) + x + (n : ℕ) * x ≥ 70 →
    3 * ((n : ℕ) * x) = 2 * x + 25 →
    (2 * x) + x + (n : ℕ) * x = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_tourist_base_cottages_l1158_115890


namespace NUMINAMATH_CALUDE_mary_berry_cost_l1158_115819

/-- The amount Mary paid for berries, given her total payment, peach cost, and change received. -/
theorem mary_berry_cost (total_paid change peach_cost : ℚ) 
  (h1 : total_paid = 20)
  (h2 : change = 598/100)
  (h3 : peach_cost = 683/100) :
  total_paid - change - peach_cost = 719/100 := by
  sorry


end NUMINAMATH_CALUDE_mary_berry_cost_l1158_115819


namespace NUMINAMATH_CALUDE_problem_solution_l1158_115858

-- Define propositions P and Q
def P (x a : ℝ) : Prop := 2 * x^2 - 5 * a * x - 3 * a^2 < 0

def Q (x : ℝ) : Prop := 2 * Real.sin x > 1 ∧ x^2 - x - 2 < 0

theorem problem_solution (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, a = 2 ∧ P x a ∧ Q x → π / 6 < x ∧ x < 2) ∧
  ((∀ x : ℝ, ¬(P x a) → ¬(Q x)) ∧ (∃ x : ℝ, Q x ∧ P x a) → 2 / 3 ≤ a) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1158_115858


namespace NUMINAMATH_CALUDE_monday_hours_calculation_l1158_115810

def hourly_wage : ℝ := 10
def monday_tips : ℝ := 18
def tuesday_hours : ℝ := 5
def tuesday_tips : ℝ := 12
def wednesday_hours : ℝ := 7
def wednesday_tips : ℝ := 20
def total_earnings : ℝ := 240

theorem monday_hours_calculation (monday_hours : ℝ) :
  hourly_wage * monday_hours + monday_tips +
  hourly_wage * tuesday_hours + tuesday_tips +
  hourly_wage * wednesday_hours + wednesday_tips = total_earnings →
  monday_hours = 7 := by
sorry

end NUMINAMATH_CALUDE_monday_hours_calculation_l1158_115810


namespace NUMINAMATH_CALUDE_print_shop_price_difference_l1158_115894

def print_shop_x_price : ℚ := 120 / 100
def print_shop_y_price : ℚ := 170 / 100
def number_of_copies : ℕ := 40

theorem print_shop_price_difference :
  number_of_copies * print_shop_y_price - number_of_copies * print_shop_x_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_price_difference_l1158_115894


namespace NUMINAMATH_CALUDE_number_puzzle_l1158_115879

theorem number_puzzle : ∃! x : ℝ, 3 * (2 * x + 9) = 69 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l1158_115879


namespace NUMINAMATH_CALUDE_max_visible_sum_l1158_115854

-- Define a cube type
structure Cube :=
  (faces : Fin 6 → Nat)

-- Define the block type
structure Block :=
  (cubes : Fin 4 → Cube)

-- Function to calculate the sum of visible faces
def sumVisibleFaces (b : Block) : Nat :=
  sorry

-- Theorem statement
theorem max_visible_sum :
  ∃ (b : Block),
    (∀ i : Fin 6, ∀ c : Fin 4, 1 ≤ (b.cubes c).faces i ∧ (b.cubes c).faces i ≤ 6) ∧
    (∀ c1 c2 : Fin 4, c1 ≠ c2 → ∃ i j : Fin 6, (b.cubes c1).faces i = (b.cubes c2).faces j) ∧
    (sumVisibleFaces b = 68) ∧
    (∀ b' : Block, sumVisibleFaces b' ≤ sumVisibleFaces b) :=
  sorry


end NUMINAMATH_CALUDE_max_visible_sum_l1158_115854


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l1158_115868

noncomputable def f (a x : ℝ) : ℝ := (a + 1/a) * Real.log x + 1/x - x

theorem f_monotonicity_and_extrema :
  ∀ a : ℝ, a > 0 →
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 1/a ∧ 1/a < x₂ ∧ x₂ < 1 → f a x₁ < f a (1/a) ∧ f a (1/a) < f a x₂) ∧
    (f a (1/a) = -(a + 1/a) * Real.log a + a - 1/a) ∧
    (f a a = (a + 1/a) * Real.log a + 1/a - a) ∧
    (∀ x : ℝ, x > 0 → f a x ≥ f a (1/a) ∧ f a x ≤ f a a) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l1158_115868


namespace NUMINAMATH_CALUDE_boys_count_l1158_115853

/-- Represents the number of skips in a jump rope competition. -/
structure SkipCompetition where
  boyAvg : ℕ
  girlAvg : ℕ
  totalAvg : ℕ
  boyCount : ℕ
  girlCount : ℕ

/-- Theorem stating the number of boys in the skip competition. -/
theorem boys_count (comp : SkipCompetition) 
  (h1 : comp.boyAvg = 85)
  (h2 : comp.girlAvg = 92)
  (h3 : comp.totalAvg = 88)
  (h4 : comp.boyCount = comp.girlCount + 10)
  (h5 : (comp.boyAvg * comp.boyCount + comp.girlAvg * comp.girlCount) / (comp.boyCount + comp.girlCount) = comp.totalAvg) :
  comp.boyCount = 40 := by
  sorry

end NUMINAMATH_CALUDE_boys_count_l1158_115853


namespace NUMINAMATH_CALUDE_walter_bus_time_l1158_115835

def wake_up_time : Nat := 6 * 60
def leave_time : Nat := 7 * 60
def return_time : Nat := 16 * 60 + 30
def num_classes : Nat := 7
def class_duration : Nat := 45
def lunch_duration : Nat := 45
def additional_time : Nat := 90

def total_away_time : Nat := return_time - leave_time
def total_school_time : Nat := num_classes * class_duration + lunch_duration + additional_time

theorem walter_bus_time :
  total_away_time - total_school_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_walter_bus_time_l1158_115835


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l1158_115814

theorem max_value_of_exponential_difference (x : ℝ) : 5^x - 25^x ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l1158_115814


namespace NUMINAMATH_CALUDE_circle_symmetry_range_l1158_115876

/-- A circle with equation x^2 + y^2 - 2x + 6y + 5a = 0 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 + 6*p.2 + 5*a = 0}

/-- A line with equation y = x + 2b -/
def Line (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 2*b}

/-- The circle is symmetric about the line -/
def IsSymmetric (a b : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), center ∈ Circle a ∧ center ∈ Line b

theorem circle_symmetry_range (a b : ℝ) :
  IsSymmetric a b → a - b ∈ Set.Iio 4 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_range_l1158_115876


namespace NUMINAMATH_CALUDE_last_digit_for_multiple_of_five_l1158_115823

theorem last_digit_for_multiple_of_five (n : ℕ) : 
  (71360 ≤ n ∧ n ≤ 71369) ∧ (n % 5 = 0) → (n % 10 = 0 ∨ n % 10 = 5) :=
by sorry

end NUMINAMATH_CALUDE_last_digit_for_multiple_of_five_l1158_115823


namespace NUMINAMATH_CALUDE_similar_terms_and_system_solution_l1158_115802

theorem similar_terms_and_system_solution :
  ∀ (m n : ℤ) (a b : ℝ) (x y : ℝ),
    (m - 1 = n - 2*m ∧ m + n = 3*m + n - 4) →
    (m*x + (n-2)*y = 24 ∧ 2*m*x + n*y = 46) →
    (x = 9 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_similar_terms_and_system_solution_l1158_115802


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1158_115852

/-- Given a principal amount, time period, and final amount, 
    calculate the annual interest rate for compound interest. -/
theorem compound_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (final_amount : ℝ) 
  (h1 : principal = 8000) 
  (h2 : time = 2) 
  (h3 : final_amount = 8820) : 
  ∃ (rate : ℝ), 
    final_amount = principal * (1 + rate) ^ time ∧ 
    rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1158_115852


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l1158_115857

theorem alpha_beta_sum (α β : ℝ) : 
  (∀ x : ℝ, x ≠ 30 → (x - α) / (x + β) = (x^2 - 120*x + 3600) / (x^2 + 70*x - 2300)) →
  α + β = 137 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l1158_115857


namespace NUMINAMATH_CALUDE_plane_equation_l1158_115811

def point_on_plane (A B C D : ℤ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

def perpendicular_planes (A1 B1 C1 D1 A2 B2 C2 D2 : ℤ) : Prop :=
  A1 * A2 + B1 * B2 + C1 * C2 = 0

theorem plane_equation : ∃ (A B C D : ℤ),
  (A > 0) ∧
  (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1) ∧
  point_on_plane A B C D 0 0 0 ∧
  point_on_plane A B C D 2 (-2) 2 ∧
  perpendicular_planes A B C D 2 (-1) 3 4 ∧
  A = 2 ∧ B = -1 ∧ C = 1 ∧ D = 0 :=
sorry

end NUMINAMATH_CALUDE_plane_equation_l1158_115811


namespace NUMINAMATH_CALUDE_digit_fraction_statement_l1158_115840

theorem digit_fraction_statement : 
  ∃ (a b c : ℕ) (f g h : ℚ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    f + 2 * g + h = 1 ∧
    f = 1/2 ∧
    g = 1/5 ∧
    h = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_digit_fraction_statement_l1158_115840


namespace NUMINAMATH_CALUDE_botany_zoology_ratio_l1158_115829

theorem botany_zoology_ratio : 
  ∀ (total_books zoology_books botany_books n : ℕ),
  total_books = 80 →
  zoology_books = 16 →
  botany_books = n * zoology_books →
  total_books = botany_books + zoology_books →
  botany_books / zoology_books = 4 := by
sorry

end NUMINAMATH_CALUDE_botany_zoology_ratio_l1158_115829


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1158_115809

/-- The eccentricity of a hyperbola passing through the focus of a parabola -/
theorem hyperbola_eccentricity (a : ℝ) (h_a : a > 0) :
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 = 1
  let parabola := fun (x y : ℝ) ↦ y^2 = 8 * x
  let focus : ℝ × ℝ := (2, 0)
  hyperbola focus.1 focus.2 →
  let c := Real.sqrt (a^2 + 1)
  c / a = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1158_115809


namespace NUMINAMATH_CALUDE_draw_all_red_probability_l1158_115887

-- Define the number of red and green chips
def num_red : ℕ := 3
def num_green : ℕ := 2

-- Define the total number of chips
def total_chips : ℕ := num_red + num_green

-- Define the probability of drawing all red chips before both green chips
def prob_all_red : ℚ := 3 / 10

-- Theorem statement
theorem draw_all_red_probability :
  prob_all_red = (num_red * (num_red - 1) * (num_red - 2)) / 
    (total_chips * (total_chips - 1) * (total_chips - 2)) :=
by sorry

end NUMINAMATH_CALUDE_draw_all_red_probability_l1158_115887


namespace NUMINAMATH_CALUDE_price_per_diaper_l1158_115884

def boxes : ℕ := 30
def packs_per_box : ℕ := 40
def diapers_per_pack : ℕ := 160
def total_revenue : ℕ := 960000

def total_diapers : ℕ := boxes * packs_per_box * diapers_per_pack

theorem price_per_diaper :
  total_revenue / total_diapers = 5 :=
by sorry

end NUMINAMATH_CALUDE_price_per_diaper_l1158_115884


namespace NUMINAMATH_CALUDE_expression_decrease_decrease_percentage_l1158_115849

theorem expression_decrease (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : 
  (2 * ((1/2 * x)^2) * (1/2 * y)) / (2 * x^2 * y) = 1/4 :=
sorry

theorem decrease_percentage : (1 - 1/4) * 100 = 87.5 :=
sorry

end NUMINAMATH_CALUDE_expression_decrease_decrease_percentage_l1158_115849


namespace NUMINAMATH_CALUDE_coinciding_rest_days_count_l1158_115817

/-- Craig's work cycle in days -/
def craig_cycle : ℕ := 6

/-- Dana's work cycle in days -/
def dana_cycle : ℕ := 6

/-- Number of days Craig works in his cycle -/
def craig_work_days : ℕ := 4

/-- Number of days Dana works in her cycle -/
def dana_work_days : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 1000

/-- The number of days both Craig and Dana have rest-days on the same day -/
def coinciding_rest_days : ℕ := total_days / craig_cycle

theorem coinciding_rest_days_count :
  coinciding_rest_days = 166 := by sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_count_l1158_115817


namespace NUMINAMATH_CALUDE_clock_time_after_2016_hours_l1158_115893

theorem clock_time_after_2016_hours (current_time : ℕ) (hours_passed : ℕ) : 
  current_time = 7 → hours_passed = 2016 → (current_time + hours_passed) % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_clock_time_after_2016_hours_l1158_115893


namespace NUMINAMATH_CALUDE_expression_evaluation_l1158_115825

theorem expression_evaluation : 200 * (200 - 8) - (200 * 200 + 8) = -1608 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1158_115825


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1158_115897

/-- A point in the fourth quadrant with given conditions has coordinates (7, -3) -/
theorem point_in_fourth_quadrant (x y : ℝ) (h1 : x > 0) (h2 : y < 0) 
  (h3 : |x| = 7) (h4 : y^2 = 9) : (x, y) = (7, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1158_115897


namespace NUMINAMATH_CALUDE_wages_decrease_percentage_l1158_115846

theorem wages_decrease_percentage (W : ℝ) (x : ℝ) 
  (h1 : W > 0)  -- Wages are positive
  (h2 : 0 ≤ x ∧ x ≤ 100)  -- Percentage decrease is between 0 and 100
  (h3 : 0.30 * (W * (1 - x / 100)) = 1.80 * (0.15 * W)) :  -- Condition from the problem
  x = 10 := by sorry

end NUMINAMATH_CALUDE_wages_decrease_percentage_l1158_115846


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l1158_115828

/-- Given three two-digit numbers and a fourth unknown two-digit number,
    if the sum of the digits of all four numbers is 1/4 of their total sum,
    then the smallest possible value for the unknown number is 70. -/
theorem smallest_fourth_number (x : ℕ) :
  x ≥ 10 ∧ x < 100 →
  (34 + 21 + 63 + x : ℕ) = 4 * ((3 + 4 + 2 + 1 + 6 + 3 + (x / 10) + (x % 10)) : ℕ) →
  ∀ y : ℕ, y ≥ 10 ∧ y < 100 →
    (34 + 21 + 63 + y : ℕ) = 4 * ((3 + 4 + 2 + 1 + 6 + 3 + (y / 10) + (y % 10)) : ℕ) →
    x ≤ y →
  x = 70 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l1158_115828


namespace NUMINAMATH_CALUDE_gcd_1337_382_l1158_115822

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1337_382_l1158_115822


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l1158_115870

/-- Proves that given a mixture with 20% water content, if adding 8.333333333333334 gallons
    of water increases the water percentage to 25%, then the initial volume of the mixture
    is 125 gallons. -/
theorem initial_mixture_volume
  (initial_water_percentage : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_water_percentage = 0.20)
  (h2 : added_water = 8.333333333333334)
  (h3 : final_water_percentage = 0.25)
  (h4 : ∀ v : ℝ, final_water_percentage * (v + added_water) = initial_water_percentage * v + added_water) :
  ∃ v : ℝ, v = 125 := by
  sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l1158_115870


namespace NUMINAMATH_CALUDE_special_house_additional_profit_l1158_115895

/-- The additional profit made by building and selling a special house compared to a regular house -/
theorem special_house_additional_profit
  (C : ℝ)  -- Regular house construction cost
  (regular_selling_price : ℝ)
  (special_selling_price : ℝ)
  (h1 : regular_selling_price = 350000)
  (h2 : special_selling_price = 1.8 * regular_selling_price)
  : (special_selling_price - (C + 200000)) - (regular_selling_price - C) = 80000 := by
  sorry

#check special_house_additional_profit

end NUMINAMATH_CALUDE_special_house_additional_profit_l1158_115895


namespace NUMINAMATH_CALUDE_songs_added_l1158_115859

/-- Calculates the number of new songs added to an mp3 player. -/
theorem songs_added (initial : ℕ) (deleted : ℕ) (final : ℕ) : 
  initial = 11 → deleted = 9 → final = 10 → final - (initial - deleted) = 8 := by
  sorry

end NUMINAMATH_CALUDE_songs_added_l1158_115859


namespace NUMINAMATH_CALUDE_response_rate_increase_l1158_115898

/-- Calculate the percentage increase in response rate between two surveys -/
theorem response_rate_increase (customers1 customers2 respondents1 respondents2 : ℕ) :
  customers1 = 80 →
  customers2 = 63 →
  respondents1 = 7 →
  respondents2 = 9 →
  let rate1 := (respondents1 : ℝ) / customers1
  let rate2 := (respondents2 : ℝ) / customers2
  let increase := (rate2 - rate1) / rate1 * 100
  ∃ ε > 0, |increase - 63.24| < ε :=
by sorry

end NUMINAMATH_CALUDE_response_rate_increase_l1158_115898


namespace NUMINAMATH_CALUDE_banana_muffins_count_l1158_115850

/-- Represents the types of pastries in the shop -/
inductive Pastry
  | PlainDoughnut
  | GlazedDoughnut
  | ChocolateChipCookie
  | OatmealCookie
  | BlueberryMuffin
  | BananaMuffin

/-- The ratio of pastries in the shop -/
def pastryRatio : Pastry → ℕ
  | Pastry.PlainDoughnut => 5
  | Pastry.GlazedDoughnut => 4
  | Pastry.ChocolateChipCookie => 3
  | Pastry.OatmealCookie => 2
  | Pastry.BlueberryMuffin => 1
  | Pastry.BananaMuffin => 2

/-- The number of plain doughnuts in the shop -/
def numPlainDoughnuts : ℕ := 50

/-- Theorem stating that the number of banana muffins is 20 -/
theorem banana_muffins_count :
  (numPlainDoughnuts / pastryRatio Pastry.PlainDoughnut) * pastryRatio Pastry.BananaMuffin = 20 := by
  sorry

end NUMINAMATH_CALUDE_banana_muffins_count_l1158_115850


namespace NUMINAMATH_CALUDE_smallest_n_for_sum_equation_l1158_115815

theorem smallest_n_for_sum_equation : ∃ (n : ℕ), n = 835 ∧ 
  (∀ (S : Finset ℕ), S.card = n → (∀ x ∈ S, x ≥ 1 ∧ x ≤ 999) → 
    (∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      a + 2*b + 3*c = d)) ∧
  (∀ (m : ℕ), m < n → 
    ∃ (T : Finset ℕ), T.card = m ∧ (∀ x ∈ T, x ≥ 1 ∧ x ≤ 999) ∧
      ¬(∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧ 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
        a + 2*b + 3*c = d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_sum_equation_l1158_115815


namespace NUMINAMATH_CALUDE_cricket_game_overs_l1158_115877

/-- Proves that the number of overs played in the first part of a cricket game is 10,
    given the specified conditions. -/
theorem cricket_game_overs (total_target : ℝ) (first_run_rate : ℝ) 
  (remaining_overs : ℝ) (remaining_run_rate : ℝ) 
  (h1 : total_target = 282)
  (h2 : first_run_rate = 3.2)
  (h3 : remaining_overs = 40)
  (h4 : remaining_run_rate = 6.25) :
  (total_target - remaining_overs * remaining_run_rate) / first_run_rate = 10 := by
sorry

end NUMINAMATH_CALUDE_cricket_game_overs_l1158_115877


namespace NUMINAMATH_CALUDE_orthogonal_matrix_sum_of_squares_l1158_115899

theorem orthogonal_matrix_sum_of_squares (p q r s : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![p, q; r, s]
  (B.transpose = B⁻¹) → (p = s) → p^2 + q^2 + r^2 + s^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_orthogonal_matrix_sum_of_squares_l1158_115899


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_theorem_l1158_115851

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (inside_plane : Line → Plane → Prop)
variable (skew_lines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_theorem 
  (a b : Line) (α : Plane) 
  (h1 : parallel_line_plane a α) 
  (h2 : inside_plane b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_theorem_l1158_115851
