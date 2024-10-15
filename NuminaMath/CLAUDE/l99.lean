import Mathlib

namespace NUMINAMATH_CALUDE_collinear_vectors_m_equals_six_l99_9966

/-- Two vectors are collinear if the determinant of their components is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given plane vectors a and b, if they are collinear, then m = 6 -/
theorem collinear_vectors_m_equals_six :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-3, m)
  collinear a b → m = 6 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_equals_six_l99_9966


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_two_variables_l99_9948

theorem arithmetic_geometric_mean_inequality_two_variables
  (a b : ℝ) : (a^2 + b^2) / 2 ≥ a * b ∧ 
  ((a^2 + b^2) / 2 = a * b ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_two_variables_l99_9948


namespace NUMINAMATH_CALUDE_alla_boris_meeting_l99_9989

/-- The number of lampposts along the alley -/
def total_lampposts : ℕ := 400

/-- The lamppost number where Alla is observed -/
def alla_observed : ℕ := 55

/-- The lamppost number where Boris is observed -/
def boris_observed : ℕ := 321

/-- The function to calculate the meeting point of Alla and Boris -/
def meeting_point : ℕ :=
  let intervals_covered := (alla_observed - 1) + (total_lampposts - boris_observed)
  let total_intervals := total_lampposts - 1
  (intervals_covered * 3) + 1

/-- Theorem stating that Alla and Boris meet at lamppost 163 -/
theorem alla_boris_meeting :
  meeting_point = 163 := by sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_l99_9989


namespace NUMINAMATH_CALUDE_quadratic_sum_l99_9946

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4 * x^2 - 28 * x - 48

-- Define the completed square form
def g (x a b c : ℝ) : ℝ := a * (x + b)^2 + c

-- Theorem statement
theorem quadratic_sum (a b c : ℝ) :
  (∀ x, f x = g x a b c) → a + b + c = -96.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l99_9946


namespace NUMINAMATH_CALUDE_inequality_proof_l99_9969

theorem inequality_proof (x a b : ℝ) (h1 : x < b) (h2 : b < a) (h3 : a < 0) :
  x^2 > a*b ∧ a*b > a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l99_9969


namespace NUMINAMATH_CALUDE_constant_derivative_implies_linear_l99_9953

/-- A function whose derivative is zero everywhere has a straight line graph -/
theorem constant_derivative_implies_linear (f : ℝ → ℝ) :
  (∀ x, deriv f x = 0) → ∃ a b : ℝ, ∀ x, f x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_constant_derivative_implies_linear_l99_9953


namespace NUMINAMATH_CALUDE_scatter_plot_suitable_for_linear_relationship_only_scatter_plot_suitable_for_linear_relationship_l99_9986

/-- A type representing different types of plots --/
inductive PlotType
  | ScatterPlot
  | StemAndLeafPlot
  | FrequencyDistributionHistogram
  | FrequencyDistributionLineChart

/-- A function that determines if a plot type is suitable for identifying linear relationships --/
def isSuitableForLinearRelationship (plot : PlotType) : Prop :=
  match plot with
  | PlotType.ScatterPlot => true
  | _ => false

/-- Theorem stating that a scatter plot is suitable for identifying linear relationships --/
theorem scatter_plot_suitable_for_linear_relationship :
  isSuitableForLinearRelationship PlotType.ScatterPlot :=
sorry

/-- Theorem stating that a scatter plot is the only suitable plot type for identifying linear relationships --/
theorem only_scatter_plot_suitable_for_linear_relationship (plot : PlotType) :
  isSuitableForLinearRelationship plot → plot = PlotType.ScatterPlot :=
sorry

end NUMINAMATH_CALUDE_scatter_plot_suitable_for_linear_relationship_only_scatter_plot_suitable_for_linear_relationship_l99_9986


namespace NUMINAMATH_CALUDE_max_distance_a_c_theorem_l99_9976

def max_distance_a_c (a b c : ℝ × ℝ) : Prop :=
  let norm := λ v : ℝ × ℝ => Real.sqrt (v.1^2 + v.2^2)
  let dot := λ u v : ℝ × ℝ => u.1 * v.1 + u.2 * v.2
  norm a = 2 ∧ 
  norm b = 2 ∧ 
  dot a b = 2 ∧ 
  dot c (a + 2 • b - 2 • c) = 2 →
  (∀ c', dot c' (a + 2 • b - 2 • c') = 2 → 
    norm (a - c) ≤ (Real.sqrt 3 + Real.sqrt 7) / 2) ∧
  (∃ c', dot c' (a + 2 • b - 2 • c') = 2 ∧ 
    norm (a - c') = (Real.sqrt 3 + Real.sqrt 7) / 2)

theorem max_distance_a_c_theorem (a b c : ℝ × ℝ) : 
  max_distance_a_c a b c := by sorry

end NUMINAMATH_CALUDE_max_distance_a_c_theorem_l99_9976


namespace NUMINAMATH_CALUDE_exam_failure_percentage_l99_9960

theorem exam_failure_percentage :
  let total_candidates : ℕ := 2000
  let girls : ℕ := 900
  let boys : ℕ := total_candidates - girls
  let boys_pass_rate : ℚ := 34 / 100
  let girls_pass_rate : ℚ := 32 / 100
  let passed_candidates : ℚ := boys_pass_rate * boys + girls_pass_rate * girls
  let failed_candidates : ℚ := total_candidates - passed_candidates
  let failure_percentage : ℚ := failed_candidates / total_candidates * 100
  failure_percentage = 669 / 10 := by sorry

end NUMINAMATH_CALUDE_exam_failure_percentage_l99_9960


namespace NUMINAMATH_CALUDE_circle_equation_l99_9980

/-- Prove that a circle with given properties has the equation (x+5)^2 + y^2 = 5 -/
theorem circle_equation (a : ℝ) (h1 : a < 0) :
  let O' : ℝ × ℝ := (a, 0)
  let r : ℝ := Real.sqrt 5
  let line : ℝ × ℝ → Prop := λ p => p.1 + 2 * p.2 = 0
  (∀ p, line p → (p.1 - O'.1)^2 + (p.2 - O'.2)^2 = r^2) →
  (∀ x y, (x + 5)^2 + y^2 = 5 ↔ (x - a)^2 + y^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l99_9980


namespace NUMINAMATH_CALUDE_sum_of_consecutive_iff_not_power_of_two_l99_9905

def is_sum_of_consecutive (n : ℕ) : Prop :=
  ∃ (a k : ℕ), k > 0 ∧ n = (k * (2 * a + k - 1)) / 2

def is_power_of_two (n : ℕ) : Prop :=
  ∃ (s : ℕ), n = 2^s

theorem sum_of_consecutive_iff_not_power_of_two (n : ℕ) :
  ¬(is_sum_of_consecutive n) ↔ is_power_of_two n :=
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_iff_not_power_of_two_l99_9905


namespace NUMINAMATH_CALUDE_student_line_length_l99_9981

/-- The length of a line of students, given the number of students and the distance between them. -/
def line_length (num_students : ℕ) (distance : ℝ) : ℝ :=
  (num_students - 1 : ℝ) * distance

/-- Theorem stating that the length of a line formed by 51 students with 3 meters between each adjacent pair is 150 meters. -/
theorem student_line_length : line_length 51 3 = 150 := by
  sorry

#eval line_length 51 3

end NUMINAMATH_CALUDE_student_line_length_l99_9981


namespace NUMINAMATH_CALUDE_smallest_integer_l99_9996

theorem smallest_integer (a b : ℕ) (ha : a = 80) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 40) :
  ∃ (m : ℕ), m ≥ b ∧ m = 50 ∧ Nat.lcm a m / Nat.gcd a m = 40 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l99_9996


namespace NUMINAMATH_CALUDE_elizabeth_haircut_l99_9979

theorem elizabeth_haircut (first_cut second_cut : ℝ) 
  (h1 : first_cut = 0.375)
  (h2 : second_cut = 0.5) :
  first_cut + second_cut = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_haircut_l99_9979


namespace NUMINAMATH_CALUDE_subtracted_value_l99_9930

theorem subtracted_value (x y : ℤ) (h1 : x = 122) (h2 : 2 * x - y = 106) : y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l99_9930


namespace NUMINAMATH_CALUDE_point_four_units_from_negative_two_l99_9985

theorem point_four_units_from_negative_two (x : ℝ) : 
  (|x - (-2)| = 4) ↔ (x = 2 ∨ x = -6) := by sorry

end NUMINAMATH_CALUDE_point_four_units_from_negative_two_l99_9985


namespace NUMINAMATH_CALUDE_stickers_per_page_l99_9921

theorem stickers_per_page (total_pages : ℕ) (remaining_stickers : ℕ) : 
  total_pages = 12 →
  remaining_stickers = 220 →
  (total_pages - 1) * (remaining_stickers / (total_pages - 1)) = remaining_stickers →
  remaining_stickers / (total_pages - 1) = 20 := by
sorry

end NUMINAMATH_CALUDE_stickers_per_page_l99_9921


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l99_9964

theorem divisibility_by_twelve (n : Nat) : n ≤ 9 → (512 * 10 + n) % 12 = 0 ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l99_9964


namespace NUMINAMATH_CALUDE_smallest_non_odd_ending_digit_l99_9954

def is_odd_ending_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def is_digit (n : ℕ) : Prop :=
  n ≤ 9

theorem smallest_non_odd_ending_digit :
  ∀ d : ℕ, is_digit d → 
    (¬is_odd_ending_digit d → d ≥ 0) ∧
    (∀ d' : ℕ, is_digit d' → ¬is_odd_ending_digit d' → d ≤ d') :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_odd_ending_digit_l99_9954


namespace NUMINAMATH_CALUDE_cookie_distribution_l99_9925

theorem cookie_distribution (boxes : ℕ) (classes : ℕ) 
  (h1 : boxes = 3) (h2 : classes = 4) :
  (boxes : ℚ) / classes = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l99_9925


namespace NUMINAMATH_CALUDE_right_triangle_one_one_sqrt_two_l99_9995

theorem right_triangle_one_one_sqrt_two :
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 2
  a^2 + b^2 = c^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_one_one_sqrt_two_l99_9995


namespace NUMINAMATH_CALUDE_rectangle_area_from_circular_wire_l99_9924

/-- The area of a rectangle formed by bending a circular wire -/
theorem rectangle_area_from_circular_wire (r : ℝ) (ratio_l : ℝ) (ratio_b : ℝ) : 
  r = 3.5 → 
  ratio_l = 6 → 
  ratio_b = 5 → 
  let circumference := 2 * π * r
  let length := (circumference * ratio_l) / (2 * (ratio_l + ratio_b))
  let breadth := (circumference * ratio_b) / (2 * (ratio_l + ratio_b))
  length * breadth = (735 * π^2) / 242 := by
  sorry

#check rectangle_area_from_circular_wire

end NUMINAMATH_CALUDE_rectangle_area_from_circular_wire_l99_9924


namespace NUMINAMATH_CALUDE_game_probability_l99_9950

/-- Represents the probability of winning for each player -/
structure PlayerProbabilities where
  alex : ℝ
  mel : ℝ
  chelsea : ℝ
  sam : ℝ

/-- Calculates the probability of a specific outcome in the game -/
def probability_of_outcome (probs : PlayerProbabilities) : ℝ :=
  probs.alex^3 * probs.mel^2 * probs.chelsea^2 * probs.sam

/-- The number of ways to arrange the wins -/
def number_of_arrangements : ℕ := 420

theorem game_probability (probs : PlayerProbabilities) 
  (h1 : probs.alex = 1/3)
  (h2 : probs.mel = 3 * probs.sam)
  (h3 : probs.chelsea = probs.sam)
  (h4 : probs.alex + probs.mel + probs.chelsea + probs.sam = 1) :
  (probability_of_outcome probs) * (number_of_arrangements : ℝ) = 13440/455625 := by
  sorry


end NUMINAMATH_CALUDE_game_probability_l99_9950


namespace NUMINAMATH_CALUDE_project_budget_l99_9908

theorem project_budget (total_spent : ℕ) (over_budget : ℕ) : 
  total_spent = 6580 →
  over_budget = 280 →
  ∃ (monthly_allocation : ℕ),
    monthly_allocation * 6 = total_spent - over_budget ∧
    monthly_allocation * 12 = 12600 := by
  sorry

end NUMINAMATH_CALUDE_project_budget_l99_9908


namespace NUMINAMATH_CALUDE_julys_husband_age_l99_9901

/-- Given information about Hannah and July's ages, and July's husband's age relative to July,
    prove that July's husband is 25 years old. -/
theorem julys_husband_age :
  ∀ (hannah_initial_age : ℕ) 
    (july_initial_age : ℕ) 
    (years_passed : ℕ) 
    (age_difference_husband : ℕ),
  hannah_initial_age = 6 →
  hannah_initial_age = 2 * july_initial_age →
  years_passed = 20 →
  age_difference_husband = 2 →
  july_initial_age + years_passed + age_difference_husband = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_julys_husband_age_l99_9901


namespace NUMINAMATH_CALUDE_curve_C_properties_l99_9952

noncomputable section

/-- Curve C in parametric form -/
def curve_C (φ : ℝ) : ℝ × ℝ := (3 * Real.cos φ, 3 + 3 * Real.sin φ)

/-- Polar equation of a curve -/
structure PolarEquation where
  f : ℝ → ℝ

/-- Line with slope angle and passing through a point -/
structure Line where
  slope_angle : ℝ
  point : ℝ × ℝ

/-- Intersection points of a line and a curve -/
structure Intersection where
  M : ℝ × ℝ
  N : ℝ × ℝ

/-- Main theorem statement -/
theorem curve_C_properties :
  ∃ (polar_eq : PolarEquation) (l : Line) (int : Intersection),
    (∀ θ : ℝ, polar_eq.f θ = 6 * Real.sin θ) ∧
    l.slope_angle = 135 * π / 180 ∧
    l.point = (1, 2) ∧
    (let (xM, yM) := int.M
     let (xN, yN) := int.N
     1 / Real.sqrt ((xM - 1)^2 + (yM - 2)^2) +
     1 / Real.sqrt ((xN - 1)^2 + (yN - 2)^2) = 6 / 7) := by
  sorry

end

end NUMINAMATH_CALUDE_curve_C_properties_l99_9952


namespace NUMINAMATH_CALUDE_find_S_value_l99_9934

/-- Represents the relationship between R, S, and T -/
def relationship (R S T : ℝ) : Prop :=
  ∃ (c : ℝ), R = c * S / T

theorem find_S_value (R₁ S₁ T₁ R₂ T₂ : ℝ) :
  relationship R₁ S₁ T₁ →
  R₁ = 4/3 →
  S₁ = 3/7 →
  T₁ = 9/14 →
  R₂ = Real.sqrt 48 →
  T₂ = Real.sqrt 75 →
  ∃ (S₂ : ℝ), relationship R₂ S₂ T₂ ∧ S₂ = 30 :=
by sorry

end NUMINAMATH_CALUDE_find_S_value_l99_9934


namespace NUMINAMATH_CALUDE_midpoint_movement_l99_9999

/-- Given two points A and B with midpoint M, prove the new midpoint M' and distance between M and M' after moving A and B -/
theorem midpoint_movement (a b c d m n : ℝ) :
  m = (a + c) / 2 →
  n = (b + d) / 2 →
  let m' := (a + 4 + c - 15) / 2
  let n' := (b + 12 + d - 5) / 2
  (m' = m - 11 / 2 ∧ n' = n + 7 / 2) ∧
  Real.sqrt ((m' - m) ^ 2 + (n' - n) ^ 2) = Real.sqrt 42.5 := by
  sorry


end NUMINAMATH_CALUDE_midpoint_movement_l99_9999


namespace NUMINAMATH_CALUDE_lace_makers_combined_time_l99_9956

theorem lace_makers_combined_time (t1 t2 T : ℚ) : 
  t1 = 8 → t2 = 13 → (1 / t1 + 1 / t2) * T = 1 → T = 104 / 21 := by
  sorry

end NUMINAMATH_CALUDE_lace_makers_combined_time_l99_9956


namespace NUMINAMATH_CALUDE_mitten_plug_difference_l99_9942

theorem mitten_plug_difference (mittens : ℕ) (added_plugs : ℕ) (total_plugs : ℕ) : 
  mittens = 150 → added_plugs = 30 → total_plugs = 400 →
  (total_plugs / 2 - added_plugs) - mittens = 20 := by
  sorry

end NUMINAMATH_CALUDE_mitten_plug_difference_l99_9942


namespace NUMINAMATH_CALUDE_exam_time_allocation_l99_9965

theorem exam_time_allocation (total_questions : ℕ) (exam_duration_hours : ℕ) 
  (type_a_problems : ℕ) (h1 : total_questions = 200) (h2 : exam_duration_hours = 3) 
  (h3 : type_a_problems = 25) :
  let exam_duration_minutes : ℕ := exam_duration_hours * 60
  let type_b_problems : ℕ := total_questions - type_a_problems
  let x : ℚ := (exam_duration_minutes : ℚ) / (type_a_problems * 2 + type_b_problems)
  (2 * x * type_a_problems : ℚ) = 40 := by
  sorry

#check exam_time_allocation

end NUMINAMATH_CALUDE_exam_time_allocation_l99_9965


namespace NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l99_9958

theorem square_difference_formula_inapplicable :
  ∀ (a b : ℝ), ¬∃ (x y : ℝ), (-a + b) * (-b + a) = x^2 - y^2 := by
  sorry

#check square_difference_formula_inapplicable

end NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l99_9958


namespace NUMINAMATH_CALUDE_brittany_second_test_score_l99_9929

/-- Proves that given the conditions of Brittany's test scores, her second test score must be 83. -/
theorem brittany_second_test_score
  (first_test_score : ℝ)
  (first_test_weight : ℝ)
  (second_test_weight : ℝ)
  (final_weighted_average : ℝ)
  (h1 : first_test_score = 78)
  (h2 : first_test_weight = 0.4)
  (h3 : second_test_weight = 0.6)
  (h4 : final_weighted_average = 81)
  (h5 : first_test_weight + second_test_weight = 1) :
  ∃ (second_test_score : ℝ),
    first_test_weight * first_test_score + second_test_weight * second_test_score = final_weighted_average ∧
    second_test_score = 83 :=
by sorry

end NUMINAMATH_CALUDE_brittany_second_test_score_l99_9929


namespace NUMINAMATH_CALUDE_circle_a_l99_9971

theorem circle_a (x y : ℝ) : 
  (x - 3)^2 + (y + 2)^2 = 16 → (∃ (center : ℝ × ℝ) (radius : ℝ), center = (3, -2) ∧ radius = 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_a_l99_9971


namespace NUMINAMATH_CALUDE_hotel_assignment_count_l99_9997

/-- Represents a hotel with a specific number of rooms and guests -/
structure Hotel :=
  (num_rooms : ℕ)
  (num_guests : ℕ)

/-- Represents the constraints for room assignments -/
structure RoomConstraints :=
  (max_guests_regular : ℕ)
  (min_guests_deluxe : ℕ)
  (max_guests_deluxe : ℕ)

/-- Calculates the number of valid room assignments -/
def count_valid_assignments (h : Hotel) (c : RoomConstraints) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem hotel_assignment_count :
  let h : Hotel := ⟨7, 7⟩
  let c : RoomConstraints := ⟨3, 2, 3⟩
  count_valid_assignments h c = 27720 :=
sorry

end NUMINAMATH_CALUDE_hotel_assignment_count_l99_9997


namespace NUMINAMATH_CALUDE_handshakes_at_event_l99_9961

/-- Represents the number of married couples at the event -/
def num_couples : ℕ := 15

/-- Calculates the total number of handshakes at the event -/
def total_handshakes (n : ℕ) : ℕ :=
  let num_men := n
  let num_women := n
  let handshakes_among_men := n * (n - 1) / 2
  let handshakes_men_women := n * (n - 1)
  handshakes_among_men + handshakes_men_women

/-- Theorem stating that the total number of handshakes is 315 -/
theorem handshakes_at_event : 
  total_handshakes num_couples = 315 := by
  sorry

#eval total_handshakes num_couples

end NUMINAMATH_CALUDE_handshakes_at_event_l99_9961


namespace NUMINAMATH_CALUDE_expected_black_pairs_in_circular_deal_l99_9940

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of black cards in a standard deck -/
def BlackCards : ℕ := 26

/-- Expected number of pairs of adjacent black cards in a circular deal -/
def ExpectedBlackPairs : ℚ := 650 / 51

theorem expected_black_pairs_in_circular_deal :
  let total_cards := StandardDeck
  let black_cards := BlackCards
  let prob_next_black : ℚ := (black_cards - 1) / (total_cards - 1)
  black_cards * prob_next_black = ExpectedBlackPairs :=
sorry

end NUMINAMATH_CALUDE_expected_black_pairs_in_circular_deal_l99_9940


namespace NUMINAMATH_CALUDE_min_triangulation_l99_9932

/-- A regular polygon with n sides, where n ≥ 5 -/
structure RegularPolygon where
  n : ℕ
  n_ge_5 : n ≥ 5

/-- A triangulation of a regular polygon -/
structure Triangulation (p : RegularPolygon) where
  num_triangles : ℕ
  is_valid : Bool  -- Represents the validity of the triangulation

/-- The number of acute triangles in a valid triangulation is at least n -/
def min_acute_triangles (p : RegularPolygon) (t : Triangulation p) : Prop :=
  t.is_valid → t.num_triangles ≥ p.n

/-- The number of obtuse triangles in a valid triangulation is at least n -/
def min_obtuse_triangles (p : RegularPolygon) (t : Triangulation p) : Prop :=
  t.is_valid → t.num_triangles ≥ p.n

/-- The main theorem: both acute and obtuse triangulations have a minimum of n triangles -/
theorem min_triangulation (p : RegularPolygon) :
  (∀ t : Triangulation p, min_acute_triangles p t) ∧
  (∀ t : Triangulation p, min_obtuse_triangles p t) :=
sorry

end NUMINAMATH_CALUDE_min_triangulation_l99_9932


namespace NUMINAMATH_CALUDE_function_equation_solution_l99_9947

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) * (x + f y) = x^2 * f y + y^2 * f x) →
  (∀ x : ℝ, f x = 0 ∨ f x = x) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l99_9947


namespace NUMINAMATH_CALUDE_angle_east_southwest_is_135_l99_9945

/-- Represents a circle with 8 equally spaced rays --/
structure EightRayCircle where
  /-- The measure of the angle between adjacent rays in degrees --/
  angle_between_rays : ℝ
  /-- The angle between adjacent rays is 45° --/
  angle_is_45 : angle_between_rays = 45

/-- The measure of the smaller angle between East and Southwest rays in degrees --/
def angle_east_southwest (circle : EightRayCircle) : ℝ :=
  3 * circle.angle_between_rays

theorem angle_east_southwest_is_135 (circle : EightRayCircle) :
  angle_east_southwest circle = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_east_southwest_is_135_l99_9945


namespace NUMINAMATH_CALUDE_max_sum_of_functions_l99_9949

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem max_sum_of_functions :
  (∀ x, -6 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -3 ≤ g x ∧ g x ≤ 2) →
  (∃ d, ∀ x, f x + g x ≤ d ∧ ∃ y, f y + g y = d) →
  ∃ d, d = 6 ∧ ∀ x, f x + g x ≤ d ∧ ∃ y, f y + g y = d :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_functions_l99_9949


namespace NUMINAMATH_CALUDE_sanchez_grade_calculation_l99_9909

theorem sanchez_grade_calculation (total_students : ℕ) (below_b_percentage : ℚ) 
  (h1 : total_students = 60) 
  (h2 : below_b_percentage = 40 / 100) : 
  ↑total_students * (1 - below_b_percentage) = 36 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_grade_calculation_l99_9909


namespace NUMINAMATH_CALUDE_function_inequality_l99_9903

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h' : ∀ x, (x - 2) * deriv f x ≤ 0) : 
  f (-3) + f 3 ≤ 2 * f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l99_9903


namespace NUMINAMATH_CALUDE_car_price_proof_l99_9963

-- Define the original cost price
def original_price : ℝ := 52325.58

-- Define the first sale price (14% loss)
def first_sale_price : ℝ := original_price * 0.86

-- Define the second sale price (20% gain from first sale)
def second_sale_price : ℝ := 54000

-- Theorem statement
theorem car_price_proof :
  (first_sale_price * 1.2 = second_sale_price) ∧
  (original_price > 0) ∧
  (first_sale_price > 0) ∧
  (second_sale_price > 0) :=
sorry

end NUMINAMATH_CALUDE_car_price_proof_l99_9963


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_20_6_sixty_divisible_by_12_20_6_sixty_is_smallest_l99_9923

theorem smallest_divisible_by_12_20_6 : ∀ n : ℕ, n > 0 → (12 ∣ n) → (20 ∣ n) → (6 ∣ n) → n ≥ 60 := by
  sorry

theorem sixty_divisible_by_12_20_6 : (12 ∣ 60) ∧ (20 ∣ 60) ∧ (6 ∣ 60) := by
  sorry

theorem sixty_is_smallest :
  ∀ n : ℕ, n > 0 → (12 ∣ n) → (20 ∣ n) → (6 ∣ n) → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_20_6_sixty_divisible_by_12_20_6_sixty_is_smallest_l99_9923


namespace NUMINAMATH_CALUDE_exists_counterexample_l99_9900

-- Define a structure for the set S with the binary operation *
structure BinarySystem where
  S : Type u
  op : S → S → S
  at_least_two_elements : ∃ (a b : S), a ≠ b
  property : ∀ (a b : S), op a (op b a) = b

-- State the theorem
theorem exists_counterexample (B : BinarySystem) :
  ∃ (a b : B.S), B.op (B.op a b) a ≠ a := by sorry

end NUMINAMATH_CALUDE_exists_counterexample_l99_9900


namespace NUMINAMATH_CALUDE_circle_symmetry_l99_9902

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Define the symmetrical circle
def symmetrical_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Define symmetry with respect to the origin
def symmetrical_wrt_origin (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ g (-x) (-y)

-- Theorem statement
theorem circle_symmetry :
  symmetrical_wrt_origin original_circle symmetrical_circle :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l99_9902


namespace NUMINAMATH_CALUDE_company_a_profit_share_l99_9916

/-- Prove that Company A's share of combined profits is 60% given the conditions -/
theorem company_a_profit_share :
  ∀ (total_profit : ℝ) (company_b_profit : ℝ) (company_a_profit : ℝ),
    company_b_profit = 0.4 * total_profit →
    company_b_profit = 60000 →
    company_a_profit = 90000 →
    company_a_profit / total_profit = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_company_a_profit_share_l99_9916


namespace NUMINAMATH_CALUDE_dice_configuration_dots_l99_9920

/-- Represents a die face with a number of dots -/
structure DieFace where
  dots : Nat
  valid : dots ≥ 1 ∧ dots ≤ 6

/-- Represents a die with six faces -/
structure Die where
  faces : Fin 6 → DieFace
  sum_opposite : ∀ i : Fin 3, (faces i).dots + (faces (i + 3)).dots = 7

/-- Represents the configuration of 4 dice glued together -/
structure DiceConfiguration where
  dice : Fin 4 → Die
  face_c : DieFace
  face_c_is_six : face_c.dots = 6

/-- The theorem to be proved -/
theorem dice_configuration_dots (config : DiceConfiguration) :
  ∃ (face_a face_b face_d : DieFace),
    face_a.dots = 3 ∧
    face_b.dots = 5 ∧
    config.face_c.dots = 6 ∧
    face_d.dots = 5 := by
  sorry

end NUMINAMATH_CALUDE_dice_configuration_dots_l99_9920


namespace NUMINAMATH_CALUDE_root_negative_implies_inequality_l99_9911

theorem root_negative_implies_inequality (a : ℝ) : 
  (∃ x : ℝ, x - 2*a + 4 = 0 ∧ x < 0) → (a - 3) * (a - 4) > 0 := by
  sorry

end NUMINAMATH_CALUDE_root_negative_implies_inequality_l99_9911


namespace NUMINAMATH_CALUDE_stating_days_worked_when_net_zero_l99_9951

/-- Represents the number of days in the work period -/
def total_days : ℕ := 30

/-- Represents the daily wage in su -/
def daily_wage : ℕ := 24

/-- Represents the daily penalty for skipping work in su -/
def daily_penalty : ℕ := 6

/-- 
Theorem stating that if a worker's net earnings are zero after the work period,
given the specified daily wage and penalty, then the number of days worked is 6.
-/
theorem days_worked_when_net_zero : 
  ∀ (days_worked : ℕ), 
    days_worked ≤ total_days →
    (daily_wage * days_worked - daily_penalty * (total_days - days_worked) = 0) →
    days_worked = 6 := by
  sorry

end NUMINAMATH_CALUDE_stating_days_worked_when_net_zero_l99_9951


namespace NUMINAMATH_CALUDE_divisible_by_six_l99_9939

theorem divisible_by_six (n : ℕ) : ∃ k : ℤ, (n - 1) * n * (n^3 + 1) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l99_9939


namespace NUMINAMATH_CALUDE_total_notes_count_l99_9915

/-- Given a total amount of 192 rupees in equal numbers of 1-rupee, 5-rupee, and 10-rupee notes,
    prove that the total number of notes is 36. -/
theorem total_notes_count (total_amount : ℕ) (note_count : ℕ) : 
  total_amount = 192 →
  note_count * 1 + note_count * 5 + note_count * 10 = total_amount →
  3 * note_count = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_total_notes_count_l99_9915


namespace NUMINAMATH_CALUDE_fruit_cost_l99_9944

/-- The cost of fruit combinations -/
theorem fruit_cost (x y z : ℚ) : 
  (2 * x + y + 4 * z = 6) →
  (4 * x + 2 * y + 2 * z = 4) →
  (4 * x + 2 * y + 5 * z = 8) :=
by sorry

end NUMINAMATH_CALUDE_fruit_cost_l99_9944


namespace NUMINAMATH_CALUDE_math_homework_pages_l99_9917

-- Define the variables
def reading_pages : ℕ := 4
def problems_per_page : ℕ := 3
def total_problems : ℕ := 30

-- Define the theorem
theorem math_homework_pages :
  ∃ (math_pages : ℕ), 
    math_pages * problems_per_page + reading_pages * problems_per_page = total_problems ∧
    math_pages = 6 := by
  sorry

end NUMINAMATH_CALUDE_math_homework_pages_l99_9917


namespace NUMINAMATH_CALUDE_least_k_cube_divisible_by_120_l99_9973

theorem least_k_cube_divisible_by_120 :
  ∀ k : ℕ, k > 0 → k^3 % 120 = 0 → k ≥ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_least_k_cube_divisible_by_120_l99_9973


namespace NUMINAMATH_CALUDE_min_value_inequality_l99_9993

def f (x : ℝ) : ℝ := |3*x - 1| + |x + 1|

def g (x : ℝ) : ℝ := f x + 2*|x + 1|

theorem min_value_inequality (a b : ℝ) 
  (h1 : ∀ x, g x ≥ a^2 + b^2) 
  (h2 : ∃ x, g x = a^2 + b^2) : 
  1 / (a^2 + 1) + 4 / (b^2 + 1) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l99_9993


namespace NUMINAMATH_CALUDE_notebooks_distribution_l99_9955

theorem notebooks_distribution (C : ℕ) (N : ℕ) : 
  (N / C = C / 8) →  -- Each child got one-eighth of the number of children in notebooks
  (N / (C / 2) = 16) →  -- If number of children halved, each would get 16 notebooks
  N = 512 := by  -- Total notebooks distributed is 512
sorry

end NUMINAMATH_CALUDE_notebooks_distribution_l99_9955


namespace NUMINAMATH_CALUDE_star_drawing_probability_l99_9928

def total_stars : ℕ := 12
def red_stars : ℕ := 3
def gold_stars : ℕ := 4
def silver_stars : ℕ := 5
def stars_drawn : ℕ := 6

theorem star_drawing_probability : 
  (red_stars / total_stars) * 
  (Nat.choose gold_stars 3 * Nat.choose silver_stars 2) / 
  (Nat.choose (total_stars - 1) (stars_drawn - 1)) = 5 / 231 := by
  sorry

end NUMINAMATH_CALUDE_star_drawing_probability_l99_9928


namespace NUMINAMATH_CALUDE_circle_radius_problem_l99_9926

theorem circle_radius_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  (π * x^2 = π * y^2) →   -- Circles have the same area
  (2 * π * x = 12 * π) →  -- Circumference of circle x is 12π
  (∃ v, y = 2 * v) →      -- Radius of circle y is twice some value v
  (∃ v, y = 2 * v ∧ v = 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l99_9926


namespace NUMINAMATH_CALUDE_lenny_money_left_l99_9984

/-- Calculates the amount of money Lenny has left after his expenses -/
def money_left (initial : ℕ) (expense1 : ℕ) (expense2 : ℕ) : ℕ :=
  initial - (expense1 + expense2)

/-- Proves that Lenny has $39 left after his expenses -/
theorem lenny_money_left :
  money_left 84 24 21 = 39 := by
  sorry

end NUMINAMATH_CALUDE_lenny_money_left_l99_9984


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l99_9936

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt ((2 / x) + 2) = 3 / 2 → x = 8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l99_9936


namespace NUMINAMATH_CALUDE_red_ball_probability_l99_9910

/-- Given a bag of balls with the following properties:
  * There are n total balls
  * There are m white balls
  * The probability of drawing at least one red ball when two balls are drawn is 3/5
  * The expected number of white balls in 6 draws with replacement is 4
  Prove that the probability of drawing a red ball on the second draw,
  given that the first draw was red, is 1/5. -/
theorem red_ball_probability (n m : ℕ) 
  (h1 : 1 - (m.choose 2 : ℚ) / (n.choose 2 : ℚ) = 3/5)
  (h2 : 6 * (m : ℚ) / (n : ℚ) = 4) :
  (n - m : ℚ) / ((n - 1) : ℚ) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_l99_9910


namespace NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l99_9970

theorem product_of_sums_equal_difference_of_powers : 
  (5 + 3) * (5^2 + 3^2) * (5^4 + 3^4) * (5^8 + 3^8) * 
  (5^16 + 3^16) * (5^32 + 3^32) * (5^64 + 3^64) = 5^128 - 3^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l99_9970


namespace NUMINAMATH_CALUDE_min_value_of_sequence_l99_9941

theorem min_value_of_sequence (a : ℕ → ℝ) :
  a 1 = 2 →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2 * n) →
  (∀ n : ℕ, n ≥ 1 → a n / n ≥ 2) ∧ (∃ n : ℕ, n ≥ 1 ∧ a n / n = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sequence_l99_9941


namespace NUMINAMATH_CALUDE_triangle_inequality_l99_9962

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) :
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l99_9962


namespace NUMINAMATH_CALUDE_lars_production_l99_9972

/-- Represents the baking rates and working hours of Lars' bakeshop --/
structure BakeshopData where
  bread_rate : ℕ  -- loaves of bread per hour
  baguette_rate : ℕ  -- baguettes per 2 hours
  croissant_rate : ℕ  -- croissants per 75 minutes
  working_hours : ℕ  -- hours worked per day

/-- Calculates the daily production of baked goods --/
def daily_production (data : BakeshopData) : ℕ × ℕ × ℕ :=
  let bread := data.bread_rate * data.working_hours
  let baguettes := data.baguette_rate * (data.working_hours / 2)
  let croissants := data.croissant_rate * (data.working_hours * 60 / 75)
  (bread, baguettes, croissants)

/-- Theorem stating Lars' daily production --/
theorem lars_production :
  let data : BakeshopData := {
    bread_rate := 10,
    baguette_rate := 30,
    croissant_rate := 20,
    working_hours := 6
  }
  daily_production data = (60, 90, 80) := by
  sorry

end NUMINAMATH_CALUDE_lars_production_l99_9972


namespace NUMINAMATH_CALUDE_point_transformation_l99_9967

def rotate90CounterClockwise (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

def reflectAboutNegativeDiagonal (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90CounterClockwise a b 1 5
  let (x₂, y₂) := reflectAboutNegativeDiagonal x₁ y₁
  (x₂ = -6 ∧ y₂ = 3) → b - a = -5 := by sorry

end NUMINAMATH_CALUDE_point_transformation_l99_9967


namespace NUMINAMATH_CALUDE_pie_slices_served_yesterday_l99_9927

def slices_served_yesterday (lunch_today dinner_today total_today : ℕ) : ℕ :=
  total_today - (lunch_today + dinner_today)

theorem pie_slices_served_yesterday : 
  slices_served_yesterday 7 5 12 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_pie_slices_served_yesterday_l99_9927


namespace NUMINAMATH_CALUDE_julia_trip_euros_l99_9988

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Exchange rate from USD to EUR -/
def exchange_rate : ℚ := 8 / 5

theorem julia_trip_euros (d' : ℕ) : 
  (exchange_rate * d' - 80 : ℚ) = d' → sum_of_digits d' = 7 := by
  sorry

end NUMINAMATH_CALUDE_julia_trip_euros_l99_9988


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l99_9935

theorem negation_of_universal_proposition (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (¬ ∀ x : ℝ, a^x > 0) ↔ (∃ x₀ : ℝ, a^x₀ ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l99_9935


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_four_l99_9919

def binary_number : ℕ := 3005 -- 110110111101₂ in decimal

theorem remainder_of_binary_div_four :
  binary_number % 4 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_four_l99_9919


namespace NUMINAMATH_CALUDE_smallest_sum_m_p_l99_9994

/-- The function f(x) = arcsin(log_m(px)) has a domain that is a closed interval of length 1/1007 -/
def domain_length (m p : ℕ) : ℚ := (m^2 - 1 : ℚ) / (m * p)

/-- The theorem statement -/
theorem smallest_sum_m_p :
  ∀ m p : ℕ,
  m > 1 ∧ 
  p > 0 ∧ 
  domain_length m p = 1 / 1007 →
  m + p ≥ 2031 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_m_p_l99_9994


namespace NUMINAMATH_CALUDE_four_points_left_of_origin_l99_9906

theorem four_points_left_of_origin : 
  let points : List ℝ := [-(-8), (-1)^2023, -(3^2), -1-11, -2/5]
  (points.filter (· < 0)).length = 4 := by
sorry

end NUMINAMATH_CALUDE_four_points_left_of_origin_l99_9906


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l99_9931

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (9 / a + 1 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 9 / a₀ + 1 / b₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l99_9931


namespace NUMINAMATH_CALUDE_matching_segment_exists_l99_9991

/-- A 20-digit binary number -/
def BinaryNumber := Fin 20 → Bool

/-- A is a 20-digit binary number with 10 zeros and 10 ones -/
def is_valid_A (A : BinaryNumber) : Prop :=
  (Finset.filter (λ i => A i = false) Finset.univ).card = 10 ∧
  (Finset.filter (λ i => A i = true) Finset.univ).card = 10

/-- B is any 20-digit binary number -/
def B : BinaryNumber := sorry

/-- C is a 40-digit binary number formed by concatenating B with itself -/
def C : Fin 40 → Bool :=
  λ i => B (Fin.val i % 20)

/-- Count matching bits between two binary numbers -/
def count_matches (X Y : BinaryNumber) : Nat :=
  (Finset.filter (λ i => X i = Y i) Finset.univ).card

/-- Theorem: There exists a 20-bit segment of C with at least 10 matching bits with A -/
theorem matching_segment_exists (A : BinaryNumber) (h : is_valid_A A) :
  ∃ k : Fin 21, count_matches A (λ i => C (i + k)) ≥ 10 := by sorry

end NUMINAMATH_CALUDE_matching_segment_exists_l99_9991


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l99_9922

/-- The area of a rectangular plot with length thrice its breadth and breadth of 14 meters is 588 square meters. -/
theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 14 →
  length = 3 * breadth →
  area = length * breadth →
  area = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l99_9922


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_range_of_m_when_B_subset_A_l99_9998

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 < x ∧ x < m + 1}

-- Theorem 1: When m = 1, A ∩ B = { x | 1 < x < 2 }
theorem intersection_when_m_is_one :
  A ∩ B 1 = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 2: If B ⊆ A, then m ∈ [-1, +∞)
theorem range_of_m_when_B_subset_A :
  (∀ m : ℝ, B m ⊆ A) → {m : ℝ | -1 ≤ m} = Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_range_of_m_when_B_subset_A_l99_9998


namespace NUMINAMATH_CALUDE_min_value_a_plus_8b_l99_9975

theorem min_value_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = 2 * a + b) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x * y = 2 * x + y → a + 8 * b ≤ x + 8 * y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x * y = 2 * x + y ∧ x + 8 * y = 25 / 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_8b_l99_9975


namespace NUMINAMATH_CALUDE_five_integers_problem_l99_9990

theorem five_integers_problem : 
  ∃ (a b c d e : ℤ), 
    ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
      {3, 8, 9, 16, 17, 17, 18, 22, 23, 31} ∧
    a * b * c * d * e = 3360 := by
  sorry

end NUMINAMATH_CALUDE_five_integers_problem_l99_9990


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l99_9978

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let firstDescent := initialHeight
  let firstAscent := initialHeight * reboundFactor
  let secondDescent := firstAscent
  let secondAscent := firstAscent * reboundFactor
  let thirdDescent := secondAscent
  firstDescent + firstAscent + secondDescent + secondAscent + thirdDescent

/-- The theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  totalDistance 90 0.5 2 = 225 := by
  sorry

#eval totalDistance 90 0.5 2

end NUMINAMATH_CALUDE_ball_bounce_distance_l99_9978


namespace NUMINAMATH_CALUDE_cost_is_ten_l99_9957

/-- Represents the cost of piano lessons -/
structure LessonCost where
  lessons_per_week : ℕ
  lesson_duration_hours : ℕ
  weeks : ℕ
  total_earnings : ℕ

/-- Calculates the cost per half-hour of teaching -/
def cost_per_half_hour (lc : LessonCost) : ℚ :=
  lc.total_earnings / (2 * lc.lessons_per_week * lc.lesson_duration_hours * lc.weeks)

/-- Theorem: The cost per half-hour of teaching is $10 -/
theorem cost_is_ten (lc : LessonCost) 
  (h1 : lc.lessons_per_week = 1)
  (h2 : lc.lesson_duration_hours = 1)
  (h3 : lc.weeks = 5)
  (h4 : lc.total_earnings = 100) : 
  cost_per_half_hour lc = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_is_ten_l99_9957


namespace NUMINAMATH_CALUDE_f_monotonic_k_range_l99_9914

def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

theorem f_monotonic_k_range :
  ∀ k : ℝ, (monotonic_on (f k) 1 2) → k ≤ 8 ∨ k ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_f_monotonic_k_range_l99_9914


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l99_9937

theorem ratio_a_to_c (a b c d : ℝ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 8) :
  a / c = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l99_9937


namespace NUMINAMATH_CALUDE_product_evaluation_l99_9907

theorem product_evaluation : 
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * 
  (2^32 + 3^32) * (2^64 + 3^64) * (2 + 1) = 3^129 - 3 * 2^128 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l99_9907


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l99_9912

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => (x^3 - 4*x^2*(Real.sqrt 3) + 12*x - 8*(Real.sqrt 3)) + (2*x - 2*(Real.sqrt 3))
  ∃ (z₁ z₂ z₃ : ℂ),
    f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0 ∧
    z₁ = 2 * Real.sqrt 3 ∧
    z₂ = 2 * Real.sqrt 3 + Complex.I * Real.sqrt 2 ∧
    z₃ = 2 * Real.sqrt 3 - Complex.I * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l99_9912


namespace NUMINAMATH_CALUDE_solve_for_y_l99_9933

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = 5) : y = 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l99_9933


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l99_9918

/-- A fraction a/b is reducible if gcd(a,b) > 1 -/
def IsReducible (a b : ℤ) : Prop := Int.gcd a b > 1

/-- The numerator of our fraction -/
def Numerator (m : ℕ) : ℤ := m - 17

/-- The denominator of our fraction -/
def Denominator (m : ℕ) : ℤ := 6 * m + 7

theorem least_reducible_fraction :
  (∀ m : ℕ, m < 126 → ¬(IsReducible (Numerator m) (Denominator m))) ∧
  IsReducible (Numerator 126) (Denominator 126) := by
  sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l99_9918


namespace NUMINAMATH_CALUDE_trigonometric_equation_has_solution_l99_9913

theorem trigonometric_equation_has_solution :
  ∃ x : ℝ, 2 * Real.sin x * Real.cos (3 * Real.pi / 2 + x) -
           3 * Real.sin (Real.pi - x) * Real.cos x +
           Real.sin (Real.pi / 2 + x) * Real.cos x = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_has_solution_l99_9913


namespace NUMINAMATH_CALUDE_max_value_polynomial_l99_9974

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (z w : ℝ), z + w = 5 ∧ 
    z^4*w + z^3*w + z^2*w + z*w + z*w^2 + z*w^3 + z*w^4 ≥ 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4) ∧
  (∀ (z w : ℝ), z + w = 5 → 
    z^4*w + z^3*w + z^2*w + z*w + z*w^2 + z*w^3 + z*w^4 ≤ 6084/17) :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l99_9974


namespace NUMINAMATH_CALUDE_four_must_be_in_A_l99_9968

/-- A type representing the circles in the diagram -/
inductive Circle : Type
  | A | B | C | D | E | F | G

/-- The set of numbers to be placed in the circles -/
def NumberSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- A function that assigns a number to each circle -/
def Assignment := Circle → ℕ

/-- Predicate to check if an assignment is valid -/
def IsValidAssignment (f : Assignment) : Prop :=
  (∀ n ∈ NumberSet, ∃ c : Circle, f c = n) ∧
  (∀ c : Circle, f c ∈ NumberSet) ∧
  (f Circle.A + f Circle.D + f Circle.E = 
   f Circle.A + f Circle.C + f Circle.F) ∧
  (f Circle.A + f Circle.D + f Circle.E = 
   f Circle.A + f Circle.B + f Circle.G) ∧
  (f Circle.D + f Circle.C + f Circle.B = 
   f Circle.E + f Circle.F + f Circle.G)

theorem four_must_be_in_A (f : Assignment) 
  (h : IsValidAssignment f) : 
  f Circle.A = 4 ∧ f Circle.E ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_four_must_be_in_A_l99_9968


namespace NUMINAMATH_CALUDE_mississippi_arrangements_l99_9987

/-- The number of unique arrangements of the letters in MISSISSIPPI -/
def mississippiArrangements : ℕ := 34650

/-- The total number of letters in MISSISSIPPI -/
def totalLetters : ℕ := 11

/-- The number of occurrences of 'I' in MISSISSIPPI -/
def countI : ℕ := 4

/-- The number of occurrences of 'S' in MISSISSIPPI -/
def countS : ℕ := 4

/-- The number of occurrences of 'P' in MISSISSIPPI -/
def countP : ℕ := 2

/-- The number of occurrences of 'M' in MISSISSIPPI -/
def countM : ℕ := 1

theorem mississippi_arrangements :
  mississippiArrangements = Nat.factorial totalLetters / (Nat.factorial countI * Nat.factorial countS * Nat.factorial countP) :=
by sorry

end NUMINAMATH_CALUDE_mississippi_arrangements_l99_9987


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l99_9943

theorem bernoulli_inequality (n : ℕ+) (x : ℝ) (h : x > -1) :
  (1 + x)^(n : ℝ) ≥ 1 + n * x :=
by sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l99_9943


namespace NUMINAMATH_CALUDE_product_uvw_l99_9992

theorem product_uvw (a c x y : ℝ) (u v w : ℤ) : 
  (a^8*x*y - a^7*y - a^6*x = a^5*(c^5 - 1)) ∧ 
  ((a^u*x - a^v)*(a^w*y - a^3) = a^5*c^5) →
  u*v*w = 6 :=
by sorry

end NUMINAMATH_CALUDE_product_uvw_l99_9992


namespace NUMINAMATH_CALUDE_no_integer_solution_l99_9983

theorem no_integer_solution : ∀ x y : ℤ, x^2 + 3*x*y - 2*y^2 ≠ 122 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l99_9983


namespace NUMINAMATH_CALUDE_x_squared_mod_26_l99_9982

theorem x_squared_mod_26 (x : ℤ) (h1 : 5 * x ≡ 9 [ZMOD 26]) (h2 : 4 * x ≡ 15 [ZMOD 26]) :
  x^2 ≡ 10 [ZMOD 26] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_26_l99_9982


namespace NUMINAMATH_CALUDE_orchestra_members_count_l99_9959

theorem orchestra_members_count : ∃! x : ℕ, 
  150 < x ∧ x < 250 ∧ 
  x % 4 = 2 ∧ 
  x % 5 = 3 ∧ 
  x % 8 = 4 ∧ 
  x % 9 = 5 ∧ 
  x = 58 :=
by sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l99_9959


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l99_9904

theorem rectangular_plot_breadth :
  ∀ (length breadth area : ℝ),
  length = 3 * breadth →
  area = length * breadth →
  area = 2028 →
  breadth = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l99_9904


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_B_l99_9938

def B : Set ℕ := {n : ℕ | ∀ p : ℕ, Nat.Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 7}

theorem sum_of_reciprocals_B : ∑' (n : B), (1 : ℚ) / n = 7 / 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_B_l99_9938


namespace NUMINAMATH_CALUDE_linear_function_inequality_l99_9977

/-- A linear function passing through first, second, and fourth quadrants -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  first_quadrant : ∃ x y, x > 0 ∧ y > 0 ∧ y = a * x + b
  second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = a * x + b
  fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = a * x + b
  x_intercept : a * 2 + b = 0

/-- The solution set of a(x-1)-b > 0 for a LinearFunction is x < -1 -/
theorem linear_function_inequality (f : LinearFunction) :
  {x : ℝ | f.a * (x - 1) - f.b > 0} = {x : ℝ | x < -1} := by
  sorry

end NUMINAMATH_CALUDE_linear_function_inequality_l99_9977
