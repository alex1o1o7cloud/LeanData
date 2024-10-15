import Mathlib

namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1847_184789

/-- Given that a² and √b vary inversely, if a = 3 when b = 64, then b = 18 when ab = 72 -/
theorem inverse_variation_problem (a b : ℝ) : 
  (∃ k : ℝ, ∀ a b : ℝ, a^2 * Real.sqrt b = k) →  -- a² and √b vary inversely
  (3^2 * Real.sqrt 64 = 3 * 64) →                -- a = 3 when b = 64
  (a * b = 72) →                                 -- ab = 72
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1847_184789


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l1847_184755

theorem smaller_solution_of_quadratic (x : ℝ) : 
  x^2 + 20*x - 72 = 0 → (∃ y : ℝ, y^2 + 20*y - 72 = 0 ∧ y ≤ x) → x = -24 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l1847_184755


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1847_184775

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  8 * x^4 + 12 * y^4 + 18 * z^4 + 25 / (x * y * z) ≥ 30 :=
by sorry

theorem min_value_attained :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  8 * x^4 + 12 * y^4 + 18 * z^4 + 25 / (x * y * z) = 30 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1847_184775


namespace NUMINAMATH_CALUDE_cubic_sum_implies_linear_sum_l1847_184722

theorem cubic_sum_implies_linear_sum (x : ℝ) (h : x^3 + 1/x^3 = 52) : x + 1/x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_implies_linear_sum_l1847_184722


namespace NUMINAMATH_CALUDE_smallest_invertible_domain_l1847_184760

def g (x : ℝ) : ℝ := (2*x - 3)^2 - 4

theorem smallest_invertible_domain (c : ℝ) : 
  (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ∧ 
  (∀ c' : ℝ, c' < c → ∃ x y, x ≥ c' ∧ y ≥ c' ∧ x ≠ y ∧ g x = g y) → 
  c = 3/2 :=
sorry

end NUMINAMATH_CALUDE_smallest_invertible_domain_l1847_184760


namespace NUMINAMATH_CALUDE_new_quadratic_equation_l1847_184701

theorem new_quadratic_equation (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let new_eq := fun y => a * y^2 + b * y + (c - a + Real.sqrt (b^2 - 4*a*c))
  (new_eq (x₁ - 1) = 0) ∧ (new_eq (x₂ + 1) = 0) := by
sorry

end NUMINAMATH_CALUDE_new_quadratic_equation_l1847_184701


namespace NUMINAMATH_CALUDE_largest_area_chord_construction_l1847_184743

/-- Represents a direction in 2D space -/
structure Direction where
  angle : Real

/-- Represents an ellipse -/
structure Ellipse where
  semi_major_axis : Real
  semi_minor_axis : Real

/-- Represents a chord of an ellipse -/
structure Chord where
  direction : Direction
  length : Real

/-- Represents a triangle -/
structure Triangle where
  base : Real
  height : Real

/-- Calculates the area of a triangle -/
def triangle_area (t : Triangle) : Real :=
  0.5 * t.base * t.height

/-- Finds the chord that forms the triangle with the largest area -/
def largest_area_chord (e : Ellipse) (i : Direction) : Chord :=
  sorry

/-- Theorem: The chord that forms the largest area triangle is constructed by 
    finding an intersection point and creating two mirrored right triangles -/
theorem largest_area_chord_construction (e : Ellipse) (i : Direction) :
  ∃ (c : Chord), c = largest_area_chord e i ∧
  ∃ (t1 t2 : Triangle), 
    triangle_area t1 = triangle_area t2 ∧
    t1.base = t2.base ∧
    t1.height = t2.height ∧
    t1.base * t1.base + t1.height * t1.height = c.length * c.length :=
  sorry

end NUMINAMATH_CALUDE_largest_area_chord_construction_l1847_184743


namespace NUMINAMATH_CALUDE_selection_theorem_l1847_184762

/-- The number of ways to select k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of male students. -/
def num_males : ℕ := 4

/-- The number of female students. -/
def num_females : ℕ := 3

/-- The total number of students to be selected. -/
def num_selected : ℕ := 3

/-- The number of ways to select students with both genders represented. -/
def num_ways : ℕ := 
  choose num_males 2 * choose num_females 1 + 
  choose num_males 1 * choose num_females 2

theorem selection_theorem : num_ways = 30 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l1847_184762


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1847_184767

theorem unique_solution_quadratic (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1847_184767


namespace NUMINAMATH_CALUDE_complex_fraction_power_l1847_184715

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_power (h : i * i = -1) : ((1 + i) / (1 - i)) ^ 2013 = i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l1847_184715


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1847_184756

theorem simplify_complex_fraction : 
  1 / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 7 - 2))) = 3 / (9 * Real.sqrt 5 + 4 * Real.sqrt 7 - 10) :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1847_184756


namespace NUMINAMATH_CALUDE_event_probability_estimate_l1847_184721

-- Define the frequency function
def frequency : ℕ → ℝ
| 20 => 0.300
| 50 => 0.360
| 100 => 0.350
| 300 => 0.350
| 500 => 0.352
| 1000 => 0.351
| 5000 => 0.351
| _ => 0  -- For other values, we set the frequency to 0

-- Define the set of trial numbers
def trialNumbers : Set ℕ := {20, 50, 100, 300, 500, 1000, 5000}

-- Theorem statement
theorem event_probability_estimate :
  ∀ ε > 0, ∃ N ∈ trialNumbers, ∀ n ∈ trialNumbers, n ≥ N → |frequency n - 0.35| < ε :=
sorry

end NUMINAMATH_CALUDE_event_probability_estimate_l1847_184721


namespace NUMINAMATH_CALUDE_quarters_count_l1847_184746

/-- Given a sum of $3.35 consisting of quarters and dimes, with a total of 23 coins, 
    prove that the number of quarters is 7. -/
theorem quarters_count (total_value : ℚ) (total_coins : ℕ) 
  (h1 : total_value = 335/100) 
  (h2 : total_coins = 23) : ∃ (quarters dimes : ℕ),
  quarters + dimes = total_coins ∧ 
  (25 * quarters + 10 * dimes : ℚ) / 100 = total_value ∧
  quarters = 7 := by
  sorry

end NUMINAMATH_CALUDE_quarters_count_l1847_184746


namespace NUMINAMATH_CALUDE_co_molecular_weight_l1847_184737

-- Define the atomic weights
def atomic_weight_carbon : ℝ := 12.01
def atomic_weight_oxygen : ℝ := 16.00

-- Define the molecular weight calculation function
def molecular_weight (carbon_atoms : ℕ) (oxygen_atoms : ℕ) : ℝ :=
  carbon_atoms * atomic_weight_carbon + oxygen_atoms * atomic_weight_oxygen

-- Theorem statement
theorem co_molecular_weight :
  molecular_weight 1 1 = 28.01 := by sorry

end NUMINAMATH_CALUDE_co_molecular_weight_l1847_184737


namespace NUMINAMATH_CALUDE_non_swimmers_playing_soccer_l1847_184703

/-- Represents the percentage of children who play soccer at Lakeview Summer Camp -/
def soccer_players : ℝ := 0.7

/-- Represents the percentage of children who swim at Lakeview Summer Camp -/
def swimmers : ℝ := 0.5

/-- Represents the percentage of soccer players who also swim -/
def soccer_swimmers : ℝ := 0.3

/-- Theorem stating that the percentage of non-swimmers who play soccer is 98% -/
theorem non_swimmers_playing_soccer :
  (soccer_players - soccer_players * soccer_swimmers) / (1 - swimmers) = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_non_swimmers_playing_soccer_l1847_184703


namespace NUMINAMATH_CALUDE_equation_solution_l1847_184754

theorem equation_solution : ∃ x : ℝ, (1 / 7 + 4 / x = 12 / x + 1 / 14) ∧ x = 112 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1847_184754


namespace NUMINAMATH_CALUDE_fraction_value_at_sqrt_two_l1847_184736

theorem fraction_value_at_sqrt_two :
  let x := Real.sqrt 2
  (x^2 - 1) / (x^2 - x) - 1 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_sqrt_two_l1847_184736


namespace NUMINAMATH_CALUDE_definitely_rain_next_tuesday_is_false_l1847_184724

-- Define a proposition representing the statement "It will definitely rain next Tuesday"
def definitely_rain_next_tuesday : Prop := True

-- Define a proposition representing the uncertainty of future events
def future_events_are_uncertain : Prop := True

-- Theorem stating that the original statement is false
theorem definitely_rain_next_tuesday_is_false : 
  future_events_are_uncertain → ¬definitely_rain_next_tuesday := by
  sorry

end NUMINAMATH_CALUDE_definitely_rain_next_tuesday_is_false_l1847_184724


namespace NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l1847_184749

/-- Given a bucket with known weights at different fill levels, 
    calculate the weight when it's full. -/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let three_fourths_weight := p
  let one_third_weight := q
  let full_weight := (8 * p - 11 * q) / 5
  full_weight

/-- Prove that the calculated full weight is correct -/
theorem bucket_weight_proof (p q : ℝ) : 
  bucket_weight p q = (8 * p - 11 * q) / 5 := by
  sorry

end NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l1847_184749


namespace NUMINAMATH_CALUDE_pie_chart_best_for_part_whole_l1847_184783

-- Define the types of statistical graphs
inductive StatisticalGraph
  | BarGraph
  | PieChart
  | LineGraph
  | FrequencyDistributionHistogram

-- Define a property for highlighting part-whole relationships
def highlightsPartWholeRelationship (graph : StatisticalGraph) : Prop :=
  match graph with
  | StatisticalGraph.PieChart => true
  | _ => false

-- Theorem statement
theorem pie_chart_best_for_part_whole : 
  ∀ (graph : StatisticalGraph), 
    highlightsPartWholeRelationship graph → graph = StatisticalGraph.PieChart :=
by
  sorry


end NUMINAMATH_CALUDE_pie_chart_best_for_part_whole_l1847_184783


namespace NUMINAMATH_CALUDE_f_2_equals_neg_26_l1847_184799

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_2_equals_neg_26 (a b : ℝ) :
  f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_2_equals_neg_26_l1847_184799


namespace NUMINAMATH_CALUDE_abscissa_of_point_M_l1847_184786

/-- Given a point M with coordinates (1,1), prove that its abscissa is 1 -/
theorem abscissa_of_point_M (M : ℝ × ℝ) (h : M = (1, 1)) : M.1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_abscissa_of_point_M_l1847_184786


namespace NUMINAMATH_CALUDE_min_PQ_distance_l1847_184761

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := x + Real.log x

-- Define the distance between P and Q
def PQ_distance (a x₁ x₂ : ℝ) : ℝ :=
  |x₂ - x₁|

-- State the theorem
theorem min_PQ_distance :
  ∃ (a : ℝ), ∀ (x₁ x₂ : ℝ),
    f x₁ = a → g x₂ = a →
    (∀ (y₁ y₂ : ℝ), f y₁ = a → g y₂ = a → PQ_distance a x₁ x₂ ≤ PQ_distance a y₁ y₂) →
    PQ_distance a x₁ x₂ = 2 :=
sorry

end

end NUMINAMATH_CALUDE_min_PQ_distance_l1847_184761


namespace NUMINAMATH_CALUDE_square_side_increase_l1847_184700

theorem square_side_increase (p : ℝ) : 
  ((1 + p / 100) ^ 2 = 1.44) → p = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_side_increase_l1847_184700


namespace NUMINAMATH_CALUDE_equation_solution_l1847_184794

theorem equation_solution (x : ℝ) : 
  (8 / (Real.sqrt (x - 8) - 10) + 3 / (Real.sqrt (x - 8) - 5) + 
   4 / (Real.sqrt (x - 8) + 5) + 15 / (Real.sqrt (x - 8) + 10) = 0) ↔ 
  (x = 33 ∨ x = 108) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1847_184794


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l1847_184784

theorem cubic_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4*a*b := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l1847_184784


namespace NUMINAMATH_CALUDE_line_through_circle_center_l1847_184776

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 1 = 0

-- Define the line l
def line_l (x y m : ℝ) : Prop := x + m*y + 1 = 0

-- Define the center of the circle
def center_C : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem line_through_circle_center (m : ℝ) :
  line_l (center_C.1) (center_C.2) m → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l1847_184776


namespace NUMINAMATH_CALUDE_exactly_one_subset_exactly_one_element_l1847_184770

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

-- Theorem for part 1
theorem exactly_one_subset (a : ℝ) : (∃! (S : Set ℝ), S ⊆ A a) ↔ a > 1 := by sorry

-- Theorem for part 2
theorem exactly_one_element (a : ℝ) : (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_exactly_one_subset_exactly_one_element_l1847_184770


namespace NUMINAMATH_CALUDE_parabola_vertex_l1847_184730

/-- The equation of a parabola in the form y^2 - 4y + 2x + 9 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 4*y + 2*x + 9 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  eq x y ∧ ∀ x' y', eq x' y' → y ≤ y'

theorem parabola_vertex :
  is_vertex (-5/2) 2 parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1847_184730


namespace NUMINAMATH_CALUDE_select_four_with_one_girl_l1847_184734

/-- The number of ways to select 4 people from two groups with exactly 1 girl -/
def select_with_one_girl (boys_a boys_b girls_a girls_b : ℕ) : ℕ :=
  (girls_a.choose 1 * boys_a.choose 1 * boys_b.choose 2) +
  (boys_a.choose 2 * girls_b.choose 1 * boys_b.choose 1)

/-- Theorem stating the correct number of selections for the given group compositions -/
theorem select_four_with_one_girl :
  select_with_one_girl 5 6 3 2 = 345 := by
  sorry

end NUMINAMATH_CALUDE_select_four_with_one_girl_l1847_184734


namespace NUMINAMATH_CALUDE_car_sales_second_day_l1847_184796

theorem car_sales_second_day 
  (total_sales : ℕ) 
  (first_day_sales : ℕ) 
  (third_day_sales : ℕ) 
  (h1 : total_sales = 57)
  (h2 : first_day_sales = 14)
  (h3 : third_day_sales = 27) :
  total_sales - first_day_sales - third_day_sales = 16 := by
  sorry

end NUMINAMATH_CALUDE_car_sales_second_day_l1847_184796


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1847_184788

theorem sum_of_three_numbers (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b = 24) (h2 : a * c = 36) (h3 : b * c = 54) : a + b + c = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1847_184788


namespace NUMINAMATH_CALUDE_star_equation_solution_l1847_184725

/-- The star operation defined on real numbers -/
def star (x y : ℝ) : ℝ := 5*x - 2*y + 2*x*y

/-- Theorem stating that 4 star y = 22 if and only if y = 1/3 -/
theorem star_equation_solution :
  ∀ y : ℝ, star 4 y = 22 ↔ y = 1/3 := by sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1847_184725


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l1847_184777

/-- Given an arithmetic sequence of 25 terms with first term 5 and last term 77,
    prove that the 10th term is 32. -/
theorem arithmetic_sequence_10th_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
    (a 1 = 5) →                          -- first term is 5
    (a 25 = 77) →                        -- last term is 77
    (a 10 = 32) :=                       -- 10th term is 32
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l1847_184777


namespace NUMINAMATH_CALUDE_r_amount_l1847_184720

def total_amount : ℝ := 9000

theorem r_amount (p q r : ℝ) 
  (h1 : p + q + r = total_amount)
  (h2 : r = (2/3) * (p + q)) :
  r = 3600 := by
  sorry

end NUMINAMATH_CALUDE_r_amount_l1847_184720


namespace NUMINAMATH_CALUDE_banana_popsicles_count_l1847_184774

theorem banana_popsicles_count (grape_count cherry_count total_count : ℕ) 
  (h1 : grape_count = 2)
  (h2 : cherry_count = 13)
  (h3 : total_count = 17) :
  total_count - (grape_count + cherry_count) = 2 := by
  sorry

end NUMINAMATH_CALUDE_banana_popsicles_count_l1847_184774


namespace NUMINAMATH_CALUDE_field_length_correct_l1847_184759

/-- The length of a rectangular field -/
def field_length : ℝ := 75

/-- The width of the rectangular field -/
def field_width : ℝ := 15

/-- The number of times the field is circled -/
def laps : ℕ := 3

/-- The total distance covered -/
def total_distance : ℝ := 540

/-- Theorem stating that the field length is correct given the conditions -/
theorem field_length_correct : 
  2 * (field_length + field_width) * laps = total_distance :=
by sorry

end NUMINAMATH_CALUDE_field_length_correct_l1847_184759


namespace NUMINAMATH_CALUDE_monet_paintings_consecutive_probability_l1847_184702

/-- The probability of consecutive Monet paintings in a random arrangement -/
theorem monet_paintings_consecutive_probability 
  (total_pieces : ℕ) 
  (monet_paintings : ℕ) 
  (h1 : total_pieces = 12) 
  (h2 : monet_paintings = 4) :
  (monet_paintings.factorial * (total_pieces - monet_paintings + 1)) / total_pieces.factorial = 18 / 95 := by
  sorry

end NUMINAMATH_CALUDE_monet_paintings_consecutive_probability_l1847_184702


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_l1847_184793

theorem negation_of_universal_quantifier :
  (¬ ∀ x : ℝ, x^2 - 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_l1847_184793


namespace NUMINAMATH_CALUDE_marbles_given_proof_l1847_184717

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 776 - 593

/-- Connie's initial number of marbles -/
def initial_marbles : ℕ := 776

/-- The number of marbles Connie has left -/
def remaining_marbles : ℕ := 593

theorem marbles_given_proof : 
  marbles_given = initial_marbles - remaining_marbles :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_proof_l1847_184717


namespace NUMINAMATH_CALUDE_solution_set_and_range_l1847_184709

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

def g (x : ℝ) : ℝ := |2*x - 3|

theorem solution_set_and_range :
  (∀ x, x ∈ {y : ℝ | 0 ≤ y ∧ y ≤ 3} ↔ f 3 x ≤ 6) ∧
  (∀ a, (∀ x, f a x + g x ≥ 5) ↔ a ≥ 11/3) := by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l1847_184709


namespace NUMINAMATH_CALUDE_least_number_with_remainder_four_l1847_184711

theorem least_number_with_remainder_four (n : ℕ) : 
  (∀ m : ℕ, m > 0 → n % m = 4) → 
  (n % 12 = 4) → 
  n ≥ 40 :=
by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_four_l1847_184711


namespace NUMINAMATH_CALUDE_museum_visitors_survey_l1847_184766

theorem museum_visitors_survey (V : ℕ) : 
  (∃ E : ℕ, 
    V = E + 140 ∧ 
    3 * V = 4 * E) →
  V = 560 :=
by
  sorry

end NUMINAMATH_CALUDE_museum_visitors_survey_l1847_184766


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1847_184787

theorem opposite_of_negative_two : 
  ∃ x : ℤ, -x = -2 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1847_184787


namespace NUMINAMATH_CALUDE_magnitude_BC_l1847_184706

/-- Given two vectors BA and AC in R², prove that the magnitude of BC is 5. -/
theorem magnitude_BC (BA AC : ℝ × ℝ) (h1 : BA = (3, -2)) (h2 : AC = (0, 6)) : 
  ‖BA + AC‖ = 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_BC_l1847_184706


namespace NUMINAMATH_CALUDE_remaining_tanning_time_l1847_184713

/-- Calculates the remaining tanning time for the last two weeks of a month. -/
theorem remaining_tanning_time 
  (monthly_limit : ℕ) 
  (daily_time : ℕ) 
  (days_per_week : ℕ) 
  (weeks : ℕ) 
  (h1 : monthly_limit = 200)
  (h2 : daily_time = 30)
  (h3 : days_per_week = 2)
  (h4 : weeks = 2) :
  monthly_limit - (daily_time * days_per_week * weeks) = 80 := by
  sorry

#check remaining_tanning_time

end NUMINAMATH_CALUDE_remaining_tanning_time_l1847_184713


namespace NUMINAMATH_CALUDE_walnut_trees_planted_park_walnut_trees_l1847_184753

/-- The number of walnut trees planted in a park -/
def trees_planted (initial_trees final_trees : ℕ) : ℕ :=
  final_trees - initial_trees

/-- Theorem: The number of trees planted is the difference between the final and initial number of trees -/
theorem walnut_trees_planted (initial_trees final_trees : ℕ) 
  (h : initial_trees ≤ final_trees) :
  trees_planted initial_trees final_trees = final_trees - initial_trees :=
by sorry

/-- The specific case for the park problem -/
theorem park_walnut_trees : trees_planted 22 77 = 55 :=
by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_park_walnut_trees_l1847_184753


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l1847_184732

/-- Represents a participant's score in a math competition --/
structure Score where
  points : ℕ
  total : ℕ
  h_positive : points > 0
  h_valid : points ≤ total

/-- Represents a participant's scores over three days --/
structure ThreeDayScore where
  day1 : Score
  day2 : Score
  day3 : Score
  h_total : day1.total + day2.total + day3.total = 600

def success_ratio (s : Score) : ℚ :=
  s.points / s.total

def three_day_success_ratio (s : ThreeDayScore) : ℚ :=
  (s.day1.points + s.day2.points + s.day3.points) / 600

theorem delta_max_success_ratio 
  (charlie : ThreeDayScore)
  (delta : ThreeDayScore)
  (h_charlie_day1 : charlie.day1 = ⟨210, 350, by sorry, by sorry⟩)
  (h_charlie_day2 : charlie.day2 = ⟨170, 250, by sorry, by sorry⟩)
  (h_charlie_day3 : charlie.day3 = ⟨0, 0, by sorry, by sorry⟩)
  (h_delta_day1 : success_ratio delta.day1 < success_ratio charlie.day1)
  (h_delta_day2 : success_ratio delta.day2 < success_ratio charlie.day2)
  (h_delta_day3_positive : delta.day3.points > 0) :
  three_day_success_ratio delta ≤ 378 / 600 :=
sorry

end NUMINAMATH_CALUDE_delta_max_success_ratio_l1847_184732


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1847_184712

/-- Given a line L1 defined by 3x - 2y = 6, and a line L2 perpendicular to L1 with y-intercept 4,
    the x-intercept of L2 is 6. -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 2 * y = 6
  let m1 : ℝ := 3 / 2  -- slope of L1
  let m2 : ℝ := -2 / 3  -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = m2 * x + 4  -- equation of L2
  let x_intercept : ℝ := 6
  (∀ x y, L2 x y ↔ y = m2 * x + 4) →  -- L2 is defined correctly
  (m1 * m2 = -1) →  -- L1 and L2 are perpendicular
  L2 x_intercept 0  -- (6, 0) satisfies the equation of L2
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1847_184712


namespace NUMINAMATH_CALUDE_cube_roots_unity_sum_l1847_184780

theorem cube_roots_unity_sum (x y : ℂ) : 
  x = (-1 + Complex.I * Real.sqrt 3) / 2 →
  y = (-1 - Complex.I * Real.sqrt 3) / 2 →
  x^9 + y^9 ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_cube_roots_unity_sum_l1847_184780


namespace NUMINAMATH_CALUDE_sport_drink_water_amount_l1847_184728

/-- Represents the composition of a sport drink -/
structure SportDrink where
  flavoringRatio : ℚ
  cornSyrupRatio : ℚ
  waterRatio : ℚ
  cornSyrupOunces : ℚ

/-- Calculates the amount of water in a sport drink -/
def waterAmount (drink : SportDrink) : ℚ :=
  (drink.waterRatio / drink.cornSyrupRatio) * drink.cornSyrupOunces

/-- Theorem stating the amount of water in the sport drink -/
theorem sport_drink_water_amount 
  (drink : SportDrink)
  (h1 : drink.flavoringRatio = 1)
  (h2 : drink.cornSyrupRatio = 4)
  (h3 : drink.waterRatio = 60)
  (h4 : drink.cornSyrupOunces = 7) :
  waterAmount drink = 105 := by
  sorry

#check sport_drink_water_amount

end NUMINAMATH_CALUDE_sport_drink_water_amount_l1847_184728


namespace NUMINAMATH_CALUDE_units_digits_divisible_by_eight_l1847_184735

theorem units_digits_divisible_by_eight :
  ∃ (S : Finset Nat), (∀ n : Nat, n % 10 ∈ S ↔ ∃ m : Nat, m % 8 = 0 ∧ m % 10 = n % 10) ∧ Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digits_divisible_by_eight_l1847_184735


namespace NUMINAMATH_CALUDE_examination_attendance_l1847_184705

theorem examination_attendance :
  ∀ (total_students : ℕ) (passed_percentage : ℚ) (failed_count : ℕ),
    passed_percentage = 35 / 100 →
    failed_count = 520 →
    (1 - passed_percentage) * total_students = failed_count →
    total_students = 800 := by
  sorry

end NUMINAMATH_CALUDE_examination_attendance_l1847_184705


namespace NUMINAMATH_CALUDE_root_equation_k_value_l1847_184779

theorem root_equation_k_value :
  ∀ k : ℝ, ((-2)^2 - k*(-2) + 2 = 0) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_k_value_l1847_184779


namespace NUMINAMATH_CALUDE_percent_not_working_projects_l1847_184748

/-- Represents the survey results of employees working on projects -/
structure ProjectSurvey where
  total : ℕ
  projectA : ℕ
  projectB : ℕ
  bothProjects : ℕ

/-- Calculates the percentage of employees not working on either project -/
def percentNotWorking (survey : ProjectSurvey) : ℚ :=
  let workingOnEither := survey.projectA + survey.projectB - survey.bothProjects
  let notWorking := survey.total - workingOnEither
  (notWorking : ℚ) / survey.total * 100

/-- The theorem stating the percentage of employees not working on either project -/
theorem percent_not_working_projects (survey : ProjectSurvey) 
    (h1 : survey.total = 150)
    (h2 : survey.projectA = 90)
    (h3 : survey.projectB = 50)
    (h4 : survey.bothProjects = 30) :
    percentNotWorking survey = 26.67 := by
  sorry


end NUMINAMATH_CALUDE_percent_not_working_projects_l1847_184748


namespace NUMINAMATH_CALUDE_base_sum_theorem_l1847_184773

/-- Given two integer bases R₁ and R₂, if certain fractions have specific representations
    in these bases, then the sum of R₁ and R₂ is 21. -/
theorem base_sum_theorem (R₁ R₂ : ℕ) : 
  (R₁ > 1) → 
  (R₂ > 1) →
  ((4 * R₁ + 8) / (R₁^2 - 1) = (3 * R₂ + 6) / (R₂^2 - 1)) →
  ((8 * R₁ + 4) / (R₁^2 - 1) = (6 * R₂ + 3) / (R₂^2 - 1)) →
  R₁ + R₂ = 21 :=
by sorry

#check base_sum_theorem

end NUMINAMATH_CALUDE_base_sum_theorem_l1847_184773


namespace NUMINAMATH_CALUDE_simplify_fraction_l1847_184727

theorem simplify_fraction : (125 : ℚ) / 10000 * 40 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1847_184727


namespace NUMINAMATH_CALUDE_function_range_l1847_184750

theorem function_range (a : ℝ) (f : ℝ → ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x, f x = a^(-x)) (h4 : f (-2) > f (-3)) : 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1847_184750


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_formula_implies_t_equals_5_l1847_184740

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), n ≥ 1 → a (n + 1) = r * a n

-- Define the sum formula for the first n terms
def sum_formula (S : ℕ → ℝ) (t : ℝ) : Prop :=
  ∀ (n : ℕ), S n = t * 5^n - 2

-- Theorem statement
theorem geometric_sequence_sum_formula_implies_t_equals_5 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : sum_formula S t) : 
  t = 5 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_formula_implies_t_equals_5_l1847_184740


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l1847_184745

theorem smallest_angle_in_triangle (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) 
  (h_ratio : Real.sin A / Real.sin B = 2 / Real.sqrt 6 ∧ 
             Real.sin B / Real.sin C = Real.sqrt 6 / (Real.sqrt 3 + 1)) : 
  min A (min B C) = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l1847_184745


namespace NUMINAMATH_CALUDE_restaurant_bill_l1847_184791

theorem restaurant_bill (food_cost : ℝ) (service_fee_percent : ℝ) (tip : ℝ) : 
  food_cost = 50 ∧ service_fee_percent = 12 ∧ tip = 5 →
  food_cost + (service_fee_percent / 100) * food_cost + tip = 61 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_l1847_184791


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1847_184790

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1847_184790


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l1847_184752

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) : 
  (2 * r) ^ 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l1847_184752


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_inequality_l1847_184797

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_cubic_inequality : 
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_inequality_l1847_184797


namespace NUMINAMATH_CALUDE_fare_660_equals_3_miles_unique_distance_for_660_l1847_184716

/-- Calculates the taxi fare for a given distance -/
def taxi_fare (distance : ℚ) : ℚ :=
  1 + 0.4 * (5 * distance - 1)

/-- Proves that a fare of $6.60 corresponds to a distance of 3 miles -/
theorem fare_660_equals_3_miles :
  taxi_fare 3 = 6.6 :=
sorry

/-- Proves that 3 miles is the unique distance that results in a fare of $6.60 -/
theorem unique_distance_for_660 :
  ∀ d : ℚ, taxi_fare d = 6.6 → d = 3 :=
sorry

end NUMINAMATH_CALUDE_fare_660_equals_3_miles_unique_distance_for_660_l1847_184716


namespace NUMINAMATH_CALUDE_sin_cos_pi_over_12_l1847_184781

theorem sin_cos_pi_over_12 : 
  Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sin_cos_pi_over_12_l1847_184781


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1847_184768

/-- The ratio of a man's age to his son's age in two years -/
def age_ratio (son_age : ℕ) (age_difference : ℕ) : ℚ :=
  let man_age := son_age + age_difference
  (man_age + 2) / (son_age + 2)

/-- Theorem: The ratio of the man's age to his son's age in two years is 2:1 -/
theorem age_ratio_is_two_to_one (son_age : ℕ) (age_difference : ℕ)
  (h1 : son_age = 23)
  (h2 : age_difference = 25) :
  age_ratio son_age age_difference = 2 := by
  sorry

#eval age_ratio 23 25

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1847_184768


namespace NUMINAMATH_CALUDE_frog_count_l1847_184729

theorem frog_count : ∀ (N : ℕ), 
  (∃ (T : ℝ), T > 0 ∧
    50 * (0.3 * T / 50) ≤ 0.43 * T / (N - 94) ∧
    0.43 * T / (N - 94) ≤ 44 * (0.27 * T / 44) ∧
    N > 94)
  → N = 165 := by
sorry

end NUMINAMATH_CALUDE_frog_count_l1847_184729


namespace NUMINAMATH_CALUDE_max_value_determines_parameter_l1847_184798

/-- Given a system of linear inequalities and an objective function,
    prove that the maximum value of the objective function
    determines the value of a parameter. -/
theorem max_value_determines_parameter
  (x y z a : ℝ)
  (h1 : x - 3 ≤ 0)
  (h2 : y - a ≤ 0)
  (h3 : x + y ≥ 0)
  (h4 : z = 2*x + y)
  (h5 : ∀ x' y' z', x' - 3 ≤ 0 → y' - a ≤ 0 → x' + y' ≥ 0 → z' = 2*x' + y' → z' ≤ 10)
  (h6 : ∃ x' y' z', x' - 3 ≤ 0 ∧ y' - a ≤ 0 ∧ x' + y' ≥ 0 ∧ z' = 2*x' + y' ∧ z' = 10) :
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_determines_parameter_l1847_184798


namespace NUMINAMATH_CALUDE_sailboat_sails_height_l1847_184742

theorem sailboat_sails_height (rectangular_length rectangular_width first_triangular_base second_triangular_base second_triangular_height total_canvas : ℝ) 
  (h1 : rectangular_length = 8)
  (h2 : rectangular_width = 5)
  (h3 : first_triangular_base = 3)
  (h4 : second_triangular_base = 4)
  (h5 : second_triangular_height = 6)
  (h6 : total_canvas = 58) :
  let rectangular_area := rectangular_length * rectangular_width
  let second_triangular_area := (second_triangular_base * second_triangular_height) / 2
  let first_triangular_area := total_canvas - rectangular_area - second_triangular_area
  first_triangular_area = (first_triangular_base * 4) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sailboat_sails_height_l1847_184742


namespace NUMINAMATH_CALUDE_juanita_dessert_cost_l1847_184795

/-- Represents the menu prices and discounts for the brownie dessert --/
structure BrownieMenu where
  brownie_base : ℝ := 2.50
  regular_scoop : ℝ := 1.00
  premium_scoop : ℝ := 1.25
  deluxe_scoop : ℝ := 1.50
  syrup : ℝ := 0.50
  nuts : ℝ := 1.50
  whipped_cream : ℝ := 0.75
  cherry : ℝ := 0.25
  tuesday_discount : ℝ := 0.10
  wednesday_discount : ℝ := 0.50
  sunday_discount : ℝ := 0.25

/-- Represents Juanita's order --/
structure JuanitaOrder where
  regular_scoops : ℕ := 2
  premium_scoops : ℕ := 1
  deluxe_scoops : ℕ := 0
  syrups : ℕ := 2
  has_nuts : Bool := true
  has_whipped_cream : Bool := true
  has_cherry : Bool := true

/-- Calculates the total cost of Juanita's dessert --/
def calculate_total_cost (menu : BrownieMenu) (order : JuanitaOrder) : ℝ :=
  let discounted_brownie := menu.brownie_base * (1 - menu.tuesday_discount)
  let ice_cream_cost := order.regular_scoops * menu.regular_scoop + 
                        order.premium_scoops * menu.premium_scoop + 
                        order.deluxe_scoops * menu.deluxe_scoop
  let syrup_cost := order.syrups * menu.syrup
  let topping_cost := (if order.has_nuts then menu.nuts else 0) +
                      (if order.has_whipped_cream then menu.whipped_cream else 0) +
                      (if order.has_cherry then menu.cherry else 0)
  discounted_brownie + ice_cream_cost + syrup_cost + topping_cost

/-- Theorem stating that Juanita's dessert costs $9.00 --/
theorem juanita_dessert_cost (menu : BrownieMenu) (order : JuanitaOrder) :
  calculate_total_cost menu order = 9.00 := by
  sorry


end NUMINAMATH_CALUDE_juanita_dessert_cost_l1847_184795


namespace NUMINAMATH_CALUDE_gcd_840_1764_l1847_184731

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l1847_184731


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1847_184763

theorem complex_equation_solution (a : ℝ) : 
  (Complex.mk 2 a) * (Complex.mk a (-2)) = Complex.I * (-4) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1847_184763


namespace NUMINAMATH_CALUDE_vikas_questions_l1847_184723

/-- Prove that given a total of 24 questions submitted in the ratio 7 : 3 : 2,
    the number of questions submitted by the person corresponding to the second part of the ratio is 6. -/
theorem vikas_questions (total : ℕ) (r v a : ℕ) : 
  total = 24 →
  r + v + a = total →
  r = 7 * (total / (7 + 3 + 2)) →
  v = 3 * (total / (7 + 3 + 2)) →
  a = 2 * (total / (7 + 3 + 2)) →
  v = 6 :=
by sorry

end NUMINAMATH_CALUDE_vikas_questions_l1847_184723


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1847_184739

theorem complex_number_quadrant (z : ℂ) (h : z * Complex.I = -2 + Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1847_184739


namespace NUMINAMATH_CALUDE_abs_T_equals_1024_l1847_184769

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^18 - (1 - i)^18

-- Theorem statement
theorem abs_T_equals_1024 : Complex.abs T = 1024 := by
  sorry

end NUMINAMATH_CALUDE_abs_T_equals_1024_l1847_184769


namespace NUMINAMATH_CALUDE_expression_evaluation_l1847_184726

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : 3 * x - y / 3 ≠ 0) :
  (3 * x - y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹)^2 = (y + 9 * x)^2 / (3 * x^2 * y^2 * (9 * x - y)) :=
by sorry


end NUMINAMATH_CALUDE_expression_evaluation_l1847_184726


namespace NUMINAMATH_CALUDE_graph_vertical_shift_l1847_184744

-- Define a continuous function f on the real line
variable (f : ℝ → ℝ)
variable (h : Continuous f)

-- Define the vertical shift operation
def verticalShift (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

-- Theorem statement
theorem graph_vertical_shift :
  ∀ (x y : ℝ), y = f x + 2 ↔ y = (verticalShift f 2) x :=
by sorry

end NUMINAMATH_CALUDE_graph_vertical_shift_l1847_184744


namespace NUMINAMATH_CALUDE_scalene_triangle_angle_difference_l1847_184765

/-- A scalene triangle with one angle of 80 degrees can have a difference of 80 degrees between its other two angles. -/
theorem scalene_triangle_angle_difference : ∃ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- angles are positive
  a + b + c = 180 ∧  -- sum of angles in a triangle is 180°
  a = 80 ∧  -- one angle is 80°
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧  -- all angles are different (scalene)
  |b - c| = 80  -- difference between other two angles is 80°
:= by sorry

end NUMINAMATH_CALUDE_scalene_triangle_angle_difference_l1847_184765


namespace NUMINAMATH_CALUDE_leopard_arrangement_count_l1847_184704

/-- The number of snow leopards -/
def total_leopards : ℕ := 9

/-- The number of leopards with special placement requirements -/
def special_leopards : ℕ := 3

/-- The number of ways to arrange the shortest two leopards at the ends -/
def shortest_arrangements : ℕ := 2

/-- The number of ways to place the tallest leopard in the middle -/
def tallest_arrangement : ℕ := 1

/-- The number of remaining leopards to be arranged -/
def remaining_leopards : ℕ := total_leopards - special_leopards

/-- Theorem: The number of ways to arrange the leopards is 1440 -/
theorem leopard_arrangement_count : 
  shortest_arrangements * tallest_arrangement * Nat.factorial remaining_leopards = 1440 := by
  sorry

end NUMINAMATH_CALUDE_leopard_arrangement_count_l1847_184704


namespace NUMINAMATH_CALUDE_percentage_change_condition_l1847_184782

theorem percentage_change_condition
  (p q r M : ℝ)
  (hp : p > 0)
  (hq : 0 < q ∧ q < 100)
  (hr : 0 < r ∧ r < 100)
  (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) * (1 - r / 100) > M ↔
  p > (100 * (q + r)) / (100 - q - r) :=
by sorry

end NUMINAMATH_CALUDE_percentage_change_condition_l1847_184782


namespace NUMINAMATH_CALUDE_modified_riemann_zeta_sum_l1847_184772

noncomputable def ξ (x : ℝ) : ℝ := ∑' n, (1 : ℝ) / (2 * n) ^ x

theorem modified_riemann_zeta_sum (h : ∀ x > 2, ξ x = ∑' n, (1 : ℝ) / (2 * n) ^ x) :
  ∑' k, ξ (2 * k + 1) = 1 := by sorry

end NUMINAMATH_CALUDE_modified_riemann_zeta_sum_l1847_184772


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_one_i_l1847_184710

/-- A complex number z is purely imaginary if and only if it has the form i (i.e., a = 1 and b = 0) -/
theorem purely_imaginary_iff_one_i (z : ℂ) : 
  (∃ (a b : ℝ), z = Complex.I * a + b) → 
  (z.re = 0 ∧ z.im ≠ 0) ↔ z = Complex.I :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_one_i_l1847_184710


namespace NUMINAMATH_CALUDE_four_digit_sum_2008_l1847_184741

theorem four_digit_sum_2008 : ∃ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (n + (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) = 2008) ∧
  (∃ m : ℕ, m ≠ n ∧ 
    (1000 ≤ m ∧ m < 10000) ∧ 
    (m + (m / 1000 + (m / 100 % 10) + (m / 10 % 10) + (m % 10)) = 2008)) :=
by sorry

#check four_digit_sum_2008

end NUMINAMATH_CALUDE_four_digit_sum_2008_l1847_184741


namespace NUMINAMATH_CALUDE_simplify_expressions_l1847_184778

theorem simplify_expressions (x y : ℝ) :
  (3 * x - 2 * y + 1 + 3 * y - 2 * x - 5 = x + y - 4) ∧
  ((2 * x^4 - 5 * x^2 - 4 * x + 3) - (3 * x^3 - 5 * x^2 - 4 * x) = 2 * x^4 - 3 * x^3 + 3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1847_184778


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1847_184757

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a * b * c) ≥ 343 :=
sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a * b * c) = 343 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1847_184757


namespace NUMINAMATH_CALUDE_student_polynomial_correction_l1847_184738

/-- Given a polynomial P(x) that satisfies P(x) - 3x^2 = x^2 - 4x + 1,
    prove that P(x) * (-3x^2) = -12x^4 + 12x^3 - 3x^2 -/
theorem student_polynomial_correction (P : ℝ → ℝ) :
  (∀ x, P x - 3 * x^2 = x^2 - 4 * x + 1) →
  (∀ x, P x * (-3 * x^2) = -12 * x^4 + 12 * x^3 - 3 * x^2) :=
by sorry

end NUMINAMATH_CALUDE_student_polynomial_correction_l1847_184738


namespace NUMINAMATH_CALUDE_daves_remaining_apps_l1847_184764

/-- Represents the number of apps and files on Dave's phone -/
structure PhoneContent where
  apps : ℕ
  files : ℕ

/-- The initial state of Dave's phone -/
def initial : PhoneContent := { apps := 11, files := 3 }

/-- The final state of Dave's phone after deletion -/
def final : PhoneContent := { apps := 2, files := 24 }

/-- Theorem stating that the final number of apps on Dave's phone is 2 -/
theorem daves_remaining_apps :
  final.apps = 2 ∧
  final.files = 24 ∧
  final.files = final.apps + 22 :=
by sorry

end NUMINAMATH_CALUDE_daves_remaining_apps_l1847_184764


namespace NUMINAMATH_CALUDE_wire_cutting_l1847_184708

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) :
  total_length = 70 →
  ratio = 2 / 3 →
  shorter_piece + (shorter_piece + ratio * shorter_piece) = total_length →
  shorter_piece = 26.25 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l1847_184708


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1847_184719

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1847_184719


namespace NUMINAMATH_CALUDE_increasing_digits_mod_1000_l1847_184707

/-- The number of ways to distribute n identical objects into k distinct boxes -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of 8-digit positive integers with digits in increasing order (1-8, repetition allowed) -/
def M : ℕ := starsAndBars 8 8

theorem increasing_digits_mod_1000 : M % 1000 = 435 := by sorry

end NUMINAMATH_CALUDE_increasing_digits_mod_1000_l1847_184707


namespace NUMINAMATH_CALUDE_yella_computer_usage_l1847_184771

def days_in_week : ℕ := 7
def hours_per_day_this_week : ℕ := 8
def hours_difference : ℕ := 35

def computer_usage_last_week : ℕ := days_in_week * hours_per_day_this_week + hours_difference

theorem yella_computer_usage :
  computer_usage_last_week = 91 := by
sorry

end NUMINAMATH_CALUDE_yella_computer_usage_l1847_184771


namespace NUMINAMATH_CALUDE_factorial_divisibility_l1847_184751

theorem factorial_divisibility (n : ℕ) (p : ℕ) (h_pos : n > 0) (h_prime : Nat.Prime p) 
  (h_div : p ^ p ∣ Nat.factorial n) : p ^ (p + 1) ∣ Nat.factorial n := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l1847_184751


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l1847_184718

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem line_parallel_to_intersection_of_parallel_planes
  (a : Line) (α β : Plane) (b : Line)
  (h1 : parallel_line_plane a α)
  (h2 : parallel_line_plane a β)
  (h3 : intersect α β = b) :
  parallel_line_line a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l1847_184718


namespace NUMINAMATH_CALUDE_solution_set_is_closed_unit_interval_l1847_184714

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- Define the set of a that satisfy f(1) ≤ f(a)
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {a | f 1 ≤ f a}

-- State the theorem
theorem solution_set_is_closed_unit_interval
  (f : ℝ → ℝ) (h_even : is_even f) (h_incr : increasing_on_nonpositive f) :
  solution_set f = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_closed_unit_interval_l1847_184714


namespace NUMINAMATH_CALUDE_tangent_sum_equality_l1847_184785

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define the tangent line
def TangentLine (p : ℝ × ℝ) (c : Circle) : ℝ × ℝ → Prop := sorry

-- State that the circles are tangent
def CirclesTangent (c1 c2 : Circle) : Prop := sorry

-- State that the triangle is equilateral
def IsEquilateral (t : Triangle) : Prop := sorry

-- State that the triangle is inscribed in the larger circle
def Inscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Define the length of a tangent line
def TangentLength (p : ℝ × ℝ) (c : Circle) : ℝ := sorry

-- Main theorem
theorem tangent_sum_equality 
  (c1 c2 : Circle) 
  (t : Triangle) 
  (h1 : CirclesTangent c1 c2) 
  (h2 : IsEquilateral t) 
  (h3 : Inscribed t c1) :
  ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    TangentLength (t.vertices i) c2 = 
    TangentLength (t.vertices j) c2 + TangentLength (t.vertices k) c2 :=
sorry

end NUMINAMATH_CALUDE_tangent_sum_equality_l1847_184785


namespace NUMINAMATH_CALUDE_triple_overlap_area_is_six_l1847_184733

/-- Represents a rectangular rug with width and height -/
structure Rug where
  width : ℝ
  height : ℝ

/-- Represents the hall and the rugs placed in it -/
structure HallWithRugs where
  hallSize : ℝ
  rug1 : Rug
  rug2 : Rug
  rug3 : Rug

/-- Calculates the area covered by all three rugs in the hall -/
def tripleOverlapArea (hall : HallWithRugs) : ℝ :=
  2 * 3

/-- Theorem stating that the area covered by all three rugs is 6 square meters -/
theorem triple_overlap_area_is_six (hall : HallWithRugs) 
  (h1 : hall.hallSize = 10)
  (h2 : hall.rug1 = ⟨6, 8⟩)
  (h3 : hall.rug2 = ⟨6, 6⟩)
  (h4 : hall.rug3 = ⟨5, 7⟩) :
  tripleOverlapArea hall = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_overlap_area_is_six_l1847_184733


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1847_184747

theorem average_of_remaining_numbers
  (total : ℝ)
  (avg_all : ℝ)
  (avg_first_two : ℝ)
  (avg_second_two : ℝ)
  (h1 : total = 6)
  (h2 : avg_all = 2.80)
  (h3 : avg_first_two = 2.4)
  (h4 : avg_second_two = 2.3) :
  (total * avg_all - 2 * avg_first_two - 2 * avg_second_two) / 2 = 3.7 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1847_184747


namespace NUMINAMATH_CALUDE_jasons_shopping_expenses_l1847_184792

theorem jasons_shopping_expenses (total_spent jacket_price : ℚ) 
  (h1 : total_spent = 14.28)
  (h2 : jacket_price = 4.74) :
  total_spent - jacket_price = 9.54 := by
  sorry

end NUMINAMATH_CALUDE_jasons_shopping_expenses_l1847_184792


namespace NUMINAMATH_CALUDE_existence_of_sequence_l1847_184758

theorem existence_of_sequence (α : ℝ) (n : ℕ) (h_α : 0 < α ∧ α < 1) (h_n : 0 < n) :
  ∃ (a : ℕ → ℕ), 
    (∀ i ∈ Finset.range n, 1 ≤ a i) ∧
    (∀ i ∈ Finset.range (n-1), a i < a (i+1)) ∧
    (∀ i ∈ Finset.range n, a i ≤ 2^(n-1)) ∧
    (∀ i ∈ Finset.range (n-1), ⌊(α^(i+1) : ℝ) * (a (i+1) : ℝ)⌋ ≥ ⌊(α^i : ℝ) * (a i : ℝ)⌋) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l1847_184758
