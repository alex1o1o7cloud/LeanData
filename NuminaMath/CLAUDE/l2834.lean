import Mathlib

namespace parallel_vectors_y_value_l2834_283402

theorem parallel_vectors_y_value :
  ∀ (y : ℝ),
  let a : Fin 2 → ℝ := ![(-1), 3]
  let b : Fin 2 → ℝ := ![2, y]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, a i = k * b i)) →
  y = -6 := by
sorry

end parallel_vectors_y_value_l2834_283402


namespace complement_intersection_theorem_l2834_283485

def I : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {3,4,5}
def B : Set Nat := {1,3,6}

theorem complement_intersection_theorem : 
  (I \ A) ∩ (I \ B) = {2,7,8} := by sorry

end complement_intersection_theorem_l2834_283485


namespace linda_fourth_day_chocolates_l2834_283486

/-- The number of chocolates Linda ate on the first day -/
def first_day_chocolates : ℕ → ℕ := λ c => c - 24

/-- The total number of chocolates Linda ate over five days -/
def total_chocolates (c : ℕ) : ℕ :=
  (first_day_chocolates c) + (c - 16) + (c - 8) + c + (c + 8)

/-- Theorem stating that Linda ate 38 chocolates on the fourth day -/
theorem linda_fourth_day_chocolates :
  ∃ c : ℕ, total_chocolates c = 150 ∧ c = 38 := by
  sorry

end linda_fourth_day_chocolates_l2834_283486


namespace paintbrush_cost_l2834_283421

/-- The amount spent on paintbrushes given the total spent and costs of other items. -/
theorem paintbrush_cost (total_spent : ℝ) (canvas_cost : ℝ) (paint_cost : ℝ) (easel_cost : ℝ) 
  (h1 : total_spent = 90)
  (h2 : canvas_cost = 40)
  (h3 : paint_cost = canvas_cost / 2)
  (h4 : easel_cost = 15) :
  total_spent - (canvas_cost + paint_cost + easel_cost) = 15 :=
by sorry

end paintbrush_cost_l2834_283421


namespace a_a_a_zero_l2834_283465

def a (k : ℕ) : ℕ := (2 * k + 1) ^ k

theorem a_a_a_zero : a (a (a 0)) = 343 := by sorry

end a_a_a_zero_l2834_283465


namespace isosceles_triangle_relationship_l2834_283424

/-- Represents an isosceles triangle with given perimeter and slant length -/
structure IsoscelesTriangle where
  perimeter : ℝ
  slantLength : ℝ

/-- The base length of an isosceles triangle given its perimeter and slant length -/
def baseLength (triangle : IsoscelesTriangle) : ℝ :=
  triangle.perimeter - 2 * triangle.slantLength

/-- Theorem stating the functional relationship and valid range for an isosceles triangle -/
theorem isosceles_triangle_relationship (triangle : IsoscelesTriangle)
    (h_perimeter : triangle.perimeter = 12)
    (h_valid_slant : 3 < triangle.slantLength ∧ triangle.slantLength < 6) :
    baseLength triangle = 12 - 2 * triangle.slantLength ∧
    3 < triangle.slantLength ∧ triangle.slantLength < 6 := by
  sorry

end isosceles_triangle_relationship_l2834_283424


namespace quadratic_roots_property_l2834_283423

theorem quadratic_roots_property (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ + k + 1 = 0 ∧ 
    x₂^2 - 4*x₂ + k + 1 = 0 ∧
    3/x₁ + 3/x₂ = x₁*x₂ - 4) →
  k = -3 ∧ k ≤ 3 := by
sorry

end quadratic_roots_property_l2834_283423


namespace pet_store_siamese_cats_l2834_283418

/-- The number of Siamese cats initially in the pet store -/
def initial_siamese_cats : ℕ := 13

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℕ := 5

/-- The total number of cats sold during the sale -/
def cats_sold : ℕ := 10

/-- The number of cats remaining after the sale -/
def cats_remaining : ℕ := 8

/-- Theorem stating that the initial number of Siamese cats is 13 -/
theorem pet_store_siamese_cats :
  initial_siamese_cats = 13 ∧
  initial_siamese_cats + initial_house_cats = cats_sold + cats_remaining :=
by sorry

end pet_store_siamese_cats_l2834_283418


namespace skew_edge_prob_is_4_11_l2834_283454

/-- A cube with 12 edges -/
structure Cube :=
  (edges : Finset (Fin 12))
  (edge_count : edges.card = 12)

/-- Two edges of a cube are skew if they don't intersect and are not in the same plane -/
def are_skew (c : Cube) (e1 e2 : Fin 12) : Prop := sorry

/-- The number of edges skew to any given edge in a cube -/
def skew_edge_count (c : Cube) : ℕ := 4

/-- The probability of selecting two skew edges from a cube -/
def skew_edge_probability (c : Cube) : ℚ :=
  (skew_edge_count c : ℚ) / (c.edges.card - 1 : ℚ)

/-- Theorem: The probability of selecting two skew edges from a cube is 4/11 -/
theorem skew_edge_prob_is_4_11 (c : Cube) : 
  skew_edge_probability c = 4 / 11 := by sorry

end skew_edge_prob_is_4_11_l2834_283454


namespace max_red_socks_l2834_283477

/-- The maximum number of red socks in a dresser with specific conditions -/
theorem max_red_socks (t : ℕ) (h1 : t ≤ 2500) :
  let p := 12 / 23
  ∃ r : ℕ, r ≤ t ∧
    (r * (r - 1) + (t - r) * (t - r - 1)) / (t * (t - 1)) = p ∧
    (∀ r' : ℕ, r' ≤ t →
      (r' * (r' - 1) + (t - r') * (t - r' - 1)) / (t * (t - 1)) = p →
      r' ≤ r) ∧
    r = 1225 :=
sorry

end max_red_socks_l2834_283477


namespace parity_of_F_l2834_283422

/-- F(n) is the number of ways to express n as the sum of three different positive integers -/
def F (n : ℕ) : ℕ := sorry

/-- Main theorem about the parity of F(n) -/
theorem parity_of_F (n : ℕ) (hn : n > 0) :
  (n % 6 = 2 ∨ n % 6 = 4 → F n % 2 = 0) ∧
  (n % 6 = 0 → F n % 2 = 1) := by
  sorry

end parity_of_F_l2834_283422


namespace modular_arithmetic_properties_l2834_283419

theorem modular_arithmetic_properties (a b c d m : ℤ) 
  (h1 : a ≡ b [ZMOD m]) 
  (h2 : c ≡ d [ZMOD m]) : 
  (a + c ≡ b + d [ZMOD m]) ∧ (a * c ≡ b * d [ZMOD m]) := by
  sorry

end modular_arithmetic_properties_l2834_283419


namespace correct_equation_l2834_283482

/-- Represents the meeting problem of two people walking towards each other -/
def meeting_problem (total_distance : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : Prop :=
  time * (speed1 + speed2) = total_distance

theorem correct_equation : 
  let total_distance : ℝ := 25
  let time : ℝ := 3
  let speed1 : ℝ := 4
  let speed2 : ℝ := x
  meeting_problem total_distance time speed1 speed2 ↔ 3 * (4 + x) = 25 := by
  sorry

end correct_equation_l2834_283482


namespace sum_of_numbers_with_lcm_and_ratio_l2834_283476

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ+) : 
  Nat.lcm a b = 42 → 
  a * 3 = b * 2 → 
  (a:ℝ) + (b:ℝ) = 70 := by
sorry

end sum_of_numbers_with_lcm_and_ratio_l2834_283476


namespace time_calculation_correct_l2834_283471

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The start date and time -/
def startDateTime : DateTime :=
  { year := 2021, month := 1, day := 5, hour := 15, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 5050

/-- The expected end date and time -/
def expectedEndDateTime : DateTime :=
  { year := 2021, month := 1, day := 9, hour := 3, minute := 10 }

theorem time_calculation_correct :
  addMinutes startDateTime minutesToAdd = expectedEndDateTime := by sorry

end time_calculation_correct_l2834_283471


namespace lcm_factor_problem_l2834_283435

theorem lcm_factor_problem (A B : ℕ) (X : ℕ+) :
  A = 400 →
  Nat.gcd A B = 25 →
  Nat.lcm A B = 25 * X * 16 →
  X = 1 := by
  sorry

end lcm_factor_problem_l2834_283435


namespace intersection_of_M_and_N_l2834_283415

def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

theorem intersection_of_M_and_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l2834_283415


namespace base8_digit_product_l2834_283488

/-- Convert a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculate the product of a list of natural numbers -/
def productList (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 8 representation of 6543₁₀ is 168 -/
theorem base8_digit_product :
  productList (toBase8 6543) = 168 :=
sorry

end base8_digit_product_l2834_283488


namespace fixed_distance_point_l2834_283409

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given vectors a and b, and a vector p satisfying ||p - b|| = 3||p - a||,
    p is at a fixed distance from (9/8)a + (-1/8)b. -/
theorem fixed_distance_point (a b p : V) 
    (h : ‖p - b‖ = 3 * ‖p - a‖) : 
    ∃ (c : ℝ), ∀ (q : V), ‖p - q‖ = c ↔ q = (9/8 : ℝ) • a + (-1/8 : ℝ) • b :=
sorry

end fixed_distance_point_l2834_283409


namespace inequality_proof_l2834_283470

theorem inequality_proof (a b c : ℝ) (h : a ≠ b) :
  Real.sqrt ((a - c)^2 + b^2) + Real.sqrt (a^2 + (b - c)^2) > Real.sqrt 2 * abs (a - b) := by
  sorry

end inequality_proof_l2834_283470


namespace road_travel_cost_example_l2834_283498

/-- Calculates the cost of traveling two intersecting roads on a rectangular lawn. -/
def road_travel_cost (lawn_length lawn_width road_width cost_per_sqm : ℝ) : ℝ :=
  let road1_area := road_width * lawn_width
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_road_area := road1_area + road2_area - intersection_area
  total_road_area * cost_per_sqm

/-- Theorem stating that the cost of traveling two intersecting roads on a specific rectangular lawn is 4500. -/
theorem road_travel_cost_example : road_travel_cost 100 60 10 3 = 4500 := by
  sorry

end road_travel_cost_example_l2834_283498


namespace integral_equality_l2834_283478

theorem integral_equality : ∫ x in (1 : ℝ)..Real.sqrt 3, 
  (x^(2*x^2 + 1) + Real.log (x^(2*x^(2*x^2 + 1)))) = 13 := by sorry

end integral_equality_l2834_283478


namespace repeating_decimal_equality_l2834_283438

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℚ
  repeatingPart : ℚ
  repeatingPartLength : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def RepeatingDecimal.toRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + x.repeatingPart / (1 - (1 / 10 ^ x.repeatingPartLength))

/-- Theorem stating that 0.3̅206̅ is equal to 5057/9990 -/
theorem repeating_decimal_equality : 
  let x : RepeatingDecimal := ⟨3/10, 206/1000, 3⟩
  x.toRational = 5057 / 9990 := by
  sorry

end repeating_decimal_equality_l2834_283438


namespace plate_arrangement_theorem_l2834_283455

/-- The number of ways to arrange plates around a circular table. -/
def circularArrangements (blue red green yellow : ℕ) : ℕ :=
  Nat.factorial (blue + red + green + yellow - 1) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial yellow)

/-- The number of ways to arrange plates around a circular table with adjacent green plates. -/
def circularArrangementsWithAdjacentGreen (blue red green yellow : ℕ) : ℕ :=
  Nat.factorial (blue + red + 1 + yellow - 1) /
  (Nat.factorial blue * Nat.factorial red * Nat.factorial yellow) *
  Nat.factorial green

/-- The number of ways to arrange plates around a circular table without adjacent green plates. -/
def circularArrangementsWithoutAdjacentGreen (blue red green yellow : ℕ) : ℕ :=
  circularArrangements blue red green yellow -
  circularArrangementsWithAdjacentGreen blue red green yellow

theorem plate_arrangement_theorem :
  circularArrangementsWithoutAdjacentGreen 4 3 3 1 = 2520 :=
by sorry

end plate_arrangement_theorem_l2834_283455


namespace quadruple_prime_equation_l2834_283487

theorem quadruple_prime_equation :
  ∀ p q r : ℕ, ∀ n : ℕ+,
    Prime p ∧ Prime q ∧ Prime r →
    p^2 = q^2 + r^(n : ℕ) →
    (p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4) := by
  sorry

end quadruple_prime_equation_l2834_283487


namespace pet_store_dogs_l2834_283411

/-- The number of dogs in a pet store after receiving additional dogs over two days -/
def total_dogs (initial : ℕ) (sunday_addition : ℕ) (monday_addition : ℕ) : ℕ :=
  initial + sunday_addition + monday_addition

/-- Theorem stating that starting with 2 dogs, adding 5 on Sunday and 3 on Monday results in 10 dogs -/
theorem pet_store_dogs : total_dogs 2 5 3 = 10 := by
  sorry

end pet_store_dogs_l2834_283411


namespace triangle_side_value_l2834_283457

/-- A triangle with sides a, b, and c satisfies the triangle inequality -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of possible values for x -/
def possible_values : Set ℕ := {2, 4, 6, 8}

/-- The theorem statement -/
theorem triangle_side_value (x : ℕ) (hx : x ∈ possible_values) :
  is_triangle 2 x 6 ↔ x = 6 := by
  sorry


end triangle_side_value_l2834_283457


namespace decagon_triangle_probability_l2834_283401

/-- A decagon is a polygon with 10 sides -/
def Decagon : Type := Fin 10

/-- The probability of choosing 3 distinct vertices from a decagon that form a triangle
    with sides that are all edges of the decagon -/
theorem decagon_triangle_probability : 
  (Nat.choose 10 3 : ℚ)⁻¹ * 10 = 1 / 12 := by sorry

end decagon_triangle_probability_l2834_283401


namespace lines_parallel_implies_a_eq_one_lines_perpendicular_implies_a_eq_zero_l2834_283462

/-- Two lines in the xy-plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define the first line l₁: x + ay - 2a - 2 = 0 -/
def l₁ (a : ℝ) : Line2D := ⟨1, a, -2*a - 2⟩

/-- Define the second line l₂: ax + y - 1 - a = 0 -/
def l₂ (a : ℝ) : Line2D := ⟨a, 1, -1 - a⟩

/-- Two lines are parallel if their slopes are equal -/
def parallel (l₁ l₂ : Line2D) : Prop := l₁.a * l₂.b = l₂.a * l₁.b

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (l₁ l₂ : Line2D) : Prop := l₁.a * l₂.a + l₁.b * l₂.b = 0

theorem lines_parallel_implies_a_eq_one :
  ∀ a : ℝ, parallel (l₁ a) (l₂ a) → a = 1 := by sorry

theorem lines_perpendicular_implies_a_eq_zero :
  ∀ a : ℝ, perpendicular (l₁ a) (l₂ a) → a = 0 := by sorry

end lines_parallel_implies_a_eq_one_lines_perpendicular_implies_a_eq_zero_l2834_283462


namespace tan_45_plus_half_inv_plus_abs_neg_two_equals_five_l2834_283417

theorem tan_45_plus_half_inv_plus_abs_neg_two_equals_five :
  Real.tan (π / 4) + (1 / 2)⁻¹ + |(-2)| = 5 := by
  sorry

end tan_45_plus_half_inv_plus_abs_neg_two_equals_five_l2834_283417


namespace six_digit_divisibility_l2834_283481

theorem six_digit_divisibility (a b c : Nat) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≥ 0) (h4 : b ≤ 9) (h5 : c ≥ 0) (h6 : c ≤ 9) :
  ∃ k : Nat, 1001 * k = a * 100000 + b * 10000 + c * 1000 + a * 100 + b * 10 + c :=
sorry

end six_digit_divisibility_l2834_283481


namespace prop_p_iff_prop_q_l2834_283445

theorem prop_p_iff_prop_q (m : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 1| ≥ m) ↔
  (∃ x : ℝ, x^2 - 2*m*x + m^2 + m - 3 = 0) :=
by sorry

end prop_p_iff_prop_q_l2834_283445


namespace complex_magnitude_equation_l2834_283473

theorem complex_magnitude_equation (x : ℝ) :
  x > 0 ∧ Complex.abs (3 + x * Complex.I) = 7 ↔ x = 2 * Real.sqrt 10 := by
  sorry

end complex_magnitude_equation_l2834_283473


namespace problem_statement_problem_statement_2_l2834_283427

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem problem_statement (a : ℝ) (h1 : a > 1) :
  (∀ x ∈ Set.Icc 1 a, f a x ∈ Set.Icc 1 a ∧ ∀ y ∈ Set.Icc 1 a, ∃ x ∈ Set.Icc 1 a, f a x = y) →
  a = 2 :=
sorry

theorem problem_statement_2 (a : ℝ) (h1 : a > 1) :
  (∀ x y : ℝ, x < y ∧ y ≤ 2 → f a x > f a y) ∧
  (∀ x ∈ Set.Icc 1 2, f a x ≤ 0) →
  a ≥ 3 :=
sorry

end problem_statement_problem_statement_2_l2834_283427


namespace derivative_of_y_l2834_283451

noncomputable def y (x : ℝ) : ℝ := Real.exp (-5 * x + 2)

theorem derivative_of_y (x : ℝ) :
  deriv y x = -5 * Real.exp (-5 * x + 2) := by
  sorry

end derivative_of_y_l2834_283451


namespace brick_length_l2834_283453

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: For a brick with width 4 cm, height 2 cm, and surface area 136 square centimeters, the length of the brick is 10 cm -/
theorem brick_length : 
  ∃ (l : ℝ), surface_area l 4 2 = 136 ∧ l = 10 :=
sorry

end brick_length_l2834_283453


namespace tailor_cut_difference_l2834_283446

theorem tailor_cut_difference (skirt_cut pants_cut : ℝ) : 
  skirt_cut = 0.75 → pants_cut = 0.5 → skirt_cut - pants_cut = 0.25 := by
  sorry

end tailor_cut_difference_l2834_283446


namespace essay_word_limit_l2834_283439

/-- The word limit for Vinnie's essay --/
def word_limit (saturday_words sunday_words exceeded_words : ℕ) : ℕ :=
  saturday_words + sunday_words - exceeded_words

/-- Theorem: The word limit for Vinnie's essay is 1000 words --/
theorem essay_word_limit :
  word_limit 450 650 100 = 1000 := by
  sorry

end essay_word_limit_l2834_283439


namespace jane_score_is_12_l2834_283404

/-- Represents the score calculation for a modified AMC 8 contest --/
def modified_amc_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - (incorrect : ℚ) / 2

/-- Theorem stating that Jane's score in the modified AMC 8 contest is 12 --/
theorem jane_score_is_12 :
  let total_questions : ℕ := 35
  let correct_answers : ℕ := 18
  let incorrect_answers : ℕ := 12
  let unanswered_questions : ℕ := 5
  modified_amc_score correct_answers incorrect_answers unanswered_questions = 12 := by
  sorry

#eval modified_amc_score 18 12 5

end jane_score_is_12_l2834_283404


namespace max_2012_gons_sharing_vertices_not_sides_target_2012_gons_impossible_l2834_283458

/-- The number of vertices in each polygon -/
def n : ℕ := 2012

/-- The maximum number of different polygons we want to prove is impossible -/
def target_polygons : ℕ := 1006

/-- The actual maximum number of polygons possible -/
def max_polygons : ℕ := 1005

theorem max_2012_gons_sharing_vertices_not_sides :
  ∀ (num_polygons : ℕ),
    (∀ (v : Fin n), num_polygons * 2 ≤ n - 1) →
    num_polygons ≤ max_polygons :=
by sorry

theorem target_2012_gons_impossible :
  ¬(∀ (v : Fin n), target_polygons * 2 ≤ n - 1) :=
by sorry

end max_2012_gons_sharing_vertices_not_sides_target_2012_gons_impossible_l2834_283458


namespace locus_of_tangent_points_l2834_283405

/-- Given a parabola y^2 = 2px and a constant k, prove that the locus of points P(x, y) 
    from which tangents can be drawn to the parabola with slopes m1 and m2 satisfying 
    m1 * m2^2 + m1^2 * m2 = k, is the parabola x^2 = (p / (2k)) * y -/
theorem locus_of_tangent_points (p k : ℝ) (hp : p > 0) (hk : k ≠ 0) :
  ∀ x y m1 m2 : ℝ,
  (∃ x1 y1 x2 y2 : ℝ,
    y1^2 = 2 * p * x1 ∧ 
    y2^2 = 2 * p * x2 ∧
    m1 = p / y1 ∧
    m2 = p / y2 ∧
    2 * y = y1 + y2 ∧
    x^2 = x1 * x2 ∧
    m1 * m2^2 + m1^2 * m2 = k) →
  x^2 = (p / (2 * k)) * y := by
  sorry

end locus_of_tangent_points_l2834_283405


namespace line_intersects_circle_l2834_283434

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The line equation -/
def line_equation (a x y : ℝ) : Prop :=
  a*x + y - 5 = 0

/-- The chord length when the line intersects the circle -/
def chord_length : ℝ := 4

/-- The theorem statement -/
theorem line_intersects_circle (a : ℝ) :
  (∃ x y : ℝ, circle_equation x y ∧ line_equation a x y) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ ∧ line_equation a x₁ y₁ ∧
    circle_equation x₂ y₂ ∧ line_equation a x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2) →
  a = 2 :=
sorry

end line_intersects_circle_l2834_283434


namespace election_votes_total_l2834_283441

theorem election_votes_total (total votes_in_favor votes_against votes_neutral : ℕ) : 
  votes_in_favor = votes_against + 78 →
  votes_against = (375 * total) / 1000 →
  votes_neutral = (125 * total) / 1000 →
  total = votes_in_favor + votes_against + votes_neutral →
  total = 624 := by
sorry

end election_votes_total_l2834_283441


namespace inequality_solution_range_l2834_283456

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x + |x - 1| ≤ a) → a ≥ 1 := by sorry

end inequality_solution_range_l2834_283456


namespace functional_equation_solution_l2834_283443

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  g 0 = 1 ∧ ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y

/-- The main theorem stating that g(x) = 5^x - 3^x is the unique solution -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
    ∀ x : ℝ, g x = 5^x - 3^x := by
  sorry

end functional_equation_solution_l2834_283443


namespace line_tangent_to_circle_l2834_283450

/-- The line equation x cos θ + y sin θ = 1 is tangent to the circle x² + y² = 1 -/
theorem line_tangent_to_circle :
  ∀ θ : ℝ, 
  (∀ x y : ℝ, x * Real.cos θ + y * Real.sin θ = 1 → x^2 + y^2 = 1) ∧
  (∃ x y : ℝ, x * Real.cos θ + y * Real.sin θ = 1 ∧ x^2 + y^2 = 1) :=
by sorry

end line_tangent_to_circle_l2834_283450


namespace tangent_line_equation_l2834_283447

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 2*x + 2

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := f' x₀
  (4 : ℝ) * x - y - 1 = 0 ↔ y - y₀ = m * (x - x₀) :=
by sorry

end tangent_line_equation_l2834_283447


namespace stone_pile_impossibility_l2834_283463

theorem stone_pile_impossibility :
  ∀ (n : ℕ) (stones piles : ℕ → ℕ),
  (stones 0 = 1001 ∧ piles 0 = 1) →
  (∀ k, stones (k + 1) + piles (k + 1) = stones k + piles k) →
  (∀ k, stones (k + 1) = stones k - 1) →
  (∀ k, piles (k + 1) = piles k + 1) →
  ¬∃ k, stones k = 3 * piles k :=
by sorry

end stone_pile_impossibility_l2834_283463


namespace inequality_proof_l2834_283467

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 3) :
  1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a) ≥ 3 / 2 := by
  sorry

end inequality_proof_l2834_283467


namespace N_is_perfect_square_l2834_283499

/-- Constructs the number N with n 1s, n+1 2s, ending with 25 -/
def constructN (n : ℕ) : ℕ :=
  (10^(2*n+2) + 10^(n+1)) / 9 + 25

/-- Theorem: For any positive n, the constructed N is a perfect square -/
theorem N_is_perfect_square (n : ℕ+) : ∃ m : ℕ, (constructN n) = m^2 := by
  sorry

end N_is_perfect_square_l2834_283499


namespace sunglasses_and_hats_probability_l2834_283494

theorem sunglasses_and_hats_probability (total_sunglasses : ℕ) (total_hats : ℕ) 
  (prob_hat_and_sunglasses : ℚ) :
  total_sunglasses = 75 →
  total_hats = 60 →
  prob_hat_and_sunglasses = 1 / 3 →
  (prob_hat_and_sunglasses * total_hats : ℚ) / total_sunglasses = 4 / 15 := by
sorry

end sunglasses_and_hats_probability_l2834_283494


namespace james_letter_frequency_l2834_283460

/-- Calculates how many times per week James writes letters to his friends -/
def letters_per_week (pages_per_year : ℕ) (weeks_per_year : ℕ) (pages_per_letter : ℕ) (num_friends : ℕ) : ℕ :=
  (pages_per_year / weeks_per_year) / (pages_per_letter * num_friends)

/-- Theorem stating that James writes letters 2 times per week -/
theorem james_letter_frequency :
  letters_per_week 624 52 3 2 = 2 := by
  sorry

end james_letter_frequency_l2834_283460


namespace jackson_holidays_l2834_283480

/-- The number of holidays Jackson takes per month -/
def holidays_per_month : ℕ := 3

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The total number of holidays Jackson takes in a year -/
def total_holidays : ℕ := holidays_per_month * months_per_year

theorem jackson_holidays : total_holidays = 36 := by
  sorry

end jackson_holidays_l2834_283480


namespace expected_baby_hawks_l2834_283493

/-- The number of kettles being tracked -/
def num_kettles : ℕ := 6

/-- The average number of pregnancies per kettle -/
def pregnancies_per_kettle : ℕ := 15

/-- The number of babies per pregnancy -/
def babies_per_pregnancy : ℕ := 4

/-- The percentage of babies lost -/
def loss_percentage : ℚ := 1/4

/-- The expected number of baby hawks this season -/
def expected_babies : ℕ := 270

theorem expected_baby_hawks :
  expected_babies = num_kettles * pregnancies_per_kettle * babies_per_pregnancy - 
    (num_kettles * pregnancies_per_kettle * babies_per_pregnancy * loss_percentage).floor := by
  sorry

end expected_baby_hawks_l2834_283493


namespace rectangle_area_with_circles_l2834_283483

/-- The area of a rectangle surrounded by four circles -/
theorem rectangle_area_with_circles (r : ℝ) (h1 : r = 3) : ∃ (length width : ℝ),
  length = 2 * r * 2 ∧ 
  width = 2 * r ∧
  length * width = 72 := by
  sorry

end rectangle_area_with_circles_l2834_283483


namespace completing_square_equivalence_l2834_283410

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 6*x - 1 = 0) ↔ ((x - 3)^2 = 10) := by sorry

end completing_square_equivalence_l2834_283410


namespace popped_kernel_red_probability_l2834_283413

theorem popped_kernel_red_probability
  (total_kernels : ℝ)
  (red_ratio : ℝ)
  (green_ratio : ℝ)
  (red_pop_ratio : ℝ)
  (green_pop_ratio : ℝ)
  (h1 : red_ratio = 3/4)
  (h2 : green_ratio = 1/4)
  (h3 : red_pop_ratio = 3/5)
  (h4 : green_pop_ratio = 3/4)
  (h5 : red_ratio + green_ratio = 1) :
  let red_kernels := red_ratio * total_kernels
  let green_kernels := green_ratio * total_kernels
  let popped_red := red_pop_ratio * red_kernels
  let popped_green := green_pop_ratio * green_kernels
  let total_popped := popped_red + popped_green
  (popped_red / total_popped) = 12/17 :=
by sorry

end popped_kernel_red_probability_l2834_283413


namespace family_weight_calculation_l2834_283495

/-- The total weight of a family consisting of a grandmother, her daughter, and her grandchild. -/
def total_weight (mother_weight daughter_weight grandchild_weight : ℝ) : ℝ :=
  mother_weight + daughter_weight + grandchild_weight

/-- Theorem stating the total weight of the family under given conditions. -/
theorem family_weight_calculation :
  ∀ (mother_weight daughter_weight grandchild_weight : ℝ),
    daughter_weight + grandchild_weight = 60 →
    grandchild_weight = (1 / 5) * mother_weight →
    daughter_weight = 48 →
    total_weight mother_weight daughter_weight grandchild_weight = 120 :=
by
  sorry

end family_weight_calculation_l2834_283495


namespace intersection_line_equation_l2834_283432

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (A B : ℝ × ℝ),
    circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
    circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
    A ≠ B →
    ∀ (P : ℝ × ℝ),
      (∃ t : ℝ, P = t • A + (1 - t) • B) ↔ line P.1 P.2 :=
by sorry


end intersection_line_equation_l2834_283432


namespace factorial_350_trailing_zeros_l2834_283492

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_350_trailing_zeros :
  trailing_zeros 350 = 86 := by
  sorry

end factorial_350_trailing_zeros_l2834_283492


namespace function_composition_equality_l2834_283472

/-- Given a function f(x) = ax^2 - √3 where a > 0, prove that f(f(√3)) = -√3 implies a = √3/3 -/
theorem function_composition_equality (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - Real.sqrt 3
  f (f (Real.sqrt 3)) = -Real.sqrt 3 → a = Real.sqrt 3 / 3 := by
  sorry

end function_composition_equality_l2834_283472


namespace point_placement_theorem_l2834_283484

theorem point_placement_theorem : ∃ n : ℕ+, 9 * n - 8 = 82 := by
  sorry

end point_placement_theorem_l2834_283484


namespace fountain_area_l2834_283448

theorem fountain_area (AB CD : ℝ) (h1 : AB = 20) (h2 : CD = 12) : ∃ (r : ℝ), r^2 = 244 ∧ π * r^2 = 244 * π := by
  sorry

end fountain_area_l2834_283448


namespace three_leaf_clover_count_l2834_283403

/-- The number of leaves on a three-leaf clover -/
def three_leaf_count : ℕ := 3

/-- The number of leaves on a four-leaf clover -/
def four_leaf_count : ℕ := 4

/-- The total number of leaves collected -/
def total_leaves : ℕ := 100

/-- The number of four-leaf clovers found -/
def four_leaf_clovers : ℕ := 1

theorem three_leaf_clover_count :
  (total_leaves - four_leaf_count * four_leaf_clovers) / three_leaf_count = 32 := by
  sorry

end three_leaf_clover_count_l2834_283403


namespace matchstick_pattern_l2834_283452

/-- 
Given a sequence where:
- The first term is 5
- Each subsequent term increases by 3
Prove that the 20th term is 62
-/
theorem matchstick_pattern (a : ℕ → ℕ) 
  (h1 : a 1 = 5)
  (h2 : ∀ n : ℕ, n ≥ 2 → a n = a (n-1) + 3) :
  a 20 = 62 := by
  sorry

end matchstick_pattern_l2834_283452


namespace f_upper_bound_l2834_283491

theorem f_upper_bound (x : ℝ) (hx : x > 1) : Real.log x + Real.sqrt x - 1 < (3/2) * (x - 1) := by
  sorry

end f_upper_bound_l2834_283491


namespace inverse_g_at_neg_138_l2834_283420

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x^3 - 3

-- State the theorem
theorem inverse_g_at_neg_138 :
  g⁻¹ (-138) = -3 :=
sorry

end inverse_g_at_neg_138_l2834_283420


namespace sqrt_32_div_sqrt_2_equals_4_l2834_283497

theorem sqrt_32_div_sqrt_2_equals_4 : Real.sqrt 32 / Real.sqrt 2 = 4 := by
  sorry

end sqrt_32_div_sqrt_2_equals_4_l2834_283497


namespace wang_trip_distance_l2834_283468

/-- The distance between Mr. Wang's home and location A -/
def distance : ℝ := 330

theorem wang_trip_distance : 
  ∀ x : ℝ, 
  x > 0 → 
  (x / 100 + x / 120) - (x / 150 + 2 * x / 198) = 31 / 60 → 
  x = distance := by
sorry

end wang_trip_distance_l2834_283468


namespace max_shape_pairs_l2834_283431

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a pair of shapes: a corner and a 2x2 square -/
structure ShapePair where
  corner : Unit
  square : Unit

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the area occupied by a single ShapePair -/
def ShapePair.area : ℕ := 7  -- 3 for corner + 4 for 2x2 square

/-- The main theorem to prove -/
theorem max_shape_pairs (r : Rectangle) (h1 : r.width = 3) (h2 : r.height = 100) :
  ∃ (n : ℕ), n = 33 ∧ 
  n * ShapePair.area ≤ r.area ∧
  ∀ (m : ℕ), m * ShapePair.area ≤ r.area → m ≤ n :=
by sorry


end max_shape_pairs_l2834_283431


namespace fourDigitPermutationsFromSixIs360_l2834_283406

/-- The number of permutations of 4 digits chosen from a set of 6 digits -/
def fourDigitPermutationsFromSix : ℕ :=
  6 * 5 * 4 * 3

/-- Theorem stating that the number of four-digit numbers without repeating digits
    from the set {1, 2, 3, 4, 5, 6} is equal to 360 -/
theorem fourDigitPermutationsFromSixIs360 : fourDigitPermutationsFromSix = 360 := by
  sorry


end fourDigitPermutationsFromSixIs360_l2834_283406


namespace circle_within_circle_l2834_283461

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point is inside a circle if its distance from the center is less than the radius -/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

/-- A circle is contained within another circle if all its points are inside the larger circle -/
def is_contained (inner outer : Circle) : Prop :=
  ∀ p : ℝ × ℝ, is_inside p inner → is_inside p outer

theorem circle_within_circle (C : Circle) (A B : ℝ × ℝ) 
    (hA : is_inside A C) (hB : is_inside B C) :
  ∃ D : Circle, is_inside A D ∧ is_inside B D ∧ is_contained D C := by
  sorry

end circle_within_circle_l2834_283461


namespace randy_quiz_average_l2834_283430

/-- The number of quizzes Randy wants to have the average for -/
def n : ℕ := 5

/-- The sum of Randy's first four quiz scores -/
def initial_sum : ℕ := 374

/-- Randy's desired average -/
def desired_average : ℕ := 94

/-- Randy's next quiz score -/
def next_score : ℕ := 96

theorem randy_quiz_average : 
  (initial_sum + next_score : ℚ) / n = desired_average := by sorry

end randy_quiz_average_l2834_283430


namespace abc_reciprocal_sum_l2834_283437

theorem abc_reciprocal_sum (a b c : ℝ) 
  (h1 : a + 1/b = 9)
  (h2 : b + 1/c = 10)
  (h3 : c + 1/a = 11) :
  a * b * c + 1 / (a * b * c) = 960 := by
  sorry

end abc_reciprocal_sum_l2834_283437


namespace counterexample_exists_l2834_283429

theorem counterexample_exists : ∃ x y : ℝ, x + y = 5 ∧ ¬(x = 1 ∧ y = 4) := by
  sorry

end counterexample_exists_l2834_283429


namespace local_extremum_values_l2834_283440

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

-- State the theorem
theorem local_extremum_values (a b : ℝ) :
  (f a b 1 = 10) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≥ f a b 1) →
  a = -4 ∧ b = 11 := by
  sorry

end local_extremum_values_l2834_283440


namespace cube_surface_area_l2834_283489

/-- The surface area of a cube with edge length 2a is 24a² -/
theorem cube_surface_area (a : ℝ) : 
  6 * (2 * a)^2 = 24 * a^2 := by sorry

end cube_surface_area_l2834_283489


namespace at_least_one_genuine_certain_l2834_283408

theorem at_least_one_genuine_certain (total : ℕ) (genuine : ℕ) (defective : ℕ) (selected : ℕ)
  (h1 : total = 8)
  (h2 : genuine = 5)
  (h3 : defective = 3)
  (h4 : total = genuine + defective)
  (h5 : selected = 4) :
  ∀ (selection : Finset (Fin total)),
    selection.card = selected →
    ∃ (i : Fin total), i ∈ selection ∧ i.val < genuine :=
by sorry

end at_least_one_genuine_certain_l2834_283408


namespace min_blocks_for_wall_l2834_283459

/-- Represents a block in the wall -/
inductive Block
| OneFootBlock
| TwoFootBlock

/-- Represents a row of blocks in the wall -/
def Row := List Block

/-- The wall specification -/
structure WallSpec where
  length : Nat
  height : Nat
  blockHeight : Nat
  evenEnds : Bool
  staggeredJoins : Bool

/-- Checks if a row of blocks is valid according to the wall specification -/
def isValidRow (spec : WallSpec) (row : Row) : Prop := sorry

/-- Checks if a list of rows forms a valid wall according to the specification -/
def isValidWall (spec : WallSpec) (rows : List Row) : Prop := sorry

/-- Counts the total number of blocks in a list of rows -/
def countBlocks (rows : List Row) : Nat := sorry

/-- The main theorem to be proved -/
theorem min_blocks_for_wall (spec : WallSpec) : 
  spec.length = 102 ∧ 
  spec.height = 8 ∧ 
  spec.blockHeight = 1 ∧ 
  spec.evenEnds = true ∧ 
  spec.staggeredJoins = true → 
  ∃ (rows : List Row), 
    isValidWall spec rows ∧ 
    countBlocks rows = 416 ∧ 
    ∀ (otherRows : List Row), 
      isValidWall spec otherRows → 
      countBlocks otherRows ≥ 416 := by sorry

end min_blocks_for_wall_l2834_283459


namespace shaded_area_of_semicircle_l2834_283464

theorem shaded_area_of_semicircle (total_area : ℝ) (h : total_area > 0) :
  let num_parts : ℕ := 6
  let excluded_fraction : ℝ := 2 / 3
  let shaded_area : ℝ := total_area * (1 - excluded_fraction)
  shaded_area = total_area / 3 :=
by sorry

end shaded_area_of_semicircle_l2834_283464


namespace isosceles_triangle_base_length_l2834_283469

/-- An isosceles triangle with specific heights -/
structure IsoscelesTriangle where
  -- The height drawn to the base
  baseHeight : ℝ
  -- The height drawn to one of the equal sides
  sideHeight : ℝ
  -- Assumption that the triangle is isosceles
  isIsosceles : True

/-- The base of the triangle -/
def baseLength (triangle : IsoscelesTriangle) : ℝ :=
  7.5

/-- Theorem stating that for an isosceles triangle with given heights, the base length is 7.5 -/
theorem isosceles_triangle_base_length 
  (triangle : IsoscelesTriangle) 
  (h1 : triangle.baseHeight = 5) 
  (h2 : triangle.sideHeight = 6) : 
  baseLength triangle = 7.5 := by
  sorry

end isosceles_triangle_base_length_l2834_283469


namespace west_notation_l2834_283466

-- Define a type for direction
inductive Direction
  | East
  | West

-- Define a function that represents the notation for walking in a given direction
def walkNotation (dir : Direction) (distance : ℝ) : ℝ :=
  match dir with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem west_notation (d : ℝ) :
  walkNotation Direction.East d = d →
  walkNotation Direction.West d = -d :=
by sorry

end west_notation_l2834_283466


namespace min_value_expression_equality_condition_l2834_283433

theorem min_value_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^2 + y^2 + 4/x^2 + 2*y/x ≥ 2 * Real.sqrt 3 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^2 + y^2 + 4/x^2 + 2*y/x = 2 * Real.sqrt 3 ↔ 
  (x = Real.sqrt (Real.sqrt 3) ∨ x = -Real.sqrt (Real.sqrt 3)) ∧
  (y = -1 / x) :=
by sorry

end min_value_expression_equality_condition_l2834_283433


namespace equal_spacing_ratio_l2834_283416

/-- Given 6 equally spaced points on a number line from 0 to 1, 
    the ratio of the 3rd point's value to the 6th point's value is 0.5 -/
theorem equal_spacing_ratio : 
  ∀ (P Q R S T U : ℝ), 
    0 ≤ P ∧ P < Q ∧ Q < R ∧ R < S ∧ S < T ∧ T < U ∧ U = 1 →
    Q - P = R - Q ∧ R - Q = S - R ∧ S - R = T - S ∧ T - S = U - T →
    R / U = 1 / 2 := by
  sorry

end equal_spacing_ratio_l2834_283416


namespace white_balls_count_l2834_283428

theorem white_balls_count (total : ℕ) (p_red p_black : ℚ) (h_total : total = 50)
  (h_red : p_red = 15/100) (h_black : p_black = 45/100) :
  (total : ℚ) * (1 - p_red - p_black) = 20 := by
  sorry

end white_balls_count_l2834_283428


namespace hyperbola_eccentricity_l2834_283474

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola with the given properties is 2 + √3 -/
theorem hyperbola_eccentricity (h : Hyperbola) (F₁ F₂ A B : Point) :
  -- F₁ and F₂ are the left and right foci respectively
  -- A line passes through F₁ at a 60° angle
  -- The line intersects the y-axis at A and the right branch of the hyperbola at B
  -- A is the midpoint of F₁B
  (∃ (θ : ℝ), θ = Real.pi / 3 ∧ 
    A.x = 0 ∧
    B.x > 0 ∧
    (B.x - F₁.x) * Real.cos θ = (B.y - F₁.y) * Real.sin θ ∧
    A.x = (F₁.x + B.x) / 2 ∧
    A.y = (F₁.y + B.y) / 2) →
  -- The eccentricity of the hyperbola is 2 + √3
  h.a / Real.sqrt (h.a^2 + h.b^2) = 2 + Real.sqrt 3 := by
  sorry

end hyperbola_eccentricity_l2834_283474


namespace sum_of_nine_zero_seven_digits_l2834_283496

/-- A function that checks if a real number uses only digits 0 and 7 in base 10 --/
def uses_only_0_and_7 (a : ℝ) : Prop := sorry

/-- Theorem stating that any real number can be expressed as the sum of nine numbers,
    each using only digits 0 and 7 in base 10 --/
theorem sum_of_nine_zero_seven_digits (x : ℝ) : 
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ), 
    x = a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ ∧ 
    uses_only_0_and_7 a₁ ∧ uses_only_0_and_7 a₂ ∧ uses_only_0_and_7 a₃ ∧ 
    uses_only_0_and_7 a₄ ∧ uses_only_0_and_7 a₅ ∧ uses_only_0_and_7 a₆ ∧ 
    uses_only_0_and_7 a₇ ∧ uses_only_0_and_7 a₈ ∧ uses_only_0_and_7 a₉ := by
  sorry

end sum_of_nine_zero_seven_digits_l2834_283496


namespace exponent_division_l2834_283425

theorem exponent_division (a : ℝ) (m n : ℕ) :
  a ^ m / a ^ n = a ^ (m - n) :=
sorry

end exponent_division_l2834_283425


namespace interval_contains_integer_l2834_283490

theorem interval_contains_integer (a : ℝ) : 
  (∃ n : ℤ, 3*a < n ∧ n < 5*a - 2) ↔ (a > 1.2 ∧ a < 4/3) ∨ a > 1.4 :=
sorry

end interval_contains_integer_l2834_283490


namespace angle_sum_from_tangents_l2834_283426

theorem angle_sum_from_tangents (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan α = 2 → 
  Real.tan β = 3 → 
  α + β = 3*π/4 := by
  sorry

end angle_sum_from_tangents_l2834_283426


namespace other_frisbee_price_is_3_l2834_283407

/-- Represents the price and sales of frisbees in a sporting goods store --/
structure FrisbeeSales where
  total_sold : ℕ
  total_receipts : ℕ
  price_other : ℚ
  min_sold_at_4 : ℕ

/-- Checks if the given FrisbeeSales satisfies the problem conditions --/
def is_valid_sale (sale : FrisbeeSales) : Prop :=
  sale.total_sold = 60 ∧
  sale.total_receipts = 204 ∧
  sale.min_sold_at_4 = 24 ∧
  sale.price_other * (sale.total_sold - sale.min_sold_at_4) + 4 * sale.min_sold_at_4 = sale.total_receipts

/-- Theorem stating that the price of the other frisbees is $3 --/
theorem other_frisbee_price_is_3 :
  ∀ (sale : FrisbeeSales), is_valid_sale sale → sale.price_other = 3 := by
  sorry

end other_frisbee_price_is_3_l2834_283407


namespace lemonade_stand_problem_l2834_283475

theorem lemonade_stand_problem (bea_price dawn_price : ℚ) (bea_glasses : ℕ) (earnings_difference : ℚ) :
  bea_price = 25 / 100 →
  dawn_price = 28 / 100 →
  bea_glasses = 10 →
  earnings_difference = 26 / 100 →
  ∃ dawn_glasses : ℕ,
    dawn_glasses = 8 ∧
    bea_price * bea_glasses = dawn_price * dawn_glasses + earnings_difference :=
by
  sorry

#check lemonade_stand_problem

end lemonade_stand_problem_l2834_283475


namespace Z_equals_S_l2834_283412

-- Define the set of functions F
def F : Set (ℝ → ℝ) := {f | ∀ x y, f (x + f y) = f x + f y}

-- Define the set of rational numbers q
def Z : Set ℚ := {q | ∀ f ∈ F, ∃ z : ℝ, f z = q * z}

-- Define the set S
def S : Set ℚ := {q | ∃ n : ℤ, n ≠ 0 ∧ q = (n + 1) / n}

-- State the theorem
theorem Z_equals_S : Z = S := by sorry

end Z_equals_S_l2834_283412


namespace derivative_value_l2834_283414

theorem derivative_value (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 2 + x^3) :
  f' 2 = -12 := by
  sorry

end derivative_value_l2834_283414


namespace harry_iguanas_l2834_283436

/-- The number of iguanas Harry owns -/
def num_iguanas : ℕ := 2

/-- The number of geckos Harry owns -/
def num_geckos : ℕ := 3

/-- The number of snakes Harry owns -/
def num_snakes : ℕ := 4

/-- The cost to feed each snake per month -/
def snake_feed_cost : ℕ := 10

/-- The cost to feed each iguana per month -/
def iguana_feed_cost : ℕ := 5

/-- The cost to feed each gecko per month -/
def gecko_feed_cost : ℕ := 15

/-- The total yearly cost to feed all pets -/
def yearly_feed_cost : ℕ := 1140

theorem harry_iguanas :
  num_iguanas * iguana_feed_cost * 12 +
  num_geckos * gecko_feed_cost * 12 +
  num_snakes * snake_feed_cost * 12 = yearly_feed_cost :=
by sorry

end harry_iguanas_l2834_283436


namespace math_test_score_l2834_283442

theorem math_test_score (korean_score english_score : ℕ)
  (h1 : (korean_score + english_score) / 2 = 92)
  (h2 : (korean_score + english_score + math_score) / 3 = 94)
  : math_score = 98 := by
  sorry

end math_test_score_l2834_283442


namespace a_range_l2834_283449

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + 1 = 0}

-- Define the set B
def B : Set ℝ := {1, 2}

-- Define the theorem
theorem a_range (a : ℝ) : (A a ⊆ B) ↔ a ∈ Set.Icc (-2) 2 ∧ a ≠ 2 := by
  sorry

end a_range_l2834_283449


namespace factor_calculation_l2834_283400

theorem factor_calculation (f : ℚ) : f * (2 * 16 + 5) = 111 ↔ f = 3 := by sorry

end factor_calculation_l2834_283400


namespace office_employees_l2834_283444

theorem office_employees (total_employees : ℕ) : 
  (total_employees : ℝ) * 0.25 * 0.6 = 120 → total_employees = 800 := by
  sorry

end office_employees_l2834_283444


namespace stamp_coverage_possible_l2834_283479

/-- Represents a square grid -/
structure Grid (n : ℕ) :=
  (cells : Fin n → Fin n → Bool)

/-- Represents a stamp with black cells -/
structure Stamp (n : ℕ) :=
  (cells : Fin n → Fin n → Bool)
  (black_count : ℕ)
  (black_count_eq : black_count = 102)

/-- Applies a stamp to a grid at a specific position -/
def apply_stamp (g : Grid n) (s : Stamp m) (pos_x pos_y : ℕ) : Grid n :=
  sorry

/-- Checks if a grid is fully covered except for one corner -/
def is_covered_except_corner (g : Grid n) : Prop :=
  sorry

/-- Main theorem: It's possible to cover a 101x101 grid except for one corner
    using a 102-cell stamp 100 times -/
theorem stamp_coverage_possible :
  ∃ (g : Grid 101) (s : Stamp 102) (stamps : List (ℕ × ℕ)),
    stamps.length = 100 ∧
    is_covered_except_corner (stamps.foldl (λ acc (x, y) => apply_stamp acc s x y) g) :=
  sorry

end stamp_coverage_possible_l2834_283479
