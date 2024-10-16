import Mathlib

namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_two_l2619_261976

theorem simplify_and_evaluate (a : ℝ) (h : a ≠ 1) :
  (1 - 2 / (a + 1)) / ((a^2 - 2*a + 1) / (a + 1)) = 1 / (a - 1) :=
sorry

theorem evaluate_at_two :
  (1 - 2 / (2 + 1)) / ((2^2 - 2*2 + 1) / (2 + 1)) = 1 :=
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_two_l2619_261976


namespace NUMINAMATH_CALUDE_cube_root_nested_expression_l2619_261949

theorem cube_root_nested_expression (x : ℝ) (h : x ≥ 0) :
  (x * (x * x^(1/3))^(1/2))^(1/3) = x^(5/9) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_nested_expression_l2619_261949


namespace NUMINAMATH_CALUDE_sum_of_squares_of_rates_l2619_261954

theorem sum_of_squares_of_rates : ∀ (c j s : ℕ),
  3 * c + 2 * j + 2 * s = 80 →
  3 * j + 2 * s + 4 * c = 104 →
  c^2 + j^2 + s^2 = 592 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_rates_l2619_261954


namespace NUMINAMATH_CALUDE_power_sum_difference_l2619_261992

theorem power_sum_difference : 2^6 + 2^6 + 2^6 + 2^6 - 4^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l2619_261992


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2619_261985

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 36*x + 320 ≤ 16} = Set.Icc 16 19 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2619_261985


namespace NUMINAMATH_CALUDE_unique_perfect_square_Q_l2619_261956

/-- The polynomial Q(x) = x^4 + 6x^3 + 13x^2 + 3x - 19 -/
def Q (x : ℤ) : ℤ := x^4 + 6*x^3 + 13*x^2 + 3*x - 19

/-- A function that checks if a given integer is a perfect square -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, m^2 = n

/-- Theorem stating that there is exactly one integer x for which Q(x) is a perfect square -/
theorem unique_perfect_square_Q : ∃! x : ℤ, is_perfect_square (Q x) :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_Q_l2619_261956


namespace NUMINAMATH_CALUDE_yellow_highlighters_count_l2619_261972

/-- The number of yellow highlighters in Kaya's teacher's desk -/
def yellow_highlighters (total pink blue : ℕ) : ℕ := total - pink - blue

/-- Theorem stating the number of yellow highlighters -/
theorem yellow_highlighters_count :
  yellow_highlighters 22 9 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_yellow_highlighters_count_l2619_261972


namespace NUMINAMATH_CALUDE_parabola_tangent_line_existence_l2619_261951

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the isosceles right triangle condition
def isosceles_right_triangle (F₁ F₂ F : ℝ × ℝ) : Prop :=
  (F₁.1 = -1 ∧ F₁.2 = 0) ∧ (F₂.1 = 1 ∧ F₂.2 = 0) ∧ (F.1 = 0 ∧ F.2 = 1)

-- Define the line passing through E(-2, 0)
def line_through_E (x y : ℝ) : Prop := y = (1/2) * (x + 2)

-- Define the perpendicular tangent lines condition
def perpendicular_tangents (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 = -4

-- Main theorem
theorem parabola_tangent_line_existence :
  ∃ (A B : ℝ × ℝ),
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    line_through_E A.1 A.2 ∧
    line_through_E B.1 B.2 ∧
    perpendicular_tangents A B :=
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_existence_l2619_261951


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l2619_261970

theorem factorization_of_cubic (x : ℝ) : 2 * x^3 - 4 * x^2 - 6 * x = 2 * x * (x - 3) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l2619_261970


namespace NUMINAMATH_CALUDE_quadratic_symmetry_implies_ordering_l2619_261971

-- Define the quadratic function f
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_symmetry_implies_ordering (b c : ℝ) :
  (∀ x : ℝ, f b c (1 + x) = f b c (1 - x)) →
  f b c 4 > f b c 2 ∧ f b c 2 > f b c 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_implies_ordering_l2619_261971


namespace NUMINAMATH_CALUDE_equal_distribution_l2619_261904

def total_amount : ℕ := 42900
def num_persons : ℕ := 22
def amount_per_person : ℕ := 1950

theorem equal_distribution :
  total_amount / num_persons = amount_per_person := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_l2619_261904


namespace NUMINAMATH_CALUDE_pentagon_area_sqrt_sum_m_n_l2619_261953

/-- A pentagon constructed from 11 line segments of length 2 --/
structure Pentagon where
  /-- The number of line segments --/
  num_segments : ℕ
  /-- The length of each segment --/
  segment_length : ℝ
  /-- Assertion that the pentagon is constructed from 11 segments of length 2 --/
  h_segments : num_segments = 11 ∧ segment_length = 2

/-- The area of the pentagon --/
noncomputable def area (p : Pentagon) : ℝ := sorry

/-- Theorem stating that the area of the pentagon can be expressed as √11 + √12 --/
theorem pentagon_area_sqrt (p : Pentagon) : 
  area p = Real.sqrt 11 + Real.sqrt 12 := by sorry

/-- Corollary showing that m + n = 23 --/
theorem sum_m_n (p : Pentagon) : 
  ∃ (m n : ℕ), (m > 0 ∧ n > 0) ∧ area p = Real.sqrt m + Real.sqrt n ∧ m + n = 23 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_sqrt_sum_m_n_l2619_261953


namespace NUMINAMATH_CALUDE_x_squared_eq_5_is_quadratic_l2619_261931

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = 5 -/
def f (x : ℝ) : ℝ := x^2 - 5

theorem x_squared_eq_5_is_quadratic :
  is_quadratic_one_var f :=
sorry

end NUMINAMATH_CALUDE_x_squared_eq_5_is_quadratic_l2619_261931


namespace NUMINAMATH_CALUDE_lcm_18_24_l2619_261910

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l2619_261910


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2619_261916

/-- A line in 2D space represented by the equation f(x,y) = 0 -/
structure Line2D where
  f : ℝ → ℝ → ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The equation f(x,y) - f(x₁,y₁) - f(x₂,y₂) = 0 represents a line parallel to 
    the original line and passing through P₂ -/
theorem parallel_line_through_point (l : Line2D) (P₁ P₂ : Point2D) 
  (h₁ : l.f P₁.x P₁.y = 0)  -- P₁ is on the line l
  (h₂ : l.f P₂.x P₂.y ≠ 0)  -- P₂ is not on the line l
  : ∃ (m : Line2D), 
    (∀ x y, m.f x y = l.f x y - l.f P₁.x P₁.y - l.f P₂.x P₂.y) ∧ 
    (m.f P₂.x P₂.y = 0) ∧
    (∃ k : ℝ, ∀ x y, m.f x y = k * l.f x y) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2619_261916


namespace NUMINAMATH_CALUDE_minimum_cactus_species_l2619_261901

theorem minimum_cactus_species (n : ℕ) (h1 : n = 80) : ∃ (k : ℕ),
  (∀ (S : Finset (Finset ℕ)), S.card = n → 
    (∀ i : ℕ, i ≤ k → ∃ s ∈ S, i ∉ s) ∧ 
    (∀ T : Finset (Finset ℕ), T ⊆ S → T.card = 15 → ∃ i : ℕ, i ≤ k ∧ ∀ s ∈ T, i ∈ s)) ∧
  k = 16 ∧ 
  (∀ m : ℕ, m < k → ¬∃ (S : Finset (Finset ℕ)), S.card = n ∧ 
    (∀ i : ℕ, i ≤ m → ∃ s ∈ S, i ∉ s) ∧
    (∀ T : Finset (Finset ℕ), T ⊆ S → T.card = 15 → ∃ i : ℕ, i ≤ m ∧ ∀ s ∈ T, i ∈ s)) :=
by sorry


end NUMINAMATH_CALUDE_minimum_cactus_species_l2619_261901


namespace NUMINAMATH_CALUDE_product_digit_sum_l2619_261946

/-- Represents a 101-digit number that repeats a 3-digit pattern -/
def RepeatingNumber (a b c : Nat) : Nat :=
  -- Implementation details omitted
  sorry

/-- Returns the units digit of a number -/
def unitsDigit (n : Nat) : Nat :=
  n % 10

/-- Returns the thousands digit of a number -/
def thousandsDigit (n : Nat) : Nat :=
  (n / 1000) % 10

/-- The main theorem -/
theorem product_digit_sum :
  let n1 := RepeatingNumber 6 0 6
  let n2 := RepeatingNumber 7 0 7
  let product := n1 * n2
  (thousandsDigit product) + (unitsDigit product) = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2619_261946


namespace NUMINAMATH_CALUDE_defective_units_shipped_percentage_l2619_261980

/-- Given that 6% of units produced are defective and 0.24% of total units
    are defective and shipped for sale, prove that 4% of defective units
    are shipped for sale. -/
theorem defective_units_shipped_percentage
  (total_defective_percent : ℝ)
  (defective_shipped_percent : ℝ)
  (h1 : total_defective_percent = 6)
  (h2 : defective_shipped_percent = 0.24) :
  defective_shipped_percent / total_defective_percent * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_percentage_l2619_261980


namespace NUMINAMATH_CALUDE_cos_80_cos_20_plus_sin_80_sin_20_l2619_261908

theorem cos_80_cos_20_plus_sin_80_sin_20 : Real.cos (80 * π / 180) * Real.cos (20 * π / 180) + Real.sin (80 * π / 180) * Real.sin (20 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_80_cos_20_plus_sin_80_sin_20_l2619_261908


namespace NUMINAMATH_CALUDE_wall_distance_equals_height_l2619_261957

-- Define the variables
variable (w : ℝ) -- Distance between walls
variable (a : ℝ) -- Length of the ladder
variable (k : ℝ) -- Distance from Q to ground
variable (h : ℝ) -- Distance from R to ground

-- Define the conditions
axiom ladder_length_45 : a = k * Real.sqrt 2
axiom ladder_length_75 : a = h * Real.sqrt (4 - 2 * Real.sqrt 3)
axiom angle_difference : 75 - 45 = 30

-- State the theorem
theorem wall_distance_equals_height : w = h := by
  sorry

end NUMINAMATH_CALUDE_wall_distance_equals_height_l2619_261957


namespace NUMINAMATH_CALUDE_pyarelal_loss_calculation_l2619_261991

/-- Calculates Pyarelal's share of the loss given the total loss and the ratio of investments -/
def pyarelal_loss (total_loss : ℚ) (ashok_ratio : ℚ) : ℚ :=
  let pyarelal_ratio := 1
  let total_ratio := pyarelal_ratio + ashok_ratio
  (pyarelal_ratio / total_ratio) * total_loss

/-- Proves that Pyarelal's share of the loss is 1440 given the conditions -/
theorem pyarelal_loss_calculation :
  let total_loss : ℚ := 1600
  let ashok_ratio : ℚ := 1 / 9
  pyarelal_loss total_loss ashok_ratio = 1440 := by
  sorry

#eval pyarelal_loss 1600 (1/9)

end NUMINAMATH_CALUDE_pyarelal_loss_calculation_l2619_261991


namespace NUMINAMATH_CALUDE_math_scores_properties_l2619_261902

def scores : List ℝ := [60, 60, 60, 65, 65, 70, 70, 70, 75, 75, 75, 75, 75, 80, 80, 80, 80, 85, 85, 90]

def group_a : List ℝ := [60, 60, 60, 65, 65, 70, 70, 70]

def mode (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem math_scores_properties :
  (mode scores = 75) ∧ (variance group_a = 75/4) := by sorry

end NUMINAMATH_CALUDE_math_scores_properties_l2619_261902


namespace NUMINAMATH_CALUDE_complex_cube_root_of_unity_sum_l2619_261960

theorem complex_cube_root_of_unity_sum (ω : ℂ) : 
  ω = (-1 + Complex.I * Real.sqrt 3) / 2 →
  ω^3 = 1 →
  ω^2 + ω + 1 = 0 →
  ((-1 + Complex.I * Real.sqrt 3) / 2)^4 + ((-1 - Complex.I * Real.sqrt 3) / 2)^4 = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_of_unity_sum_l2619_261960


namespace NUMINAMATH_CALUDE_crow_nest_ditch_distance_crow_problem_solution_l2619_261974

/-- The distance between a crow's nest and a ditch, given the crow's flying pattern and speed. -/
theorem crow_nest_ditch_distance (trips : ℕ) (time : ℝ) (speed : ℝ) : ℝ :=
  let distance_km := speed * time / (2 * trips)
  let distance_m := distance_km * 1000
  200

/-- Proof that the distance between the nest and the ditch is 200 meters. -/
theorem crow_problem_solution :
  crow_nest_ditch_distance 15 1.5 4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_crow_nest_ditch_distance_crow_problem_solution_l2619_261974


namespace NUMINAMATH_CALUDE_sin_sum_pi_minus_plus_alpha_l2619_261911

theorem sin_sum_pi_minus_plus_alpha (α : ℝ) : 
  Real.sin (π - α) + Real.sin (π + α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_pi_minus_plus_alpha_l2619_261911


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2619_261943

/-- Theorem: Weight of the new person in a group replacement scenario -/
theorem weight_of_new_person
  (n : ℕ) -- Number of people in the group
  (w : ℝ) -- Total weight of the original group
  (r : ℝ) -- Weight of the person being replaced
  (i : ℝ) -- Increase in average weight
  (h1 : n = 15) -- There are 15 people initially
  (h2 : r = 75) -- The replaced person weighs 75 kg
  (h3 : i = 3.2) -- The average weight increases by 3.2 kg
  (h4 : (w - r + (w / n + n * i)) / n = w / n + i) -- Equation for the new average weight
  : w / n + n * i - r = 123 := by
  sorry

#check weight_of_new_person

end NUMINAMATH_CALUDE_weight_of_new_person_l2619_261943


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l2619_261938

theorem binomial_expansion_terms (x a : ℝ) (n : ℕ) :
  (Nat.choose n 2 : ℝ) * x^(n - 2) * a^2 = 84 ∧
  (Nat.choose n 3 : ℝ) * x^(n - 3) * a^3 = 280 ∧
  (Nat.choose n 4 : ℝ) * x^(n - 4) * a^4 = 560 →
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l2619_261938


namespace NUMINAMATH_CALUDE_drug_price_reduction_l2619_261914

theorem drug_price_reduction (initial_price final_price : ℝ) (x : ℝ) :
  initial_price = 63800 →
  final_price = 3900 →
  final_price = initial_price * (1 - x)^2 :=
by sorry

end NUMINAMATH_CALUDE_drug_price_reduction_l2619_261914


namespace NUMINAMATH_CALUDE_gum_ratio_proof_l2619_261963

def gum_ratio (total_gum : ℕ) (shane_chewed : ℕ) (shane_left : ℕ) : ℚ :=
  let shane_total := shane_chewed + shane_left
  let rick_total := shane_total * 2
  rick_total / total_gum

theorem gum_ratio_proof :
  gum_ratio 100 11 14 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_gum_ratio_proof_l2619_261963


namespace NUMINAMATH_CALUDE_max_values_on_unit_circle_l2619_261998

theorem max_values_on_unit_circle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2 = 1) :
  (∀ x y, x > 0 → y > 0 → x^2 + y^2 = 1 → a + b ≥ x + y) ∧
  (a + b ≤ Real.sqrt 2) ∧
  (∀ x y, x > 0 → y > 0 → x^2 + y^2 = 1 → a * b ≥ x * y) ∧
  (a * b ≤ 1/2) := by
sorry


end NUMINAMATH_CALUDE_max_values_on_unit_circle_l2619_261998


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2619_261929

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  commonDiff : ℕ
  numTerms : ℕ
  sumOddTerms : ℕ
  sumEvenTerms : ℕ
  isEven : Even numTerms
  diffIs2 : commonDiff = 2

/-- Theorem stating the number of terms in the sequence with given conditions -/
theorem arithmetic_sequence_terms
  (seq : ArithmeticSequence)
  (h1 : seq.sumOddTerms = 15)
  (h2 : seq.sumEvenTerms = 35) :
  seq.numTerms = 20 := by
  sorry

#check arithmetic_sequence_terms

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2619_261929


namespace NUMINAMATH_CALUDE_michelle_initial_ride_fee_l2619_261940

/-- A taxi ride with an initial fee and per-mile charge. -/
structure TaxiRide where
  distance : ℝ
  chargePerMile : ℝ
  totalPaid : ℝ

/-- Calculate the initial ride fee for a taxi ride. -/
def initialRideFee (ride : TaxiRide) : ℝ :=
  ride.totalPaid - ride.distance * ride.chargePerMile

/-- Theorem: The initial ride fee for Michelle's taxi ride is $2. -/
theorem michelle_initial_ride_fee :
  let ride : TaxiRide := {
    distance := 4,
    chargePerMile := 2.5,
    totalPaid := 12
  }
  initialRideFee ride = 2 := by
  sorry

end NUMINAMATH_CALUDE_michelle_initial_ride_fee_l2619_261940


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l2619_261961

/-- The total path length of a vertex of an equilateral triangle rotating inside a square -/
theorem triangle_rotation_path_length 
  (triangle_side : ℝ) 
  (square_side : ℝ) 
  (h1 : triangle_side = 3) 
  (h2 : square_side = 6) 
  (h3 : triangle_side > 0) 
  (h4 : square_side > triangle_side) : 
  (4 : ℝ) * (π / 2) * triangle_side = 6 * π := by
sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l2619_261961


namespace NUMINAMATH_CALUDE_box_side_area_l2619_261969

theorem box_side_area (L W H : ℕ) : 
  L * H = (L * W) / 2 →  -- front area is half of top area
  L * W = (3/2) * (H * W) →  -- top area is 1.5 times side area
  3 * H = 2 * L →  -- length to height ratio is 3:2
  L * W * H = 3000 →  -- volume is 3000
  H * W = 200 :=  -- side area is 200
by sorry

end NUMINAMATH_CALUDE_box_side_area_l2619_261969


namespace NUMINAMATH_CALUDE_shelby_total_stars_l2619_261967

/-- The number of gold stars Shelby earned yesterday -/
def stars_yesterday : ℕ := 4

/-- The number of gold stars Shelby earned today -/
def stars_today : ℕ := 3

/-- The total number of gold stars Shelby earned -/
def total_stars : ℕ := stars_yesterday + stars_today

/-- Theorem stating that the total number of gold stars Shelby earned is 7 -/
theorem shelby_total_stars : total_stars = 7 := by sorry

end NUMINAMATH_CALUDE_shelby_total_stars_l2619_261967


namespace NUMINAMATH_CALUDE_count_integers_with_digit_sum_two_is_correct_l2619_261941

/-- The count of positive integers between 10^7 and 10^8 whose sum of digits is equal to 2 -/
def count_integers_with_digit_sum_two : ℕ :=
  let lower_bound := 10^7
  let upper_bound := 10^8 - 1
  let digit_sum := 2
  28  -- The actual count, to be proven

/-- Theorem stating that the count of positive integers between 10^7 and 10^8
    whose sum of digits is equal to 2 is correct -/
theorem count_integers_with_digit_sum_two_is_correct :
  count_integers_with_digit_sum_two = 
    (Finset.filter (fun n => (Finset.sum (Finset.range 8) (fun i => (n / 10^i) % 10) = 2))
      (Finset.range (10^8 - 10^7))).card :=
by
  sorry

#eval count_integers_with_digit_sum_two

end NUMINAMATH_CALUDE_count_integers_with_digit_sum_two_is_correct_l2619_261941


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l2619_261950

/-- The quadratic function y = x^2 - 4x + 3 is equivalent to y = (x-2)^2 - 1 -/
theorem quadratic_equivalence :
  ∀ x : ℝ, x^2 - 4*x + 3 = (x - 2)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l2619_261950


namespace NUMINAMATH_CALUDE_range_a_theorem_l2619_261923

/-- The range of a satisfying both conditions -/
def range_a : Set ℝ :=
  {a | (0 ≤ a ∧ a < 1) ∨ (3 < a ∧ a < 4)}

/-- Condition 1: For all x ∈ ℝ, ax^2 + ax + 1 > 0 -/
def condition1 (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Condition 2: The standard equation of the hyperbola is (x^2)/(1-a) + (y^2)/(a-3) = 1 -/
def condition2 (a : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (1 - a) + y^2 / (a - 3) = 1

/-- The main theorem stating that if both conditions are satisfied, then a is in the specified range -/
theorem range_a_theorem (a : ℝ) :
  condition1 a ∧ condition2 a → a ∈ range_a :=
sorry

end NUMINAMATH_CALUDE_range_a_theorem_l2619_261923


namespace NUMINAMATH_CALUDE_g_inequality_range_l2619_261934

noncomputable def g (x : ℝ) : ℝ := 2^x + 2^(-x) + |x|

theorem g_inequality_range : 
  {x : ℝ | g (2*x - 1) < g 3} = Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_g_inequality_range_l2619_261934


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l2619_261977

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_function_inequality
  (f : ℝ → ℝ) (h_dec : is_decreasing f) (m n : ℝ)
  (h_ineq : f m - f n > f (-m) - f (-n)) :
  m - n < 0 :=
sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l2619_261977


namespace NUMINAMATH_CALUDE_odd_function_implies_a_zero_l2619_261982

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = (x^2+1)(x+a) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ (x^2 + 1) * (x + a)

theorem odd_function_implies_a_zero (a : ℝ) : IsOdd (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_zero_l2619_261982


namespace NUMINAMATH_CALUDE_solutions_of_x_squared_equals_x_l2619_261966

theorem solutions_of_x_squared_equals_x : 
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_solutions_of_x_squared_equals_x_l2619_261966


namespace NUMINAMATH_CALUDE_parabola_focus_construction_l2619_261993

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : Line

def reflect_line (l : Line) (t : Line) : Line :=
  sorry

def intersection_point (l1 : Line) (l2 : Line) : Point :=
  sorry

def is_tangent (p : Parabola) (t : Line) : Prop :=
  sorry

theorem parabola_focus_construction 
  (p : Parabola) (t1 t2 : Line) 
  (h1 : is_tangent p t1) 
  (h2 : is_tangent p t2) :
  p.focus = intersection_point 
    (reflect_line p.directrix t1) 
    (reflect_line p.directrix t2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_construction_l2619_261993


namespace NUMINAMATH_CALUDE_solution_set_correct_l2619_261989

/-- The solution set of the equation 3sin(x) = 1 + cos(2x) -/
def SolutionSet : Set ℝ :=
  {x | ∃ k : ℤ, x = k * Real.pi + (-1)^k * (Real.pi / 6)}

/-- The original equation -/
def OriginalEquation (x : ℝ) : Prop :=
  3 * Real.sin x = 1 + Real.cos (2 * x)

theorem solution_set_correct :
  ∀ x : ℝ, x ∈ SolutionSet ↔ OriginalEquation x := by
  sorry

end NUMINAMATH_CALUDE_solution_set_correct_l2619_261989


namespace NUMINAMATH_CALUDE_ashton_pencils_l2619_261958

def pencil_problem (initial_boxes : Nat) (pencils_per_box : Nat) (given_to_brother : Nat) (distributed_to_friends : Nat) : Prop :=
  let initial_total := initial_boxes * pencils_per_box
  let after_giving_to_brother := initial_total - given_to_brother
  let remaining := after_giving_to_brother - distributed_to_friends
  remaining = 24

theorem ashton_pencils :
  pencil_problem 3 14 6 12 := by
  sorry

end NUMINAMATH_CALUDE_ashton_pencils_l2619_261958


namespace NUMINAMATH_CALUDE_power_simplification_l2619_261928

theorem power_simplification : 16^6 * 4^6 * 16^10 * 4^10 = 64^16 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l2619_261928


namespace NUMINAMATH_CALUDE_right_angled_isosceles_unique_indivisible_l2619_261932

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  a : ℝ  -- side length
  b : ℝ  -- base length
  ha : a > 0
  hb : b > 0

/-- A right-angled isosceles triangle -/
def RightAngledIsoscelesTriangle (t : IsoscelesTriangle) : Prop :=
  t.a = t.b * Real.sqrt 2 / 2

/-- Predicate for a triangle that can be divided into three isosceles triangles with equal side lengths -/
def CanBeDividedIntoThreeIsosceles (t : IsoscelesTriangle) : Prop :=
  ∃ (t1 t2 t3 : IsoscelesTriangle),
    t1.a = t2.a ∧ t2.a = t3.a ∧
    -- Additional conditions to ensure the three triangles form a partition of t
    sorry

/-- Theorem stating that only right-angled isosceles triangles cannot be divided into three isosceles triangles with equal side lengths -/
theorem right_angled_isosceles_unique_indivisible (t : IsoscelesTriangle) :
  ¬(CanBeDividedIntoThreeIsosceles t) ↔ RightAngledIsoscelesTriangle t :=
sorry

end NUMINAMATH_CALUDE_right_angled_isosceles_unique_indivisible_l2619_261932


namespace NUMINAMATH_CALUDE_three_tenths_plus_four_thousandths_l2619_261909

theorem three_tenths_plus_four_thousandths : 
  (3 : ℚ) / 10 + (4 : ℚ) / 1000 = (304 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_three_tenths_plus_four_thousandths_l2619_261909


namespace NUMINAMATH_CALUDE_sophia_ate_five_percent_l2619_261903

-- Define the percentages eaten by each person
def caden_percent : ℝ := 20
def zoe_percent : ℝ := caden_percent + 0.5 * caden_percent
def noah_percent : ℝ := zoe_percent + 0.5 * zoe_percent
def sophia_percent : ℝ := 100 - (caden_percent + zoe_percent + noah_percent)

-- Theorem statement
theorem sophia_ate_five_percent :
  sophia_percent = 5 := by
  sorry

end NUMINAMATH_CALUDE_sophia_ate_five_percent_l2619_261903


namespace NUMINAMATH_CALUDE_children_who_got_off_bus_l2619_261979

/-- Proves that 10 children got off the bus given the initial, final, and additional children counts -/
theorem children_who_got_off_bus 
  (initial_children : ℕ) 
  (children_who_got_on : ℕ) 
  (final_children : ℕ) 
  (h1 : initial_children = 21)
  (h2 : children_who_got_on = 5)
  (h3 : final_children = 16) :
  initial_children - final_children + children_who_got_on = 10 :=
by sorry

end NUMINAMATH_CALUDE_children_who_got_off_bus_l2619_261979


namespace NUMINAMATH_CALUDE_parabola_directrix_l2619_261978

/-- The directrix of a parabola with equation y = (1/4)x^2 -/
def directrix_of_parabola (x y : ℝ) : Prop :=
  y = (1/4) * x^2 → y = -1

theorem parabola_directrix : 
  ∀ x y : ℝ, directrix_of_parabola x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2619_261978


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a11_l2619_261988

theorem arithmetic_sequence_a11 (a : ℕ → ℚ) 
  (h_arith : ∀ n, (a (n+4) + 1)⁻¹ = ((a n + 1)⁻¹ + (a (n+8) + 1)⁻¹) / 2)
  (h_a3 : a 3 = 2)
  (h_a7 : a 7 = 1) :
  a 11 = 1/2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a11_l2619_261988


namespace NUMINAMATH_CALUDE_function_transformation_l2619_261945

open Real

theorem function_transformation (x : ℝ) :
  let f (x : ℝ) := sin (2 * x + π / 3)
  let g (x : ℝ) := 2 * f (x - π / 6)
  g x = 2 * sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l2619_261945


namespace NUMINAMATH_CALUDE_price_adjustment_solution_l2619_261905

/-- Selling prices before and after adjustment in places A and B -/
structure Prices where
  a_before : ℝ
  b_before : ℝ
  a_after : ℝ
  b_after : ℝ

/-- Conditions of the price adjustment problem -/
def PriceAdjustmentConditions (p : Prices) : Prop :=
  p.a_after = p.a_before * 1.1 ∧
  p.b_after = p.b_before - 5 ∧
  p.b_before - p.a_before = 10 ∧
  p.b_after - p.a_after = 1

/-- Theorem stating the solution to the price adjustment problem -/
theorem price_adjustment_solution :
  ∃ (p : Prices), PriceAdjustmentConditions p ∧ p.a_before = 40 ∧ p.b_before = 50 := by
  sorry

end NUMINAMATH_CALUDE_price_adjustment_solution_l2619_261905


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2619_261997

/-- The sum of the geometric series 15 + 15r + 15r^2 + 15r^3 + ... for -1 < r < 1 -/
noncomputable def S (r : ℝ) : ℝ := 15 / (1 - r)

/-- For -1 < a < 1, if S(a)S(-a) = 2025, then S(a) + S(-a) = 270 -/
theorem geometric_series_sum (a : ℝ) (h1 : -1 < a) (h2 : a < 1) 
  (h3 : S a * S (-a) = 2025) : S a + S (-a) = 270 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2619_261997


namespace NUMINAMATH_CALUDE_cookie_sales_difference_l2619_261981

/-- The number of cookie boxes sold by Kim -/
def kim_boxes : ℕ := 54

/-- The number of cookie boxes sold by Jennifer -/
def jennifer_boxes : ℕ := 71

/-- Theorem stating the difference in cookie sales between Jennifer and Kim -/
theorem cookie_sales_difference :
  jennifer_boxes > kim_boxes ∧
  jennifer_boxes - kim_boxes = 17 :=
sorry

end NUMINAMATH_CALUDE_cookie_sales_difference_l2619_261981


namespace NUMINAMATH_CALUDE_intersection_M_N_l2619_261926

-- Define the universal set U
def U : Set Int := {-1, 0, 1, 2, 3, 4}

-- Define the complement of M with respect to U
def M_complement : Set Int := {-1, 1}

-- Define set N
def N : Set Int := {0, 1, 2, 3}

-- Define set M based on its complement
def M : Set Int := U \ M_complement

-- Theorem statement
theorem intersection_M_N : M ∩ N = {0, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2619_261926


namespace NUMINAMATH_CALUDE_exactly_three_valid_sets_l2619_261937

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)

/-- The sum of a ConsecutiveSet -/
def sum_consecutive_set (s : ConsecutiveSet) : ℕ :=
  (s.length * (2 * s.start + s.length - 1)) / 2

/-- Predicate for a valid set according to our conditions -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 3 ∧ sum_consecutive_set s = 30

/-- The theorem to prove -/
theorem exactly_three_valid_sets :
  ∃! (sets : Finset ConsecutiveSet), 
    sets.card = 3 ∧ 
    (∀ s ∈ sets, is_valid_set s) ∧
    (∀ s, is_valid_set s → s ∈ sets) :=
  sorry

end NUMINAMATH_CALUDE_exactly_three_valid_sets_l2619_261937


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2619_261973

theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := c / a
  (a^2 * b^2 = (a^2 - c^2) * b^2) →  -- Ellipse equation
  (c^2 + b^2 = a^2) →               -- Right triangle condition
  e = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2619_261973


namespace NUMINAMATH_CALUDE_primality_extension_l2619_261922

theorem primality_extension (n : ℕ) (h1 : n ≥ 2) :
  (∀ k : ℕ, k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, k ≤ n - 2 → Nat.Prime (k^2 + k + n)) :=
by sorry

end NUMINAMATH_CALUDE_primality_extension_l2619_261922


namespace NUMINAMATH_CALUDE_set_intersection_range_l2619_261995

theorem set_intersection_range (m : ℝ) : 
  let A : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  let B : Set ℝ := {x | x^2 - 7*x + 10 ≤ 0}
  A ∩ B = A → 2 ≤ m ∧ m ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_set_intersection_range_l2619_261995


namespace NUMINAMATH_CALUDE_polynomial_identity_l2619_261935

theorem polynomial_identity (p : ℝ → ℝ) : 
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) →
  p 3 = 10 →
  p = fun x => x^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2619_261935


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2619_261944

theorem arithmetic_calculation : 1375 + 150 / 50 * 3 - 275 = 1109 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2619_261944


namespace NUMINAMATH_CALUDE_max_length_sum_l2619_261947

/-- Length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer -/
def length (n : ℕ) : ℕ := sorry

theorem max_length_sum :
  ∃ (x y : ℕ),
    x > 1 ∧
    y > 1 ∧
    x + 3 * y < 5000 ∧
    length x + length y = 20 ∧
    ∀ (a b : ℕ),
      a > 1 →
      b > 1 →
      a + 3 * b < 5000 →
      length a + length b ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_max_length_sum_l2619_261947


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2619_261984

/-- Given that y^4 varies inversely with ⁴√z, prove that z = 1/4096 when y = 6, given that y = 3 when z = 16 -/
theorem inverse_variation_problem (y z : ℝ) (h1 : ∃ k : ℝ, ∀ y z, y^4 * z^(1/4) = k) 
  (h2 : 3^4 * 16^(1/4) = 6^4 * z^(1/4)) : 
  y = 6 → z = 1/4096 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2619_261984


namespace NUMINAMATH_CALUDE_triangle_side_length_l2619_261936

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  B = 2 * A ∧
  a = 1 ∧ 
  b = Real.sqrt 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C ∧
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2619_261936


namespace NUMINAMATH_CALUDE_tan_sin_inequality_l2619_261959

open Real

theorem tan_sin_inequality (x : ℝ) (h : 0 < x ∧ x < π/2) :
  (tan x - x) / (x - sin x) > 2 ∧
  ∃ (a : ℝ), a = 3 ∧ ∀ b > a, ∃ y ∈ Set.Ioo 0 (π/2), tan y + 2 * sin y - b * y ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_tan_sin_inequality_l2619_261959


namespace NUMINAMATH_CALUDE_tunnel_length_l2619_261930

theorem tunnel_length : 
  ∀ (initial_speed : ℝ),
  initial_speed > 0 →
  (400 + 8600) / initial_speed = 10 →
  (400 + 8600) / (initial_speed + 0.1 * 60) = 9 :=
by sorry

end NUMINAMATH_CALUDE_tunnel_length_l2619_261930


namespace NUMINAMATH_CALUDE_equation_two_distinct_roots_l2619_261975

theorem equation_two_distinct_roots (a : ℝ) : 
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    ((Real.sqrt (6 * x₁ - x₁^2 - 4) + a - 2) * ((a - 2) * x₁ - 3 * a + 4) = 0) ∧
    ((Real.sqrt (6 * x₂ - x₂^2 - 4) + a - 2) * ((a - 2) * x₂ - 3 * a + 4) = 0)) ↔ 
  (a = 2 - Real.sqrt 5 ∨ a = 0 ∨ a = 1 ∨ (2 - 2 / Real.sqrt 5 < a ∧ a ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_equation_two_distinct_roots_l2619_261975


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l2619_261906

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 ∧ q > 0 ∧ p + 2*q = 1 ∧ 
  (45 : ℝ) * p^8 * q^2 = (120 : ℝ) * p^7 * q^3 → 
  p = 4/7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l2619_261906


namespace NUMINAMATH_CALUDE_endpoint_of_vector_l2619_261918

def vector_a : Fin 3 → ℝ := ![3, -4, 2]
def point_A : Fin 3 → ℝ := ![2, -1, 1]
def point_B : Fin 3 → ℝ := ![5, -5, 3]

theorem endpoint_of_vector (i : Fin 3) : 
  point_B i = point_A i + vector_a i :=
by
  sorry

end NUMINAMATH_CALUDE_endpoint_of_vector_l2619_261918


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2619_261917

theorem quadratic_inequality_equivalence :
  ∃ d : ℝ, ∀ x : ℝ, x * (2 * x + 4) < d ↔ x ∈ Set.Ioo (-4) 1 :=
by
  use 8
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2619_261917


namespace NUMINAMATH_CALUDE_exists_specific_figure_l2619_261994

/-- A rhombus in the mosaic --/
structure Rhombus where
  is_wide : Bool

/-- A figure in the mosaic --/
structure Figure where
  rhombuses : List Rhombus
  is_contiguous : Bool

/-- Count the number of wide rhombuses in a figure --/
def count_wide (f : Figure) : Nat :=
  f.rhombuses.filter (λ r => r.is_wide) |>.length

/-- Count the number of narrow rhombuses in a figure --/
def count_narrow (f : Figure) : Nat :=
  f.rhombuses.filter (λ r => ¬r.is_wide) |>.length

/-- Theorem stating the existence of a figure satisfying the required conditions --/
theorem exists_specific_figure :
  ∃ (f : Figure), count_wide f = 3 ∧ count_narrow f = 8 ∧ f.is_contiguous := by
  sorry

end NUMINAMATH_CALUDE_exists_specific_figure_l2619_261994


namespace NUMINAMATH_CALUDE_probability_two_pairs_one_odd_l2619_261996

def total_socks : ℕ := 12
def socks_per_color : ℕ := 3
def num_colors : ℕ := 4
def drawn_socks : ℕ := 5

def favorable_outcomes : ℕ := 324
def total_outcomes : ℕ := Nat.choose total_socks drawn_socks

theorem probability_two_pairs_one_odd (h : total_outcomes = 792) :
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_pairs_one_odd_l2619_261996


namespace NUMINAMATH_CALUDE_f_lower_bound_l2619_261986

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.log x

-- State the theorem
theorem f_lower_bound (a : ℝ) (h_a : a > 0) :
  ∀ x > 0, f a x ≥ a * (2 - Real.log a) := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_l2619_261986


namespace NUMINAMATH_CALUDE_household_coffee_expense_l2619_261927

def weekly_coffee_expense (person_a_cups : ℕ) (person_a_ounces : ℝ)
                          (person_b_cups : ℕ) (person_b_ounces : ℝ)
                          (person_c_cups : ℕ) (person_c_ounces : ℝ)
                          (person_c_days : ℕ) (coffee_cost : ℝ) : ℝ :=
  let person_a_weekly := person_a_cups * person_a_ounces * 7
  let person_b_weekly := person_b_cups * person_b_ounces * 7
  let person_c_weekly := person_c_cups * person_c_ounces * person_c_days
  let total_weekly_ounces := person_a_weekly + person_b_weekly + person_c_weekly
  total_weekly_ounces * coffee_cost

theorem household_coffee_expense :
  weekly_coffee_expense 3 0.4 1 0.6 2 0.5 5 1.25 = 22 := by
  sorry

end NUMINAMATH_CALUDE_household_coffee_expense_l2619_261927


namespace NUMINAMATH_CALUDE_circle_radius_implies_m_value_l2619_261920

/-- The equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The given equation -/
def given_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y + m = 0

theorem circle_radius_implies_m_value :
  ∀ m : ℝ, 
  (∃ h k : ℝ, ∀ x y : ℝ, given_equation x y m ↔ circle_equation x y h k 3) →
  m = -7 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_implies_m_value_l2619_261920


namespace NUMINAMATH_CALUDE_impossible_score_53_l2619_261948

/-- Represents the score of a quiz -/
structure QuizScore where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  total_questions : ℕ
  score : ℤ

/-- Calculates the score based on the number of correct, incorrect, and unanswered questions -/
def calculate_score (c i u : ℕ) : ℤ :=
  (4 : ℤ) * c - i

/-- Checks if a QuizScore is valid according to the quiz rules -/
def is_valid_score (qs : QuizScore) : Prop :=
  qs.correct + qs.incorrect + qs.unanswered = qs.total_questions ∧
  qs.score = calculate_score qs.correct qs.incorrect qs.unanswered

/-- Theorem: It's impossible to achieve a score of 53 in the given quiz -/
theorem impossible_score_53 :
  ¬ ∃ (qs : QuizScore), qs.total_questions = 15 ∧ is_valid_score qs ∧ qs.score = 53 :=
by sorry

end NUMINAMATH_CALUDE_impossible_score_53_l2619_261948


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l2619_261942

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 2*a| + |x + 3| < 5)) ↔ (a ≤ -4 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l2619_261942


namespace NUMINAMATH_CALUDE_points_symmetric_about_x_axis_l2619_261987

/-- Two points are symmetric about the x-axis if they have the same x-coordinate
    and their y-coordinates are negatives of each other. -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

/-- Given points P₁(-4, 3) and P₂(-4, -3), prove they are symmetric about the x-axis. -/
theorem points_symmetric_about_x_axis :
  let p1 : ℝ × ℝ := (-4, 3)
  let p2 : ℝ × ℝ := (-4, -3)
  symmetric_about_x_axis p1 p2 := by
  sorry


end NUMINAMATH_CALUDE_points_symmetric_about_x_axis_l2619_261987


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l2619_261913

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + 4*a else Real.log x / Real.log a

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/6) (1/2) ∧ a ≠ 1/2 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l2619_261913


namespace NUMINAMATH_CALUDE_paula_candies_l2619_261919

theorem paula_candies (x : ℕ) : 
  (x + 4 = 6 * 4) → x = 20 := by
sorry

end NUMINAMATH_CALUDE_paula_candies_l2619_261919


namespace NUMINAMATH_CALUDE_total_subjects_l2619_261962

theorem total_subjects (average_all : ℝ) (average_five : ℝ) (last_subject : ℝ) 
  (h1 : average_all = 77)
  (h2 : average_five = 74)
  (h3 : last_subject = 92) :
  ∃ n : ℕ, n = 6 ∧ 
    n * average_all = (n - 1) * average_five + last_subject :=
by sorry

end NUMINAMATH_CALUDE_total_subjects_l2619_261962


namespace NUMINAMATH_CALUDE_remainder_98_pow_50_mod_100_l2619_261907

theorem remainder_98_pow_50_mod_100 : 98^50 % 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_pow_50_mod_100_l2619_261907


namespace NUMINAMATH_CALUDE_quadratic_has_two_distinct_real_roots_l2619_261955

theorem quadratic_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 - 5*x₁ - 1 = 0) ∧ (x₂^2 - 5*x₂ - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_distinct_real_roots_l2619_261955


namespace NUMINAMATH_CALUDE_two_number_cards_totaling_twelve_probability_l2619_261939

/-- Represents a standard deck of cards -/
def StandardDeck : Type := Unit

/-- Number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- Set of card values that are numbers (2 through 10) -/
def numberCards : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10}

/-- Number of cards of each value in the deck -/
def cardsPerValue : ℕ := 4

/-- Predicate for two cards totaling 12 -/
def totalTwelve (card1 card2 : ℕ) : Prop := card1 + card2 = 12

/-- The probability of the event -/
def probabilityTwoNumberCardsTotalingTwelve (deck : StandardDeck) : ℚ :=
  35 / 663

theorem two_number_cards_totaling_twelve_probability 
  (deck : StandardDeck) : 
  probabilityTwoNumberCardsTotalingTwelve deck = 35 / 663 := by
  sorry

end NUMINAMATH_CALUDE_two_number_cards_totaling_twelve_probability_l2619_261939


namespace NUMINAMATH_CALUDE_expression_evaluation_l2619_261921

theorem expression_evaluation (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y) / (1 / x) + 2 = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2619_261921


namespace NUMINAMATH_CALUDE_sheep_wool_production_l2619_261912

/-- Calculates the amount of wool produced per sheep given the total number of sheep,
    payment to the shearer, price per pound of wool, and total profit. -/
def wool_per_sheep (num_sheep : ℕ) (shearer_payment : ℕ) (price_per_pound : ℕ) (profit : ℕ) : ℕ :=
  ((profit + shearer_payment) / price_per_pound) / num_sheep

/-- Proves that given 200 sheep, $2000 paid to shearer, $20 per pound of wool,
    and $38000 profit, each sheep produces 10 pounds of wool. -/
theorem sheep_wool_production :
  wool_per_sheep 200 2000 20 38000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sheep_wool_production_l2619_261912


namespace NUMINAMATH_CALUDE_row_time_ratio_l2619_261964

/-- Proves the ratio of time taken to row up and down a river -/
theorem row_time_ratio (man_speed : ℝ) (stream_speed : ℝ)
  (h1 : man_speed = 24)
  (h2 : stream_speed = 12) :
  (man_speed - stream_speed) / (man_speed + stream_speed) = 1 / 3 := by
  sorry

#check row_time_ratio

end NUMINAMATH_CALUDE_row_time_ratio_l2619_261964


namespace NUMINAMATH_CALUDE_vector_problem_l2619_261924

/-- Given vectors AB and BC in R², prove that -1/2 * AC equals the specified vector. -/
theorem vector_problem (AB BC : ℝ × ℝ) (h1 : AB = (3, 7)) (h2 : BC = (-2, 3)) :
  let AC := (AB.1 + BC.1, AB.2 + BC.2)
  (-1/2 : ℝ) • AC = (-1/2, -5) := by sorry

end NUMINAMATH_CALUDE_vector_problem_l2619_261924


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l2619_261900

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  let d_on_ab := ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ t.D = k • t.A + (1 - k) • t.B
  let ad_bisects_bac := ∃ k : ℝ, k > 0 ∧ k • (t.C - t.A) = t.D - t.A
  let bd_length := dist t.B t.D = 36
  let bc_length := dist t.B t.C = 45
  let ac_length := dist t.A t.C = 40
  d_on_ab ∧ ad_bisects_bac ∧ bd_length ∧ bc_length ∧ ac_length

-- State the theorem
theorem angle_bisector_theorem (t : Triangle) :
  satisfies_conditions t → dist t.A t.D = 68 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_l2619_261900


namespace NUMINAMATH_CALUDE_sum_and_difference_problem_l2619_261990

theorem sum_and_difference_problem (a b : ℤ) : 
  a + b = 84 ∧ a = b + 12 ∧ a = 36 → b = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_difference_problem_l2619_261990


namespace NUMINAMATH_CALUDE_tracy_book_collection_l2619_261965

theorem tracy_book_collection (x : ℕ) (h : x + 10 * x = 99) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_tracy_book_collection_l2619_261965


namespace NUMINAMATH_CALUDE_play_area_size_l2619_261999

/-- Represents the configuration of a rectangular play area with fence posts -/
structure PlayArea where
  total_posts : ℕ
  post_spacing : ℕ
  shorter_side_posts : ℕ
  longer_side_posts : ℕ

/-- Calculates the area of the play area given its configuration -/
def calculate_area (pa : PlayArea) : ℕ :=
  (pa.shorter_side_posts - 1) * pa.post_spacing * ((pa.longer_side_posts - 1) * pa.post_spacing)

/-- Theorem stating that the play area with given specifications has an area of 324 square yards -/
theorem play_area_size (pa : PlayArea) :
  pa.total_posts = 24 ∧
  pa.post_spacing = 3 ∧
  pa.longer_side_posts = 2 * pa.shorter_side_posts ∧
  pa.total_posts = 2 * pa.shorter_side_posts + 2 * pa.longer_side_posts - 4 →
  calculate_area pa = 324 := by
  sorry

end NUMINAMATH_CALUDE_play_area_size_l2619_261999


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2619_261915

-- 1. 0.175÷0.25÷4 = 0.175
theorem problem_1 : (0.175 / 0.25) / 4 = 0.175 := by sorry

-- 2. 1.4×99+1.4 = 140
theorem problem_2 : 1.4 * 99 + 1.4 = 140 := by sorry

-- 3. 3.6÷4-1.2×6 = -6.3
theorem problem_3 : 3.6 / 4 - 1.2 * 6 = -6.3 := by sorry

-- 4. (3.2+0.16)÷0.8 = 4.2
theorem problem_4 : (3.2 + 0.16) / 0.8 = 4.2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2619_261915


namespace NUMINAMATH_CALUDE_evelyn_skittles_l2619_261968

def skittles_problem (starting_skittles shared_skittles : ℕ) : ℕ :=
  starting_skittles - shared_skittles

theorem evelyn_skittles :
  skittles_problem 76 72 = 4 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_skittles_l2619_261968


namespace NUMINAMATH_CALUDE_value_of_c_l2619_261933

theorem value_of_c (a b c : ℝ) : 
  8 = (4 / 100) * a →
  4 = (8 / 100) * b →
  c = b / a →
  c = 0.25 := by
sorry

end NUMINAMATH_CALUDE_value_of_c_l2619_261933


namespace NUMINAMATH_CALUDE_solve_books_problem_l2619_261952

def books_problem (initial_books : ℕ) (new_books : ℕ) : Prop :=
  let after_nephew := initial_books - (initial_books / 4)
  let after_library := after_nephew - (after_nephew / 5)
  let after_neighbor := after_library - (after_library / 6)
  let final_books := after_neighbor + new_books
  final_books = 68

theorem solve_books_problem :
  books_problem 120 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_books_problem_l2619_261952


namespace NUMINAMATH_CALUDE_not_in_set_A_l2619_261925

-- Define the set A
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 - 5}

-- Theorem statement
theorem not_in_set_A :
  (1, -5) ∉ A ∧ (2, 1) ∈ A ∧ (3, 4) ∈ A ∧ (4, 7) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_not_in_set_A_l2619_261925


namespace NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l2619_261983

theorem twenty_is_eighty_percent_of_twentyfive : 
  ∃ x : ℝ, (20 : ℝ) / x = (80 : ℝ) / 100 ∧ x = 25 := by sorry

end NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l2619_261983
