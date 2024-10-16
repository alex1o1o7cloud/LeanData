import Mathlib

namespace NUMINAMATH_CALUDE_six_legged_creatures_count_l2041_204169

/-- Represents the number of creatures with 6 legs -/
def creatures_with_6_legs : ℕ := sorry

/-- Represents the number of creatures with 10 legs -/
def creatures_with_10_legs : ℕ := sorry

/-- The total number of creatures -/
def total_creatures : ℕ := 20

/-- The total number of legs -/
def total_legs : ℕ := 156

/-- Theorem stating that the number of creatures with 6 legs is 11 -/
theorem six_legged_creatures_count : 
  creatures_with_6_legs = 11 ∧ 
  creatures_with_6_legs + creatures_with_10_legs = total_creatures ∧
  6 * creatures_with_6_legs + 10 * creatures_with_10_legs = total_legs := by
  sorry

end NUMINAMATH_CALUDE_six_legged_creatures_count_l2041_204169


namespace NUMINAMATH_CALUDE_ln_inequality_l2041_204131

theorem ln_inequality (x y a b : ℝ) 
  (hx : 0 < x) (hy : x < y) (hy1 : y < 1)
  (hb : 1 < b) (ha : b < a) : 
  (Real.log x) / b < (Real.log y) / a :=
sorry

end NUMINAMATH_CALUDE_ln_inequality_l2041_204131


namespace NUMINAMATH_CALUDE_sin_translation_problem_l2041_204164

open Real

theorem sin_translation_problem (φ : ℝ) : 
  (0 < φ) → (φ < π / 2) →
  (∃ x₁ x₂ : ℝ, |sin (2 * x₁) - sin (2 * x₂ - 2 * φ)| = 2) →
  (∀ x₁ x₂ : ℝ, |sin (2 * x₁) - sin (2 * x₂ - 2 * φ)| = 2 → |x₁ - x₂| ≥ π / 3) →
  (∃ x₁ x₂ : ℝ, |sin (2 * x₁) - sin (2 * x₂ - 2 * φ)| = 2 ∧ |x₁ - x₂| = π / 3) →
  φ = π / 6 := by
sorry

end NUMINAMATH_CALUDE_sin_translation_problem_l2041_204164


namespace NUMINAMATH_CALUDE_multiples_of_three_never_reach_one_l2041_204135

def operation (n : ℕ) : ℕ :=
  (n + 3 * (5 - n % 5) % 5) / 5

theorem multiples_of_three_never_reach_one (k : ℕ) :
  ∀ n : ℕ, (∃ m : ℕ, n = operation^[m] (3 * k)) → n ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_multiples_of_three_never_reach_one_l2041_204135


namespace NUMINAMATH_CALUDE_perpendicular_line_parallel_planes_l2041_204152

structure Plane where
  -- Define a plane

structure Line where
  -- Define a line

def perpendicular (l : Line) (p : Plane) : Prop :=
  -- Define what it means for a line to be perpendicular to a plane
  sorry

def parallel (p1 p2 : Plane) : Prop :=
  -- Define what it means for two planes to be parallel
  sorry

def contains (p : Plane) (l : Line) : Prop :=
  -- Define what it means for a plane to contain a line
  sorry

def perpendicular_lines (l1 l2 : Line) : Prop :=
  -- Define what it means for two lines to be perpendicular
  sorry

theorem perpendicular_line_parallel_planes 
  (m : Line) (n : Line) (α β : Plane) :
  perpendicular m α → contains β n → parallel α β → perpendicular_lines m n :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_parallel_planes_l2041_204152


namespace NUMINAMATH_CALUDE_adam_figurines_count_l2041_204196

/-- Adam's wood carving shop problem -/
theorem adam_figurines_count :
  -- Define the number of figurines per block for each wood type
  let basswood_figurines : ℕ := 3
  let butternut_figurines : ℕ := 4
  let aspen_figurines : ℕ := 2 * basswood_figurines
  let oak_figurines : ℕ := 5
  let cherry_figurines : ℕ := 7

  -- Define the number of blocks for each wood type
  let basswood_blocks : ℕ := 25
  let butternut_blocks : ℕ := 30
  let aspen_blocks : ℕ := 35
  let oak_blocks : ℕ := 40
  let cherry_blocks : ℕ := 45

  -- Calculate total figurines
  let total_figurines : ℕ := 
    basswood_blocks * basswood_figurines +
    butternut_blocks * butternut_figurines +
    aspen_blocks * aspen_figurines +
    oak_blocks * oak_figurines +
    cherry_blocks * cherry_figurines

  -- Prove that the total number of figurines is 920
  total_figurines = 920 := by
  sorry


end NUMINAMATH_CALUDE_adam_figurines_count_l2041_204196


namespace NUMINAMATH_CALUDE_triangle_relation_l2041_204163

-- Define the triangles and their properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- State the theorem
theorem triangle_relation (abc : Triangle) (a'b'c' : Triangle) 
  (h1 : abc.angleB = a'b'c'.angleB) 
  (h2 : abc.angleA + a'b'c'.angleA = π) : 
  abc.a * a'b'c'.a = abc.b * a'b'c'.b + abc.c * a'b'c'.c := by
  sorry


end NUMINAMATH_CALUDE_triangle_relation_l2041_204163


namespace NUMINAMATH_CALUDE_time_to_see_again_is_120_l2041_204189

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a person walking -/
structure Walker where
  position : Point
  speed : ℝ

/-- The scenario of Jenny and Kenny walking -/
structure WalkingScenario where
  jenny : Walker
  kenny : Walker
  buildingRadius : ℝ
  pathDistance : ℝ

/-- The time when Jenny and Kenny can see each other again -/
def timeToSeeAgain (scenario : WalkingScenario) : ℝ :=
  sorry

/-- The theorem stating that the time to see again is 120 seconds -/
theorem time_to_see_again_is_120 (scenario : WalkingScenario) :
  scenario.jenny.speed = 2 →
  scenario.kenny.speed = 4 →
  scenario.pathDistance = 300 →
  scenario.buildingRadius = 75 →
  scenario.jenny.position = Point.mk (-75) (-150) →
  scenario.kenny.position = Point.mk (-75) 150 →
  timeToSeeAgain scenario = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_time_to_see_again_is_120_l2041_204189


namespace NUMINAMATH_CALUDE_sculpture_cost_equivalence_l2041_204171

/-- Represents the exchange rate between US dollars and Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Represents the exchange rate between US dollars and Chinese yuan -/
def usd_to_cny : ℚ := 5

/-- Represents the cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 160

/-- Theorem stating the equivalence of the sculpture's cost in Chinese yuan -/
theorem sculpture_cost_equivalence :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 100 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_equivalence_l2041_204171


namespace NUMINAMATH_CALUDE_card_distribution_l2041_204127

theorem card_distribution (total : ℕ) (black red : ℕ) (spades diamonds hearts clubs : ℕ) : 
  total = 13 →
  black = 7 →
  red = 6 →
  diamonds = 2 * spades →
  hearts = 2 * diamonds →
  total = spades + diamonds + hearts + clubs →
  black = spades + clubs →
  red = diamonds + hearts →
  clubs = 6 := by
sorry

end NUMINAMATH_CALUDE_card_distribution_l2041_204127


namespace NUMINAMATH_CALUDE_second_storm_duration_l2041_204108

theorem second_storm_duration 
  (storm1_rate : ℝ) 
  (storm2_rate : ℝ) 
  (total_time : ℝ) 
  (total_rainfall : ℝ) 
  (h1 : storm1_rate = 30) 
  (h2 : storm2_rate = 15) 
  (h3 : total_time = 45) 
  (h4 : total_rainfall = 975) :
  ∃ (storm1_duration storm2_duration : ℝ),
    storm1_duration + storm2_duration = total_time ∧
    storm1_rate * storm1_duration + storm2_rate * storm2_duration = total_rainfall ∧
    storm2_duration = 25 := by
  sorry

#check second_storm_duration

end NUMINAMATH_CALUDE_second_storm_duration_l2041_204108


namespace NUMINAMATH_CALUDE_graduation_ceremony_chairs_l2041_204168

/-- The number of graduates at a ceremony -/
def graduates : ℕ := 50

/-- The number of parents per graduate -/
def parents_per_graduate : ℕ := 2

/-- The number of teachers attending -/
def teachers : ℕ := 20

/-- The number of administrators attending -/
def administrators : ℕ := teachers / 2

/-- The total number of chairs available -/
def total_chairs : ℕ := 180

theorem graduation_ceremony_chairs :
  graduates + graduates * parents_per_graduate + teachers + administrators = total_chairs :=
sorry

end NUMINAMATH_CALUDE_graduation_ceremony_chairs_l2041_204168


namespace NUMINAMATH_CALUDE_gcf_of_7_factorial_and_8_factorial_l2041_204133

theorem gcf_of_7_factorial_and_8_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 :=
by sorry

end NUMINAMATH_CALUDE_gcf_of_7_factorial_and_8_factorial_l2041_204133


namespace NUMINAMATH_CALUDE_solve_for_a_l2041_204167

theorem solve_for_a : ∃ a : ℝ, (3 * 2 + 2 * a = 0) ∧ (a = -3) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2041_204167


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2041_204134

theorem arithmetic_sequence_sum (d : ℤ) : ∃ (S : ℕ → ℤ) (a : ℕ → ℤ), 
  (∀ n, a (n + 1) = a n + d) ∧  -- Arithmetic sequence definition
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) ∧  -- Sum formula
  a 1 = 190 ∧  -- First term
  S 20 > 0 ∧  -- S₂₀ > 0
  S 24 < 0 ∧  -- S₂₄ < 0
  d = -17  -- One possible value for d
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2041_204134


namespace NUMINAMATH_CALUDE_permutations_of_four_objects_l2041_204160

theorem permutations_of_four_objects : Nat.factorial 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_four_objects_l2041_204160


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l2041_204184

theorem polygon_interior_angles (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 720 → (180 * (n - 2) : ℝ) = sum_angles → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l2041_204184


namespace NUMINAMATH_CALUDE_set_of_multiples_of_six_l2041_204129

def is_closed_under_addition_subtraction (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S ∧ (x - y) ∈ S

def smallest_positive (S : Set ℝ) (a : ℝ) : Prop :=
  a ∈ S ∧ a > 0 ∧ ∀ x ∈ S, x > 0 → x ≥ a

theorem set_of_multiples_of_six (S : Set ℝ) :
  S.Nonempty →
  is_closed_under_addition_subtraction S →
  smallest_positive S 6 →
  S = {x : ℝ | ∃ n : ℤ, x = 6 * n} :=
by sorry

end NUMINAMATH_CALUDE_set_of_multiples_of_six_l2041_204129


namespace NUMINAMATH_CALUDE_pizza_slices_left_per_person_l2041_204165

theorem pizza_slices_left_per_person 
  (small_pizza : ℕ) 
  (large_pizza : ℕ) 
  (people : ℕ) 
  (slices_eaten_per_person : ℕ) 
  (h1 : small_pizza = 8) 
  (h2 : large_pizza = 14) 
  (h3 : people = 2) 
  (h4 : slices_eaten_per_person = 9) : 
  ((small_pizza + large_pizza) - (people * slices_eaten_per_person)) / people = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_per_person_l2041_204165


namespace NUMINAMATH_CALUDE_business_investment_l2041_204138

/-- Prove that the total investment is 90000 given the conditions of the business problem -/
theorem business_investment (a b c : ℕ) (total_profit a_share : ℕ) : 
  a = b + 6000 →
  c = b + 3000 →
  total_profit = 8640 →
  a_share = 3168 →
  a_share * (a + b + c) = a * total_profit →
  a + b + c = 90000 :=
by sorry

end NUMINAMATH_CALUDE_business_investment_l2041_204138


namespace NUMINAMATH_CALUDE_rectangle_area_l2041_204170

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 246) : L * B = 3650 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2041_204170


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_two_l2041_204153

theorem sum_of_solutions_is_two :
  ∃ (x y : ℤ), x^2 = x + 224 ∧ y^2 = y + 224 ∧ x + y = 2 ∧
  ∀ (z : ℤ), z^2 = z + 224 → z = x ∨ z = y :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_two_l2041_204153


namespace NUMINAMATH_CALUDE_abs_opposite_neg_six_l2041_204179

theorem abs_opposite_neg_six : |-(- 6)| = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_opposite_neg_six_l2041_204179


namespace NUMINAMATH_CALUDE_gabriel_pages_read_l2041_204111

theorem gabriel_pages_read (beatrix_pages cristobal_pages gabriel_pages : ℕ) : 
  beatrix_pages = 704 →
  cristobal_pages = 3 * beatrix_pages + 15 →
  gabriel_pages = 3 * (cristobal_pages + beatrix_pages) →
  gabriel_pages = 8493 := by
  sorry

end NUMINAMATH_CALUDE_gabriel_pages_read_l2041_204111


namespace NUMINAMATH_CALUDE_twelfth_term_of_sequence_l2041_204113

def arithmetic_sequence (a₁ a₂ a₃ : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

theorem twelfth_term_of_sequence (a₁ a₂ a₃ : ℚ) 
  (h₁ : a₁ = 1/2) 
  (h₂ : a₂ = 5/6) 
  (h₃ : a₃ = 7/6) 
  (h₄ : arithmetic_sequence a₁ a₂ a₃ 3 = a₃) :
  arithmetic_sequence a₁ a₂ a₃ 12 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_sequence_l2041_204113


namespace NUMINAMATH_CALUDE_square_equation_implies_m_equals_negative_one_l2041_204194

theorem square_equation_implies_m_equals_negative_one :
  (∀ a : ℝ, a^2 + m * a + 1/4 = (a - 1/2)^2) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_implies_m_equals_negative_one_l2041_204194


namespace NUMINAMATH_CALUDE_prob_one_male_is_three_fifths_l2041_204139

/-- Represents the class composition and sampling results -/
structure ClassSampling where
  total_students : ℕ
  male_students : ℕ
  selected_students : ℕ
  chosen_students : ℕ

/-- Calculates the number of male students selected in stratified sampling -/
def male_selected (c : ClassSampling) : ℕ :=
  (c.selected_students * c.male_students) / c.total_students

/-- Calculates the number of female students selected in stratified sampling -/
def female_selected (c : ClassSampling) : ℕ :=
  c.selected_students - male_selected c

/-- Calculates the probability of selecting exactly one male student from the chosen students -/
def prob_one_male (c : ClassSampling) : ℚ :=
  (male_selected c * female_selected c : ℚ) / (Nat.choose c.selected_students c.chosen_students : ℚ)

/-- Theorem stating the probability of selecting exactly one male student is 3/5 -/
theorem prob_one_male_is_three_fifths (c : ClassSampling) 
  (h1 : c.total_students = 50)
  (h2 : c.male_students = 30)
  (h3 : c.selected_students = 5)
  (h4 : c.chosen_students = 2) :
  prob_one_male c = 3/5 := by
  sorry

#eval prob_one_male ⟨50, 30, 5, 2⟩

end NUMINAMATH_CALUDE_prob_one_male_is_three_fifths_l2041_204139


namespace NUMINAMATH_CALUDE_men_sent_to_project_l2041_204137

/-- Represents the number of men sent to another project -/
def men_sent : ℕ := 33

/-- Represents the original number of men -/
def original_men : ℕ := 50

/-- Represents the original number of days to complete the work -/
def original_days : ℕ := 10

/-- Represents the new number of days to complete the work -/
def new_days : ℕ := 30

/-- Theorem stating that given the original conditions and the new completion time,
    the number of men sent to another project is 33 -/
theorem men_sent_to_project :
  (original_men * original_days = (original_men - men_sent) * new_days) →
  men_sent = 33 := by
  sorry


end NUMINAMATH_CALUDE_men_sent_to_project_l2041_204137


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l2041_204112

theorem quadratic_equation_solutions :
  let eq1 : ℂ → Prop := λ x ↦ x^2 - 6*x + 13 = 0
  let eq2 : ℂ → Prop := λ x ↦ 9*x^2 + 12*x + 29 = 0
  let sol1 : Set ℂ := {3 - 2*I, 3 + 2*I}
  let sol2 : Set ℂ := {-2/3 - 5/3*I, -2/3 + 5/3*I}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l2041_204112


namespace NUMINAMATH_CALUDE_salmon_count_l2041_204158

theorem salmon_count (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261) 
  (h2 : female_salmon = 259378) : 
  male_salmon + female_salmon = 971639 := by
  sorry

end NUMINAMATH_CALUDE_salmon_count_l2041_204158


namespace NUMINAMATH_CALUDE_bill_donut_order_combinations_l2041_204109

/-- The number of ways to distribute remaining donuts after ensuring at least one of each kind -/
def donut_combinations (total_donuts : ℕ) (donut_kinds : ℕ) (remaining_donuts : ℕ) : ℕ :=
  Nat.choose (remaining_donuts + donut_kinds - 1) (donut_kinds - 1)

/-- Theorem stating the number of combinations for Bill's donut order -/
theorem bill_donut_order_combinations :
  donut_combinations 8 5 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_bill_donut_order_combinations_l2041_204109


namespace NUMINAMATH_CALUDE_parabola_chord_midpoint_tangent_intersection_l2041_204125

/-- Given a parabola y² = 2px and a chord with endpoints P₁(x₁, y₁) and P₂(x₂, y₂),
    the line y = (y₁ + y₂)/2 passing through the midpoint M of the chord
    also passes through the intersection point of the tangents at P₁ and P₂. -/
theorem parabola_chord_midpoint_tangent_intersection
  (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : y₁^2 = 2*p*x₁)
  (h₂ : y₂^2 = 2*p*x₂)
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂)) :
  let midpoint_y := (y₁ + y₂) / 2
  let tangent₁ := fun x y ↦ y₁ * y = p * (x + x₁)
  let tangent₂ := fun x y ↦ y₂ * y = p * (x + x₂)
  let intersection := fun x y ↦ tangent₁ x y ∧ tangent₂ x y
  ∃ x, intersection x midpoint_y :=
sorry

end NUMINAMATH_CALUDE_parabola_chord_midpoint_tangent_intersection_l2041_204125


namespace NUMINAMATH_CALUDE_base_seven_246_equals_132_l2041_204105

/-- Converts a digit in base 7 to base 10 -/
def toBase10Digit (d : Nat) : Nat := d

/-- Converts a 3-digit number in base 7 to base 10 -/
def baseSevenToTen (d₂ d₁ d₀ : Nat) : Nat :=
  toBase10Digit d₂ * 7^2 + toBase10Digit d₁ * 7^1 + toBase10Digit d₀ * 7^0

theorem base_seven_246_equals_132 :
  baseSevenToTen 2 4 6 = 132 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_246_equals_132_l2041_204105


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2041_204192

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (∃ r : ℝ, r ≠ 0 ∧ a 5 = r * a 1 ∧ a 17 = r * a 5) →
  (∃ r : ℝ, r ≠ 0 ∧ a 5 = r * a 1 ∧ a 17 = r * a 5 ∧ r = 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2041_204192


namespace NUMINAMATH_CALUDE_solve_equation_l2041_204166

theorem solve_equation : 
  ∃ y : ℚ, (40 : ℚ) / 60 = Real.sqrt ((y / 60) - (10 : ℚ) / 60) → y = 110 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2041_204166


namespace NUMINAMATH_CALUDE_combined_tax_rate_l2041_204144

/-- Given Mork's and Mindy's tax rates and relative incomes, compute their combined tax rate -/
theorem combined_tax_rate (mork_rate mindy_rate : ℚ) (income_ratio : ℕ) :
  mork_rate = 45/100 →
  mindy_rate = 25/100 →
  income_ratio = 4 →
  (mork_rate + income_ratio * mindy_rate) / (1 + income_ratio) = 29/100 := by
sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l2041_204144


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2041_204148

/-- Proves that (x^4 + 12x^2 + 144)(x^2 - 12) = x^6 - 1728 for all real x. -/
theorem polynomial_multiplication (x : ℝ) : 
  (x^4 + 12*x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2041_204148


namespace NUMINAMATH_CALUDE_scientific_notation_of_43300000_l2041_204180

theorem scientific_notation_of_43300000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 43300000 = a * (10 : ℝ) ^ n ∧ a = 4.33 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_43300000_l2041_204180


namespace NUMINAMATH_CALUDE_rope_length_ratio_l2041_204145

/-- Given three ropes with lengths A, B, and C, where A is the longest, B is the middle, and C is the shortest,
    if A + C = B + 100 and C = 80, then the ratio of their lengths is (B + 20):B:80. -/
theorem rope_length_ratio (A B C : ℕ) (h1 : A ≥ B) (h2 : B ≥ C) (h3 : A + C = B + 100) (h4 : C = 80) :
  ∃ (k : ℕ), k > 0 ∧ A = k * (B + 20) ∧ B = k * B ∧ C = k * 80 :=
sorry

end NUMINAMATH_CALUDE_rope_length_ratio_l2041_204145


namespace NUMINAMATH_CALUDE_probability_specific_selection_l2041_204103

def num_shirts : ℕ := 6
def num_shorts : ℕ := 8
def num_socks : ℕ := 7
def total_items : ℕ := num_shirts + num_shorts + num_socks
def items_chosen : ℕ := 4

theorem probability_specific_selection :
  (Nat.choose num_shirts 1 * Nat.choose num_shorts 2 * Nat.choose num_socks 1) / 
  Nat.choose total_items items_chosen = 392 / 1995 := by
  sorry

end NUMINAMATH_CALUDE_probability_specific_selection_l2041_204103


namespace NUMINAMATH_CALUDE_chocolate_cost_is_75_cents_l2041_204154

/-- The cost of a candy bar in cents -/
def candy_bar_cost : ℕ := 25

/-- The cost of a pack of juice in cents -/
def juice_cost : ℕ := 50

/-- The total cost in cents for 3 candy bars, 2 pieces of chocolate, and 1 pack of juice -/
def total_cost : ℕ := 275

/-- The number of candy bars purchased -/
def num_candy_bars : ℕ := 3

/-- The number of chocolate pieces purchased -/
def num_chocolates : ℕ := 2

/-- The number of juice packs purchased -/
def num_juice_packs : ℕ := 1

theorem chocolate_cost_is_75_cents :
  ∃ (chocolate_cost : ℕ),
    chocolate_cost * num_chocolates + 
    candy_bar_cost * num_candy_bars + 
    juice_cost * num_juice_packs = total_cost ∧
    chocolate_cost = 75 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_is_75_cents_l2041_204154


namespace NUMINAMATH_CALUDE_area_inscribed_triangle_in_octagon_l2041_204176

/-- An equilateral octagon -/
structure EquilateralOctagon where
  side_length : ℝ
  is_positive : 0 < side_length

/-- An equilateral triangle formed by three diagonals of an equilateral octagon -/
def InscribedEquilateralTriangle (octagon : EquilateralOctagon) : Set (ℝ × ℝ) :=
  sorry

/-- The area of an inscribed equilateral triangle in an equilateral octagon -/
def area_inscribed_triangle (octagon : EquilateralOctagon) : ℝ :=
  sorry

/-- Theorem: The area of an inscribed equilateral triangle in an equilateral octagon
    with side length 60 is 900 -/
theorem area_inscribed_triangle_in_octagon :
  ∀ (octagon : EquilateralOctagon),
    octagon.side_length = 60 →
    area_inscribed_triangle octagon = 900 :=
by sorry

end NUMINAMATH_CALUDE_area_inscribed_triangle_in_octagon_l2041_204176


namespace NUMINAMATH_CALUDE_trip_length_calculation_l2041_204136

theorem trip_length_calculation (total : ℚ) 
  (h1 : total / 4 + 16 + total / 6 = total) : total = 192 / 7 := by
  sorry

end NUMINAMATH_CALUDE_trip_length_calculation_l2041_204136


namespace NUMINAMATH_CALUDE_ellipse_m_value_l2041_204151

/-- An ellipse with equation x² + y²/m = 1, foci on x-axis, and major axis twice the minor axis -/
structure Ellipse (m : ℝ) :=
  (equation : ∀ (x y : ℝ), x^2 + y^2/m = 1)
  (foci_on_x_axis : True)  -- This is a placeholder, as we can't directly represent this geometrically
  (major_twice_minor : True)  -- This is a placeholder for the condition

/-- The value of m for the given ellipse properties is 1/4 -/
theorem ellipse_m_value :
  ∀ (m : ℝ), Ellipse m → m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l2041_204151


namespace NUMINAMATH_CALUDE_veranda_area_l2041_204130

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) 
  (h1 : room_length = 20)
  (h2 : room_width = 12)
  (h3 : veranda_width = 2) : 
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - 
  room_length * room_width = 144 := by
  sorry

end NUMINAMATH_CALUDE_veranda_area_l2041_204130


namespace NUMINAMATH_CALUDE_pool_tiles_l2041_204198

theorem pool_tiles (total_needed : ℕ) (blue_tiles : ℕ) (additional_needed : ℕ) 
  (h1 : total_needed = 100)
  (h2 : blue_tiles = 48)
  (h3 : additional_needed = 20) :
  total_needed - additional_needed - blue_tiles = 32 := by
  sorry

#check pool_tiles

end NUMINAMATH_CALUDE_pool_tiles_l2041_204198


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l2041_204199

theorem different_color_chips_probability : 
  let total_chips := 20
  let blue_chips := 4
  let red_chips := 3
  let yellow_chips := 2
  let green_chips := 5
  let orange_chips := 6
  let prob_diff_color := 
    (blue_chips / total_chips) * ((total_chips - blue_chips) / total_chips) +
    (red_chips / total_chips) * ((total_chips - red_chips) / total_chips) +
    (yellow_chips / total_chips) * ((total_chips - yellow_chips) / total_chips) +
    (green_chips / total_chips) * ((total_chips - green_chips) / total_chips) +
    (orange_chips / total_chips) * ((total_chips - orange_chips) / total_chips)
  prob_diff_color = 31 / 40 := by
  sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l2041_204199


namespace NUMINAMATH_CALUDE_max_area_rectangle_in_345_triangle_l2041_204161

/-- The maximum area of a rectangle inscribed in a 3-4-5 right triangle -/
theorem max_area_rectangle_in_345_triangle : 
  ∃ (A : ℝ), A = 3 ∧ 
  ∀ (x y : ℝ), 
    0 ≤ x ∧ 0 ≤ y ∧ 
    (x ≤ 4 ∧ y ≤ 3 - (3/4) * x) ∨ (y ≤ 3 ∧ x ≤ 4 - (4/3) * y) →
    x * y ≤ A :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_in_345_triangle_l2041_204161


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2041_204173

theorem sqrt_product_equality : Real.sqrt 12 * Real.sqrt 8 = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2041_204173


namespace NUMINAMATH_CALUDE_total_teaching_time_l2041_204177

/-- Represents a teacher's class schedule -/
structure Schedule where
  math_classes : ℕ
  science_classes : ℕ
  history_classes : ℕ
  math_duration : ℝ
  science_duration : ℝ
  history_duration : ℝ

/-- Calculates the total teaching time for a given schedule -/
def total_time (s : Schedule) : ℝ :=
  s.math_classes * s.math_duration +
  s.science_classes * s.science_duration +
  s.history_classes * s.history_duration

/-- Eduardo's teaching schedule -/
def eduardo : Schedule :=
  { math_classes := 3
    science_classes := 4
    history_classes := 2
    math_duration := 1
    science_duration := 1.5
    history_duration := 2 }

/-- Frankie's teaching schedule (double of Eduardo's) -/
def frankie : Schedule :=
  { math_classes := 2 * eduardo.math_classes
    science_classes := 2 * eduardo.science_classes
    history_classes := 2 * eduardo.history_classes
    math_duration := eduardo.math_duration
    science_duration := eduardo.science_duration
    history_duration := eduardo.history_duration }

/-- Theorem: The total teaching time for Eduardo and Frankie is 39 hours -/
theorem total_teaching_time : total_time eduardo + total_time frankie = 39 := by
  sorry


end NUMINAMATH_CALUDE_total_teaching_time_l2041_204177


namespace NUMINAMATH_CALUDE_area_integral_solution_l2041_204155

theorem area_integral_solution (k : ℝ) : k > 0 →
  (∫ x in Set.Icc k (1/2), 1/x) = 2 * Real.log 2 ∨
  (∫ x in Set.Icc (1/2) k, 1/x) = 2 * Real.log 2 →
  k = 1/8 ∨ k = 2 := by
sorry

end NUMINAMATH_CALUDE_area_integral_solution_l2041_204155


namespace NUMINAMATH_CALUDE_no_number_with_specific_digit_sums_l2041_204140

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: No natural number exists with sum of digits 1000 and sum of square's digits 1000000 -/
theorem no_number_with_specific_digit_sums :
  ¬ ∃ n : ℕ, sumOfDigits n = 1000 ∧ sumOfDigits (n^2) = 1000000 := by sorry

end NUMINAMATH_CALUDE_no_number_with_specific_digit_sums_l2041_204140


namespace NUMINAMATH_CALUDE_rectangular_solid_depth_l2041_204128

/-- The surface area of a rectangular solid -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem rectangular_solid_depth :
  ∃ (h : ℝ), h > 0 ∧ surface_area 5 4 h = 58 → h = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_depth_l2041_204128


namespace NUMINAMATH_CALUDE_birthday_stickers_l2041_204195

/-- Represents the number of stickers Luke has at different stages --/
structure StickerCount where
  initial : ℕ
  bought : ℕ
  birthday : ℕ
  givenAway : ℕ
  used : ℕ
  final : ℕ

/-- Theorem stating the number of stickers Luke got for his birthday --/
theorem birthday_stickers (s : StickerCount) 
  (h1 : s.initial = 20)
  (h2 : s.bought = 12)
  (h3 : s.givenAway = 5)
  (h4 : s.used = 8)
  (h5 : s.final = 39)
  (h6 : s.final = s.initial + s.bought + s.birthday - s.givenAway - s.used) :
  s.birthday = 20 := by
  sorry

end NUMINAMATH_CALUDE_birthday_stickers_l2041_204195


namespace NUMINAMATH_CALUDE_natural_number_representation_l2041_204197

theorem natural_number_representation (n : ℕ) : 
  ∃ (x y : ℕ), n = x^3 / y^4 := by sorry

end NUMINAMATH_CALUDE_natural_number_representation_l2041_204197


namespace NUMINAMATH_CALUDE_cubic_equation_integer_roots_l2041_204118

theorem cubic_equation_integer_roots :
  ∀ x : ℤ, x^3 - 3*x^2 - 10*x + 20 = 0 ↔ x = -2 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_roots_l2041_204118


namespace NUMINAMATH_CALUDE_final_number_calculation_l2041_204114

theorem final_number_calculation (initial_number : ℕ) : 
  initial_number = 8 → 3 * (2 * initial_number + 9) = 75 :=
by sorry

end NUMINAMATH_CALUDE_final_number_calculation_l2041_204114


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2041_204101

theorem sum_of_fractions : (3 / 100 : ℚ) + (5 / 1000 : ℚ) + (7 / 10000 : ℚ) = (357 / 10000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2041_204101


namespace NUMINAMATH_CALUDE_order_of_trig_powers_l2041_204110

theorem order_of_trig_powers (α : Real) (h : π/4 < α ∧ α < π/2) :
  (Real.cos α) ^ (Real.sin α) < (Real.cos α) ^ (Real.cos α) ∧
  (Real.cos α) ^ (Real.cos α) < (Real.sin α) ^ (Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_order_of_trig_powers_l2041_204110


namespace NUMINAMATH_CALUDE_tenth_row_sum_l2041_204116

/-- The function representing the first term of the n-th row -/
def f (n : ℕ) : ℕ := 2 * n^2 - 3 * n + 3

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem tenth_row_sum :
  let first_term : ℕ := f 10
  let num_terms : ℕ := 2 * 10
  let common_diff : ℕ := 2
  arithmetic_sum first_term common_diff num_terms = 3840 := by
sorry

#eval arithmetic_sum (f 10) 2 (2 * 10)

end NUMINAMATH_CALUDE_tenth_row_sum_l2041_204116


namespace NUMINAMATH_CALUDE_exam_failure_count_l2041_204174

theorem exam_failure_count (total : ℕ) (pass_percentage : ℚ) (fail_count : ℕ) : 
  total = 400 → pass_percentage = 35 / 100 → fail_count = total - (pass_percentage * total).floor → fail_count = 260 := by
  sorry

end NUMINAMATH_CALUDE_exam_failure_count_l2041_204174


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2041_204104

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 3 * y = 11 :=
by
  -- The unique solution is y = 1.5
  use 1.5
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2041_204104


namespace NUMINAMATH_CALUDE_pineapple_problem_l2041_204181

theorem pineapple_problem (pineapple_cost : ℕ) (rings_per_pineapple : ℕ) 
  (rings_per_sale : ℕ) (sale_price : ℕ) (total_profit : ℕ) :
  pineapple_cost = 3 →
  rings_per_pineapple = 12 →
  rings_per_sale = 4 →
  sale_price = 5 →
  total_profit = 72 →
  ∃ (num_pineapples : ℕ),
    num_pineapples * (rings_per_pineapple / rings_per_sale * sale_price - pineapple_cost) = total_profit ∧
    num_pineapples = 6 := by
  sorry

end NUMINAMATH_CALUDE_pineapple_problem_l2041_204181


namespace NUMINAMATH_CALUDE_triangle_inequality_l2041_204120

open Real

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b^2 + c^2) / a + (c^2 + a^2) / b + (a^2 + b^2) / c ≥ 2 * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2041_204120


namespace NUMINAMATH_CALUDE_tan_195_in_terms_of_cos_165_l2041_204122

theorem tan_195_in_terms_of_cos_165 (a : ℝ) (h : Real.cos (165 * π / 180) = a) :
  Real.tan (195 * π / 180) = -Real.sqrt (1 - a^2) / a := by
  sorry

end NUMINAMATH_CALUDE_tan_195_in_terms_of_cos_165_l2041_204122


namespace NUMINAMATH_CALUDE_B_determines_xy_l2041_204183

/-- Function B that determines x and y --/
def B (x y : ℕ) : ℕ := (x + y) * (x + y + 1) - y

/-- Theorem stating that B(x, y) uniquely determines x and y --/
theorem B_determines_xy (x y : ℕ) : 
  ∀ a b : ℕ, B x y = B a b → x = a ∧ y = b := by sorry

end NUMINAMATH_CALUDE_B_determines_xy_l2041_204183


namespace NUMINAMATH_CALUDE_subject_selection_l2041_204186

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem subject_selection (n : ℕ) (h : n = 7) :
  -- Total number of ways to choose any three subjects
  choose n 3 = choose 7 3 ∧
  -- If at least one of two specific subjects is chosen
  choose 2 1 * choose 5 2 + choose 2 2 * choose 5 1 = choose 2 1 * choose 5 2 + choose 2 2 * choose 5 1 ∧
  -- If two specific subjects cannot be chosen at the same time
  choose n 3 - choose 2 2 * choose 5 1 = choose 7 3 - choose 2 2 * choose 5 1 ∧
  -- If at least one of two specific subjects is chosen, and two other specific subjects are not chosen at the same time
  (choose 1 1 * choose 4 2 + choose 1 1 * choose 5 2 + choose 2 2 * choose 4 1 = 20) := by
  sorry

end NUMINAMATH_CALUDE_subject_selection_l2041_204186


namespace NUMINAMATH_CALUDE_work_completion_time_l2041_204193

theorem work_completion_time (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (1/a + 1/b = 1/4) → (1/b = 1/6) → (1/a = 1/12) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2041_204193


namespace NUMINAMATH_CALUDE_base4_1010_equals_68_l2041_204117

/-- Converts a base-4 digit to its decimal value -/
def base4ToDecimal (digit : Nat) : Nat :=
  match digit with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | _ => 0  -- Default case for invalid digits

/-- Converts a list of base-4 digits to a decimal number -/
def convertBase4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + base4ToDecimal d * (4 ^ i)) 0

theorem base4_1010_equals_68 : 
  convertBase4ToDecimal [0, 1, 0, 1] = 68 := by
  sorry

#eval convertBase4ToDecimal [0, 1, 0, 1]

end NUMINAMATH_CALUDE_base4_1010_equals_68_l2041_204117


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l2041_204121

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of a when f(x) ≥ 4
theorem range_of_a :
  {a : ℝ | ∃ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l2041_204121


namespace NUMINAMATH_CALUDE_expanded_lattice_equilateral_triangles_l2041_204132

/-- Represents a point in the triangular lattice --/
structure LatticePoint where
  x : ℚ
  y : ℚ

/-- The set of all points in the expanded lattice --/
def ExpandedLattice : Set LatticePoint :=
  sorry

/-- Checks if three points form an equilateral triangle --/
def IsEquilateralTriangle (p1 p2 p3 : LatticePoint) : Prop :=
  sorry

/-- Counts the number of equilateral triangles in the expanded lattice --/
def CountEquilateralTriangles (lattice : Set LatticePoint) : ℕ :=
  sorry

/-- Main theorem: The number of equilateral triangles in the expanded lattice is 14 --/
theorem expanded_lattice_equilateral_triangles :
  CountEquilateralTriangles ExpandedLattice = 14 :=
sorry

end NUMINAMATH_CALUDE_expanded_lattice_equilateral_triangles_l2041_204132


namespace NUMINAMATH_CALUDE_max_discount_rate_l2041_204162

/-- Represents the maximum discount rate problem -/
theorem max_discount_rate
  (cost_price : ℝ)
  (original_price : ℝ)
  (min_profit_margin : ℝ)
  (h1 : cost_price = 4)
  (h2 : original_price = 5)
  (h3 : min_profit_margin = 0.1)
  : ∃ (max_discount : ℝ),
    max_discount = 0.12 ∧
    ∀ (discount : ℝ),
      discount ≤ max_discount →
      (original_price * (1 - discount) - cost_price) / cost_price ≥ min_profit_margin :=
sorry

end NUMINAMATH_CALUDE_max_discount_rate_l2041_204162


namespace NUMINAMATH_CALUDE_rhombus_sum_difference_l2041_204182

theorem rhombus_sum_difference (a b c d : ℝ) (h : a + b + c + d = 2021) :
  ((a + 1) * (b + 1) + (b + 1) * (c + 1) + (c + 1) * (d + 1) + (d + 1) * (a + 1)) -
  (a * b + b * c + c * d + d * a) = 4046 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_sum_difference_l2041_204182


namespace NUMINAMATH_CALUDE_arithmetic_puzzle_l2041_204119

theorem arithmetic_puzzle : (100 - (5050 - 450)) + (5050 - (450 - 100)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_puzzle_l2041_204119


namespace NUMINAMATH_CALUDE_gumball_sales_total_l2041_204178

theorem gumball_sales_total (price1 price2 price3 price4 price5 : ℕ) 
  (h1 : price1 = 12)
  (h2 : price2 = 15)
  (h3 : price3 = 8)
  (h4 : price4 = 10)
  (h5 : price5 = 20) :
  price1 + price2 + price3 + price4 + price5 = 65 := by
  sorry

end NUMINAMATH_CALUDE_gumball_sales_total_l2041_204178


namespace NUMINAMATH_CALUDE_a_equals_two_l2041_204106

/-- The function f(x) = x^2 - 14x + 52 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 52

/-- The function g(x) = ax + b, where a and b are positive real numbers -/
def g (a b : ℝ) (x : ℝ) : ℝ := a*x + b

/-- Theorem stating that a = 2 given the conditions -/
theorem a_equals_two (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : f (g a b (-5)) = 3) (h2 : f (g a b 0) = 103) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_two_l2041_204106


namespace NUMINAMATH_CALUDE_tenth_element_is_6785_l2041_204124

/-- A list of all four-digit integers using digits 5, 6, 7, and 8 exactly once, ordered from least to greatest -/
def fourDigitList : List Nat := sorry

/-- The 10th element in the fourDigitList -/
def tenthElement : Nat := sorry

/-- Theorem stating that the 10th element in the fourDigitList is 6785 -/
theorem tenth_element_is_6785 : tenthElement = 6785 := by sorry

end NUMINAMATH_CALUDE_tenth_element_is_6785_l2041_204124


namespace NUMINAMATH_CALUDE_ab_range_l2041_204100

theorem ab_range (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 3 → a * b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_range_l2041_204100


namespace NUMINAMATH_CALUDE_grain_demand_formula_l2041_204143

/-- World grain supply and demand model -/
structure GrainModel where
  S : ℝ  -- World grain supply
  D : ℝ  -- World grain demand
  F : ℝ  -- Production fluctuations
  P : ℝ  -- Population growth
  S0 : ℝ  -- Base supply value
  D0 : ℝ  -- Initial demand value

/-- Conditions for the grain model -/
def GrainModelConditions (m : GrainModel) : Prop :=
  m.S = 0.75 * m.D ∧
  m.S = m.S0 * (1 + m.F) ∧
  m.D = m.D0 * (1 + m.P) ∧
  m.S0 = 1800000

/-- Theorem: Given the conditions, the world grain demand D can be expressed as D = (1,800,000 * (1 + F)) / 0.75 -/
theorem grain_demand_formula (m : GrainModel) (h : GrainModelConditions m) :
  m.D = (1800000 * (1 + m.F)) / 0.75 := by
  sorry


end NUMINAMATH_CALUDE_grain_demand_formula_l2041_204143


namespace NUMINAMATH_CALUDE_expression_factorization_l2041_204191

theorem expression_factorization (x : ℝ) : 
  (12 * x^6 + 30 * x^4 - 6) - (2 * x^6 - 4 * x^4 - 6) = 2 * x^4 * (5 * x^2 + 17) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2041_204191


namespace NUMINAMATH_CALUDE_unique_x_for_real_sqrt_l2041_204187

theorem unique_x_for_real_sqrt (y : ℝ) : ∃! x : ℝ, ∃ z : ℝ, z^2 = -(x + 2*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_for_real_sqrt_l2041_204187


namespace NUMINAMATH_CALUDE_min_sum_last_three_digits_equal_l2041_204142

/-- 
Given two positive integers m and n, where n > m ≥ 1, 
and the last three digits of 1978^n and 1978^m are equal,
prove that the minimum value of m + n is 106.
-/
theorem min_sum_last_three_digits_equal (m n : ℕ) : 
  m ≥ 1 → n > m → 
  (1978^n) % 1000 = (1978^m) % 1000 → 
  ∃ (m₀ n₀ : ℕ), m₀ ≥ 1 ∧ n₀ > m₀ ∧ 
    (1978^n₀) % 1000 = (1978^m₀) % 1000 ∧
    m₀ + n₀ = 106 ∧ 
    ∀ (m' n' : ℕ), m' ≥ 1 → n' > m' → 
      (1978^n') % 1000 = (1978^m') % 1000 → 
      m' + n' ≥ 106 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_last_three_digits_equal_l2041_204142


namespace NUMINAMATH_CALUDE_chess_club_mixed_groups_l2041_204147

theorem chess_club_mixed_groups 
  (total_children : Nat) 
  (total_groups : Nat) 
  (group_size : Nat)
  (boy_games : Nat)
  (girl_games : Nat) :
  total_children = 90 →
  total_groups = 30 →
  group_size = 3 →
  boy_games = 30 →
  girl_games = 14 →
  (∃ (mixed_groups : Nat), 
    mixed_groups = 23 ∧ 
    mixed_groups * 2 = total_children - boy_games - girl_games) := by
  sorry

end NUMINAMATH_CALUDE_chess_club_mixed_groups_l2041_204147


namespace NUMINAMATH_CALUDE_rotate_A_equals_B_l2041_204185

-- Define a 2x2 grid
structure Grid2x2 :=
  (cells : Fin 2 → Fin 2 → Bool)

-- Define rotations
def rotate90CounterClockwise (g : Grid2x2) : Grid2x2 :=
  { cells := λ i j => g.cells (1 - j) i }

-- Define the initial position of 'A'
def initialA : Grid2x2 :=
  { cells := λ i j => (i = 1 ∧ j = 0) ∨ (i = 1 ∧ j = 1) ∨ (i = 0 ∧ j = 1) }

-- Define the final position of 'A' (option B)
def finalA : Grid2x2 :=
  { cells := λ i j => (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 1) ∨ (i = 2 ∧ j = 1) ∨ (i = 2 ∧ j = 0) }

-- Theorem statement
theorem rotate_A_equals_B : rotate90CounterClockwise initialA = finalA := by
  sorry

end NUMINAMATH_CALUDE_rotate_A_equals_B_l2041_204185


namespace NUMINAMATH_CALUDE_remainder_of_98765432101_mod_240_l2041_204159

theorem remainder_of_98765432101_mod_240 :
  98765432101 % 240 = 61 := by sorry

end NUMINAMATH_CALUDE_remainder_of_98765432101_mod_240_l2041_204159


namespace NUMINAMATH_CALUDE_column_with_most_shaded_boxes_l2041_204188

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

def number_of_divisors (n : ℕ) : ℕ :=
  (divisors n).card

theorem column_with_most_shaded_boxes :
  ∀ n ∈ ({144, 120, 150, 96, 100} : Finset ℕ),
    number_of_divisors n ≤ number_of_divisors 120 :=
by sorry

end NUMINAMATH_CALUDE_column_with_most_shaded_boxes_l2041_204188


namespace NUMINAMATH_CALUDE_bob_mary_sheep_ratio_l2041_204107

/-- The number of sheep Mary has initially -/
def mary_initial_sheep : ℕ := 300

/-- The number of sheep Mary buys -/
def mary_bought_sheep : ℕ := 266

/-- The difference between Bob's sheep and Mary's sheep after Mary's purchase -/
def sheep_difference : ℕ := 69

/-- Bob's sheep count -/
def bob_sheep : ℕ := mary_initial_sheep + mary_bought_sheep + sheep_difference

theorem bob_mary_sheep_ratio : 
  (bob_sheep : ℚ) / mary_initial_sheep = 635 / 300 := by sorry

end NUMINAMATH_CALUDE_bob_mary_sheep_ratio_l2041_204107


namespace NUMINAMATH_CALUDE_nested_root_equality_l2041_204146

theorem nested_root_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt x)) = (x ^ 7) ^ (1 / 8) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_equality_l2041_204146


namespace NUMINAMATH_CALUDE_quadratic_general_form_l2041_204190

/-- Given a quadratic equation 3x² + 1 = 7x, its general form is 3x² - 7x + 1 = 0 -/
theorem quadratic_general_form : 
  ∀ x : ℝ, 3 * x^2 + 1 = 7 * x ↔ 3 * x^2 - 7 * x + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_general_form_l2041_204190


namespace NUMINAMATH_CALUDE_projection_of_a_onto_b_l2041_204126

/-- Prove that the projection of vector a = (3, 4) onto vector b = (0, 1) results in the vector (0, 4) -/
theorem projection_of_a_onto_b :
  let a : Fin 2 → ℝ := ![3, 4]
  let b : Fin 2 → ℝ := ![0, 1]
  let proj := (a • b) / (b • b) • b
  proj = ![0, 4] := by sorry

end NUMINAMATH_CALUDE_projection_of_a_onto_b_l2041_204126


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2041_204123

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x + 2 > a ∧ x - 1 < b) ↔ (1 < x ∧ x < 3)) → 
  (a = 3 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2041_204123


namespace NUMINAMATH_CALUDE_soup_kettle_capacity_l2041_204157

theorem soup_kettle_capacity (current_percentage : ℚ) (current_servings : ℕ) : 
  current_percentage = 55 / 100 →
  current_servings = 88 →
  (current_servings : ℚ) / current_percentage = 160 :=
by sorry

end NUMINAMATH_CALUDE_soup_kettle_capacity_l2041_204157


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2041_204172

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 10 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 10 = 0 → y = x) ↔ 
  (m = 2 + 2 * Real.sqrt 30 ∨ m = 2 - 2 * Real.sqrt 30) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2041_204172


namespace NUMINAMATH_CALUDE_circumcenter_coincidence_l2041_204149

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  distance : ℝ

/-- The circumcenter of a tetrahedron -/
def circumcenter (t : Tetrahedron) : Point3D :=
  sorry

/-- The inscribed sphere of a tetrahedron -/
def inscribedSphere (t : Tetrahedron) : Sphere :=
  sorry

/-- Points where the inscribed sphere touches the faces of the tetrahedron -/
def touchPoints (t : Tetrahedron) (s : Sphere) : (Point3D × Point3D × Point3D × Point3D) :=
  sorry

/-- Plane equidistant from a point and another plane -/
def equidistantPlane (p : Point3D) (pl : Plane) : Plane :=
  sorry

/-- Tetrahedron formed by four planes -/
def tetrahedronFromPlanes (p1 p2 p3 p4 : Plane) : Tetrahedron :=
  sorry

/-- Main theorem statement -/
theorem circumcenter_coincidence (t : Tetrahedron) : 
  let s := inscribedSphere t
  let (A₁, B₁, C₁, D₁) := touchPoints t s
  let p1 := equidistantPlane t.A (Plane.mk B₁ 0)
  let p2 := equidistantPlane t.B (Plane.mk C₁ 0)
  let p3 := equidistantPlane t.C (Plane.mk D₁ 0)
  let p4 := equidistantPlane t.D (Plane.mk A₁ 0)
  let t' := tetrahedronFromPlanes p1 p2 p3 p4
  circumcenter t = circumcenter t' :=
by
  sorry

end NUMINAMATH_CALUDE_circumcenter_coincidence_l2041_204149


namespace NUMINAMATH_CALUDE_total_days_2000_to_2003_l2041_204150

def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 ≠ 0 || year % 400 == 0)

def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

def totalDaysInRange (startYear endYear : Nat) : Nat :=
  (List.range (endYear - startYear + 1)).map (fun i => daysInYear (startYear + i))
    |>.sum

theorem total_days_2000_to_2003 :
  totalDaysInRange 2000 2003 = 1461 :=
by
  sorry

end NUMINAMATH_CALUDE_total_days_2000_to_2003_l2041_204150


namespace NUMINAMATH_CALUDE_age_difference_proof_l2041_204156

/-- Given the ages of Milena, her grandmother, and her grandfather, prove the age difference between Milena and her grandfather. -/
theorem age_difference_proof (milena_age : ℕ) (grandmother_age_factor : ℕ) (grandfather_age_difference : ℕ) 
  (h1 : milena_age = 7)
  (h2 : grandmother_age_factor = 9)
  (h3 : grandfather_age_difference = 2) :
  grandfather_age_difference + grandmother_age_factor * milena_age - milena_age = 58 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2041_204156


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2041_204141

theorem regular_polygon_sides (n : ℕ) (angle_OAB : ℝ) : 
  n > 0 → 
  angle_OAB = 72 → 
  (360 : ℝ) / angle_OAB = n → 
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2041_204141


namespace NUMINAMATH_CALUDE_number_of_boys_at_park_l2041_204102

theorem number_of_boys_at_park : 
  ∀ (girls parents groups group_size total_people boys : ℕ),
    girls = 14 →
    parents = 50 →
    groups = 3 →
    group_size = 25 →
    total_people = groups * group_size →
    boys = total_people - (girls + parents) →
    boys = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_at_park_l2041_204102


namespace NUMINAMATH_CALUDE_sum_of_m_for_integer_solutions_l2041_204115

theorem sum_of_m_for_integer_solutions : ∃ (S : Finset Int),
  (∀ m : Int, m ∈ S ↔ 
    (∃ x y : Int, x^2 - m*x + 15 = 0 ∧ y^2 - m*y + 15 = 0 ∧ x ≠ y)) ∧
  (S.sum id = 48) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_m_for_integer_solutions_l2041_204115


namespace NUMINAMATH_CALUDE_origin_on_circle_M_l2041_204175

-- Define the parabola C: y² = 2x
def parabola_C (x y : ℝ) : Prop := y^2 = 2*x

-- Define the line l passing through (2,0)
def line_l (x y : ℝ) (k : ℝ) : Prop := y = k*(x - 2)

-- Define points A and B as intersections of line l and parabola C
def point_A (k : ℝ) : ℝ × ℝ := sorry
def point_B (k : ℝ) : ℝ × ℝ := sorry

-- Define circle M with diameter AB
def circle_M (x y : ℝ) (k : ℝ) : Prop :=
  let A := point_A k
  let B := point_B k
  let center := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  let radius := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 2
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Theorem: The origin O(0,0) lies on circle M
theorem origin_on_circle_M (k : ℝ) : circle_M 0 0 k :=
  sorry


end NUMINAMATH_CALUDE_origin_on_circle_M_l2041_204175
