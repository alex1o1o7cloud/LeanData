import Mathlib

namespace product_correction_l3652_365277

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem product_correction (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (reverse_digits a * b = 221) →  -- reversed a times b is 221
  (a * b = 527 ∨ a * b = 923) :=  -- correct product is 527 or 923
by sorry

end product_correction_l3652_365277


namespace la_retail_women_ratio_l3652_365298

/-- The ratio of women working in retail to the total number of women in Los Angeles -/
def retail_women_ratio (total_population : ℕ) (women_population : ℕ) (retail_women : ℕ) : ℚ :=
  retail_women / women_population

theorem la_retail_women_ratio :
  let total_population : ℕ := 6000000
  let women_population : ℕ := total_population / 2
  let retail_women : ℕ := 1000000
  retail_women_ratio total_population women_population retail_women = 1 / 3 := by
sorry

end la_retail_women_ratio_l3652_365298


namespace investment_problem_l3652_365278

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem statement for the investment problem -/
theorem investment_problem (P : ℝ) :
  (∃ r : ℝ, simple_interest P r 2 = 520 ∧ simple_interest P r 7 = 820) →
  P = 400 := by
  sorry

end investment_problem_l3652_365278


namespace store_comparison_l3652_365241

/-- Represents the cost function for store A -/
def cost_A (x : ℕ) : ℝ :=
  if x = 0 then 0 else 140 * x + 60

/-- Represents the cost function for store B -/
def cost_B (x : ℕ) : ℝ := 150 * x

theorem store_comparison (x : ℕ) (h : x ≥ 1) :
  (cost_A x = 140 * x + 60) ∧
  (cost_B x = 150 * x) ∧
  (∀ y : ℕ, 1 ≤ y ∧ y < 6 → cost_A y < cost_B y) ∧
  (∀ z : ℕ, z > 6 → cost_A z > cost_B z) :=
by sorry

end store_comparison_l3652_365241


namespace optimal_rent_and_income_l3652_365282

def daily_net_income (rent : ℕ) : ℤ :=
  if rent ≤ 6 then
    50 * rent - 115
  else
    (50 - 3 * (rent - 6)) * rent - 115

def is_valid_rent (rent : ℕ) : Prop :=
  3 ≤ rent ∧ rent ≤ 20 ∧ daily_net_income rent > 0

theorem optimal_rent_and_income :
  ∃ (optimal_rent : ℕ) (max_income : ℤ),
    is_valid_rent optimal_rent ∧
    max_income = daily_net_income optimal_rent ∧
    optimal_rent = 11 ∧
    max_income = 270 ∧
    ∀ (rent : ℕ), is_valid_rent rent → daily_net_income rent ≤ max_income :=
by
  sorry

end optimal_rent_and_income_l3652_365282


namespace shaded_triangle_probability_l3652_365284

/-- Given a set of triangles with equal selection probability, 
    this function calculates the probability of selecting a shaded triangle -/
def probability_shaded_triangle (total_triangles : ℕ) (shaded_triangles : ℕ) : ℚ :=
  shaded_triangles / total_triangles

/-- Theorem: The probability of selecting a shaded triangle 
    given 6 total triangles and 2 shaded triangles is 1/3 -/
theorem shaded_triangle_probability :
  probability_shaded_triangle 6 2 = 1/3 := by
  sorry

#eval probability_shaded_triangle 6 2

end shaded_triangle_probability_l3652_365284


namespace greatest_x_value_l3652_365219

theorem greatest_x_value (x : ℕ+) (y : ℕ) (b : ℚ) 
  (h1 : y.Prime)
  (h2 : y = 2)
  (h3 : b = 3.56)
  (h4 : (b * y^x.val : ℚ) < 600000) :
  x.val ≤ 17 ∧ ∃ (x' : ℕ+), x'.val = 17 ∧ (b * y^x'.val : ℚ) < 600000 :=
by sorry

end greatest_x_value_l3652_365219


namespace decreasing_function_inequality_l3652_365299

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : ∀ x y, x < y → f x > f y) 
  (h_inequality : f (1 - a) < f (2 * a - 1)) : 
  a < 2/3 := by
  sorry

end decreasing_function_inequality_l3652_365299


namespace sum_of_three_numbers_l3652_365215

theorem sum_of_three_numbers : 2.12 + 0.004 + 0.345 = 2.469 := by
  sorry

end sum_of_three_numbers_l3652_365215


namespace trapezium_side_length_l3652_365262

/-- Given a trapezium with the specified properties, prove that the length of the unknown parallel side is 20 cm. -/
theorem trapezium_side_length 
  (known_side : ℝ) 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : known_side = 18) 
  (h2 : height = 12) 
  (h3 : area = 228) : 
  ∃ unknown_side : ℝ, 
    area = (1/2) * (known_side + unknown_side) * height ∧ 
    unknown_side = 20 :=
sorry

end trapezium_side_length_l3652_365262


namespace jamie_lost_balls_jamie_lost_six_balls_l3652_365214

theorem jamie_lost_balls (initial_red : ℕ) (blue_multiplier : ℕ) (yellow_bought : ℕ) (final_total : ℕ) : ℕ :=
  let initial_blue := blue_multiplier * initial_red
  let initial_total := initial_red + initial_blue + yellow_bought
  initial_total - final_total

theorem jamie_lost_six_balls : jamie_lost_balls 16 2 32 74 = 6 := by
  sorry

end jamie_lost_balls_jamie_lost_six_balls_l3652_365214


namespace sqrt_product_plus_one_equals_271_l3652_365216

theorem sqrt_product_plus_one_equals_271 : 
  Real.sqrt ((18 : ℝ) * 17 * 16 * 15 + 1) = 271 := by
  sorry

end sqrt_product_plus_one_equals_271_l3652_365216


namespace bennys_cards_l3652_365286

theorem bennys_cards (x : ℕ) : 
  (x + 4) / 2 = 34 → x = 68 := by sorry

end bennys_cards_l3652_365286


namespace children_share_sum_l3652_365232

theorem children_share_sum (total_money : ℕ) (ratio_a ratio_b ratio_c ratio_d : ℕ) : 
  total_money = 4500 → 
  ratio_a = 2 → 
  ratio_b = 4 → 
  ratio_c = 5 → 
  ratio_d = 4 → 
  (ratio_a + ratio_b) * total_money / (ratio_a + ratio_b + ratio_c + ratio_d) = 1800 := by
sorry

end children_share_sum_l3652_365232


namespace convex_lattice_nonagon_centroid_l3652_365204

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : Int
  y : Int

/-- A convex nonagon represented by 9 lattice points -/
structure ConvexLatticeNonagon where
  vertices : Fin 9 → LatticePoint
  is_convex : Bool  -- We assume this property without defining it explicitly

/-- The centroid of three points -/
def centroid (p1 p2 p3 : LatticePoint) : (Rat × Rat) :=
  ((p1.x + p2.x + p3.x) / 3, (p1.y + p2.y + p3.y) / 3)

/-- Check if a point with rational coordinates is a lattice point -/
def isLatticePoint (p : Rat × Rat) : Prop :=
  ∃ (x y : Int), p.1 = x ∧ p.2 = y

/-- Main theorem: Any convex lattice nonagon has three vertices whose centroid is a lattice point -/
theorem convex_lattice_nonagon_centroid (n : ConvexLatticeNonagon) :
  ∃ (i j k : Fin 9), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    isLatticePoint (centroid (n.vertices i) (n.vertices j) (n.vertices k)) :=
  sorry

end convex_lattice_nonagon_centroid_l3652_365204


namespace marta_book_count_l3652_365208

/-- The number of books on Marta's shelf after all changes -/
def final_book_count (initial_books added_books removed_books birthday_multiplier : ℕ) : ℕ :=
  initial_books + added_books - removed_books + birthday_multiplier * initial_books

/-- Theorem stating the final number of books on Marta's shelf -/
theorem marta_book_count : final_book_count 38 10 5 3 = 157 := by
  sorry

#eval final_book_count 38 10 5 3

end marta_book_count_l3652_365208


namespace condition_one_condition_two_l3652_365294

-- Define the lines l₁ and l₂
def l₁ (a b : ℝ) (x y : ℝ) : Prop := a * x - b * y + 4 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y + b = 0

-- Define perpendicularity condition
def perpendicular (a b : ℝ) : Prop := a * (a - 1) - b = 0

-- Define parallel condition
def parallel (a b : ℝ) : Prop := a * (a - 1) + b = 0

-- Define the condition that l₁ passes through (-3, -1)
def passes_through (a b : ℝ) : Prop := l₁ a b (-3) (-1)

-- Define the condition that intercepts are equal
def equal_intercepts (a b : ℝ) : Prop := b = -a

theorem condition_one (a b : ℝ) :
  perpendicular a b ∧ passes_through a b → a = 2 ∧ b = 2 :=
by sorry

theorem condition_two (a b : ℝ) :
  parallel a b ∧ equal_intercepts a b → a = 2 ∧ b = -2 :=
by sorry

end condition_one_condition_two_l3652_365294


namespace shoe_price_calculation_l3652_365209

theorem shoe_price_calculation (num_shoes : ℕ) (num_shirts : ℕ) (shirt_price : ℚ) (total_earnings_per_person : ℚ) :
  num_shoes = 6 →
  num_shirts = 18 →
  shirt_price = 2 →
  total_earnings_per_person = 27 →
  ∃ (shoe_price : ℚ), 
    (num_shoes * shoe_price + num_shirts * shirt_price) / 2 = total_earnings_per_person ∧
    shoe_price = 3 :=
by sorry

end shoe_price_calculation_l3652_365209


namespace x_power_27_minus_reciprocal_l3652_365263

theorem x_power_27_minus_reciprocal (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^27 - 1/(x^27) = 0 := by
  sorry

end x_power_27_minus_reciprocal_l3652_365263


namespace range_of_fraction_l3652_365288

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : 2 < b ∧ b < 4) :
  1/4 < a/b ∧ a/b < 3/2 := by
  sorry

end range_of_fraction_l3652_365288


namespace total_hair_cut_equals_41_l3652_365227

/-- Represents the hair length of a person before and after a haircut. -/
structure Haircut where
  original : ℕ  -- Original hair length in inches
  cut : ℕ       -- Amount of hair cut off in inches

/-- Calculates the total amount of hair cut off from multiple haircuts. -/
def total_hair_cut (haircuts : List Haircut) : ℕ :=
  haircuts.map (·.cut) |>.sum

/-- Theorem stating that the total hair cut off from Isabella, Damien, and Ella is 41 inches. -/
theorem total_hair_cut_equals_41 : 
  let isabella : Haircut := { original := 18, cut := 9 }
  let damien : Haircut := { original := 24, cut := 12 }
  let ella : Haircut := { original := 30, cut := 20 }
  total_hair_cut [isabella, damien, ella] = 41 := by
  sorry

end total_hair_cut_equals_41_l3652_365227


namespace unique_solution_mn_l3652_365212

theorem unique_solution_mn : 
  ∃! (m n : ℕ+), 18 * (m : ℕ) * (n : ℕ) = 72 - 9 * (m : ℕ) - 4 * (n : ℕ) ∧ m = 8 ∧ n = 36 := by
  sorry

end unique_solution_mn_l3652_365212


namespace sum_of_coefficients_l3652_365210

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 60 := by
sorry

end sum_of_coefficients_l3652_365210


namespace min_distance_complex_l3652_365295

/-- Given a complex number z satisfying |z + 3i| = 1, 
    the minimum value of |z - 1 + 2i| is √2 - 1. -/
theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 3 * Complex.I) = 1) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 2 - 1 ∧
  ∀ (w : ℂ), Complex.abs (w + 3 * Complex.I) = 1 →
  Complex.abs (w - 1 + 2 * Complex.I) ≥ min_val :=
sorry

end min_distance_complex_l3652_365295


namespace arithmetic_sequence_a8_l3652_365268

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Theorem: In an arithmetic sequence where a_7 + a_9 = 16, a_8 = 8 -/
theorem arithmetic_sequence_a8 (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : a 7 + a 9 = 16) :
  a 8 = 8 := by
  sorry

end arithmetic_sequence_a8_l3652_365268


namespace f_2005_of_2_pow_2006_l3652_365246

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- f₁(k) is the square of the sum of digits of k -/
def f₁ (k : ℕ) : ℕ := (sum_of_digits k) ^ 2

/-- fₙ₊₁(k) = f₁(fₙ(k)) for n ≥ 1 -/
def f (n : ℕ) (k : ℕ) : ℕ :=
  match n with
  | 0 => k
  | n + 1 => f₁ (f n k)

/-- The main theorem to prove -/
theorem f_2005_of_2_pow_2006 : f 2005 (2^2006) = 169 := by sorry

end f_2005_of_2_pow_2006_l3652_365246


namespace intersection_of_sets_l3652_365255

theorem intersection_of_sets : 
  let M : Set ℕ := {0, 1, 2, 3}
  let N : Set ℕ := {1, 3, 4}
  M ∩ N = {1, 3} := by
sorry

end intersection_of_sets_l3652_365255


namespace math_competition_probabilities_l3652_365273

-- Define the total number of questions and the number of questions each student answers
def total_questions : ℕ := 6
def questions_answered : ℕ := 3

-- Define the number of questions student A can correctly answer
def student_a_correct : ℕ := 4

-- Define the probability of student B correctly answering a question
def student_b_prob : ℚ := 2/3

-- Define the point values for correct answers
def points_a : ℕ := 15
def points_b : ℕ := 10

-- Define the probability that students A and B together correctly answer 3 questions
def prob_three_correct : ℚ := 31/135

-- Define the expected value of the total score
def expected_total_score : ℕ := 50

-- Theorem statement
theorem math_competition_probabilities :
  (prob_three_correct = 31/135) ∧
  (expected_total_score = 50) := by
  sorry

end math_competition_probabilities_l3652_365273


namespace quadratic_equation_solution_sum_l3652_365297

theorem quadratic_equation_solution_sum : ∃ (a b : ℝ), 
  (a^2 - 6*a + 15 = 24) ∧ 
  (b^2 - 6*b + 15 = 24) ∧ 
  (a ≥ b) ∧ 
  (3*a + 2*b = 15 + 3*Real.sqrt 2) := by
  sorry

end quadratic_equation_solution_sum_l3652_365297


namespace florist_roses_l3652_365207

theorem florist_roses (initial : ℕ) (sold : ℕ) (picked : ℕ) : 
  initial = 37 → sold = 16 → picked = 19 → initial - sold + picked = 40 := by
  sorry

end florist_roses_l3652_365207


namespace frustum_small_cone_height_l3652_365234

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  height : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the height of the small cone removed from a frustum -/
def small_cone_height (f : Frustum) : ℝ :=
  f.height

/-- Theorem: The height of the small cone removed from a frustum with given dimensions is 30 cm -/
theorem frustum_small_cone_height :
  ∀ (f : Frustum),
    f.height = 30 ∧
    f.lower_base_area = 400 * Real.pi ∧
    f.upper_base_area = 100 * Real.pi →
    small_cone_height f = 30 := by
  sorry

end frustum_small_cone_height_l3652_365234


namespace problem_statement_l3652_365269

-- Define propositions p and q
def p : Prop := 2 + 2 = 5
def q : Prop := 3 > 2

-- Theorem stating the properties of p and q
theorem problem_statement :
  ¬p ∧ q ∧ (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬p ∧ ¬¬q := by sorry

end problem_statement_l3652_365269


namespace failed_implies_no_perfect_essay_l3652_365201

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (wrote_perfect_essay : Student → Prop)
variable (passed_course : Student → Prop)

-- Define the given condition
axiom perfect_essay_implies_pass :
  ∀ (s : Student), wrote_perfect_essay s → passed_course s

-- The statement to prove
theorem failed_implies_no_perfect_essay :
  ∀ (s : Student), ¬(passed_course s) → ¬(wrote_perfect_essay s) :=
sorry

end failed_implies_no_perfect_essay_l3652_365201


namespace medicine_container_problem_l3652_365223

theorem medicine_container_problem (initial_volume : ℝ) (remaining_volume : ℝ) : 
  initial_volume = 63 ∧ remaining_volume = 28 →
  ∃ (x : ℝ), x = 18 ∧ 
    initial_volume * (1 - x / initial_volume) * (1 - x / initial_volume) = remaining_volume :=
by sorry

end medicine_container_problem_l3652_365223


namespace triangle_count_for_2016_30_triangle_count_formula_l3652_365281

/-- Represents the number of non-overlapping triangles in a mesh region --/
def f (m n : ℕ) : ℕ := 2 * m - n - 2

/-- The theorem states that for 2016 points forming a 30-gon convex hull, 
    the number of non-overlapping triangles is 4000 --/
theorem triangle_count_for_2016_30 :
  f 2016 30 = 4000 := by
  sorry

/-- A more general theorem about the formula for f(m, n) --/
theorem triangle_count_formula {m n : ℕ} (h_m : m > 2) (h_n : 3 ≤ n ∧ n ≤ m) :
  f m n = 2 * m - n - 2 := by
  sorry

end triangle_count_for_2016_30_triangle_count_formula_l3652_365281


namespace negation_existence_absolute_value_l3652_365296

theorem negation_existence_absolute_value (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, |x| ≥ 1) :=
by sorry

end negation_existence_absolute_value_l3652_365296


namespace hyperbolas_M_value_l3652_365254

/-- Two hyperbolas with the same asymptotes -/
def hyperbolas_same_asymptotes (M : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧
  (∀ (x y : ℝ), x^2/9 - y^2/16 = 1 → y = k*x ∨ y = -k*x) ∧
  (∀ (x y : ℝ), y^2/25 - x^2/M = 1 → y = k*x ∨ y = -k*x)

/-- The theorem stating that M must equal 225/16 for the hyperbolas to have the same asymptotes -/
theorem hyperbolas_M_value :
  hyperbolas_same_asymptotes (225/16) ∧
  ∀ M : ℝ, hyperbolas_same_asymptotes M → M = 225/16 :=
sorry

end hyperbolas_M_value_l3652_365254


namespace vector_magnitude_equivalence_l3652_365239

/-- Given non-zero, non-collinear vectors a and b, prove that |a| = |b| if and only if |a + 2b| = |2a + b| -/
theorem vector_magnitude_equivalence {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] 
  (a b : n) (ha : a ≠ 0) (hb : b ≠ 0) (hnc : ¬ Collinear ℝ {0, a, b}) :
  ‖a‖ = ‖b‖ ↔ ‖a + 2 • b‖ = ‖2 • a + b‖ := by
  sorry

end vector_magnitude_equivalence_l3652_365239


namespace fish_tank_water_l3652_365250

theorem fish_tank_water (initial_water : ℝ) (added_water : ℝ) :
  initial_water = 7.75 →
  added_water = 7 →
  initial_water + added_water = 14.75 :=
by
  sorry

end fish_tank_water_l3652_365250


namespace construction_time_for_330_meters_l3652_365249

/-- Represents the daily progress of road construction in meters -/
def daily_progress : ℕ := 30

/-- Calculates the cumulative progress given the number of days -/
def cumulative_progress (days : ℕ) : ℕ :=
  daily_progress * days

/-- Theorem stating that 330 meters of cumulative progress corresponds to 11 days of construction -/
theorem construction_time_for_330_meters :
  cumulative_progress 11 = 330 ∧ cumulative_progress 10 ≠ 330 := by
  sorry

end construction_time_for_330_meters_l3652_365249


namespace expression_decrease_l3652_365231

theorem expression_decrease (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let x' := 0.9 * x
  let y' := 0.7 * y
  (x' ^ 2 * y' ^ 3) / (x ^ 2 * y ^ 3) = 0.27783 := by
sorry

end expression_decrease_l3652_365231


namespace compound_has_two_hydrogen_l3652_365265

/-- Represents a chemical compound with hydrogen, carbon, and oxygen atoms. -/
structure Compound where
  hydrogen : ℕ
  carbon : ℕ
  oxygen : ℕ
  molecular_weight : ℕ

/-- Atomic weights of elements in g/mol -/
def atomic_weight (element : String) : ℕ :=
  match element with
  | "H" => 1
  | "C" => 12
  | "O" => 16
  | _ => 0

/-- Calculates the molecular weight of a compound based on its composition -/
def calculate_weight (c : Compound) : ℕ :=
  c.hydrogen * atomic_weight "H" +
  c.carbon * atomic_weight "C" +
  c.oxygen * atomic_weight "O"

/-- Theorem stating that a compound with 1 Carbon, 3 Oxygen, and 62 g/mol molecular weight has 2 Hydrogen atoms -/
theorem compound_has_two_hydrogen :
  ∀ (c : Compound),
    c.carbon = 1 →
    c.oxygen = 3 →
    c.molecular_weight = 62 →
    calculate_weight c = c.molecular_weight →
    c.hydrogen = 2 :=
by
  sorry

end compound_has_two_hydrogen_l3652_365265


namespace intersection_A_B_l3652_365224

def A : Set ℕ := {1, 2, 3, 4, 5}

def B : Set ℕ := {x : ℕ | (x - 1) * (x - 4) < 0}

theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end intersection_A_B_l3652_365224


namespace midpoint_to_directrix_distance_l3652_365270

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define the point P
def point_P : ℝ × ℝ := (2, 0)

-- Define the directrix of parabola C
def directrix_C : ℝ := -3

-- Theorem statement
theorem midpoint_to_directrix_distance :
  ∃ (A B : ℝ × ℝ),
    parabola_C A.1 A.2 ∧
    parabola_C B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    (A.1 + B.1) / 2 - directrix_C = 11 :=
sorry

end midpoint_to_directrix_distance_l3652_365270


namespace power_sum_problem_l3652_365259

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 25)
  (h4 : a * x^4 + b * y^4 = 59) :
  a * x^5 + b * y^5 = 145 := by
  sorry

end power_sum_problem_l3652_365259


namespace inequality_implication_l3652_365267

theorem inequality_implication (a b c d : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) 
  (h : a + b * Real.sqrt 5 < c + d * Real.sqrt 5) : 
  a < c ∧ b < d := by sorry

end inequality_implication_l3652_365267


namespace original_triangle_area_l3652_365291

theorem original_triangle_area (original_area new_area : ℝ) : 
  (new_area = 32) → 
  (new_area = 4 * original_area) → 
  (original_area = 8) := by
sorry

end original_triangle_area_l3652_365291


namespace point_on_circle_l3652_365222

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Check if a point lies on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  squaredDistance p c.center = c.radius^2

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- The given point P(-3,4) -/
def pointP : Point := ⟨-3, 4⟩

/-- The circle with center at origin and radius 5 -/
def circleO : Circle := ⟨origin, 5⟩

theorem point_on_circle : isOnCircle pointP circleO := by
  sorry

end point_on_circle_l3652_365222


namespace birdseed_theorem_l3652_365203

/-- Calculates the amount of birdseed Peter needs to buy for a week -/
def birdseed_for_week : ℕ :=
  let parakeet_daily_consumption : ℕ := 2
  let parrot_daily_consumption : ℕ := 14
  let finch_daily_consumption : ℕ := parakeet_daily_consumption / 2
  let num_parakeets : ℕ := 3
  let num_parrots : ℕ := 2
  let num_finches : ℕ := 4
  let days_in_week : ℕ := 7
  
  let total_daily_consumption : ℕ := 
    num_parakeets * parakeet_daily_consumption +
    num_parrots * parrot_daily_consumption +
    num_finches * finch_daily_consumption

  total_daily_consumption * days_in_week

/-- Theorem stating that the amount of birdseed Peter needs to buy for a week is 266 grams -/
theorem birdseed_theorem : birdseed_for_week = 266 := by
  sorry

end birdseed_theorem_l3652_365203


namespace hexagons_from_circle_points_l3652_365238

/-- The number of points on the circle -/
def n : ℕ := 15

/-- The number of vertices in a hexagon -/
def k : ℕ := 6

/-- A function to calculate binomial coefficient -/
def binomial_coefficient (n k : ℕ) : ℕ := (Nat.choose n k)

/-- Theorem: The number of distinct convex hexagons formed from 15 points on a circle is 5005 -/
theorem hexagons_from_circle_points : binomial_coefficient n k = 5005 := by
  sorry

#eval binomial_coefficient n k  -- This should output 5005

end hexagons_from_circle_points_l3652_365238


namespace intersection_of_M_and_N_l3652_365244

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N :
  M ∩ N = {(3, -1)} := by sorry

end intersection_of_M_and_N_l3652_365244


namespace sum_of_specific_pair_l3652_365287

theorem sum_of_specific_pair : 
  ∃ (pairs : List (ℕ × ℕ)), 
    (pairs.length = 300) ∧ 
    (∀ (p : ℕ × ℕ), p ∈ pairs → 
      p.1 < 1500 ∧ p.2 < 1500 ∧ 
      p.2 = p.1 + 1 ∧ 
      (p.1 + p.2) % 5 = 0) ∧
    (57, 58) ∈ pairs →
    57 + 58 = 115 := by
  sorry

end sum_of_specific_pair_l3652_365287


namespace binomial_factorial_product_l3652_365292

theorem binomial_factorial_product : Nat.choose 20 6 * Nat.factorial 6 = 27907200 := by
  sorry

end binomial_factorial_product_l3652_365292


namespace inequality_proof_l3652_365236

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) :
  Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) ≤ Real.sqrt (x + y + z) := by
  sorry

end inequality_proof_l3652_365236


namespace f_is_decreasing_and_odd_l3652_365275

-- Define the function f(x) = -x
def f (x : ℝ) : ℝ := -x

-- State the theorem
theorem f_is_decreasing_and_odd :
  (∀ x y : ℝ, x < y → f x > f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end f_is_decreasing_and_odd_l3652_365275


namespace angle_A_value_range_of_b_squared_plus_c_squared_l3652_365293

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  (t.a + t.b) * (Real.sin t.A - Real.sin t.B) = t.c * (Real.sin t.C - Real.sin t.B)

-- Theorem 1: Prove that A = π/3
theorem angle_A_value (t : Triangle) (h : given_condition t) : t.A = π / 3 := by
  sorry

-- Theorem 2: Prove the range of b² + c² when a = 4
theorem range_of_b_squared_plus_c_squared (t : Triangle) (h1 : given_condition t) (h2 : t.a = 4) :
  16 < t.b^2 + t.c^2 ∧ t.b^2 + t.c^2 ≤ 32 := by
  sorry

end angle_A_value_range_of_b_squared_plus_c_squared_l3652_365293


namespace fraction_subtraction_l3652_365261

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end fraction_subtraction_l3652_365261


namespace mom_bought_71_packages_l3652_365272

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The total number of t-shirts Mom has -/
def total_shirts : ℕ := 426

/-- The number of packages Mom bought -/
def num_packages : ℕ := total_shirts / shirts_per_package

theorem mom_bought_71_packages : num_packages = 71 := by
  sorry

end mom_bought_71_packages_l3652_365272


namespace count_integers_satisfying_inequality_l3652_365205

theorem count_integers_satisfying_inequality : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, Real.sqrt (2 * n) ≤ Real.sqrt (5 * n - 8) ∧ 
               Real.sqrt (5 * n - 8) < Real.sqrt (3 * n + 7)) ∧
    S.card = 5 := by
  sorry

end count_integers_satisfying_inequality_l3652_365205


namespace election_theorem_l3652_365237

/-- Represents the number of boys in the club -/
def num_boys : ℕ := 15

/-- Represents the number of girls in the club -/
def num_girls : ℕ := 15

/-- Represents the total number of members in the club -/
def total_members : ℕ := num_boys + num_girls

/-- Calculates the number of ways to elect officials under the given constraints -/
def election_ways : ℕ := 2 * num_boys * num_girls * (num_boys - 1)

/-- Theorem stating the number of ways to elect officials -/
theorem election_theorem : election_ways = 6300 := by sorry

end election_theorem_l3652_365237


namespace quadratic_sum_of_coefficients_l3652_365242

/-- The quadratic function f(x) = -3x^2 + 24x - 45 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 24 * x - 45

/-- The same function in completed square form a(x+b)^2 + c -/
def g (x a b c : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum_of_coefficients :
  ∃ (a b c : ℝ), (∀ x, f x = g x a b c) ∧ (a + b + c = 4) := by sorry

end quadratic_sum_of_coefficients_l3652_365242


namespace freshman_class_size_l3652_365226

theorem freshman_class_size (N : ℕ) 
  (h1 : N > 0) 
  (h2 : 90 ≤ N) 
  (h3 : 100 ≤ N) :
  (90 : ℝ) / N * (20 : ℝ) / 100 = (20 : ℝ) / N → N = 450 := by
  sorry

end freshman_class_size_l3652_365226


namespace sixth_train_departure_l3652_365266

def train_departure_time (start_time : Nat) (interval : Nat) (n : Nat) : Nat :=
  start_time + (n - 1) * interval

theorem sixth_train_departure :
  let start_time := 10 * 60  -- 10:00 AM in minutes
  let interval := 30         -- 30 minutes
  let sixth_train := 6
  train_departure_time start_time interval sixth_train = 12 * 60 + 30  -- 12:30 PM in minutes
  := by sorry

end sixth_train_departure_l3652_365266


namespace bean_game_uniqueness_l3652_365221

/-- Represents the state of beans on an infinite row of squares -/
def BeanState := ℤ → ℕ

/-- Represents a single move in the bean game -/
def Move := ℤ

/-- Applies a move to a given state -/
def applyMove (state : BeanState) (move : Move) : BeanState :=
  sorry

/-- Checks if a state is terminal (no square has more than one bean) -/
def isTerminal (state : BeanState) : Prop :=
  ∀ i : ℤ, state i ≤ 1

/-- Represents a sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to an initial state -/
def applyMoveSequence (initial : BeanState) (moves : MoveSequence) : BeanState :=
  sorry

/-- The final state after applying a sequence of moves -/
def finalState (initial : BeanState) (moves : MoveSequence) : BeanState :=
  applyMoveSequence initial moves

/-- The number of steps (moves) in a sequence -/
def numSteps (moves : MoveSequence) : ℕ :=
  moves.length

/-- Theorem: All valid move sequences result in the same final state and number of steps -/
theorem bean_game_uniqueness (initial : BeanState) 
    (moves1 moves2 : MoveSequence) 
    (h1 : isTerminal (finalState initial moves1))
    (h2 : isTerminal (finalState initial moves2)) :
    finalState initial moves1 = finalState initial moves2 ∧ 
    numSteps moves1 = numSteps moves2 :=
  sorry

end bean_game_uniqueness_l3652_365221


namespace hyperbola_midpoint_l3652_365217

def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem hyperbola_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ ∧
    ¬∃ (x₁' y₁' x₂' y₂' : ℝ),
      hyperbola x₁' y₁' ∧
      hyperbola x₂' y₂' ∧
      (is_midpoint 1 1 x₁' y₁' x₂' y₂' ∨
       is_midpoint (-1) 2 x₁' y₁' x₂' y₂' ∨
       is_midpoint 1 3 x₁' y₁' x₂' y₂') :=
by sorry

end hyperbola_midpoint_l3652_365217


namespace quadratic_intersection_l3652_365271

theorem quadratic_intersection
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0) :
  ∃! p : ℝ × ℝ,
    (λ x y => y = a * x^2 + b * x + c) p.1 p.2 ∧
    (λ x y => y = a * x^2 - b * x + c + d) p.1 p.2 ∧
    p.1 ≠ 0 ∧ p.2 ≠ 0 :=
by sorry

end quadratic_intersection_l3652_365271


namespace orchid_bushes_planted_l3652_365279

/-- The number of orchid bushes planted in the park -/
theorem orchid_bushes_planted (initial : ℕ) (final : ℕ) (planted : ℕ) : 
  initial = 2 → final = 6 → planted = final - initial → planted = 4 := by
  sorry

end orchid_bushes_planted_l3652_365279


namespace equilateral_triangle_on_parabola_l3652_365211

/-- An equilateral triangle with one vertex at the origin and the other two on the parabola x^2 = 2y has side length 4√3. -/
theorem equilateral_triangle_on_parabola :
  ∃ (a : ℝ) (v1 v2 : ℝ × ℝ),
    a > 0 ∧
    v1.1^2 = 2 * v1.2 ∧
    v2.1^2 = 2 * v2.2 ∧
    (v1.1 - 0)^2 + (v1.2 - 0)^2 = a^2 ∧
    (v2.1 - 0)^2 + (v2.2 - 0)^2 = a^2 ∧
    (v2.1 - v1.1)^2 + (v2.2 - v1.2)^2 = a^2 ∧
    a = 4 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_on_parabola_l3652_365211


namespace sum_pairwise_ratios_lower_bound_l3652_365233

theorem sum_pairwise_ratios_lower_bound {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 := by
  sorry

end sum_pairwise_ratios_lower_bound_l3652_365233


namespace tenth_term_of_geometric_sequence_l3652_365251

/-- Given a geometric sequence with first term 3 and common ratio 5/2, 
    the tenth term is equal to 5859375/512. -/
theorem tenth_term_of_geometric_sequence : 
  let a : ℚ := 3
  let r : ℚ := 5/2
  let n : ℕ := 10
  let a_n := a * r^(n - 1)
  a_n = 5859375/512 := by sorry

end tenth_term_of_geometric_sequence_l3652_365251


namespace birds_meeting_point_l3652_365213

/-- The distance between West-town and East-town in kilometers -/
def total_distance : ℝ := 20

/-- The speed of the first bird in kilometers per minute -/
def speed_bird1 : ℝ := 4

/-- The speed of the second bird in kilometers per minute -/
def speed_bird2 : ℝ := 1

/-- The distance traveled by the first bird before meeting -/
def distance_bird1 : ℝ := 16

/-- The distance traveled by the second bird before meeting -/
def distance_bird2 : ℝ := 4

theorem birds_meeting_point :
  distance_bird1 + distance_bird2 = total_distance ∧
  distance_bird1 / speed_bird1 = distance_bird2 / speed_bird2 ∧
  distance_bird1 = 16 := by sorry

end birds_meeting_point_l3652_365213


namespace factor_36_minus_9x_squared_l3652_365264

theorem factor_36_minus_9x_squared (x : ℝ) : 36 - 9 * x^2 = 9 * (2 - x) * (2 + x) := by
  sorry

end factor_36_minus_9x_squared_l3652_365264


namespace sin_m_equals_cos_714_l3652_365229

theorem sin_m_equals_cos_714 (m : ℤ) :
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.cos (714 * π / 180) →
  m = 96 ∨ m = 84 := by
  sorry

end sin_m_equals_cos_714_l3652_365229


namespace quadratic_equation_1_l3652_365235

theorem quadratic_equation_1 : 
  ∃ x₁ x₂ : ℝ, x₁^2 - 2*x₁ - 99 = 0 ∧ x₂^2 - 2*x₂ - 99 = 0 ∧ x₁ ≠ x₂ :=
by sorry

end quadratic_equation_1_l3652_365235


namespace pascal_triangle_52nd_number_l3652_365245

/-- The number of elements in the row of Pascal's triangle we're considering --/
def row_length : ℕ := 55

/-- The index of the number we're looking for in the row (0-indexed) --/
def target_index : ℕ := 51

/-- The row number in Pascal's triangle (0-indexed) --/
def row_number : ℕ := row_length - 1

/-- The binomial coefficient we need to calculate --/
def pascal_number : ℕ := Nat.choose row_number target_index

theorem pascal_triangle_52nd_number : pascal_number = 24804 := by
  sorry

end pascal_triangle_52nd_number_l3652_365245


namespace greatest_integer_for_integer_fraction_l3652_365243

theorem greatest_integer_for_integer_fraction : 
  ∃ (x : ℤ), x = 53 ∧ 
  (∀ (y : ℤ), y > 53 → ¬(∃ (z : ℤ), (y^2 + 2*y + 13) / (y - 5) = z)) ∧
  (∃ (z : ℤ), (x^2 + 2*x + 13) / (x - 5) = z) :=
sorry

end greatest_integer_for_integer_fraction_l3652_365243


namespace harkamal_purchase_l3652_365258

/-- The total cost of a purchase given the quantity and price per unit -/
def totalCost (quantity : ℕ) (pricePerUnit : ℕ) : ℕ :=
  quantity * pricePerUnit

theorem harkamal_purchase : 
  let grapeQuantity : ℕ := 8
  let grapePrice : ℕ := 70
  let mangoQuantity : ℕ := 9
  let mangoPrice : ℕ := 60
  totalCost grapeQuantity grapePrice + totalCost mangoQuantity mangoPrice = 1100 := by
  sorry

end harkamal_purchase_l3652_365258


namespace divisibility_by_1947_l3652_365280

theorem divisibility_by_1947 (n : ℕ) : 
  (46 * 2^(n+1) + 296 * 13 * 2^(n+1)) % 1947 = 0 := by
sorry

end divisibility_by_1947_l3652_365280


namespace condition_necessary_not_sufficient_l3652_365220

theorem condition_necessary_not_sufficient (a b : ℝ) :
  (∃ a b : ℝ, (a > 0 ∨ b > 0) ∧ ¬(a + b > 0 ∧ a * b > 0)) ∧
  (∀ a b : ℝ, (a + b > 0 ∧ a * b > 0) → (a > 0 ∨ b > 0)) :=
by sorry

end condition_necessary_not_sufficient_l3652_365220


namespace function_inequality_l3652_365253

open Real

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧
  (∀ x, (x - 1) * (deriv f x - f x) > 0) ∧
  (∀ x, f (2 - x) = f x * Real.exp (2 - 2*x))

/-- The main theorem -/
theorem function_inequality (f : ℝ → ℝ) (h : satisfies_conditions f) :
  f 3 < Real.exp 3 * f 0 := by
  sorry

end function_inequality_l3652_365253


namespace quadratic_minimum_l3652_365256

/-- The quadratic function f(x) = x^2 - 8x + 18 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 18

theorem quadratic_minimum :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∃ (x_min : ℝ), f x_min = 2) ∧
  (∀ (x : ℝ), f x = 2 → x = 4) :=
by sorry

end quadratic_minimum_l3652_365256


namespace solution_set_when_a_is_2_range_of_a_l3652_365202

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of values for a
theorem range_of_a (h : ∀ x : ℝ, f x a ≥ 4) :
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end solution_set_when_a_is_2_range_of_a_l3652_365202


namespace teachers_daughter_age_l3652_365200

theorem teachers_daughter_age 
  (P : ℤ → ℤ)  -- P is a function from integers to integers
  (a : ℕ+)     -- a is a positive natural number
  (p : ℕ)      -- p is a natural number
  (h_poly : ∀ x y : ℤ, (x - y) ∣ (P x - P y))  -- P is a polynomial with integer coefficients
  (h_pa : P a = a)    -- P(a) = a
  (h_p0 : P 0 = p)    -- P(0) = p
  (h_prime : Nat.Prime p)  -- p is prime
  (h_p_gt_a : p > a)  -- p > a
  : a = 1 :=
by sorry

end teachers_daughter_age_l3652_365200


namespace inequality_count_l3652_365252

theorem inequality_count (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x^2 < a^2) (hyb : |y| < |b|) : 
  ∃! (n : ℕ), n = 2 ∧ 
  (n = (if x^2 + y^2 < a^2 + b^2 then 1 else 0) +
       (if x^2 - y^2 < a^2 - b^2 then 1 else 0) +
       (if x^2 * y^2 < a^2 * b^2 then 1 else 0) +
       (if x^2 / y^2 < a^2 / b^2 then 1 else 0)) :=
sorry

end inequality_count_l3652_365252


namespace wendys_cupcakes_l3652_365230

theorem wendys_cupcakes :
  ∀ (cupcakes cookies_baked pastries_left pastries_sold : ℕ),
    cookies_baked = 29 →
    pastries_left = 24 →
    pastries_sold = 9 →
    cupcakes + cookies_baked = pastries_left + pastries_sold →
    cupcakes = 4 := by
  sorry

end wendys_cupcakes_l3652_365230


namespace alvin_marbles_l3652_365290

def marble_game (initial : ℕ) (game1 : ℤ) (game2 : ℤ) (game3 : ℤ) (game4 : ℤ) (give : ℕ) (receive : ℕ) : ℕ :=
  (initial : ℤ) + game1 + game2 + game3 + game4 - give + receive |>.toNat

theorem alvin_marbles : 
  marble_game 57 (-18) 25 (-12) 15 10 8 = 65 := by
  sorry

end alvin_marbles_l3652_365290


namespace garage_sale_dvd_average_price_l3652_365283

/-- Calculate the average price of DVDs bought at a garage sale --/
theorem garage_sale_dvd_average_price : 
  let box1_count : ℕ := 10
  let box1_price : ℚ := 2
  let box2_count : ℕ := 5
  let box2_price : ℚ := 5
  let box3_count : ℕ := 3
  let box3_price : ℚ := 7
  let box4_count : ℕ := 4
  let box4_price : ℚ := 7/2
  let discount_rate : ℚ := 15/100
  let tax_rate : ℚ := 10/100
  let total_count : ℕ := box1_count + box2_count + box3_count + box4_count
  let total_cost : ℚ := 
    box1_count * box1_price + 
    box2_count * box2_price + 
    box3_count * box3_price + 
    box4_count * box4_price
  let discounted_cost : ℚ := total_cost * (1 - discount_rate)
  let final_cost : ℚ := discounted_cost * (1 + tax_rate)
  let average_price : ℚ := final_cost / total_count
  average_price = 17/5 := by sorry

end garage_sale_dvd_average_price_l3652_365283


namespace impossible_to_raise_average_l3652_365274

def current_scores : List ℝ := [82, 75, 88, 91, 78]
def max_score : ℝ := 100
def target_increase : ℝ := 5

theorem impossible_to_raise_average (scores : List ℝ) (max_score : ℝ) (target_increase : ℝ) :
  let current_avg := scores.sum / scores.length
  let new_sum := scores.sum + max_score
  let new_avg := new_sum / (scores.length + 1)
  new_avg < current_avg + target_increase :=
by sorry

end impossible_to_raise_average_l3652_365274


namespace angle4_is_60_l3652_365257

/-- Represents a quadrilateral with specific angle properties -/
structure SpecialQuadrilateral where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ
  sum_property : angle1 + angle2 + angle3 = 360
  equal_angles : angle3 = angle4 ∧ angle3 = angle5
  angle1_value : angle1 = 100
  angle2_value : angle2 = 80

/-- Theorem: In a SpecialQuadrilateral, angle4 equals 60 degrees -/
theorem angle4_is_60 (q : SpecialQuadrilateral) : q.angle4 = 60 := by
  sorry


end angle4_is_60_l3652_365257


namespace not_perfect_square_9999xxxx_l3652_365218

theorem not_perfect_square_9999xxxx : 
  ∀ n : ℕ, 99990000 ≤ n ∧ n ≤ 99999999 → ¬∃ m : ℕ, n = m * m := by
  sorry

end not_perfect_square_9999xxxx_l3652_365218


namespace line_tangent_to_parabola_l3652_365247

/-- The parabola y = 2x^2 -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2

/-- The point A(-1, 2) -/
def point_A : ℝ × ℝ := (-1, 2)

/-- The line l: 4x + y + 2 = 0 -/
def line_l (x y : ℝ) : Prop := 4 * x + y + 2 = 0

/-- Theorem stating that the line l passes through point A and is tangent to the parabola -/
theorem line_tangent_to_parabola :
  line_l point_A.1 point_A.2 ∧
  parabola point_A.1 point_A.2 ∧
  ∃ (t : ℝ), t ≠ point_A.1 ∧
    (∀ (x y : ℝ), x ≠ point_A.1 → line_l x y → parabola x y → x = t) :=
sorry

end line_tangent_to_parabola_l3652_365247


namespace square_difference_l3652_365225

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 45) (h2 : x * y = 10) :
  (x - y)^2 = 5 := by
  sorry

end square_difference_l3652_365225


namespace units_digit_3_pow_34_l3652_365260

def units_digit (n : ℕ) : ℕ := n % 10

def power_3_cycle (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur due to the modulo operation

theorem units_digit_3_pow_34 :
  units_digit (3^34) = 9 :=
by sorry

end units_digit_3_pow_34_l3652_365260


namespace fraction_equation_solution_l3652_365285

theorem fraction_equation_solution :
  ∃! x : ℚ, (x - 3) / (x + 2) + (3*x - 9) / (x - 3) = 2 ∧ x = 1/2 := by
sorry

end fraction_equation_solution_l3652_365285


namespace luncheon_seating_capacity_l3652_365276

theorem luncheon_seating_capacity 
  (invited : ℕ) 
  (no_shows : ℕ) 
  (tables : ℕ) 
  (h1 : invited = 47) 
  (h2 : no_shows = 7) 
  (h3 : tables = 8) :
  (invited - no_shows) / tables = 5 := by
  sorry

end luncheon_seating_capacity_l3652_365276


namespace composition_of_even_is_even_l3652_365289

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end composition_of_even_is_even_l3652_365289


namespace smallest_product_l3652_365228

def S : Finset Int := {-10, -4, 0, 2, 6}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int) (hx : x ∈ S) (hy : y ∈ S), x * y ≤ a * b ∧ x * y = -60 := by
  sorry

end smallest_product_l3652_365228


namespace triangle_area_bounds_l3652_365240

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 4

/-- The area of the triangle formed by the parabola and the line y = r -/
def triangleArea (r : ℝ) : ℝ := (r + 4)^(3/2)

/-- Theorem stating the relationship between r and the triangle area -/
theorem triangle_area_bounds (r : ℝ) :
  (16 ≤ triangleArea r ∧ triangleArea r ≤ 128) ↔ (8/3 ≤ r ∧ r ≤ 52/3) :=
sorry

end triangle_area_bounds_l3652_365240


namespace parallel_vectors_x_equals_9_l3652_365206

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, prove that if they are parallel, then x = 9 -/
theorem parallel_vectors_x_equals_9 (x : ℝ) :
  let a : ℝ × ℝ := (x, 3)
  let b : ℝ × ℝ := (3, 1)
  parallel a b → x = 9 := by
  sorry

end parallel_vectors_x_equals_9_l3652_365206


namespace increasing_order_x_z_y_l3652_365248

theorem increasing_order_x_z_y (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  x < x^(x^x) ∧ x^(x^x) < x^x := by
  sorry

end increasing_order_x_z_y_l3652_365248
