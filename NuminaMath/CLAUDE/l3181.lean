import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_a_l3181_318187

theorem solve_for_a (a b d : ℝ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3181_318187


namespace NUMINAMATH_CALUDE_marble_ratio_l3181_318136

/-- Represents the number of marbles of each color in a box -/
structure MarbleBox where
  red : ℕ
  green : ℕ
  yellow : ℕ
  other : ℕ

/-- Conditions for the marble box problem -/
def MarbleBoxConditions (box : MarbleBox) : Prop :=
  box.red = 20 ∧
  box.yellow = box.green / 5 ∧
  box.red + box.green + box.yellow + box.other = 3 * box.green ∧
  box.other = 88

theorem marble_ratio (box : MarbleBox) 
  (h : MarbleBoxConditions box) : 
  box.green = 3 * box.red := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l3181_318136


namespace NUMINAMATH_CALUDE_floor_breadth_correct_l3181_318118

/-- The length of the rectangular floor in meters -/
def floor_length : ℝ := 16.25

/-- The number of square tiles required to cover the floor -/
def number_of_tiles : ℕ := 3315

/-- The breadth of the rectangular floor in meters -/
def floor_breadth : ℝ := 204

/-- Theorem stating that the given breadth is correct for the rectangular floor -/
theorem floor_breadth_correct : 
  floor_length * floor_breadth = (number_of_tiles : ℝ) := by sorry

end NUMINAMATH_CALUDE_floor_breadth_correct_l3181_318118


namespace NUMINAMATH_CALUDE_stickers_on_last_page_l3181_318192

def total_books : Nat := 10
def pages_per_book : Nat := 30
def initial_stickers_per_page : Nat := 5
def new_stickers_per_page : Nat := 8
def full_books_after_rearrange : Nat := 6
def full_pages_in_seventh_book : Nat := 25

theorem stickers_on_last_page :
  let total_stickers := total_books * pages_per_book * initial_stickers_per_page
  let stickers_in_full_books := full_books_after_rearrange * pages_per_book * new_stickers_per_page
  let remaining_stickers := total_stickers - stickers_in_full_books
  let stickers_in_full_pages_of_seventh_book := (remaining_stickers / new_stickers_per_page) * new_stickers_per_page
  remaining_stickers - stickers_in_full_pages_of_seventh_book = 4 := by
  sorry

end NUMINAMATH_CALUDE_stickers_on_last_page_l3181_318192


namespace NUMINAMATH_CALUDE_root_implies_expression_value_l3181_318189

theorem root_implies_expression_value (a : ℝ) : 
  (1^2 - 5*a*1 + a^2 = 0) → (3*a^2 - 15*a - 7 = -10) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_expression_value_l3181_318189


namespace NUMINAMATH_CALUDE_shirt_price_l3181_318101

/-- Given a shirt and sweater with a total cost of $80.34, where the shirt costs $7.43 less than the sweater, the price of the shirt is $36.455. -/
theorem shirt_price (total_cost sweater_price shirt_price : ℝ) : 
  total_cost = 80.34 →
  sweater_price = shirt_price + 7.43 →
  total_cost = sweater_price + shirt_price →
  shirt_price = 36.455 := by sorry

end NUMINAMATH_CALUDE_shirt_price_l3181_318101


namespace NUMINAMATH_CALUDE_michael_crates_thursday_l3181_318152

/-- The number of crates Michael bought on Thursday -/
def crates_bought_thursday (initial_crates : ℕ) (crates_given : ℕ) (eggs_per_crate : ℕ) (final_eggs : ℕ) : ℕ :=
  (final_eggs - (initial_crates - crates_given) * eggs_per_crate) / eggs_per_crate

theorem michael_crates_thursday :
  crates_bought_thursday 6 2 30 270 = 5 := by
  sorry

end NUMINAMATH_CALUDE_michael_crates_thursday_l3181_318152


namespace NUMINAMATH_CALUDE_least_divisible_by_3_4_5_6_8_divisible_by_3_4_5_6_8_120_least_number_120_l3181_318173

theorem least_divisible_by_3_4_5_6_8 : ∀ n : ℕ, n > 0 → (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (8 ∣ n) → n ≥ 120 :=
by
  sorry

theorem divisible_by_3_4_5_6_8_120 : (3 ∣ 120) ∧ (4 ∣ 120) ∧ (5 ∣ 120) ∧ (6 ∣ 120) ∧ (8 ∣ 120) :=
by
  sorry

theorem least_number_120 : ∀ n : ℕ, n > 0 → (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (8 ∣ n) → n = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_3_4_5_6_8_divisible_by_3_4_5_6_8_120_least_number_120_l3181_318173


namespace NUMINAMATH_CALUDE_no_seven_edge_polyhedron_exists_polyhedron_with_2n_and_2n_plus_3_edges_l3181_318138

-- Define a convex polyhedron
structure ConvexPolyhedron where
  edges : ℕ
  is_convex : Bool

-- Theorem 1: A convex polyhedron cannot have exactly 7 edges
theorem no_seven_edge_polyhedron :
  ¬∃ (p : ConvexPolyhedron), p.edges = 7 ∧ p.is_convex = true :=
sorry

-- Theorem 2: For any integer n ≥ 3, there exists a convex polyhedron 
-- with 2n edges and another with 2n + 3 edges
theorem exists_polyhedron_with_2n_and_2n_plus_3_edges (n : ℕ) (h : n ≥ 3) :
  (∃ (p : ConvexPolyhedron), p.edges = 2 * n ∧ p.is_convex = true) ∧
  (∃ (q : ConvexPolyhedron), q.edges = 2 * n + 3 ∧ q.is_convex = true) :=
sorry

end NUMINAMATH_CALUDE_no_seven_edge_polyhedron_exists_polyhedron_with_2n_and_2n_plus_3_edges_l3181_318138


namespace NUMINAMATH_CALUDE_chair_arrangement_l3181_318168

theorem chair_arrangement (total_chairs : ℕ) (h : total_chairs = 10000) :
  ∃ (n : ℕ), n * n = total_chairs :=
sorry

end NUMINAMATH_CALUDE_chair_arrangement_l3181_318168


namespace NUMINAMATH_CALUDE_brown_family_seating_l3181_318195

/-- The number of ways to arrange boys and girls in a row --/
def seating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating the number of valid seating arrangements for 6 boys and 5 girls --/
theorem brown_family_seating :
  seating_arrangements 6 5 = 39830400 := by
  sorry

end NUMINAMATH_CALUDE_brown_family_seating_l3181_318195


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_two_disjoint_implies_a_leq_one_l3181_318104

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for case (1)
theorem subset_implies_a_geq_two (a : ℝ) : A ⊆ B a → a ≥ 2 := by
  sorry

-- Theorem for case (2)
theorem disjoint_implies_a_leq_one (a : ℝ) : A ∩ B a = ∅ → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_two_disjoint_implies_a_leq_one_l3181_318104


namespace NUMINAMATH_CALUDE_fraction_equality_l3181_318198

theorem fraction_equality (a b : ℝ) (h : (1/a + 1/b)/(1/a - 1/b) = 1009) :
  (a + b)/(a - b) = -1009 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3181_318198


namespace NUMINAMATH_CALUDE_square_times_square_minus_one_div_12_l3181_318146

theorem square_times_square_minus_one_div_12 (k : ℤ) : 
  12 ∣ (k^2 * (k^2 - 1)) := by sorry

end NUMINAMATH_CALUDE_square_times_square_minus_one_div_12_l3181_318146


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3181_318142

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 0 ≤ x}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3181_318142


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3181_318178

theorem geometric_sequence_fourth_term :
  ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- first term
  a 8 = 3888 →                         -- last term
  a 4 = 648 :=                         -- fourth term
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3181_318178


namespace NUMINAMATH_CALUDE_m_range_l3181_318153

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 - m) * x + 1 < (2 - m) * y + 1

-- Define the theorem
theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 
  1 < m ∧ m < 2 := by
  sorry


end NUMINAMATH_CALUDE_m_range_l3181_318153


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_l3181_318111

theorem no_solution_to_inequality (x : ℝ) :
  x ≥ -1/4 → ¬(-1 - 1/(3*x + 4) < 2) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_l3181_318111


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3181_318165

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x - x - 1 ≥ 0) ↔ (∃ x : ℝ, Real.exp x - x - 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3181_318165


namespace NUMINAMATH_CALUDE_inequality_proof_l3181_318143

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3181_318143


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3181_318174

theorem fraction_equivalence (b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∀ x : ℝ, x ≠ -c → x ≠ -3*c → (x + 2*b) / (x + 3*c) = (x + b) / (x + c)) ↔ b = 2*c :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3181_318174


namespace NUMINAMATH_CALUDE_total_cost_one_large_three_small_l3181_318109

/-- The cost of a large puzzle, in dollars -/
def large_puzzle_cost : ℕ := 15

/-- The cost of a small puzzle and a large puzzle together, in dollars -/
def combined_cost : ℕ := 23

/-- The cost of a small puzzle, in dollars -/
def small_puzzle_cost : ℕ := combined_cost - large_puzzle_cost

theorem total_cost_one_large_three_small :
  large_puzzle_cost + 3 * small_puzzle_cost = 39 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_one_large_three_small_l3181_318109


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_answer_is_valid_l3181_318164

def is_valid_number (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), 
    digits.length = 10 ∧ 
    digits.toFinset = Finset.range 10 ∧
    n = digits.foldl (λ acc d => acc * 10 + d) 0

def is_multiple_of_12 (n : ℕ) : Prop :=
  n % 12 = 0

theorem largest_multiple_of_12 :
  ∀ n : ℕ, 
    is_valid_number n → 
    is_multiple_of_12 n → 
    n ≤ 9876543120 :=
by sorry

theorem answer_is_valid :
  is_valid_number 9876543120 ∧ is_multiple_of_12 9876543120 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_answer_is_valid_l3181_318164


namespace NUMINAMATH_CALUDE_junior_score_l3181_318180

theorem junior_score (n : ℝ) (junior_ratio : ℝ) (senior_ratio : ℝ) 
  (class_avg : ℝ) (senior_avg : ℝ) (junior_score : ℝ) :
  junior_ratio = 0.2 →
  senior_ratio = 0.8 →
  junior_ratio + senior_ratio = 1 →
  class_avg = 75 →
  senior_avg = 72 →
  class_avg * n = senior_avg * (senior_ratio * n) + junior_score * (junior_ratio * n) →
  junior_score = 87 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l3181_318180


namespace NUMINAMATH_CALUDE_point_on_line_with_equal_distances_l3181_318126

theorem point_on_line_with_equal_distances (P : ℝ × ℝ) :
  P.1 + 3 * P.2 = 0 →
  (P.1^2 + P.2^2).sqrt = |P.1 + 3 * P.2 - 2| / (1^2 + 3^2).sqrt →
  (P = (3/5, -1/5) ∨ P = (-3/5, 1/5)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_with_equal_distances_l3181_318126


namespace NUMINAMATH_CALUDE_selling_price_is_14_l3181_318140

/-- Calculates the selling price per bowl given the following conditions:
    * Total number of glass bowls bought: 110
    * Cost per bowl: Rs. 10
    * Number of bowls sold: 100
    * Number of bowls broken: 10 (remaining)
    * Percentage gain: 27.27272727272727%
-/
def calculate_selling_price_per_bowl (total_bowls : ℕ) (cost_per_bowl : ℚ) 
  (bowls_sold : ℕ) (percentage_gain : ℚ) : ℚ :=
  let total_cost : ℚ := total_bowls * cost_per_bowl
  let gain : ℚ := percentage_gain * total_cost
  let total_selling_price : ℚ := total_cost + gain
  total_selling_price / bowls_sold

/-- Theorem stating that the selling price per bowl is 14 given the problem conditions -/
theorem selling_price_is_14 :
  calculate_selling_price_per_bowl 110 10 100 (27.27272727272727 / 100) = 14 := by
  sorry

#eval calculate_selling_price_per_bowl 110 10 100 (27.27272727272727 / 100)

end NUMINAMATH_CALUDE_selling_price_is_14_l3181_318140


namespace NUMINAMATH_CALUDE_shaded_fraction_is_five_thirty_sixths_l3181_318179

/-- Represents a square quilt with a 3x3 grid of unit squares -/
structure Quilt :=
  (size : ℕ := 3)
  (total_area : ℚ := 9)

/-- Calculates the shaded area of the quilt -/
def shaded_area (q : Quilt) : ℚ :=
  let triangle_area : ℚ := 1/2
  let small_square_area : ℚ := 1/4
  let full_square_area : ℚ := 1
  2 * triangle_area + small_square_area + full_square_area

/-- Theorem stating that the shaded area is 5/36 of the total area -/
theorem shaded_fraction_is_five_thirty_sixths (q : Quilt) :
  shaded_area q / q.total_area = 5/36 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_is_five_thirty_sixths_l3181_318179


namespace NUMINAMATH_CALUDE_senegal_total_points_l3181_318141

-- Define the point values for victory and draw
def victory_points : ℕ := 3
def draw_points : ℕ := 1

-- Define Senegal's match results
def senegal_victories : ℕ := 1
def senegal_draws : ℕ := 2

-- Define the function to calculate total points
def calculate_points (victories draws : ℕ) : ℕ :=
  victories * victory_points + draws * draw_points

-- Theorem to prove
theorem senegal_total_points :
  calculate_points senegal_victories senegal_draws = 5 := by
  sorry

end NUMINAMATH_CALUDE_senegal_total_points_l3181_318141


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3181_318147

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 393000

/-- The scientific notation representation of the number -/
def scientific_form : ScientificNotation :=
  { coefficient := 3.93
    exponent := 5
    coefficient_range := by sorry }

/-- Theorem stating that the scientific notation is correct -/
theorem scientific_notation_correct :
  (scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent) = number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l3181_318147


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3181_318197

theorem rectangle_perimeter (z w : ℝ) (hz : z > 0) (hw : w > 0) (h : w < z) :
  let l := z - w
  2 * (l + w) = 2 * z :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3181_318197


namespace NUMINAMATH_CALUDE_pool_filling_rate_prove_pool_filling_rate_l3181_318117

/-- Proves that the rate of filling the pool during the second and third hours is 10 gallons/hour -/
theorem pool_filling_rate : ℝ → Prop :=
  fun (R : ℝ) ↦
    (8 : ℝ) +         -- Water added in 1st hour
    (R * 2) +         -- Water added in 2nd and 3rd hours
    (14 : ℝ) -        -- Water added in 4th hour
    (8 : ℝ) =         -- Water lost in 5th hour
    (34 : ℝ) →        -- Total water after 5 hours
    R = (10 : ℝ)      -- Rate during 2nd and 3rd hours

/-- Proof of the theorem -/
theorem prove_pool_filling_rate : pool_filling_rate (10 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_rate_prove_pool_filling_rate_l3181_318117


namespace NUMINAMATH_CALUDE_grass_cutting_cost_l3181_318185

/-- The cost of cutting grass once, given specific growth and cost conditions --/
theorem grass_cutting_cost
  (initial_height : ℝ)
  (growth_rate : ℝ)
  (cut_threshold : ℝ)
  (annual_cost : ℝ)
  (h1 : initial_height = 2)
  (h2 : growth_rate = 0.5)
  (h3 : cut_threshold = 4)
  (h4 : annual_cost = 300)
  : (annual_cost / (12 / ((cut_threshold - initial_height) / growth_rate))) = 100 := by
  sorry

end NUMINAMATH_CALUDE_grass_cutting_cost_l3181_318185


namespace NUMINAMATH_CALUDE_T_property_M_remainder_l3181_318199

/-- A sequence of positive integers where each number has exactly 9 ones in its binary representation -/
def T : ℕ → ℕ := sorry

/-- The 500th number in the sequence T -/
def M : ℕ := T 500

/-- Predicate to check if a natural number has exactly 9 ones in its binary representation -/
def has_nine_ones (n : ℕ) : Prop := sorry

theorem T_property (n : ℕ) : has_nine_ones (T n) := sorry

theorem M_remainder : M % 500 = 191 := sorry

end NUMINAMATH_CALUDE_T_property_M_remainder_l3181_318199


namespace NUMINAMATH_CALUDE_dinner_tip_calculation_l3181_318148

/-- Calculates the individual tip amount for a group dinner -/
theorem dinner_tip_calculation (julie_order : ℚ) (letitia_order : ℚ) (anton_order : ℚ) 
  (tip_percentage : ℚ) (num_people : ℕ) : 
  julie_order = 10 ∧ letitia_order = 20 ∧ anton_order = 30 ∧ 
  tip_percentage = 1/5 ∧ num_people = 3 →
  (julie_order + letitia_order + anton_order) * tip_percentage / num_people = 4 := by
  sorry

#check dinner_tip_calculation

end NUMINAMATH_CALUDE_dinner_tip_calculation_l3181_318148


namespace NUMINAMATH_CALUDE_pancakes_and_honey_cost_l3181_318114

theorem pancakes_and_honey_cost (x y : ℕ) : 25 * x + 340 * y ≤ 2000 :=
by sorry

end NUMINAMATH_CALUDE_pancakes_and_honey_cost_l3181_318114


namespace NUMINAMATH_CALUDE_existence_of_even_and_odd_composite_functions_l3181_318100

theorem existence_of_even_and_odd_composite_functions :
  ∃ (p q : ℝ → ℝ),
    (∀ x, p (-x) = p x) ∧
    (∀ x, p (q (-x)) = -(p (q x))) ∧
    (∃ x, p (q x) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_even_and_odd_composite_functions_l3181_318100


namespace NUMINAMATH_CALUDE_multiplicative_inverse_of_110_mod_667_l3181_318121

-- Define the triangle
def leg1 : ℕ := 65
def leg2 : ℕ := 156
def hypotenuse : ℕ := 169

-- Define the relation C = A + B
def relation (A B C : ℕ) : Prop := C = A + B

-- Define the modulus
def modulus : ℕ := 667

-- Define the number we're finding the inverse for
def num : ℕ := 110

-- Theorem statement
theorem multiplicative_inverse_of_110_mod_667 :
  (∃ (A B : ℕ), relation A B hypotenuse ∧ leg1^2 + leg2^2 = hypotenuse^2) →
  ∃ (n : ℕ), n < modulus ∧ (num * n) % modulus = 1 ∧ n = 608 :=
by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_of_110_mod_667_l3181_318121


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3181_318106

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ),
  (-3 * x^2 + 24 * x + 81 = a * (x + b)^2 + c) ∧ (a + b + c = 122) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3181_318106


namespace NUMINAMATH_CALUDE_complex_square_in_second_quadrant_l3181_318132

theorem complex_square_in_second_quadrant :
  let z : ℂ := (1/2 : ℝ) + (Real.sqrt 3/2 : ℝ) * Complex.I
  let w : ℂ := z^2
  (w.re < 0) ∧ (w.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_square_in_second_quadrant_l3181_318132


namespace NUMINAMATH_CALUDE_remaining_grass_area_l3181_318188

-- Define the plot and path characteristics
def plot_diameter : ℝ := 20
def path_width : ℝ := 4

-- Define the theorem
theorem remaining_grass_area :
  let plot_radius : ℝ := plot_diameter / 2
  let effective_radius : ℝ := plot_radius - path_width / 2
  (π * effective_radius^2 : ℝ) = 64 * π := by sorry

end NUMINAMATH_CALUDE_remaining_grass_area_l3181_318188


namespace NUMINAMATH_CALUDE_sector_area_l3181_318105

/-- Given a sector with central angle 2 radians and arc length 2, its area is 1. -/
theorem sector_area (θ : Real) (l : Real) (r : Real) : 
  θ = 2 → l = 2 → l = r * θ → (1/2) * r * θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3181_318105


namespace NUMINAMATH_CALUDE_sqrt_of_one_plus_three_l3181_318181

theorem sqrt_of_one_plus_three : Real.sqrt (1 + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_one_plus_three_l3181_318181


namespace NUMINAMATH_CALUDE_lcm_gcd_220_126_l3181_318125

theorem lcm_gcd_220_126 :
  (Nat.lcm 220 126 = 13860) ∧ (Nat.gcd 220 126 = 2) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_220_126_l3181_318125


namespace NUMINAMATH_CALUDE_value_of_M_l3181_318183

theorem value_of_M : ∃ M : ℝ, (0.2 * M = 0.5 * 1000) ∧ (M = 2500) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l3181_318183


namespace NUMINAMATH_CALUDE_smallest_solution_sum_is_five_l3181_318160

/-- The sum of divisors function for numbers of the form 2^i * 3^j * 5^k -/
def sum_of_divisors (i j k : ℕ) : ℕ :=
  (2^(i+1) - 1) * ((3^(j+1) - 1)/2) * ((5^(k+1) - 1)/4)

/-- Predicate to check if (i, j, k) is a valid solution -/
def is_valid_solution (i j k : ℕ) : Prop :=
  sum_of_divisors i j k = 360

/-- Predicate to check if (i, j, k) is the smallest valid solution -/
def is_smallest_solution (i j k : ℕ) : Prop :=
  is_valid_solution i j k ∧
  ∀ i' j' k', is_valid_solution i' j' k' → i + j + k ≤ i' + j' + k'

/-- The main theorem: the smallest solution sums to 5 -/
theorem smallest_solution_sum_is_five :
  ∃ i j k, is_smallest_solution i j k ∧ i + j + k = 5 := by sorry

#check smallest_solution_sum_is_five

end NUMINAMATH_CALUDE_smallest_solution_sum_is_five_l3181_318160


namespace NUMINAMATH_CALUDE_steel_experiment_golden_ratio_l3181_318151

/-- The 0.618 method calculation for a given range -/
def golden_ratio_method (lower_bound upper_bound : ℝ) : ℝ :=
  lower_bound + (upper_bound - lower_bound) * 0.618

/-- Theorem: The 0.618 method for the given steel experiment -/
theorem steel_experiment_golden_ratio :
  let lower_bound : ℝ := 500
  let upper_bound : ℝ := 1000
  golden_ratio_method lower_bound upper_bound = 809 := by
  sorry

end NUMINAMATH_CALUDE_steel_experiment_golden_ratio_l3181_318151


namespace NUMINAMATH_CALUDE_complex_equality_l3181_318120

theorem complex_equality (z : ℂ) : z = -1 + I ↔ Complex.abs (z - 2) = Complex.abs (z + 4) ∧ Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l3181_318120


namespace NUMINAMATH_CALUDE_max_difference_of_constrained_integers_l3181_318162

theorem max_difference_of_constrained_integers : 
  ∃ (P Q : ℤ), 
    (∃ (x : ℤ), x^2 ≤ 729 ∧ 729 ≤ -x^3 ∧ (x = P ∨ x = Q)) ∧
    (∀ (R S : ℤ), (∃ (y : ℤ), y^2 ≤ 729 ∧ 729 ≤ -y^3 ∧ (y = R ∨ y = S)) → 
      10 * (P - Q) ≥ 10 * (R - S)) ∧
    10 * (P - Q) = 180 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_of_constrained_integers_l3181_318162


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l3181_318191

/-- Proves that the initial ratio of milk to water is 3:2 given the conditions -/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (new_ratio_milk : ℝ) 
  (new_ratio_water : ℝ) 
  (h1 : total_volume = 155) 
  (h2 : added_water = 62) 
  (h3 : new_ratio_milk = 3) 
  (h4 : new_ratio_water = 4) : 
  ∃ (initial_milk initial_water : ℝ), 
    initial_milk + initial_water = total_volume ∧ 
    initial_milk / (initial_water + added_water) = new_ratio_milk / new_ratio_water ∧
    initial_milk / initial_water = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l3181_318191


namespace NUMINAMATH_CALUDE_factorization_equality_l3181_318193

theorem factorization_equality (x y : ℝ) :
  -12 * x * y^2 * (x + y) + 18 * x^2 * y * (x + y) = 6 * x * y * (x + y) * (3 * x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3181_318193


namespace NUMINAMATH_CALUDE_greatest_common_remainder_l3181_318175

theorem greatest_common_remainder (a b c : ℕ) (h1 : a = 41) (h2 : b = 71) (h3 : c = 113) :
  ∃ (d : ℕ), d > 0 ∧ 
  (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
  (∀ (k : ℕ), k > 0 → (∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s) → k ≤ d) ∧
  d = Nat.gcd (b - a) (Nat.gcd (c - b) (c - a)) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_remainder_l3181_318175


namespace NUMINAMATH_CALUDE_tree_height_calculation_l3181_318176

/-- Given a flagpole and a tree, calculate the height of the tree using similar triangles -/
theorem tree_height_calculation (flagpole_height flagpole_shadow tree_shadow : ℝ) 
  (h1 : flagpole_height = 4)
  (h2 : flagpole_shadow = 6)
  (h3 : tree_shadow = 12) :
  (flagpole_height / flagpole_shadow) * tree_shadow = 8 :=
by sorry

end NUMINAMATH_CALUDE_tree_height_calculation_l3181_318176


namespace NUMINAMATH_CALUDE_equation_solutions_l3181_318159

theorem equation_solutions : 
  {(x, y) : ℕ × ℕ | x > 0 ∧ y > 0 ∧ 2*x^2 + 5*x*y + 2*y^2 = 2006} = 
  {(28, 3), (3, 28)} :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3181_318159


namespace NUMINAMATH_CALUDE_cone_height_l3181_318150

theorem cone_height (s : ℝ) (a : ℝ) (h : s = 13 ∧ a = 65 * Real.pi) :
  Real.sqrt (s^2 - (a / (s * Real.pi))^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l3181_318150


namespace NUMINAMATH_CALUDE_worksheets_graded_l3181_318107

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 16 →
  problems_per_worksheet = 4 →
  problems_left = 32 →
  total_worksheets * problems_per_worksheet - problems_left = 8 * problems_per_worksheet :=
by sorry

end NUMINAMATH_CALUDE_worksheets_graded_l3181_318107


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l3181_318186

def fair_coin_probability : ℚ := 1 / 2
def fair_die_probability : ℚ := 1 / 6

theorem coin_and_die_probability :
  let p_tails := fair_coin_probability
  let p_one_or_two := 2 * fair_die_probability
  p_tails * p_one_or_two = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l3181_318186


namespace NUMINAMATH_CALUDE_monotone_increasing_iff_a_in_range_l3181_318108

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1/2 then a^x else (2*a - 1)*x

theorem monotone_increasing_iff_a_in_range (a : ℝ) :
  Monotone (f a) ↔ a ∈ Set.Ici ((2 + Real.sqrt 3) / 2) :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_iff_a_in_range_l3181_318108


namespace NUMINAMATH_CALUDE_total_arrangements_l3181_318170

/-- The number of ways to arrange 3 events in 4 venues with at most 2 events per venue -/
def arrangeEvents : ℕ := sorry

/-- The total number of arrangements is 60 -/
theorem total_arrangements : arrangeEvents = 60 := by sorry

end NUMINAMATH_CALUDE_total_arrangements_l3181_318170


namespace NUMINAMATH_CALUDE_correct_bouquet_flowers_l3181_318137

def flowers_for_bouquets (tulips roses extra : ℕ) : ℕ :=
  tulips + roses - extra

theorem correct_bouquet_flowers :
  flowers_for_bouquets 39 49 7 = 81 := by
  sorry

end NUMINAMATH_CALUDE_correct_bouquet_flowers_l3181_318137


namespace NUMINAMATH_CALUDE_positive_number_square_sum_l3181_318172

theorem positive_number_square_sum (n : ℝ) : n > 0 ∧ n^2 + n = 210 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_square_sum_l3181_318172


namespace NUMINAMATH_CALUDE_find_t_l3181_318155

-- Define variables
variable (t : ℝ)

-- Define functions for hours worked and hourly rates
def my_hours : ℝ := t + 2
def my_rate : ℝ := 4*t - 4
def bob_hours : ℝ := 4*t - 7
def bob_rate : ℝ := t + 3

-- State the theorem
theorem find_t : 
  my_hours * my_rate = bob_hours * bob_rate + 3 → t = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_t_l3181_318155


namespace NUMINAMATH_CALUDE_jorge_corn_yield_l3181_318158

/-- Calculates the total corn yield from Jorge's land -/
theorem jorge_corn_yield (total_land : ℝ) (good_soil_yield : ℝ) 
  (clay_soil_fraction : ℝ) (h1 : total_land = 60) 
  (h2 : good_soil_yield = 400) (h3 : clay_soil_fraction = 1/3) : 
  total_land * (clay_soil_fraction * (good_soil_yield / 2) + 
  (1 - clay_soil_fraction) * good_soil_yield) = 20000 := by
  sorry

#check jorge_corn_yield

end NUMINAMATH_CALUDE_jorge_corn_yield_l3181_318158


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3181_318149

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 24538

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation := {
  coefficient := 2.4538,
  exponent := 4,
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3181_318149


namespace NUMINAMATH_CALUDE_unmeasurable_weights_theorem_l3181_318161

def available_weights : List Nat := [1, 2, 3, 8, 16, 32]

def is_measurable (n : Nat) (weights : List Nat) : Prop :=
  ∃ (subset : List Nat), subset.Sublist weights ∧ subset.sum = n

def unmeasurable_weights : Set Nat :=
  {n | n ≤ 60 ∧ ¬(is_measurable n available_weights)}

theorem unmeasurable_weights_theorem :
  unmeasurable_weights = {7, 15, 23, 31, 39, 47, 55} := by
  sorry

end NUMINAMATH_CALUDE_unmeasurable_weights_theorem_l3181_318161


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3181_318116

/-- A right triangle with side lengths a, b, and c (a < b < c) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  a_lt_b : a < b
  b_lt_c : b < c
  pythagoras : a^2 + b^2 = c^2

/-- The condition a:b:c = 3:4:5 -/
def is_345_ratio (t : RightTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k

/-- The condition that a, b, c form an arithmetic progression -/
def is_arithmetic_progression (t : RightTriangle) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ t.b - t.a = d ∧ t.c - t.b = d

theorem sufficient_not_necessary :
  (∀ t : RightTriangle, is_345_ratio t → is_arithmetic_progression t) ∧
  (∃ t : RightTriangle, is_arithmetic_progression t ∧ ¬is_345_ratio t) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3181_318116


namespace NUMINAMATH_CALUDE_log11_not_expressible_l3181_318133

-- Define the given logarithmic values
def log5 : ℝ := 0.6990
def log6 : ℝ := 0.7782

-- Define a type for basic logarithmic expressions
inductive LogExpr
| Const : ℝ → LogExpr
| Log5 : LogExpr
| Log6 : LogExpr
| Add : LogExpr → LogExpr → LogExpr
| Sub : LogExpr → LogExpr → LogExpr
| Mul : ℝ → LogExpr → LogExpr

-- Function to evaluate a LogExpr
def eval : LogExpr → ℝ
| LogExpr.Const r => r
| LogExpr.Log5 => log5
| LogExpr.Log6 => log6
| LogExpr.Add e1 e2 => eval e1 + eval e2
| LogExpr.Sub e1 e2 => eval e1 - eval e2
| LogExpr.Mul r e => r * eval e

-- Theorem stating that log 11 cannot be expressed using log 5 and log 6
theorem log11_not_expressible : ∀ e : LogExpr, eval e ≠ Real.log 11 := by
  sorry

end NUMINAMATH_CALUDE_log11_not_expressible_l3181_318133


namespace NUMINAMATH_CALUDE_sum_of_bases_l3181_318128

/-- Given two bases R₁ and R₂, and two fractions F₁ and F₂, prove that R₁ + R₂ = 21 -/
theorem sum_of_bases (R₁ R₂ : ℕ) (F₁ F₂ : ℚ) : R₁ + R₂ = 21 :=
  by
  have h1 : F₁ = (4 * R₁ + 7) / (R₁^2 - 1) := by sorry
  have h2 : F₂ = (7 * R₁ + 4) / (R₁^2 - 1) := by sorry
  have h3 : F₁ = (R₂ + 6) / (R₂^2 - 1) := by sorry
  have h4 : F₂ = (6 * R₂ + 1) / (R₂^2 - 1) := by sorry
  sorry

end NUMINAMATH_CALUDE_sum_of_bases_l3181_318128


namespace NUMINAMATH_CALUDE_fraction_product_equality_l3181_318139

theorem fraction_product_equality : (3 + 5 + 7) / (2 + 4 + 6) * (4 + 8 + 12) / (1 + 3 + 5) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l3181_318139


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l3181_318131

theorem quadratic_negative_root (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ x^2 - 6*a*x - 2 + 2*a + 9*a^2 = 0) ↔ 
  a < (-1 + Real.sqrt 19) / 9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l3181_318131


namespace NUMINAMATH_CALUDE_business_investment_l3181_318177

theorem business_investment (p q : ℕ) (h1 : q = 15000) (h2 : p / q = 4) : p = 60000 := by
  sorry

end NUMINAMATH_CALUDE_business_investment_l3181_318177


namespace NUMINAMATH_CALUDE_parabola_translation_l3181_318171

/-- A parabola defined by a quadratic function -/
def Parabola (a b c : ℝ) := fun (x : ℝ) => a * x^2 + b * x + c

/-- Translation of a function -/
def translate (f : ℝ → ℝ) (dx dy : ℝ) := fun (x : ℝ) => f (x - dx) + dy

theorem parabola_translation (x : ℝ) :
  translate (Parabola 2 4 1) 1 3 x = Parabola 2 0 0 x := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l3181_318171


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3181_318103

theorem equation_solution_exists : ∃ x : ℝ, 
  (0.76 : ℝ)^3 - (0.1 : ℝ)^3 / (0.76 : ℝ)^2 + x + (0.1 : ℝ)^2 = 0.66 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3181_318103


namespace NUMINAMATH_CALUDE_mary_received_more_than_mike_l3181_318196

/-- Represents the profit distribution in a partnership --/
def profit_distribution (mary_investment mike_investment total_profit : ℚ) : ℚ :=
  let equal_share := (1/3) * total_profit / 2
  let ratio_share := (2/3) * total_profit
  let mary_ratio := mary_investment / (mary_investment + mike_investment)
  let mike_ratio := mike_investment / (mary_investment + mike_investment)
  let mary_total := equal_share + mary_ratio * ratio_share
  let mike_total := equal_share + mike_ratio * ratio_share
  mary_total - mike_total

/-- Theorem stating that Mary received $800 more than Mike --/
theorem mary_received_more_than_mike :
  profit_distribution 700 300 3000 = 800 := by
  sorry


end NUMINAMATH_CALUDE_mary_received_more_than_mike_l3181_318196


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3181_318182

theorem quadratic_transformation (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (h k r : ℝ) (hr : r ≠ 0), ∀ x : ℝ,
    a * x^2 + b * x + c = a * ((x - h)^2 / r^2) + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3181_318182


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_1320_l3181_318110

def sum_of_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_of_prime_factors_1320 :
  sum_of_prime_factors 1320 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_1320_l3181_318110


namespace NUMINAMATH_CALUDE_team_not_losing_probability_l3181_318194

/-- Represents the positions Player A can play -/
inductive Position
| CenterForward
| Winger
| AttackingMidfielder

/-- The appearance rate for each position -/
def appearanceRate (pos : Position) : ℝ :=
  match pos with
  | .CenterForward => 0.3
  | .Winger => 0.5
  | .AttackingMidfielder => 0.2

/-- The probability of the team losing when Player A plays in each position -/
def losingProbability (pos : Position) : ℝ :=
  match pos with
  | .CenterForward => 0.3
  | .Winger => 0.2
  | .AttackingMidfielder => 0.2

/-- The probability of the team not losing when Player A participates -/
def teamNotLosingProbability : ℝ :=
  (appearanceRate Position.CenterForward * (1 - losingProbability Position.CenterForward)) +
  (appearanceRate Position.Winger * (1 - losingProbability Position.Winger)) +
  (appearanceRate Position.AttackingMidfielder * (1 - losingProbability Position.AttackingMidfielder))

theorem team_not_losing_probability :
  teamNotLosingProbability = 0.77 := by
  sorry

end NUMINAMATH_CALUDE_team_not_losing_probability_l3181_318194


namespace NUMINAMATH_CALUDE_income_data_correction_l3181_318130

theorem income_data_correction (T : ℝ) : 
  let num_families : ℕ := 1200
  let largest_correct_income : ℝ := 102000
  let largest_incorrect_income : ℝ := 1020000
  let processing_fee : ℝ := 500
  let corrected_mean := (T + (largest_correct_income - processing_fee)) / num_families
  let incorrect_mean := (T + (largest_incorrect_income - processing_fee)) / num_families
  incorrect_mean - corrected_mean = 765 := by
sorry

end NUMINAMATH_CALUDE_income_data_correction_l3181_318130


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_l3181_318166

theorem smallest_x_absolute_value (x : ℝ) : 
  (|x - 10| = 15) → (x ≥ -5 ∧ (∃ y : ℝ, |y - 10| = 15 ∧ y = -5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_l3181_318166


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3181_318154

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 9 / b) ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3181_318154


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_plane_parallel_plane_parallel_line_in_plane_implies_line_parallel_plane_l3181_318144

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Axioms
axiom distinct_lines (m n : Line) : m ≠ n
axiom non_coincident_planes (α β : Plane) : α ≠ β

-- Theorem 1
theorem perpendicular_parallel_implies_plane_parallel 
  (m n : Line) (α β : Plane) :
  perpendicular m α → perpendicular n β → parallel m n → 
  plane_parallel α β :=
sorry

-- Theorem 2
theorem plane_parallel_line_in_plane_implies_line_parallel_plane 
  (m : Line) (α β : Plane) :
  plane_parallel α β → contains α m → line_parallel_plane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_plane_parallel_plane_parallel_line_in_plane_implies_line_parallel_plane_l3181_318144


namespace NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l3181_318135

theorem a_positive_sufficient_not_necessary_for_abs_a_positive :
  (∃ a : ℝ, (a > 0 → abs a > 0) ∧ ¬(abs a > 0 → a > 0)) := by
  sorry

end NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l3181_318135


namespace NUMINAMATH_CALUDE_negative_four_squared_times_negative_one_power_2022_l3181_318156

theorem negative_four_squared_times_negative_one_power_2022 :
  -4^2 * (-1)^2022 = -16 := by
  sorry

end NUMINAMATH_CALUDE_negative_four_squared_times_negative_one_power_2022_l3181_318156


namespace NUMINAMATH_CALUDE_floor_of_5_7_l3181_318134

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_5_7_l3181_318134


namespace NUMINAMATH_CALUDE_combined_mean_l3181_318115

theorem combined_mean (set1_count : ℕ) (set1_mean : ℝ) (set2_count : ℕ) (set2_mean : ℝ) :
  set1_count = 8 →
  set2_count = 10 →
  set1_mean = 17 →
  set2_mean = 23 →
  let total_count := set1_count + set2_count
  let combined_mean := (set1_count * set1_mean + set2_count * set2_mean) / total_count
  combined_mean = (8 * 17 + 10 * 23) / 18 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_mean_l3181_318115


namespace NUMINAMATH_CALUDE_square_root_problem_l3181_318190

theorem square_root_problem (m : ℝ) (x : ℝ) 
  (h1 : m > 0) 
  (h2 : Real.sqrt m = x + 1) 
  (h3 : Real.sqrt m = x - 3) : 
  m = 4 := by sorry

end NUMINAMATH_CALUDE_square_root_problem_l3181_318190


namespace NUMINAMATH_CALUDE_floor_inequality_l3181_318129

theorem floor_inequality (α β : ℝ) : 
  ⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋ := by sorry

end NUMINAMATH_CALUDE_floor_inequality_l3181_318129


namespace NUMINAMATH_CALUDE_purchase_group_equation_l3181_318145

/-- A group of people buying an item -/
structure PurchaseGroup where
  price : ℝ  -- Price of the item in coins
  excess_contribution : ℝ := 8  -- Contribution that exceeds the price
  excess_amount : ℝ := 3  -- Amount by which the excess contribution exceeds the price
  shortfall_contribution : ℝ := 7  -- Contribution that falls short of the price
  shortfall_amount : ℝ := 4  -- Amount by which the shortfall contribution falls short of the price

/-- The equation holds for a purchase group -/
theorem purchase_group_equation (g : PurchaseGroup) :
  (g.price + g.excess_amount) / g.excess_contribution = (g.price - g.shortfall_amount) / g.shortfall_contribution :=
sorry

end NUMINAMATH_CALUDE_purchase_group_equation_l3181_318145


namespace NUMINAMATH_CALUDE_a_spends_95_percent_l3181_318119

/-- Represents the salaries and spending percentages of two individuals A and B -/
structure SalaryData where
  total_salary : ℝ
  a_salary : ℝ
  b_spend_percent : ℝ
  a_spend_percent : ℝ

/-- Calculates the savings of an individual given their salary and spending percentage -/
def savings (salary : ℝ) (spend_percent : ℝ) : ℝ :=
  salary * (1 - spend_percent)

/-- Theorem stating that under given conditions, A spends 95% of their salary -/
theorem a_spends_95_percent (data : SalaryData) 
  (h1 : data.total_salary = 3000)
  (h2 : data.a_salary = 2250)
  (h3 : data.b_spend_percent = 0.85)
  (h4 : savings data.a_salary data.a_spend_percent = 
        savings (data.total_salary - data.a_salary) data.b_spend_percent) :
  data.a_spend_percent = 0.95 := by
  sorry


end NUMINAMATH_CALUDE_a_spends_95_percent_l3181_318119


namespace NUMINAMATH_CALUDE_largest_divisor_of_three_consecutive_even_integers_l3181_318112

theorem largest_divisor_of_three_consecutive_even_integers :
  ∃ (d : ℕ), d = 24 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (2*n) * (2*n + 2) * (2*n + 4)) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (2*m) * (2*m + 2) * (2*m + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_three_consecutive_even_integers_l3181_318112


namespace NUMINAMATH_CALUDE_min_cos_C_in_triangle_l3181_318122

theorem min_cos_C_in_triangle (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sin : Real.sin A + 2 * Real.sin B = 3 * Real.sin C) : 
  (2 * Real.sqrt 10 - 2) / 9 ≤ Real.cos C :=
sorry

end NUMINAMATH_CALUDE_min_cos_C_in_triangle_l3181_318122


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3181_318127

/-- An arithmetic sequence {a_n} with a_1 = 1 and a_3 = a_2^2 - 4 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  a 3 = (a 2)^2 - 4 ∧
  ∀ n m : ℕ, n < m → a n < a m

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) :
  ∀ n : ℕ, a n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3181_318127


namespace NUMINAMATH_CALUDE_circumscribed_diagonals_center_implies_rhombus_l3181_318113

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Check if a quadrilateral is circumscribed around a circle -/
def isCircumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Check if the diagonals of a quadrilateral intersect at a given point -/
def diagonalsIntersectAt (q : Quadrilateral) (p : ℝ × ℝ) : Prop := sorry

/-- Check if a quadrilateral is a rhombus -/
def isRhombus (q : Quadrilateral) : Prop := sorry

/-- Main theorem -/
theorem circumscribed_diagonals_center_implies_rhombus (q : Quadrilateral) (c : Circle) :
  isCircumscribed q c → diagonalsIntersectAt q c.center → isRhombus q := by sorry

end NUMINAMATH_CALUDE_circumscribed_diagonals_center_implies_rhombus_l3181_318113


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_subset_condition_l3181_318124

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | -1 - 2*a ≤ x ∧ x ≤ a - 2}

-- Statement for part (1)
theorem sufficient_not_necessary (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ∧ (∃ x, x ∈ B a ∧ x ∉ A) ↔ a ≥ 7 :=
sorry

-- Statement for part (2)
theorem subset_condition (a : ℝ) :
  B a ⊆ A ↔ a < 1/3 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_subset_condition_l3181_318124


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l3181_318184

theorem min_value_sum_fractions (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l3181_318184


namespace NUMINAMATH_CALUDE_min_height_box_l3181_318167

def box_height (side_length : ℝ) : ℝ := 2 * side_length

def surface_area (side_length : ℝ) : ℝ := 10 * side_length^2

theorem min_height_box (min_area : ℝ) (h_min_area : min_area = 120) :
  ∃ (h : ℝ), h = box_height (Real.sqrt (min_area / 10)) ∧
             h = 8 ∧
             ∀ (s : ℝ), surface_area s ≥ min_area → box_height s ≥ h :=
by sorry

end NUMINAMATH_CALUDE_min_height_box_l3181_318167


namespace NUMINAMATH_CALUDE_front_wheel_perimeter_front_wheel_perimeter_is_30_l3181_318102

/-- The perimeter of the front wheel of a bicycle, given the perimeter of the back wheel
    and the number of revolutions each wheel makes to cover the same distance. -/
theorem front_wheel_perimeter (back_wheel_perimeter : ℝ) 
    (front_wheel_revolutions : ℝ) (back_wheel_revolutions : ℝ) : ℝ :=
  let front_wheel_perimeter := (back_wheel_perimeter * back_wheel_revolutions) / front_wheel_revolutions
  have back_wheel_perimeter_eq : back_wheel_perimeter = 20 := by sorry
  have front_wheel_revolutions_eq : front_wheel_revolutions = 240 := by sorry
  have back_wheel_revolutions_eq : back_wheel_revolutions = 360 := by sorry
  have equal_distance : front_wheel_perimeter * front_wheel_revolutions = 
                        back_wheel_perimeter * back_wheel_revolutions := by sorry
  30

theorem front_wheel_perimeter_is_30 : front_wheel_perimeter 20 240 360 = 30 := by sorry

end NUMINAMATH_CALUDE_front_wheel_perimeter_front_wheel_perimeter_is_30_l3181_318102


namespace NUMINAMATH_CALUDE_contractor_fine_calculation_l3181_318169

/-- Calculates the daily fine for a contractor given contract details -/
def calculate_daily_fine (contract_duration : ℕ) (daily_pay : ℕ) (total_payment : ℕ) (days_absent : ℕ) : ℚ :=
  let days_worked := contract_duration - days_absent
  let total_earned := days_worked * daily_pay
  ((total_earned - total_payment) : ℚ) / days_absent

theorem contractor_fine_calculation :
  let contract_duration : ℕ := 30
  let daily_pay : ℕ := 25
  let total_payment : ℕ := 425
  let days_absent : ℕ := 10
  calculate_daily_fine contract_duration daily_pay total_payment days_absent = 15/2 := by
  sorry

#eval calculate_daily_fine 30 25 425 10

end NUMINAMATH_CALUDE_contractor_fine_calculation_l3181_318169


namespace NUMINAMATH_CALUDE_solution_set_l3181_318163

theorem solution_set (x : ℝ) : 
  33 * 32 ≤ x ∧ 
  Int.floor x + Int.ceil x = 5 → 
  2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l3181_318163


namespace NUMINAMATH_CALUDE_battleship_max_ships_l3181_318123

/-- Represents a game board --/
structure Board :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a ship --/
structure Ship :=
  (length : Nat)
  (width : Nat)

/-- Calculates the maximum number of ships that can be placed on a board --/
def maxShips (board : Board) (ship : Ship) : Nat :=
  (board.rows * board.cols) / (ship.length * ship.width)

theorem battleship_max_ships :
  let board : Board := ⟨10, 10⟩
  let ship : Ship := ⟨4, 1⟩
  maxShips board ship = 25 := by
  sorry

#eval maxShips ⟨10, 10⟩ ⟨4, 1⟩

end NUMINAMATH_CALUDE_battleship_max_ships_l3181_318123


namespace NUMINAMATH_CALUDE_triangle_properties_l3181_318157

/-- Properties of a triangle ABC -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = t.b * Real.cos t.C + Real.sqrt 3 * t.c * Real.sin t.B)
  (h2 : t.b = 2)
  (h3 : t.a = Real.sqrt 3 * t.c) : 
  t.B = π / 6 ∧ t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3181_318157
