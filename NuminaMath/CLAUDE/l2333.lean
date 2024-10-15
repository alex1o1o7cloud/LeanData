import Mathlib

namespace NUMINAMATH_CALUDE_special_hexagon_area_l2333_233338

/-- A hexagon with specific properties -/
structure SpecialHexagon where
  -- Each angle measures 120°
  angle_measure : ℝ
  angle_measure_eq : angle_measure = 120
  -- Sides alternately measure 1 cm and √3 cm
  side_length1 : ℝ
  side_length2 : ℝ
  side_length1_eq : side_length1 = 1
  side_length2_eq : side_length2 = Real.sqrt 3

/-- The area of the special hexagon -/
noncomputable def area (h : SpecialHexagon) : ℝ := 3 + Real.sqrt 3

/-- Theorem stating that the area of the special hexagon is 3 + √3 cm² -/
theorem special_hexagon_area (h : SpecialHexagon) : area h = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_area_l2333_233338


namespace NUMINAMATH_CALUDE_remaining_money_l2333_233366

def initial_amount : ℕ := 11
def spent_amount : ℕ := 2
def lost_amount : ℕ := 6

theorem remaining_money :
  initial_amount - spent_amount - lost_amount = 3 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l2333_233366


namespace NUMINAMATH_CALUDE_monica_reading_plan_l2333_233368

def books_last_year : ℕ := 16

def books_this_year : ℕ := 2 * books_last_year

def books_next_year : ℕ := 2 * books_this_year + 5

theorem monica_reading_plan : books_next_year = 69 := by
  sorry

end NUMINAMATH_CALUDE_monica_reading_plan_l2333_233368


namespace NUMINAMATH_CALUDE_estimate_percentage_negative_attitude_l2333_233362

theorem estimate_percentage_negative_attitude 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (negative_attitude_count : ℕ) 
  (h1 : total_population = 2500)
  (h2 : sample_size = 400)
  (h3 : negative_attitude_count = 360) :
  (negative_attitude_count : ℝ) / (sample_size : ℝ) * 100 = 90 := by
sorry

end NUMINAMATH_CALUDE_estimate_percentage_negative_attitude_l2333_233362


namespace NUMINAMATH_CALUDE_gem_stone_necklaces_count_l2333_233308

/-- The number of gem stone necklaces sold by Faye -/
def gem_stone_necklaces : ℕ := 7

/-- The number of bead necklaces sold by Faye -/
def bead_necklaces : ℕ := 3

/-- The price of each necklace in dollars -/
def necklace_price : ℕ := 7

/-- The total earnings from all necklaces in dollars -/
def total_earnings : ℕ := 70

/-- Theorem stating that the number of gem stone necklaces sold is 7 -/
theorem gem_stone_necklaces_count :
  gem_stone_necklaces = (total_earnings - bead_necklaces * necklace_price) / necklace_price :=
by sorry

end NUMINAMATH_CALUDE_gem_stone_necklaces_count_l2333_233308


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l2333_233345

theorem smaller_solution_quadratic (x : ℝ) : 
  x^2 + 15*x + 36 = 0 ∧ x ≠ -3 → x = -12 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l2333_233345


namespace NUMINAMATH_CALUDE_ernie_circles_l2333_233347

theorem ernie_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) 
  (ali_circles : ℕ) (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) 
  (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) : 
  (total_boxes - ali_circles * ali_boxes_per_circle) / ernie_boxes_per_circle = 4 := by
  sorry

end NUMINAMATH_CALUDE_ernie_circles_l2333_233347


namespace NUMINAMATH_CALUDE_five_students_three_not_adjacent_l2333_233352

/-- The number of ways to arrange n elements --/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange 5 students in a line --/
def totalArrangements : ℕ := factorial 5

/-- The number of ways to arrange 3 students together and 2 separately --/
def restrictedArrangements : ℕ := factorial 3 * factorial 3

/-- The number of ways to arrange 5 students where 3 are not adjacent --/
def validArrangements : ℕ := totalArrangements - restrictedArrangements

theorem five_students_three_not_adjacent :
  validArrangements = 84 :=
sorry

end NUMINAMATH_CALUDE_five_students_three_not_adjacent_l2333_233352


namespace NUMINAMATH_CALUDE_benny_piggy_bank_l2333_233321

theorem benny_piggy_bank (january_amount february_amount total_amount : ℕ) 
  (h1 : january_amount = 19)
  (h2 : february_amount = january_amount)
  (h3 : total_amount = 46) : 
  total_amount - (january_amount + february_amount) = 8 := by
  sorry

end NUMINAMATH_CALUDE_benny_piggy_bank_l2333_233321


namespace NUMINAMATH_CALUDE_first_day_exceeding_500_l2333_233342

def bacteria_growth (n : ℕ) : ℕ := 5 * 3^n

theorem first_day_exceeding_500 :
  (∀ k < 6, bacteria_growth k ≤ 500) ∧ bacteria_growth 6 > 500 := by sorry

end NUMINAMATH_CALUDE_first_day_exceeding_500_l2333_233342


namespace NUMINAMATH_CALUDE_root_in_interval_l2333_233318

def f (x : ℝ) := x^3 - 3*x + 1

theorem root_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l2333_233318


namespace NUMINAMATH_CALUDE_factor_expression_l2333_233315

theorem factor_expression (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2333_233315


namespace NUMINAMATH_CALUDE_root_value_theorem_l2333_233326

theorem root_value_theorem (m : ℝ) (h : 2 * m^2 - 7 * m + 1 = 0) :
  m * (2 * m - 7) + 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l2333_233326


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l2333_233375

theorem normal_distribution_std_dev (μ σ x : ℝ) (hμ : μ = 17.5) (hσ : σ = 2.5) (hx : x = 12.5) :
  (x - μ) / σ = -2 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l2333_233375


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2333_233353

/-- 
Given a quadratic equation x^2 - 6x + k = 0 with two equal real roots,
prove that k = 9
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 6*y + k = 0 → y = x) → 
  k = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2333_233353


namespace NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_perpendicular_line_plane_from_intersection_l2333_233309

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)
variable (contained_in : Line → Plane → Prop)

-- Theorem for statement ②
theorem parallel_planes_from_perpendicular_lines 
  (l m : Line) (α β : Plane) :
  parallel l m →
  perpendicular_line_plane m α →
  perpendicular_line_plane l β →
  parallel_plane α β :=
sorry

-- Theorem for statement ④
theorem perpendicular_line_plane_from_intersection 
  (l : Line) (α β : Plane) (m : Line) :
  perpendicular_plane α β →
  intersection α β = m →
  contained_in l β →
  perpendicular l m →
  perpendicular_line_plane l α :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_perpendicular_line_plane_from_intersection_l2333_233309


namespace NUMINAMATH_CALUDE_relationship_abc_l2333_233358

theorem relationship_abc (a b c : ℝ) 
  (h : Real.exp a + a = Real.log b + b ∧ Real.log b + b = Real.sqrt c + c ∧ Real.sqrt c + c = Real.sin 1) : 
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2333_233358


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2333_233305

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = 0.5875 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2333_233305


namespace NUMINAMATH_CALUDE_phi_subset_singleton_zero_l2333_233363

-- Define Φ as a set
variable (Φ : Set ℕ)

-- Theorem stating that Φ is a subset of {0}
theorem phi_subset_singleton_zero : Φ ⊆ {0} := by
  sorry

end NUMINAMATH_CALUDE_phi_subset_singleton_zero_l2333_233363


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2333_233311

/-- Calculates the simple interest rate given loan amounts, durations, and total interest received. -/
theorem simple_interest_rate 
  (loan_b loan_c : ℕ) 
  (duration_b duration_c : ℕ) 
  (total_interest : ℕ) : 
  loan_b = 5000 → 
  loan_c = 3000 → 
  duration_b = 2 → 
  duration_c = 4 → 
  total_interest = 1540 → 
  ∃ (rate : ℚ), 
    rate = 7 ∧ 
    (loan_b * duration_b * rate + loan_c * duration_c * rate) / 100 = total_interest :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2333_233311


namespace NUMINAMATH_CALUDE_problem_solution_l2333_233334

theorem problem_solution :
  ∀ (a b c d : ℝ),
    1000 * a = 85^2 - 15^2 →
    5 * a + 2 * b = 41 →
    (-3)^2 + 6 * (-3) + c = 0 →
    d^2 = (5 - c)^2 + (4 - 1)^2 →
    a = 7 ∧ b = 3 ∧ c = 9 ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2333_233334


namespace NUMINAMATH_CALUDE_point_translation_l2333_233360

def initial_point : ℝ × ℝ := (-5, 1)
def x_translation : ℝ := 2
def y_translation : ℝ := -4

theorem point_translation (P : ℝ × ℝ) (dx dy : ℝ) :
  P = initial_point →
  (P.1 + dx, P.2 + dy) = (-3, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_translation_l2333_233360


namespace NUMINAMATH_CALUDE_apples_total_weight_l2333_233369

def apple_weight : ℕ := 4
def orange_weight : ℕ := 3
def plum_weight : ℕ := 2
def bag_capacity : ℕ := 49
def num_bags : ℕ := 5

def fruit_set_weight : ℕ := apple_weight + orange_weight + plum_weight

def fruits_per_bag : ℕ := (bag_capacity / fruit_set_weight) * fruit_set_weight

theorem apples_total_weight :
  fruits_per_bag / fruit_set_weight * apple_weight * num_bags = 80 := by sorry

end NUMINAMATH_CALUDE_apples_total_weight_l2333_233369


namespace NUMINAMATH_CALUDE_no_integer_solution_exists_l2333_233310

theorem no_integer_solution_exists (a b : ℤ) : 
  ∃ c : ℤ, ∀ m n : ℤ, m^2 + a*m + b ≠ 2*n^2 + 2*n + c :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_exists_l2333_233310


namespace NUMINAMATH_CALUDE_parabola_vertex_l2333_233354

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x + 1)^2 - 6

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-1, -6)

/-- Theorem: The vertex of the parabola y = -2(x+1)^2 - 6 is at the point (-1, -6) -/
theorem parabola_vertex :
  let (h, k) := vertex
  ∀ x y, parabola x y → (x - h)^2 ≤ (y - k) / (-2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2333_233354


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2333_233391

def A : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2333_233391


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2333_233371

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2333_233371


namespace NUMINAMATH_CALUDE_total_painting_cost_l2333_233377

/-- Calculate the last term of an arithmetic sequence -/
def lastTerm (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

/-- Count the number of digits in a natural number -/
def digitCount (n : ℕ) : ℕ :=
  if n < 10 then 1 else if n < 100 then 2 else 3

/-- Calculate the cost of painting numbers for one side of the street -/
def sideCost (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  let lastNum := lastTerm a₁ d n
  let twoDigitCount := (min 99 lastNum - a₁) / d + 1
  let threeDigitCount := n - twoDigitCount
  2 * (2 * twoDigitCount + 3 * threeDigitCount)

/-- The main theorem stating the total cost for painting all house numbers -/
theorem total_painting_cost : 
  sideCost 5 7 30 + sideCost 6 8 30 = 312 := by sorry

end NUMINAMATH_CALUDE_total_painting_cost_l2333_233377


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l2333_233322

/-- The volume of a tetrahedron with vertices on the positive coordinate axes -/
theorem tetrahedron_volume (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = 25) (h5 : b^2 + c^2 = 36) (h6 : c^2 + a^2 = 49) :
  (1 / 6 : ℝ) * a * b * c = Real.sqrt 95 := by
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l2333_233322


namespace NUMINAMATH_CALUDE_tan_beta_value_l2333_233359

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l2333_233359


namespace NUMINAMATH_CALUDE_vector_properties_l2333_233301

def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (-1, -1)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 ≠ 2) ∧
  (∃ (k : ℝ), a ≠ k • b) ∧
  (b.1 * (a.1 + b.1) + b.2 * (a.2 + b.2) = 0) ∧
  (a.1^2 + a.2^2 ≠ b.1^2 + b.2^2) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l2333_233301


namespace NUMINAMATH_CALUDE_ant_movement_l2333_233397

-- Define the type for a 2D position
def Position := ℝ × ℝ

-- Define the initial position
def initial_position : Position := (-2, 4)

-- Define the horizontal movement
def horizontal_movement : ℝ := 3

-- Define the vertical movement
def vertical_movement : ℝ := -2

-- Define the function to calculate the final position
def final_position (initial : Position) (horizontal : ℝ) (vertical : ℝ) : Position :=
  (initial.1 + horizontal, initial.2 + vertical)

-- Theorem statement
theorem ant_movement :
  final_position initial_position horizontal_movement vertical_movement = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_ant_movement_l2333_233397


namespace NUMINAMATH_CALUDE_sector_perimeter_to_circumference_ratio_l2333_233333

theorem sector_perimeter_to_circumference_ratio (r : ℝ) (hr : r > 0) :
  let circumference := 2 * π * r
  let sector_arc_length := circumference / 3
  let sector_perimeter := sector_arc_length + 2 * r
  sector_perimeter / circumference = (π + 3) / (3 * π) := by
sorry

end NUMINAMATH_CALUDE_sector_perimeter_to_circumference_ratio_l2333_233333


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2333_233323

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2333_233323


namespace NUMINAMATH_CALUDE_min_value_theorem_l2333_233316

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  b / a + 1 / b ≥ 3 ∧ (b / a + 1 / b = 3 ↔ a = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2333_233316


namespace NUMINAMATH_CALUDE_total_fruits_l2333_233331

theorem total_fruits (apples bananas grapes : ℕ) 
  (h1 : apples = 5) 
  (h2 : bananas = 4) 
  (h3 : grapes = 6) : 
  apples + bananas + grapes = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_l2333_233331


namespace NUMINAMATH_CALUDE_expression_simplification_l2333_233339

theorem expression_simplification (a : ℤ) (h : a = 2021) :
  (((a + 1 : ℚ) / a + 1 / (a + 1)) - a / (a + 1)) = (a^2 + a + 2 : ℚ) / (a * (a + 1)) ∧
  a^2 + a + 2 = 4094865 :=
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2333_233339


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2333_233381

theorem complex_product_theorem (a b : ℝ) :
  let z₁ : ℂ := Complex.mk a b
  let z₂ : ℂ := Complex.mk a (-b)
  let z₃ : ℂ := Complex.mk (-a) b
  let z₄ : ℂ := Complex.mk (-a) (-b)
  (z₁ * z₂ * z₃ * z₄).re = (a^2 + b^2)^2 ∧ (z₁ * z₂ * z₃ * z₄).im = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2333_233381


namespace NUMINAMATH_CALUDE_number_puzzle_l2333_233378

theorem number_puzzle : ∃ N : ℚ, N = (3/8) * N + (1/4) * N + 15 ∧ N = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2333_233378


namespace NUMINAMATH_CALUDE_yellow_bows_count_l2333_233367

theorem yellow_bows_count (total : ℚ) :
  (1 / 6 : ℚ) * total +  -- yellow bows
  (1 / 3 : ℚ) * total +  -- purple bows
  (1 / 8 : ℚ) * total +  -- orange bows
  40 = total →           -- black bows
  (1 / 6 : ℚ) * total = 160 / 9 := by
sorry

end NUMINAMATH_CALUDE_yellow_bows_count_l2333_233367


namespace NUMINAMATH_CALUDE_scout_saturday_hours_scout_saturday_hours_is_four_l2333_233319

/-- Scout's delivery earnings over a weekend --/
theorem scout_saturday_hours : ℕ :=
  let base_pay : ℕ := 10  -- Base pay per hour in dollars
  let tip_per_customer : ℕ := 5  -- Tip per customer in dollars
  let saturday_customers : ℕ := 5  -- Number of customers on Saturday
  let sunday_hours : ℕ := 5  -- Hours worked on Sunday
  let sunday_customers : ℕ := 8  -- Number of customers on Sunday
  let total_earnings : ℕ := 155  -- Total earnings for the weekend in dollars

  let saturday_hours : ℕ := 
    (total_earnings - 
     (base_pay * sunday_hours + tip_per_customer * sunday_customers + 
      tip_per_customer * saturday_customers)) / base_pay

  saturday_hours

/-- Proof that Scout worked 4 hours on Saturday --/
theorem scout_saturday_hours_is_four : scout_saturday_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_scout_saturday_hours_scout_saturday_hours_is_four_l2333_233319


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2333_233344

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x y : ℝ, 16*x^2 + m*x*y + 25*y^2 = (a*x + b*y)^2) → 
  m = 40 ∨ m = -40 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2333_233344


namespace NUMINAMATH_CALUDE_second_year_percentage_correct_l2333_233387

/-- The number of second-year students studying numeric methods -/
def numeric_methods : ℕ := 250

/-- The number of second-year students studying automatic control of airborne vehicles -/
def automatic_control : ℕ := 423

/-- The number of second-year students studying both subjects -/
def both_subjects : ℕ := 134

/-- The total number of students in the faculty -/
def total_students : ℕ := 673

/-- The percentage of second-year students in the faculty -/
def second_year_percentage : ℚ :=
  (numeric_methods + automatic_control - both_subjects : ℚ) / total_students * 100

theorem second_year_percentage_correct :
  second_year_percentage = (250 + 423 - 134 : ℚ) / 673 * 100 :=
by sorry

end NUMINAMATH_CALUDE_second_year_percentage_correct_l2333_233387


namespace NUMINAMATH_CALUDE_second_die_sides_l2333_233343

theorem second_die_sides (n : ℕ) : 
  n > 0 → 
  (1 : ℚ) / 6 * (1 : ℚ) / n = 0.023809523809523808 → 
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_die_sides_l2333_233343


namespace NUMINAMATH_CALUDE_election_winner_margin_l2333_233396

theorem election_winner_margin (total_votes : ℕ) (winner_votes : ℕ) (winner_percentage : ℚ) : 
  winner_percentage = 62 / 100 ∧ 
  winner_votes = 775 ∧ 
  winner_votes = (winner_percentage * total_votes).floor →
  winner_votes - (total_votes - winner_votes) = 300 := by
sorry

end NUMINAMATH_CALUDE_election_winner_margin_l2333_233396


namespace NUMINAMATH_CALUDE_E_is_true_l2333_233394

-- Define the statements as propositions
variable (A B C D E : Prop)

-- Define the condition that only one statement is true
def only_one_true (A B C D E : Prop) : Prop :=
  (A ∧ ¬B ∧ ¬C ∧ ¬D ∧ ¬E) ∨
  (¬A ∧ B ∧ ¬C ∧ ¬D ∧ ¬E) ∨
  (¬A ∧ ¬B ∧ C ∧ ¬D ∧ ¬E) ∨
  (¬A ∧ ¬B ∧ ¬C ∧ D ∧ ¬E) ∨
  (¬A ∧ ¬B ∧ ¬C ∧ ¬D ∧ E)

-- Define the content of each statement
def statement_definitions (A B C D E : Prop) : Prop :=
  (A ↔ B) ∧
  (B ↔ ¬E) ∧
  (C ↔ (A ∧ B ∧ C ∧ D ∧ E)) ∧
  (D ↔ (¬A ∧ ¬B ∧ ¬C ∧ ¬D ∧ ¬E)) ∧
  (E ↔ ¬A)

-- Theorem stating that E is the only true statement
theorem E_is_true (A B C D E : Prop) 
  (h1 : only_one_true A B C D E) 
  (h2 : statement_definitions A B C D E) : 
  E ∧ ¬A ∧ ¬B ∧ ¬C ∧ ¬D :=
sorry

end NUMINAMATH_CALUDE_E_is_true_l2333_233394


namespace NUMINAMATH_CALUDE_power_difference_equality_l2333_233346

theorem power_difference_equality : (3^4)^4 - (4^3)^3 = 42792577 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l2333_233346


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2333_233351

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 2, 4}

-- Define set N
def N : Finset Nat := {2, 4, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2333_233351


namespace NUMINAMATH_CALUDE_steps_to_distance_l2333_233349

/-- Given that 625 steps correspond to 500 meters, prove that 10,000 steps at the same rate will result in a distance of 8 km. -/
theorem steps_to_distance (steps_short : ℕ) (distance_short : ℝ) (steps_long : ℕ) :
  steps_short = 625 →
  distance_short = 500 →
  steps_long = 10000 →
  (distance_short / steps_short) * steps_long = 8000 :=
by sorry

end NUMINAMATH_CALUDE_steps_to_distance_l2333_233349


namespace NUMINAMATH_CALUDE_eight_faucets_fill_time_correct_l2333_233380

/-- The time (in seconds) it takes for eight faucets to fill a 50-gallon tank,
    given that four faucets fill a 200-gallon tank in 8 minutes and all faucets
    dispense water at the same rate. -/
def eight_faucets_fill_time : ℕ := by sorry

/-- Four faucets fill a 200-gallon tank in 8 minutes. -/
def four_faucets_fill_time : ℕ := 8 * 60  -- in seconds

/-- The volume of the tank filled by four faucets. -/
def four_faucets_volume : ℕ := 200  -- in gallons

/-- The volume of the tank to be filled by eight faucets. -/
def eight_faucets_volume : ℕ := 50  -- in gallons

/-- All faucets dispense water at the same rate. -/
axiom faucets_equal_rate : True

theorem eight_faucets_fill_time_correct :
  eight_faucets_fill_time = 60 := by sorry

end NUMINAMATH_CALUDE_eight_faucets_fill_time_correct_l2333_233380


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2333_233385

/-- Given a line passing through points (1, -2) and (3, 4), 
    prove that the sum of its slope and y-intercept is -2 -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
  (∀ (x y : ℝ), y = m * x + b → 
    ((x = 1 ∧ y = -2) ∨ (x = 3 ∧ y = 4))) → 
  m + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2333_233385


namespace NUMINAMATH_CALUDE_travis_payment_l2333_233372

def payment_calculation (total_bowls glass_bowls ceramic_bowls base_fee safe_delivery_fee
                         broken_glass_charge broken_ceramic_charge lost_glass_charge lost_ceramic_charge
                         additional_glass_fee additional_ceramic_fee lost_glass lost_ceramic
                         broken_glass broken_ceramic : ℕ) : ℚ :=
  let safe_glass := glass_bowls - lost_glass - broken_glass
  let safe_ceramic := ceramic_bowls - lost_ceramic - broken_ceramic
  let safe_delivery_payment := (safe_glass + safe_ceramic) * safe_delivery_fee
  let broken_lost_charges := broken_glass * broken_glass_charge + broken_ceramic * broken_ceramic_charge +
                             lost_glass * lost_glass_charge + lost_ceramic * lost_ceramic_charge
  let additional_moving_fee := glass_bowls * additional_glass_fee + ceramic_bowls * additional_ceramic_fee
  (base_fee + safe_delivery_payment - broken_lost_charges + additional_moving_fee : ℚ)

theorem travis_payment :
  payment_calculation 638 375 263 100 3 5 4 6 3 (1/2) (1/4) 9 3 10 5 = 2053.25 := by
  sorry

end NUMINAMATH_CALUDE_travis_payment_l2333_233372


namespace NUMINAMATH_CALUDE_bake_sale_cookies_l2333_233330

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 48

/-- The number of boxes Abigail collected -/
def abigail_boxes : ℕ := 2

/-- The number of quarter boxes Grayson collected -/
def grayson_quarter_boxes : ℕ := 3

/-- The number of boxes Olivia collected -/
def olivia_boxes : ℕ := 3

/-- The total number of cookies collected -/
def total_cookies : ℕ := 276

theorem bake_sale_cookies : 
  cookies_per_box * abigail_boxes + 
  (cookies_per_box / 4) * grayson_quarter_boxes + 
  cookies_per_box * olivia_boxes = total_cookies := by
sorry

end NUMINAMATH_CALUDE_bake_sale_cookies_l2333_233330


namespace NUMINAMATH_CALUDE_fraction_simplification_l2333_233383

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (2 / y + 1 / x) / (1 / x) = 13 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2333_233383


namespace NUMINAMATH_CALUDE_rotation_maps_points_l2333_233356

-- Define points in R²
def C : ℝ × ℝ := (3, -2)
def C' : ℝ × ℝ := (-3, 2)
def D : ℝ × ℝ := (4, -5)
def D' : ℝ × ℝ := (-4, 5)

-- Define rotation by 180°
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement
theorem rotation_maps_points :
  rotate180 C = C' ∧ rotate180 D = D' :=
sorry

end NUMINAMATH_CALUDE_rotation_maps_points_l2333_233356


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2333_233392

theorem absolute_value_equation_solutions :
  ∃! (s : Set ℝ), s = {x : ℝ | |x + 1| = |x - 2| + |x - 5| + |x - 6|} ∧ s = {4, 7} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2333_233392


namespace NUMINAMATH_CALUDE_calculate_principal_l2333_233307

/-- Given simple interest, rate, and time, calculate the principal sum -/
theorem calculate_principal (simple_interest rate time : ℝ) : 
  simple_interest = 16065 * rate * time / 100 →
  rate = 5 →
  time = 5 →
  simple_interest = 4016.25 := by
  sorry

#check calculate_principal

end NUMINAMATH_CALUDE_calculate_principal_l2333_233307


namespace NUMINAMATH_CALUDE_four_common_tangents_l2333_233348

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y - 2 = 0

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 4 = 0

/-- The number of common tangent lines between C₁ and C₂ -/
def num_common_tangents : ℕ := 4

/-- Theorem stating that the number of common tangent lines between C₁ and C₂ is 4 -/
theorem four_common_tangents : num_common_tangents = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_common_tangents_l2333_233348


namespace NUMINAMATH_CALUDE_total_hats_bought_l2333_233325

theorem total_hats_bought (blue_cost green_cost total_price green_hats : ℕ) 
  (h1 : blue_cost = 6)
  (h2 : green_cost = 7)
  (h3 : total_price = 550)
  (h4 : green_hats = 40) :
  ∃ (blue_hats : ℕ), blue_cost * blue_hats + green_cost * green_hats = total_price ∧
                     blue_hats + green_hats = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_hats_bought_l2333_233325


namespace NUMINAMATH_CALUDE_min_red_to_blue_l2333_233340

/-- Represents the color of a chameleon -/
inductive Color
| Red
| Blue
| Other1
| Other2
| Other3

/-- Represents the result of a bite interaction between two chameleons -/
def bite_result : Color → Color → Color := sorry

/-- Represents a sequence of bites -/
def BiteSequence := List (Nat × Nat)

/-- The number of colors available -/
def num_colors : Nat := 5

/-- The given number of red chameleons that can become blue -/
def given_red_count : Nat := 2023

/-- Checks if a sequence of bites transforms all chameleons to blue -/
def all_blue (initial : List Color) (sequence : BiteSequence) : Prop := sorry

/-- The theorem to be proved -/
theorem min_red_to_blue :
  ∀ (k : Nat),
  (k ≥ 5) →
  (∃ (sequence : BiteSequence), all_blue (List.replicate k Color.Red) sequence) ∧
  (∀ (j : Nat), j < 5 →
    ¬∃ (sequence : BiteSequence), all_blue (List.replicate j Color.Red) sequence) :=
sorry

end NUMINAMATH_CALUDE_min_red_to_blue_l2333_233340


namespace NUMINAMATH_CALUDE_largest_number_proof_l2333_233357

theorem largest_number_proof (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  (Nat.gcd a b = 42) → 
  (∃ k : ℕ, Nat.lcm a b = 42 * 11 * 12 * k) →
  (max a b = 504) := by
sorry

end NUMINAMATH_CALUDE_largest_number_proof_l2333_233357


namespace NUMINAMATH_CALUDE_palindromic_four_digit_squares_l2333_233384

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def ends_with_0_4_or_6 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 4 ∨ n % 10 = 6

def satisfies_conditions (n : ℕ) : Prop :=
  is_square n ∧ is_four_digit n ∧ is_palindrome n ∧ ends_with_0_4_or_6 n

theorem palindromic_four_digit_squares :
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n, n ∈ s ↔ satisfies_conditions n :=
sorry

end NUMINAMATH_CALUDE_palindromic_four_digit_squares_l2333_233384


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_with_16_divisors_l2333_233341

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def last_three_digits (n : ℕ) : ℕ := n % 1000

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_five_digit_multiple_with_16_divisors :
  ∃ (n : ℕ), is_five_digit n ∧ 2014 ∣ n ∧ count_divisors (last_three_digits n) = 16 ∧
  ∀ (m : ℕ), is_five_digit m ∧ 2014 ∣ m ∧ count_divisors (last_three_digits m) = 16 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_with_16_divisors_l2333_233341


namespace NUMINAMATH_CALUDE_circle_divides_rectangle_sides_l2333_233399

/-- A circle touching two adjacent sides of a rectangle --/
structure CircleTouchingRectangle where
  radius : ℝ
  rect_side1 : ℝ
  rect_side2 : ℝ
  (radius_positive : 0 < radius)
  (rect_sides_positive : 0 < rect_side1 ∧ 0 < rect_side2)
  (radius_fits : radius < rect_side1 ∧ radius < rect_side2)

/-- The segments into which the circle divides the rectangle sides --/
structure RectangleSegments where
  seg1 : ℝ
  seg2 : ℝ
  seg3 : ℝ
  seg4 : ℝ
  seg5 : ℝ
  seg6 : ℝ

/-- Theorem stating how the circle divides the rectangle sides --/
theorem circle_divides_rectangle_sides (c : CircleTouchingRectangle) 
  (h : c.radius = 26 ∧ c.rect_side1 = 36 ∧ c.rect_side2 = 60) :
  ∃ (s : RectangleSegments), 
    s.seg1 = 26 ∧ s.seg2 = 34 ∧ 
    s.seg3 = 26 ∧ s.seg4 = 10 ∧ 
    s.seg5 = 2 ∧ s.seg6 = 48 :=
sorry

end NUMINAMATH_CALUDE_circle_divides_rectangle_sides_l2333_233399


namespace NUMINAMATH_CALUDE_maxwell_walking_speed_l2333_233373

-- Define the given conditions
def total_distance : ℝ := 65
def maxwell_distance : ℝ := 26
def brad_speed : ℝ := 3

-- Define Maxwell's speed as a variable
def maxwell_speed : ℝ := sorry

-- Theorem to prove
theorem maxwell_walking_speed :
  (maxwell_distance / maxwell_speed = (total_distance - maxwell_distance) / brad_speed) →
  maxwell_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_maxwell_walking_speed_l2333_233373


namespace NUMINAMATH_CALUDE_add_fractions_l2333_233303

theorem add_fractions : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_add_fractions_l2333_233303


namespace NUMINAMATH_CALUDE_problem_solution_l2333_233389

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 6 * x^2 + 12 * x * y + 6 * y^2 = x^3 + 3 * x^2 * y + 3 * x * y^2) :
  x = 24 / 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2333_233389


namespace NUMINAMATH_CALUDE_trains_crossing_time_l2333_233314

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 200) 
  (h2 : length2 = 160) 
  (h3 : speed1 = 68 * 1000 / 3600) 
  (h4 : speed2 = 40 * 1000 / 3600) : 
  (length1 + length2) / (speed1 + speed2) = 12 :=
by sorry

end NUMINAMATH_CALUDE_trains_crossing_time_l2333_233314


namespace NUMINAMATH_CALUDE_tangent_double_angle_identity_l2333_233336

theorem tangent_double_angle_identity (α : Real) (h : 0 < α ∧ α < π/4) : 
  Real.tan (2 * α) / Real.tan α = 1 + 1 / Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_tangent_double_angle_identity_l2333_233336


namespace NUMINAMATH_CALUDE_pass_rate_two_procedures_l2333_233386

theorem pass_rate_two_procedures (a b : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  let pass_rate := (1 - a) * (1 - b)
  0 ≤ pass_rate ∧ pass_rate ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_pass_rate_two_procedures_l2333_233386


namespace NUMINAMATH_CALUDE_total_carrots_l2333_233327

theorem total_carrots (sally_carrots fred_carrots : ℕ) 
  (h1 : sally_carrots = 6) 
  (h2 : fred_carrots = 4) : 
  sally_carrots + fred_carrots = 10 := by
sorry

end NUMINAMATH_CALUDE_total_carrots_l2333_233327


namespace NUMINAMATH_CALUDE_circle_equation_correct_l2333_233328

-- Define the center and radius of the circle
def center : ℝ × ℝ := (1, -2)
def radius : ℝ := 3

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Theorem to prove
theorem circle_equation_correct :
  ∀ x y : ℝ, circle_equation x y ↔ (x - 1)^2 + (y + 2)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l2333_233328


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l2333_233382

theorem exactly_one_greater_than_one (x₁ x₂ x₃ : ℝ) 
  (h_positive : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0)
  (h_product : x₁ * x₂ * x₃ = 1)
  (h_sum : x₁ + x₂ + x₃ > 1/x₁ + 1/x₂ + 1/x₃) :
  (x₁ > 1 ∧ x₂ ≤ 1 ∧ x₃ ≤ 1) ∨
  (x₁ ≤ 1 ∧ x₂ > 1 ∧ x₃ ≤ 1) ∨
  (x₁ ≤ 1 ∧ x₂ ≤ 1 ∧ x₃ > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l2333_233382


namespace NUMINAMATH_CALUDE_brush_cost_is_correct_l2333_233361

/-- The cost of a set of brushes for Maria's painting project -/
def brush_cost : ℝ := 20

/-- The cost of canvas for Maria's painting project -/
def canvas_cost : ℝ := 3 * brush_cost

/-- The cost of paint for Maria's painting project -/
def paint_cost : ℝ := 40

/-- The total cost of materials for Maria's painting project -/
def total_cost : ℝ := brush_cost + canvas_cost + paint_cost

/-- Theorem stating that the brush cost is correct given the problem conditions -/
theorem brush_cost_is_correct :
  brush_cost = 20 ∧
  canvas_cost = 3 * brush_cost ∧
  paint_cost = 40 ∧
  total_cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_brush_cost_is_correct_l2333_233361


namespace NUMINAMATH_CALUDE_b_join_time_l2333_233379

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Represents A's initial investment in Rupees -/
def aInvestment : ℕ := 45000

/-- Represents B's initial investment in Rupees -/
def bInvestment : ℕ := 27000

/-- Represents the ratio of profit sharing between A and B -/
def profitRatio : ℚ := 2 / 1

/-- 
Proves that B joined 2 months after A started the business, given the initial investments
and profit ratio.
-/
theorem b_join_time : 
  ∀ x : ℕ, 
  (aInvestment * monthsInYear) / (bInvestment * (monthsInYear - x)) = profitRatio → 
  x = 2 :=
by sorry

end NUMINAMATH_CALUDE_b_join_time_l2333_233379


namespace NUMINAMATH_CALUDE_expression_nonnegative_l2333_233304

theorem expression_nonnegative (a b c d e : ℝ) : 
  (a-b)*(a-c)*(a-d)*(a-e) + (b-a)*(b-c)*(b-d)*(b-e) + (c-a)*(c-b)*(c-d)*(c-e) +
  (d-a)*(d-b)*(d-c)*(d-e) + (e-a)*(e-b)*(e-c)*(e-d) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonnegative_l2333_233304


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l2333_233332

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3, 
    ∀ y ∈ Set.Ioo (-1 : ℝ) 3, 
      x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l2333_233332


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l2333_233320

theorem quadratic_inequality_always_negative : ∀ x : ℝ, -6 * x^2 + 2 * x - 4 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l2333_233320


namespace NUMINAMATH_CALUDE_total_tickets_sold_l2333_233376

theorem total_tickets_sold (adult_tickets student_tickets : ℕ) 
  (h1 : adult_tickets = 410)
  (h2 : student_tickets = 436) :
  adult_tickets + student_tickets = 846 := by
  sorry

#check total_tickets_sold

end NUMINAMATH_CALUDE_total_tickets_sold_l2333_233376


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_5_l2333_233317

theorem units_digit_of_7_to_5 : 7^5 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_5_l2333_233317


namespace NUMINAMATH_CALUDE_brand_preference_survey_l2333_233302

theorem brand_preference_survey (total : ℕ) (ratio : ℚ) (brand_x : ℕ) : 
  total = 250 → 
  ratio = 4/1 → 
  brand_x = total * (ratio / (1 + ratio)) → 
  brand_x = 200 := by
sorry

end NUMINAMATH_CALUDE_brand_preference_survey_l2333_233302


namespace NUMINAMATH_CALUDE_matching_pair_probability_l2333_233312

def black_socks : ℕ := 12
def blue_socks : ℕ := 10
def total_socks : ℕ := black_socks + blue_socks

def matching_pairs : ℕ := (black_socks * (black_socks - 1)) / 2 + (blue_socks * (blue_socks - 1)) / 2
def total_combinations : ℕ := (total_socks * (total_socks - 1)) / 2

theorem matching_pair_probability :
  (matching_pairs : ℚ) / total_combinations = 111 / 231 := by sorry

end NUMINAMATH_CALUDE_matching_pair_probability_l2333_233312


namespace NUMINAMATH_CALUDE_bianca_extra_flowers_l2333_233374

/-- The number of extra flowers Bianca picked -/
def extra_flowers (tulips roses daffodils sunflowers used : ℕ) : ℕ :=
  tulips + roses + daffodils + sunflowers - used

/-- Proof that Bianca picked 29 extra flowers -/
theorem bianca_extra_flowers :
  extra_flowers 57 73 45 35 181 = 29 := by
  sorry

end NUMINAMATH_CALUDE_bianca_extra_flowers_l2333_233374


namespace NUMINAMATH_CALUDE_positive_difference_problem_l2333_233306

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0))

theorem positive_difference_problem (x : ℕ) 
  (h1 : (45 + x) / 2 = 50) 
  (h2 : is_prime x) : 
  Int.natAbs (x - 45) = 8 := by
sorry

end NUMINAMATH_CALUDE_positive_difference_problem_l2333_233306


namespace NUMINAMATH_CALUDE_continuity_at_seven_l2333_233365

/-- The function f(x) = 4x^2 + 6 is continuous at x₀ = 7 -/
theorem continuity_at_seven (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, |x - 7| < δ → |4 * x^2 + 6 - (4 * 7^2 + 6)| < ε := by
  sorry

end NUMINAMATH_CALUDE_continuity_at_seven_l2333_233365


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l2333_233337

def cricket_team_problem (team_size : ℕ) (avg_age : ℚ) (wicket_keeper_age_diff : ℚ) : Prop :=
  let total_age : ℚ := team_size * avg_age
  let wicket_keeper_age : ℚ := avg_age + wicket_keeper_age_diff
  let remaining_total_age : ℚ := total_age - wicket_keeper_age - avg_age
  let remaining_team_size : ℕ := team_size - 2
  let remaining_avg_age : ℚ := remaining_total_age / remaining_team_size
  (avg_age - remaining_avg_age) = 0.3

theorem cricket_team_age_difference :
  cricket_team_problem 11 24 3 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l2333_233337


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_satisfying_conditions_l2333_233350

theorem smallest_three_digit_number_satisfying_conditions : ∃ n : ℕ,
  (100 ≤ n ∧ n ≤ 999) ∧
  (∃ k : ℕ, n + 7 = 9 * k) ∧
  (∃ m : ℕ, n - 6 = 7 * m) ∧
  (∀ x : ℕ, (100 ≤ x ∧ x < n) → ¬((∃ k : ℕ, x + 7 = 9 * k) ∧ (∃ m : ℕ, x - 6 = 7 * m))) ∧
  n = 116 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_satisfying_conditions_l2333_233350


namespace NUMINAMATH_CALUDE_prob_at_least_three_marbles_l2333_233388

def num_green : ℕ := 5
def num_purple : ℕ := 7
def total_marbles : ℕ := num_green + num_purple
def num_draws : ℕ := 5

def prob_purple : ℚ := num_purple / total_marbles
def prob_green : ℚ := num_green / total_marbles

def prob_exactly (k : ℕ) : ℚ :=
  (Nat.choose num_draws k) * (prob_purple ^ k) * (prob_green ^ (num_draws - k))

def prob_at_least_three : ℚ :=
  prob_exactly 3 + prob_exactly 4 + prob_exactly 5

theorem prob_at_least_three_marbles :
  prob_at_least_three = 162582 / 2985984 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_marbles_l2333_233388


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2333_233355

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 36) (h3 : x > 0) (h4 : y > 0) :
  1 / x + 1 / y = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2333_233355


namespace NUMINAMATH_CALUDE_opposite_roots_imply_k_value_l2333_233300

theorem opposite_roots_imply_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 + (k^2 - 4)*x + (k - 1) = 0 ∧ 
             ∃ y : ℝ, y^2 + (k^2 - 4)*y + (k - 1) = 0 ∧ 
             x = -y ∧ x ≠ y) → 
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_opposite_roots_imply_k_value_l2333_233300


namespace NUMINAMATH_CALUDE_program_output_l2333_233364

theorem program_output : ∀ (a b : ℕ), a = 1 → b = 2 → a + b = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_program_output_l2333_233364


namespace NUMINAMATH_CALUDE_abes_age_l2333_233335

theorem abes_age (present_age : ℕ) : 
  present_age + (present_age - 7) = 29 → present_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_abes_age_l2333_233335


namespace NUMINAMATH_CALUDE_binary_calculation_l2333_233370

theorem binary_calculation : 
  (0b110101 * 0b1101) + 0b1010 = 0b10010111111 := by sorry

end NUMINAMATH_CALUDE_binary_calculation_l2333_233370


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2333_233395

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2333_233395


namespace NUMINAMATH_CALUDE_y_decreases_as_x_increases_l2333_233398

def linear_function (x : ℝ) : ℝ := -3 * x + 6

theorem y_decreases_as_x_increases :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function x₁ > linear_function x₂ := by
  sorry

end NUMINAMATH_CALUDE_y_decreases_as_x_increases_l2333_233398


namespace NUMINAMATH_CALUDE_mike_picked_52_peaches_l2333_233329

/-- The number of peaches Mike picked -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

/-- Proof that Mike picked 52 peaches -/
theorem mike_picked_52_peaches (initial final : ℕ) 
  (h1 : initial = 34) 
  (h2 : final = 86) : 
  peaches_picked initial final = 52 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_52_peaches_l2333_233329


namespace NUMINAMATH_CALUDE_solve_for_y_l2333_233324

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 2 = y - 4) (h2 : x = -3) : y = 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2333_233324


namespace NUMINAMATH_CALUDE_quadratic_one_root_l2333_233390

theorem quadratic_one_root (m : ℝ) : m > 0 ∧ 
  (∃! x : ℝ, x^2 + 6*m*x + 3*m = 0) ↔ m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l2333_233390


namespace NUMINAMATH_CALUDE_miriam_monday_pushups_l2333_233393

/-- Represents the number of push-ups Miriam did on each day of the week --/
structure PushUps where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of push-ups done before Thursday --/
def totalBeforeThursday (p : PushUps) : ℕ :=
  p.monday + p.tuesday + p.wednesday

/-- Represents Miriam's push-up routine for the week --/
def miriamPushUps (monday : ℕ) : PushUps :=
  { monday := monday
  , tuesday := 7
  , wednesday := 2 * 7
  , thursday := (totalBeforeThursday { monday := monday, tuesday := 7, wednesday := 2 * 7, thursday := 0, friday := 0 }) / 2
  , friday := 39
  }

/-- Theorem stating that Miriam did 5 push-ups on Monday --/
theorem miriam_monday_pushups :
  ∃ (p : PushUps), p = miriamPushUps 5 ∧
    p.monday + p.tuesday + p.wednesday + p.thursday = p.friday :=
  sorry

end NUMINAMATH_CALUDE_miriam_monday_pushups_l2333_233393


namespace NUMINAMATH_CALUDE_equation_consequences_l2333_233313

theorem equation_consequences (x y : ℝ) (h : x^2 + y^2 - x*y = 1) :
  (-2 : ℝ) ≤ x + y ∧ x + y ≤ 2 ∧ 2/3 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_consequences_l2333_233313
