import Mathlib

namespace NUMINAMATH_CALUDE_nested_average_equals_29_18_l3100_310014

def average_2 (a b : ℚ) : ℚ := (a + b) / 2

def average_3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem nested_average_equals_29_18 : 
  average_3 (average_3 (-1) 2 3) (average_2 2 3) 1 = 29 / 18 := by
  sorry

end NUMINAMATH_CALUDE_nested_average_equals_29_18_l3100_310014


namespace NUMINAMATH_CALUDE_class_election_votes_l3100_310084

theorem class_election_votes (total_votes : ℕ) (fiona_votes : ℕ) : 
  fiona_votes = 48 → 
  (fiona_votes : ℚ) / total_votes = 2 / 5 → 
  total_votes = 120 := by
sorry

end NUMINAMATH_CALUDE_class_election_votes_l3100_310084


namespace NUMINAMATH_CALUDE_rational_function_equality_l3100_310021

theorem rational_function_equality (x : ℝ) (h : x ≠ 1 ∧ x ≠ -2) : 
  (x^2 + 5) / (x^3 - 3*x + 2) = 1 / (x + 2) + 2 / (x - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_rational_function_equality_l3100_310021


namespace NUMINAMATH_CALUDE_product_cde_value_l3100_310096

theorem product_cde_value (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.75) :
  c * d * e = 1000 := by
  sorry

end NUMINAMATH_CALUDE_product_cde_value_l3100_310096


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l3100_310092

/-- A rectangle inscribed in a semicircle -/
structure InscribedRectangle where
  /-- Length of side MO of the rectangle -/
  mo : ℝ
  /-- Length of MG (equal to KO) -/
  mg : ℝ
  /-- The rectangle is inscribed in a semicircle -/
  inscribed : mo > 0 ∧ mg > 0

/-- The area of the inscribed rectangle is 240 -/
theorem inscribed_rectangle_area
  (rect : InscribedRectangle)
  (h1 : rect.mo = 20)
  (h2 : rect.mg = 12) :
  rect.mo * (rect.mg * rect.mg / rect.mo) = 240 :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l3100_310092


namespace NUMINAMATH_CALUDE_rubble_purchase_l3100_310091

/-- Calculates the remaining money after a purchase. -/
def remaining_money (initial_amount notebook_cost pen_cost : ℚ) : ℚ :=
  initial_amount - (2 * notebook_cost + 2 * pen_cost)

/-- Proves that Rubble will have $4.00 left after his purchase. -/
theorem rubble_purchase : remaining_money 15 4 (3/2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_rubble_purchase_l3100_310091


namespace NUMINAMATH_CALUDE_min_tiles_for_coverage_l3100_310032

-- Define the grid size
def grid_size : ℕ := 8

-- Define the size of small squares
def small_square_size : ℕ := 2

-- Define the number of cells covered by each L-shaped tile
def cells_per_tile : ℕ := 3

-- Calculate the number of small squares in the grid
def num_small_squares : ℕ := (grid_size * grid_size) / (small_square_size * small_square_size)

-- Define the minimum number of cells that need to be covered
def min_cells_to_cover : ℕ := 2 * num_small_squares

-- Define the minimum number of L-shaped tiles needed
def min_tiles_needed : ℕ := (min_cells_to_cover + cells_per_tile - 1) / cells_per_tile

-- Theorem statement
theorem min_tiles_for_coverage : min_tiles_needed = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_coverage_l3100_310032


namespace NUMINAMATH_CALUDE_subtraction_problem_l3100_310069

theorem subtraction_problem (x N V : ℝ) : 
  x = 10 → 3 * x = (N - x) + V → V = 0 → N = 40 := by
sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3100_310069


namespace NUMINAMATH_CALUDE_metal_weight_in_compound_l3100_310028

/-- The molecular weight of the metal element in a compound with formula (OH)2 -/
def metal_weight (total_weight : ℝ) : ℝ :=
  total_weight - 2 * (16 + 1)

/-- Theorem: The molecular weight of the metal element in a compound with formula (OH)2
    and total molecular weight of 171 g/mol is 171 - 2 * (16 + 1) g/mol -/
theorem metal_weight_in_compound : metal_weight 171 = 137 := by
  sorry

end NUMINAMATH_CALUDE_metal_weight_in_compound_l3100_310028


namespace NUMINAMATH_CALUDE_betty_age_l3100_310063

theorem betty_age (albert : ℕ) (mary : ℕ) (betty : ℕ)
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 22) :
  betty = 11 := by
  sorry

end NUMINAMATH_CALUDE_betty_age_l3100_310063


namespace NUMINAMATH_CALUDE_calculate_expression_l3100_310031

theorem calculate_expression : -1^4 - (1 - 0.4) * (1/3) * (2 - 3^2) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3100_310031


namespace NUMINAMATH_CALUDE_book_thickness_theorem_l3100_310023

/-- Calculates the number of pages per inch of thickness for a stack of books --/
def pages_per_inch (num_books : ℕ) (avg_pages : ℕ) (total_thickness : ℕ) : ℕ :=
  (num_books * avg_pages) / total_thickness

/-- Theorem: Given a stack of 6 books with an average of 160 pages each and a total thickness of 12 inches,
    the number of pages per inch of thickness is 80. --/
theorem book_thickness_theorem :
  pages_per_inch 6 160 12 = 80 := by
  sorry

end NUMINAMATH_CALUDE_book_thickness_theorem_l3100_310023


namespace NUMINAMATH_CALUDE_rational_segment_existence_l3100_310049

theorem rational_segment_existence (f : ℚ → ℤ) : ∃ x y : ℚ, f x + f y ≤ 2 * f ((x + y) / 2) := by
  sorry

end NUMINAMATH_CALUDE_rational_segment_existence_l3100_310049


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3100_310090

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 14) → 
  p + q = 69 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3100_310090


namespace NUMINAMATH_CALUDE_quarters_to_nickels_difference_l3100_310034

/-- The difference in money (in nickels) between two people with different numbers of quarters -/
theorem quarters_to_nickels_difference (q : ℚ) : 
  5 * ((7 * q + 2) - (3 * q + 7)) = 20 * (q - 1.25) := by
  sorry

end NUMINAMATH_CALUDE_quarters_to_nickels_difference_l3100_310034


namespace NUMINAMATH_CALUDE_map_area_ratio_map_area_ratio_not_scale_l3100_310046

/-- Represents the scale of a map --/
structure MapScale where
  ratio : ℚ

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℚ
  width : ℚ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℚ := r.length * r.width

/-- Theorem: For a map with scale 1:500, the ratio of map area to actual area is 1:250000 --/
theorem map_area_ratio (scale : MapScale) (map_rect : Rectangle) (actual_rect : Rectangle)
    (h_scale : scale.ratio = 1 / 500)
    (h_length : map_rect.length * 500 = actual_rect.length)
    (h_width : map_rect.width * 500 = actual_rect.width) :
    area map_rect / area actual_rect = 1 / 250000 := by
  sorry

/-- The ratio of map area to actual area is not 1:500 --/
theorem map_area_ratio_not_scale (scale : MapScale) (map_rect : Rectangle) (actual_rect : Rectangle)
    (h_scale : scale.ratio = 1 / 500)
    (h_length : map_rect.length * 500 = actual_rect.length)
    (h_width : map_rect.width * 500 = actual_rect.width) :
    area map_rect / area actual_rect ≠ 1 / 500 := by
  sorry

end NUMINAMATH_CALUDE_map_area_ratio_map_area_ratio_not_scale_l3100_310046


namespace NUMINAMATH_CALUDE_tuesday_necklaces_l3100_310048

/-- The number of beaded necklaces Kylie made on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of beaded bracelets Kylie made on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of beaded earrings Kylie made on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed to make one beaded necklace -/
def beads_per_necklace : ℕ := 20

/-- The number of beads needed to make one beaded bracelet -/
def beads_per_bracelet : ℕ := 10

/-- The number of beads needed to make one beaded earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads Kylie used to make her jewelry -/
def total_beads : ℕ := 325

/-- Theorem: The number of beaded necklaces Kylie made on Tuesday is 2 -/
theorem tuesday_necklaces : 
  (total_beads - (monday_necklaces * beads_per_necklace + 
    wednesday_bracelets * beads_per_bracelet + 
    wednesday_earrings * beads_per_earring)) / beads_per_necklace = 2 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_necklaces_l3100_310048


namespace NUMINAMATH_CALUDE_square_sum_equals_150_l3100_310057

theorem square_sum_equals_150 (u v : ℝ) 
  (h1 : u * (u + v) = 50) 
  (h2 : v * (u + v) = 100) : 
  (u + v)^2 = 150 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_150_l3100_310057


namespace NUMINAMATH_CALUDE_solve_class_problem_l3100_310078

def class_problem (N : ℕ) : Prop :=
  ∃ (taqeesha_score : ℕ),
    N > 1 ∧
    (77 * (N - 1) + taqeesha_score) / N = 78 ∧
    N - 1 = 16

theorem solve_class_problem :
  ∃ (N : ℕ), class_problem N ∧ N = 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_class_problem_l3100_310078


namespace NUMINAMATH_CALUDE_hannah_farm_animals_l3100_310053

def farm_animals (num_pigs : ℕ) : ℕ :=
  let num_cows := 2 * num_pigs - 3
  let num_goats := num_cows + 6
  num_pigs + num_cows + num_goats

theorem hannah_farm_animals :
  farm_animals 10 = 50 :=
by sorry

end NUMINAMATH_CALUDE_hannah_farm_animals_l3100_310053


namespace NUMINAMATH_CALUDE_difference_of_squares_l3100_310005

theorem difference_of_squares (m : ℝ) : m^2 - 9 = (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3100_310005


namespace NUMINAMATH_CALUDE_linear_inequality_solution_l3100_310038

theorem linear_inequality_solution (a : ℝ) : 
  (|2 + 3 * a| = 1) ↔ (a = -1 ∨ a = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_l3100_310038


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l3100_310059

theorem junk_mail_distribution (total_mail : ℕ) (total_houses : ℕ) 
  (h1 : total_mail = 48) (h2 : total_houses = 8) :
  total_mail / total_houses = 6 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l3100_310059


namespace NUMINAMATH_CALUDE_teenager_toddler_ratio_l3100_310089

theorem teenager_toddler_ratio (total_children : ℕ) (toddlers : ℕ) (newborns : ℕ) : 
  total_children = 40 → toddlers = 6 → newborns = 4 → 
  (total_children - toddlers - newborns) / toddlers = 5 := by
  sorry

end NUMINAMATH_CALUDE_teenager_toddler_ratio_l3100_310089


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3100_310050

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 + p^2 - 2*p - 1 = 0) →
  (q^3 + q^2 - 2*q - 1 = 0) →
  (r^3 + r^2 - 2*r - 1 = 0) →
  p*(q-r)^2 + q*(r-p)^2 + r*(p-q)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3100_310050


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3100_310040

/-- The lateral surface area of a cone with an equilateral triangle cross-section --/
theorem cone_lateral_surface_area (r h : Real) : 
  r^2 + h^2 = 1 →  -- Condition for equilateral triangle with side length 2
  r * h = 1/2 →    -- Condition for equilateral triangle with side length 2
  2 * π * r = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3100_310040


namespace NUMINAMATH_CALUDE_solution_set_linear_inequalities_l3100_310073

theorem solution_set_linear_inequalities :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_linear_inequalities_l3100_310073


namespace NUMINAMATH_CALUDE_smallest_valid_club_size_l3100_310003

def is_valid_club_size (N : ℕ) : Prop :=
  N < 50 ∧
  ((N - 5) % 6 = 0 ∨ (N - 5) % 7 = 0) ∧
  N % 8 = 7

theorem smallest_valid_club_size :
  ∀ n : ℕ, is_valid_club_size n → n ≥ 47 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_club_size_l3100_310003


namespace NUMINAMATH_CALUDE_complex_calculation_l3100_310088

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*I) (hb : b = 2 - 3*I) :
  3*a - 4*b = 1 + 18*I := by
sorry

end NUMINAMATH_CALUDE_complex_calculation_l3100_310088


namespace NUMINAMATH_CALUDE_local_extremum_implies_b_minus_a_l3100_310033

/-- A function with a local extremum -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x - b

theorem local_extremum_implies_b_minus_a (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → b - a = 15 := by
  sorry

end NUMINAMATH_CALUDE_local_extremum_implies_b_minus_a_l3100_310033


namespace NUMINAMATH_CALUDE_unique_quadruple_solution_l3100_310025

theorem unique_quadruple_solution :
  ∃! (a b c d : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
    a^2 + b^2 + c^2 + d^2 = 4 ∧
    (a + b + c + d) * (a^4 + b^4 + c^4 + d^4) = 32 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadruple_solution_l3100_310025


namespace NUMINAMATH_CALUDE_mango_seller_loss_percentage_l3100_310035

/-- Calculates the percentage of loss for a fruit seller selling mangoes -/
theorem mango_seller_loss_percentage
  (selling_price : ℝ)
  (profit_price : ℝ)
  (h1 : selling_price = 16)
  (h2 : profit_price = 21.818181818181817)
  (h3 : profit_price = 1.2 * (profit_price / 1.2)) :
  (((profit_price / 1.2) - selling_price) / (profit_price / 1.2)) * 100 = 12 :=
by sorry

end NUMINAMATH_CALUDE_mango_seller_loss_percentage_l3100_310035


namespace NUMINAMATH_CALUDE_discount_percentage_l3100_310016

def coffee_cost : ℝ := 6
def cheesecake_cost : ℝ := 10
def discounted_price : ℝ := 12

theorem discount_percentage : 
  (1 - discounted_price / (coffee_cost + cheesecake_cost)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l3100_310016


namespace NUMINAMATH_CALUDE_circles_common_internal_tangent_l3100_310002

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def circle_O₂ (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = 9

-- Define the center of circle O₂
def center_O₂ : ℝ × ℝ := (3, 3)

-- Define the property of being externally tangent
def externally_tangent (O₁ O₂ : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), O₁ x y ∧ O₂ x y ∧
  ∀ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) → ¬(O₁ x' y' ∧ O₂ x' y')

-- Define the common internal tangent line
def common_internal_tangent (x y : ℝ) : Prop := 3*x + 4*y - 21 = 0

-- State the theorem
theorem circles_common_internal_tangent :
  externally_tangent circle_O₁ circle_O₂ →
  ∀ (x y : ℝ), common_internal_tangent x y ↔
    (∃ (t : ℝ), circle_O₁ (x + t) (y - (3/4)*t) ∧
               circle_O₂ (x - t) (y + (3/4)*t)) :=
sorry

end NUMINAMATH_CALUDE_circles_common_internal_tangent_l3100_310002


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_l3100_310030

theorem sum_mod_thirteen : (5678 + 5679 + 5680 + 5681) % 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_l3100_310030


namespace NUMINAMATH_CALUDE_range_of_a_l3100_310068

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x

-- Define the proposition P
def P (a : ℝ) : Prop := ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂

-- Define the function inside the logarithm
def g (a : ℝ) (x : ℝ) : ℝ := a*x^2 - x + a

-- Define the proposition Q
def Q (a : ℝ) : Prop := ∀ x, g a x > 0

-- Main theorem
theorem range_of_a (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ≤ 1/2 ∨ a > 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3100_310068


namespace NUMINAMATH_CALUDE_complex_number_properties_l3100_310009

theorem complex_number_properties : ∃ (z : ℂ), 
  z = 2 / (1 - Complex.I) ∧ 
  Complex.abs z = Real.sqrt 2 ∧
  z^2 = 2 * Complex.I ∧
  z^2 - 2*z + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3100_310009


namespace NUMINAMATH_CALUDE_election_margin_l3100_310020

theorem election_margin (total_votes : ℕ) (vote_swing : ℕ) (final_margin_percent : ℚ) : 
  total_votes = 15000 →
  vote_swing = 3000 →
  final_margin_percent = 20 →
  let initial_winner_votes := (total_votes + vote_swing) / 2 + vote_swing / 2
  let initial_loser_votes := (total_votes - vote_swing) / 2 - vote_swing / 2
  let initial_margin := initial_winner_votes - initial_loser_votes
  initial_margin * 100 / total_votes = final_margin_percent :=
by sorry

end NUMINAMATH_CALUDE_election_margin_l3100_310020


namespace NUMINAMATH_CALUDE_amys_final_money_l3100_310083

def amys_money (initial_amount : ℚ) (chore_payment : ℚ) (num_neighbors : ℕ) 
  (birthday_money : ℚ) (investment_percentage : ℚ) (investment_return : ℚ) 
  (toy_cost : ℚ) (grandparent_multiplier : ℚ) (donation_percentage : ℚ) : ℚ :=
  let total_before_investment := initial_amount + chore_payment * num_neighbors + birthday_money
  let invested_amount := total_before_investment * investment_percentage
  let investment_value := invested_amount * (1 + investment_return)
  let remaining_after_toy := total_before_investment - toy_cost
  let after_grandparent_gift := remaining_after_toy * grandparent_multiplier
  let total_before_donation := after_grandparent_gift + investment_value
  let final_amount := total_before_donation * (1 - donation_percentage)
  final_amount

theorem amys_final_money :
  amys_money 2 13 5 3 (20/100) (10/100) 12 2 (25/100) = 98.55 := by
  sorry

end NUMINAMATH_CALUDE_amys_final_money_l3100_310083


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3100_310074

/-- An isosceles triangle with sides 6 and 3 has perimeter 15 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 6 → b = 3 → c = 6 →  -- Two sides are 6, one is 3
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a = c →  -- Isosceles condition
  a + b + c = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3100_310074


namespace NUMINAMATH_CALUDE_corn_height_after_three_weeks_l3100_310058

/-- The height of corn plants after three weeks of growth -/
def corn_height (initial_height week1_growth : ℕ) : ℕ :=
  let week2_growth := 2 * week1_growth
  let week3_growth := 4 * week2_growth
  initial_height + week1_growth + week2_growth + week3_growth

/-- Theorem stating that the corn height after three weeks is 22 inches -/
theorem corn_height_after_three_weeks :
  corn_height 0 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_corn_height_after_three_weeks_l3100_310058


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l3100_310075

/-- The probability of stopping at box n, given n ≥ 50 -/
def Q (n : ℕ) : ℚ := 2 / (n + 2)

/-- The smallest n ≥ 50 such that Q(n) < 1/2023 is 1011 -/
theorem smallest_n_for_Q_less_than_threshold : 
  (∀ k : ℕ, k ≥ 50 → k < 1011 → Q k ≥ 1/2023) ∧ 
  (Q 1011 < 1/2023) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l3100_310075


namespace NUMINAMATH_CALUDE_inequality_properties_l3100_310093

theorem inequality_properties (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  (a + b < a * b) ∧
  (abs a ≤ abs b) ∧
  (a ≥ b) ∧
  (b/a + a/b > 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_properties_l3100_310093


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3100_310010

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Theorem statement
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -2 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3100_310010


namespace NUMINAMATH_CALUDE_lunch_scores_pigeonhole_l3100_310022

theorem lunch_scores_pigeonhole (n : ℕ) (scores : Fin n → ℕ) 
  (h1 : ∀ i : Fin n, scores i < n) : 
  ∃ i j : Fin n, i ≠ j ∧ scores i = scores j :=
by
  sorry

end NUMINAMATH_CALUDE_lunch_scores_pigeonhole_l3100_310022


namespace NUMINAMATH_CALUDE_mikes_games_l3100_310086

/-- Given Mike's earnings, expenses, and game cost, prove the number of games he can buy -/
theorem mikes_games (earnings : ℕ) (blade_cost : ℕ) (game_cost : ℕ) 
  (h1 : earnings = 101)
  (h2 : blade_cost = 47)
  (h3 : game_cost = 6) :
  (earnings - blade_cost) / game_cost = 9 := by
  sorry

end NUMINAMATH_CALUDE_mikes_games_l3100_310086


namespace NUMINAMATH_CALUDE_water_added_calculation_l3100_310008

def initial_volume : ℝ := 340
def initial_water_percentage : ℝ := 0.80
def initial_kola_percentage : ℝ := 0.06
def added_sugar : ℝ := 3.2
def added_kola : ℝ := 6.8
def final_sugar_percentage : ℝ := 0.14111111111111112

theorem water_added_calculation (water_added : ℝ) : 
  let initial_sugar_percentage := 1 - initial_water_percentage - initial_kola_percentage
  let initial_sugar := initial_sugar_percentage * initial_volume
  let total_sugar := initial_sugar + added_sugar
  let final_volume := initial_volume + water_added + added_sugar + added_kola
  final_sugar_percentage * final_volume = total_sugar →
  water_added = 10 := by sorry

end NUMINAMATH_CALUDE_water_added_calculation_l3100_310008


namespace NUMINAMATH_CALUDE_base_10_satisfies_equation_l3100_310098

def base_x_addition (x : ℕ) (a b c : ℕ) : Prop :=
  a + b = c

def to_base_10 (x : ℕ) (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 * x^3 + d2 * x^2 + d3 * x + d4

theorem base_10_satisfies_equation : 
  ∃ x : ℕ, x > 1 ∧ base_x_addition x 
    (to_base_10 x 8374) 
    (to_base_10 x 6250) 
    (to_base_10 x 15024) :=
by
  sorry

end NUMINAMATH_CALUDE_base_10_satisfies_equation_l3100_310098


namespace NUMINAMATH_CALUDE_desk_purchase_price_l3100_310065

/-- Given a desk with a selling price that includes a 25% markup and results in a gross profit of $33.33, prove that the purchase price of the desk is $99.99. -/
theorem desk_purchase_price (purchase_price selling_price : ℝ) : 
  selling_price = purchase_price + 0.25 * selling_price →
  selling_price - purchase_price = 33.33 →
  purchase_price = 99.99 := by
sorry

end NUMINAMATH_CALUDE_desk_purchase_price_l3100_310065


namespace NUMINAMATH_CALUDE_f_composition_negative_three_equals_zero_l3100_310080

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x + 2/x - 3
  else Real.log (x^2 + 1) / Real.log 10

-- State the theorem
theorem f_composition_negative_three_equals_zero :
  f (f (-3)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_three_equals_zero_l3100_310080


namespace NUMINAMATH_CALUDE_remainder_27_pow_27_plus_27_mod_28_l3100_310015

theorem remainder_27_pow_27_plus_27_mod_28 :
  (27^27 + 27) % 28 = 26 := by
sorry

end NUMINAMATH_CALUDE_remainder_27_pow_27_plus_27_mod_28_l3100_310015


namespace NUMINAMATH_CALUDE_fraction_problem_l3100_310067

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/3) * F * N = 18) 
  (h2 : (3/10) * N = 64.8) : 
  F = 1/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3100_310067


namespace NUMINAMATH_CALUDE_no_overlap_for_y_l3100_310004

theorem no_overlap_for_y (y : ℝ) : 
  200 ≤ y ∧ y ≤ 300 → 
  ⌊Real.sqrt y⌋ = 16 → 
  ⌊Real.sqrt (50 * y)⌋ ≠ 226 := by
sorry

end NUMINAMATH_CALUDE_no_overlap_for_y_l3100_310004


namespace NUMINAMATH_CALUDE_reporters_not_covering_politics_l3100_310076

/-- The percentage of reporters who cover local politics in country X -/
def local_politics_coverage : ℝ := 30

/-- The percentage of reporters who cover politics but not local politics in country X -/
def non_local_politics_coverage : ℝ := 25

/-- Theorem stating that 60% of reporters do not cover politics -/
theorem reporters_not_covering_politics :
  let total_reporters : ℝ := 100
  let reporters_covering_local_politics : ℝ := local_politics_coverage
  let reporters_covering_politics : ℝ := reporters_covering_local_politics / (1 - non_local_politics_coverage / 100)
  let reporters_not_covering_politics : ℝ := total_reporters - reporters_covering_politics
  reporters_not_covering_politics / total_reporters = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_reporters_not_covering_politics_l3100_310076


namespace NUMINAMATH_CALUDE_expected_wealth_difference_10_days_l3100_310095

/-- Represents the daily outcome for an agent --/
inductive DailyOutcome
  | Win
  | Lose
  | Reset

/-- Represents the state of wealth for both agents --/
structure WealthState :=
  (cat : ℤ)
  (fox : ℤ)

/-- Defines the probability distribution for daily outcomes --/
def dailyProbability : DailyOutcome → ℝ
  | DailyOutcome.Win => 0.25
  | DailyOutcome.Lose => 0.25
  | DailyOutcome.Reset => 0.5

/-- Updates the wealth state based on the daily outcome --/
def updateWealth (state : WealthState) (outcome : DailyOutcome) : WealthState :=
  match outcome with
  | DailyOutcome.Win => { cat := state.cat + 1, fox := state.fox }
  | DailyOutcome.Lose => { cat := state.cat, fox := state.fox + 1 }
  | DailyOutcome.Reset => { cat := 0, fox := 0 }

/-- Calculates the expected value of the absolute difference in wealth after n days --/
def expectedWealthDifference (n : ℕ) : ℝ :=
  sorry

theorem expected_wealth_difference_10_days :
  expectedWealthDifference 10 = 1 :=
sorry

end NUMINAMATH_CALUDE_expected_wealth_difference_10_days_l3100_310095


namespace NUMINAMATH_CALUDE_students_registered_correct_registration_l3100_310047

theorem students_registered (students_yesterday : ℕ) (absent_today : ℕ) : ℕ :=
  let twice_yesterday := 2 * students_yesterday
  let ten_percent := twice_yesterday / 10
  let attending_today := twice_yesterday - ten_percent
  let total_registered := attending_today + absent_today
  total_registered

theorem correct_registration : students_registered 70 30 = 156 := by
  sorry

end NUMINAMATH_CALUDE_students_registered_correct_registration_l3100_310047


namespace NUMINAMATH_CALUDE_max_sum_of_diagonals_l3100_310011

/-- A rhombus with side length 5 and diagonals d1 and d2 where d1 ≤ 6 and d2 ≥ 6 -/
structure Rhombus where
  side_length : ℝ
  d1 : ℝ
  d2 : ℝ
  side_is_5 : side_length = 5
  d1_le_6 : d1 ≤ 6
  d2_ge_6 : d2 ≥ 6

/-- The maximum sum of diagonals in the given rhombus is 14 -/
theorem max_sum_of_diagonals (r : Rhombus) : (r.d1 + r.d2 ≤ 14) ∧ (∃ (s : Rhombus), s.d1 + s.d2 = 14) := by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_diagonals_l3100_310011


namespace NUMINAMATH_CALUDE_starting_lineup_count_l3100_310066

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def special_players : ℕ := 3

theorem starting_lineup_count : 
  (lineup_size.choose (team_size - special_players)) + 
  (special_players * (lineup_size - 1).choose (team_size - special_players)) = 2277 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l3100_310066


namespace NUMINAMATH_CALUDE_min_value_implications_l3100_310081

theorem min_value_implications (a b : ℝ) 
  (h_a : a > 0) (h_b : b > 0) 
  (h_min : ∀ x, |x + a| + |x - b| ≥ 2) : 
  (3 * a^2 + b^2 ≥ 3) ∧ (4 / (a + 1) + 1 / b ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_implications_l3100_310081


namespace NUMINAMATH_CALUDE_crease_line_theorem_l3100_310045

/-- Given a circle with radius R and a point A inside the circle at distance a from the center,
    this function represents the set of points (x, y) on the crease lines formed by folding
    points on the circumference onto A. -/
def creaseLinePoints (R a : ℝ) (x y : ℝ) : Prop :=
  (x - a/2)^2 / (R/2)^2 + y^2 / ((R/2)^2 - (a/2)^2) ≥ 1

/-- Theorem stating that the set of points on the crease lines satisfies the given inequality. -/
theorem crease_line_theorem (R a : ℝ) (h₁ : R > 0) (h₂ : 0 < a ∧ a < R) :
  ∀ x y : ℝ, creaseLinePoints R a x y ↔
    ∃ A' : ℝ × ℝ, (A'.1 - R)^2 + A'.2^2 = R^2 ∧
      (x - a/2)^2 + y^2 = (x - A'.1/2)^2 + (y - A'.2/2)^2 :=
by sorry

end NUMINAMATH_CALUDE_crease_line_theorem_l3100_310045


namespace NUMINAMATH_CALUDE_aarti_work_completion_l3100_310041

/-- The number of days Aarti needs to complete one piece of work -/
def days_for_one_work : ℕ := 5

/-- The number of times the work is multiplied -/
def work_multiplier : ℕ := 3

/-- Theorem: Aarti will complete three times the work in 15 days -/
theorem aarti_work_completion :
  days_for_one_work * work_multiplier = 15 := by
  sorry

end NUMINAMATH_CALUDE_aarti_work_completion_l3100_310041


namespace NUMINAMATH_CALUDE_probability_kings_or_aces_l3100_310072

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Number of cards drawn -/
def CardsDrawn : ℕ := 3

/-- Probability of drawing three kings or at least two aces -/
def prob_kings_or_aces : ℚ := 6 / 425

/-- Theorem stating the probability of drawing three kings or at least two aces -/
theorem probability_kings_or_aces :
  (KingsInDeck.choose CardsDrawn) / (StandardDeck.choose CardsDrawn) +
  ((AcesInDeck.choose 2 * (StandardDeck - AcesInDeck).choose 1) +
   AcesInDeck.choose 3) / (StandardDeck.choose CardsDrawn) = prob_kings_or_aces := by
  sorry

end NUMINAMATH_CALUDE_probability_kings_or_aces_l3100_310072


namespace NUMINAMATH_CALUDE_saras_baking_days_l3100_310097

/-- Proves the number of weekdays Sara makes cakes given the problem conditions -/
theorem saras_baking_days (cakes_per_day : ℕ) (price_per_cake : ℕ) (total_collected : ℕ) 
  (h1 : cakes_per_day = 4)
  (h2 : price_per_cake = 8)
  (h3 : total_collected = 640) :
  total_collected / price_per_cake / cakes_per_day = 20 := by
  sorry

end NUMINAMATH_CALUDE_saras_baking_days_l3100_310097


namespace NUMINAMATH_CALUDE_tank_weight_l3100_310087

/-- Given a tank with the following properties:
  * When four-fifths full, it weighs p kilograms
  * When two-thirds full, it weighs q kilograms
  * The empty tank and other contents weigh r kilograms
  Prove that the total weight of the tank when completely full is (5/2)p + (3/2)q -/
theorem tank_weight (p q r : ℝ) : 
  (∃ (x y : ℝ), x + (4/5) * y = p ∧ x + (2/3) * y = q ∧ x = r) →
  (∃ (z : ℝ), z = (5/2) * p + (3/2) * q ∧ 
    (∀ (x y : ℝ), x + (4/5) * y = p ∧ x + (2/3) * y = q → x + y = z)) :=
by sorry

end NUMINAMATH_CALUDE_tank_weight_l3100_310087


namespace NUMINAMATH_CALUDE_erasers_problem_l3100_310055

theorem erasers_problem (initial_erasers bought_erasers final_erasers : ℕ) : 
  bought_erasers = 42 ∧ final_erasers = 137 → initial_erasers = 95 :=
by sorry

end NUMINAMATH_CALUDE_erasers_problem_l3100_310055


namespace NUMINAMATH_CALUDE_unique_solution_l3100_310064

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (heq : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 12 ∧ y = 10 ∧ z = 8 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l3100_310064


namespace NUMINAMATH_CALUDE_halfway_between_one_fourth_and_one_seventh_l3100_310079

/-- The fraction halfway between two fractions is their average -/
def halfway (a b : ℚ) : ℚ := (a + b) / 2

/-- The fraction halfway between 1/4 and 1/7 is 11/56 -/
theorem halfway_between_one_fourth_and_one_seventh :
  halfway (1/4) (1/7) = 11/56 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_fourth_and_one_seventh_l3100_310079


namespace NUMINAMATH_CALUDE_triangle_side_length_l3100_310027

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  a = 2 →
  B = π / 3 →
  c = 3 →
  b = Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3100_310027


namespace NUMINAMATH_CALUDE_rectangular_park_area_l3100_310044

theorem rectangular_park_area :
  ∀ (width length : ℝ),
    length = 4 * width + 15 →
    2 * (length + width) = 780 →
    width * length = 23625 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_park_area_l3100_310044


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solutions_equation_three_solution_l3100_310070

-- Equation 1
theorem equation_one_solution (x : ℝ) :
  (x^2 + 2) * |2*x - 5| = 0 ↔ x = 5/2 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  (x - 3)^3 * x = 0 ↔ x = 0 ∨ x = 3 := by sorry

-- Equation 3
theorem equation_three_solution (x : ℝ) :
  |x^4 + 1| = x^4 + x ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solutions_equation_three_solution_l3100_310070


namespace NUMINAMATH_CALUDE_westward_plane_speed_l3100_310012

/-- Given two planes traveling in opposite directions, this theorem calculates
    the speed of the westward-traveling plane. -/
theorem westward_plane_speed
  (east_speed : ℝ)
  (time : ℝ)
  (total_distance : ℝ)
  (h1 : east_speed = 325)
  (h2 : time = 3.5)
  (h3 : total_distance = 2100)
  : ∃ (west_speed : ℝ),
    west_speed = 275 ∧
    total_distance = (east_speed + west_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_westward_plane_speed_l3100_310012


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3100_310061

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 1 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 3 / 5 →
  n + k = 8 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3100_310061


namespace NUMINAMATH_CALUDE_abs_z_equals_sqrt_two_l3100_310001

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem abs_z_equals_sqrt_two (z : ℂ) (h : z * (1 + i) = 2 * i) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_sqrt_two_l3100_310001


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3100_310029

theorem rectangular_field_area (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : y = 9) : 
  x * y = 99 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3100_310029


namespace NUMINAMATH_CALUDE_range_of_m_l3100_310054

/-- The proposition p: "The equation x^2 + 2mx + 1 = 0 has two distinct positive roots" -/
def p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + 2*m*x₁ + 1 = 0 ∧ x₂^2 + 2*m*x₂ + 1 = 0

/-- The proposition q: "The equation x^2 + 2(m-2)x - 3m + 10 = 0 has no real roots" -/
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2*(m-2)*x - 3*m + 10 ≠ 0

/-- The set representing the range of m -/
def S : Set ℝ := {m | m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)}

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ S := by sorry

end NUMINAMATH_CALUDE_range_of_m_l3100_310054


namespace NUMINAMATH_CALUDE_shirt_difference_l3100_310099

theorem shirt_difference (alex_shirts joe_shirts ben_shirts : ℕ) : 
  alex_shirts = 4 → 
  ben_shirts = 15 → 
  ben_shirts = joe_shirts + 8 → 
  joe_shirts - alex_shirts = 3 := by
sorry

end NUMINAMATH_CALUDE_shirt_difference_l3100_310099


namespace NUMINAMATH_CALUDE_green_peppers_weight_l3100_310082

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_peppers : ℝ := 0.3333333333333333

/-- The weight of red peppers bought by Hannah's Vegetarian Restaurant -/
def red_peppers : ℝ := 0.3333333333333333

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_peppers : ℝ := 0.6666666666666666

/-- Theorem stating that the weight of green peppers is the difference between
    the total weight of peppers and the weight of red peppers -/
theorem green_peppers_weight :
  green_peppers = total_peppers - red_peppers := by
  sorry

end NUMINAMATH_CALUDE_green_peppers_weight_l3100_310082


namespace NUMINAMATH_CALUDE_book_page_difference_l3100_310094

/-- The number of pages in Selena's book -/
def selena_pages : ℕ := 400

/-- The number of pages in Harry's book -/
def harry_pages : ℕ := 180

/-- The difference between half of Selena's pages and Harry's pages -/
def page_difference : ℕ := selena_pages / 2 - harry_pages

theorem book_page_difference : page_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_book_page_difference_l3100_310094


namespace NUMINAMATH_CALUDE_sum_equals_350_l3100_310071

theorem sum_equals_350 : 247 + 53 + 47 + 3 = 350 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_350_l3100_310071


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3100_310018

/-- The area of a square inscribed in the ellipse x^2/5 + y^2/10 = 1, with its diagonals parallel to the coordinate axes, is 40/3. -/
theorem inscribed_square_area (x y : ℝ) :
  (x^2 / 5 + y^2 / 10 = 1) →  -- ellipse equation
  (∃ (a : ℝ), x = a ∧ y = a) →  -- square vertices on the ellipse
  (40 : ℝ) / 3 = 4 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3100_310018


namespace NUMINAMATH_CALUDE_sandy_fish_count_l3100_310052

def initial_fish : ℕ := 26
def bought_fish : ℕ := 6

theorem sandy_fish_count : initial_fish + bought_fish = 32 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fish_count_l3100_310052


namespace NUMINAMATH_CALUDE_red_rose_theatre_ticket_sales_l3100_310013

theorem red_rose_theatre_ticket_sales 
  (price_low : ℝ) 
  (price_high : ℝ) 
  (total_sales : ℝ) 
  (low_price_tickets : ℕ) 
  (h1 : price_low = 4.5)
  (h2 : price_high = 6)
  (h3 : total_sales = 1972.5)
  (h4 : low_price_tickets = 205) :
  ∃ (high_price_tickets : ℕ),
    (low_price_tickets : ℝ) * price_low + (high_price_tickets : ℝ) * price_high = total_sales ∧
    low_price_tickets + high_price_tickets = 380 :=
by sorry

end NUMINAMATH_CALUDE_red_rose_theatre_ticket_sales_l3100_310013


namespace NUMINAMATH_CALUDE_gumball_packages_l3100_310051

theorem gumball_packages (gumballs_per_package : ℕ) (gumballs_eaten : ℕ) : 
  gumballs_per_package = 5 → gumballs_eaten = 20 → 
  (gumballs_eaten / gumballs_per_package : ℕ) = 4 := by
sorry

end NUMINAMATH_CALUDE_gumball_packages_l3100_310051


namespace NUMINAMATH_CALUDE_zeros_of_f_l3100_310037

def f (x : ℝ) := -x^2 + 5*x - 6

theorem zeros_of_f :
  ∃ (a b : ℝ), (∀ x, f x = 0 ↔ x = a ∨ x = b) ∧ a = 2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_l3100_310037


namespace NUMINAMATH_CALUDE_calculator_price_proof_l3100_310042

theorem calculator_price_proof (total_calculators : ℕ) (total_sales : ℕ) 
  (first_type_count : ℕ) (first_type_price : ℕ) (second_type_count : ℕ) :
  total_calculators = 85 →
  total_sales = 3875 →
  first_type_count = 35 →
  first_type_price = 15 →
  second_type_count = total_calculators - first_type_count →
  (first_type_count * first_type_price + second_type_count * 67 = total_sales) :=
by
  sorry

#check calculator_price_proof

end NUMINAMATH_CALUDE_calculator_price_proof_l3100_310042


namespace NUMINAMATH_CALUDE_expand_expression_l3100_310062

theorem expand_expression (x y : ℝ) : (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3100_310062


namespace NUMINAMATH_CALUDE_younger_person_age_l3100_310000

theorem younger_person_age (y e : ℕ) : 
  e = y + 20 → 
  e - 4 = 5 * (y - 4) → 
  y = 9 := by
sorry

end NUMINAMATH_CALUDE_younger_person_age_l3100_310000


namespace NUMINAMATH_CALUDE_inequality_proof_l3100_310006

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_3 : a + b + c = 3) :
  (a^2 + 9) / (2*a^2 + (b + c)^2) + (b^2 + 9) / (2*b^2 + (c + a)^2) + (c^2 + 9) / (2*c^2 + (a + b)^2) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3100_310006


namespace NUMINAMATH_CALUDE_correct_calculation_l3100_310036

theorem correct_calculation : -5 * (-4) * (-2) * (-2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3100_310036


namespace NUMINAMATH_CALUDE_intersection_equals_universal_set_l3100_310007

theorem intersection_equals_universal_set {α : Type*} (S A B : Set α) 
  (h_universal : ∀ x, x ∈ S) 
  (h_intersection : A ∩ B = S) : 
  A = S ∧ B = S := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_universal_set_l3100_310007


namespace NUMINAMATH_CALUDE_celestia_badges_l3100_310085

theorem celestia_badges (total : ℕ) (hermione : ℕ) (luna : ℕ) (celestia : ℕ)
  (h_total : total = 83)
  (h_hermione : hermione = 14)
  (h_luna : luna = 17)
  (h_sum : total = hermione + luna + celestia) :
  celestia = 52 := by
sorry

end NUMINAMATH_CALUDE_celestia_badges_l3100_310085


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l3100_310019

theorem smallest_distance_between_complex_numbers (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 2*Complex.I) = 2)
  (hw : Complex.abs (w - 5 - 6*Complex.I) = 2) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 113 - 4 ∧ 
    ∀ (z' w' : ℂ), Complex.abs (z' + 2 + 2*Complex.I) = 2 →
      Complex.abs (w' - 5 - 6*Complex.I) = 2 →
      Complex.abs (z' - w') ≥ min_dist := by
  sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l3100_310019


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3100_310043

theorem log_equality_implies_ratio_one (p q : ℝ) 
  (hp : p > 0) (hq : q > 0)
  (h : Real.log p / Real.log 4 = Real.log q / Real.log 6 ∧ 
       Real.log q / Real.log 6 = Real.log (p * q) / Real.log 8) : 
  q / p = 1 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3100_310043


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3100_310024

theorem rectangle_dimensions : ∀ w l : ℝ,
  w > 0 →
  l = 2 * w →
  2 * (l + w) = 3 * (l * w) →
  w = 1 ∧ l = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3100_310024


namespace NUMINAMATH_CALUDE_calm_snakes_not_blue_l3100_310026

-- Define the universe of snakes
variable (Snake : Type)

-- Define properties of snakes
variable (isBlue : Snake → Prop)
variable (isCalm : Snake → Prop)
variable (canMultiply : Snake → Prop)
variable (canDivide : Snake → Prop)

-- State the theorem
theorem calm_snakes_not_blue 
  (h1 : ∀ s : Snake, isCalm s → canMultiply s)
  (h2 : ∀ s : Snake, isBlue s → ¬canDivide s)
  (h3 : ∀ s : Snake, ¬canDivide s → ¬canMultiply s) :
  ∀ s : Snake, isCalm s → ¬isBlue s :=
by
  sorry


end NUMINAMATH_CALUDE_calm_snakes_not_blue_l3100_310026


namespace NUMINAMATH_CALUDE_digit_swap_difference_multiple_of_nine_l3100_310077

theorem digit_swap_difference_multiple_of_nine (a b : ℕ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) : 
  ∃ k : ℤ, (10 * a + b) - (10 * b + a) = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_digit_swap_difference_multiple_of_nine_l3100_310077


namespace NUMINAMATH_CALUDE_system_solution_l3100_310056

theorem system_solution :
  let x : ℚ := -89/43
  let y : ℚ := -202/129
  (4 * x - 3 * y = -14) ∧ (5 * x + 7 * y = 3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3100_310056


namespace NUMINAMATH_CALUDE_correct_seat_increase_l3100_310039

/-- Represents the seating arrangement in a theater --/
structure Theater where
  first_row_seats : ℕ
  last_row_seats : ℕ
  total_seats : ℕ
  seat_increase_per_row : ℕ

/-- Calculates the number of rows in the theater --/
def num_rows (t : Theater) : ℕ :=
  (t.last_row_seats - t.first_row_seats) / t.seat_increase_per_row + 1

/-- Calculates the sum of seats in all rows --/
def sum_of_seats (t : Theater) : ℕ :=
  (num_rows t * (t.first_row_seats + t.last_row_seats)) / 2

/-- Theorem stating the correct seat increase per row --/
theorem correct_seat_increase (t : Theater) 
  (h1 : t.first_row_seats = 12)
  (h2 : t.last_row_seats = 48)
  (h3 : t.total_seats = 570)
  (h4 : sum_of_seats t = t.total_seats) :
  t.seat_increase_per_row = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_seat_increase_l3100_310039


namespace NUMINAMATH_CALUDE_number_of_baskets_l3100_310017

theorem number_of_baskets (green_per_basket : ℕ) (total_green : ℕ) (h1 : green_per_basket = 2) (h2 : total_green = 14) :
  total_green / green_per_basket = 7 := by
sorry

end NUMINAMATH_CALUDE_number_of_baskets_l3100_310017


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l3100_310060

theorem inscribed_triangle_area (r : ℝ) (A B C : ℝ) :
  r = 18 / Real.pi →
  A = 60 * Real.pi / 180 →
  B = 120 * Real.pi / 180 →
  C = 180 * Real.pi / 180 →
  (1/2) * r^2 * (Real.sin A + Real.sin B + Real.sin C) = 162 * Real.sqrt 3 / Real.pi^2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l3100_310060
