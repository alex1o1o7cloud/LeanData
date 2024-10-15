import Mathlib

namespace NUMINAMATH_GPT_find_interest_rate_l1416_141669

noncomputable def interest_rate_solution : ℝ :=
  let P := 800
  let A := 1760
  let t := 4
  let n := 1
  (A / P) ^ (1 / (n * t)) - 1

theorem find_interest_rate : interest_rate_solution = 0.1892 := 
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l1416_141669


namespace NUMINAMATH_GPT_browser_usage_information_is_false_l1416_141653

def num_people_using_A : ℕ := 316
def num_people_using_B : ℕ := 478
def num_people_using_both_A_and_B : ℕ := 104
def num_people_only_using_one_browser : ℕ := 567

theorem browser_usage_information_is_false :
  num_people_only_using_one_browser ≠ (num_people_using_A - num_people_using_both_A_and_B) + (num_people_using_B - num_people_using_both_A_and_B) :=
by
  sorry

end NUMINAMATH_GPT_browser_usage_information_is_false_l1416_141653


namespace NUMINAMATH_GPT_staircase_steps_l1416_141662

theorem staircase_steps (x : ℕ) (h1 : x + 2 * x + (2 * x - 10) = 2 * 45) : x = 20 :=
by 
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_staircase_steps_l1416_141662


namespace NUMINAMATH_GPT_linda_coats_l1416_141646

variable (wall_area : ℝ) (cover_per_gallon : ℝ) (gallons_bought : ℝ)

theorem linda_coats (h1 : wall_area = 600)
                    (h2 : cover_per_gallon = 400)
                    (h3 : gallons_bought = 3) :
  (gallons_bought / (wall_area / cover_per_gallon)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_linda_coats_l1416_141646


namespace NUMINAMATH_GPT_number_of_boxes_of_nectarines_l1416_141677

namespace ProofProblem

/-- Define the given conditions: -/
def crates : Nat := 12
def oranges_per_crate : Nat := 150
def nectarines_per_box : Nat := 30
def total_fruit : Nat := 2280

/-- Define the number of oranges: -/
def total_oranges : Nat := crates * oranges_per_crate

/-- Calculate the number of nectarines: -/
def total_nectarines : Nat := total_fruit - total_oranges

/-- Calculate the number of boxes of nectarines: -/
def boxes_of_nectarines : Nat := total_nectarines / nectarines_per_box

-- Theorem stating that given the conditions, the number of boxes of nectarines is 16.
theorem number_of_boxes_of_nectarines :
  boxes_of_nectarines = 16 := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_number_of_boxes_of_nectarines_l1416_141677


namespace NUMINAMATH_GPT_sally_weekly_bread_l1416_141664

-- Define the conditions
def monday_bread : Nat := 3
def tuesday_bread : Nat := 2
def wednesday_bread : Nat := 4
def thursday_bread : Nat := 2
def friday_bread : Nat := 1
def saturday_bread : Nat := 2 * 2  -- 2 sandwiches, 2 pieces each
def sunday_bread : Nat := 2

-- Define the total bread count
def total_bread : Nat := 
  monday_bread + 
  tuesday_bread + 
  wednesday_bread + 
  thursday_bread + 
  friday_bread + 
  saturday_bread + 
  sunday_bread

-- The proof statement
theorem sally_weekly_bread : total_bread = 18 := by
  sorry

end NUMINAMATH_GPT_sally_weekly_bread_l1416_141664


namespace NUMINAMATH_GPT_solve_for_x_l1416_141654

theorem solve_for_x (x : ℝ) (h : (3 * x - 17) / 4 = (x + 12) / 5) : x = 12.09 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1416_141654


namespace NUMINAMATH_GPT_original_price_of_wand_l1416_141675

theorem original_price_of_wand (x : ℝ) (h : x / 8 = 12) : x = 96 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_wand_l1416_141675


namespace NUMINAMATH_GPT_initial_investment_l1416_141697

theorem initial_investment
  (P r : ℝ)
  (h1 : P + (P * r * 2) / 100 = 600)
  (h2 : P + (P * r * 7) / 100 = 850) :
  P = 500 :=
sorry

end NUMINAMATH_GPT_initial_investment_l1416_141697


namespace NUMINAMATH_GPT_find_d_div_a_l1416_141610
noncomputable def quad_to_square_form (x : ℝ) : ℝ :=
  x^2 + 1500 * x + 1800

theorem find_d_div_a : 
  ∃ (a d : ℝ), (∀ x : ℝ, quad_to_square_form x = (x + a)^2 + d) 
  ∧ a = 750 
  ∧ d = -560700 
  ∧ d / a = -560700 / 750 := 
sorry

end NUMINAMATH_GPT_find_d_div_a_l1416_141610


namespace NUMINAMATH_GPT_sin_identity_l1416_141638

theorem sin_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 4) :
  Real.sin (2 * α + π / 6) = 7 / 8 := 
by
  sorry

end NUMINAMATH_GPT_sin_identity_l1416_141638


namespace NUMINAMATH_GPT_symmetric_point_condition_l1416_141631

theorem symmetric_point_condition (a b : ℝ) (l : ℝ → ℝ → Prop) 
  (H_line: ∀ x y, l x y ↔ x + y + 1 = 0)
  (H_symmetric: l a b ∧ l (2*(-a-1) + a) (2*(-b-1) + b))
  : a + b = -1 :=
by 
  sorry

end NUMINAMATH_GPT_symmetric_point_condition_l1416_141631


namespace NUMINAMATH_GPT_bus_waiting_probability_l1416_141680

-- Definitions
def arrival_time_range := (0, 90)  -- minutes from 1:00 to 2:30
def bus_wait_time := 20             -- bus waits for 20 minutes

noncomputable def probability_bus_there_when_Laura_arrives : ℚ :=
  let total_area := 90 * 90
  let trapezoid_area := 1400
  let triangle_area := 200
  (trapezoid_area + triangle_area) / total_area

-- Theorem statement
theorem bus_waiting_probability : probability_bus_there_when_Laura_arrives = 16 / 81 := by
  sorry

end NUMINAMATH_GPT_bus_waiting_probability_l1416_141680


namespace NUMINAMATH_GPT_prove_correct_operation_l1416_141624

def correct_operation (a b : ℕ) : Prop :=
  (a^3 * a^2 ≠ a^6) ∧
  ((a * b^2)^2 = a^2 * b^4) ∧
  (a^10 / a^5 ≠ a^2) ∧
  (a^2 + a ≠ a^3)

theorem prove_correct_operation (a b : ℕ) : correct_operation a b :=
by {
  sorry
}

end NUMINAMATH_GPT_prove_correct_operation_l1416_141624


namespace NUMINAMATH_GPT_attendance_rate_comparison_l1416_141668

theorem attendance_rate_comparison (attendees_A total_A attendees_B total_B : ℕ) 
  (hA : (attendees_A / total_A: ℚ) > (attendees_B / total_B: ℚ)) : 
  (attendees_A > attendees_B) → false :=
by
  sorry

end NUMINAMATH_GPT_attendance_rate_comparison_l1416_141668


namespace NUMINAMATH_GPT_sector_area_is_4_l1416_141628

/-- Given a sector of a circle with perimeter 8 and central angle 2 radians,
    the area of the sector is 4. -/
theorem sector_area_is_4 (r l : ℝ) (h1 : l + 2 * r = 8) (h2 : l / r = 2) : 
    (1 / 2) * l * r = 4 :=
sorry

end NUMINAMATH_GPT_sector_area_is_4_l1416_141628


namespace NUMINAMATH_GPT_geo_seq_condition_l1416_141674

-- Definitions based on conditions
variable (a b c : ℝ)

-- Condition of forming a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, -1 * r = a ∧ a * r = b ∧ b * r = c ∧ c * r = -9

-- Proof problem statement
theorem geo_seq_condition (h : geometric_sequence a b c) : b = -3 ∧ a * c = 9 :=
sorry

end NUMINAMATH_GPT_geo_seq_condition_l1416_141674


namespace NUMINAMATH_GPT_Milly_spends_135_minutes_studying_l1416_141679

-- Definitions of homework times
def mathHomeworkTime := 60
def geographyHomeworkTime := mathHomeworkTime / 2
def scienceHomeworkTime := (mathHomeworkTime + geographyHomeworkTime) / 2

-- Definition of Milly's total study time
def totalStudyTime := mathHomeworkTime + geographyHomeworkTime + scienceHomeworkTime

-- Theorem stating that Milly spends 135 minutes studying
theorem Milly_spends_135_minutes_studying : totalStudyTime = 135 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_Milly_spends_135_minutes_studying_l1416_141679


namespace NUMINAMATH_GPT_clownfish_ratio_l1416_141667

theorem clownfish_ratio (C B : ℕ) (h₁ : C = B) (h₂ : C + B = 100) (h₃ : C = B) : 
  (let B := 50; 
  let initially_clownfish := B - 26; -- Number of clownfish that initially joined display tank
  let swam_back := (B - 26) - 16; -- Number of clownfish that swam back
  initially_clownfish > 0 → 
  swam_back > 0 → 
  (swam_back : ℚ) / (initially_clownfish : ℚ) = 1 / 3) :=
by 
  sorry

end NUMINAMATH_GPT_clownfish_ratio_l1416_141667


namespace NUMINAMATH_GPT_each_piece_of_paper_weight_l1416_141698

noncomputable def paper_weight : ℚ :=
 sorry

theorem each_piece_of_paper_weight (w : ℚ) (n : ℚ) (envelope_weight : ℚ) (stamps_needed : ℚ) (paper_pieces : ℚ) :
  paper_pieces = 8 →
  envelope_weight = 2/5 →
  stamps_needed = 2 →
  n = paper_pieces * w + envelope_weight →
  n ≤ stamps_needed →
  w = 1/5 :=
by sorry

end NUMINAMATH_GPT_each_piece_of_paper_weight_l1416_141698


namespace NUMINAMATH_GPT_number_of_possible_flags_l1416_141684

-- Define the number of colors available
def num_colors : ℕ := 3

-- Define the number of stripes on the flag
def num_stripes : ℕ := 3

-- Define the total number of possible flags
def total_flags : ℕ := num_colors ^ num_stripes

-- The statement we need to prove
theorem number_of_possible_flags : total_flags = 27 := by
  sorry

end NUMINAMATH_GPT_number_of_possible_flags_l1416_141684


namespace NUMINAMATH_GPT_pieces_after_cuts_l1416_141620

theorem pieces_after_cuts (n : ℕ) (h : n = 10) : (n + 1) = 11 := by
  sorry

end NUMINAMATH_GPT_pieces_after_cuts_l1416_141620


namespace NUMINAMATH_GPT_original_digit_sum_six_and_product_is_1008_l1416_141671

theorem original_digit_sum_six_and_product_is_1008 (x : ℕ) :
  (2 ∣ x / 10) → (4 ∣ x / 10) → 
  (x % 10 + (x / 10) = 6) →
  ((x % 10) * 10 + (x / 10)) * ((x / 10) * 10 + (x % 10)) = 1008 →
  x = 42 ∨ x = 24 :=
by
  intro h1 h2 h3 h4
  sorry


end NUMINAMATH_GPT_original_digit_sum_six_and_product_is_1008_l1416_141671


namespace NUMINAMATH_GPT_second_piece_cost_l1416_141655

theorem second_piece_cost
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (single_piece1 : ℕ)
  (single_piece2 : ℕ)
  (remaining_piece_count : ℕ)
  (remaining_piece_cost : ℕ)
  (total_cost : total_spent = 610)
  (number_of_items : num_pieces = 7)
  (first_item_cost : single_piece1 = 49)
  (remaining_piece_item_cost : remaining_piece_cost = 96)
  (first_item_total_cost : remaining_piece_count = 5)
  (sum_equation : single_piece1 + single_piece2 + (remaining_piece_count * remaining_piece_cost) = total_spent) :
  single_piece2 = 81 := 
  sorry

end NUMINAMATH_GPT_second_piece_cost_l1416_141655


namespace NUMINAMATH_GPT_find_interest_rate_l1416_141627

theorem find_interest_rate
  (P : ℝ) (A : ℝ) (n t : ℕ) (hP : P = 3000) (hA : A = 3307.5) (hn : n = 2) (ht : t = 1) :
  ∃ r : ℝ, r = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l1416_141627


namespace NUMINAMATH_GPT_ellipse_constants_sum_l1416_141640

/-- Given the center of the ellipse at (h, k) = (3, -5),
    the semi-major axis a = 7,
    and the semi-minor axis b = 4,
    prove that h + k + a + b = 9. -/
theorem ellipse_constants_sum :
  let h := 3
  let k := -5
  let a := 7
  let b := 4
  h + k + a + b = 9 :=
by
  let h := 3
  let k := -5
  let a := 7
  let b := 4
  sorry

end NUMINAMATH_GPT_ellipse_constants_sum_l1416_141640


namespace NUMINAMATH_GPT_rectangle_area_l1416_141643

theorem rectangle_area (P : ℕ) (w : ℕ) (h : ℕ) (A : ℕ) 
  (hP : P = 28) 
  (hw : w = 6)
  (hW : P = 2 * (h + w)) 
  (hA : A = h * w) : 
  A = 48 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1416_141643


namespace NUMINAMATH_GPT_value_of_x_l1416_141633

theorem value_of_x (x y : ℕ) (h1 : x / y = 7 / 3) (h2 : y = 21) : x = 49 := sorry

end NUMINAMATH_GPT_value_of_x_l1416_141633


namespace NUMINAMATH_GPT_age_difference_ratio_l1416_141657

theorem age_difference_ratio (h : ℕ) (f : ℕ) (m : ℕ) 
  (harry_age : h = 50) 
  (father_age : f = h + 24) 
  (mother_age : m = 22 + h) :
  (f - m) / h = 1 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_age_difference_ratio_l1416_141657


namespace NUMINAMATH_GPT_polynomial_remainder_l1416_141642

noncomputable def divisionRemainder (f g : Polynomial ℝ) : Polynomial ℝ := Polynomial.modByMonic f g

theorem polynomial_remainder :
  divisionRemainder (Polynomial.X ^ 5 + 2) (Polynomial.X ^ 2 - 4 * Polynomial.X + 7) = -29 * Polynomial.X - 54 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1416_141642


namespace NUMINAMATH_GPT_cost_of_each_skirt_l1416_141682

-- Problem definitions based on conditions
def cost_of_art_supplies : ℕ := 20
def total_expenditure : ℕ := 50
def number_of_skirts : ℕ := 2

-- Proving the cost of each skirt
theorem cost_of_each_skirt (cost_of_each_skirt : ℕ) : 
  number_of_skirts * cost_of_each_skirt + cost_of_art_supplies = total_expenditure → 
  cost_of_each_skirt = 15 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_each_skirt_l1416_141682


namespace NUMINAMATH_GPT_min_acute_triangles_for_isosceles_l1416_141694

noncomputable def isosceles_triangle_acute_division : ℕ :=
  sorry

theorem min_acute_triangles_for_isosceles {α : ℝ} (hα : α = 108) (isosceles : ∀ β γ : ℝ, β = γ) :
  isosceles_triangle_acute_division = 7 :=
sorry

end NUMINAMATH_GPT_min_acute_triangles_for_isosceles_l1416_141694


namespace NUMINAMATH_GPT_calculate_expression_l1416_141676

theorem calculate_expression : 61 + 5 * 12 / (180 / 3) = 62 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1416_141676


namespace NUMINAMATH_GPT_line_tangent_through_A_l1416_141656

theorem line_tangent_through_A {A : ℝ × ℝ} (hA : A = (1, 2)) : 
  ∃ m b : ℝ, (b = 2) ∧ (∀ x : ℝ, y = m * x + b) ∧ (∀ y x : ℝ, y^2 = 4*x → y = 2) :=
by
  sorry

end NUMINAMATH_GPT_line_tangent_through_A_l1416_141656


namespace NUMINAMATH_GPT_max_sum_of_distances_l1416_141687

theorem max_sum_of_distances (x1 x2 y1 y2 : ℝ)
  (h1 : x1^2 + y1^2 = 1)
  (h2 : x2^2 + y2^2 = 1)
  (h3 : x1 * x2 + y1 * y2 = 1 / 2) :
  (|x1 + y1 - 1| / Real.sqrt 2 + |x2 + y2 - 1| / Real.sqrt 2) ≤ Real.sqrt 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_sum_of_distances_l1416_141687


namespace NUMINAMATH_GPT_how_many_oranges_put_back_l1416_141651

variables (A O x : ℕ)

-- Conditions: prices and initial selection.
def price_apple (A : ℕ) : ℕ := 40 * A
def price_orange (O : ℕ) : ℕ := 60 * O
def total_fruit := 20
def average_price_initial : ℕ := 56 -- Average price in cents

-- Conditions: equation from initial average price.
def total_initial_cost := total_fruit * average_price_initial
axiom initial_cost_eq : price_apple A + price_orange O = total_initial_cost
axiom total_fruit_eq : A + O = total_fruit

-- New conditions: desired average price and number of fruits
def average_price_new : ℕ := 52 -- Average price in cents
axiom new_cost_eq : price_apple A + price_orange (O - x) = (total_fruit - x) * average_price_new

-- The statement to be proven
theorem how_many_oranges_put_back : 40 * A + 60 * (O - 10) = (total_fruit - 10) * 52 → x = 10 :=
sorry

end NUMINAMATH_GPT_how_many_oranges_put_back_l1416_141651


namespace NUMINAMATH_GPT_intersection_of_sets_l1416_141616

def SetA : Set ℝ := {x | 0 < x ∧ x < 3}
def SetB : Set ℝ := {x | x > 2}
def SetC : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_of_sets :
  SetA ∩ SetB = SetC :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1416_141616


namespace NUMINAMATH_GPT_not_product_of_two_integers_l1416_141607

theorem not_product_of_two_integers (n : ℕ) (hn : n > 0) :
  ∀ t k : ℕ, t * (t + k) = n^2 + n + 1 → k ≥ 2 * Nat.sqrt n :=
by
  sorry

end NUMINAMATH_GPT_not_product_of_two_integers_l1416_141607


namespace NUMINAMATH_GPT_price_without_and_with_coupon_l1416_141637

theorem price_without_and_with_coupon
  (commission_rate sale_tax_rate discount_rate : ℝ)
  (cost producer_price shipping_fee: ℝ)
  (S: ℝ)
  (h_commission: commission_rate = 0.20)
  (h_sale_tax: sale_tax_rate = 0.08)
  (h_discount: discount_rate = 0.10)
  (h_producer_price: producer_price = 20)
  (h_shipping_fee: shipping_fee = 5)
  (h_total_cost: cost = producer_price + shipping_fee)
  (h_profit: 0.20 * cost = 5)
  (h_total_earn: cost + sale_tax_rate * S + 5 = 0.80 * S)
  (h_S: S = 41.67):
  S = 41.67 ∧ 0.90 * S = 37.50 :=
by
  sorry

end NUMINAMATH_GPT_price_without_and_with_coupon_l1416_141637


namespace NUMINAMATH_GPT_carrie_strawberry_harvest_l1416_141670

/-- Carrie has a rectangular garden that measures 10 feet by 7 feet.
    She plants the entire garden with strawberry plants. Carrie is able to
    plant 5 strawberry plants per square foot, and she harvests an average of
    12 strawberries per plant. How many strawberries can she expect to harvest?
-/
theorem carrie_strawberry_harvest :
  let width := 10
  let length := 7
  let plants_per_sqft := 5
  let strawberries_per_plant := 12
  let area := width * length
  let total_plants := plants_per_sqft * area
  let total_strawberries := strawberries_per_plant * total_plants
  total_strawberries = 4200 :=
by
  sorry

end NUMINAMATH_GPT_carrie_strawberry_harvest_l1416_141670


namespace NUMINAMATH_GPT_geometric_seq_fourth_term_l1416_141617

-- Define the conditions
def first_term (a1 : ℝ) : Prop := a1 = 512
def sixth_term (a1 r : ℝ) : Prop := a1 * r^5 = 32

-- Define the claim
def fourth_term (a1 r a4 : ℝ) : Prop := a4 = a1 * r^3

-- State the theorem
theorem geometric_seq_fourth_term :
  ∀ a1 r a4 : ℝ, first_term a1 → sixth_term a1 r → fourth_term a1 r a4 → a4 = 64 :=
by
  intros a1 r a4 h1 h2 h3
  rw [first_term, sixth_term, fourth_term] at *
  sorry

end NUMINAMATH_GPT_geometric_seq_fourth_term_l1416_141617


namespace NUMINAMATH_GPT_tile_covering_possible_l1416_141673

theorem tile_covering_possible (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  ((m % 6 = 0) ∨ (n % 6 = 0)) := 
sorry

end NUMINAMATH_GPT_tile_covering_possible_l1416_141673


namespace NUMINAMATH_GPT_stacy_has_2_more_than_triple_steve_l1416_141609

-- Definitions based on the given conditions
def skylar_berries : ℕ := 20
def steve_berries : ℕ := skylar_berries / 2
def stacy_berries : ℕ := 32

-- Statement to be proved
theorem stacy_has_2_more_than_triple_steve :
  stacy_berries = 3 * steve_berries + 2 := by
  sorry

end NUMINAMATH_GPT_stacy_has_2_more_than_triple_steve_l1416_141609


namespace NUMINAMATH_GPT_regular_polygon_sides_l1416_141683

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1416_141683


namespace NUMINAMATH_GPT_sweeties_remainder_l1416_141693

theorem sweeties_remainder (m k : ℤ) (h : m = 12 * k + 11) :
  (4 * m) % 12 = 8 :=
by
  -- The proof steps will go here
  sorry

end NUMINAMATH_GPT_sweeties_remainder_l1416_141693


namespace NUMINAMATH_GPT_pastries_calculation_l1416_141634

theorem pastries_calculation 
    (G : ℕ) (C : ℕ) (P : ℕ) (F : ℕ)
    (hG : G = 30) 
    (hC : C = G - 5)
    (hP : P = G - 5)
    (htotal : C + P + F + G = 97) :
    C - F = 8 ∧ P - F = 8 :=
by
  sorry

end NUMINAMATH_GPT_pastries_calculation_l1416_141634


namespace NUMINAMATH_GPT_time_spent_per_piece_l1416_141658

-- Conditions
def number_of_chairs : ℕ := 7
def number_of_tables : ℕ := 3
def total_furniture : ℕ := number_of_chairs + number_of_tables
def total_time_spent : ℕ := 40

-- Proof statement
theorem time_spent_per_piece : total_time_spent / total_furniture = 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_time_spent_per_piece_l1416_141658


namespace NUMINAMATH_GPT_calculate_probability_two_cards_sum_to_15_l1416_141649

-- Define the probability calculation as per the problem statement
noncomputable def probability_two_cards_sum_to_15 : ℚ :=
  let total_cards := 52
  let number_cards := 36 -- 9 values (2 through 10) each with 4 cards
  let card_combinations := (number_cards * (number_cards - 1)) / 2 -- Total pairs to choose from
  let favourable_combinations := 144 -- Manually calculated from cases in the solution
  favourable_combinations / card_combinations

theorem calculate_probability_two_cards_sum_to_15 :
  probability_two_cards_sum_to_15 = 8 / 221 :=
by
  -- Here we ignore the proof steps and directly state it assuming the provided assumption
  admit

end NUMINAMATH_GPT_calculate_probability_two_cards_sum_to_15_l1416_141649


namespace NUMINAMATH_GPT_correct_definition_of_regression_independence_l1416_141619

-- Definitions
def regression_analysis (X Y : Type) := ∃ r : X → Y, true -- Placeholder, ideal definition studies correlation
def independence_test (X Y : Type) := ∃ rel : X → Y → Prop, true -- Placeholder, ideal definition examines relationship

-- Theorem statement
theorem correct_definition_of_regression_independence (X Y : Type) :
  (∃ r : X → Y, true) ∧ (∃ rel : X → Y → Prop, true)
  → "Regression analysis studies the correlation between two variables, and independence tests examine whether there is some kind of relationship between two variables" = "C" :=
sorry

end NUMINAMATH_GPT_correct_definition_of_regression_independence_l1416_141619


namespace NUMINAMATH_GPT_find_a_l1416_141608

noncomputable def givenConditions (a b c R : ℝ) : Prop :=
  (a^2 / (b * c) - c / b - b / c = Real.sqrt 3) ∧ (R = 3)

theorem find_a (a b c : ℝ) (R : ℝ) (h : givenConditions a b c R) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1416_141608


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1416_141602

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → x^2 + x - 2 > 0) ∧ (∃ y, y < -2 ∧ y^2 + y - 2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1416_141602


namespace NUMINAMATH_GPT_average_cookies_per_package_l1416_141647

def cookies_per_package : List ℕ := [9, 11, 14, 12, 0, 18, 15, 16, 19, 21]

theorem average_cookies_per_package :
  (cookies_per_package.sum : ℚ) / cookies_per_package.length = 13.5 := by
  sorry

end NUMINAMATH_GPT_average_cookies_per_package_l1416_141647


namespace NUMINAMATH_GPT_polynomial_root_triples_l1416_141636

theorem polynomial_root_triples (a b c : ℝ) :
  (∀ x : ℝ, x > 0 → (x^4 + a * x^3 + b * x^2 + c * x + b = 0)) ↔ (a, b, c) = (-21, 112, -204) ∨ (a, b, c) = (-12, 48, -80) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_root_triples_l1416_141636


namespace NUMINAMATH_GPT_floor_equality_iff_l1416_141625

variable (x : ℝ)

theorem floor_equality_iff :
  (⌊3 * x + 4⌋ = ⌊5 * x - 1⌋) ↔
  (11 / 5 ≤ x ∧ x < 7 / 3) ∨
  (12 / 5 ≤ x ∧ x < 13 / 5) ∨
  (17 / 5 ≤ x ∧ x < 18 / 5) := by
  sorry

end NUMINAMATH_GPT_floor_equality_iff_l1416_141625


namespace NUMINAMATH_GPT_symmetric_point_l1416_141648

theorem symmetric_point (a b : ℝ) (h1 : a = 2) (h2 : 3 = -b) : (a + b) ^ 2023 = -1 := 
by
  sorry

end NUMINAMATH_GPT_symmetric_point_l1416_141648


namespace NUMINAMATH_GPT_zero_integers_in_range_such_that_expr_is_perfect_square_l1416_141689

theorem zero_integers_in_range_such_that_expr_is_perfect_square :
  (∃ n : ℕ, 5 ≤ n ∧ n ≤ 15 ∧ ∃ m : ℕ, 2 * n ^ 2 + n + 2 = m ^ 2) → False :=
by sorry

end NUMINAMATH_GPT_zero_integers_in_range_such_that_expr_is_perfect_square_l1416_141689


namespace NUMINAMATH_GPT_service_station_location_l1416_141618

/-- The first exit is at milepost 35. -/
def first_exit_milepost : ℕ := 35

/-- The eighth exit is at milepost 275. -/
def eighth_exit_milepost : ℕ := 275

/-- The expected milepost of the service station built halfway between the first exit and the eighth exit is 155. -/
theorem service_station_location : (first_exit_milepost + (eighth_exit_milepost - first_exit_milepost) / 2) = 155 := by
  sorry

end NUMINAMATH_GPT_service_station_location_l1416_141618


namespace NUMINAMATH_GPT_mixed_alcohol_solution_l1416_141606

theorem mixed_alcohol_solution 
    (vol_x : ℝ) (vol_y : ℝ) (conc_x : ℝ) (conc_y : ℝ) (target_conc : ℝ) (vol_y_given : vol_y = 750) 
    (conc_x_given : conc_x = 0.10) (conc_y_given : conc_y = 0.30) (target_conc_given : target_conc = 0.25) : 
    vol_x = 250 → 
    (conc_x * vol_x + conc_y * vol_y) / (vol_x + vol_y) = target_conc :=
by
  intros h_x
  rw [vol_y_given, conc_x_given, conc_y_given, target_conc_given, h_x]
  sorry

end NUMINAMATH_GPT_mixed_alcohol_solution_l1416_141606


namespace NUMINAMATH_GPT_probability_at_least_one_six_l1416_141660

theorem probability_at_least_one_six (h: ℚ) : h = 91 / 216 :=
by 
  sorry

end NUMINAMATH_GPT_probability_at_least_one_six_l1416_141660


namespace NUMINAMATH_GPT_simplify_expression_l1416_141621

theorem simplify_expression :
  (Real.sqrt 5 * 5^(1/2) + 20 / 4 * 3 - 9^(3/2)) = -7 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1416_141621


namespace NUMINAMATH_GPT_balance_balls_l1416_141661

variable (G B Y W P : ℝ)

-- Given conditions
def cond1 : 4 * G = 9 * B := sorry
def cond2 : 3 * Y = 8 * B := sorry
def cond3 : 7 * B = 5 * W := sorry
def cond4 : 4 * P = 10 * B := sorry

-- Theorem we need to prove
theorem balance_balls : 5 * G + 3 * Y + 3 * W + P = 26 * B :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_balance_balls_l1416_141661


namespace NUMINAMATH_GPT_which_is_right_triangle_l1416_141650

-- Definitions for each group of numbers
def sides_A := (1, 2, 3)
def sides_B := (3, 4, 5)
def sides_C := (4, 5, 6)
def sides_D := (7, 8, 9)

-- Definition of a condition for right triangle using the converse of the Pythagorean theorem
def is_right_triangle (a b c: ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem which_is_right_triangle :
    ¬is_right_triangle 1 2 3 ∧
    ¬is_right_triangle 4 5 6 ∧
    ¬is_right_triangle 7 8 9 ∧
    is_right_triangle 3 4 5 :=
by
  sorry

end NUMINAMATH_GPT_which_is_right_triangle_l1416_141650


namespace NUMINAMATH_GPT_find_x2_plus_y2_l1416_141630

theorem find_x2_plus_y2 
  (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x * y + x + y = 83)
  (h2 : x^2 * y + x * y^2 = 1056) :
  x^2 + y^2 = 458 :=
by
  sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l1416_141630


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1416_141632

theorem boat_speed_in_still_water (V_b : ℝ) : 
  (∀ t : ℝ, t = 26 / (V_b + 6) → t = 14 / (V_b - 6)) → V_b = 20 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1416_141632


namespace NUMINAMATH_GPT_cube_of_product_of_ab_l1416_141695

theorem cube_of_product_of_ab (a b c : ℕ) (h1 : a * b * c = 180) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : (a * b) ^ 3 = 216 := 
sorry

end NUMINAMATH_GPT_cube_of_product_of_ab_l1416_141695


namespace NUMINAMATH_GPT_pattern_equation_l1416_141635

theorem pattern_equation (n : ℕ) (h : n ≥ 1) : 
  (Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2))) :=
by
  sorry

end NUMINAMATH_GPT_pattern_equation_l1416_141635


namespace NUMINAMATH_GPT_number_of_hens_l1416_141663

-- Conditions as Lean definitions
def total_heads (H C : ℕ) : Prop := H + C = 48
def total_feet (H C : ℕ) : Prop := 2 * H + 4 * C = 136

-- Mathematically equivalent proof problem
theorem number_of_hens (H C : ℕ) (h1 : total_heads H C) (h2 : total_feet H C) : H = 28 :=
by
  sorry

end NUMINAMATH_GPT_number_of_hens_l1416_141663


namespace NUMINAMATH_GPT_farm_distance_l1416_141691

theorem farm_distance (a x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (triangle_ineq1 : x + z = 85)
  (triangle_ineq2 : x + y = 4 * z)
  (triangle_ineq3 : z + y = x + a) :
  0 < a ∧ a < 85 ∧
  x = (340 - a) / 6 ∧
  y = (2 * a + 85) / 3 ∧
  z = (170 + a) / 6 :=
sorry

end NUMINAMATH_GPT_farm_distance_l1416_141691


namespace NUMINAMATH_GPT_trigonometric_identity_l1416_141605

theorem trigonometric_identity (α : ℝ) (h : Real.sin (π + α) = -1/3) : Real.sin (2 * α) / Real.cos α = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1416_141605


namespace NUMINAMATH_GPT_find_sum_abc_l1416_141666

noncomputable def f (x a b c : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x + c

theorem find_sum_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (habc_distinct : a ≠ b) (hfa : f a a b c = a^3) (hfb : f b a b c = b^3) : 
  a + b + c = 18 := 
sorry

end NUMINAMATH_GPT_find_sum_abc_l1416_141666


namespace NUMINAMATH_GPT_boxes_sold_l1416_141699

def case_size : ℕ := 12
def remaining_boxes : ℕ := 7

theorem boxes_sold (sold_boxes : ℕ) : ∃ n : ℕ, sold_boxes = n * case_size + remaining_boxes :=
sorry

end NUMINAMATH_GPT_boxes_sold_l1416_141699


namespace NUMINAMATH_GPT_train_length_l1416_141613

/-- Given a train that can cross an electric pole in 15 seconds and has a speed of 72 km/h, prove that the length of the train is 300 meters. -/
theorem train_length 
  (time_to_cross_pole : ℝ)
  (train_speed_kmh : ℝ)
  (h1 : time_to_cross_pole = 15)
  (h2 : train_speed_kmh = 72)
  : (train_speed_kmh * 1000 / 3600) * time_to_cross_pole = 300 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_train_length_l1416_141613


namespace NUMINAMATH_GPT_hyperbola_eccentricity_correct_l1416_141612

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_asymptote : b / a = Real.tan (Real.pi / 6)) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_correct
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_asymptote : b / a = Real.tan (Real.pi / 6)) :
  hyperbola_eccentricity a b h_a h_b h_asymptote = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_correct_l1416_141612


namespace NUMINAMATH_GPT_larger_number_is_1891_l1416_141623

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem larger_number_is_1891 :
  ∃ L S : ℕ, (L - S = 1355) ∧ (L = 6 * S + 15) ∧ is_prime (sum_of_digits L) ∧ sum_of_digits L ≠ 12
  :=
sorry

end NUMINAMATH_GPT_larger_number_is_1891_l1416_141623


namespace NUMINAMATH_GPT_number_of_teams_l1416_141696

-- Define the statement representing the problem and conditions
theorem number_of_teams (n : ℕ) (h : 2 * n * (n - 1) = 9800) : n = 50 :=
sorry

end NUMINAMATH_GPT_number_of_teams_l1416_141696


namespace NUMINAMATH_GPT_contribution_amount_l1416_141600

theorem contribution_amount (x : ℝ) (S : ℝ) :
  (S = 10 * x) ∧ (S = 15 * (x - 100)) → x = 300 :=
by
  sorry

end NUMINAMATH_GPT_contribution_amount_l1416_141600


namespace NUMINAMATH_GPT_more_oranges_than_apples_l1416_141622

def apples : ℕ := 14
def oranges : ℕ := 2 * 12

theorem more_oranges_than_apples : oranges - apples = 10 :=
by
  sorry

end NUMINAMATH_GPT_more_oranges_than_apples_l1416_141622


namespace NUMINAMATH_GPT_base6_base5_subtraction_in_base10_l1416_141629

def base6_to_nat (n : ℕ) : ℕ :=
  3 * 6^2 + 2 * 6^1 + 5 * 6^0

def base5_to_nat (n : ℕ) : ℕ :=
  2 * 5^2 + 3 * 5^1 + 1 * 5^0

theorem base6_base5_subtraction_in_base10 : base6_to_nat 325 - base5_to_nat 231 = 59 := by
  sorry

end NUMINAMATH_GPT_base6_base5_subtraction_in_base10_l1416_141629


namespace NUMINAMATH_GPT_sphere_center_ratio_l1416_141644

/-
Let O be the origin and let (a, b, c) be a fixed point.
A plane with the equation x + 2y + 3z = 6 passes through (a, b, c)
and intersects the x-axis, y-axis, and z-axis at A, B, and C, respectively, all distinct from O.
Let (p, q, r) be the center of the sphere passing through A, B, C, and O.
Prove: a / p + b / q + c / r = 2
-/
theorem sphere_center_ratio (a b c : ℝ) (p q r : ℝ)
  (h_plane : a + 2 * b + 3 * c = 6) 
  (h_p : p = 3)
  (h_q : q = 1.5)
  (h_r : r = 1) :
  a / p + b / q + c / r = 2 :=
by
  sorry

end NUMINAMATH_GPT_sphere_center_ratio_l1416_141644


namespace NUMINAMATH_GPT_total_amount_paid_l1416_141652

-- Definitions based on the conditions.
def cost_per_pizza : ℝ := 12
def delivery_charge : ℝ := 2
def distance_threshold : ℝ := 1000 -- distance in meters
def park_distance : ℝ := 100
def building_distance : ℝ := 2000

def pizzas_at_park : ℕ := 3
def pizzas_at_building : ℕ := 2

-- The proof problem stating the total amount paid to Jimmy.
theorem total_amount_paid :
  let total_pizzas := pizzas_at_park + pizzas_at_building
  let cost_without_delivery := total_pizzas * cost_per_pizza
  let park_charge := if park_distance > distance_threshold then pizzas_at_park * delivery_charge else 0
  let building_charge := if building_distance > distance_threshold then pizzas_at_building * delivery_charge else 0
  let total_cost := cost_without_delivery + park_charge + building_charge
  total_cost = 64 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1416_141652


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1416_141601

variable (x : ℝ)

theorem sufficient_not_necessary_condition :
  (x > 2 → x > 1) ∧ (¬ (x > 1 → x > 2)) := by
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1416_141601


namespace NUMINAMATH_GPT_age_difference_l1416_141678

theorem age_difference (d : ℕ) (h1 : 18 + (18 - d) + (18 - 2 * d) + (18 - 3 * d) = 48) : d = 4 :=
sorry

end NUMINAMATH_GPT_age_difference_l1416_141678


namespace NUMINAMATH_GPT_work_completion_time_l1416_141645

theorem work_completion_time (work_per_day_A : ℚ) (work_per_day_B : ℚ) (work_per_day_C : ℚ) 
(days_A_worked: ℚ) (days_C_worked: ℚ) :
work_per_day_A = 1 / 20 ∧ work_per_day_B = 1 / 30 ∧ work_per_day_C = 1 / 10 ∧
days_A_worked = 2 ∧ days_C_worked = 4  → 
(work_per_day_A * days_A_worked + work_per_day_B * days_A_worked + work_per_day_C * days_A_worked +
work_per_day_B * (days_C_worked - days_A_worked) + work_per_day_C * (days_C_worked - days_A_worked) +
(1 - 
(work_per_day_A * days_A_worked + work_per_day_B * days_A_worked + work_per_day_C * days_A_worked +
work_per_day_B * (days_C_worked - days_A_worked) + work_per_day_C * (days_C_worked - days_A_worked)))
/ work_per_day_B + days_C_worked) 
= 15 := by
sorry

end NUMINAMATH_GPT_work_completion_time_l1416_141645


namespace NUMINAMATH_GPT_varphi_solution_l1416_141614

noncomputable def varphi (x : ℝ) (m n : ℝ) : ℝ :=
  m * x + n / x

theorem varphi_solution :
  ∃ (m n : ℝ), (varphi 1 m n = 8) ∧ (varphi 16 m n = 16) ∧ (∀ x, varphi x m n = 3 * x + 5 / x) :=
sorry

end NUMINAMATH_GPT_varphi_solution_l1416_141614


namespace NUMINAMATH_GPT_geom_sixth_term_is_31104_l1416_141692

theorem geom_sixth_term_is_31104 :
  ∃ (r : ℝ), 4 * r^8 = 39366 ∧ 4 * r^(6-1) = 31104 :=
by
  sorry

end NUMINAMATH_GPT_geom_sixth_term_is_31104_l1416_141692


namespace NUMINAMATH_GPT_min_value_of_y_l1416_141603

variable {x k : ℝ}

theorem min_value_of_y (h₁ : ∀ x > 0, 0 < k) 
  (h₂ : ∀ x > 0, (x^2 + k / x) ≥ 3) : k = 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_y_l1416_141603


namespace NUMINAMATH_GPT_cube_sum_identity_l1416_141659

theorem cube_sum_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^3 + 1/r^3 = 2 * Real.sqrt 5 ∨ r^3 + 1/r^3 = -2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_cube_sum_identity_l1416_141659


namespace NUMINAMATH_GPT_probability_of_both_l1416_141665

variable (A B : Prop)

-- Assumptions
def p_A : ℝ := 0.55
def p_B : ℝ := 0.60

-- Probability of both A and B telling the truth at the same time
theorem probability_of_both : p_A * p_B = 0.33 := by
  sorry

end NUMINAMATH_GPT_probability_of_both_l1416_141665


namespace NUMINAMATH_GPT_factorize_xy_l1416_141681

theorem factorize_xy (x y : ℕ): xy - x + y - 1 = (x + 1) * (y - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_xy_l1416_141681


namespace NUMINAMATH_GPT_eunice_pots_l1416_141639

theorem eunice_pots (total_seeds pots_with_3_seeds last_pot_seeds : ℕ)
  (h1 : total_seeds = 10)
  (h2 : pots_with_3_seeds * 3 + last_pot_seeds = total_seeds)
  (h3 : last_pot_seeds = 1) : pots_with_3_seeds + 1 = 4 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_eunice_pots_l1416_141639


namespace NUMINAMATH_GPT_inequality_proof_l1416_141690

variables (a b c d : ℝ)

theorem inequality_proof 
  (h1 : a + b > abs (c - d)) 
  (h2 : c + d > abs (a - b)) : 
  a + c > abs (b - d) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1416_141690


namespace NUMINAMATH_GPT_juice_drinks_costs_2_l1416_141685

-- Define the conditions and the proof problem
theorem juice_drinks_costs_2 (given_amount : ℕ) (amount_returned : ℕ) 
                            (pizza_cost : ℕ) (number_of_pizzas : ℕ) 
                            (number_of_juice_packs : ℕ) 
                            (total_spent_on_juice : ℕ) (cost_per_pack : ℕ) 
                            (h1 : given_amount = 50) (h2 : amount_returned = 22)
                            (h3 : pizza_cost = 12) (h4 : number_of_pizzas = 2)
                            (h5 : number_of_juice_packs = 2) 
                            (h6 : given_amount - amount_returned - number_of_pizzas * pizza_cost = total_spent_on_juice) 
                            (h7 : total_spent_on_juice / number_of_juice_packs = cost_per_pack) : 
                            cost_per_pack = 2 := by
  sorry

end NUMINAMATH_GPT_juice_drinks_costs_2_l1416_141685


namespace NUMINAMATH_GPT_phi_value_l1416_141611

open Real

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := sin (2 * x + φ)

theorem phi_value (φ : ℝ) (h : |φ| < π / 2) :
  (∀ x : ℝ, f (x + π / 3) φ = f (-(x + π / 3)) φ) → φ = -(π / 6) :=
by
  intro h'
  sorry

end NUMINAMATH_GPT_phi_value_l1416_141611


namespace NUMINAMATH_GPT_probability_of_three_heads_in_eight_tosses_l1416_141672

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end NUMINAMATH_GPT_probability_of_three_heads_in_eight_tosses_l1416_141672


namespace NUMINAMATH_GPT_L_shaped_region_area_l1416_141641

-- Define the conditions
def square_area (side_length : ℕ) : ℕ := side_length * side_length

def WXYZ_side_length : ℕ := 6
def XUVW_side_length : ℕ := 2
def TYXZ_side_length : ℕ := 3

-- Define the areas of the squares
def WXYZ_area : ℕ := square_area WXYZ_side_length
def XUVW_area : ℕ := square_area XUVW_side_length
def TYXZ_area : ℕ := square_area TYXZ_side_length

-- Lean statement to prove the area of the L-shaped region
theorem L_shaped_region_area : WXYZ_area - XUVW_area - TYXZ_area = 23 := by
  sorry

end NUMINAMATH_GPT_L_shaped_region_area_l1416_141641


namespace NUMINAMATH_GPT_milk_fraction_correct_l1416_141615

def fraction_of_milk_in_coffee_cup (coffee_initial : ℕ) (milk_initial : ℕ) : ℚ :=
  let coffee_transferred := coffee_initial / 3
  let milk_cup_after_transfer := milk_initial + coffee_transferred
  let coffee_left := coffee_initial - coffee_transferred
  let total_mixed := milk_cup_after_transfer
  let transfer_back := total_mixed / 2
  let coffee_back := transfer_back * (coffee_transferred / total_mixed)
  let milk_back := transfer_back * (milk_initial / total_mixed)
  let coffee_final := coffee_left + coffee_back
  let milk_final := milk_back
  milk_final / (coffee_final + milk_final)

theorem milk_fraction_correct (coffee_initial : ℕ) (milk_initial : ℕ)
  (h_coffee : coffee_initial = 6) (h_milk : milk_initial = 3) :
  fraction_of_milk_in_coffee_cup coffee_initial milk_initial = 3 / 13 :=
by
  sorry

end NUMINAMATH_GPT_milk_fraction_correct_l1416_141615


namespace NUMINAMATH_GPT_treasure_distribution_l1416_141626

noncomputable def calculate_share (investment total_investment total_value : ℝ) : ℝ :=
  (investment / total_investment) * total_value

theorem treasure_distribution 
  (investment_fonzie investment_aunt_bee investment_lapis investment_skylar investment_orion total_treasure : ℝ)
  (total_investment : ℝ)
  (h : total_investment = investment_fonzie + investment_aunt_bee + investment_lapis + investment_skylar + investment_orion) :
  calculate_share investment_fonzie total_investment total_treasure = 210000 ∧
  calculate_share investment_aunt_bee total_investment total_treasure = 255000 ∧
  calculate_share investment_lapis total_investment total_treasure = 270000 ∧
  calculate_share investment_skylar total_investment total_treasure = 225000 ∧
  calculate_share investment_orion total_investment total_treasure = 240000 :=
by
  sorry

end NUMINAMATH_GPT_treasure_distribution_l1416_141626


namespace NUMINAMATH_GPT_problem_l1416_141688

-- Definition for condition 1
def condition1 (uniform_band : Prop) (appropriate_model : Prop) := 
  uniform_band → appropriate_model

-- Definition for condition 2
def condition2 (smaller_residual : Prop) (better_fit : Prop) :=
  smaller_residual → better_fit

-- Formal statement of the problem
theorem problem (uniform_band appropriate_model smaller_residual better_fit : Prop)
  (h1 : condition1 uniform_band appropriate_model)
  (h2 : condition2 smaller_residual better_fit)
  (h3 : uniform_band ∧ smaller_residual) :
  appropriate_model ∧ better_fit :=
  sorry

end NUMINAMATH_GPT_problem_l1416_141688


namespace NUMINAMATH_GPT_probability_red_in_both_jars_l1416_141686

def original_red_buttons : ℕ := 6
def original_blue_buttons : ℕ := 10
def total_original_buttons : ℕ := original_red_buttons + original_blue_buttons
def remaining_buttons : ℕ := (2 * total_original_buttons) / 3
def moved_buttons : ℕ := total_original_buttons - remaining_buttons
def moved_red_buttons : ℕ := 2
def moved_blue_buttons : ℕ := 3

theorem probability_red_in_both_jars :
  moved_red_buttons = moved_blue_buttons →
  remaining_buttons = 11 →
  (∃ m n : ℚ, m / remaining_buttons = 4 / 11 ∧ n / (moved_red_buttons + moved_blue_buttons) = 2 / 5 ∧ (m / remaining_buttons) * (n / (moved_red_buttons + moved_blue_buttons)) = 8 / 55) :=
by sorry

end NUMINAMATH_GPT_probability_red_in_both_jars_l1416_141686


namespace NUMINAMATH_GPT_vector_perpendicular_to_a_l1416_141604

theorem vector_perpendicular_to_a :
  let a := (4, 3)
  let b := (3, -4)
  a.1 * b.1 + a.2 * b.2 = 0 := by
  let a := (4, 3)
  let b := (3, -4)
  sorry

end NUMINAMATH_GPT_vector_perpendicular_to_a_l1416_141604
