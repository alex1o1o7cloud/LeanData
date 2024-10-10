import Mathlib

namespace geometric_sequence_property_l1790_179019

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 * a 4 = 16 →
  (a 2 * a 3 * a 4 = 64 ∨ a 2 * a 3 * a 4 = -64) :=
by sorry

end geometric_sequence_property_l1790_179019


namespace car_dealership_problem_l1790_179017

/- Define the prices of models A and B -/
def price_A : ℝ := 20
def price_B : ℝ := 15

/- Define the sales data for two weeks -/
def week1_sales : ℝ := 65
def week1_units_A : ℕ := 1
def week1_units_B : ℕ := 3

def week2_sales : ℝ := 155
def week2_units_A : ℕ := 4
def week2_units_B : ℕ := 5

/- Define the company's purchase constraints -/
def total_units : ℕ := 8
def min_cost : ℝ := 145
def max_cost : ℝ := 153

/- Define a function to calculate the cost of a purchase plan -/
def purchase_cost (units_A : ℕ) : ℝ :=
  price_A * units_A + price_B * (total_units - units_A)

/- Define a function to check if a purchase plan is valid -/
def is_valid_plan (units_A : ℕ) : Prop :=
  units_A ≤ total_units ∧ 
  min_cost ≤ purchase_cost units_A ∧ 
  purchase_cost units_A ≤ max_cost

/- Theorem statement -/
theorem car_dealership_problem :
  /- Prices satisfy the sales data -/
  (price_A * week1_units_A + price_B * week1_units_B = week1_sales) ∧
  (price_A * week2_units_A + price_B * week2_units_B = week2_sales) ∧
  /- Exactly two valid purchase plans exist -/
  (∃ (plan1 plan2 : ℕ), 
    plan1 ≠ plan2 ∧ 
    is_valid_plan plan1 ∧ 
    is_valid_plan plan2 ∧
    (∀ (plan : ℕ), is_valid_plan plan → plan = plan1 ∨ plan = plan2)) ∧
  /- The most cost-effective plan is 5 units of A and 3 units of B -/
  (∀ (plan : ℕ), is_valid_plan plan → purchase_cost 5 ≤ purchase_cost plan) :=
by sorry

end car_dealership_problem_l1790_179017


namespace complement_of_P_in_U_l1790_179081

def U : Finset Int := {-1, 0, 1, 2}

def P : Set Int := {x | -Real.sqrt 2 < x ∧ x < Real.sqrt 2}

theorem complement_of_P_in_U : 
  (U.toSet \ P) = {2} := by sorry

end complement_of_P_in_U_l1790_179081


namespace circular_table_dice_probability_l1790_179032

/-- The number of people seated around the table -/
def num_people : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 8

/-- The probability of adjacent people not rolling the same number -/
def prob_not_same : ℚ := 7 / 8

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ := (prob_not_same ^ (num_people - 1))

theorem circular_table_dice_probability :
  prob_no_adjacent_same = 2401 / 4096 := by sorry

end circular_table_dice_probability_l1790_179032


namespace valid_base5_number_l1790_179075

def is_base5_digit (d : Nat) : Prop := d ≤ 4

def is_base5_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 5 → is_base5_digit d

theorem valid_base5_number : is_base5_number 2134 := by sorry

end valid_base5_number_l1790_179075


namespace power_sum_equality_l1790_179050

theorem power_sum_equality : (-2 : ℤ) ^ (4 ^ 2) + 2 ^ (3 ^ 2) = 66048 := by
  sorry

end power_sum_equality_l1790_179050


namespace fraction_equation_solution_l1790_179057

theorem fraction_equation_solution (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ y ≠ x) :
  1/x - 1/y = 1/z → z = x*y/(y-x) := by
  sorry

end fraction_equation_solution_l1790_179057


namespace heels_savings_per_month_l1790_179097

theorem heels_savings_per_month 
  (months_saved : ℕ) 
  (sister_contribution : ℕ) 
  (total_spent : ℕ) : 
  months_saved = 3 → 
  sister_contribution = 50 → 
  total_spent = 260 → 
  (total_spent - sister_contribution) / months_saved = 70 :=
by sorry

end heels_savings_per_month_l1790_179097


namespace sum_a_d_equals_ten_l1790_179026

theorem sum_a_d_equals_ten (a b c d : ℝ) 
  (h1 : a + b = 16) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) : 
  a + d = 10 := by
  sorry

end sum_a_d_equals_ten_l1790_179026


namespace binary_addition_subtraction_l1790_179074

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

theorem binary_addition_subtraction :
  let a := binary_to_decimal [true, true, false, true, true]
  let b := binary_to_decimal [true, false, true, true]
  let c := binary_to_decimal [true, true, true, false, false]
  let d := binary_to_decimal [true, false, true, false, true]
  let e := binary_to_decimal [true, false, false, true]
  let result := binary_to_decimal [true, true, true, true, false]
  a + b - c + d - e = result :=
by sorry

end binary_addition_subtraction_l1790_179074


namespace simplification_exponent_sum_l1790_179036

-- Define the expression
def original_expression (a b c : ℝ) : ℝ := (40 * a^5 * b^8 * c^14) ^ (1/3)

-- Define the simplified expression
def simplified_expression (a b c : ℝ) : ℝ := 2 * a * b^2 * c^4 * ((5 * a * b^2 * c^2) ^ (1/3))

-- State the theorem
theorem simplification_exponent_sum :
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 →
  original_expression a b c = simplified_expression a b c ∧
  (1 + 2 + 4 = 7) := by sorry

end simplification_exponent_sum_l1790_179036


namespace quarrel_between_opposite_houses_l1790_179061

/-- Represents a house in the square yard -/
inductive House : Type
| A : House
| B : House
| C : House
| D : House

/-- Represents a quarrel between two houses -/
structure Quarrel :=
  (house1 : House)
  (house2 : House)

/-- Checks if two houses are neighbors -/
def are_neighbors (h1 h2 : House) : Prop :=
  (h1 = House.A ∧ (h2 = House.B ∨ h2 = House.D)) ∨
  (h1 = House.B ∧ (h2 = House.A ∨ h2 = House.C)) ∨
  (h1 = House.C ∧ (h2 = House.B ∨ h2 = House.D)) ∨
  (h1 = House.D ∧ (h2 = House.A ∨ h2 = House.C))

/-- Checks if two houses are opposite -/
def are_opposite (h1 h2 : House) : Prop :=
  (h1 = House.A ∧ h2 = House.C) ∨ (h1 = House.C ∧ h2 = House.A) ∨
  (h1 = House.B ∧ h2 = House.D) ∨ (h1 = House.D ∧ h2 = House.B)

theorem quarrel_between_opposite_houses 
  (total_friends : Nat)
  (quarrels : List Quarrel)
  (h_total_friends : total_friends = 77)
  (h_quarrels_count : quarrels.length = 365)
  (h_different_houses : ∀ q ∈ quarrels, q.house1 ≠ q.house2)
  (h_no_neighbor_friends : ∀ h1 h2, are_neighbors h1 h2 → 
    ∃ q ∈ quarrels, (q.house1 = h1 ∧ q.house2 = h2) ∨ (q.house1 = h2 ∧ q.house2 = h1))
  : ∃ q ∈ quarrels, are_opposite q.house1 q.house2 :=
by sorry

end quarrel_between_opposite_houses_l1790_179061


namespace complex_fraction_equality_l1790_179041

theorem complex_fraction_equality : ∃ (i : ℂ), i * i = -1 ∧ (2 * i) / (1 - i) = -1 + i := by
  sorry

end complex_fraction_equality_l1790_179041


namespace negation_of_existence_negation_of_quadratic_equation_l1790_179039

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - 2*x = 0) ↔ (∀ x : ℝ, x^2 - 2*x ≠ 0) := by sorry

end negation_of_existence_negation_of_quadratic_equation_l1790_179039


namespace triangle_side_value_l1790_179083

/-- In a triangle ABC, given specific conditions, prove that a = 2√3 -/
theorem triangle_side_value (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  A = 2 * C ∧  -- Given condition
  c = 2 ∧  -- Given condition
  a^2 = 4*b - 4 ∧  -- Given condition
  a / (Real.sin A) = b / (Real.sin B) ∧  -- Sine law
  a / (Real.sin A) = c / (Real.sin C) ∧  -- Sine law
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) ∧  -- Cosine law
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧  -- Cosine law
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)  -- Cosine law
  → a = 2 * Real.sqrt 3 := by sorry

end triangle_side_value_l1790_179083


namespace cubic_identity_l1790_179011

theorem cubic_identity (a b : ℝ) : (a + b) * (a^2 - a*b + b^2) = a^3 + b^3 := by
  sorry

end cubic_identity_l1790_179011


namespace margarets_mean_score_l1790_179071

def scores : List ℕ := [85, 87, 92, 93, 94, 98]

theorem margarets_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℕ), 
        cyprian_scores.length = 3 ∧ 
        margaret_scores.length = 3 ∧ 
        cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℕ), 
        cyprian_scores.length = 3 ∧ 
        cyprian_scores.sum / cyprian_scores.length = 90) :
  ∃ (margaret_scores : List ℕ), 
    margaret_scores.length = 3 ∧ 
    margaret_scores.sum / margaret_scores.length = 93 :=
sorry

end margarets_mean_score_l1790_179071


namespace tangent_slope_at_one_l1790_179038

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x - 2

-- State the theorem
theorem tangent_slope_at_one : 
  (deriv f) 1 = 3 := by sorry

end tangent_slope_at_one_l1790_179038


namespace line_slope_l1790_179060

theorem line_slope (α : Real) (h : Real.sin α + Real.cos α = 1/5) :
  Real.tan α = -4/3 := by
  sorry

end line_slope_l1790_179060


namespace inverse_of_A_l1790_179073

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; 4, -2]

theorem inverse_of_A : 
  (A⁻¹) = !![(-1 : ℝ), (3/2 : ℝ); (-2 : ℝ), (5/2 : ℝ)] := by sorry

end inverse_of_A_l1790_179073


namespace shorter_container_radius_l1790_179021

-- Define the containers
structure Container where
  radius : ℝ
  height : ℝ

-- Define the problem
theorem shorter_container_radius 
  (c1 c2 : Container) -- Two containers
  (h_volume : c1.radius ^ 2 * c1.height = c2.radius ^ 2 * c2.height) -- Equal volume
  (h_height : c2.height = 2 * c1.height) -- One height is double the other
  (h_tall_radius : c2.radius = 10) -- Radius of taller container is 10
  : c1.radius = 10 * Real.sqrt 2 := by
  sorry

end shorter_container_radius_l1790_179021


namespace sand_dune_probability_l1790_179023

/-- The probability that a sand dune remains -/
def P_remain : ℚ := 1 / 3

/-- The probability that a blown-out sand dune has a treasure -/
def P_treasure : ℚ := 1 / 5

/-- The probability that a sand dune has lucky coupons -/
def P_coupons : ℚ := 2 / 3

/-- The probability that a dune is formed in the evening -/
def P_evening : ℚ := 70 / 100

/-- The probability that a dune is formed in the morning -/
def P_morning : ℚ := 1 - P_evening

/-- The combined probability that a blown-out sand dune contains both the treasure and lucky coupons -/
def P_combined : ℚ := P_treasure * P_morning * P_coupons

theorem sand_dune_probability : P_combined = 2 / 25 := by
  sorry

end sand_dune_probability_l1790_179023


namespace colorings_count_l1790_179035

/-- Represents the colors available for coloring the cells -/
inductive Color
| Blue
| Red
| White

/-- Represents a cell in the figure -/
structure Cell :=
  (x : ℕ)
  (y : ℕ)

/-- Represents the entire figure to be colored -/
structure Figure :=
  (cells : List Cell)
  (neighbors : Cell → Cell → Bool)

/-- A coloring of the figure -/
def Coloring := Cell → Color

/-- Checks if a coloring is valid for the given figure -/
def is_valid_coloring (f : Figure) (c : Coloring) : Prop :=
  ∀ cell1 cell2, f.neighbors cell1 cell2 → c cell1 ≠ c cell2

/-- The specific figure described in the problem -/
def problem_figure : Figure := sorry

/-- The number of valid colorings for the problem figure -/
def num_valid_colorings (f : Figure) : ℕ := sorry

/-- The main theorem to be proved -/
theorem colorings_count :
  num_valid_colorings problem_figure = 3 * 48^4 := by sorry

end colorings_count_l1790_179035


namespace modulus_of_complex_number_l1790_179022

theorem modulus_of_complex_number (x : ℝ) (i : ℂ) : 
  i * i = -1 →
  (∃ (y : ℝ), (x + i) * (2 + i) = y * i) →
  Complex.abs (2 * x - i) = Real.sqrt 2 := by
sorry

end modulus_of_complex_number_l1790_179022


namespace calculate_expression_l1790_179024

theorem calculate_expression : 3000 * (3000 ^ 3000) + 3000 ^ 2 = 3000 ^ 3001 := by
  sorry

end calculate_expression_l1790_179024


namespace total_people_needed_l1790_179065

def people_per_car : ℕ := 5

def people_per_truck (people_per_car : ℕ) : ℕ := 2 * people_per_car

def people_for_cars (num_cars : ℕ) (people_per_car : ℕ) : ℕ :=
  num_cars * people_per_car

def people_for_trucks (num_trucks : ℕ) (people_per_truck : ℕ) : ℕ :=
  num_trucks * people_per_truck

theorem total_people_needed (num_cars num_trucks : ℕ) :
  people_for_cars num_cars people_per_car +
  people_for_trucks num_trucks (people_per_truck people_per_car) = 60 :=
by
  sorry

end total_people_needed_l1790_179065


namespace veronica_photos_l1790_179084

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem veronica_photos (x : ℕ) 
  (h1 : choose x 3 + choose x 4 = 15) : x = 7 := by
  sorry

end veronica_photos_l1790_179084


namespace translate_line_upward_l1790_179094

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically --/
def translateLine (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

theorem translate_line_upward (original : Line) (shift : ℝ) :
  original.slope = -2 ∧ shift = 4 →
  translateLine original shift = { slope := -2, intercept := 4 } := by
  sorry

#check translate_line_upward

end translate_line_upward_l1790_179094


namespace fraction_difference_l1790_179096

def fractions : List ℚ := [2/3, 3/4, 4/5, 5/7, 7/10, 11/13, 14/19]

theorem fraction_difference : 
  (List.maximum fractions).get! - (List.minimum fractions).get! = 11/13 - 2/3 := by
  sorry

end fraction_difference_l1790_179096


namespace exam_problem_l1790_179088

/-- Proves that given the conditions of the exam problem, the number of students is 56 -/
theorem exam_problem (N : ℕ) (T : ℕ) : 
  T = 80 * N →                        -- The total marks equal 80 times the number of students
  (T - 160) / (N - 8) = 90 →          -- After excluding 8 students, the new average is 90
  N = 56 :=                           -- The number of students is 56
by
  sorry

#check exam_problem

end exam_problem_l1790_179088


namespace project_budget_increase_l1790_179042

/-- Proves that the annual increase in budget for project Q is $50,000 --/
theorem project_budget_increase (initial_q initial_v annual_decrease_v : ℕ) 
  (h1 : initial_q = 540000)
  (h2 : initial_v = 780000)
  (h3 : annual_decrease_v = 10000)
  (h4 : ∃ (annual_increase_q : ℕ), 
    initial_q + 4 * annual_increase_q = initial_v - 4 * annual_decrease_v) :
  ∃ (annual_increase_q : ℕ), annual_increase_q = 50000 := by
  sorry

end project_budget_increase_l1790_179042


namespace total_water_volume_l1790_179020

theorem total_water_volume (num_boxes : ℕ) (bottles_per_box : ℕ) (bottle_capacity : ℝ) (fill_ratio : ℝ) : 
  num_boxes = 10 →
  bottles_per_box = 50 →
  bottle_capacity = 12 →
  fill_ratio = 3/4 →
  (num_boxes * bottles_per_box * bottle_capacity * fill_ratio : ℝ) = 4500 := by
  sorry

end total_water_volume_l1790_179020


namespace inverse_sum_product_l1790_179054

theorem inverse_sum_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 2 * x + y / 2 ≠ 0) :
  (2 * x + y / 2)⁻¹ * ((2 * x)⁻¹ + (y / 2)⁻¹) = (x * y)⁻¹ := by
  sorry

end inverse_sum_product_l1790_179054


namespace couple_seating_arrangements_l1790_179012

/-- Represents a couple (a boy and a girl) -/
structure Couple :=
  (boy : Nat)
  (girl : Nat)

/-- Represents a seating arrangement on the bench -/
structure Arrangement :=
  (seat1 : Nat)
  (seat2 : Nat)
  (seat3 : Nat)
  (seat4 : Nat)

/-- Checks if a given arrangement is valid (each couple sits together) -/
def isValidArrangement (c1 c2 : Couple) (arr : Arrangement) : Prop :=
  (arr.seat1 = c1.boy ∧ arr.seat2 = c1.girl ∧ arr.seat3 = c2.boy ∧ arr.seat4 = c2.girl) ∨
  (arr.seat1 = c1.girl ∧ arr.seat2 = c1.boy ∧ arr.seat3 = c2.boy ∧ arr.seat4 = c2.girl) ∨
  (arr.seat1 = c1.boy ∧ arr.seat2 = c1.girl ∧ arr.seat3 = c2.girl ∧ arr.seat4 = c2.boy) ∨
  (arr.seat1 = c1.girl ∧ arr.seat2 = c1.boy ∧ arr.seat3 = c2.girl ∧ arr.seat4 = c2.boy) ∨
  (arr.seat1 = c2.boy ∧ arr.seat2 = c2.girl ∧ arr.seat3 = c1.boy ∧ arr.seat4 = c1.girl) ∨
  (arr.seat1 = c2.girl ∧ arr.seat2 = c2.boy ∧ arr.seat3 = c1.boy ∧ arr.seat4 = c1.girl) ∨
  (arr.seat1 = c2.boy ∧ arr.seat2 = c2.girl ∧ arr.seat3 = c1.girl ∧ arr.seat4 = c1.boy) ∨
  (arr.seat1 = c2.girl ∧ arr.seat2 = c2.boy ∧ arr.seat3 = c1.girl ∧ arr.seat4 = c1.boy)

/-- The main theorem: there are exactly 8 valid seating arrangements -/
theorem couple_seating_arrangements (c1 c2 : Couple) :
  ∃! (arrangements : Finset Arrangement), 
    (∀ arr ∈ arrangements, isValidArrangement c1 c2 arr) ∧
    arrangements.card = 8 :=
by
  sorry

end couple_seating_arrangements_l1790_179012


namespace boxes_with_neither_l1790_179080

theorem boxes_with_neither (total : ℕ) (with_stickers : ℕ) (with_cards : ℕ) (with_both : ℕ) :
  total = 15 →
  with_stickers = 8 →
  with_cards = 5 →
  with_both = 3 →
  total - (with_stickers + with_cards - with_both) = 5 :=
by
  sorry

end boxes_with_neither_l1790_179080


namespace complex_magnitude_sum_reciprocal_l1790_179033

theorem complex_magnitude_sum_reciprocal (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
  sorry

end complex_magnitude_sum_reciprocal_l1790_179033


namespace exactly_one_two_black_mutually_exclusive_not_complementary_l1790_179014

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
def DrawOutcome := Prod BallColor BallColor

/-- The set of all possible outcomes when drawing two balls -/
def SampleSpace : Set DrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def ExactlyOneBlack : Set DrawOutcome := sorry

/-- The event of drawing exactly two black balls -/
def ExactlyTwoBlack : Set DrawOutcome := sorry

/-- Two events are mutually exclusive if their intersection is empty -/
def MutuallyExclusive (A B : Set DrawOutcome) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary if their union is the entire sample space -/
def Complementary (A B : Set DrawOutcome) : Prop :=
  A ∪ B = SampleSpace

theorem exactly_one_two_black_mutually_exclusive_not_complementary :
  MutuallyExclusive ExactlyOneBlack ExactlyTwoBlack ∧
  ¬Complementary ExactlyOneBlack ExactlyTwoBlack :=
sorry

end exactly_one_two_black_mutually_exclusive_not_complementary_l1790_179014


namespace probability_not_all_same_dice_probability_not_all_same_five_eight_sided_dice_l1790_179062

theorem probability_not_all_same_dice (n : ℕ) (s : ℕ) (hn : n > 0) (hs : s > 0) : 
  1 - (s : ℚ) / (s ^ n : ℚ) = (s ^ n - s : ℚ) / (s ^ n : ℚ) :=
by sorry

-- The probability of not getting all the same numbers when rolling five fair 8-sided dice
theorem probability_not_all_same_five_eight_sided_dice : 
  1 - (8 : ℚ) / (8^5 : ℚ) = 4095 / 4096 :=
by sorry

end probability_not_all_same_dice_probability_not_all_same_five_eight_sided_dice_l1790_179062


namespace power_product_squared_l1790_179092

theorem power_product_squared : (3^5 * 6^5)^2 = 3570467226624 := by
  sorry

end power_product_squared_l1790_179092


namespace dave_winfield_home_runs_l1790_179095

/-- Dave Winfield's career home run count -/
def dave_winfield_hr : ℕ := 465

/-- Hank Aaron's career home run count -/
def hank_aaron_hr : ℕ := 755

/-- Theorem stating Dave Winfield's home run count based on the given conditions -/
theorem dave_winfield_home_runs :
  dave_winfield_hr = 465 ∧
  hank_aaron_hr = 2 * dave_winfield_hr - 175 :=
by sorry

end dave_winfield_home_runs_l1790_179095


namespace compare_expressions_l1790_179082

theorem compare_expressions (m x : ℝ) : x^2 - x + 1 > -2*m^2 - 2*m*x := by
  sorry

end compare_expressions_l1790_179082


namespace smallest_lcm_with_gcd_5_l1790_179009

theorem smallest_lcm_with_gcd_5 (k l : ℕ) : 
  k ≥ 1000 ∧ k < 10000 ∧ l ≥ 1000 ∧ l < 10000 ∧ Nat.gcd k l = 5 →
  Nat.lcm k l ≥ 201000 := by
sorry

end smallest_lcm_with_gcd_5_l1790_179009


namespace coin_division_problem_l1790_179005

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (∀ m : ℕ, m > 0 → m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 8 = 6) →
  (n % 7 = 5) →
  (n % 9 = 0) :=
by sorry

end coin_division_problem_l1790_179005


namespace price_reduction_equation_l1790_179018

theorem price_reduction_equation (x : ℝ) : 
  (∀ (original_price final_price : ℝ),
    original_price = 100 ∧ 
    final_price = 81 ∧ 
    final_price = original_price * (1 - x)^2) →
  100 * (1 - x)^2 = 81 :=
by sorry

end price_reduction_equation_l1790_179018


namespace no_real_sqrt_negative_quadratic_l1790_179030

theorem no_real_sqrt_negative_quadratic :
  ∀ x : ℝ, ¬ ∃ y : ℝ, y ^ 2 = -(x ^ 2 + 2 * x + 5) :=
by sorry

end no_real_sqrt_negative_quadratic_l1790_179030


namespace juans_number_problem_l1790_179008

theorem juans_number_problem (n : ℝ) : 
  (2 * (n + 3)^2 - 3) / 2 = 49 → n = Real.sqrt (101 / 2) - 3 := by sorry

end juans_number_problem_l1790_179008


namespace log_sum_equality_l1790_179049

theorem log_sum_equality : Real.log 3 / Real.log 2 * (Real.log 4 / Real.log 3) + Real.log 8 / Real.log 4 + (5 : ℝ) ^ (Real.log 2 / Real.log 5) = 11 / 2 := by
  sorry

end log_sum_equality_l1790_179049


namespace sqrt_x_plus_4_meaningful_l1790_179007

theorem sqrt_x_plus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 4) ↔ x ≥ -4 := by sorry

end sqrt_x_plus_4_meaningful_l1790_179007


namespace parentheses_removal_l1790_179037

theorem parentheses_removal (a b c : ℝ) : 3*a - (2*b - c) = 3*a - 2*b + c := by
  sorry

end parentheses_removal_l1790_179037


namespace A_minus_B_equality_A_minus_B_at_negative_two_l1790_179047

-- Define A and B as functions of x
def A (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 2
def B (x : ℝ) : ℝ := x^2 - 3 * x - 2

-- Theorem 1: A - B = x² + 4 for all real x
theorem A_minus_B_equality (x : ℝ) : A x - B x = x^2 + 4 := by
  sorry

-- Theorem 2: A - B = 8 when x = -2
theorem A_minus_B_at_negative_two : A (-2) - B (-2) = 8 := by
  sorry

end A_minus_B_equality_A_minus_B_at_negative_two_l1790_179047


namespace strawberry_loss_l1790_179048

theorem strawberry_loss (total_weight : ℕ) (marco_weight : ℕ) (dad_weight : ℕ) 
  (h1 : total_weight = 36)
  (h2 : marco_weight = 12)
  (h3 : dad_weight = 16) :
  total_weight - (marco_weight + dad_weight) = 8 := by
  sorry

end strawberry_loss_l1790_179048


namespace quadratic_inequality_solution_set_l1790_179067

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 + x - 12 ≥ 0 ↔ x ≤ -4 ∨ x ≥ 3 := by sorry

end quadratic_inequality_solution_set_l1790_179067


namespace gold_coin_puzzle_l1790_179045

theorem gold_coin_puzzle (n : ℕ) (c : ℕ) : 
  (∃ k : ℕ, n = 11 * (c - 3) ∧ k = c - 3) ∧ 
  n = 7 * c + 5 →
  n = 75 :=
by sorry

end gold_coin_puzzle_l1790_179045


namespace perimeter_of_square_d_l1790_179068

/-- Given a square C with perimeter 32 cm and a square D with area equal to one-third the area of square C, 
    the perimeter of square D is (32√3)/3 cm. -/
theorem perimeter_of_square_d (C D : Real) : 
  (C = 32) →  -- perimeter of square C is 32 cm
  (D^2 = (C/4)^2 / 3) →  -- area of square D is one-third the area of square C
  (4 * D = 32 * Real.sqrt 3 / 3) := by  -- perimeter of square D is (32√3)/3 cm
sorry

end perimeter_of_square_d_l1790_179068


namespace bread_slices_eaten_for_breakfast_l1790_179015

theorem bread_slices_eaten_for_breakfast 
  (total_slices : ℕ) 
  (lunch_slices : ℕ) 
  (remaining_slices : ℕ) 
  (h1 : total_slices = 12)
  (h2 : lunch_slices = 2)
  (h3 : remaining_slices = 6) :
  (total_slices - (remaining_slices + lunch_slices)) / total_slices = 1 / 3 := by
  sorry

end bread_slices_eaten_for_breakfast_l1790_179015


namespace abs_value_difference_l1790_179001

theorem abs_value_difference (a b : ℝ) (ha : |a| = 3) (hb : |b| = 5) (hab : a > b) :
  a - b = 8 := by
  sorry

end abs_value_difference_l1790_179001


namespace sin_150_degrees_l1790_179089

theorem sin_150_degrees : Real.sin (150 * Real.pi / 180) = 1 / 2 := by sorry

end sin_150_degrees_l1790_179089


namespace shaded_area_fraction_l1790_179056

theorem shaded_area_fraction (length width : ℕ) (quarter_shaded_fraction : ℚ) (unshaded_squares : ℕ) :
  length = 15 →
  width = 20 →
  quarter_shaded_fraction = 1/4 →
  unshaded_squares = 9 →
  (quarter_shaded_fraction * (1/4 * (length * width)) - unshaded_squares) / (length * width) = 13/400 :=
by sorry

end shaded_area_fraction_l1790_179056


namespace geometric_sequence_a7_l1790_179029

/-- Geometric sequence with a_3 = 1 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ a 3 = 1

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h : geometric_sequence a) 
  (h_prod : a 6 * a 8 = 64) : a 7 = 8 := by
  sorry

end geometric_sequence_a7_l1790_179029


namespace circle_radius_bounds_l1790_179085

/-- Given a quadrilateral ABCD circumscribed around a circle, where the
    tangency points divide AB into segments a and b, and AD into segments a and c,
    prove that the radius r of the circle satisfies the given inequality. -/
theorem circle_radius_bounds (a b c r : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) : 
  Real.sqrt ((a * b * c) / (a + b + c)) < r ∧ 
  r < Real.sqrt (a * b + b * c + c * a) := by
  sorry

end circle_radius_bounds_l1790_179085


namespace evenResultCombinations_l1790_179051

def Operation : Type := Nat → Nat

def increaseBy2 : Operation := λ n => n + 2
def increaseBy3 : Operation := λ n => n + 3
def multiplyBy2 : Operation := λ n => n * 2

def applyOperations (ops : List Operation) (initial : Nat) : Nat :=
  ops.foldl (λ acc op => op acc) initial

def isEven (n : Nat) : Bool := n % 2 = 0

def allCombinations (n : Nat) : List (List Operation) :=
  sorry -- Implementation of all combinations of 6 operations

theorem evenResultCombinations :
  let initial := 1
  let operations := [increaseBy2, increaseBy3, multiplyBy2]
  let combinations := allCombinations 6
  (combinations.filter (λ ops => isEven (applyOperations ops initial))).length = 486 := by
  sorry

end evenResultCombinations_l1790_179051


namespace vector_computation_l1790_179028

theorem vector_computation :
  4 • !![3, -9] - 3 • !![2, -8] + 2 • !![1, -6] = !![8, -24] := by
  sorry

end vector_computation_l1790_179028


namespace right_triangle_perimeter_l1790_179070

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) :
  area = 150 →
  leg1 = 30 →
  area = (1 / 2) * leg1 * leg2 →
  hypotenuse^2 = leg1^2 + leg2^2 →
  leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10 :=
by sorry

end right_triangle_perimeter_l1790_179070


namespace trigonometric_identity_l1790_179090

theorem trigonometric_identity : 
  (Real.sin (47 * π / 180) - Real.sin (17 * π / 180) * Real.cos (30 * π / 180)) / 
  Real.cos (17 * π / 180) = 1/2 := by
  sorry

end trigonometric_identity_l1790_179090


namespace house_cost_is_280k_l1790_179055

/-- Calculates the total cost of a house given the initial deposit, mortgage duration, and monthly payment. -/
def house_cost (deposit : ℕ) (duration_years : ℕ) (monthly_payment : ℕ) : ℕ :=
  deposit + duration_years * 12 * monthly_payment

/-- Proves that the total cost of the house is $280,000 given the specified conditions. -/
theorem house_cost_is_280k :
  house_cost 40000 10 2000 = 280000 :=
by sorry

end house_cost_is_280k_l1790_179055


namespace smaller_solution_quadratic_l1790_179058

theorem smaller_solution_quadratic (x : ℝ) : 
  x^2 - 9*x + 20 = 0 → x = 4 ∨ x = 5 → min x (15 - x) = 4 := by
  sorry

end smaller_solution_quadratic_l1790_179058


namespace smallest_solution_comparison_l1790_179087

theorem smallest_solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0) :
  (∃ x y : ℝ, x < y ∧ p * x^2 + q = 0 ∧ p' * y^2 + q' = 0 ∧
    (∀ z : ℝ, p * z^2 + q = 0 → x ≤ z) ∧
    (∀ w : ℝ, p' * w^2 + q' = 0 → y ≤ w)) ↔
  Real.sqrt (q' / p') < Real.sqrt (q / p) :=
sorry

end smallest_solution_comparison_l1790_179087


namespace rajan_income_l1790_179044

/-- Represents the financial situation of two individuals --/
structure FinancialSituation where
  income_ratio : Rat
  expenditure_ratio : Rat
  savings : ℕ

/-- Calculates the income of the first person given a financial situation --/
def calculate_income (situation : FinancialSituation) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, Rajan's income is $7000 --/
theorem rajan_income (situation : FinancialSituation) 
  (h1 : situation.income_ratio = 7 / 6)
  (h2 : situation.expenditure_ratio = 6 / 5)
  (h3 : situation.savings = 1000) :
  calculate_income situation = 7000 :=
sorry

end rajan_income_l1790_179044


namespace circle_area_tangent_to_hyperbola_and_xaxis_l1790_179093

/-- A hyperbola in the xy-plane -/
def Hyperbola (x y : ℝ) : Prop := x^2 - 20*y^2 = 24

/-- A circle in the xy-plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- A point is on the x-axis if its y-coordinate is 0 -/
def OnXAxis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A circle is tangent to the hyperbola if there exists a point that satisfies both equations -/
def TangentToHyperbola (c : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ c ∧ Hyperbola p.1 p.2

/-- A circle is tangent to the x-axis if there exists a point on the circle that is also on the x-axis -/
def TangentToXAxis (c : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ c ∧ OnXAxis p

theorem circle_area_tangent_to_hyperbola_and_xaxis :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    let c := Circle center radius
    TangentToHyperbola c ∧ TangentToXAxis c ∧ π * radius^2 = 504 * π :=
by sorry

end circle_area_tangent_to_hyperbola_and_xaxis_l1790_179093


namespace greatest_integer_satisfying_inequality_l1790_179064

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ+, (x : ℝ)^6 / (x : ℝ)^3 < 18 → x ≤ 2 :=
by
  sorry

#check greatest_integer_satisfying_inequality

end greatest_integer_satisfying_inequality_l1790_179064


namespace sqrt_x_squared_nonnegative_l1790_179013

theorem sqrt_x_squared_nonnegative (x : ℝ) : 0 ≤ Real.sqrt (x^2) := by sorry

end sqrt_x_squared_nonnegative_l1790_179013


namespace solid_color_marbles_l1790_179052

theorem solid_color_marbles (total_marbles : ℕ) (solid_color_percent : ℚ) (solid_yellow_percent : ℚ)
  (h1 : solid_color_percent = 90 / 100)
  (h2 : solid_yellow_percent = 5 / 100) :
  solid_color_percent - solid_yellow_percent = 85 / 100 := by
  sorry

end solid_color_marbles_l1790_179052


namespace p_sufficient_not_necessary_l1790_179043

-- Define the real number x
variable (x : ℝ)

-- Define condition p
def p (x : ℝ) : Prop := |x - 2| < 1

-- Define condition q
def q (x : ℝ) : Prop := 1 < x ∧ x < 5

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by
  sorry


end p_sufficient_not_necessary_l1790_179043


namespace problem_statement_l1790_179086

theorem problem_statement (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x * (y + z) = 132)
  (h2 : z * (x + y) = 180)
  (h3 : x * y * z = 160) : 
  y * (z + x) = 160 := by
sorry

end problem_statement_l1790_179086


namespace gcd_seven_eight_factorial_l1790_179069

theorem gcd_seven_eight_factorial : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcd_seven_eight_factorial_l1790_179069


namespace simplify_expression_l1790_179076

theorem simplify_expression (a b c : ℝ) :
  (18 * a + 72 * b + 30 * c) + (15 * a + 40 * b - 20 * c) - (12 * a + 60 * b + 25 * c) = 21 * a + 52 * b - 15 * c := by
  sorry

end simplify_expression_l1790_179076


namespace haley_laundry_loads_l1790_179072

/-- The number of loads required to wash a given number of clothing items with a fixed-capacity washing machine. -/
def loads_required (machine_capacity : ℕ) (total_items : ℕ) : ℕ :=
  (total_items + machine_capacity - 1) / machine_capacity

theorem haley_laundry_loads :
  let machine_capacity : ℕ := 7
  let shirts : ℕ := 2
  let sweaters : ℕ := 33
  let total_items : ℕ := shirts + sweaters
  loads_required machine_capacity total_items = 5 := by
sorry

end haley_laundry_loads_l1790_179072


namespace absolute_value_ratio_l1790_179091

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 10*a*b) :
  |((a + b) / (a - b))| = Real.sqrt 6 / 2 := by
  sorry

end absolute_value_ratio_l1790_179091


namespace chairs_to_remove_l1790_179079

theorem chairs_to_remove (initial_chairs : Nat) (chairs_per_row : Nat) (participants : Nat) 
  (chairs_to_remove : Nat) :
  initial_chairs = 169 →
  chairs_per_row = 13 →
  participants = 95 →
  chairs_to_remove = 65 →
  (initial_chairs - chairs_to_remove) % chairs_per_row = 0 ∧
  initial_chairs - chairs_to_remove ≥ participants ∧
  ∀ n : Nat, n < chairs_to_remove → 
    (initial_chairs - n) % chairs_per_row ≠ 0 ∨ 
    initial_chairs - n < participants := by
  sorry

end chairs_to_remove_l1790_179079


namespace count_satisfying_numbers_l1790_179025

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  thousands_nonzero : thousands > 0
  all_digits : thousands < 10 ∧ hundreds < 10 ∧ tens < 10 ∧ units < 10

/-- Checks if a four-digit number satisfies the given conditions -/
def satisfiesConditions (n : FourDigitNumber) : Prop :=
  n.thousands = 2 ∧
  n.hundreds % 2 = 0 ∧
  n.units = n.thousands + n.hundreds + n.tens

theorem count_satisfying_numbers :
  (∃ (s : Finset FourDigitNumber),
    (∀ n ∈ s, satisfiesConditions n) ∧
    s.card = 16 ∧
    (∀ n : FourDigitNumber, satisfiesConditions n → n ∈ s)) := by
  sorry

#check count_satisfying_numbers

end count_satisfying_numbers_l1790_179025


namespace equation_C_most_suitable_l1790_179046

-- Define the equations
def equation_A : ℝ → Prop := λ x ↦ 2 * x^2 = 8
def equation_B : ℝ → Prop := λ x ↦ x * (x + 2) = x + 2
def equation_C : ℝ → Prop := λ x ↦ x^2 - 2*x = 3
def equation_D : ℝ → Prop := λ x ↦ 2 * x^2 + x - 1 = 0

-- Define a predicate for suitability for completing the square method
def suitable_for_completing_square (eq : ℝ → Prop) : Prop := sorry

-- Theorem stating that equation C is most suitable for completing the square
theorem equation_C_most_suitable :
  suitable_for_completing_square equation_C ∧
  (¬suitable_for_completing_square equation_A ∨
   ¬suitable_for_completing_square equation_B ∨
   ¬suitable_for_completing_square equation_D) :=
sorry

end equation_C_most_suitable_l1790_179046


namespace exists_positive_integer_solution_l1790_179016

theorem exists_positive_integer_solution :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + 2 * y = 7 :=
by sorry

end exists_positive_integer_solution_l1790_179016


namespace tangent_circles_F_value_l1790_179004

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle C₂ with equation x² + y² - 6x - 8y + F = 0 -/
def C₂ (F : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + F = 0}

/-- Two circles are internally tangent if the distance between their centers
    is equal to the absolute difference of their radii -/
def internally_tangent (S T : Set (ℝ × ℝ)) : Prop :=
  ∃ (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
    (∀ p, p ∈ S ↔ (p.1 - c₁.1)^2 + (p.2 - c₁.2)^2 = r₁^2) ∧
    (∀ p, p ∈ T ↔ (p.1 - c₂.1)^2 + (p.2 - c₂.2)^2 = r₂^2) ∧
    (c₂.1 - c₁.1)^2 + (c₂.2 - c₁.2)^2 = (r₂ - r₁)^2

/-- If C₁ and C₂ are internally tangent, then F = -11 -/
theorem tangent_circles_F_value :
  internally_tangent C₁ (C₂ F) → F = -11 := by sorry

end tangent_circles_F_value_l1790_179004


namespace expression_factorization_l1790_179031

theorem expression_factorization (a b c : ℝ) : 
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  (a - b) * (b - c) * (c - a) * (-(a + b + c) * (a^2 + b^2 + c^2 + a*b + b*c + a*c)) := by
  sorry

end expression_factorization_l1790_179031


namespace book_selection_combinations_l1790_179010

theorem book_selection_combinations (n k : ℕ) (hn : n = 13) (hk : k = 3) :
  Nat.choose n k = 286 := by
  sorry

end book_selection_combinations_l1790_179010


namespace inequality_implication_l1790_179027

theorem inequality_implication (m n : ℝ) : -m/2 < -n/6 → 3*m > n := by
  sorry

end inequality_implication_l1790_179027


namespace lcm_gcf_product_24_36_l1790_179066

theorem lcm_gcf_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end lcm_gcf_product_24_36_l1790_179066


namespace sqrt_less_than_3y_iff_y_greater_than_one_ninth_l1790_179099

theorem sqrt_less_than_3y_iff_y_greater_than_one_ninth (y : ℝ) (h : y > 0) :
  Real.sqrt y < 3 * y ↔ y > 1/9 := by
  sorry

end sqrt_less_than_3y_iff_y_greater_than_one_ninth_l1790_179099


namespace difference_of_place_values_l1790_179002

def place_value (digit : ℕ) (place : ℕ) : ℕ := digit * (10 ^ place)

def sum_place_values_27242 : ℕ := place_value 2 0 + place_value 2 2

def sum_place_values_7232062 : ℕ := place_value 2 1 + place_value 2 6

theorem difference_of_place_values : 
  sum_place_values_7232062 - sum_place_values_27242 = 1999818 := by sorry

end difference_of_place_values_l1790_179002


namespace ellipse_sum_range_l1790_179053

theorem ellipse_sum_range (x y : ℝ) (h : 9 * x^2 + 16 * y^2 = 144) :
  5 ≤ x + y + 10 ∧ x + y + 10 ≤ 15 := by
  sorry

end ellipse_sum_range_l1790_179053


namespace multiplication_value_l1790_179059

theorem multiplication_value : ∃ x : ℚ, (5 / 6) * x = 10 ∧ x = 12 := by
  sorry

end multiplication_value_l1790_179059


namespace total_amount_calculation_l1790_179078

/-- Calculates the total amount after simple interest is applied -/
def total_amount_after_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Theorem: The total amount after interest for the given conditions -/
theorem total_amount_calculation :
  let principal : ℝ := 979.0209790209791
  let rate : ℝ := 0.06
  let time : ℝ := 2.4
  total_amount_after_interest principal rate time = 1120.0649350649352 :=
by sorry

end total_amount_calculation_l1790_179078


namespace sum_of_common_terms_equal_1472_l1790_179063

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def common_terms (seq1 seq2 : List ℕ) : List ℕ :=
  seq1.filter (fun x => seq2.contains x)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem sum_of_common_terms_equal_1472 :
  let seq1 := arithmetic_sequence 2 4 48
  let seq2 := arithmetic_sequence 2 6 34
  let common := common_terms seq1 seq2
  sum_list common = 1472 := by
  sorry

end sum_of_common_terms_equal_1472_l1790_179063


namespace rhombus_longer_diagonal_l1790_179034

/-- A rhombus with given side length and shorter diagonal has a specific longer diagonal length -/
theorem rhombus_longer_diagonal (side_length shorter_diagonal : ℝ) 
  (h1 : side_length = 65)
  (h2 : shorter_diagonal = 72) : 
  ∃ longer_diagonal : ℝ, longer_diagonal = 108 ∧ 
  longer_diagonal^2 = 4 * (side_length^2 - (shorter_diagonal/2)^2) := by
  sorry

end rhombus_longer_diagonal_l1790_179034


namespace line_equivalence_l1790_179000

/-- Given a line in vector form, prove it's equivalent to a specific slope-intercept form --/
theorem line_equivalence (x y : ℝ) : 
  4 * (x + 2) - 3 * (y - 8) = 0 ↔ y = (4/3) * x + 32/3 := by
  sorry

end line_equivalence_l1790_179000


namespace complement_union_theorem_l1790_179040

-- Define the universe
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define sets A and B
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

-- State the theorem
theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4, 5} := by sorry

end complement_union_theorem_l1790_179040


namespace joan_sofa_cost_l1790_179003

theorem joan_sofa_cost (joan karl : ℕ) 
  (h1 : joan + karl = 600)
  (h2 : 2 * joan = karl + 90) : 
  joan = 230 := by
sorry

end joan_sofa_cost_l1790_179003


namespace smallest_delightful_integer_l1790_179077

/-- Definition of a delightful integer -/
def IsDelightful (B : ℤ) : Prop :=
  ∃ n : ℕ, (n + 1) * (2 * B + n) = 6100

/-- The smallest delightful integer -/
theorem smallest_delightful_integer :
  IsDelightful (-38) ∧ ∀ B : ℤ, B < -38 → ¬IsDelightful B :=
by sorry

end smallest_delightful_integer_l1790_179077


namespace complex_modulus_one_l1790_179006

theorem complex_modulus_one (z : ℂ) (h : z * (1 + Complex.I) = (1 - Complex.I)) : Complex.abs z = 1 := by
  sorry

end complex_modulus_one_l1790_179006


namespace expression_equals_one_l1790_179098

theorem expression_equals_one (x : ℝ) (h1 : x^3 ≠ 2) (h2 : x^3 ≠ -2) :
  ((x+2)^3 * (x^2-x+2)^3 / (x^3+2)^3)^3 * ((x-2)^3 * (x^2+x+2)^3 / (x^3-2)^3)^3 = 1 := by
  sorry

end expression_equals_one_l1790_179098
