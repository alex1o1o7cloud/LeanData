import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_negatives_l3747_374760

theorem sum_of_negatives : (-4) + (-6) = -10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_negatives_l3747_374760


namespace NUMINAMATH_CALUDE_flies_needed_for_week_l3747_374732

/-- The number of flies a frog eats per day -/
def flies_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of flies Betty caught in total -/
def flies_caught : ℕ := 11

/-- The number of flies that escaped -/
def flies_escaped : ℕ := 1

/-- Theorem stating how many more flies Betty needs for a week -/
theorem flies_needed_for_week : 
  flies_per_day * days_in_week - (flies_caught - flies_escaped) = 4 := by
  sorry

end NUMINAMATH_CALUDE_flies_needed_for_week_l3747_374732


namespace NUMINAMATH_CALUDE_distance_from_negative_two_l3747_374711

-- Define the distance function on the real number line
def distance (x y : ℝ) : ℝ := |x - y|

-- Theorem statement
theorem distance_from_negative_two :
  ∀ x : ℝ, distance x (-2) = 3 ↔ x = -5 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_negative_two_l3747_374711


namespace NUMINAMATH_CALUDE_element_in_set_l3747_374739

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : (U \ M) = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l3747_374739


namespace NUMINAMATH_CALUDE_markup_rate_l3747_374746

theorem markup_rate (selling_price : ℝ) (profit_rate : ℝ) (expense_rate : ℝ) : 
  selling_price = 5 → 
  profit_rate = 0.1 → 
  expense_rate = 0.15 → 
  (selling_price / (selling_price * (1 - profit_rate - expense_rate)) - 1) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_markup_rate_l3747_374746


namespace NUMINAMATH_CALUDE_ten_n_value_l3747_374740

theorem ten_n_value (n : ℝ) (h : 2 * n = 14) : 10 * n = 70 := by sorry

end NUMINAMATH_CALUDE_ten_n_value_l3747_374740


namespace NUMINAMATH_CALUDE_vegetarian_gluten_free_fraction_is_one_twentyfifth_l3747_374744

/-- Represents the menu of a restaurant --/
structure Menu where
  total_dishes : ℕ
  vegetarian_dishes : ℕ
  gluten_free_vegetarian_dishes : ℕ

/-- The fraction of dishes that are both vegetarian and gluten-free --/
def vegetarian_gluten_free_fraction (menu : Menu) : ℚ :=
  menu.gluten_free_vegetarian_dishes / menu.total_dishes

/-- Theorem stating the fraction of vegetarian and gluten-free dishes --/
theorem vegetarian_gluten_free_fraction_is_one_twentyfifth 
  (menu : Menu) 
  (h1 : menu.vegetarian_dishes = 5)
  (h2 : menu.vegetarian_dishes = menu.total_dishes / 5)
  (h3 : menu.gluten_free_vegetarian_dishes = menu.vegetarian_dishes - 4) :
  vegetarian_gluten_free_fraction menu = 1 / 25 := by
  sorry

#check vegetarian_gluten_free_fraction_is_one_twentyfifth

end NUMINAMATH_CALUDE_vegetarian_gluten_free_fraction_is_one_twentyfifth_l3747_374744


namespace NUMINAMATH_CALUDE_swimmer_problem_l3747_374791

theorem swimmer_problem (swimmer_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  swimmer_speed = 5 →
  downstream_distance = 54 →
  upstream_distance = 6 →
  ∃ (time current_speed : ℝ),
    time > 0 ∧
    current_speed > 0 ∧
    current_speed < swimmer_speed ∧
    time = downstream_distance / (swimmer_speed + current_speed) ∧
    time = upstream_distance / (swimmer_speed - current_speed) ∧
    time = 6 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_problem_l3747_374791


namespace NUMINAMATH_CALUDE_remainder_of_2357916_div_8_l3747_374713

theorem remainder_of_2357916_div_8 : 2357916 % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2357916_div_8_l3747_374713


namespace NUMINAMATH_CALUDE_no_winning_strategy_strategy_independent_no_strategy_better_than_half_l3747_374777

/-- Represents a deck of cards with red and black suits -/
structure Deck :=
  (red : ℕ)
  (black : ℕ)

/-- A strategy is a function that decides whether to stop based on the current deck state -/
def Strategy := Deck → Bool

/-- The probability of winning given a deck state -/
def winProbability (d : Deck) : ℚ :=
  d.red / (d.red + d.black)

/-- Theorem stating that no strategy can achieve a winning probability greater than 0.5 -/
theorem no_winning_strategy (d : Deck) (s : Strategy) :
  d.red = d.black → winProbability d ≤ 1/2 := by
  sorry

/-- Theorem stating that the winning probability is independent of the strategy -/
theorem strategy_independent (d : Deck) (s₁ s₂ : Strategy) :
  winProbability d = winProbability d := by
  sorry

/-- Main theorem: No strategy exists that guarantees a winning probability greater than 0.5 -/
theorem no_strategy_better_than_half (d : Deck) :
  d.red = d.black → ∀ s : Strategy, winProbability d ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_winning_strategy_strategy_independent_no_strategy_better_than_half_l3747_374777


namespace NUMINAMATH_CALUDE_divisibility_implication_l3747_374730

theorem divisibility_implication (a b : ℕ) (h : a < 1000) :
  (b^10 ∣ a^21) → (b ∣ a^2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l3747_374730


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_real_range_l3747_374706

theorem sqrt_x_plus_one_real_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_real_range_l3747_374706


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_l3747_374712

/-- The cost of buying fruits given their prices per dozen and quantities --/
def total_cost (apple_price_per_dozen : ℕ) (pear_price_per_dozen : ℕ) (apple_quantity : ℕ) (pear_quantity : ℕ) : ℕ :=
  apple_price_per_dozen * apple_quantity + pear_price_per_dozen * pear_quantity

/-- Theorem: Given the prices and quantities of apples and pears, the total cost is 1260 dollars --/
theorem fruit_purchase_cost :
  total_cost 40 50 14 14 = 1260 := by
  sorry

#eval total_cost 40 50 14 14

end NUMINAMATH_CALUDE_fruit_purchase_cost_l3747_374712


namespace NUMINAMATH_CALUDE_bacteria_increase_l3747_374717

theorem bacteria_increase (original : ℕ) (current : ℕ) (increase : ℕ) : 
  original = 600 → current = 8917 → increase = current - original → increase = 8317 := by
sorry

end NUMINAMATH_CALUDE_bacteria_increase_l3747_374717


namespace NUMINAMATH_CALUDE_reflection_y_axis_transformation_l3747_374727

-- Define points in 2D space
def Point := ℝ × ℝ

-- Define the original points
def C : Point := (-3, 2)
def D : Point := (-4, -2)

-- Define the transformed points
def C' : Point := (3, 2)
def D' : Point := (4, -2)

-- Define reflection in y-axis
def reflect_y_axis (p : Point) : Point := (-(p.1), p.2)

-- Theorem statement
theorem reflection_y_axis_transformation :
  (reflect_y_axis C = C') ∧ (reflect_y_axis D = D') := by
  sorry

end NUMINAMATH_CALUDE_reflection_y_axis_transformation_l3747_374727


namespace NUMINAMATH_CALUDE_total_price_calculation_l3747_374795

def jewelry_original_price : ℝ := 30
def painting_original_price : ℝ := 100
def jewelry_price_increase : ℝ := 10
def painting_price_increase_percentage : ℝ := 0.20
def jewelry_sales_tax : ℝ := 0.06
def painting_sales_tax : ℝ := 0.08
def discount_percentage : ℝ := 0.10
def discount_min_amount : ℝ := 800
def jewelry_quantity : ℕ := 2
def painting_quantity : ℕ := 5

def jewelry_new_price : ℝ := jewelry_original_price + jewelry_price_increase
def painting_new_price : ℝ := painting_original_price * (1 + painting_price_increase_percentage)

def jewelry_price_with_tax : ℝ := jewelry_new_price * (1 + jewelry_sales_tax)
def painting_price_with_tax : ℝ := painting_new_price * (1 + painting_sales_tax)

def total_price : ℝ := jewelry_price_with_tax * jewelry_quantity + painting_price_with_tax * painting_quantity

theorem total_price_calculation :
  total_price = 732.80 ∧ total_price < discount_min_amount :=
sorry

end NUMINAMATH_CALUDE_total_price_calculation_l3747_374795


namespace NUMINAMATH_CALUDE_lice_check_time_l3747_374763

/-- The total number of hours required for lice checks -/
def total_hours (kindergarteners first_graders second_graders third_graders : ℕ) 
  (minutes_per_check : ℕ) : ℚ :=
  (kindergarteners + first_graders + second_graders + third_graders) * minutes_per_check / 60

/-- Theorem stating that the total time for lice checks is 3 hours -/
theorem lice_check_time : 
  total_hours 26 19 20 25 2 = 3 := by sorry

end NUMINAMATH_CALUDE_lice_check_time_l3747_374763


namespace NUMINAMATH_CALUDE_trapezium_other_side_length_l3747_374759

/-- Given a trapezium with the following properties:
  - One parallel side is 20 cm long
  - The distance between parallel sides is 17 cm
  - The area is 323 square centimeters
  Prove that the length of the other parallel side is 18 cm -/
theorem trapezium_other_side_length (a b h : ℝ) (area : ℝ) : 
  a = 20 → h = 17 → area = 323 → area = (a + b) * h / 2 → b = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_other_side_length_l3747_374759


namespace NUMINAMATH_CALUDE_no_real_roots_l3747_374750

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (3 * x + 9) + 8 / Real.sqrt (3 * x + 9) = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3747_374750


namespace NUMINAMATH_CALUDE_oranges_picked_total_l3747_374796

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 122

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 105

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_total :
  total_oranges = 227 := by sorry

end NUMINAMATH_CALUDE_oranges_picked_total_l3747_374796


namespace NUMINAMATH_CALUDE_terminating_decimal_of_fraction_l3747_374716

theorem terminating_decimal_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 65 / 1000 →
  ∃ (a b : ℕ), (n : ℚ) / d = (a : ℚ) / (10 ^ b) ∧ (a : ℚ) / (10 ^ b) = 0.065 :=
by sorry

end NUMINAMATH_CALUDE_terminating_decimal_of_fraction_l3747_374716


namespace NUMINAMATH_CALUDE_speed_increase_time_reduction_l3747_374769

theorem speed_increase_time_reduction 
  (initial_speed : ℝ) 
  (speed_increase : ℝ) 
  (distance : ℝ) 
  (h1 : initial_speed = 30)
  (h2 : speed_increase = 10)
  (h3 : distance > 0) :
  let final_speed := initial_speed + speed_increase
  let initial_time := distance / initial_speed
  let final_time := distance / final_speed
  final_time / initial_time = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_speed_increase_time_reduction_l3747_374769


namespace NUMINAMATH_CALUDE_least_multiple_72_112_l3747_374781

theorem least_multiple_72_112 : 
  (∀ k : ℕ, k > 0 ∧ k < 14 → ¬(112 ∣ 72 * k)) ∧ (112 ∣ 72 * 14) :=
sorry

end NUMINAMATH_CALUDE_least_multiple_72_112_l3747_374781


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l3747_374737

/-- Represents the structure of the cube after modifications -/
structure ModifiedCube where
  initial_size : Nat
  small_cube_size : Nat
  removed_center_cubes : Nat
  removed_per_small_cube : Nat

/-- Calculates the surface area of the modified cube structure -/
def surface_area (c : ModifiedCube) : Nat :=
  let remaining_small_cubes := c.initial_size^3 / c.small_cube_size^3 - c.removed_center_cubes
  let surface_per_small_cube := 6 * c.small_cube_size^2 + 12 -- Original surface + newly exposed
  remaining_small_cubes * surface_per_small_cube

/-- Theorem stating the surface area of the specific modified cube -/
theorem modified_cube_surface_area :
  let c : ModifiedCube := {
    initial_size := 12,
    small_cube_size := 3,
    removed_center_cubes := 7,
    removed_per_small_cube := 9
  }
  surface_area c = 3762 := by sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l3747_374737


namespace NUMINAMATH_CALUDE_stating_sock_drawing_probability_l3747_374736

/-- Represents the total number of socks -/
def total_socks : ℕ := 10

/-- Represents the number of colors -/
def num_colors : ℕ := 5

/-- Represents the number of socks per color -/
def socks_per_color : ℕ := 2

/-- Represents the number of socks drawn -/
def socks_drawn : ℕ := 5

/-- 
Theorem stating the probability of drawing 5 socks with exactly one pair 
of the same color and the rest different colors, given 10 socks with 
2 socks each of 5 colors.
-/
theorem sock_drawing_probability : 
  (total_socks = 10) → 
  (num_colors = 5) → 
  (socks_per_color = 2) → 
  (socks_drawn = 5) →
  (Prob_exactly_one_pair_rest_different : ℚ) →
  Prob_exactly_one_pair_rest_different = 10 / 63 := by
  sorry

end NUMINAMATH_CALUDE_stating_sock_drawing_probability_l3747_374736


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_l3747_374774

theorem similar_triangles_perimeter (h_small h_large p_small : ℝ) : 
  h_small / h_large = 3 / 5 →
  p_small = 12 →
  ∃ p_large : ℝ, p_large = 20 ∧ p_small / p_large = h_small / h_large :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_l3747_374774


namespace NUMINAMATH_CALUDE_cone_volume_from_slant_and_height_l3747_374755

/-- The volume of a cone given its slant height and height --/
theorem cone_volume_from_slant_and_height 
  (slant_height : ℝ) 
  (height : ℝ) 
  (h_slant : slant_height = 15) 
  (h_height : height = 9) : 
  (1/3 : ℝ) * Real.pi * (slant_height^2 - height^2) * height = 432 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_slant_and_height_l3747_374755


namespace NUMINAMATH_CALUDE_registration_scientific_notation_equality_l3747_374728

/-- The number of people registered for the national college entrance examination in 2023 -/
def registration_number : ℕ := 12910000

/-- The scientific notation representation of the registration number -/
def scientific_notation : ℝ := 1.291 * (10 ^ 7)

/-- Theorem stating that the registration number is equal to its scientific notation representation -/
theorem registration_scientific_notation_equality :
  (registration_number : ℝ) = scientific_notation :=
sorry

end NUMINAMATH_CALUDE_registration_scientific_notation_equality_l3747_374728


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3747_374709

theorem no_integer_solutions : ¬∃ (x y z : ℤ),
  (x^2 - 3*x*y + 2*y^2 - z^2 = 27) ∧
  (-x^2 + 6*y*z + 2*z^2 = 52) ∧
  (x^2 + x*y + 8*z^2 = 110) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3747_374709


namespace NUMINAMATH_CALUDE_smallest_product_is_623_l3747_374767

def Digits : Finset Nat := {7, 8, 9, 0}

def valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def two_digit_number (tens ones : Nat) : Nat :=
  10 * tens + ones

theorem smallest_product_is_623 :
  ∀ a b c d : Nat,
    valid_arrangement a b c d →
    (two_digit_number a b) * (two_digit_number c d) ≥ 623 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_is_623_l3747_374767


namespace NUMINAMATH_CALUDE_sqrt_sum_quotient_l3747_374718

theorem sqrt_sum_quotient : 
  (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 1.44) / (Real.sqrt 0.49) = 185/63 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_quotient_l3747_374718


namespace NUMINAMATH_CALUDE_conference_schedule_ways_l3747_374707

/-- Represents the number of lecturers --/
def n : ℕ := 7

/-- Represents the number of lecturers with specific ordering constraints --/
def k : ℕ := 3

/-- Calculates the number of ways to schedule n lecturers with k lecturers having specific ordering constraints --/
def schedule_ways (n : ℕ) (k : ℕ) : ℕ :=
  (n - k + 1) * Nat.factorial (n - k)

/-- Theorem stating that the number of ways to schedule 7 lecturers with 3 having specific ordering constraints is 600 --/
theorem conference_schedule_ways : schedule_ways n k = 600 := by
  sorry

end NUMINAMATH_CALUDE_conference_schedule_ways_l3747_374707


namespace NUMINAMATH_CALUDE_simplified_expression_terms_l3747_374765

/-- The number of terms in the simplified form of (x+y+z)^2010 + (x-y-z)^2010 -/
def num_terms : ℕ := 1012036

/-- The exponent used in the expression -/
def exponent : ℕ := 2010

theorem simplified_expression_terms :
  num_terms = (exponent / 2 + 1)^2 := by sorry

end NUMINAMATH_CALUDE_simplified_expression_terms_l3747_374765


namespace NUMINAMATH_CALUDE_probability_is_one_sixth_l3747_374772

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light
    during a randomly chosen five-second interval -/
def probabilityOfColorChange (cycle : TrafficLightCycle) : ℚ :=
  let totalCycleDuration := cycle.green + cycle.yellow + cycle.red
  let favorableDuration := 15 -- 5 seconds before each color change
  favorableDuration / totalCycleDuration

/-- The specific traffic light cycle from the problem -/
def problemCycle : TrafficLightCycle :=
  { green := 45
  , yellow := 5
  , red := 40 }

theorem probability_is_one_sixth :
  probabilityOfColorChange problemCycle = 1 / 6 := by
  sorry

#eval probabilityOfColorChange problemCycle

end NUMINAMATH_CALUDE_probability_is_one_sixth_l3747_374772


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_perimeter_bound_l3747_374779

/-- A rectangle in 2D space -/
structure Rectangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  d : ℝ × ℝ
  is_rectangle : sorry

/-- A quadrilateral inscribed in a rectangle -/
structure InscribedQuadrilateral (rect : Rectangle) where
  k : ℝ × ℝ
  l : ℝ × ℝ
  m : ℝ × ℝ
  n : ℝ × ℝ
  on_sides : sorry

/-- Calculate the perimeter of a quadrilateral -/
def perimeter (q : InscribedQuadrilateral rect) : ℝ := sorry

/-- Calculate the length of the diagonal of a rectangle -/
def diagonal_length (rect : Rectangle) : ℝ := sorry

/-- Theorem: The perimeter of an inscribed quadrilateral is at least twice the diagonal of the rectangle -/
theorem inscribed_quadrilateral_perimeter_bound (rect : Rectangle) (q : InscribedQuadrilateral rect) :
  perimeter q ≥ 2 * diagonal_length rect := by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_perimeter_bound_l3747_374779


namespace NUMINAMATH_CALUDE_large_triangle_altitude_proof_l3747_374764

/-- The altitude of a triangle with area 1600 square feet, composed of two identical smaller triangles each with base 40 feet -/
def largeTriangleAltitude : ℝ := 40

theorem large_triangle_altitude_proof (largeArea smallBase : ℝ) 
  (h1 : largeArea = 1600)
  (h2 : smallBase = 40)
  (h3 : largeArea = 2 * (1/2 * smallBase * largeTriangleAltitude)) :
  largeTriangleAltitude = 40 := by
  sorry

#check large_triangle_altitude_proof

end NUMINAMATH_CALUDE_large_triangle_altitude_proof_l3747_374764


namespace NUMINAMATH_CALUDE_smallest_nut_count_l3747_374742

def nut_division (N : ℕ) (i : ℕ) : ℕ :=
  match i with
  | 0 => N
  | i + 1 => (nut_division N i - 1) / 5

theorem smallest_nut_count :
  ∀ N : ℕ, (∀ i : ℕ, i ≤ 5 → nut_division N i % 5 = 1) ↔ N ≥ 15621 :=
sorry

end NUMINAMATH_CALUDE_smallest_nut_count_l3747_374742


namespace NUMINAMATH_CALUDE_cone_altitude_to_radius_ratio_l3747_374731

/-- The ratio of a cone's altitude to its base radius, given that its volume is one-third of a sphere with the same radius -/
theorem cone_altitude_to_radius_ratio (r h : ℝ) (h_pos : 0 < r) : 
  (1 / 3 * π * r^2 * h = 1 / 3 * (4 / 3 * π * r^3)) → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_altitude_to_radius_ratio_l3747_374731


namespace NUMINAMATH_CALUDE_subset_intersection_union_equivalence_l3747_374704

theorem subset_intersection_union_equivalence (A B C : Set α) :
  (B ⊆ A ∧ C ⊆ A) ↔ ((A ∩ B) ∪ (A ∩ C) = B ∪ C) := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_union_equivalence_l3747_374704


namespace NUMINAMATH_CALUDE_thumbtack_count_l3747_374787

theorem thumbtack_count (num_cans : ℕ) (boards_tested : ℕ) (tacks_per_board : ℕ) (remaining_tacks : ℕ) : 
  num_cans = 3 →
  boards_tested = 120 →
  tacks_per_board = 1 →
  remaining_tacks = 30 →
  (num_cans * (boards_tested * tacks_per_board + remaining_tacks) = 450) :=
by sorry

end NUMINAMATH_CALUDE_thumbtack_count_l3747_374787


namespace NUMINAMATH_CALUDE_correct_price_reduction_equation_l3747_374738

/-- Represents the price reduction scenario for a mobile phone -/
def price_reduction (original_price final_price x : ℝ) : Prop :=
  original_price * (1 - x)^2 = final_price

/-- Theorem stating the correct equation for the given price reduction scenario -/
theorem correct_price_reduction_equation :
  ∃ x : ℝ, price_reduction 1185 580 x :=
sorry

end NUMINAMATH_CALUDE_correct_price_reduction_equation_l3747_374738


namespace NUMINAMATH_CALUDE_existence_of_point_with_specific_distance_l3747_374724

theorem existence_of_point_with_specific_distance : ∃ (x y : ℤ),
  (x : ℝ)^2 + (y : ℝ)^2 = 2 * 2017^2 + 2 * 2018^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_point_with_specific_distance_l3747_374724


namespace NUMINAMATH_CALUDE_laurent_series_expansion_l3747_374753

open Complex

/-- The Laurent series expansion of f(z) = (z+2)/(z^2+4z+3) in the ring 2 < |z+1| < +∞ --/
theorem laurent_series_expansion (z : ℂ) (h : 2 < abs (z + 1)) :
  (z + 2) / (z^2 + 4*z + 3) = ∑' k, ((-2)^k + 1) / (z + 1)^(k + 1) := by sorry

end NUMINAMATH_CALUDE_laurent_series_expansion_l3747_374753


namespace NUMINAMATH_CALUDE_runner_stops_on_start_quarter_l3747_374710

/-- Represents the quarters of the circular track -/
inductive Quarter : Type
  | X : Quarter
  | Y : Quarter
  | Z : Quarter
  | W : Quarter

/-- The circular track -/
structure Track :=
  (circumference : ℝ)
  (quarters : Fin 4 → Quarter)

/-- Represents a runner on the track -/
structure Runner :=
  (start_quarter : Quarter)
  (distance_run : ℝ)

/-- Function to determine the quarter where a runner stops -/
def stop_quarter (track : Track) (runner : Runner) : Quarter :=
  runner.start_quarter

/-- Theorem stating that a runner stops on the same quarter they started on
    when running a multiple of the track's circumference -/
theorem runner_stops_on_start_quarter 
  (track : Track) 
  (runner : Runner) 
  (h1 : track.circumference = 200)
  (h2 : runner.distance_run = 3000) :
  stop_quarter track runner = runner.start_quarter :=
sorry

end NUMINAMATH_CALUDE_runner_stops_on_start_quarter_l3747_374710


namespace NUMINAMATH_CALUDE_exists_ratio_preserving_quadrilateral_l3747_374702

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  sides_positive : ∀ i, sides i > 0
  angles_positive : ∀ i, angles i > 0
  sides_convex : ∀ i, sides i < sides ((i + 1) % 4) + sides ((i + 2) % 4) + sides ((i + 3) % 4)
  angles_convex : ∀ i, angles i < angles ((i + 1) % 4) + angles ((i + 2) % 4) + angles ((i + 3) % 4)
  angle_sum : angles 0 + angles 1 + angles 2 + angles 3 = 2 * Real.pi

/-- The existence of a quadrilateral with side-angle ratio preservation -/
theorem exists_ratio_preserving_quadrilateral (q : ConvexQuadrilateral) :
  ∃ q' : ConvexQuadrilateral,
    ∀ i : Fin 4, (q'.sides i) / (q'.sides ((i + 1) % 4)) = (q.angles i) / (q.angles ((i + 1) % 4)) :=
sorry

end NUMINAMATH_CALUDE_exists_ratio_preserving_quadrilateral_l3747_374702


namespace NUMINAMATH_CALUDE_michaels_ride_l3747_374766

/-- Calculates the total distance traveled by a cyclist given their speed and time -/
def total_distance (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- Represents Michael's cycling scenario -/
theorem michaels_ride (total_time : ℚ) (speed : ℚ) 
    (h1 : total_time = 40) 
    (h2 : speed = 2 / 5) : 
  total_distance speed total_time = 16 := by
  sorry

#eval total_distance (2/5) 40

end NUMINAMATH_CALUDE_michaels_ride_l3747_374766


namespace NUMINAMATH_CALUDE_retail_prices_correct_l3747_374725

def calculate_retail_price (wholesale_price : ℚ) : ℚ :=
  let tax_rate : ℚ := 5 / 100
  let shipping_fee : ℚ := 10
  let profit_margin_rate : ℚ := 20 / 100
  let total_cost : ℚ := wholesale_price + (wholesale_price * tax_rate) + shipping_fee
  let profit_margin : ℚ := wholesale_price * profit_margin_rate
  total_cost + profit_margin

theorem retail_prices_correct :
  let machine1_wholesale : ℚ := 99
  let machine2_wholesale : ℚ := 150
  let machine3_wholesale : ℚ := 210
  (calculate_retail_price machine1_wholesale = 133.75) ∧
  (calculate_retail_price machine2_wholesale = 197.50) ∧
  (calculate_retail_price machine3_wholesale = 272.50) := by
  sorry

end NUMINAMATH_CALUDE_retail_prices_correct_l3747_374725


namespace NUMINAMATH_CALUDE_volume_theorem_l3747_374745

noncomputable def volume_of_body : ℝ :=
  let surface1 (x y z : ℝ) := 2 * z = x^2 + y^2
  let surface2 (z : ℝ) := z = 2
  let surface3 (x : ℝ) := x = 0
  let surface4 (x y : ℝ) := y = 2 * x
  let arctan2 := Real.arctan 2
  2 * arctan2

theorem volume_theorem :
  volume_of_body = 1.704 * Real.pi := by sorry

end NUMINAMATH_CALUDE_volume_theorem_l3747_374745


namespace NUMINAMATH_CALUDE_nancy_crayons_l3747_374722

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 15

/-- The number of packs Nancy bought -/
def packs_bought : ℕ := 41

/-- The total number of crayons Nancy bought -/
def total_crayons : ℕ := crayons_per_pack * packs_bought

theorem nancy_crayons : total_crayons = 615 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crayons_l3747_374722


namespace NUMINAMATH_CALUDE_albert_cabbage_count_l3747_374799

-- Define the number of rows in Albert's cabbage patch
def num_rows : ℕ := 12

-- Define the number of cabbage heads in each row
def heads_per_row : ℕ := 15

-- Define the total number of cabbage heads
def total_heads : ℕ := num_rows * heads_per_row

-- Theorem statement
theorem albert_cabbage_count : total_heads = 180 := by
  sorry

end NUMINAMATH_CALUDE_albert_cabbage_count_l3747_374799


namespace NUMINAMATH_CALUDE_plane_distance_l3747_374705

/-- Given a plane flying east at 300 km/h and west at 400 km/h for a total of 7 hours,
    the distance traveled from the airport is 1200 km. -/
theorem plane_distance (speed_east speed_west total_time : ℝ) 
    (h1 : speed_east = 300)
    (h2 : speed_west = 400)
    (h3 : total_time = 7) : 
  (total_time * speed_east * speed_west) / (speed_east + speed_west) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_plane_distance_l3747_374705


namespace NUMINAMATH_CALUDE_plane_sphere_ratio_sum_l3747_374700

/-- Given a plane passing through (a,b,c) and intersecting the coordinate axes, 
    prove that the sum of ratios of the fixed point coordinates to the sphere center coordinates is 2. -/
theorem plane_sphere_ratio_sum (a b c d e f p q r : ℝ) 
  (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (hdist : d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0)
  (hplane : a / d + b / e + c / f = 1)
  (hsphere : p^2 + q^2 + r^2 = (p - d)^2 + q^2 + r^2 ∧ 
             p^2 + q^2 + r^2 = p^2 + (q - e)^2 + r^2 ∧ 
             p^2 + q^2 + r^2 = p^2 + q^2 + (r - f)^2) :
  a / p + b / q + c / r = 2 := by
sorry

end NUMINAMATH_CALUDE_plane_sphere_ratio_sum_l3747_374700


namespace NUMINAMATH_CALUDE_eighth_fib_is_21_l3747_374714

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Theorem: The 8th term of the Fibonacci sequence is 21
theorem eighth_fib_is_21 : fib 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eighth_fib_is_21_l3747_374714


namespace NUMINAMATH_CALUDE_unique_parallel_line_l3747_374756

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (lies_on : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (on_plane : Point → Plane → Prop)
variable (passes_through : Line → Point → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem unique_parallel_line 
  (α β : Plane) (a : Line) (M : Point) :
  parallel α β → 
  lies_on a α → 
  on_plane M β → 
  ∃! l : Line, passes_through l M ∧ line_parallel l a :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_l3747_374756


namespace NUMINAMATH_CALUDE_andy_remaining_demerits_l3747_374729

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def max_demerits : ℕ := 50

/-- The number of demerits Andy gets per instance of being late -/
def demerits_per_late : ℕ := 2

/-- The number of times Andy was late -/
def times_late : ℕ := 6

/-- The number of demerits Andy got for making an inappropriate joke -/
def demerits_for_joke : ℕ := 15

/-- The number of additional demerits Andy can get before being fired -/
def remaining_demerits : ℕ := max_demerits - (demerits_per_late * times_late + demerits_for_joke)

theorem andy_remaining_demerits : remaining_demerits = 23 := by
  sorry

end NUMINAMATH_CALUDE_andy_remaining_demerits_l3747_374729


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3747_374747

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 2^5 * 3^2 * 7) :
  (∃ m : ℕ, n^m ∣ n! ∧ ∀ k > m, ¬(n^k ∣ n!)) →
  (∃ m : ℕ, n^m ∣ n! ∧ ∀ k > m, ¬(n^k ∣ n!) ∧ m = 334) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3747_374747


namespace NUMINAMATH_CALUDE_valid_placement_iff_even_l3747_374751

/-- Represents a chessboard with one corner cut off -/
structure Chessboard (n : ℕ) :=
  (size : ℕ := 2*n + 1)
  (corner_cut : Bool := true)

/-- Represents a domino placement on the chessboard -/
structure DominoPlacement (n : ℕ) :=
  (board : Chessboard n)
  (total_dominos : ℕ)
  (horizontal_dominos : ℕ)

/-- Checks if a domino placement is valid -/
def is_valid_placement (n : ℕ) (placement : DominoPlacement n) : Prop :=
  placement.total_dominos * 2 = placement.board.size^2 - 1 ∧
  placement.horizontal_dominos * 2 = placement.total_dominos

/-- The main theorem stating the condition for valid placement -/
theorem valid_placement_iff_even (n : ℕ) :
  (∃ (placement : DominoPlacement n), is_valid_placement n placement) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_valid_placement_iff_even_l3747_374751


namespace NUMINAMATH_CALUDE_partition_ratio_theorem_l3747_374793

theorem partition_ratio_theorem (n : ℕ) : 
  (∃ (A B : Finset ℕ), 
    (A ∪ B = Finset.range (n^2 + 1) \ {0}) ∧ 
    (A ∩ B = ∅) ∧
    (A.card = B.card) ∧
    ((A.sum id) / (B.sum id) = 39 / 64)) ↔ 
  (∃ k : ℕ, n = 206 * k) ∧ 
  Even n :=
sorry

end NUMINAMATH_CALUDE_partition_ratio_theorem_l3747_374793


namespace NUMINAMATH_CALUDE_darcies_age_l3747_374735

/-- Darcie's age problem -/
theorem darcies_age :
  ∀ (darcie_age mother_age father_age : ℚ),
    darcie_age = (1 / 6 : ℚ) * mother_age →
    mother_age = (4 / 5 : ℚ) * father_age →
    father_age = 30 →
    darcie_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_darcies_age_l3747_374735


namespace NUMINAMATH_CALUDE_apple_pie_count_l3747_374771

/-- The number of halves in an apple pie -/
def halves_per_pie : ℕ := 2

/-- The number of bite-size samples in half an apple pie -/
def samples_per_half : ℕ := 5

/-- The number of people who can taste Sedrach's apple pies -/
def people_tasting : ℕ := 130

/-- The number of apple pies Sedrach has -/
def sedrachs_pies : ℕ := 13

theorem apple_pie_count :
  sedrachs_pies * halves_per_pie * samples_per_half = people_tasting := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_count_l3747_374771


namespace NUMINAMATH_CALUDE_pears_left_l3747_374794

def initial_pears : ℕ := 35
def given_pears : ℕ := 28

theorem pears_left : initial_pears - given_pears = 7 := by
  sorry

end NUMINAMATH_CALUDE_pears_left_l3747_374794


namespace NUMINAMATH_CALUDE_drum_capacity_ratio_l3747_374798

theorem drum_capacity_ratio (c_x c_y : ℝ) : 
  c_x > 0 → c_y > 0 →
  (1/2 * c_x + 1/2 * c_y = 3/4 * c_y) →
  c_y / c_x = 2 := by
sorry

end NUMINAMATH_CALUDE_drum_capacity_ratio_l3747_374798


namespace NUMINAMATH_CALUDE_custom_op_example_l3747_374773

-- Define the custom operation ⊗
def custom_op (a b : ℤ) : ℤ := 2 * a - b

-- Theorem statement
theorem custom_op_example : custom_op 5 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l3747_374773


namespace NUMINAMATH_CALUDE_three_mn_odd_l3747_374785

theorem three_mn_odd (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  (3 * m * n) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_mn_odd_l3747_374785


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3747_374782

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3747_374782


namespace NUMINAMATH_CALUDE_triangle_property_l3747_374784

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Triangle angle condition
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a / Real.sin A = b / Real.sin B →  -- Law of sines (partial)
  a / Real.sin A = c / Real.sin C →  -- Law of sines (partial)
  (2 * c + b) * Real.cos A + a * Real.cos B = 0 →  -- Given equation
  a = Real.sqrt 3 →  -- Given side length
  A = 2 * π / 3 ∧ Real.sqrt 3 < 2 * b + c ∧ 2 * b + c < 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_property_l3747_374784


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_l3747_374726

theorem mean_equality_implies_x_value : 
  let mean1 := (8 + 15 + 21) / 3
  let mean2 := (18 + x) / 2
  mean1 = mean2 → x = 34 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_l3747_374726


namespace NUMINAMATH_CALUDE_routes_equal_choose_l3747_374792

/-- The number of routes in a 3x2 grid from top-left to bottom-right -/
def num_routes : ℕ := 10

/-- The number of ways to choose 2 items from a set of 5 items -/
def choose_2_from_5 : ℕ := Nat.choose 5 2

/-- Theorem stating that the number of routes is equal to choosing 2 from 5 -/
theorem routes_equal_choose :
  num_routes = choose_2_from_5 := by sorry

end NUMINAMATH_CALUDE_routes_equal_choose_l3747_374792


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l3747_374780

theorem seventh_root_unity_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l3747_374780


namespace NUMINAMATH_CALUDE_january_bill_is_120_l3747_374789

/-- Represents the oil bill for a month -/
structure OilBill where
  amount : ℚ

/-- Represents the oil bills for three months -/
structure ThreeMonthBills where
  january : OilBill
  february : OilBill
  march : OilBill

/-- The conditions given in the problem -/
def satisfiesConditions (bills : ThreeMonthBills) : Prop :=
  let j := bills.january.amount
  let f := bills.february.amount
  let m := bills.march.amount
  f / j = 3 / 2 ∧
  f / m = 4 / 5 ∧
  (f + 20) / j = 5 / 3 ∧
  (f + 20) / m = 2 / 3

/-- The theorem to be proved -/
theorem january_bill_is_120 (bills : ThreeMonthBills) :
  satisfiesConditions bills → bills.january.amount = 120 := by
  sorry

end NUMINAMATH_CALUDE_january_bill_is_120_l3747_374789


namespace NUMINAMATH_CALUDE_train_journey_time_l3747_374790

/-- Proves that if a train moving at 6/7 of its usual speed is 10 minutes late, then its usual journey time is 1 hour -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 → usual_time > 0 →
  (6 / 7 * usual_speed) * (usual_time + 1 / 6) = usual_speed * usual_time →
  usual_time = 1 := by
sorry

end NUMINAMATH_CALUDE_train_journey_time_l3747_374790


namespace NUMINAMATH_CALUDE_candle_equality_l3747_374797

/-- Represents the number of times each candle is used over n Sundays -/
def total_usage (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the number of times each individual candle is used -/
def individual_usage (n : ℕ) : ℚ := (n + 1) / 2

/-- Theorem stating that for all candles to be of equal length after n Sundays,
    n must be a positive odd integer -/
theorem candle_equality (n : ℕ) (h : n > 0) :
  (∀ (i : ℕ), i ≤ n → (individual_usage n).num % (individual_usage n).den = 0) ↔
  n % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_candle_equality_l3747_374797


namespace NUMINAMATH_CALUDE_original_number_is_five_sixths_l3747_374723

theorem original_number_is_five_sixths (x : ℚ) : 
  1 + 1 / x = 11 / 5 → x = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_five_sixths_l3747_374723


namespace NUMINAMATH_CALUDE_cos_B_value_l3747_374733

-- Define the angle B
def B : ℝ := sorry

-- Define the conditions
def B_in_third_quadrant : 3 * π / 2 < B ∧ B < 2 * π := sorry
def sin_B : Real.sin B = -5/13 := sorry

-- Theorem to prove
theorem cos_B_value : Real.cos B = -12/13 := by sorry

end NUMINAMATH_CALUDE_cos_B_value_l3747_374733


namespace NUMINAMATH_CALUDE_largest_product_sum_1976_l3747_374757

theorem largest_product_sum_1976 (n : ℕ) (h : n > 0) :
  (∃ (factors : List ℕ), factors.sum = 1976 ∧ factors.prod = n) →
  n ≤ 2 * 3^658 := by
sorry

end NUMINAMATH_CALUDE_largest_product_sum_1976_l3747_374757


namespace NUMINAMATH_CALUDE_f_greater_than_one_f_monotonicity_f_non_negative_iff_l3747_374775

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (Real.exp x) / (x^2) - k * x + 2 * k * Real.log x

-- State the theorems to be proved
theorem f_greater_than_one (x : ℝ) (hx : x > 0) : f 0 x > 1 := by sorry

theorem f_monotonicity (x : ℝ) (hx : x > 0) :
  (x > 2 → (∀ y > x, f 1 y > f 1 x)) ∧
  (x < 2 → (∀ y ∈ Set.Ioo 0 x, f 1 y > f 1 x)) := by sorry

theorem f_non_negative_iff (k : ℝ) :
  (∀ x > 0, f k x ≥ 0) ↔ k ≤ Real.exp 1 := by sorry

end

end NUMINAMATH_CALUDE_f_greater_than_one_f_monotonicity_f_non_negative_iff_l3747_374775


namespace NUMINAMATH_CALUDE_circle_with_diameter_AB_l3747_374719

-- Define the line segment AB
def line_segment_AB (x y : ℝ) : Prop :=
  x + y - 2 = 0 ∧ 0 ≤ x ∧ x ≤ 2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

-- Theorem statement
theorem circle_with_diameter_AB :
  ∀ x y : ℝ, line_segment_AB x y →
  ∃ center_x center_y radius : ℝ,
    (∀ p q : ℝ, (p - center_x)^2 + (q - center_y)^2 = radius^2 ↔ circle_equation p q) :=
by sorry

end NUMINAMATH_CALUDE_circle_with_diameter_AB_l3747_374719


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3747_374758

def p (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 4

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 4) ∧
  (∀ x : ℝ, (x = -1 ∨ x = 1 ∨ x = 4) → (deriv p) x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3747_374758


namespace NUMINAMATH_CALUDE_y_derivative_l3747_374752

noncomputable def y (x : ℝ) : ℝ := 
  Real.sqrt (1 + 2*x - x^2) * Real.arcsin (x * Real.sqrt 2 / (1 + x)) - Real.sqrt 2 * Real.log (1 + x)

theorem y_derivative (x : ℝ) (h : x ≠ -1) : 
  deriv y x = (1 - x) / Real.sqrt (1 + 2*x - x^2) * Real.arcsin (x * Real.sqrt 2 / (1 + x)) := by
  sorry

end NUMINAMATH_CALUDE_y_derivative_l3747_374752


namespace NUMINAMATH_CALUDE_balloon_fraction_proof_l3747_374720

/-- Proves that the fraction of balloons that blew up in the first half hour is 1/5 --/
theorem balloon_fraction_proof (total : ℕ) (intact : ℕ) (f : ℚ) : 
  total = 200 →
  intact = 80 →
  f * total + 2 * f * total = total - intact →
  f = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_balloon_fraction_proof_l3747_374720


namespace NUMINAMATH_CALUDE_student_allowance_l3747_374754

theorem student_allowance (allowance : ℝ) : 
  (allowance * 2/5 * 2/3 * 3/4 * 9/10 = 1.20) → 
  allowance = 60 := by
sorry

end NUMINAMATH_CALUDE_student_allowance_l3747_374754


namespace NUMINAMATH_CALUDE_product_of_polynomials_l3747_374721

theorem product_of_polynomials (k j : ℝ) :
  (∀ e : ℝ, (8 * e^2 - 4 * e + k) * (4 * e^2 + j * e - 9) = 32 * e^4 - 52 * e^3 + 23 * e^2 + 6 * e - 27) →
  k + j = -7 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l3747_374721


namespace NUMINAMATH_CALUDE_tan_sum_pi_fractions_l3747_374770

theorem tan_sum_pi_fractions : 
  Real.tan (π / 12) + Real.tan (7 * π / 12) = -(4 * (3 - Real.sqrt 3)) / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_fractions_l3747_374770


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3747_374783

theorem quadratic_inequality_range (α : Real) (h : 0 ≤ α ∧ α ≤ π) :
  (∀ x : Real, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) ↔
  (0 ≤ α ∧ α ≤ π / 6) ∨ (5 * π / 6 ≤ α ∧ α ≤ π) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3747_374783


namespace NUMINAMATH_CALUDE_variance_invariant_under_translation_regression_line_minimizes_squared_residuals_sum_residuals_zero_l3747_374749

-- Define a data set as a list of real numbers
def DataSet := List ℝ

-- Define the sample variance
def sampleVariance (data : DataSet) : ℝ := sorry

-- Define a function to subtract a constant from each data point
def subtractConstant (data : DataSet) (c : ℝ) : DataSet := sorry

-- Define a type for a regression line
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

-- Define a function to calculate residuals
def residuals (data : DataSet) (line : RegressionLine) : DataSet := sorry

-- Define a function to calculate the sum of squared residuals
def sumSquaredResiduals (data : DataSet) (line : RegressionLine) : ℝ := sorry

-- Define a function to find the least squares regression line
def leastSquaresRegressionLine (data : DataSet) : RegressionLine := sorry

-- Theorem 1: Subtracting a constant doesn't change the sample variance
theorem variance_invariant_under_translation (data : DataSet) (c : ℝ) :
  sampleVariance (subtractConstant data c) = sampleVariance data := by sorry

-- Theorem 2: The regression line minimizes the sum of squared residuals
theorem regression_line_minimizes_squared_residuals (data : DataSet) :
  ∀ line : RegressionLine,
    sumSquaredResiduals data (leastSquaresRegressionLine data) ≤ sumSquaredResiduals data line := by sorry

-- Theorem 3: The sum of residuals for the least squares regression line is zero
theorem sum_residuals_zero (data : DataSet) :
  (residuals data (leastSquaresRegressionLine data)).sum = 0 := by sorry

end NUMINAMATH_CALUDE_variance_invariant_under_translation_regression_line_minimizes_squared_residuals_sum_residuals_zero_l3747_374749


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3747_374748

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x < -1 → 2 * x^2 + x - 1 > 0) ∧
  (∃ x, 2 * x^2 + x - 1 > 0 ∧ x ≥ -1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3747_374748


namespace NUMINAMATH_CALUDE_sqrt_450_equals_15_sqrt_2_l3747_374762

theorem sqrt_450_equals_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_equals_15_sqrt_2_l3747_374762


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3747_374768

-- Define the triangle ABC
theorem triangle_angle_measure (A B C : Real) (a b c : Real) :
  -- Conditions
  (a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = (Real.sqrt 3 / 2) * b) →
  (c > b) →
  -- Conclusion
  B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3747_374768


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3747_374788

theorem quadratic_equation_solution :
  ∃! x : ℚ, x > 0 ∧ 7 * x^2 + 13 * x - 30 = 0 :=
by
  -- The unique solution is x = 10/7
  use 10/7
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3747_374788


namespace NUMINAMATH_CALUDE_sons_age_l3747_374734

/-- Proves that given the conditions, the son's present age is 22 years -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l3747_374734


namespace NUMINAMATH_CALUDE_triangle_inequality_l3747_374741

/-- Given a triangle ABC with sides a, b, c, heights h_a, h_b, h_c, area Δ, and a positive real number n,
    the inequality (ah_b)^n + (bh_c)^n + (ch_a)^n ≥ 3 * 2^n * Δ^n holds. -/
theorem triangle_inequality (a b c h_a h_b h_c Δ : ℝ) (n : ℝ) 
    (h_pos : n > 0)
    (h_heights : h_a = 2 * Δ / a ∧ h_b = 2 * Δ / b ∧ h_c = 2 * Δ / c)
    (h_area : Δ = a * h_a / 2 ∧ Δ = b * h_b / 2 ∧ Δ = c * h_c / 2) :
  (a * h_b)^n + (b * h_c)^n + (c * h_a)^n ≥ 3 * 2^n * Δ^n := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3747_374741


namespace NUMINAMATH_CALUDE_complex_product_range_l3747_374786

theorem complex_product_range (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ < 1)
  (h₂ : Complex.abs z₂ < 1)
  (h₃ : ∃ (r : ℝ), z₁ + z₂ = r)
  (h₄ : z₁ + z₂ + z₁ * z₂ = 0) :
  ∃ (x : ℝ), z₁ * z₂ = x ∧ -1/2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_range_l3747_374786


namespace NUMINAMATH_CALUDE_manager_salary_calculation_l3747_374743

/-- Calculates the manager's salary given the number of employees, their average salary,
    and the increase in average salary when the manager is included. -/
def manager_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) : ℚ :=
  (avg_salary + avg_increase) * (num_employees + 1) - avg_salary * num_employees

/-- Theorem stating that given 25 employees with an average salary of 2500,
    if adding a manager's salary increases the average by 400,
    then the manager's salary is 12900. -/
theorem manager_salary_calculation :
  manager_salary 25 2500 400 = 12900 := by
  sorry

end NUMINAMATH_CALUDE_manager_salary_calculation_l3747_374743


namespace NUMINAMATH_CALUDE_total_working_days_l3747_374761

/-- Represents the commute options for a worker over a period of working days. -/
structure CommuteData where
  /-- Number of days the worker drove to work in the morning -/
  morning_drives : ℕ
  /-- Number of days the worker took the subway home in the afternoon -/
  afternoon_subways : ℕ
  /-- Total number of subway commutes (morning or afternoon) -/
  total_subway_commutes : ℕ

/-- Theorem stating that given the specific commute data, the total number of working days is 15 -/
theorem total_working_days (data : CommuteData) 
  (h1 : data.morning_drives = 12)
  (h2 : data.afternoon_subways = 20)
  (h3 : data.total_subway_commutes = 15) :
  data.morning_drives + (data.total_subway_commutes - data.morning_drives) = 15 := by
  sorry

#check total_working_days

end NUMINAMATH_CALUDE_total_working_days_l3747_374761


namespace NUMINAMATH_CALUDE_angelinas_speed_to_gym_l3747_374715

-- Define the distances and time difference
def distance_home_to_grocery : ℝ := 1200
def distance_grocery_to_gym : ℝ := 480
def time_difference : ℝ := 40

-- Define the relationship between speeds
def speed_grocery_to_gym (v : ℝ) : ℝ := 2 * v

-- Theorem statement
theorem angelinas_speed_to_gym :
  ∃ v : ℝ, v > 0 ∧
  distance_home_to_grocery / v - distance_grocery_to_gym / (speed_grocery_to_gym v) = time_difference ∧
  speed_grocery_to_gym v = 48 := by
  sorry

end NUMINAMATH_CALUDE_angelinas_speed_to_gym_l3747_374715


namespace NUMINAMATH_CALUDE_inner_square_prob_10x10_l3747_374703

/-- Represents a square checkerboard -/
structure Checkerboard where
  size : ℕ
  total_squares : ℕ
  edge_squares : ℕ
  inner_squares : ℕ

/-- Calculates the probability of choosing an inner square -/
def inner_square_probability (board : Checkerboard) : ℚ :=
  board.inner_squares / board.total_squares

/-- Properties of a 10x10 checkerboard -/
def board_10x10 : Checkerboard :=
  { size := 10
  , total_squares := 100
  , edge_squares := 36
  , inner_squares := 64 }

/-- Theorem: The probability of choosing an inner square on a 10x10 board is 16/25 -/
theorem inner_square_prob_10x10 :
  inner_square_probability board_10x10 = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inner_square_prob_10x10_l3747_374703


namespace NUMINAMATH_CALUDE_parabola_above_line_l3747_374701

theorem parabola_above_line (a : ℝ) : 
  (∀ x ∈ Set.Icc a (a + 1), x^2 - a*x + 3 > 9/4) ↔ a > -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_above_line_l3747_374701


namespace NUMINAMATH_CALUDE_special_triangle_sides_special_triangle_right_l3747_374708

/-- A triangle with sides in arithmetic progression and area 6 -/
structure SpecialTriangle where
  a : ℝ
  area : ℝ
  sides_arithmetic : a > 0 ∧ area = 6 ∧ a * (a + 1) * (a + 2) / 4 = area

/-- The sides of the special triangle are 3, 4, and 5 -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 3 ∧ t.a + 1 = 4 ∧ t.a + 2 = 5 :=
sorry

/-- The special triangle is a right triangle -/
theorem special_triangle_right (t : SpecialTriangle) : 
  t.a ^ 2 + (t.a + 1) ^ 2 = (t.a + 2) ^ 2 :=
sorry

end NUMINAMATH_CALUDE_special_triangle_sides_special_triangle_right_l3747_374708


namespace NUMINAMATH_CALUDE_tan_alpha_max_value_l3747_374778

open Real

theorem tan_alpha_max_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : tan (α + β) = 9 * tan β) : 
  ∃ (max_tan_α : Real), max_tan_α = 4/3 ∧ ∀ (γ : Real), 
    (0 < γ ∧ γ < π/2 ∧ ∃ (δ : Real), (0 < δ ∧ δ < π/2 ∧ tan (γ + δ) = 9 * tan δ)) → 
    tan γ ≤ max_tan_α := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_max_value_l3747_374778


namespace NUMINAMATH_CALUDE_optimal_launch_angle_l3747_374776

/-- 
Given a target at horizontal distance A and height B, 
the angle α that minimizes the initial speed of a projectile to hit the target 
is given by α = arctan((B + √(A² + B²))/A).
-/
theorem optimal_launch_angle (A B : ℝ) (hA : A > 0) (hB : B ≥ 0) :
  let C := Real.sqrt (A^2 + B^2)
  let α := Real.arctan ((B + C) / A)
  ∀ θ : ℝ, 
    0 < θ ∧ θ < π / 2 → 
    (Real.sin θ)^2 * (A^2 + B^2) ≤ (Real.sin (2*α)) * (A^2 + B^2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_optimal_launch_angle_l3747_374776
