import Mathlib

namespace selling_price_example_l1832_183247

/-- Calculates the selling price of an article given the gain and gain percentage. -/
def selling_price (gain : ℚ) (gain_percentage : ℚ) : ℚ :=
  let cost_price := gain / (gain_percentage / 100)
  cost_price + gain

/-- Theorem stating that given a gain of $75 and a gain percentage of 50%, 
    the selling price of an article is $225. -/
theorem selling_price_example : selling_price 75 50 = 225 := by
  sorry

end selling_price_example_l1832_183247


namespace ribbon_gifts_l1832_183204

theorem ribbon_gifts (total_ribbon : ℕ) (ribbon_per_gift : ℕ) (remaining_ribbon : ℕ) : 
  total_ribbon = 18 ∧ ribbon_per_gift = 2 ∧ remaining_ribbon = 6 →
  (total_ribbon - remaining_ribbon) / ribbon_per_gift = 6 :=
by sorry

end ribbon_gifts_l1832_183204


namespace apples_basket_value_l1832_183264

/-- Given a total number of apples, number of baskets, and price per apple,
    calculates the value of apples in one basket. -/
def value_of_basket (total_apples : ℕ) (num_baskets : ℕ) (price_per_apple : ℕ) : ℕ :=
  (total_apples / num_baskets) * price_per_apple

/-- Theorem stating that the value of apples in one basket is 6000 won
    given the specific conditions of the problem. -/
theorem apples_basket_value :
  value_of_basket 180 6 200 = 6000 := by
  sorry

end apples_basket_value_l1832_183264


namespace cricket_game_remaining_overs_l1832_183258

def cricket_game (total_overs : ℕ) (target_runs : ℕ) (initial_overs : ℕ) (initial_run_rate : ℚ) : Prop :=
  let runs_scored := initial_run_rate * initial_overs
  let remaining_runs := target_runs - runs_scored
  let remaining_overs := total_overs - initial_overs
  remaining_overs = 40

theorem cricket_game_remaining_overs :
  cricket_game 50 282 10 (32/10) :=
sorry

end cricket_game_remaining_overs_l1832_183258


namespace edward_initial_amount_l1832_183257

def initial_amount (books_cost pens_cost remaining : ℕ) : ℕ :=
  books_cost + pens_cost + remaining

theorem edward_initial_amount :
  initial_amount 6 16 19 = 41 :=
by sorry

end edward_initial_amount_l1832_183257


namespace number_problem_l1832_183246

theorem number_problem (x : ℚ) : x - (3/5) * x = 62 ↔ x = 155 := by
  sorry

end number_problem_l1832_183246


namespace solution_implies_a_value_l1832_183282

theorem solution_implies_a_value (a : ℝ) : 
  (2 * 1 + 3 * a = -1) → a = -1 := by
  sorry

end solution_implies_a_value_l1832_183282


namespace shelter_cats_l1832_183217

theorem shelter_cats (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 15 / 7 ∧ 
  cats / (dogs + 12) = 15 / 11 → 
  cats = 45 := by
sorry

end shelter_cats_l1832_183217


namespace watermelon_sales_theorem_l1832_183231

/-- Calculates the total income from selling watermelons -/
def watermelon_income (weight : ℕ) (price_per_pound : ℕ) (num_watermelons : ℕ) : ℕ :=
  weight * price_per_pound * num_watermelons

/-- Proves that selling 18 watermelons of 23 pounds each at $2 per pound yields $828 -/
theorem watermelon_sales_theorem :
  watermelon_income 23 2 18 = 828 := by
  sorry

end watermelon_sales_theorem_l1832_183231


namespace no_integer_solution_l1832_183286

theorem no_integer_solution : ¬ ∃ (n : ℤ), (n + 15 > 20) ∧ (-3*n > -9) := by
  sorry

end no_integer_solution_l1832_183286


namespace silver_cube_gold_coating_value_l1832_183249

/-- Calculate the combined value of a silver cube with gold coating and markup -/
theorem silver_cube_gold_coating_value
  (cube_side : ℝ)
  (silver_density : ℝ)
  (silver_price : ℝ)
  (gold_coating_coverage : ℝ)
  (gold_coating_weight : ℝ)
  (gold_price : ℝ)
  (markup : ℝ)
  (h_cube_side : cube_side = 3)
  (h_silver_density : silver_density = 6)
  (h_silver_price : silver_price = 25)
  (h_gold_coating_coverage : gold_coating_coverage = 1/2)
  (h_gold_coating_weight : gold_coating_weight = 0.1)
  (h_gold_price : gold_price = 1800)
  (h_markup : markup = 1.1)
  : ∃ (total_value : ℝ), total_value = 18711 :=
by
  sorry

end silver_cube_gold_coating_value_l1832_183249


namespace y_derivative_l1832_183230

open Real

noncomputable def y (x : ℝ) : ℝ := 
  (sin (tan (1/7)) * (cos (16*x))^2) / (32 * sin (32*x))

theorem y_derivative (x : ℝ) : 
  deriv y x = -sin (tan (1/7)) / (4 * (sin (16*x))^2) :=
sorry

end y_derivative_l1832_183230


namespace point_coordinates_l1832_183200

def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

theorem point_coordinates :
  ∀ (p : ℝ × ℝ),
    second_quadrant p →
    distance_to_x_axis p = 3 →
    distance_to_y_axis p = 1 →
    p = (-1, 3) :=
by
  sorry

end point_coordinates_l1832_183200


namespace teapot_sale_cost_comparison_l1832_183277

/-- Represents the cost calculation for promotional methods in a teapot and teacup sale. -/
structure TeapotSale where
  teapot_price : ℝ
  teacup_price : ℝ
  discount_rate : ℝ
  teapots_bought : ℕ
  min_teacups : ℕ

/-- Calculates the cost under promotional method 1 (buy 1 teapot, get 1 teacup free) -/
def cost_method1 (sale : TeapotSale) (x : ℕ) : ℝ :=
  sale.teapot_price * sale.teapots_bought + sale.teacup_price * (x - sale.teapots_bought)

/-- Calculates the cost under promotional method 2 (9.2% discount on total price) -/
def cost_method2 (sale : TeapotSale) (x : ℕ) : ℝ :=
  (sale.teapot_price * sale.teapots_bought + sale.teacup_price * x) * (1 - sale.discount_rate)

/-- Theorem stating the relationship between costs of two promotional methods -/
theorem teapot_sale_cost_comparison (sale : TeapotSale)
    (h_teapot : sale.teapot_price = 20)
    (h_teacup : sale.teacup_price = 5)
    (h_discount : sale.discount_rate = 0.092)
    (h_teapots : sale.teapots_bought = 4)
    (h_min_teacups : sale.min_teacups = 4) :
    ∀ x : ℕ, x ≥ sale.min_teacups →
      (cost_method1 sale x < cost_method2 sale x ↔ x < 34) ∧
      (cost_method1 sale x = cost_method2 sale x ↔ x = 34) ∧
      (cost_method1 sale x > cost_method2 sale x ↔ x > 34) := by
  sorry


end teapot_sale_cost_comparison_l1832_183277


namespace circle_C_radius_l1832_183255

-- Define the circle C
def Circle_C : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (-1, 1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y + 7 = 0

-- State that A is on the circle
axiom A_on_circle : A ∈ Circle_C

-- State that B is on the circle and the tangent line
axiom B_on_circle : B ∈ Circle_C
axiom B_on_tangent : tangent_line B.1 B.2

-- Define the radius of the circle
def radius (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem circle_C_radius : radius Circle_C = 5 := by sorry

end circle_C_radius_l1832_183255


namespace camp_boys_percentage_l1832_183296

theorem camp_boys_percentage (total : ℕ) (added_boys : ℕ) (girl_percentage : ℚ) : 
  total = 60 →
  added_boys = 60 →
  girl_percentage = 5 / 100 →
  (girl_percentage * (total + added_boys) : ℚ) = (total - (9 * total / 10) : ℚ) →
  (9 * total / 10 : ℚ) / total = 9 / 10 :=
by sorry

end camp_boys_percentage_l1832_183296


namespace seedlings_per_packet_l1832_183259

theorem seedlings_per_packet (total_seedlings : ℕ) (num_packets : ℕ) 
  (h1 : total_seedlings = 420) (h2 : num_packets = 60) :
  total_seedlings / num_packets = 7 := by
  sorry

end seedlings_per_packet_l1832_183259


namespace monitor_pixels_l1832_183292

/-- Calculates the total number of pixels on a monitor given its dimensions and resolution. -/
def totalPixels (width : ℕ) (height : ℕ) (dotsPerInch : ℕ) : ℕ :=
  (width * dotsPerInch) * (height * dotsPerInch)

/-- Theorem stating that a 21x12 inch monitor with 100 dots per inch has 2,520,000 pixels. -/
theorem monitor_pixels :
  totalPixels 21 12 100 = 2520000 := by
  sorry

end monitor_pixels_l1832_183292


namespace expression_factorization_l1832_183222

theorem expression_factorization (x : ℝ) : 
  (4 * x^3 + 75 * x^2 - 12) - (-5 * x^3 + 3 * x^2 - 12) = 9 * x^2 * (x + 8) := by
  sorry

end expression_factorization_l1832_183222


namespace recurrence_necessary_not_sufficient_l1832_183267

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property that a sequence satisfies a_n = 2a_{n-1} for n ≥ 2 -/
def SatisfiesRecurrence (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

/-- The property that a sequence is geometric with common ratio 2 -/
def IsGeometricSequenceWithRatio2 (a : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a n = r * (2 ^ n)

/-- The main theorem stating that SatisfiesRecurrence is necessary but not sufficient
    for IsGeometricSequenceWithRatio2 -/
theorem recurrence_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometricSequenceWithRatio2 a → SatisfiesRecurrence a) ∧
  (∃ a : Sequence, SatisfiesRecurrence a ∧ ¬IsGeometricSequenceWithRatio2 a) :=
by sorry

end recurrence_necessary_not_sufficient_l1832_183267


namespace largest_coefficient_in_expansion_l1832_183288

theorem largest_coefficient_in_expansion (a : ℝ) : 
  (a - 1)^5 = 32 → 
  ∃ (r : ℕ), r ≤ 5 ∧ 
    ∀ (s : ℕ), s ≤ 5 → 
      |(-1)^r * a^(5-r) * (Nat.choose 5 r)| ≥ |(-1)^s * a^(5-s) * (Nat.choose 5 s)| ∧
      (-1)^r * a^(5-r) * (Nat.choose 5 r) = 270 :=
by sorry

end largest_coefficient_in_expansion_l1832_183288


namespace new_difference_greater_than_original_l1832_183251

theorem new_difference_greater_than_original
  (x y a b : ℝ)
  (h_x_pos : x > 0)
  (h_y_pos : y > 0)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_x_gt_y : x > y)
  (h_a_neq_b : a ≠ b) :
  (x + a) - (y - b) > x - y :=
sorry

end new_difference_greater_than_original_l1832_183251


namespace same_solution_equations_l1832_183281

theorem same_solution_equations (x c : ℝ) : 
  (3 * x + 11 = 5) ∧ (c * x - 14 = -4) → c = -5 := by
  sorry

end same_solution_equations_l1832_183281


namespace ball_drawing_properties_l1832_183260

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : Nat
  white : Nat
  white_gt_3 : white > 3

/-- Represents the possible outcomes when drawing two balls -/
inductive DrawOutcome
  | SameColor
  | DifferentColors

/-- Calculates the probability of an outcome given the bag contents -/
def probability (b : BagContents) (o : DrawOutcome) : Rat :=
  sorry

/-- Calculates the probability of drawing at least one red ball -/
def probabilityAtLeastOneRed (b : BagContents) : Rat :=
  sorry

theorem ball_drawing_properties (n : Nat) (h : n > 3) :
  let b : BagContents := ⟨3, n, h⟩
  -- Events "same color" and "different colors" are mutually exclusive
  (probability b DrawOutcome.SameColor + probability b DrawOutcome.DifferentColors = 1) ∧
  -- When P(SameColor) = P(DifferentColors), P(AtLeastOneRed) = 7/12
  (probability b DrawOutcome.SameColor = probability b DrawOutcome.DifferentColors →
   probabilityAtLeastOneRed b = 7/12) :=
by
  sorry

end ball_drawing_properties_l1832_183260


namespace class_artworks_l1832_183218

/-- Represents the number of artworks created by a class of students -/
def total_artworks (num_students : ℕ) (artworks_group1 : ℕ) (artworks_group2 : ℕ) : ℕ :=
  (num_students / 2) * artworks_group1 + (num_students / 2) * artworks_group2

/-- Theorem stating that a class of 10 students, where half make 3 artworks and half make 4, creates 35 artworks in total -/
theorem class_artworks : total_artworks 10 3 4 = 35 := by
  sorry

end class_artworks_l1832_183218


namespace alternating_sequence_sum_l1832_183212

def alternating_sequence (n : ℕ) : ℤ := 
  if n % 2 = 0 then (n : ℤ) else -((n + 1) : ℤ)

def sequence_sum (n : ℕ) : ℤ := 
  (List.range n).map alternating_sequence |>.sum

theorem alternating_sequence_sum : sequence_sum 50 = 25 := by
  sorry

end alternating_sequence_sum_l1832_183212


namespace november_rainfall_l1832_183201

/-- The total rainfall in November for a northwestern town -/
def total_rainfall (first_half_daily_rainfall : ℝ) (days_in_november : ℕ) : ℝ :=
  let first_half := 15
  let second_half := days_in_november - first_half
  let first_half_total := first_half * first_half_daily_rainfall
  let second_half_total := second_half * (2 * first_half_daily_rainfall)
  first_half_total + second_half_total

/-- Theorem stating the total rainfall in November is 180 inches -/
theorem november_rainfall : total_rainfall 4 30 = 180 := by
  sorry

end november_rainfall_l1832_183201


namespace fraction_of_loss_l1832_183279

/-- Given the selling price and cost price of an item, calculate the fraction of loss. -/
theorem fraction_of_loss (selling_price cost_price : ℚ) 
  (h1 : selling_price = 15)
  (h2 : cost_price = 16) :
  (cost_price - selling_price) / cost_price = 1 / 16 := by
  sorry

end fraction_of_loss_l1832_183279


namespace complex_number_modulus_l1832_183236

theorem complex_number_modulus (b : ℝ) : 
  let z : ℂ := (3 - b * Complex.I) / (2 + Complex.I)
  (z.re = z.im) → Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end complex_number_modulus_l1832_183236


namespace platform_length_l1832_183206

/-- Given a train and platform with specific properties, prove the length of the platform. -/
theorem platform_length 
  (train_length : ℝ) 
  (time_platform : ℝ) 
  (time_pole : ℝ) 
  (h1 : train_length = 300)
  (h2 : time_platform = 51)
  (h3 : time_pole = 18) :
  ∃ (platform_length : ℝ), platform_length = 550 := by
  sorry

end platform_length_l1832_183206


namespace average_income_b_and_c_l1832_183265

/-- Proves that given the conditions, the average monthly income of B and C is 6250 --/
theorem average_income_b_and_c (income_a income_b income_c : ℝ) : 
  (income_a + income_b) / 2 = 5050 →
  (income_a + income_c) / 2 = 5200 →
  income_a = 4000 →
  (income_b + income_c) / 2 = 6250 := by
sorry

end average_income_b_and_c_l1832_183265


namespace regression_equation_change_l1832_183213

theorem regression_equation_change (x y : ℝ) :
  y = 3 - 5 * x →
  (3 - 5 * (x + 1)) = y - 5 := by
sorry

end regression_equation_change_l1832_183213


namespace least_positive_angle_l1832_183219

theorem least_positive_angle (x : Real) (a b : Real) : 
  Real.tan x = a / b →
  Real.tan (2 * x) = b / (a + b) →
  Real.tan (3 * x) = (a - b) / (a + b) →
  ∃ k, k = 13 / 9 ∧ x = Real.arctan k ∧ 
    ∀ y, y > 0 → Real.tan y = a / b → Real.tan (2 * y) = b / (a + b) → 
    Real.tan (3 * y) = (a - b) / (a + b) → y ≥ x :=
by sorry

end least_positive_angle_l1832_183219


namespace linear_function_not_in_quadrant_II_l1832_183202

/-- Represents a linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Checks if a point (x, y) is in Quadrant II -/
def isInQuadrantII (x : ℝ) (y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- Theorem: The linear function y = 3x - 2 does not pass through Quadrant II -/
theorem linear_function_not_in_quadrant_II :
  let f : LinearFunction := { m := 3, b := -2 }
  ∀ x y : ℝ, y = f.m * x + f.b → ¬(isInQuadrantII x y) :=
by
  sorry


end linear_function_not_in_quadrant_II_l1832_183202


namespace limit_at_two_l1832_183256

/-- The limit of (3x^2 - 5x - 2) / (x - 2) as x approaches 2 is 7 -/
theorem limit_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 2 →
    0 < |x - 2| ∧ |x - 2| < δ →
    |((3 * x^2 - 5 * x - 2) / (x - 2)) - 7| < ε :=
by sorry

end limit_at_two_l1832_183256


namespace remaining_cube_volume_l1832_183278

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) (angle : ℝ) :
  cube_side = 5 →
  cylinder_radius = 1 →
  angle = Real.pi / 4 →
  ∃ (remaining_volume : ℝ),
    remaining_volume = cube_side^3 - cylinder_radius^2 * Real.pi * (cube_side * Real.sqrt 2) ∧
    remaining_volume = 125 - 5 * Real.sqrt 2 * Real.pi :=
by sorry

end remaining_cube_volume_l1832_183278


namespace john_school_year_hours_l1832_183228

/-- Calculates the number of hours John needs to work per week during the school year -/
def school_year_hours (summer_hours : ℕ) (summer_weeks : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_earnings : ℕ) : ℕ :=
  let summer_hourly_rate := summer_earnings / (summer_hours * summer_weeks)
  let school_year_weekly_earnings := school_year_earnings / school_year_weeks
  school_year_weekly_earnings / summer_hourly_rate

/-- Theorem stating that John needs to work 8 hours per week during the school year -/
theorem john_school_year_hours : 
  school_year_hours 40 10 4000 50 4000 = 8 := by
  sorry

end john_school_year_hours_l1832_183228


namespace sandwiches_bought_l1832_183244

theorem sandwiches_bought (sandwich_cost : ℝ) (soda_count : ℕ) (soda_cost : ℝ) (total_cost : ℝ)
  (h1 : sandwich_cost = 2.44)
  (h2 : soda_count = 4)
  (h3 : soda_cost = 0.87)
  (h4 : total_cost = 8.36)
  : ∃ (sandwich_count : ℕ), 
    sandwich_count * sandwich_cost + soda_count * soda_cost = total_cost ∧ 
    sandwich_count = 2 := by
  sorry

end sandwiches_bought_l1832_183244


namespace dave_files_left_l1832_183295

/-- The number of files Dave has left on his phone -/
def files_left : ℕ := 24

/-- The number of apps Dave has left on his phone -/
def apps_left : ℕ := 2

/-- The difference between the number of files and apps left -/
def file_app_difference : ℕ := 22

theorem dave_files_left :
  files_left = apps_left + file_app_difference :=
by sorry

end dave_files_left_l1832_183295


namespace two_numbers_equation_l1832_183250

theorem two_numbers_equation (α β : ℝ) : 
  (α + β) / 2 = 8 → 
  Real.sqrt (α * β) = 15 → 
  ∃ (x : ℝ), x^2 - 16*x + 225 = 0 ∧ (x = α ∨ x = β) := by
  sorry

end two_numbers_equation_l1832_183250


namespace arithmetic_mean_problem_l1832_183220

theorem arithmetic_mean_problem (original_list : List ℝ) (x y z : ℝ) :
  original_list.length = 15 →
  original_list.sum / original_list.length = 70 →
  (original_list.sum + x + y + z) / (original_list.length + 3) = 80 →
  (x + y + z) / 3 = 130 := by
  sorry

end arithmetic_mean_problem_l1832_183220


namespace angle_inequality_l1832_183274

theorem angle_inequality (x y : Real) (h1 : x ≤ 90 * Real.pi / 180) (h2 : Real.sin y = 3/4 * Real.sin x) : y > x/2 := by
  sorry

end angle_inequality_l1832_183274


namespace min_a_for_simplest_quadratic_root_l1832_183239

-- Define the property of being the simplest quadratic root
def is_simplest_quadratic_root (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ ∀ (m : ℕ), m < n → ¬(∃ (q : ℚ), q * q = m)

-- Define the main theorem
theorem min_a_for_simplest_quadratic_root :
  ∃ (a : ℤ), (∀ (b : ℤ), is_simplest_quadratic_root (Real.sqrt (3 * b + 1)) → a ≤ b) ∧
             is_simplest_quadratic_root (Real.sqrt (3 * a + 1)) ∧
             a = 2 :=
sorry

end min_a_for_simplest_quadratic_root_l1832_183239


namespace omega_range_l1832_183262

theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∃ a b : ℝ, π ≤ a ∧ a < b ∧ b ≤ 2*π ∧ Real.sin (ω*a) + Real.sin (ω*b) = 2) →
  (9/4 ≤ ω ∧ ω < 5/2) ∨ (13/4 ≤ ω) :=
by sorry

end omega_range_l1832_183262


namespace same_start_end_words_count_l1832_183242

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The number of freely chosen letters in each word -/
def free_choices : ℕ := word_length - 2

/-- The number of five-letter words that begin and end with the same letter -/
def same_start_end_words : ℕ := alphabet_size ^ free_choices

theorem same_start_end_words_count :
  same_start_end_words = 456976 :=
sorry

end same_start_end_words_count_l1832_183242


namespace min_omega_for_max_values_l1832_183287

theorem min_omega_for_max_values (ω : ℝ) (h1 : ω > 0) :
  (∀ x ∈ Set.Icc 0 1, ∃ (n : ℕ), n ≥ 50 ∧ 
    (∀ y ∈ Set.Icc 0 1, Real.sin (ω * x) ≥ Real.sin (ω * y))) →
  ω ≥ 197 * Real.pi / 2 :=
sorry

end min_omega_for_max_values_l1832_183287


namespace greatest_three_digit_multiple_of_17_l1832_183238

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n → n ≤ 986 :=
by sorry

end greatest_three_digit_multiple_of_17_l1832_183238


namespace yogurt_combinations_l1832_183271

theorem yogurt_combinations (num_flavors : Nat) (num_toppings : Nat) (num_sizes : Nat) :
  num_flavors = 6 →
  num_toppings = 8 →
  num_sizes = 2 →
  num_flavors * (num_toppings.choose 2) * num_sizes = 336 := by
  sorry

#eval Nat.choose 8 2

end yogurt_combinations_l1832_183271


namespace ed_lost_marbles_ed_lost_eleven_marbles_l1832_183253

theorem ed_lost_marbles (doug : ℕ) : ℕ :=
  let ed_initial := doug + 19
  let ed_final := doug + 8
  ed_initial - ed_final

theorem ed_lost_eleven_marbles (doug : ℕ) :
  ed_lost_marbles doug = 11 := by
  sorry

end ed_lost_marbles_ed_lost_eleven_marbles_l1832_183253


namespace prime_iff_no_equal_products_l1832_183203

theorem prime_iff_no_equal_products (p : ℕ) (h : p > 1) :
  Nat.Prime p ↔ 
  ∀ (a b c d : ℕ), 
    a > 0 → b > 0 → c > 0 → d > 0 → 
    a + b + c + d = p → 
    (a * b ≠ c * d ∧ a * c ≠ b * d ∧ a * d ≠ b * c) :=
by sorry

end prime_iff_no_equal_products_l1832_183203


namespace angle_symmetry_l1832_183223

/-- Given that the terminal side of angle α is symmetric to the terminal side of angle -690° about the y-axis, prove that α = k * 360° + 150° for some integer k. -/
theorem angle_symmetry (α : Real) : 
  (∃ k : ℤ, α = k * 360 + 150) ↔ 
  (∃ n : ℤ, α + (-690) = n * 360 + 180) :=
by sorry

end angle_symmetry_l1832_183223


namespace matrix_power_four_l1832_183280

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A^4 = !![0, -9; 9, -9] := by sorry

end matrix_power_four_l1832_183280


namespace prime_factors_of_factorial_30_l1832_183297

theorem prime_factors_of_factorial_30 : 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 
  (Nat.factors 30).toFinset.card := by sorry

end prime_factors_of_factorial_30_l1832_183297


namespace animal_shelter_count_l1832_183275

theorem animal_shelter_count : 645 + 567 + 316 + 120 = 1648 := by
  sorry

end animal_shelter_count_l1832_183275


namespace debt_average_payment_l1832_183299

theorem debt_average_payment 
  (total_installments : ℕ) 
  (first_payment_count : ℕ) 
  (first_payment_amount : ℚ) 
  (payment_increase : ℚ) :
  total_installments = 65 →
  first_payment_count = 20 →
  first_payment_amount = 410 →
  payment_increase = 65 →
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + payment_increase
  let total_amount := first_payment_count * first_payment_amount + 
                      remaining_payment_count * remaining_payment_amount
  total_amount / total_installments = 455 := by
sorry

end debt_average_payment_l1832_183299


namespace sum_of_powers_of_i_l1832_183226

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem sum_of_powers_of_i :
  i^1520 + i^1521 + i^1522 + i^1523 + i^1524 = (2 : ℂ) := by sorry

end sum_of_powers_of_i_l1832_183226


namespace cube_of_negative_two_x_l1832_183285

theorem cube_of_negative_two_x (x : ℝ) : (-2 * x)^3 = -8 * x^3 := by
  sorry

end cube_of_negative_two_x_l1832_183285


namespace roots_and_minimum_value_l1832_183227

def f (a : ℝ) (x : ℝ) : ℝ := |x^2 - x| - a*x

theorem roots_and_minimum_value :
  (∀ x, f (1/3) x = 0 ↔ x = 0 ∨ x = 2/3 ∨ x = 4/3) ∧
  (∀ a, a ≤ -1 →
    (∀ x ∈ Set.Icc (-2) 3, f a x ≥ 
      (if a ≤ -5 then 2*a + 6 else -(a+1)^2/4)) ∧
    (∃ x ∈ Set.Icc (-2) 3, f a x = 
      (if a ≤ -5 then 2*a + 6 else -(a+1)^2/4))) := by
  sorry

end roots_and_minimum_value_l1832_183227


namespace magnified_tissue_diameter_l1832_183241

/-- Given a circular piece of tissue with an actual diameter and a magnification factor,
    calculate the diameter of the magnified image. -/
def magnified_diameter (actual_diameter : ℝ) (magnification_factor : ℝ) : ℝ :=
  actual_diameter * magnification_factor

/-- Theorem: The diameter of a circular piece of tissue with an actual diameter of 0.001 cm,
    when magnified 1,000 times, is 1 cm. -/
theorem magnified_tissue_diameter :
  magnified_diameter 0.001 1000 = 1 := by
  sorry

end magnified_tissue_diameter_l1832_183241


namespace a_greater_equal_two_l1832_183273

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

theorem a_greater_equal_two (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a (-1) ≤ f a x) ∧
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ f a 1) →
  a ≥ 2 := by sorry

end a_greater_equal_two_l1832_183273


namespace sum_of_two_squares_equivalence_l1832_183252

theorem sum_of_two_squares_equivalence (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2*x = u^2 + v^2) :=
by sorry

end sum_of_two_squares_equivalence_l1832_183252


namespace product_of_powers_l1832_183245

theorem product_of_powers (n : ℕ) : (500 ^ 50) * (2 ^ 100) = 10 ^ 75 := by
  sorry

end product_of_powers_l1832_183245


namespace notebook_savings_theorem_l1832_183266

/-- Calculates the savings when buying notebooks on sale compared to regular price -/
def calculate_savings (original_price : ℚ) (regular_quantity : ℕ) (sale_quantity : ℕ) 
  (sale_discount : ℚ) (extra_discount : ℚ) : ℚ :=
  let regular_cost := original_price * regular_quantity
  let discounted_price := original_price * (1 - sale_discount)
  let sale_cost := if sale_quantity > 10
    then discounted_price * sale_quantity * (1 - extra_discount)
    else discounted_price * sale_quantity
  regular_cost - sale_cost

theorem notebook_savings_theorem : 
  let original_price : ℚ := 3
  let regular_quantity : ℕ := 8
  let sale_quantity : ℕ := 12
  let sale_discount : ℚ := 1/4
  let extra_discount : ℚ := 1/20
  calculate_savings original_price regular_quantity sale_quantity sale_discount extra_discount = 10.35 := by
  sorry

end notebook_savings_theorem_l1832_183266


namespace base_10_to_12_conversion_l1832_183225

/-- Represents a digit in base 12 -/
inductive Base12Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

/-- Converts a Base12Digit to its corresponding natural number -/
def Base12Digit.toNat : Base12Digit → Nat
| D0 => 0
| D1 => 1
| D2 => 2
| D3 => 3
| D4 => 4
| D5 => 5
| D6 => 6
| D7 => 7
| D8 => 8
| D9 => 9
| A => 10
| B => 11

/-- Represents a number in base 12 -/
def Base12Number := List Base12Digit

/-- Converts a Base12Number to its corresponding natural number -/
def Base12Number.toNat : Base12Number → Nat
| [] => 0
| d::ds => d.toNat * (12 ^ ds.length) + Base12Number.toNat ds

theorem base_10_to_12_conversion :
  Base12Number.toNat [Base12Digit.B, Base12Digit.D5] = 173 :=
by sorry

end base_10_to_12_conversion_l1832_183225


namespace sum_x_coordinates_q3_l1832_183283

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Creates a new polygon from the midpoints of the sides of a given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Calculates the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

/-- The main theorem -/
theorem sum_x_coordinates_q3 (q1 : Polygon) 
  (h1 : q1.vertices.length = 45)
  (h2 : sumXCoordinates q1 = 135) :
  let q2 := midpointPolygon q1
  let q3 := midpointPolygon q2
  sumXCoordinates q3 = 135 := by
  sorry

end sum_x_coordinates_q3_l1832_183283


namespace inequality_solution_l1832_183229

theorem inequality_solution :
  ∀ x : ℝ, (x / 2 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Ici (-6) ∩ Set.Iio (-3/2) :=
by sorry

end inequality_solution_l1832_183229


namespace constant_sum_l1832_183254

theorem constant_sum (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 1 ↔ x = -1) ∧ (a + b / x = 5 ↔ x = -5)) →
  a + b = 11 := by
sorry

end constant_sum_l1832_183254


namespace otimes_self_otimes_self_l1832_183290

def otimes (x y : ℝ) : ℝ := x^2 - y^2

theorem otimes_self_otimes_self (h : ℝ) : otimes h (otimes h h) = h^2 := by
  sorry

end otimes_self_otimes_self_l1832_183290


namespace problem_solution_l1832_183237

-- Define proposition p
def p : Prop := ∀ a b : ℝ, (a > b ∧ b > 0) → (1/a < 1/b)

-- Define proposition q
def q : Prop := ∀ f : ℝ → ℝ, (∀ x : ℝ, f (x - 1) = f (-(x - 1))) → 
  (∀ x : ℝ, f x = f (2 - x))

-- Theorem to prove
theorem problem_solution : p ∨ ¬q := by sorry

end problem_solution_l1832_183237


namespace parallel_lines_circle_distance_l1832_183214

theorem parallel_lines_circle_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ),
    chord1 = 38 ∧ 
    chord2 = 38 ∧ 
    chord3 = 34 ∧
    chord1 * 38 * chord1 / 4 + (d / 2) * 38 * (d / 2) = chord1 * r^2 ∧
    chord3 * 34 * chord3 / 4 + (3 * d / 2) * 34 * (3 * d / 2) = chord3 * r^2) →
  d = 6 := by
sorry

end parallel_lines_circle_distance_l1832_183214


namespace multiply_mixed_number_l1832_183215

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end multiply_mixed_number_l1832_183215


namespace parallel_tangents_sum_l1832_183270

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1/a) * Real.log x + 1/x - x

theorem parallel_tangents_sum (a : ℝ) (h : a ≥ 3) :
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  (deriv (f a)) x₁ = (deriv (f a)) x₂ ∧
  x₁ + x₂ > 6/5 := by sorry

end parallel_tangents_sum_l1832_183270


namespace greatest_three_digit_multiple_of_17_l1832_183221

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n ≤ 999 ∧ 
  100 ≤ n ∧ 
  n % 17 = 0 ∧ 
  ∀ m : ℕ, m ≤ 999 ∧ 100 ≤ m ∧ m % 17 = 0 → m ≤ n :=
by
  use 986
  sorry

end greatest_three_digit_multiple_of_17_l1832_183221


namespace range_of_m_l1832_183284

def p (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ m - 2 > 0 ∧ 6 - m > 0 ∧
  ∀ x y : ℝ, x^2 / (m - 2) + y^2 / (6 - m) = 1 ↔ (x / a)^2 + (y / b)^2 = 1

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m > 0

theorem range_of_m :
  ∃ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ (1 < m ∧ m ≤ 2) ∨ m ≥ 4 := by sorry

end range_of_m_l1832_183284


namespace find_number_l1832_183209

theorem find_number : ∃ x : ℝ, 0.5 * x = 0.2 * 650 + 190 := by
  sorry

end find_number_l1832_183209


namespace no_rain_probability_l1832_183289

theorem no_rain_probability (p : ℝ) (n : ℕ) (h1 : p = 2/3) (h2 : n = 5) :
  (1 - p)^n = 1/243 := by
  sorry

end no_rain_probability_l1832_183289


namespace range_of_a_l1832_183294

open Real

noncomputable def f (a x : ℝ) : ℝ := exp x * (2 * x - 1) - 2 * a * x + 2 * a

theorem range_of_a (a : ℝ) :
  (a < 1) →
  (∃! (x₀ : ℤ), f a (x₀ : ℝ) < 0) →
  a ∈ Set.Icc (3 / (4 * exp 1)) (1 / 2) :=
sorry

end range_of_a_l1832_183294


namespace sum_inequality_l1832_183211

theorem sum_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  n * (n + 1) / 2 ≠ m * (m + 1) := by
  sorry

end sum_inequality_l1832_183211


namespace dannys_age_l1832_183248

/-- Proves Danny's current age given Jane's age and their age relationship 19 years ago -/
theorem dannys_age (jane_age : ℕ) (h1 : jane_age = 26) 
  (h2 : ∃ (danny_age : ℕ), danny_age - 19 = 3 * (jane_age - 19)) : 
  ∃ (danny_age : ℕ), danny_age = 40 := by
  sorry

end dannys_age_l1832_183248


namespace boys_to_girls_ratio_l1832_183263

/-- Theorem: Ratio of boys to girls in a class --/
theorem boys_to_girls_ratio 
  (boys_avg : ℝ) 
  (girls_avg : ℝ) 
  (class_avg : ℝ) 
  (missing_scores : ℕ) 
  (missing_avg : ℝ) 
  (h1 : boys_avg = 90) 
  (h2 : girls_avg = 96) 
  (h3 : class_avg = 94) 
  (h4 : missing_scores = 3) 
  (h5 : missing_avg = 92) :
  ∃ (boys girls : ℕ), 
    boys > 0 ∧ 
    girls > 0 ∧ 
    (boys : ℝ) / girls = 1 / 5 ∧
    class_avg * (boys + girls + missing_scores : ℝ) = 
      boys_avg * boys + girls_avg * girls + missing_avg * missing_scores :=
by sorry

end boys_to_girls_ratio_l1832_183263


namespace min_value_of_sum_l1832_183269

theorem min_value_of_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  x + 3 * y + 5 * z ≥ 14 * (40 / 3) ^ (1 / 3) :=
by sorry

end min_value_of_sum_l1832_183269


namespace least_number_divisible_by_five_primes_l1832_183216

theorem least_number_divisible_by_five_primes : 
  ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    2310 = p₁ * p₂ * p₃ * p₄ * p₅ ∧
    ∀ (n : ℕ), n > 0 → (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ), 
      Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧
      n % q₁ = 0 ∧ n % q₂ = 0 ∧ n % q₃ = 0 ∧ n % q₄ = 0 ∧ n % q₅ = 0) → 
    n ≥ 2310 := by
  sorry

end least_number_divisible_by_five_primes_l1832_183216


namespace different_types_of_players_l1832_183293

/-- Represents the types of players in the game. -/
inductive PlayerType
  | Cricket
  | Hockey
  | Football
  | Softball

/-- The number of players for each type. -/
def num_players (t : PlayerType) : ℕ :=
  match t with
  | .Cricket => 12
  | .Hockey => 17
  | .Football => 11
  | .Softball => 10

/-- The total number of players on the ground. -/
def total_players : ℕ := 50

/-- The list of all player types. -/
def all_player_types : List PlayerType :=
  [PlayerType.Cricket, PlayerType.Hockey, PlayerType.Football, PlayerType.Softball]

theorem different_types_of_players :
  (List.length all_player_types = 4) ∧
  (List.sum (List.map num_players all_player_types) = total_players) := by
  sorry

end different_types_of_players_l1832_183293


namespace average_difference_l1832_183298

theorem average_difference (a b c : ℝ) 
  (hab : (a + b) / 2 = 80) 
  (hbc : (b + c) / 2 = 180) : 
  a - c = -200 := by sorry

end average_difference_l1832_183298


namespace max_x_minus_y_l1832_183272

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l1832_183272


namespace inequality_proof_l1832_183268

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^2 + y^2 + z^2 + x*y + y*z + z*x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) := by
  sorry

end inequality_proof_l1832_183268


namespace complement_P_in_U_l1832_183207

-- Define the sets U and P
def U : Set ℝ := {y | ∃ x > 1, y = Real.log x / Real.log 2}
def P : Set ℝ := {y | ∃ x > 2, y = 1 / x}

-- State the theorem
theorem complement_P_in_U : 
  (U \ P) = {y | y ∈ Set.Ici (1/2)} := by sorry

end complement_P_in_U_l1832_183207


namespace dilation_image_l1832_183276

def dilation (center : ℂ) (scale : ℝ) (point : ℂ) : ℂ :=
  center + scale * (point - center)

theorem dilation_image : 
  let center : ℂ := -1 + 2*I
  let scale : ℝ := 2
  let point : ℂ := 3 + 4*I
  dilation center scale point = 7 + 6*I := by
  sorry

end dilation_image_l1832_183276


namespace section_B_students_l1832_183240

/-- The number of students in section A -/
def students_A : ℕ := 36

/-- The average weight of students in section A (in kg) -/
def avg_weight_A : ℚ := 40

/-- The average weight of students in section B (in kg) -/
def avg_weight_B : ℚ := 35

/-- The average weight of the whole class (in kg) -/
def avg_weight_total : ℚ := 37.25

/-- The number of students in section B -/
def students_B : ℕ := 44

theorem section_B_students : 
  (students_A : ℚ) * avg_weight_A + (students_B : ℚ) * avg_weight_B = 
  ((students_A : ℚ) + students_B) * avg_weight_total := by
  sorry

end section_B_students_l1832_183240


namespace smallest_prime_perimeter_isosceles_triangle_l1832_183232

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define an isosceles triangle with prime side lengths
def isoscelesTrianglePrime (a b : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ (a + a + b > a) ∧ (a + b > a)

-- Define the perimeter of the triangle
def perimeter (a b : ℕ) : ℕ := a + a + b

-- State the theorem
theorem smallest_prime_perimeter_isosceles_triangle :
  ∀ a b : ℕ, isoscelesTrianglePrime a b → isPrime (perimeter a b) →
  perimeter a b ≥ 11 :=
sorry

end smallest_prime_perimeter_isosceles_triangle_l1832_183232


namespace incorrect_height_proof_l1832_183210

/-- Given a class of boys with an incorrect average height and one boy's height
    recorded incorrectly, prove the value of the incorrectly recorded height. -/
theorem incorrect_height_proof (n : ℕ) (incorrect_avg real_avg actual_height : ℝ) 
    (hn : n = 35)
    (hi : incorrect_avg = 182)
    (hr : real_avg = 180)
    (ha : actual_height = 106) :
  ∃ (incorrect_height : ℝ),
    incorrect_height = 176 ∧
    n * real_avg = (n - 1) * incorrect_avg + actual_height - incorrect_height :=
by
  sorry

end incorrect_height_proof_l1832_183210


namespace paradise_park_capacity_l1832_183208

/-- A Ferris wheel in paradise park -/
structure FerrisWheel where
  total_seats : ℕ
  broken_seats : ℕ
  people_per_seat : ℕ

/-- The capacity of a Ferris wheel is the number of people it can hold on functioning seats -/
def FerrisWheel.capacity (fw : FerrisWheel) : ℕ :=
  (fw.total_seats - fw.broken_seats) * fw.people_per_seat

/-- The paradise park with its three Ferris wheels -/
def paradise_park : List FerrisWheel :=
  [{ total_seats := 18, broken_seats := 10, people_per_seat := 15 },
   { total_seats := 25, broken_seats := 7,  people_per_seat := 15 },
   { total_seats := 30, broken_seats := 12, people_per_seat := 15 }]

/-- The total capacity of all Ferris wheels in paradise park -/
def total_park_capacity : ℕ :=
  (paradise_park.map FerrisWheel.capacity).sum

theorem paradise_park_capacity :
  total_park_capacity = 660 := by
  sorry

end paradise_park_capacity_l1832_183208


namespace inequality_proof_l1832_183243

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2) ∧
  ((a * b * c = 1) → (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ 3 / 2)) ∧
  ((a * b * c = 1) → (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2)) :=
by sorry

end inequality_proof_l1832_183243


namespace rectangle_area_error_percent_l1832_183234

/-- Given a rectangle with actual length L and width W, if the measured length is 1.09L
    and the measured width is 0.92W, then the error percent in the calculated area
    compared to the actual area is 0.28%. -/
theorem rectangle_area_error_percent (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let measured_length := 1.09 * L
  let measured_width := 0.92 * W
  let actual_area := L * W
  let calculated_area := measured_length * measured_width
  let error_percent := (calculated_area - actual_area) / actual_area * 100
  error_percent = 0.28 := by sorry

end rectangle_area_error_percent_l1832_183234


namespace apple_delivery_proof_l1832_183224

/-- Represents the number of apples delivered by the truck -/
def apples_delivered : ℕ → ℕ → ℕ → ℕ
  | initial_green, initial_red, final_green_excess =>
    final_green_excess + initial_red - initial_green

theorem apple_delivery_proof :
  let initial_green := 32
  let initial_red := initial_green + 200
  let final_green_excess := 140
  apples_delivered initial_green initial_red final_green_excess = 340 := by
sorry

#eval apples_delivered 32 232 140

end apple_delivery_proof_l1832_183224


namespace complex_power_sum_l1832_183261

theorem complex_power_sum : 
  let i : ℂ := Complex.I
  ((1 + i) / 2) ^ 8 + ((1 - i) / 2) ^ 8 = (1 : ℂ) / 8 := by
  sorry

end complex_power_sum_l1832_183261


namespace first_digit_after_500_erasure_l1832_183235

/-- Calculates the total number of digits when writing numbers from 1 to n in sequence -/
def totalDigits (n : ℕ) : ℕ := sorry

/-- Finds the first digit after erasing a certain number of digits from the sequence -/
def firstDigitAfterErasure (totalNumbers : ℕ) (erasedDigits : ℕ) : ℕ := sorry

theorem first_digit_after_500_erasure :
  firstDigitAfterErasure 500 500 = 3 := by sorry

end first_digit_after_500_erasure_l1832_183235


namespace skittles_given_to_karen_l1832_183233

/-- The number of Skittles Pamela initially had -/
def initial_skittles : ℕ := 50

/-- The number of Skittles Pamela has now -/
def remaining_skittles : ℕ := 43

/-- The number of Skittles Pamela gave to Karen -/
def skittles_given : ℕ := initial_skittles - remaining_skittles

theorem skittles_given_to_karen : skittles_given = 7 := by
  sorry

end skittles_given_to_karen_l1832_183233


namespace expression_simplification_l1832_183291

theorem expression_simplification (x : ℝ) :
  4 * x^3 + 5 * x + 9 - (3 * x^3 - 2 * x + 1) + 2 * x^2 - (x^2 - 4 * x - 6) =
  x^3 + x^2 + 11 * x + 14 := by
  sorry

end expression_simplification_l1832_183291


namespace travel_ways_theorem_l1832_183205

/-- Represents the number of transportation options between two cities -/
structure TransportOptions where
  buses : Nat
  trains : Nat
  ferries : Nat

/-- The total number of ways to travel between two cities -/
def total_ways (options : TransportOptions) : Nat :=
  options.buses + options.trains + options.ferries

theorem travel_ways_theorem (ab_morning : TransportOptions) (bc_afternoon : TransportOptions)
  (h1 : ab_morning.buses = 5)
  (h2 : ab_morning.trains = 2)
  (h3 : ab_morning.ferries = 0)
  (h4 : bc_afternoon.buses = 3)
  (h5 : bc_afternoon.trains = 0)
  (h6 : bc_afternoon.ferries = 2) :
  (total_ways ab_morning) * (total_ways bc_afternoon) = 35 := by
  sorry

#check travel_ways_theorem

end travel_ways_theorem_l1832_183205
