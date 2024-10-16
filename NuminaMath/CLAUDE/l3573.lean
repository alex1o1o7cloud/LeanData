import Mathlib

namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3573_357334

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex polygon with 9 sides -/
structure Nonagon where
  sides : ℕ
  convex : Bool
  is_nonagon : sides = 9 ∧ convex = true

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals (n : Nonagon) : diagonals_in_nonagon = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3573_357334


namespace NUMINAMATH_CALUDE_no_four_digit_numbers_divisible_by_11_sum_10_l3573_357341

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- Checks if a four-digit number is divisible by 11 -/
def isDivisibleBy11 (n : FourDigitNumber) : Prop :=
  (1000 * n.a + 100 * n.b + 10 * n.c + n.d) % 11 = 0

/-- Checks if the sum of digits of a four-digit number is 10 -/
def sumOfDigitsIs10 (n : FourDigitNumber) : Prop :=
  n.a + n.b + n.c + n.d = 10

/-- Theorem: There are no four-digit numbers divisible by 11 with digits summing to 10 -/
theorem no_four_digit_numbers_divisible_by_11_sum_10 :
  ¬ ∃ (n : FourDigitNumber), isDivisibleBy11 n ∧ sumOfDigitsIs10 n := by
  sorry

end NUMINAMATH_CALUDE_no_four_digit_numbers_divisible_by_11_sum_10_l3573_357341


namespace NUMINAMATH_CALUDE_kays_aerobics_time_l3573_357346

/-- Given Kay's weekly exercise routine, calculate the time spent on aerobics. -/
theorem kays_aerobics_time (total_time : ℕ) (aerobics_ratio : ℕ) (weight_ratio : ℕ) 
  (h1 : total_time = 250)
  (h2 : aerobics_ratio = 3)
  (h3 : weight_ratio = 2) :
  (aerobics_ratio * total_time) / (aerobics_ratio + weight_ratio) = 150 := by
  sorry

#check kays_aerobics_time

end NUMINAMATH_CALUDE_kays_aerobics_time_l3573_357346


namespace NUMINAMATH_CALUDE_p_shape_points_for_10cm_square_l3573_357375

/-- Calculates the number of unique points on a "P" shape formed from a square --/
def count_points_on_p_shape (side_length : ℕ) : ℕ :=
  let points_per_side := side_length + 1
  let total_points := points_per_side * 3
  let corner_points := 2
  total_points - corner_points

/-- Theorem stating that a "P" shape formed from a 10 cm square has 31 unique points --/
theorem p_shape_points_for_10cm_square :
  count_points_on_p_shape 10 = 31 := by
  sorry

#eval count_points_on_p_shape 10

end NUMINAMATH_CALUDE_p_shape_points_for_10cm_square_l3573_357375


namespace NUMINAMATH_CALUDE_nuts_mixed_with_raisins_l3573_357388

/-- The number of pounds of nuts mixed with raisins -/
def pounds_of_nuts : ℝ := 4

/-- The number of pounds of raisins -/
def pounds_of_raisins : ℝ := 5

/-- The cost ratio of nuts to raisins -/
def cost_ratio : ℝ := 3

/-- The fraction of the total cost that the raisins represent -/
def raisin_cost_fraction : ℝ := 0.29411764705882354

/-- Proves that the number of pounds of nuts mixed with 5 pounds of raisins is 4 -/
theorem nuts_mixed_with_raisins :
  let r := 1  -- Cost of 1 pound of raisins (arbitrary unit)
  let n := cost_ratio * r  -- Cost of 1 pound of nuts
  pounds_of_nuts * n / (pounds_of_nuts * n + pounds_of_raisins * r) = 1 - raisin_cost_fraction :=
by sorry

end NUMINAMATH_CALUDE_nuts_mixed_with_raisins_l3573_357388


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_equality_l3573_357361

-- Define the GDP value in yuan
def gdp : ℝ := 121 * 10^12

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.21 * 10^14

-- Theorem stating that the GDP is equal to its scientific notation representation
theorem gdp_scientific_notation_equality : gdp = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_equality_l3573_357361


namespace NUMINAMATH_CALUDE_corrected_mean_l3573_357300

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value correct_value : ℝ) :
  n = 40 ∧ original_mean = 36 ∧ incorrect_value = 20 ∧ correct_value = 34 →
  (n : ℝ) * original_mean - incorrect_value + correct_value = n * 36.35 :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l3573_357300


namespace NUMINAMATH_CALUDE_expression_evaluation_l3573_357391

theorem expression_evaluation :
  let a : ℝ := 1
  let b : ℝ := -1
  5 * (3 * a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) + 1 = -17 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3573_357391


namespace NUMINAMATH_CALUDE_new_car_distance_l3573_357327

theorem new_car_distance (old_car_speed : ℝ) (old_car_distance : ℝ) (new_car_speed : ℝ) : 
  old_car_distance = 150 →
  new_car_speed = old_car_speed * 1.3 →
  new_car_speed * (old_car_distance / old_car_speed) = 195 := by
sorry

end NUMINAMATH_CALUDE_new_car_distance_l3573_357327


namespace NUMINAMATH_CALUDE_suitcase_electronics_weight_l3573_357358

/-- Given a suitcase with books, clothes, and electronics, prove the weight of electronics. -/
theorem suitcase_electronics_weight 
  (B C E : ℝ) -- Weights of books, clothes, and electronics
  (h1 : B / C = 5 / 4) -- Initial ratio of books to clothes
  (h2 : C / E = 4 / 2) -- Initial ratio of clothes to electronics
  (h3 : B / (C - 9) = 10 / 4) -- New ratio after removing 9 pounds of clothes
  : E = 9 := by
  sorry

end NUMINAMATH_CALUDE_suitcase_electronics_weight_l3573_357358


namespace NUMINAMATH_CALUDE_quadratic_max_iff_a_neg_l3573_357380

/-- A quadratic function -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Definition of having a maximum value for a quadratic function -/
def has_maximum (f : QuadraticFunction) : Prop :=
  ∃ x₀ : ℝ, ∀ x : ℝ, f.a * x^2 + f.b * x + f.c ≤ f.a * x₀^2 + f.b * x₀ + f.c

/-- Theorem: A quadratic function has a maximum value if and only if a < 0 -/
theorem quadratic_max_iff_a_neg (f : QuadraticFunction) :
  has_maximum f ↔ f.a < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_iff_a_neg_l3573_357380


namespace NUMINAMATH_CALUDE_discussions_probability_l3573_357316

def word := "DISCUSSIONS"

theorem discussions_probability : 
  let total_arrangements := Nat.factorial 11 / (Nat.factorial 4 * Nat.factorial 2)
  let favorable_arrangements := Nat.factorial 8 / Nat.factorial 2
  (favorable_arrangements : ℚ) / total_arrangements = 4 / 165 := by
  sorry

end NUMINAMATH_CALUDE_discussions_probability_l3573_357316


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_4sqrt6_l3573_357368

theorem sqrt_sum_equals_4sqrt6 :
  Real.sqrt (16 - 12 * Real.sqrt 3) + Real.sqrt (16 + 12 * Real.sqrt 3) = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_4sqrt6_l3573_357368


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l3573_357396

theorem complex_magnitude_equality (m : ℝ) (h : m > 0) :
  Complex.abs (4 + m * Complex.I) = 4 * Real.sqrt 5 → m = 8 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l3573_357396


namespace NUMINAMATH_CALUDE_prob_both_red_given_one_red_l3573_357389

/-- Represents a card with two sides -/
structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

/-- Represents the box of cards -/
def box : List Card := [
  {side1 := false, side2 := false},
  {side1 := false, side2 := false},
  {side1 := false, side2 := false},
  {side1 := false, side2 := false},
  {side1 := true,  side2 := false},
  {side1 := true,  side2 := false},
  {side1 := true,  side2 := true},
  {side1 := true,  side2 := true},
  {side1 := true,  side2 := true}
]

/-- The probability of drawing a card with a red side -/
def probRedSide : Rat := 8 / 18

/-- The probability that both sides are red, given that one side is red -/
theorem prob_both_red_given_one_red :
  (3 : Rat) / 4 = (List.filter (fun c => c.side1 ∧ c.side2) box).length / 
                  (List.filter (fun c => c.side1 ∨ c.side2) box).length :=
by sorry

end NUMINAMATH_CALUDE_prob_both_red_given_one_red_l3573_357389


namespace NUMINAMATH_CALUDE_mr_mcpherson_contribution_l3573_357322

def total_rent : ℝ := 1200
def mrs_mcpherson_percentage : ℝ := 30

theorem mr_mcpherson_contribution :
  total_rent - (mrs_mcpherson_percentage / 100 * total_rent) = 840 := by
  sorry

end NUMINAMATH_CALUDE_mr_mcpherson_contribution_l3573_357322


namespace NUMINAMATH_CALUDE_tulips_to_add_l3573_357312

def tulip_to_daisy_ratio : ℚ := 3 / 4
def initial_daisies : ℕ := 32
def added_daisies : ℕ := 24

theorem tulips_to_add (tulips_added : ℕ) : 
  (tulip_to_daisy_ratio * (initial_daisies + added_daisies : ℚ)).num = 
  (tulip_to_daisy_ratio * initial_daisies).num + tulips_added → 
  tulips_added = 18 :=
by sorry

end NUMINAMATH_CALUDE_tulips_to_add_l3573_357312


namespace NUMINAMATH_CALUDE_expression_evaluation_l3573_357337

theorem expression_evaluation :
  let x : ℚ := -1/2
  3 * x^2 - (5*x - 3*(2*x - 1) + 7*x^2) = -9/2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3573_357337


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l3573_357321

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_deg_has_20_sides :
  ∀ n : ℕ,
  n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l3573_357321


namespace NUMINAMATH_CALUDE_common_integer_root_l3573_357317

theorem common_integer_root (a : ℤ) : 
  (∃ x : ℤ, a * x + a = 7 ∧ 3 * x - a = 17) ↔ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_common_integer_root_l3573_357317


namespace NUMINAMATH_CALUDE_twenty_fifth_term_is_173_l3573_357353

/-- The nth term of an arithmetic progression -/
def arithmetic_progression (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 25th term of the arithmetic progression with first term 5 and common difference 7 is 173 -/
theorem twenty_fifth_term_is_173 :
  arithmetic_progression 5 7 25 = 173 := by
  sorry

end NUMINAMATH_CALUDE_twenty_fifth_term_is_173_l3573_357353


namespace NUMINAMATH_CALUDE_article_sale_loss_percentage_l3573_357384

theorem article_sale_loss_percentage 
  (cost : ℝ) 
  (original_price : ℝ) 
  (discounted_price : ℝ) 
  (h1 : original_price = cost * 1.35)
  (h2 : discounted_price = original_price * (2/3)) :
  (cost - discounted_price) / cost * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_article_sale_loss_percentage_l3573_357384


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l3573_357329

/-- The set of functions satisfying the given conditions -/
def S : Set (ℕ → ℝ) :=
  {f | f 1 = 2 ∧ ∀ n, f (n + 1) ≥ f n ∧ f n ≥ (n / (n + 1 : ℝ)) * f (2 * n)}

/-- The smallest natural number M such that f(n) < M for all f ∈ S and n ∈ ℕ -/
theorem smallest_upper_bound : ∃! M : ℕ, 
  (∀ f ∈ S, ∀ n, f n < M) ∧ 
  (∀ M' : ℕ, (∀ f ∈ S, ∀ n, f n < M') → M ≤ M') :=
by
  use 10
  sorry

#check smallest_upper_bound

end NUMINAMATH_CALUDE_smallest_upper_bound_l3573_357329


namespace NUMINAMATH_CALUDE_theater_promotion_l3573_357395

theorem theater_promotion (capacity : ℕ) (ticket_interval : ℕ) (popcorn_interval : ℕ) (soda_interval : ℕ) :
  capacity = 3600 →
  ticket_interval = 90 →
  popcorn_interval = 36 →
  soda_interval = 60 →
  (capacity / (Nat.lcm ticket_interval (Nat.lcm popcorn_interval soda_interval))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_theater_promotion_l3573_357395


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3573_357373

theorem necessary_not_sufficient_condition (x : ℝ) : 
  (∀ y : ℝ, y > 2 → y > 1) ∧ (∃ z : ℝ, z > 1 ∧ z ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3573_357373


namespace NUMINAMATH_CALUDE_red_beads_in_necklace_l3573_357352

/-- Represents the number of red beads in each group -/
def redBeadsInGroup (n : ℕ) : ℕ := 2 * n

/-- Represents the total number of red beads up to the nth group -/
def totalRedBeads (n : ℕ) : ℕ := n * (n + 1)

/-- Represents the total number of beads (red and white) up to the nth group -/
def totalBeads (n : ℕ) : ℕ := n + totalRedBeads n

theorem red_beads_in_necklace :
  ∃ n : ℕ, totalBeads n ≤ 99 ∧ totalBeads (n + 1) > 99 ∧ totalRedBeads n = 90 := by
  sorry

end NUMINAMATH_CALUDE_red_beads_in_necklace_l3573_357352


namespace NUMINAMATH_CALUDE_evaluate_complex_expression_l3573_357339

theorem evaluate_complex_expression :
  ∃ (x : ℝ), x > 0 ∧ x^2 = (297 - 99*Real.sqrt 5 + 108*Real.sqrt 6 - 36*Real.sqrt 30) / 64 ∧
  x = (3*(Real.sqrt 3 + Real.sqrt 8)) / (4*Real.sqrt (3 + Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_evaluate_complex_expression_l3573_357339


namespace NUMINAMATH_CALUDE_magnitude_of_vector_AB_l3573_357398

theorem magnitude_of_vector_AB (OA OB : ℝ × ℝ) : 
  OA = (Real.cos (15 * π / 180), Real.sin (15 * π / 180)) →
  OB = (Real.cos (75 * π / 180), Real.sin (75 * π / 180)) →
  Real.sqrt ((OB.1 - OA.1)^2 + (OB.2 - OA.2)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_AB_l3573_357398


namespace NUMINAMATH_CALUDE_books_from_second_shop_l3573_357381

/-- Proves the number of books bought from the second shop given the conditions -/
theorem books_from_second_shop 
  (first_shop_books : ℕ) 
  (first_shop_cost : ℕ) 
  (second_shop_cost : ℕ) 
  (average_price : ℚ) 
  (h1 : first_shop_books = 65)
  (h2 : first_shop_cost = 1150)
  (h3 : second_shop_cost = 920)
  (h4 : average_price = 18) : 
  ℕ := by
  sorry

#check books_from_second_shop

end NUMINAMATH_CALUDE_books_from_second_shop_l3573_357381


namespace NUMINAMATH_CALUDE_greatest_a_for_equation_l3573_357372

theorem greatest_a_for_equation :
  ∃ (a : ℝ), 
    (∀ (x : ℝ), (5 * Real.sqrt ((2 * x)^2 + 1) - 4 * x^2 - 1) / (Real.sqrt (1 + 4 * x^2) + 3) = 3 → x ≤ a) ∧
    (5 * Real.sqrt ((2 * a)^2 + 1) - 4 * a^2 - 1) / (Real.sqrt (1 + 4 * a^2) + 3) = 3 ∧
    a = Real.sqrt ((5 + Real.sqrt 10) / 2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_a_for_equation_l3573_357372


namespace NUMINAMATH_CALUDE_money_distribution_l3573_357366

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (ac_sum : A + C = 200)
  (bc_sum : B + C = 330) :
  C = 30 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3573_357366


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3573_357367

theorem negation_of_universal_proposition :
  (¬ ∀ a : ℝ, a ≥ 0 → a^4 + a^2 ≥ 0) ↔ (∃ a : ℝ, a ≥ 0 ∧ a^4 + a^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3573_357367


namespace NUMINAMATH_CALUDE_max_sum_cubes_l3573_357330

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  ∃ (M : ℝ), M = 5 * Real.sqrt 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 ≤ M ∧
  ∃ (a' b' c' d' e' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 5 ∧
                           a'^3 + b'^3 + c'^3 + d'^3 + e'^3 = M :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l3573_357330


namespace NUMINAMATH_CALUDE_circle_radius_l3573_357394

/-- Given a circle with area A = k π r², where k = 4 and A = 225π, prove that the radius r is 7.5 units. -/
theorem circle_radius (k : ℝ) (A : ℝ) (r : ℝ) (h1 : k = 4) (h2 : A = 225 * Real.pi) (h3 : A = k * Real.pi * r^2) : r = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3573_357394


namespace NUMINAMATH_CALUDE_food_fraction_proof_l3573_357347

def initial_amount : ℝ := 499.9999999999999

theorem food_fraction_proof (clothes_fraction : ℝ) (travel_fraction : ℝ) (food_fraction : ℝ) 
  (h1 : clothes_fraction = 1/3)
  (h2 : travel_fraction = 1/4)
  (h3 : initial_amount * (1 - clothes_fraction) * (1 - food_fraction) * (1 - travel_fraction) = 200) :
  food_fraction = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_food_fraction_proof_l3573_357347


namespace NUMINAMATH_CALUDE_statue_cost_l3573_357349

theorem statue_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 620 →
  profit_percentage = 25 →
  selling_price = original_cost * (1 + profit_percentage / 100) →
  original_cost = 496 := by
sorry

end NUMINAMATH_CALUDE_statue_cost_l3573_357349


namespace NUMINAMATH_CALUDE_joe_remaining_money_l3573_357325

def joe_pocket_money : ℚ := 450

def chocolate_fraction : ℚ := 1/9
def fruit_fraction : ℚ := 2/5

def remaining_money : ℚ := joe_pocket_money - (chocolate_fraction * joe_pocket_money) - (fruit_fraction * joe_pocket_money)

theorem joe_remaining_money :
  remaining_money = 220 :=
by sorry

end NUMINAMATH_CALUDE_joe_remaining_money_l3573_357325


namespace NUMINAMATH_CALUDE_intersection_M_N_l3573_357387

def M : Set ℝ := {x | x^2 + x - 2 = 0}
def N : Set ℝ := {x | x < 0}

theorem intersection_M_N : M ∩ N = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3573_357387


namespace NUMINAMATH_CALUDE_jack_books_left_l3573_357324

/-- The number of books left in Jack's classics section -/
def books_left (authors : ℕ) (books_per_author : ℕ) (lent_books : ℕ) (misplaced_books : ℕ) : ℕ :=
  authors * books_per_author - (lent_books + misplaced_books)

theorem jack_books_left :
  books_left 10 45 17 8 = 425 := by
  sorry

end NUMINAMATH_CALUDE_jack_books_left_l3573_357324


namespace NUMINAMATH_CALUDE_cube_split_2017_l3573_357332

theorem cube_split_2017 (m : ℕ) (h1 : m > 1) : 
  (m^3 = (m - 1)*(m^2 + m + 1) + (m - 1)^2 + (m - 1)^2 + 1) → 
  ((m - 1)*(m^2 + m + 1) = 2017 ∨ (m - 1)^2 = 2017 ∨ (m - 1)^2 + 2 = 2017) → 
  m = 46 := by
sorry

end NUMINAMATH_CALUDE_cube_split_2017_l3573_357332


namespace NUMINAMATH_CALUDE_pirate_treasure_ratio_l3573_357355

theorem pirate_treasure_ratio : 
  let total_gold : ℕ := 3500
  let num_chests : ℕ := 5
  let total_silver : ℕ := 500
  let coins_per_chest : ℕ := 1000
  let gold_per_chest : ℕ := total_gold / num_chests
  let silver_per_chest : ℕ := total_silver / num_chests
  let bronze_per_chest : ℕ := coins_per_chest - gold_per_chest - silver_per_chest
  bronze_per_chest = 2 * silver_per_chest :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_ratio_l3573_357355


namespace NUMINAMATH_CALUDE_seven_story_pagoda_top_lights_verify_total_lights_l3573_357314

/-- Represents a pagoda with a given number of stories and a total number of lights -/
structure Pagoda where
  stories : ℕ
  total_lights : ℕ
  lights_ratio : ℕ

/-- Calculates the number of lights at the top of the pagoda -/
def top_lights (p : Pagoda) : ℕ :=
  p.total_lights / (2^p.stories - 1)

/-- Theorem stating that a 7-story pagoda with 381 total lights and a doubling ratio has 3 lights at the top -/
theorem seven_story_pagoda_top_lights :
  let p := Pagoda.mk 7 381 2
  top_lights p = 3 := by
  sorry

/-- Verifies that the sum of lights across all stories equals the total lights -/
theorem verify_total_lights (p : Pagoda) :
  (top_lights p) * (2^p.stories - 1) = p.total_lights := by
  sorry

end NUMINAMATH_CALUDE_seven_story_pagoda_top_lights_verify_total_lights_l3573_357314


namespace NUMINAMATH_CALUDE_calculation_proof_l3573_357362

theorem calculation_proof (initial_amount : ℝ) (first_percentage : ℝ) 
  (discount_percentage : ℝ) (target_percentage : ℝ) (tax_rate : ℝ) : 
  initial_amount = 4000 ∧ 
  first_percentage = 0.15 ∧ 
  discount_percentage = 0.25 ∧ 
  target_percentage = 0.07 ∧ 
  tax_rate = 0.10 → 
  (1 + tax_rate) * (target_percentage * (1 - discount_percentage) * (first_percentage * initial_amount)) = 34.65 := by
sorry

#eval (1 + 0.10) * (0.07 * (1 - 0.25) * (0.15 * 4000))

end NUMINAMATH_CALUDE_calculation_proof_l3573_357362


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3573_357348

/-- Represents a T-shaped structure made of unit cubes -/
structure TCube where
  base : Nat
  stack : Nat

/-- Calculates the volume of the T-shaped structure -/
def volume (t : TCube) : Nat :=
  t.base + t.stack

/-- Calculates the surface area of the T-shaped structure -/
def surfaceArea (t : TCube) : Nat :=
  2 * (5 + 3) + 1 + 3 * 5

/-- The specific T-shaped structure described in the problem -/
def specificT : TCube :=
  { base := 4, stack := 4 }

theorem volume_to_surface_area_ratio :
  (volume specificT : ℚ) / (surfaceArea specificT : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3573_357348


namespace NUMINAMATH_CALUDE_no_matching_units_digits_l3573_357310

theorem no_matching_units_digits :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 100 → (x % 10 ≠ (101 - x) % 10) :=
by sorry

end NUMINAMATH_CALUDE_no_matching_units_digits_l3573_357310


namespace NUMINAMATH_CALUDE_smallest_divisible_by_million_l3573_357344

/-- A geometric sequence with first term a₁ and common ratio r -/
def geometric_sequence (a₁ : ℚ) (r : ℚ) : ℕ → ℚ :=
  λ n => a₁ * r^(n - 1)

/-- The nth term of the sequence is divisible by m -/
def is_divisible_by (seq : ℕ → ℚ) (n : ℕ) (m : ℕ) : Prop :=
  ∃ k : ℤ, seq n = k * (m : ℚ)

theorem smallest_divisible_by_million :
  let a₁ : ℚ := 3/4
  let a₂ : ℚ := 15
  let r : ℚ := a₂ / a₁
  let seq := geometric_sequence a₁ r
  (∀ n < 7, ¬ is_divisible_by seq n 1000000) ∧
  is_divisible_by seq 7 1000000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_million_l3573_357344


namespace NUMINAMATH_CALUDE_will_old_cards_l3573_357360

/-- Calculates the number of old baseball cards Will had. -/
def old_cards (cards_per_page : ℕ) (new_cards : ℕ) (pages_used : ℕ) : ℕ :=
  cards_per_page * pages_used - new_cards

/-- Theorem stating that Will had 10 old cards. -/
theorem will_old_cards : old_cards 3 8 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_will_old_cards_l3573_357360


namespace NUMINAMATH_CALUDE_bike_ride_speed_l3573_357377

theorem bike_ride_speed (joann_speed joann_time fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 2.5) :
  (joann_speed * joann_time) / fran_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_bike_ride_speed_l3573_357377


namespace NUMINAMATH_CALUDE_sum_greater_than_double_l3573_357331

theorem sum_greater_than_double (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a + b > 2*b := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_double_l3573_357331


namespace NUMINAMATH_CALUDE_floor_sum_equals_four_l3573_357363

theorem floor_sum_equals_four (x y : ℝ) : 
  (⌊x⌋^2 + ⌊y⌋^2 = 4) ↔ 
  ((2 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 1) ∨
   (-2 ≤ x ∧ x < -1 ∧ 0 ≤ y ∧ y < 1) ∨
   (0 ≤ x ∧ x < 1 ∧ 2 ≤ y ∧ y < 3) ∨
   (0 ≤ x ∧ x < 1 ∧ -2 ≤ y ∧ y < -1)) :=
by sorry

end NUMINAMATH_CALUDE_floor_sum_equals_four_l3573_357363


namespace NUMINAMATH_CALUDE_product_consecutive_integers_square_l3573_357302

theorem product_consecutive_integers_square (x : ℤ) :
  ∃ (y : ℤ), x * (x + 1) * (x + 2) = y^2 ↔ x = 0 ∨ x = -1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_product_consecutive_integers_square_l3573_357302


namespace NUMINAMATH_CALUDE_equation_solution_l3573_357326

theorem equation_solution :
  ∀ N : ℚ, (5 + 6 + 7) / 3 = (2020 + 2021 + 2022) / N → N = 1010.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3573_357326


namespace NUMINAMATH_CALUDE_fraction_simplification_l3573_357309

theorem fraction_simplification :
  (270 : ℚ) / 24 * 7 / 210 * 6 / 4 = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3573_357309


namespace NUMINAMATH_CALUDE_min_flowers_for_bouquets_l3573_357359

/-- The number of different types of flowers in the box -/
def num_flower_types : ℕ := 6

/-- The number of flowers needed to make one bouquet -/
def flowers_per_bouquet : ℕ := 5

/-- The number of bouquets we want to guarantee -/
def target_bouquets : ℕ := 10

/-- The minimum number of flowers needed to guarantee the target number of bouquets -/
def min_flowers_needed : ℕ := 70

theorem min_flowers_for_bouquets :
  ∀ (n : ℕ), n ≥ min_flowers_needed →
  ∀ (f : Fin n → Fin num_flower_types),
  ∃ (S : Finset (Fin n)),
  S.card = target_bouquets * flowers_per_bouquet ∧
  ∀ (t : Fin num_flower_types),
  (S.filter (fun i => f i = t)).card ≥ target_bouquets * flowers_per_bouquet / num_flower_types :=
by sorry

end NUMINAMATH_CALUDE_min_flowers_for_bouquets_l3573_357359


namespace NUMINAMATH_CALUDE_wendi_chicken_count_l3573_357382

/-- The number of chickens Wendi has after a series of events -/
def final_chicken_count (initial : ℕ) (doubled : ℕ) (lost : ℕ) (found : ℕ) : ℕ :=
  initial + doubled - lost + found

/-- Theorem stating the final number of chickens Wendi has -/
theorem wendi_chicken_count : 
  final_chicken_count 4 4 1 6 = 13 := by sorry

end NUMINAMATH_CALUDE_wendi_chicken_count_l3573_357382


namespace NUMINAMATH_CALUDE_log_equation_solution_l3573_357333

theorem log_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log 5
  ∀ x : ℝ, f (x^2 - 25*x) = 3 ↔ x = 5*(5 + 3*Real.sqrt 5) ∨ x = 5*(5 - 3*Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3573_357333


namespace NUMINAMATH_CALUDE_tile_ratio_l3573_357335

theorem tile_ratio (total : Nat) (yellow purple white : Nat)
  (h_total : total = 20)
  (h_yellow : yellow = 3)
  (h_purple : purple = 6)
  (h_white : white = 7) :
  (total - (yellow + purple + white)) / yellow = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tile_ratio_l3573_357335


namespace NUMINAMATH_CALUDE_angle_z_is_90_l3573_357306

-- Define a triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)

-- Define the property that the sum of angles in a triangle is 180°
axiom triangle_angle_sum (t : Triangle) : t.X + t.Y + t.Z = 180

-- Theorem: If the sum of angles X and Y is 90°, then angle Z is 90°
theorem angle_z_is_90 (t : Triangle) (h : t.X + t.Y = 90) : t.Z = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_z_is_90_l3573_357306


namespace NUMINAMATH_CALUDE_tank_capacity_is_72_liters_l3573_357397

/-- The total capacity of a water tank in liters. -/
def tank_capacity : ℝ := 72

/-- The amount of water in the tank when it's 40% full, in liters. -/
def water_at_40_percent : ℝ := 0.4 * tank_capacity

/-- The amount of water in the tank when it's 10% empty, in liters. -/
def water_at_10_percent_empty : ℝ := 0.9 * tank_capacity

/-- Theorem stating that the tank capacity is 72 liters, given the condition. -/
theorem tank_capacity_is_72_liters :
  water_at_10_percent_empty - water_at_40_percent = 36 →
  tank_capacity = 72 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_is_72_liters_l3573_357397


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l3573_357376

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m β) : 
  perpendicular n β := by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l3573_357376


namespace NUMINAMATH_CALUDE_k_value_l3573_357304

def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + k^3 * x

theorem k_value (k : ℝ) : 
  (∀ x, deriv (f k) x = 6 * x^2 + 6 * x + k^3) → 
  deriv (f k) 0 = 27 → 
  k = 3 := by sorry

end NUMINAMATH_CALUDE_k_value_l3573_357304


namespace NUMINAMATH_CALUDE_sum_of_digits_square_of_ones_l3573_357371

/-- Given a natural number n, construct a number consisting of n ones -/
def number_of_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

/-- Theorem: For a number consisting of n ones, the sum of digits of its square equals n^2 -/
theorem sum_of_digits_square_of_ones (n : ℕ) :
  sum_of_digits ((number_of_ones n)^2) = n^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_of_ones_l3573_357371


namespace NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_540_l3573_357374

theorem sum_of_two_smallest_prime_factors_of_540 :
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧
  p ∣ 540 ∧ q ∣ 540 ∧
  (∀ (r : ℕ), Nat.Prime r → r ∣ 540 → r ≥ p) ∧
  (∀ (r : ℕ), Nat.Prime r → r ∣ 540 → r ≠ p → r ≥ q) ∧
  p + q = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_540_l3573_357374


namespace NUMINAMATH_CALUDE_jason_retirement_age_l3573_357311

/-- Jason's career in the military -/
def military_career (joining_age : ℕ) (years_to_chief : ℕ) (additional_years : ℕ) : Prop :=
  let years_to_master_chief : ℕ := years_to_chief + (years_to_chief / 4)
  let total_years : ℕ := years_to_chief + years_to_master_chief + additional_years
  let retirement_age : ℕ := joining_age + total_years
  retirement_age = 46

theorem jason_retirement_age :
  military_career 18 8 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jason_retirement_age_l3573_357311


namespace NUMINAMATH_CALUDE_floor_equality_iff_in_interval_l3573_357350

theorem floor_equality_iff_in_interval (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 5/3 ≤ x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_floor_equality_iff_in_interval_l3573_357350


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3573_357307

/-- Given a geometric sequence of 10 terms, prove that if the sum of these terms is 18
    and the sum of their reciprocals is 6, then the product of these terms is (1/6)^55 -/
theorem geometric_sequence_product (a r : ℝ) (h1 : a ≠ 0) (h2 : r ≠ 0) (h3 : r ≠ 1) :
  (a * r * (r^10 - 1) / (r - 1) = 18) →
  (1 / (a * r) * (1 - 1/r^10) / (1 - 1/r) = 6) →
  (a * r)^55 = (1/6)^55 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3573_357307


namespace NUMINAMATH_CALUDE_least_N_for_probability_l3573_357328

def P (N : ℕ) : ℚ := 2 * (N / 3 + 1) / (N + 2)

def is_multiple_of_seven (N : ℕ) : Prop := ∃ k, N = 7 * k

theorem least_N_for_probability (N : ℕ) :
  is_multiple_of_seven N →
  (∀ M, is_multiple_of_seven M → M < N → P M ≥ 7/10) →
  P N < 7/10 →
  N = 700 :=
sorry

end NUMINAMATH_CALUDE_least_N_for_probability_l3573_357328


namespace NUMINAMATH_CALUDE_truncated_cone_volume_l3573_357308

/-- The volume of a truncated cone with specific diagonal properties -/
theorem truncated_cone_volume 
  (l : ℝ) 
  (α : ℝ) 
  (h_positive : l > 0)
  (h_angle : 0 < α ∧ α < π)
  (h_diagonal_ratio : ∃ (k : ℝ), k > 0 ∧ 2 * k = l ∧ k = l / 3)
  : ∃ (V : ℝ), V = (7 / 54) * π * l^3 * Real.sin α * Real.sin (α / 2) :=
sorry

end NUMINAMATH_CALUDE_truncated_cone_volume_l3573_357308


namespace NUMINAMATH_CALUDE_stella_spent_40_l3573_357351

/-- Represents the inventory and financial data of Stella's antique shop --/
structure AntiqueShop where
  dolls : ℕ
  clocks : ℕ
  glasses : ℕ
  doll_price : ℕ
  clock_price : ℕ
  glass_price : ℕ
  profit : ℕ

/-- Calculates the total revenue from selling all items --/
def total_revenue (shop : AntiqueShop) : ℕ :=
  shop.dolls * shop.doll_price + shop.clocks * shop.clock_price + shop.glasses * shop.glass_price

/-- Theorem stating that Stella spent $40 to buy everything --/
theorem stella_spent_40 (shop : AntiqueShop) 
    (h1 : shop.dolls = 3)
    (h2 : shop.clocks = 2)
    (h3 : shop.glasses = 5)
    (h4 : shop.doll_price = 5)
    (h5 : shop.clock_price = 15)
    (h6 : shop.glass_price = 4)
    (h7 : shop.profit = 25) : 
  total_revenue shop - shop.profit = 40 := by
  sorry

end NUMINAMATH_CALUDE_stella_spent_40_l3573_357351


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3573_357370

/-- The measure of the angle between vectors a = (1, √3) and b = (√3, 1) is π/6 -/
theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (1, Real.sqrt 3) →
  b = (Real.sqrt 3, 1) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l3573_357370


namespace NUMINAMATH_CALUDE_parabola_point_ordinate_l3573_357305

/-- Represents a parabola y = ax^2 with a > 0 -/
structure Parabola where
  a : ℝ
  a_pos : a > 0

/-- A point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.a * x^2

theorem parabola_point_ordinate (p : Parabola) (M : PointOnParabola p) 
    (focus_directrix_dist : (1 : ℝ) / (2 * p.a) = 1)
    (M_to_focus_dist : Real.sqrt ((M.x - 0)^2 + (M.y - 1 / (4 * p.a))^2) = 5) :
    M.y = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordinate_l3573_357305


namespace NUMINAMATH_CALUDE_sum_fifth_powers_divisible_by_30_l3573_357303

theorem sum_fifth_powers_divisible_by_30 
  (n : ℕ) 
  (a : Fin n → ℕ) 
  (h : 30 ∣ (Finset.univ.sum (λ i => a i))) : 
  30 ∣ (Finset.univ.sum (λ i => (a i)^5)) :=
sorry

end NUMINAMATH_CALUDE_sum_fifth_powers_divisible_by_30_l3573_357303


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3573_357318

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The left vertex of the hyperbola -/
def left_vertex (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- An asymptote of the hyperbola -/
def asymptote (h : Hyperbola a b) : Set (ℝ × ℝ) := sorry

/-- The projection of a point onto a line -/
def project (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The area of a triangle formed by three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  let A := left_vertex h
  let F := right_focus h
  let asym := asymptote h
  let B := project A asym
  let Q := project F asym
  let O := (0, 0)
  triangle_area A B O / triangle_area F Q O = 1 / 2 →
  eccentricity h = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3573_357318


namespace NUMINAMATH_CALUDE_no_real_roots_l3573_357301

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 2) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3573_357301


namespace NUMINAMATH_CALUDE_c_share_is_56_l3573_357336

/-- Represents the share of money for each person -/
structure Share where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the money distribution problem -/
def moneyDistribution (s : Share) : Prop :=
  s.b = 0.65 * s.a ∧ 
  s.c = 0.40 * s.a ∧ 
  s.a + s.b + s.c = 287

/-- Theorem stating that under the given conditions, C's share is 56 -/
theorem c_share_is_56 :
  ∃ s : Share, moneyDistribution s ∧ s.c = 56 := by
  sorry


end NUMINAMATH_CALUDE_c_share_is_56_l3573_357336


namespace NUMINAMATH_CALUDE_liz_additional_money_needed_l3573_357315

def original_price : ℝ := 32500
def new_car_price : ℝ := 30000
def sale_percentage : ℝ := 0.8

theorem liz_additional_money_needed :
  new_car_price - sale_percentage * original_price = 4000 := by
  sorry

end NUMINAMATH_CALUDE_liz_additional_money_needed_l3573_357315


namespace NUMINAMATH_CALUDE_jake_fewer_than_steven_peach_difference_is_twelve_l3573_357365

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 19

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := 7

/-- Jake has fewer peaches than Steven -/
theorem jake_fewer_than_steven : jake_peaches < steven_peaches := by sorry

/-- The difference between Steven's and Jake's peaches -/
def peach_difference : ℕ := steven_peaches - jake_peaches

/-- Prove that the difference between Steven's and Jake's peaches is 12 -/
theorem peach_difference_is_twelve : peach_difference = 12 := by sorry

end NUMINAMATH_CALUDE_jake_fewer_than_steven_peach_difference_is_twelve_l3573_357365


namespace NUMINAMATH_CALUDE_another_root_of_p_l3573_357379

-- Define the polynomials p and q
def p (a b : ℤ) (x : ℂ) : ℂ := x^3 + a*x^2 + b*x - 1
def q (c d : ℤ) (x : ℂ) : ℂ := x^3 + c*x^2 + d*x + 1

-- State the theorem
theorem another_root_of_p (a b c d : ℤ) (α : ℂ) :
  (∃ (a b : ℤ), p a b α = 0) →  -- α is a root of p(x) = 0
  (∀ (r : ℚ), p a b r ≠ 0) →  -- p(x) is irreducible over the rationals
  (∃ (c d : ℤ), q c d (α + 1) = 0) →  -- α + 1 is a root of q(x) = 0
  (∃ β : ℂ, p a b β = 0 ∧ (β = -1/(α+1) ∨ β = -(α+1)/α)) :=
by sorry

end NUMINAMATH_CALUDE_another_root_of_p_l3573_357379


namespace NUMINAMATH_CALUDE_second_pipe_fill_time_l3573_357345

theorem second_pipe_fill_time (pipe1_time pipe2_time outlet_time all_pipes_time : ℝ) 
  (h1 : pipe1_time = 18)
  (h2 : outlet_time = 45)
  (h3 : all_pipes_time = 0.08333333333333333)
  (h4 : 1 / pipe1_time + 1 / pipe2_time - 1 / outlet_time = 1 / all_pipes_time) :
  pipe2_time = 20 := by sorry

end NUMINAMATH_CALUDE_second_pipe_fill_time_l3573_357345


namespace NUMINAMATH_CALUDE_wednesday_most_frequent_l3573_357386

/-- Represents days of the week -/
inductive DayOfWeek
  | sunday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday

/-- Represents a date in the year 2014 -/
structure Date2014 where
  month : Nat
  day : Nat

def march_9_2014 : Date2014 := ⟨3, 9⟩

/-- The number of days in 2014 -/
def days_in_2014 : Nat := 365

/-- Function to determine the day of the week for a given date in 2014 -/
def dayOfWeek (d : Date2014) : DayOfWeek := sorry

/-- Function to count occurrences of each day of the week in 2014 -/
def countDayOccurrences (day : DayOfWeek) : Nat := sorry

/-- Theorem stating that Wednesday occurs most frequently in 2014 -/
theorem wednesday_most_frequent :
  (dayOfWeek march_9_2014 = DayOfWeek.sunday) →
  (∀ d : DayOfWeek, countDayOccurrences DayOfWeek.wednesday ≥ countDayOccurrences d) :=
by sorry

end NUMINAMATH_CALUDE_wednesday_most_frequent_l3573_357386


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l3573_357383

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = x^3 + x + 1) :
  ∀ x < 0, f x = x^3 + x - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l3573_357383


namespace NUMINAMATH_CALUDE_volleyball_count_l3573_357313

theorem volleyball_count (total : ℕ) (soccer : ℕ) (basketball : ℕ) (tennis : ℕ) (baseball : ℕ) (hockey : ℕ) (volleyball : ℕ) :
  total = 180 →
  soccer = 20 →
  basketball = soccer + 5 →
  tennis = 2 * soccer →
  baseball = soccer + 10 →
  hockey = tennis / 2 →
  volleyball = total - (soccer + basketball + tennis + baseball + hockey) →
  volleyball = 45 := by
sorry

end NUMINAMATH_CALUDE_volleyball_count_l3573_357313


namespace NUMINAMATH_CALUDE_inverse_sum_product_l3573_357393

theorem inverse_sum_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 3*x + 4*y ≠ 0) :
  (3*x + 4*y)⁻¹ * ((3*x)⁻¹ + (4*y)⁻¹) = (12*x*y)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_product_l3573_357393


namespace NUMINAMATH_CALUDE_intersection_A_B_l3573_357392

def set_A : Set ℝ := {x | x^2 - 11*x - 12 < 0}

def set_B : Set ℝ := {x | ∃ n : ℤ, x = 3*n + 1}

theorem intersection_A_B :
  set_A ∩ set_B = {1, 4, 7, 10} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3573_357392


namespace NUMINAMATH_CALUDE_sprint_distance_l3573_357354

/-- Given a constant speed and a duration, calculates the distance traveled. -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that sprinting at 6 miles per hour for 4 hours results in a distance of 24 miles. -/
theorem sprint_distance : distance_traveled 6 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sprint_distance_l3573_357354


namespace NUMINAMATH_CALUDE_doubling_points_theorem_l3573_357378

/-- Definition of a "doubling point" -/
def is_doubling_point (P Q : ℝ × ℝ) : Prop :=
  2 * (P.1 + Q.1) = P.2 + Q.2

/-- The point P₁ -/
def P₁ : ℝ × ℝ := (1, 0)

/-- Q₁ and Q₂ are specified points -/
def Q₁ : ℝ × ℝ := (3, 8)
def Q₂ : ℝ × ℝ := (-2, -2)

/-- The parabola y = x² - 2x - 3 -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem doubling_points_theorem :
  (is_doubling_point P₁ Q₁) ∧
  (is_doubling_point P₁ Q₂) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    is_doubling_point P₁ (x₁, parabola x₁) ∧
    is_doubling_point P₁ (x₂, parabola x₂)) ∧
  (∀ (Q : ℝ × ℝ), is_doubling_point P₁ Q → 
    Real.sqrt ((Q.1 - P₁.1)^2 + (Q.2 - P₁.2)^2) ≥ 4 * Real.sqrt 5 / 5) ∧
  (∃ (Q : ℝ × ℝ), is_doubling_point P₁ Q ∧
    Real.sqrt ((Q.1 - P₁.1)^2 + (Q.2 - P₁.2)^2) = 4 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_doubling_points_theorem_l3573_357378


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l3573_357399

/-- The curve function -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The derivative of the curve function -/
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + a

/-- The tangent line function -/
def tangent_line (x y : ℝ) : ℝ := x - y + 1

theorem tangent_line_at_zero (a b : ℝ) :
  (∀ x y, y = f a b x → tangent_line x y = 0 → x = 0 ∧ y = b) →
  (f' a 0 = -1) →
  a = -1 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l3573_357399


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l3573_357343

theorem half_abs_diff_squares_20_15 : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l3573_357343


namespace NUMINAMATH_CALUDE_line_circle_properties_l3573_357323

-- Define the line l and circle C
def line_l (m x y : ℝ) : Prop := (m + 2) * x + (1 - 2 * m) * y + 4 * m - 2 = 0
def circle_C (x y : ℝ) : Prop := x^2 - 2 * x + y^2 = 0

-- Define the intersection points M and N
def intersect_points (m : ℝ) : Prop := ∃ x_M y_M x_N y_N : ℝ,
  line_l m x_M y_M ∧ circle_C x_M y_M ∧
  line_l m x_N y_N ∧ circle_C x_N y_N ∧
  (x_M ≠ x_N ∨ y_M ≠ y_N)

-- Define the slopes of OM and ON
def slope_OM_ON (m : ℝ) : Prop := ∃ k₁ k₂ x_M y_M x_N y_N : ℝ,
  line_l m x_M y_M ∧ circle_C x_M y_M ∧
  line_l m x_N y_N ∧ circle_C x_N y_N ∧
  k₁ = y_M / x_M ∧ k₂ = y_N / x_N

-- Theorem statement
theorem line_circle_properties :
  (∀ m : ℝ, line_l m 0 2) ∧
  (∀ m : ℝ, intersect_points m → -(m + 2) / (1 - 2 * m) < -3/4) ∧
  (∀ m : ℝ, slope_OM_ON m → ∃ k₁ k₂ : ℝ, k₁ + k₂ = 1) :=
sorry

end NUMINAMATH_CALUDE_line_circle_properties_l3573_357323


namespace NUMINAMATH_CALUDE_swan_percentage_among_non_ducks_l3573_357357

theorem swan_percentage_among_non_ducks (total : ℝ) (ducks swans herons geese : ℝ) :
  total = 100 →
  ducks = 35 →
  swans = 30 →
  herons = 20 →
  geese = 15 →
  (swans / (total - ducks)) * 100 = 46.15 := by
  sorry

end NUMINAMATH_CALUDE_swan_percentage_among_non_ducks_l3573_357357


namespace NUMINAMATH_CALUDE_two_digit_integer_problem_l3573_357385

theorem two_digit_integer_problem (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧ 
  10 ≤ n ∧ n < 100 ∧ 
  m ≠ n ∧
  (m + n) / 2 = (m : ℚ) + n / 100 →
  min m n = 32 := by
sorry

end NUMINAMATH_CALUDE_two_digit_integer_problem_l3573_357385


namespace NUMINAMATH_CALUDE_a_minus_b_eq_neg_seven_l3573_357342

theorem a_minus_b_eq_neg_seven
  (h1 : Real.sqrt (a ^ 2) = 3)
  (h2 : Real.sqrt b = 2)
  (h3 : a * b < 0)
  : a - b = -7 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_eq_neg_seven_l3573_357342


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3573_357320

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h : (Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 2) : 
  Real.tan (α + π / 4) = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3573_357320


namespace NUMINAMATH_CALUDE_smallest_number_l3573_357364

theorem smallest_number (S : Set ℤ) (h : S = {-3, 2, -2, 0}) : 
  ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3573_357364


namespace NUMINAMATH_CALUDE_adults_fed_is_eight_l3573_357356

/-- Represents the number of adults that can be fed with one can of soup -/
def adults_per_can : ℕ := 4

/-- Represents the number of children that can be fed with one can of soup -/
def children_per_can : ℕ := 6

/-- Represents the total number of cans available -/
def total_cans : ℕ := 8

/-- Represents the number of children fed -/
def children_fed : ℕ := 24

/-- Calculates the number of adults that can be fed with the remaining soup -/
def adults_fed : ℕ :=
  let cans_used_for_children := children_fed / children_per_can
  let remaining_cans := total_cans - cans_used_for_children
  let usable_cans := remaining_cans / 2
  usable_cans * adults_per_can

theorem adults_fed_is_eight : adults_fed = 8 := by
  sorry

end NUMINAMATH_CALUDE_adults_fed_is_eight_l3573_357356


namespace NUMINAMATH_CALUDE_min_y_value_l3573_357390

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 26*y) : 
  ∃ (y_min : ℝ), y_min = 13 - Real.sqrt 269 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 20*x' + 26*y' → y' ≥ y_min := by
sorry

end NUMINAMATH_CALUDE_min_y_value_l3573_357390


namespace NUMINAMATH_CALUDE_range_of_f_l3573_357338

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.range f, -8 ≤ y ∧ y ≤ 8 ∧
  ∀ z, -8 ≤ z ∧ z ≤ 8 → ∃ x, f x = z :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l3573_357338


namespace NUMINAMATH_CALUDE_m_range_l3573_357369

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → m ∈ Set.Icc (-3) 5 := by
sorry

end NUMINAMATH_CALUDE_m_range_l3573_357369


namespace NUMINAMATH_CALUDE_max_a_value_l3573_357319

theorem max_a_value (a b : ℕ) (h : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) :
  a ≤ 20 ∧ ∃ b₀ : ℕ, 5 * Nat.lcm 20 b₀ + 2 * Nat.gcd 20 b₀ = 120 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_l3573_357319


namespace NUMINAMATH_CALUDE_complement_of_A_l3573_357340

def U : Set ℕ := {x | 0 ≤ x ∧ x < 10}

def A : Set ℕ := {2, 4, 6, 8}

theorem complement_of_A : U \ A = {1, 3, 5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3573_357340
