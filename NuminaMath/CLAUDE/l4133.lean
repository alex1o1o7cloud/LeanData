import Mathlib

namespace min_ratio_four_digit_number_l4133_413342

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest value of n/s_n for four-digit numbers is 1099/19 -/
theorem min_ratio_four_digit_number :
  ∀ n : ℕ, 1000 ≤ n → n ≤ 9999 → (n : ℚ) / (sum_of_digits n) ≥ 1099 / 19 := by
  sorry

end min_ratio_four_digit_number_l4133_413342


namespace sumata_family_vacation_miles_l4133_413305

/-- Proves that given a 5-day vacation with a total of 1250 miles driven, the average miles driven per day is 250 miles. -/
theorem sumata_family_vacation_miles (total_miles : ℕ) (num_days : ℕ) (miles_per_day : ℕ) :
  total_miles = 1250 ∧ num_days = 5 ∧ miles_per_day = total_miles / num_days →
  miles_per_day = 250 :=
by sorry

end sumata_family_vacation_miles_l4133_413305


namespace plums_added_l4133_413398

def initial_plums : ℕ := 17
def final_plums : ℕ := 21

theorem plums_added (initial : ℕ) (final : ℕ) (added : ℕ) 
  (h1 : initial = initial_plums) 
  (h2 : final = final_plums) 
  (h3 : final = initial + added) : 
  added = final - initial :=
by
  sorry

end plums_added_l4133_413398


namespace quadratic_polynomial_with_complex_root_l4133_413386

theorem quadratic_polynomial_with_complex_root : 
  ∃ (a b c : ℝ), 
    (a = 3 ∧ 
     (Complex.I : ℂ)^2 = -1 ∧
     (3 : ℂ) * ((4 + 2 * Complex.I) ^ 2 - 8 * (4 + 2 * Complex.I) + 16 + 4) = 3 * (Complex.I : ℂ)^2 + b * (Complex.I : ℂ) + c) :=
by sorry

end quadratic_polynomial_with_complex_root_l4133_413386


namespace triangle_special_angle_l4133_413391

/-- Given a triangle ABC where b = c and a² = 2b²(1 - sin A), prove that A = π/4 -/
theorem triangle_special_angle (a b c : ℝ) (A : ℝ) 
  (h1 : b = c) 
  (h2 : a^2 = 2 * b^2 * (1 - Real.sin A)) : 
  A = π/4 := by
  sorry

end triangle_special_angle_l4133_413391


namespace ounces_per_cup_l4133_413378

theorem ounces_per_cup (total_ounces : ℕ) (total_cups : ℕ) (h1 : total_ounces = 264) (h2 : total_cups = 33) :
  total_ounces / total_cups = 8 := by
sorry

end ounces_per_cup_l4133_413378


namespace product_72_difference_sum_l4133_413321

theorem product_72_difference_sum (P Q R S : ℕ+) : 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
  P * Q = 72 →
  R * S = 72 →
  (P : ℤ) - (Q : ℤ) = (R : ℤ) + (S : ℤ) →
  P = 18 :=
by sorry

end product_72_difference_sum_l4133_413321


namespace symmetry_and_evenness_l4133_413310

def symmetric_wrt_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (|x|) = f (-|x|)

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem symmetry_and_evenness (f : ℝ → ℝ) :
  (even_function f → symmetric_wrt_y_axis f) ∧
  ∃ g : ℝ → ℝ, symmetric_wrt_y_axis g ∧ ¬even_function g :=
sorry

end symmetry_and_evenness_l4133_413310


namespace negative_abs_negative_five_l4133_413301

theorem negative_abs_negative_five : -|-5| = -5 := by
  sorry

end negative_abs_negative_five_l4133_413301


namespace max_person_money_100_2000_380_l4133_413389

/-- Given a group of people and their money distribution, 
    calculate the maximum amount one person can have. -/
def maxPersonMoney (n : ℕ) (total : ℕ) (maxTen : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum amount one person can have 
    under the given conditions. -/
theorem max_person_money_100_2000_380 : 
  maxPersonMoney 100 2000 380 = 218 := by sorry

end max_person_money_100_2000_380_l4133_413389


namespace min_sum_squares_l4133_413328

theorem min_sum_squares (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 1) :
  ∃ (m : ℝ), ∀ (x y : ℝ), 0 < x → 0 < y → 2 * x + y = 1 → x^2 + y^2 ≥ m ∧ (∃ (u v : ℝ), 0 < u ∧ 0 < v ∧ 2 * u + v = 1 ∧ u^2 + v^2 = m) ∧ m = 1/5 :=
sorry

end min_sum_squares_l4133_413328


namespace similar_triangles_shortest_side_l4133_413308

theorem similar_triangles_shortest_side 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a = 24) 
  (h3 : c = 37) 
  (h4 : ∃ k, k > 0 ∧ k * c = 74) : 
  ∃ x, x > 0 ∧ x^2 = 793 ∧ 2 * x = min (2 * a) (2 * b) := by
sorry

end similar_triangles_shortest_side_l4133_413308


namespace evaluate_expression_l4133_413399

theorem evaluate_expression : 3000 * (3000 ^ 2999) ^ 2 = 3000 ^ 5999 := by
  sorry

end evaluate_expression_l4133_413399


namespace triangle_power_equality_l4133_413393

theorem triangle_power_equality (a b c : ℝ) 
  (h : ∀ n : ℕ, (a^n + b^n > c^n) ∧ (b^n + c^n > a^n) ∧ (c^n + a^n > b^n)) :
  (a = b) ∨ (b = c) ∨ (c = a) := by
sorry

end triangle_power_equality_l4133_413393


namespace division_and_subtraction_l4133_413384

theorem division_and_subtraction : (12 / (1/12)) - 5 = 139 := by
  sorry

end division_and_subtraction_l4133_413384


namespace rectangle_area_l4133_413377

/-- Represents a rectangle with given properties -/
structure Rectangle where
  breadth : ℝ
  length : ℝ
  perimeter : ℝ

/-- Theorem: Area of a specific rectangle -/
theorem rectangle_area (r : Rectangle) 
  (h1 : r.length = 3 * r.breadth) 
  (h2 : r.perimeter = 88) : 
  r.length * r.breadth = 363 := by
  sorry

#check rectangle_area

end rectangle_area_l4133_413377


namespace book_pages_calculation_l4133_413349

-- Define the number of pages read per night
def pages_per_night : ℝ := 120.0

-- Define the number of days of reading
def days_of_reading : ℝ := 10.0

-- Define the total number of pages in the book
def total_pages : ℝ := pages_per_night * days_of_reading

-- Theorem statement
theorem book_pages_calculation : total_pages = 1200.0 := by
  sorry

end book_pages_calculation_l4133_413349


namespace camera_price_theorem_l4133_413359

/-- The sticker price of the camera -/
def sticker_price : ℝ := 666.67

/-- The price at Store X after discount and rebate -/
def store_x_price (p : ℝ) : ℝ := 0.80 * p - 50

/-- The price at Store Y after discount -/
def store_y_price (p : ℝ) : ℝ := 0.65 * p

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem camera_price_theorem : 
  store_y_price sticker_price - store_x_price sticker_price = 40 := by
  sorry


end camera_price_theorem_l4133_413359


namespace power_calculation_l4133_413304

theorem power_calculation : 2^24 / 16^3 * 2^4 = 65536 := by
  sorry

end power_calculation_l4133_413304


namespace x_cubed_remainder_l4133_413346

theorem x_cubed_remainder (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^3 ≡ 8 [ZMOD 25] := by
  sorry

end x_cubed_remainder_l4133_413346


namespace manuscript_cost_theorem_l4133_413394

/-- Represents the cost calculation for manuscript printing and binding --/
def manuscript_cost (
  num_copies : ℕ
  ) (
  total_pages : ℕ
  ) (
  color_pages : ℕ
  ) (
  bw_cost : ℚ
  ) (
  color_cost : ℚ
  ) (
  binding_cost : ℚ
  ) (
  index_cost : ℚ
  ) (
  rush_copies : ℕ
  ) (
  rush_cost : ℚ
  ) (
  binding_discount_rate : ℚ
  ) (
  bundle_discount : ℚ
  ) : ℚ :=
  let bw_pages := total_pages - color_pages
  let print_cost := (bw_pages : ℚ) * bw_cost + (color_pages : ℚ) * color_cost
  let additional_cost := binding_cost + index_cost - bundle_discount
  let copy_cost := print_cost + additional_cost
  let total_before_discount := (num_copies : ℚ) * copy_cost
  let binding_discount := (num_copies : ℚ) * binding_cost * binding_discount_rate
  let rush_fee := (rush_copies : ℚ) * rush_cost
  total_before_discount - binding_discount + rush_fee

/-- Theorem stating the total cost for the manuscript printing and binding --/
theorem manuscript_cost_theorem :
  manuscript_cost 10 400 50 (5/100) (1/10) 5 2 5 3 (1/10) (1/2) = 300 :=
by sorry

end manuscript_cost_theorem_l4133_413394


namespace park_legs_count_l4133_413300

/-- Calculate the total number of legs for given numbers of dogs, cats, birds, and spiders -/
def totalLegs (dogs cats birds spiders : ℕ) : ℕ :=
  4 * dogs + 4 * cats + 2 * birds + 8 * spiders

/-- Theorem stating that the total number of legs for 109 dogs, 37 cats, 52 birds, and 19 spiders is 840 -/
theorem park_legs_count : totalLegs 109 37 52 19 = 840 := by
  sorry

end park_legs_count_l4133_413300


namespace tangent_product_upper_bound_l4133_413358

theorem tangent_product_upper_bound (α β : Real) 
  (sum_eq : α + β = Real.pi / 3)
  (α_pos : α > 0)
  (β_pos : β > 0) :
  Real.tan α * Real.tan β ≤ 1 / 3 := by
  sorry

end tangent_product_upper_bound_l4133_413358


namespace circles_externally_tangent_l4133_413376

-- Define the circles
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 1}

-- Define the centers of the circles
def center₁ : ℝ × ℝ := (0, 0)
def center₂ : ℝ × ℝ := (2, 0)

-- Define the radii of the circles
def radius₁ : ℝ := 1
def radius₂ : ℝ := 1

-- Theorem statement
theorem circles_externally_tangent :
  let d := Real.sqrt ((center₂.1 - center₁.1)^2 + (center₂.2 - center₁.2)^2)
  d = radius₁ + radius₂ := by sorry

end circles_externally_tangent_l4133_413376


namespace cards_bought_equals_difference_l4133_413352

/-- The number of baseball cards Sam bought is equal to the difference between
    Mike's initial number of cards and his current number of cards. -/
theorem cards_bought_equals_difference (initial_cards current_cards cards_bought : ℕ) :
  initial_cards = 87 →
  current_cards = 74 →
  cards_bought = initial_cards - current_cards →
  cards_bought = 13 := by
  sorry

end cards_bought_equals_difference_l4133_413352


namespace locus_of_point_P_l4133_413307

/-- Given two rays OA and OB, and a point P inside the angle AOx, prove the equation of the locus of P and its domain --/
theorem locus_of_point_P (k : ℝ) (h_k : k > 0) :
  ∃ (f : ℝ → ℝ) (domain : Set ℝ),
    (∀ x y, (y = k * x ∧ x > 0) → (y = f x → x ∈ domain)) ∧
    (∀ x y, (y = -k * x ∧ x > 0) → (y = f x → x ∈ domain)) ∧
    (∀ x y, x ∈ domain → 0 < y ∧ y < k * x ∧ y < (1/k) * x) ∧
    (∀ x y, x ∈ domain → y = f x → y = Real.sqrt (x^2 - (1 + k^2))) ∧
    (0 < k ∧ k < 1 →
      domain = {x | Real.sqrt (k^2 + 1) < x ∧ x < Real.sqrt ((k^2 + 1)/(1 - k^2))}) ∧
    (k = 1 →
      domain = {x | Real.sqrt 2 < x}) ∧
    (k > 1 →
      domain = {x | Real.sqrt (k^2 + 1) < x ∧ x < k * Real.sqrt ((k^2 + 1)/(k^2 - 1))}) :=
by sorry

end locus_of_point_P_l4133_413307


namespace batsman_highest_score_l4133_413347

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (overall_average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (i : overall_average = 63)
  (j : score_difference = 150)
  (k : average_excluding_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    (total_innings : ℚ) * overall_average = 
      ((total_innings - 2 : ℕ) : ℚ) * average_excluding_extremes + highest_score + lowest_score ∧
    highest_score = 248 := by
  sorry

end batsman_highest_score_l4133_413347


namespace sphere_volume_from_surface_area_l4133_413317

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    (4 * Real.pi * r^2 = 144 * Real.pi) →
    ((4 / 3) * Real.pi * r^3 = 288 * Real.pi) := by
  sorry

end sphere_volume_from_surface_area_l4133_413317


namespace faucet_turning_is_rotational_motion_l4133_413383

/-- A motion that involves revolving around a center and changing direction -/
structure FaucetTurning where
  revolves_around_center : Bool
  direction_changes : Bool

/-- Definition of rotational motion -/
def is_rotational_motion (motion : FaucetTurning) : Prop :=
  motion.revolves_around_center ∧ motion.direction_changes

/-- Theorem: Turning a faucet by hand is a rotational motion -/
theorem faucet_turning_is_rotational_motion :
  ∀ (faucet_turning : FaucetTurning),
  faucet_turning.revolves_around_center = true →
  faucet_turning.direction_changes = true →
  is_rotational_motion faucet_turning :=
by
  sorry

end faucet_turning_is_rotational_motion_l4133_413383


namespace digit_150_of_1_13_l4133_413361

def decimal_representation_1_13 : List ℕ := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_13 : 
  (decimal_representation_1_13[(150 - 1) % decimal_representation_1_13.length] = 3) := by
  sorry

end digit_150_of_1_13_l4133_413361


namespace box_volume_increase_l4133_413367

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + w * h + h * l) = 1800)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := by sorry

end box_volume_increase_l4133_413367


namespace sports_enthusiasts_l4133_413370

theorem sports_enthusiasts (I A B : Finset ℕ) : 
  Finset.card I = 100 → 
  Finset.card A = 63 → 
  Finset.card B = 75 → 
  38 ≤ Finset.card (A ∩ B) ∧ Finset.card (A ∩ B) ≤ 63 := by
  sorry

end sports_enthusiasts_l4133_413370


namespace john_sneezing_fit_duration_l4133_413382

/-- Calculates the duration of a sneezing fit given the time between sneezes and the number of sneezes. -/
def sneezingFitDuration (timeBetweenSneezes : ℕ) (numberOfSneezes : ℕ) : ℕ :=
  timeBetweenSneezes * numberOfSneezes

/-- Proves that a sneezing fit with 3 seconds between sneezes and 40 sneezes lasts 120 seconds. -/
theorem john_sneezing_fit_duration :
  sneezingFitDuration 3 40 = 120 := by
  sorry

#eval sneezingFitDuration 3 40

end john_sneezing_fit_duration_l4133_413382


namespace angle_C_measure_l4133_413348

theorem angle_C_measure (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Given conditions
  (a = 2 * Real.sqrt 6) →
  (b = 6) →
  (Real.cos B = -1/2) →
  -- Conclusion
  C = π/12 := by
  sorry

end angle_C_measure_l4133_413348


namespace stating_spinner_points_east_l4133_413363

/-- Represents the four cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a clockwise rotation in revolutions --/
def clockwise_rotation : ℚ := 7/2

/-- Represents a counterclockwise rotation in revolutions --/
def counterclockwise_rotation : ℚ := 11/4

/-- Represents the initial direction of the spinner --/
def initial_direction : Direction := Direction.South

/-- 
  Theorem stating that after the given rotations, 
  the spinner will point east
--/
theorem spinner_points_east : 
  ∃ (final_direction : Direction),
    final_direction = Direction.East :=
by sorry

end stating_spinner_points_east_l4133_413363


namespace quadratic_form_minimum_l4133_413302

theorem quadratic_form_minimum (x y : ℝ) : x^2 + 4*x*y + 5*y^2 - 8*x - 6*y ≥ -9/5 := by
  sorry

end quadratic_form_minimum_l4133_413302


namespace solution_set_implies_a_and_b_l4133_413327

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x + 3

-- Define the theorem
theorem solution_set_implies_a_and_b :
  ∀ a b : ℝ, 
  (∀ x : ℝ, f a x > 0 ↔ b < x ∧ x < 1) →
  (a = -7 ∧ b = -3/7) := by
sorry

-- Note: The second part of the problem is not included in the Lean statement
-- as it relies on the solution of the first part, which should not be assumed
-- in the theorem statement according to the given criteria.

end solution_set_implies_a_and_b_l4133_413327


namespace problem_statement_l4133_413371

theorem problem_statement : 
  (∃ x : ℝ, x - 2 > 0) ∧ ¬(∀ x : ℝ, x ≥ 0 → Real.sqrt x < x) := by
  sorry

end problem_statement_l4133_413371


namespace remainder_452867_div_9_l4133_413381

theorem remainder_452867_div_9 : 452867 % 9 = 5 := by
  sorry

end remainder_452867_div_9_l4133_413381


namespace exists_set_equal_partitions_l4133_413373

/-- The type of positive integers -/
def PositiveInt : Type := { n : ℕ // n > 0 }

/-- Count of partitions where each number appears at most twice -/
def countLimitedPartitions (n : ℕ) : ℕ :=
  sorry

/-- Count of partitions using elements from a set -/
def countSetPartitions (n : ℕ) (S : Set PositiveInt) : ℕ :=
  sorry

/-- The existence of a set S satisfying the partition property -/
theorem exists_set_equal_partitions :
  ∃ (S : Set PositiveInt), ∀ (n : ℕ), n > 0 →
    countLimitedPartitions n = countSetPartitions n S :=
  sorry

end exists_set_equal_partitions_l4133_413373


namespace total_cuts_after_six_operations_l4133_413365

def cuts_in_operation (n : ℕ) : ℕ :=
  3 * 4^(n - 1)

def total_cuts (n : ℕ) : ℕ :=
  (List.range n).map (cuts_in_operation ∘ (· + 1)) |> List.sum

theorem total_cuts_after_six_operations :
  total_cuts 6 = 4095 := by
  sorry

end total_cuts_after_six_operations_l4133_413365


namespace sibling_age_sum_l4133_413354

/-- Given the ages and age differences of three siblings, prove the sum of the youngest and oldest siblings' ages. -/
theorem sibling_age_sum (juliet maggie ralph : ℕ) : 
  juliet = 10 ∧ 
  juliet = maggie + 3 ∧ 
  ralph = juliet + 2 → 
  maggie + ralph = 19 := by
  sorry

end sibling_age_sum_l4133_413354


namespace lomonosov_card_puzzle_l4133_413325

theorem lomonosov_card_puzzle :
  ∃ (L O M N C B : ℕ),
    L ≠ O ∧ L ≠ M ∧ L ≠ N ∧ L ≠ C ∧ L ≠ B ∧
    O ≠ M ∧ O ≠ N ∧ O ≠ C ∧ O ≠ B ∧
    M ≠ N ∧ M ≠ C ∧ M ≠ B ∧
    N ≠ C ∧ N ≠ B ∧
    C ≠ B ∧
    L < 10 ∧ O < 10 ∧ M < 10 ∧ N < 10 ∧ C < 10 ∧ B < 10 ∧
    O < M ∧ O < C ∧
    L + O / M + O + N + O / C = 10 * O + B :=
by sorry

end lomonosov_card_puzzle_l4133_413325


namespace triangle_transformation_indefinite_l4133_413364

/-- A triangle can undergo the given transformation indefinitely iff it's equilateral -/
theorem triangle_transformation_indefinite (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  (∀ n : ℕ, ∃ a' b' c' : ℝ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    a' = (-a + b + c) / 2 ∧ 
    b' = (a - b + c) / 2 ∧ 
    c' = (a + b - c) / 2) ↔ 
  (a = b ∧ b = c) :=
by sorry

end triangle_transformation_indefinite_l4133_413364


namespace school_year_length_school_year_weeks_l4133_413303

theorem school_year_length 
  (num_children : ℕ) 
  (juice_boxes_per_child_per_day : ℕ) 
  (days_per_week : ℕ) 
  (total_juice_boxes : ℕ) : ℕ :=
  let juice_boxes_per_week := num_children * juice_boxes_per_child_per_day * days_per_week
  total_juice_boxes / juice_boxes_per_week

theorem school_year_weeks 
  (num_children : ℕ) 
  (juice_boxes_per_child_per_day : ℕ) 
  (days_per_week : ℕ) 
  (total_juice_boxes : ℕ) :
  school_year_length num_children juice_boxes_per_child_per_day days_per_week total_juice_boxes = 25 :=
by
  have h1 : num_children = 3 := by sorry
  have h2 : juice_boxes_per_child_per_day = 1 := by sorry
  have h3 : days_per_week = 5 := by sorry
  have h4 : total_juice_boxes = 375 := by sorry
  sorry

end school_year_length_school_year_weeks_l4133_413303


namespace simplify_expression_l4133_413319

theorem simplify_expression (x : ℝ) : (5 - 2*x) - (7 + 3*x) = -2 - 5*x := by
  sorry

end simplify_expression_l4133_413319


namespace value_of_a_l4133_413356

theorem value_of_a (a : ℚ) : a + (2 * a / 5) = 9 / 5 → a = 9 / 7 := by
  sorry

end value_of_a_l4133_413356


namespace book_arrangement_count_l4133_413326

def num_arrangements (n_pushkin n_tarle : ℕ) : ℕ :=
  3 * (Nat.factorial 2) * (Nat.factorial 4)

theorem book_arrangement_count :
  num_arrangements 2 4 = 144 := by
  sorry

end book_arrangement_count_l4133_413326


namespace exists_composite_prime_product_plus_one_l4133_413332

/-- pₖ denotes the k-th prime number -/
def nth_prime (k : ℕ) : ℕ := sorry

/-- Product of first n prime numbers plus 1 -/
def prime_product_plus_one (n : ℕ) : ℕ := 
  (List.range n).foldl (λ acc i => acc * nth_prime (i + 1)) 1 + 1

theorem exists_composite_prime_product_plus_one :
  ∃ n : ℕ, ¬ Nat.Prime (prime_product_plus_one n) := by sorry

end exists_composite_prime_product_plus_one_l4133_413332


namespace cubic_not_decreasing_param_range_l4133_413312

/-- Given a cubic function that is not strictly decreasing, prove the range of its parameter. -/
theorem cubic_not_decreasing_param_range (b : ℝ) : 
  (∃ x y : ℝ, x < y ∧ (-x^3 + b*x^2 - (2*b + 3)*x + 2 - b) ≤ (-y^3 + b*y^2 - (2*b + 3)*y + 2 - b)) →
  (b < -1 ∨ b > 3) := by
  sorry

end cubic_not_decreasing_param_range_l4133_413312


namespace nates_scallop_cost_l4133_413334

/-- Calculates the cost of scallops for a dinner party. -/
def scallop_cost (scallops_per_pound : ℕ) (cost_per_pound : ℚ) 
                 (scallops_per_person : ℕ) (num_people : ℕ) : ℚ :=
  let total_scallops := num_people * scallops_per_person
  let pounds_needed := total_scallops / scallops_per_pound
  pounds_needed * cost_per_pound

/-- The cost of scallops for Nate's dinner party is $48.00. -/
theorem nates_scallop_cost :
  scallop_cost 8 24 2 8 = 48 := by
  sorry

end nates_scallop_cost_l4133_413334


namespace parabola_shift_theorem_l4133_413330

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 2x^2 - 1 -/
def original_parabola : Parabola :=
  { a := 2, b := 0, c := -1 }

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h_shift : ℝ) (v_shift : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h_shift + p.b
    c := p.a * h_shift^2 - p.b * h_shift + p.c + v_shift }

theorem parabola_shift_theorem :
  let shifted := shift_parabola original_parabola 1 (-2)
  shifted.a = 2 ∧ shifted.b = 4 ∧ shifted.c = -3 := by sorry

end parabola_shift_theorem_l4133_413330


namespace second_derivative_sin_plus_cos_l4133_413368

open Real

theorem second_derivative_sin_plus_cos :
  let f : ℝ → ℝ := fun x ↦ sin x + cos x
  ∀ x : ℝ, (deriv^[2] f) x = -(cos x) - sin x := by
  sorry

end second_derivative_sin_plus_cos_l4133_413368


namespace arccos_one_over_sqrt_two_l4133_413336

theorem arccos_one_over_sqrt_two (π : Real) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end arccos_one_over_sqrt_two_l4133_413336


namespace expression_evaluation_l4133_413339

theorem expression_evaluation : 
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 := by
  sorry

end expression_evaluation_l4133_413339


namespace sin_15_30_75_product_l4133_413353

theorem sin_15_30_75_product : Real.sin (15 * π / 180) * Real.sin (30 * π / 180) * Real.sin (75 * π / 180) = 1 / 8 := by
  sorry

end sin_15_30_75_product_l4133_413353


namespace slope_angle_expression_l4133_413387

theorem slope_angle_expression (x y : ℝ) (α : ℝ) : 
  (6 * x - 2 * y - 5 = 0) →
  (Real.tan α = 3) →
  ((Real.sin (π - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (π + α)) = -2) := by
  sorry

end slope_angle_expression_l4133_413387


namespace correct_calculation_l4133_413395

theorem correct_calculation (x : ℤ) (h : x - 2 = 5) : x + 2 = 9 := by
  sorry

end correct_calculation_l4133_413395


namespace salt_added_amount_l4133_413390

/-- Represents the salt solution problem --/
structure SaltSolution where
  initial_volume : ℝ
  initial_salt_concentration : ℝ
  evaporation_fraction : ℝ
  water_added : ℝ
  final_salt_concentration : ℝ

/-- Calculates the amount of salt added to the solution --/
def salt_added (s : SaltSolution) : ℝ :=
  let initial_salt := s.initial_volume * s.initial_salt_concentration
  let water_evaporated := s.initial_volume * s.evaporation_fraction
  let remaining_volume := s.initial_volume - water_evaporated
  let new_volume := remaining_volume + s.water_added
  let final_salt := new_volume * s.final_salt_concentration
  final_salt - initial_salt

/-- The theorem stating the amount of salt added --/
theorem salt_added_amount (s : SaltSolution) 
  (h1 : s.initial_volume = 149.99999999999994)
  (h2 : s.initial_salt_concentration = 0.20)
  (h3 : s.evaporation_fraction = 0.25)
  (h4 : s.water_added = 10)
  (h5 : s.final_salt_concentration = 1/3) :
  ∃ ε > 0, |salt_added s - 10.83| < ε :=
sorry

end salt_added_amount_l4133_413390


namespace cookies_per_person_l4133_413366

theorem cookies_per_person (total_cookies : ℕ) (num_people : ℕ) 
  (h1 : total_cookies = 420) (h2 : num_people = 14) :
  total_cookies / num_people = 30 := by
  sorry

end cookies_per_person_l4133_413366


namespace geometric_sequence_fifth_term_l4133_413306

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n where a_3 = -1 and a_7 = -9, then a_5 = -3. -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_3 : a 3 = -1)
  (h_7 : a 7 = -9) :
  a 5 = -3 := by
  sorry

end geometric_sequence_fifth_term_l4133_413306


namespace simplify_expression_l4133_413388

theorem simplify_expression (x : ℝ) :
  3 * x^3 + 5 * x + 16 * x^2 + 15 - (7 - 3 * x^3 - 5 * x - 16 * x^2) = 
  6 * x^3 + 32 * x^2 + 10 * x + 8 :=
by sorry

end simplify_expression_l4133_413388


namespace range_of_g_l4133_413344

-- Define the functions
def f (x : ℝ) := x^2 - 7*x + 12
def g (x : ℝ) := x^2 - 7*x + 14

-- State the theorem
theorem range_of_g (x : ℝ) : 
  f x < 0 → ∃ y ∈ Set.Icc (1.75 : ℝ) 2, y = g x :=
sorry

end range_of_g_l4133_413344


namespace social_practice_choices_l4133_413338

/-- The number of classes in the first year of high school -/
def first_year_classes : Nat := 14

/-- The number of classes in the second year of high school -/
def second_year_classes : Nat := 14

/-- The number of classes in the third year of high school -/
def third_year_classes : Nat := 15

/-- The number of ways to choose students from 1 class to participate in social practice activities -/
def choose_one_class : Nat := first_year_classes + second_year_classes + third_year_classes

/-- The number of ways to choose students from one class in each grade to participate in social practice activities -/
def choose_one_from_each : Nat := first_year_classes * second_year_classes * third_year_classes

/-- The number of ways to choose students from 2 classes to participate in social practice activities, with the requirement that these 2 classes are from different grades -/
def choose_two_different_grades : Nat := 
  first_year_classes * second_year_classes + 
  first_year_classes * third_year_classes + 
  second_year_classes * third_year_classes

theorem social_practice_choices : 
  choose_one_class = 43 ∧ 
  choose_one_from_each = 2940 ∧ 
  choose_two_different_grades = 616 := by sorry

end social_practice_choices_l4133_413338


namespace parallel_line_through_point_l4133_413355

theorem parallel_line_through_point (x y : ℝ) : 
  let P : ℝ × ℝ := (0, 2)
  let L₁ : Set (ℝ × ℝ) := {(x, y) | 2 * x - y = 0}
  let L₂ : Set (ℝ × ℝ) := {(x, y) | 2 * x - y + 2 = 0}
  (P ∈ L₂) ∧ (∃ k : ℝ, k ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ L₁ ↔ (k * x, k * y) ∈ L₂) :=
by
  sorry

#check parallel_line_through_point

end parallel_line_through_point_l4133_413355


namespace smallest_truck_shipments_l4133_413309

theorem smallest_truck_shipments (B : ℕ) : 
  B ≥ 120 → 
  B % 5 = 0 → 
  ∃ (T : ℕ), T ≠ 5 ∧ T > 1 ∧ B % T = 0 ∧ 
  ∀ (S : ℕ), S ≠ 5 → S > 1 → B % S = 0 → T ≤ S :=
by sorry

end smallest_truck_shipments_l4133_413309


namespace min_sum_given_log_sum_l4133_413329

theorem min_sum_given_log_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.log a / Real.log 5 + Real.log b / Real.log 5 = 2) : 
  a + b ≥ 10 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
    Real.log x / Real.log 5 + Real.log y / Real.log 5 = 2 ∧ x + y = 10 := by
  sorry

end min_sum_given_log_sum_l4133_413329


namespace complex_power_2013_l4133_413375

def i : ℂ := Complex.I

theorem complex_power_2013 : ((1 + i) / (1 - i)) ^ 2013 = i := by sorry

end complex_power_2013_l4133_413375


namespace orange_seller_gain_percentage_l4133_413318

theorem orange_seller_gain_percentage
  (loss_rate : ℝ)
  (loss_quantity : ℝ)
  (gain_quantity : ℝ)
  (h_loss_rate : loss_rate = 0.04)
  (h_loss_quantity : loss_quantity = 16)
  (h_gain_quantity : gain_quantity = 12) :
  let cost_price := 1 / (1 - loss_rate)
  let gain_percentage := ((cost_price * gain_quantity) / (1 - loss_rate * cost_price) - 1) * 100
  gain_percentage = 28 := by
sorry

end orange_seller_gain_percentage_l4133_413318


namespace triangle_area_change_l4133_413320

theorem triangle_area_change (h : ℝ) (b₁ b₂ : ℝ) (a₁ a₂ : ℝ) :
  h = 8 ∧ b₁ = 16 ∧ b₂ = 5 ∧
  a₁ = 1/2 * b₁ * h ∧
  a₂ = 1/2 * b₂ * h →
  a₁ = 64 ∧ a₂ = 20 :=
by sorry

end triangle_area_change_l4133_413320


namespace fixed_point_parabola_l4133_413350

theorem fixed_point_parabola :
  ∀ (t : ℝ), 5 = 5 * (-1)^2 + 2 * t * (-1) - 5 * t := by sorry

end fixed_point_parabola_l4133_413350


namespace age_ratio_is_two_to_one_l4133_413372

def B_current_age : ℕ := 39
def A_current_age : ℕ := B_current_age + 9

def A_future_age : ℕ := A_current_age + 10
def B_past_age : ℕ := B_current_age - 10

theorem age_ratio_is_two_to_one :
  A_future_age / B_past_age = 2 ∧ B_past_age ≠ 0 :=
by sorry

end age_ratio_is_two_to_one_l4133_413372


namespace unique_solution_equation_l4133_413380

theorem unique_solution_equation (x : ℝ) :
  x ≥ 0 → (2021 * (x^2020)^(1/202) - 1 = 2020 * x) → x = 1 := by
  sorry

end unique_solution_equation_l4133_413380


namespace centroid_property_l4133_413322

/-- Definition of a point in 2D space -/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Definition of a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Calculate the centroid of a triangle -/
def centroid (t : Triangle) : Point2D :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

/-- The main theorem -/
theorem centroid_property :
  let t := Triangle.mk
    (Point2D.mk (-1) 4)
    (Point2D.mk 5 2)
    (Point2D.mk 3 10)
  let c := centroid t
  10 * c.x + c.y = 86 / 3 := by
  sorry

end centroid_property_l4133_413322


namespace album_ratio_proof_l4133_413369

/-- Prove that given the conditions, the ratio of Katrina's albums to Bridget's albums is 6:1 -/
theorem album_ratio_proof (miriam katrina bridget adele : ℕ) 
  (h1 : miriam = 5 * katrina)
  (h2 : ∃ n : ℕ, katrina = n * bridget)
  (h3 : bridget = adele - 15)
  (h4 : miriam + katrina + bridget + adele = 585)
  (h5 : adele = 30) :
  katrina / bridget = 6 := by
sorry

end album_ratio_proof_l4133_413369


namespace hillary_activities_lcm_l4133_413341

theorem hillary_activities_lcm : Nat.lcm 6 (Nat.lcm 4 (Nat.lcm 16 (Nat.lcm 12 8))) = 48 := by
  sorry

end hillary_activities_lcm_l4133_413341


namespace cell_chain_length_is_million_l4133_413314

/-- The length of a cell chain in nanometers -/
def cell_chain_length (cell_diameter : ℕ) (num_cells : ℕ) : ℕ :=
  cell_diameter * num_cells

/-- Theorem: The length of a cell chain is 10⁶ nanometers -/
theorem cell_chain_length_is_million :
  cell_chain_length 500 2000 = 1000000 := by
  sorry

end cell_chain_length_is_million_l4133_413314


namespace simplify_expression_l4133_413357

theorem simplify_expression : 5 * (14 / 3) * (9 / -42) = -5 := by
  sorry

end simplify_expression_l4133_413357


namespace max_quotient_is_21996_l4133_413340

def is_valid_divisor (d : ℕ) : Prop :=
  d ≥ 10 ∧ d < 100

def quotient_hundreds_condition (dividend : ℕ) (divisor : ℕ) : Prop :=
  let q := dividend / divisor
  (q / 100) * divisor ≥ 200 ∧ (q / 100) * divisor < 300

def max_quotient_dividend (dividends : List ℕ) : ℕ := sorry

theorem max_quotient_is_21996 :
  let dividends := [21944, 21996, 24054, 24111]
  ∃ d : ℕ, is_valid_divisor d ∧ 
           quotient_hundreds_condition (max_quotient_dividend dividends) d ∧
           max_quotient_dividend dividends = 21996 := by sorry

end max_quotient_is_21996_l4133_413340


namespace lucky_draw_probabilities_l4133_413337

def probability_wang_wins : ℝ := 0.4
def probability_zhang_wins : ℝ := 0.2

theorem lucky_draw_probabilities :
  let p_both_win := probability_wang_wins * probability_zhang_wins
  let p_only_one_wins := probability_wang_wins * (1 - probability_zhang_wins) + (1 - probability_wang_wins) * probability_zhang_wins
  let p_at_most_one_wins := 1 - p_both_win
  (p_both_win = 0.08) ∧
  (p_only_one_wins = 0.44) ∧
  (p_at_most_one_wins = 0.92) := by
  sorry

end lucky_draw_probabilities_l4133_413337


namespace min_value_sum_of_squares_over_sums_l4133_413360

theorem min_value_sum_of_squares_over_sums (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_sum : a + b + c = 9) : 
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) ≥ 9 := by
sorry

end min_value_sum_of_squares_over_sums_l4133_413360


namespace lineup_combinations_l4133_413345

/-- The number of ways to choose a starting lineup for a basketball team -/
def choose_lineup (total_players : ℕ) (center_players : ℕ) (point_guard_players : ℕ) : ℕ :=
  center_players * point_guard_players * (total_players - 2) * (total_players - 3) * (total_players - 4)

/-- Theorem stating the number of ways to choose a starting lineup -/
theorem lineup_combinations :
  choose_lineup 12 3 2 = 4320 :=
by sorry

end lineup_combinations_l4133_413345


namespace books_per_shelf_l4133_413343

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) (h1 : total_books = 12) (h2 : num_shelves = 3) :
  total_books / num_shelves = 4 := by
  sorry

end books_per_shelf_l4133_413343


namespace initial_money_calculation_l4133_413396

theorem initial_money_calculation (M : ℚ) : 
  (((M * (3/5) * (2/3) * (3/4) * (4/7)) : ℚ) = 700) → M = 24500/6 := by
  sorry

end initial_money_calculation_l4133_413396


namespace perpendicular_vectors_l4133_413313

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

theorem perpendicular_vectors (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) → k = -10/3 := by
  sorry

end perpendicular_vectors_l4133_413313


namespace f_is_even_and_decreasing_l4133_413392

def f (x : ℝ) : ℝ := -x^2 + abs x

theorem f_is_even_and_decreasing :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) :=
by sorry

end f_is_even_and_decreasing_l4133_413392


namespace fallen_cakes_ratio_l4133_413311

theorem fallen_cakes_ratio (total_cakes : ℕ) (destroyed_cakes : ℕ) : 
  total_cakes = 12 → 
  destroyed_cakes = 3 → 
  (2 * destroyed_cakes : ℚ) / total_cakes = 1 / 2 := by
  sorry

end fallen_cakes_ratio_l4133_413311


namespace union_of_sets_l4133_413315

theorem union_of_sets : 
  let A : Set ℕ := {1, 3}
  let B : Set ℕ := {3, 4}
  A ∪ B = {1, 3, 4} := by sorry

end union_of_sets_l4133_413315


namespace special_number_not_divisible_l4133_413379

/-- Represents a 70-digit number with specific digit frequency properties -/
def SpecialNumber := { n : ℕ // 
  (Nat.digits 10 n).length = 70 ∧ 
  (∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7] → (Nat.digits 10 n).count d = 10) ∧
  (∀ d : ℕ, d ∈ [8, 9, 0] → d ∉ (Nat.digits 10 n))
}

/-- Theorem stating that no SpecialNumber can divide another SpecialNumber -/
theorem special_number_not_divisible (n m : SpecialNumber) : ¬(n.val ∣ m.val) := by
  sorry

end special_number_not_divisible_l4133_413379


namespace reciprocal_problem_l4133_413397

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 4) : 200 * (1 / x) = 400 := by
  sorry

end reciprocal_problem_l4133_413397


namespace scientific_notation_of_1680000_l4133_413316

theorem scientific_notation_of_1680000 : 
  ∃ (a : ℝ) (n : ℤ), 1680000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.68 ∧ n = 6 := by
  sorry

end scientific_notation_of_1680000_l4133_413316


namespace inscribed_triangle_inequality_l4133_413335

/-- A triangle PQR inscribed in a semicircle with diameter PQ and R on the semicircle -/
structure InscribedTriangle where
  /-- The radius of the semicircle -/
  r : ℝ
  /-- Point P -/
  P : ℝ × ℝ
  /-- Point Q -/
  Q : ℝ × ℝ
  /-- Point R -/
  R : ℝ × ℝ
  /-- PQ is the diameter of the semicircle -/
  diameter : dist P Q = 2 * r
  /-- R is on the semicircle -/
  on_semicircle : dist P R = r ∨ dist Q R = r

/-- The sum of distances PR and QR -/
def t (triangle : InscribedTriangle) : ℝ :=
  dist triangle.P triangle.R + dist triangle.Q triangle.R

/-- Theorem: For all inscribed triangles, t^2 ≤ 8r^2 -/
theorem inscribed_triangle_inequality (triangle : InscribedTriangle) :
  (t triangle)^2 ≤ 8 * triangle.r^2 := by
  sorry

end inscribed_triangle_inequality_l4133_413335


namespace sum_mod_seven_l4133_413385

theorem sum_mod_seven : (8145 + 8146 + 8147 + 8148 + 8149) % 7 = 4 := by
  sorry

end sum_mod_seven_l4133_413385


namespace hyperbola_foci_l4133_413351

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(-Real.sqrt 5, 0), (Real.sqrt 5, 0)}

/-- Theorem: The foci of the hyperbola are at (±√5, 0) -/
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci_coordinates :=
by sorry

end hyperbola_foci_l4133_413351


namespace right_triangle_pythagorean_l4133_413374

theorem right_triangle_pythagorean (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → -- Ensuring positive lengths
  (a^2 + b^2 = c^2) → -- Pythagorean theorem
  ((a = 12 ∧ b = 5) → c = 13) ∧ -- Part 1
  ((c = 10 ∧ b = 9) → a = Real.sqrt 19) -- Part 2
  := by sorry

end right_triangle_pythagorean_l4133_413374


namespace quadratic_inequality_solution_l4133_413331

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ 1 < x ∧ x < 2) → b = 2 := by
  sorry

end quadratic_inequality_solution_l4133_413331


namespace remaining_statue_weight_l4133_413333

/-- Represents the weights of Hammond's statues and marble block -/
structure HammondStatues where
  initial_weight : ℝ
  first_statue : ℝ
  second_statue : ℝ
  discarded_marble : ℝ

/-- Theorem stating the weight of each remaining statue -/
theorem remaining_statue_weight (h : HammondStatues)
  (h_initial : h.initial_weight = 80)
  (h_first : h.first_statue = 10)
  (h_second : h.second_statue = 18)
  (h_discarded : h.discarded_marble = 22)
  (h_equal_remaining : ∃ x : ℝ, 
    h.initial_weight - h.discarded_marble - h.first_statue - h.second_statue = 2 * x) :
  ∃ x : ℝ, x = 15 ∧ 
    h.initial_weight - h.discarded_marble - h.first_statue - h.second_statue = 2 * x :=
by sorry

end remaining_statue_weight_l4133_413333


namespace lana_extra_flowers_l4133_413323

/-- The number of extra flowers Lana picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Theorem stating that Lana picked 280 extra flowers -/
theorem lana_extra_flowers :
  extra_flowers 860 920 1500 = 280 := by
  sorry

end lana_extra_flowers_l4133_413323


namespace sin_double_angle_plus_5pi_6_l4133_413324

theorem sin_double_angle_plus_5pi_6 (α : Real) 
  (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.sin (2 * α + 5 * π / 6) = 7 / 9 := by
  sorry

end sin_double_angle_plus_5pi_6_l4133_413324


namespace shaded_region_perimeter_l4133_413362

-- Define the circumference of each circle
def circle_circumference : ℝ := 48

-- Define the angle subtended by each arc (in degrees)
def arc_angle : ℝ := 90

-- Define the number of circles
def num_circles : ℕ := 3

-- Theorem statement
theorem shaded_region_perimeter :
  let arc_length := (arc_angle / 360) * circle_circumference
  (num_circles : ℝ) * arc_length = 36 := by sorry

end shaded_region_perimeter_l4133_413362
