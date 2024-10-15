import Mathlib

namespace NUMINAMATH_CALUDE_largest_integer_divisibility_l3033_303367

theorem largest_integer_divisibility : ∃ (n : ℕ), n = 14 ∧ 
  (∀ (m : ℕ), m > n → ¬(∃ (k : ℤ), (m - 2)^2 * (m + 1) = k * (2*m - 1))) ∧
  (∃ (k : ℤ), (n - 2)^2 * (n + 1) = k * (2*n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_divisibility_l3033_303367


namespace NUMINAMATH_CALUDE_line_equation_l3033_303310

/-- The distance between intersection points of x = k with y = x^2 + 4x + 4 and y = mx + b is 10 -/
def intersection_distance (m b k : ℝ) : Prop :=
  |k^2 + 4*k + 4 - (m*k + b)| = 10

/-- The line y = mx + b passes through the point (1, 6) -/
def passes_through_point (m b : ℝ) : Prop :=
  m * 1 + b = 6

theorem line_equation (m b : ℝ) (h1 : ∃ k, intersection_distance m b k)
    (h2 : passes_through_point m b) (h3 : b ≠ 0) :
    m = 4 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l3033_303310


namespace NUMINAMATH_CALUDE_triangle_inequality_l3033_303363

theorem triangle_inequality (a b c r R s : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ R > 0 ∧ s > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : s = (a + b + c) / 2)
  (h_inradius : r = (a * b * c) / (4 * s))
  (h_circumradius : R = (a * b * c) / (4 * area))
  (h_area : area = Real.sqrt (s * (s - a) * (s - b) * (s - c))) :
  (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≤ 
  (r / (16 * R * s) + s / (16 * R * r) + 11 / (8 * s)) ∧
  ((1 / (a + b) + 1 / (a + c) + 1 / (b + c) = 
    r / (16 * R * s) + s / (16 * R * r) + 11 / (8 * s)) ↔ 
   (a = b ∧ b = c)) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3033_303363


namespace NUMINAMATH_CALUDE_janet_movie_cost_l3033_303315

/-- Calculates the total cost of filming Janet's newest movie given the following conditions:
  * Janet's previous movie was 2 hours long
  * The new movie is 60% longer than the previous movie
  * The previous movie cost $50 per minute to film
  * The new movie cost twice as much per minute to film as the previous movie
-/
def total_cost_newest_movie (previous_movie_length : Real) 
                            (length_increase_percent : Real)
                            (previous_cost_per_minute : Real)
                            (new_cost_multiplier : Real) : Real :=
  let new_movie_length := previous_movie_length * (1 + length_increase_percent)
  let new_movie_length_minutes := new_movie_length * 60
  let new_cost_per_minute := previous_cost_per_minute * new_cost_multiplier
  new_movie_length_minutes * new_cost_per_minute

theorem janet_movie_cost :
  total_cost_newest_movie 2 0.6 50 2 = 19200 := by
  sorry

end NUMINAMATH_CALUDE_janet_movie_cost_l3033_303315


namespace NUMINAMATH_CALUDE_cookies_per_bag_l3033_303336

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 703) (h2 : num_bags = 37) :
  total_cookies / num_bags = 19 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l3033_303336


namespace NUMINAMATH_CALUDE_evaluate_expression_l3033_303391

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 2 * y^x = 533 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3033_303391


namespace NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l3033_303374

/-- Triangle with positive integer side lengths --/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Isosceles triangle where two sides are equal --/
def IsoscelesTriangle (t : IntegerTriangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Perimeter of a triangle --/
def Perimeter (t : IntegerTriangle) : ℕ :=
  t.a.val + t.b.val + t.c.val

/-- Angle bisector theorem relation --/
def AngleBisectorRelation (t : IntegerTriangle) (bisectorLength : ℕ+) : Prop :=
  ∃ (x y : ℕ+), x + y = t.c ∧ bisectorLength * t.c = t.a * y

/-- Main theorem --/
theorem smallest_perimeter_isosceles_triangle :
  ∀ (t : IntegerTriangle),
    IsoscelesTriangle t →
    AngleBisectorRelation t 8 →
    (∀ (t' : IntegerTriangle),
      IsoscelesTriangle t' →
      AngleBisectorRelation t' 8 →
      Perimeter t ≤ Perimeter t') →
    Perimeter t = 108 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l3033_303374


namespace NUMINAMATH_CALUDE_cloth_cost_calculation_l3033_303350

theorem cloth_cost_calculation (length : Real) (price_per_meter : Real) :
  length = 9.25 ∧ price_per_meter = 43 → length * price_per_meter = 397.75 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_calculation_l3033_303350


namespace NUMINAMATH_CALUDE_vase_discount_percentage_l3033_303392

theorem vase_discount_percentage 
  (original_price : ℝ) 
  (total_payment : ℝ) 
  (sales_tax_rate : ℝ) 
  (h1 : original_price = 200)
  (h2 : total_payment = 165)
  (h3 : sales_tax_rate = 0.1)
  : ∃ (discount_percentage : ℝ), 
    discount_percentage = 25 ∧ 
    total_payment = (original_price * (1 - discount_percentage / 100)) * (1 + sales_tax_rate) := by
  sorry

end NUMINAMATH_CALUDE_vase_discount_percentage_l3033_303392


namespace NUMINAMATH_CALUDE_prob_three_odd_dice_l3033_303300

/-- The number of dice being rolled -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 1/2

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 1/2

/-- The number of ways to choose 3 dice out of 4 -/
def choose_3_from_4 : ℕ := 4

theorem prob_three_odd_dice :
  (choose_3_from_4 : ℚ) * prob_odd^3 * prob_even^(num_dice - 3) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_dice_l3033_303300


namespace NUMINAMATH_CALUDE_product_b3_b17_l3033_303303

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem product_b3_b17 (a b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_geom : geometric_sequence b)
    (h_eq : 3 * a 1 - (a 8)^2 + 3 * a 15 = 0)
    (h_a8_b10 : a 8 = b 10) :
  b 3 * b 17 = 36 := by
sorry

end NUMINAMATH_CALUDE_product_b3_b17_l3033_303303


namespace NUMINAMATH_CALUDE_large_cube_surface_area_l3033_303334

-- Define the volume of a small cube
def small_cube_volume : ℝ := 512

-- Define the number of small cubes
def num_small_cubes : ℕ := 8

-- Define the function to calculate the side length of a cube given its volume
def side_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Define the function to calculate the surface area of a cube given its side length
def surface_area (side : ℝ) : ℝ := 6 * side^2

-- Theorem statement
theorem large_cube_surface_area :
  let small_side := side_length small_cube_volume
  let large_side := small_side * (num_small_cubes ^ (1/3))
  surface_area large_side = 1536 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_surface_area_l3033_303334


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3033_303383

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, m ≠ 0 ∧ n ≠ 0 →
  (m^2 + n) * (m + n^2) = (m - n)^3 →
  (m = -1 ∧ n = -1) ∨ (m = 8 ∧ n = -10) ∨ (m = 9 ∧ n = -6) ∨ (m = 9 ∧ n = -21) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3033_303383


namespace NUMINAMATH_CALUDE_medicine_box_theorem_l3033_303364

/-- Represents the number of tablets of each medicine type in a box -/
structure MedicineBox where
  tabletA : ℕ
  tabletB : ℕ

/-- Calculates the minimum number of tablets to extract to ensure at least two of each type -/
def minExtract (box : MedicineBox) : ℕ :=
  box.tabletA + 6

theorem medicine_box_theorem (box : MedicineBox) 
  (h1 : box.tabletA = 10)
  (h2 : minExtract box = 16) :
  box.tabletB ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_medicine_box_theorem_l3033_303364


namespace NUMINAMATH_CALUDE_cyclic_ratio_sum_geq_two_l3033_303362

theorem cyclic_ratio_sum_geq_two (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_ratio_sum_geq_two_l3033_303362


namespace NUMINAMATH_CALUDE_sum_consecutive_integers_n_plus_3_l3033_303387

theorem sum_consecutive_integers_n_plus_3 (n : ℕ) (h : n = 1) :
  (List.range (n + 3 + 1)).sum = ((n + 3) * (n + 4)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_integers_n_plus_3_l3033_303387


namespace NUMINAMATH_CALUDE_triangle_angle_sum_bound_l3033_303326

theorem triangle_angle_sum_bound (A B C : Real) (h_triangle : A + B + C = Real.pi)
  (h_sin_sum : Real.sin A + Real.sin B + Real.sin C ≤ 1) :
  min (A + B) (min (B + C) (C + A)) < Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_bound_l3033_303326


namespace NUMINAMATH_CALUDE_moses_extra_amount_l3033_303340

def total_amount : ℝ := 50
def moses_percentage : ℝ := 0.4

theorem moses_extra_amount :
  let moses_share := moses_percentage * total_amount
  let remainder := total_amount - moses_share
  let esther_share := remainder / 2
  moses_share - esther_share = 5 := by sorry

end NUMINAMATH_CALUDE_moses_extra_amount_l3033_303340


namespace NUMINAMATH_CALUDE_cricketer_average_score_l3033_303317

/-- Proves that the overall average score for a cricketer who played 7 matches
    with given averages for the first 4 and last 3 matches is 56. -/
theorem cricketer_average_score 
  (total_matches : ℕ)
  (first_matches : ℕ)
  (last_matches : ℕ)
  (first_average : ℚ)
  (last_average : ℚ)
  (h1 : total_matches = 7)
  (h2 : first_matches = 4)
  (h3 : last_matches = 3)
  (h4 : first_matches + last_matches = total_matches)
  (h5 : first_average = 46)
  (h6 : last_average = 69333333333333 / 1000000000000) : 
  (first_average * first_matches + last_average * last_matches) / total_matches = 56 := by
sorry

#eval (46 * 4 + 69333333333333 / 1000000000000 * 3) / 7

end NUMINAMATH_CALUDE_cricketer_average_score_l3033_303317


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l3033_303393

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 257 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 257 := by sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l3033_303393


namespace NUMINAMATH_CALUDE_largest_four_digit_multiple_of_3_and_5_l3033_303379

theorem largest_four_digit_multiple_of_3_and_5 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ 3 ∣ n ∧ 5 ∣ n → n ≤ 9990 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_multiple_of_3_and_5_l3033_303379


namespace NUMINAMATH_CALUDE_weight_of_substance_a_l3033_303388

/-- Given a mixture of substances a and b in the ratio 9:11 with a total weight,
    calculate the weight of substance a in the mixture. -/
theorem weight_of_substance_a (total_weight : ℝ) : 
  total_weight = 58.00000000000001 →
  (9 : ℝ) / (9 + 11) * total_weight = 26.1 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_substance_a_l3033_303388


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3033_303366

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, r > 0 ∧ a = 30 * r ∧ 7/4 = a * r) : a = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3033_303366


namespace NUMINAMATH_CALUDE_youseff_distance_to_office_l3033_303301

theorem youseff_distance_to_office (x : ℝ) 
  (walk_time : ℝ → ℝ) 
  (bike_time : ℝ → ℝ) 
  (h1 : ∀ d, walk_time d = d) 
  (h2 : ∀ d, bike_time d = d / 3) 
  (h3 : walk_time x = bike_time x + 14) : 
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_youseff_distance_to_office_l3033_303301


namespace NUMINAMATH_CALUDE_triangle_angle_side_inequality_l3033_303316

/-- Theorem: For any triangle, the weighted sum of angles divided by the sum of sides 
    is bounded between π/3 and π/2 -/
theorem triangle_angle_side_inequality (A B C a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- sides are positive
  a + b > c ∧ b + c > a ∧ c + a > b →  -- triangle inequality
  A + B + C = π →  -- sum of angles
  0 < A ∧ 0 < B ∧ 0 < C →  -- angles are positive
  π / 3 ≤ (A * a + B * b + C * c) / (a + b + c) ∧ 
  (A * a + B * b + C * c) / (a + b + c) < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_side_inequality_l3033_303316


namespace NUMINAMATH_CALUDE_number_of_pupils_is_40_l3033_303321

/-- The number of pupils in a class, given a specific mark entry error and its effect on the class average. -/
def number_of_pupils : ℕ :=
  let incorrect_mark : ℕ := 83
  let correct_mark : ℕ := 63
  let average_increase : ℚ := 1/2
  40

/-- Theorem stating that the number of pupils is 40 under the given conditions. -/
theorem number_of_pupils_is_40 :
  let n := number_of_pupils
  let incorrect_mark : ℕ := 83
  let correct_mark : ℕ := 63
  let mark_difference : ℕ := incorrect_mark - correct_mark
  let average_increase : ℚ := 1/2
  (mark_difference : ℚ) / n = average_increase → n = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_is_40_l3033_303321


namespace NUMINAMATH_CALUDE_unique_cube_prime_l3033_303377

theorem unique_cube_prime (n : ℕ) : (∃ p : ℕ, Nat.Prime p ∧ 2^n + n^2 + 25 = p^3) ↔ n = 6 :=
  sorry

end NUMINAMATH_CALUDE_unique_cube_prime_l3033_303377


namespace NUMINAMATH_CALUDE_savings_comparison_l3033_303332

theorem savings_comparison (last_year_salary : ℝ) (last_year_savings_rate : ℝ) 
  (salary_increase_rate : ℝ) (this_year_savings_rate : ℝ) 
  (h1 : last_year_savings_rate = 0.06)
  (h2 : salary_increase_rate = 0.20)
  (h3 : this_year_savings_rate = 0.05) :
  (this_year_savings_rate * (1 + salary_increase_rate) * last_year_salary) / 
  (last_year_savings_rate * last_year_salary) = 1 := by
sorry

end NUMINAMATH_CALUDE_savings_comparison_l3033_303332


namespace NUMINAMATH_CALUDE_log_function_range_l3033_303360

/-- The function f(x) = lg(ax^2 - 2x + 2) has a range of ℝ if and only if a ∈ (0, 1/2] -/
theorem log_function_range (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, y = Real.log (a * x^2 - 2 * x + 2)) ↔ (0 < a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_log_function_range_l3033_303360


namespace NUMINAMATH_CALUDE_garrett_peanut_granola_bars_l3033_303331

/-- The number of granola bars Garrett bought in total -/
def total_granola_bars : ℕ := 14

/-- The number of oatmeal raisin granola bars Garrett bought -/
def oatmeal_raisin_bars : ℕ := 6

/-- The number of peanut granola bars Garrett bought -/
def peanut_granola_bars : ℕ := total_granola_bars - oatmeal_raisin_bars

theorem garrett_peanut_granola_bars : peanut_granola_bars = 8 := by
  sorry

end NUMINAMATH_CALUDE_garrett_peanut_granola_bars_l3033_303331


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3033_303333

-- Define the universal set U
def U : Set ℤ := {-1, -2, -3, 0, 1}

-- Define set M
def M (a : ℤ) : Set ℤ := {-1, 0, a^2 + 1}

-- Theorem statement
theorem complement_of_M_in_U (a : ℤ) (h : M a ⊆ U) :
  U \ M a = {-2, -3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3033_303333


namespace NUMINAMATH_CALUDE_z_value_range_l3033_303370

theorem z_value_range (x y z : ℝ) (sum_eq : x + y + z = 3) (sum_sq_eq : x^2 + y^2 + z^2 = 18) :
  ∃ (z_max z_min : ℝ), 
    (∀ z', (∃ x' y', x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≤ z_max) ∧
    (∀ z', (∃ x' y', x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≥ z_min) ∧
    z_max - z_min = 6 :=
sorry

end NUMINAMATH_CALUDE_z_value_range_l3033_303370


namespace NUMINAMATH_CALUDE_chromosomal_variations_l3033_303344

/-- Represents a biological process or condition -/
inductive BiologicalProcess
| AntherCulture
| DNABaseChange
| NonHomologousRecombination
| CrossingOver
| DownSyndrome

/-- Defines what constitutes a chromosomal variation -/
def isChromosomalVariation (p : BiologicalProcess) : Prop :=
  match p with
  | BiologicalProcess.AntherCulture => true
  | BiologicalProcess.DNABaseChange => false
  | BiologicalProcess.NonHomologousRecombination => false
  | BiologicalProcess.CrossingOver => false
  | BiologicalProcess.DownSyndrome => true

/-- The main theorem stating which processes are chromosomal variations -/
theorem chromosomal_variations :
  (isChromosomalVariation BiologicalProcess.AntherCulture) ∧
  (¬ isChromosomalVariation BiologicalProcess.DNABaseChange) ∧
  (¬ isChromosomalVariation BiologicalProcess.NonHomologousRecombination) ∧
  (¬ isChromosomalVariation BiologicalProcess.CrossingOver) ∧
  (isChromosomalVariation BiologicalProcess.DownSyndrome) :=
by sorry

end NUMINAMATH_CALUDE_chromosomal_variations_l3033_303344


namespace NUMINAMATH_CALUDE_set_union_problem_l3033_303319

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 4} →
  B = {0, 1, a} →
  A ∪ B = {0, 1, 4} →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l3033_303319


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l3033_303382

/-- The cost of one pen in rupees -/
def pen_cost : ℕ := 65

/-- The ratio of the cost of one pen to one pencil -/
def pen_pencil_ratio : ℚ := 5/1

/-- The cost of 3 pens and some pencils in rupees -/
def total_cost : ℕ := 260

/-- The number of pens in a dozen -/
def dozen : ℕ := 12

/-- Theorem stating that the cost of one dozen pens is 780 rupees -/
theorem cost_of_dozen_pens : pen_cost * dozen = 780 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l3033_303382


namespace NUMINAMATH_CALUDE_apples_per_child_l3033_303371

theorem apples_per_child (total_apples : ℕ) (num_children : ℕ) (num_adults : ℕ) (apples_per_adult : ℕ)
  (h1 : total_apples = 450)
  (h2 : num_children = 33)
  (h3 : num_adults = 40)
  (h4 : apples_per_adult = 3) :
  (total_apples - num_adults * apples_per_adult) / num_children = 10 := by
sorry

end NUMINAMATH_CALUDE_apples_per_child_l3033_303371


namespace NUMINAMATH_CALUDE_garden_breadth_l3033_303398

/-- Given a rectangular garden with perimeter 900 m and length 260 m, prove its breadth is 190 m. -/
theorem garden_breadth (perimeter length breadth : ℝ) : 
  perimeter = 900 ∧ length = 260 ∧ perimeter = 2 * (length + breadth) → breadth = 190 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l3033_303398


namespace NUMINAMATH_CALUDE_expand_expression_l3033_303304

theorem expand_expression (x : ℝ) : (9*x + 4) * (2*x^2) = 18*x^3 + 8*x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3033_303304


namespace NUMINAMATH_CALUDE_red_cards_taken_out_l3033_303368

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = 52)
  (h_red : red_cards = total_cards / 2)

/-- Represents the state after some red cards were taken out -/
structure RemainingCards :=
  (remaining_red : ℕ)
  (h_remaining : remaining_red = 16)

theorem red_cards_taken_out (d : Deck) (r : RemainingCards) :
  d.red_cards - r.remaining_red = 10 := by
  sorry

end NUMINAMATH_CALUDE_red_cards_taken_out_l3033_303368


namespace NUMINAMATH_CALUDE_age_ratio_after_time_l3033_303386

/-- Represents a person's age -/
structure Age where
  value : ℕ

/-- Represents the ratio between two ages -/
structure AgeRatio where
  numerator : ℕ
  denominator : ℕ

def Age.addYears (a : Age) (years : ℕ) : Age :=
  ⟨a.value + years⟩

def Age.subtractYears (a : Age) (years : ℕ) : Age :=
  ⟨a.value - years⟩

def AgeRatio.fromAges (a b : Age) : AgeRatio :=
  ⟨a.value, b.value⟩

theorem age_ratio_after_time (sandy_age molly_age : Age) 
    (h1 : AgeRatio.fromAges sandy_age molly_age = ⟨7, 2⟩)
    (h2 : (sandy_age.subtractYears 6).value = 78) :
    AgeRatio.fromAges (sandy_age.addYears 16) (molly_age.addYears 16) = ⟨5, 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_after_time_l3033_303386


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3033_303390

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 625) : 
  (1.2 * L) * (0.8 * W) = 600 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3033_303390


namespace NUMINAMATH_CALUDE_solution_range_l3033_303351

theorem solution_range (x : ℝ) : 
  x ≥ 1 → 
  Real.sqrt (x + 2 - 5 * Real.sqrt (x - 1)) + Real.sqrt (x + 5 - 7 * Real.sqrt (x - 1)) = 2 → 
  5 ≤ x ∧ x ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l3033_303351


namespace NUMINAMATH_CALUDE_delivery_driver_stops_l3033_303327

theorem delivery_driver_stops (initial_stops total_stops : ℕ) 
  (h1 : initial_stops = 3)
  (h2 : total_stops = 7) :
  total_stops - initial_stops = 4 := by
  sorry

end NUMINAMATH_CALUDE_delivery_driver_stops_l3033_303327


namespace NUMINAMATH_CALUDE_cost_of_450_candies_l3033_303372

/-- Represents the cost calculation for chocolate candies --/
def chocolate_cost (candies_per_box : ℕ) (discount_threshold : ℕ) (regular_price : ℚ) (discount_price : ℚ) (total_candies : ℕ) : ℚ :=
  let boxes := total_candies / candies_per_box
  if boxes ≥ discount_threshold then
    (boxes : ℚ) * discount_price
  else
    (boxes : ℚ) * regular_price

/-- Theorem stating the cost of 450 chocolate candies --/
theorem cost_of_450_candies :
  chocolate_cost 15 10 5 4 450 = 120 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_450_candies_l3033_303372


namespace NUMINAMATH_CALUDE_horizontal_line_inclination_l3033_303355

def line (x y : ℝ) : Prop := y + 3 = 0

def angle_of_inclination (f : ℝ → ℝ → Prop) : ℝ := sorry

theorem horizontal_line_inclination :
  angle_of_inclination line = 0 := by sorry

end NUMINAMATH_CALUDE_horizontal_line_inclination_l3033_303355


namespace NUMINAMATH_CALUDE_x_squared_eq_zero_is_quadratic_l3033_303353

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: x^2 = 0 is a quadratic equation in one variable -/
theorem x_squared_eq_zero_is_quadratic : is_quadratic_equation f :=
sorry

end NUMINAMATH_CALUDE_x_squared_eq_zero_is_quadratic_l3033_303353


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3033_303335

-- Define the variables
variable (a : ℕ+) -- a is a positive integer
variable (A B : ℝ) -- A and B are real numbers
variable (x y z : ℕ+) -- x, y, z are positive integers

-- Define the system of equations
def equation1 (x y z : ℕ+) (a : ℕ+) : Prop :=
  (x : ℝ)^2 + (y : ℝ)^2 + (z : ℝ)^2 = (13 * (a : ℝ))^2

def equation2 (x y z : ℕ+) (a : ℕ+) (A B : ℝ) : Prop :=
  (x : ℝ)^2 * (A * (x : ℝ)^2 + B * (y : ℝ)^2) +
  (y : ℝ)^2 * (A * (y : ℝ)^2 + B * (z : ℝ)^2) +
  (z : ℝ)^2 * (A * (z : ℝ)^2 + B * (x : ℝ)^2) =
  1/4 * (2 * A + B) * (13 * (a : ℝ))^4

-- Theorem statement
theorem necessary_and_sufficient_condition :
  (∃ x y z : ℕ+, equation1 x y z a ∧ equation2 x y z a A B) ↔ B = 2 * A :=
sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3033_303335


namespace NUMINAMATH_CALUDE_electricity_relationship_l3033_303361

/-- Represents the relationship between electricity consumption and fee -/
structure ElectricityRelation where
  consumption : ℝ  -- Electricity consumption in kWh
  fee : ℝ          -- Electricity fee in yuan
  linear : fee = 0.55 * consumption  -- Linear relationship

/-- Proves the functional relationship and calculates consumption for a given fee -/
theorem electricity_relationship (r : ElectricityRelation) :
  r.fee = 0.55 * r.consumption ∧ 
  (r.fee = 40.7 → r.consumption = 74) := by
  sorry

#check electricity_relationship

end NUMINAMATH_CALUDE_electricity_relationship_l3033_303361


namespace NUMINAMATH_CALUDE_last_twelve_average_l3033_303384

theorem last_twelve_average (total_average : ℝ) (first_twelve_average : ℝ) (thirteenth_result : ℝ) :
  total_average = 20 →
  first_twelve_average = 14 →
  thirteenth_result = 128 →
  (25 * total_average - 12 * first_twelve_average - thirteenth_result) / 12 = 17 := by
sorry

end NUMINAMATH_CALUDE_last_twelve_average_l3033_303384


namespace NUMINAMATH_CALUDE_prob_A_shot_twice_correct_l3033_303358

def prob_A : ℚ := 3/4
def prob_B : ℚ := 4/5

def prob_A_shot_twice : ℚ := 19/400

theorem prob_A_shot_twice_correct :
  let p_A_miss := 1 - prob_A
  let p_B_miss := 1 - prob_B
  prob_A_shot_twice = p_A_miss * p_B_miss * prob_A + p_A_miss * p_B_miss * p_A_miss * prob_B :=
by sorry

end NUMINAMATH_CALUDE_prob_A_shot_twice_correct_l3033_303358


namespace NUMINAMATH_CALUDE_consecutive_squares_difference_l3033_303338

theorem consecutive_squares_difference (n : ℕ) : 
  (n > 0) → 
  (n + (n + 1) < 150) → 
  ((n + 1)^2 - n^2 = 129 ∨ (n + 1)^2 - n^2 = 147) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_difference_l3033_303338


namespace NUMINAMATH_CALUDE_negation_existence_gt_one_l3033_303318

theorem negation_existence_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_gt_one_l3033_303318


namespace NUMINAMATH_CALUDE_billys_candy_count_l3033_303352

/-- The total number of candy pieces given the number of boxes and pieces per box -/
def total_candy (boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  boxes * pieces_per_box

/-- Theorem: Billy's total candy pieces -/
theorem billys_candy_count :
  total_candy 7 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_billys_candy_count_l3033_303352


namespace NUMINAMATH_CALUDE_circle_placement_possible_l3033_303373

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Theorem: In a 20x25 rectangle with 120 unit squares, there exists a point
    that is at least 0.5 units away from any edge and at least √2/2 units
    away from the center of any unit square -/
theorem circle_placement_possible (rect : Rectangle) 
    (squares : Finset Point) : 
    rect.width = 20 ∧ rect.height = 25 ∧ squares.card = 120 →
    ∃ p : Point, 
      0.5 ≤ p.x ∧ p.x ≤ rect.width - 0.5 ∧
      0.5 ≤ p.y ∧ p.y ≤ rect.height - 0.5 ∧
      ∀ s ∈ squares, (p.x - s.x)^2 + (p.y - s.y)^2 ≥ 0.5 := by
  sorry

end NUMINAMATH_CALUDE_circle_placement_possible_l3033_303373


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l3033_303354

theorem triangle_abc_proof (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  0 < A ∧ A < π →
  m = (Real.cos A, Real.sin A) →
  n = (Real.sqrt 2 - Real.sin A, Real.cos A) →
  Real.sqrt ((m.1 + n.1)^2 + (m.2 + n.2)^2) = 2 →
  b = 4 * Real.sqrt 2 →
  c = Real.sqrt 2 * a →
  A = π / 4 ∧ (1/2 * b * a = 16) := by sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l3033_303354


namespace NUMINAMATH_CALUDE_skateboard_ramp_speeds_l3033_303345

theorem skateboard_ramp_speeds (S₁ S₂ S₃ : ℝ) :
  (S₁ + S₂ + S₃) / 3 + 4 = 40 →
  ∃ (T₁ T₂ T₃ : ℝ), (T₁ + T₂ + T₃) / 3 + 4 = 40 ∧ (T₁ ≠ S₁ ∨ T₂ ≠ S₂ ∨ T₃ ≠ S₃) :=
by sorry

end NUMINAMATH_CALUDE_skateboard_ramp_speeds_l3033_303345


namespace NUMINAMATH_CALUDE_spatial_relations_l3033_303395

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular parallel subset : ∀ {T U : Type}, T → U → Prop)

-- Define the given conditions
variable (α β γ : Plane)
variable (m n : Line)
variable (h1 : m ≠ n)

-- Define the main theorem
theorem spatial_relations :
  (∀ α β γ : Plane, perpendicular α β → parallel α γ ∧ perpendicular α γ) →
  ((parallel m n ∧ subset n α) → parallel m α) ∧
  ((perpendicular m α ∧ parallel n α) → perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_spatial_relations_l3033_303395


namespace NUMINAMATH_CALUDE_four_points_probability_l3033_303339

-- Define a circle
def Circle : Type := Unit

-- Define a point on a circle
def Point (c : Circle) : Type := Unit

-- Define a function to choose n points uniformly at random on a circle
def chooseRandomPoints (c : Circle) (n : ℕ) : Type := 
  Fin n → Point c

-- Define a predicate for two points and the center forming an obtuse triangle
def isObtuse (c : Circle) (p1 p2 : Point c) : Prop := sorry

-- Define a function to calculate the probability of an event
def probability (event : Prop) : ℝ := sorry

-- The main theorem
theorem four_points_probability (c : Circle) :
  probability (∀ (points : chooseRandomPoints c 4),
    ∀ (i j : Fin 4), i ≠ j → ¬isObtuse c (points i) (points j)) = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_four_points_probability_l3033_303339


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3033_303320

/-- The common difference of an arithmetic sequence with general term a_n = 5 - 4n is -4. -/
theorem arithmetic_sequence_common_difference :
  ∀ (a : ℕ → ℝ), (∀ n, a n = 5 - 4 * n) →
  ∃ d : ℝ, ∀ n, a (n + 1) - a n = d ∧ d = -4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3033_303320


namespace NUMINAMATH_CALUDE_burrito_count_l3033_303330

theorem burrito_count (cheese_per_burrito cheese_per_taco total_cheese : ℕ) 
  (h1 : cheese_per_burrito = 4)
  (h2 : cheese_per_taco = 9)
  (h3 : total_cheese = 37) :
  ∃ (num_burritos : ℕ), 
    num_burritos * cheese_per_burrito + cheese_per_taco = total_cheese ∧ 
    num_burritos = 7 := by
  sorry

end NUMINAMATH_CALUDE_burrito_count_l3033_303330


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_l3033_303308

/-- The fraction of the total cost that is the cost of raisins in a mixture of raisins and nuts -/
theorem raisin_cost_fraction (raisin_pounds : ℝ) (nut_pounds : ℝ) (raisin_cost : ℝ) :
  raisin_pounds = 3 →
  nut_pounds = 4 →
  raisin_cost > 0 →
  (raisin_pounds * raisin_cost) / ((raisin_pounds * raisin_cost) + (nut_pounds * (2 * raisin_cost))) = 3 / 11 := by
  sorry

#check raisin_cost_fraction

end NUMINAMATH_CALUDE_raisin_cost_fraction_l3033_303308


namespace NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l3033_303394

theorem arccos_of_one_eq_zero : Real.arccos 1 = 0 := by sorry

end NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l3033_303394


namespace NUMINAMATH_CALUDE_sum_of_threes_place_values_l3033_303359

def number : ℕ := 63130

def first_three_place_value : ℕ := 3000
def second_three_place_value : ℕ := 30

theorem sum_of_threes_place_values :
  first_three_place_value + second_three_place_value = 3030 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_threes_place_values_l3033_303359


namespace NUMINAMATH_CALUDE_hot_dog_remainder_l3033_303348

theorem hot_dog_remainder : 25197641 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_remainder_l3033_303348


namespace NUMINAMATH_CALUDE_total_pencils_l3033_303313

theorem total_pencils (drawer : ℕ) (desk_initial : ℕ) (desk_added : ℕ) 
  (h1 : drawer = 43)
  (h2 : desk_initial = 19)
  (h3 : desk_added = 16) :
  drawer + desk_initial + desk_added = 78 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l3033_303313


namespace NUMINAMATH_CALUDE_david_boxes_l3033_303356

/-- Given a total number of dogs and the number of dogs per box, 
    calculate the number of boxes needed. -/
def calculate_boxes (total_dogs : ℕ) (dogs_per_box : ℕ) : ℕ :=
  total_dogs / dogs_per_box

/-- Theorem stating that given 28 total dogs and 4 dogs per box, 
    the number of boxes is 7. -/
theorem david_boxes : calculate_boxes 28 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_david_boxes_l3033_303356


namespace NUMINAMATH_CALUDE_cans_per_bag_l3033_303312

theorem cans_per_bag (total_cans : ℕ) (total_bags : ℕ) (h1 : total_cans = 63) (h2 : total_bags = 7) :
  total_cans / total_bags = 9 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_bag_l3033_303312


namespace NUMINAMATH_CALUDE_least_prime_for_integer_roots_l3033_303376

theorem least_prime_for_integer_roots : 
  ∃ (P : ℕ), 
    Prime P ∧ 
    (∃ (x : ℤ), x^2 + 2*(P+1)*x + P^2 - P - 14 = 0) ∧
    (∀ (Q : ℕ), Prime Q ∧ Q < P → ¬∃ (y : ℤ), y^2 + 2*(Q+1)*y + Q^2 - Q - 14 = 0) ∧
    P = 7 :=
sorry

end NUMINAMATH_CALUDE_least_prime_for_integer_roots_l3033_303376


namespace NUMINAMATH_CALUDE_minor_arc_circumference_l3033_303341

theorem minor_arc_circumference (r : ℝ) (θ : ℝ) (h_r : r = 12) (h_θ : θ = 110 * π / 180) :
  let circle_circumference := 2 * π * r
  let arc_length := circle_circumference * θ / (2 * π)
  arc_length = 22 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_minor_arc_circumference_l3033_303341


namespace NUMINAMATH_CALUDE_alloy_mixture_problem_l3033_303314

/-- Proves that the amount of the first alloy used is 15 kg given the conditions of the problem -/
theorem alloy_mixture_problem (x : ℝ) : 
  (0.10 * x + 0.08 * 35 = 0.086 * (x + 35)) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_problem_l3033_303314


namespace NUMINAMATH_CALUDE_arithmetic_sequence_l3033_303346

def a (n : ℕ) : ℤ := 3 * n + 1

theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) - a n = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_l3033_303346


namespace NUMINAMATH_CALUDE_handshake_count_l3033_303342

/-- Represents a gathering of married couples -/
structure Gathering where
  couples : ℕ
  no_male_handshakes : Bool

/-- Calculates the total number of handshakes in the gathering -/
def total_handshakes (g : Gathering) : ℕ :=
  let women := g.couples
  let men := g.couples
  let women_handshakes := women.choose 2
  let men_women_handshakes := men * (women - 1)
  women_handshakes + men_women_handshakes

/-- Theorem stating that in a gathering of 15 married couples with the given conditions, 
    the total number of handshakes is 315 -/
theorem handshake_count (g : Gathering) :
  g.couples = 15 ∧ g.no_male_handshakes = true → total_handshakes g = 315 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l3033_303342


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l3033_303311

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l3033_303311


namespace NUMINAMATH_CALUDE_equation_solutions_l3033_303325

theorem equation_solutions : 
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 5 ∧ x2 = 2 - Real.sqrt 5 ∧ 
    x1^2 - 4*x1 - 1 = 0 ∧ x2^2 - 4*x2 - 1 = 0) ∧ 
  (∃ x1 x2 : ℝ, x1 = -3 ∧ x2 = 6 ∧ 
    (x1 + 3)*(x1 - 3) = 3*(x1 + 3) ∧ (x2 + 3)*(x2 - 3) = 3*(x2 + 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3033_303325


namespace NUMINAMATH_CALUDE_triangle_side_value_l3033_303305

open Real

/-- Prove that in triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*sin(B) = √2*sin(C), cos(C) = 1/3, and the area of the triangle is 4, then c = 6. -/
theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) :
  a * sin B = sqrt 2 * sin C →
  cos C = 1 / 3 →
  1 / 2 * a * b * sin C = 4 →
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l3033_303305


namespace NUMINAMATH_CALUDE_impossible_inequalities_l3033_303349

theorem impossible_inequalities (a b c : ℝ) : ¬(|a| < |b - c| ∧ |b| < |c - a| ∧ |c| < |a - b|) := by
  sorry

end NUMINAMATH_CALUDE_impossible_inequalities_l3033_303349


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3033_303378

theorem arithmetic_expression_equality : 76 + (144 / 12) + (15 * 19)^2 - 350 - (270 / 6) = 80918 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3033_303378


namespace NUMINAMATH_CALUDE_probability_theorem_l3033_303397

/-- The number of possible outcomes when rolling a single six-sided die -/
def dice_outcomes : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 7

/-- The probability of rolling seven standard six-sided dice and getting at least one pair
    but not a three-of-a-kind -/
def probability_at_least_one_pair_no_three_of_a_kind : ℚ :=
  6426 / 13997

/-- Theorem stating that the probability of rolling seven standard six-sided dice
    and getting at least one pair but not a three-of-a-kind is 6426/13997 -/
theorem probability_theorem :
  probability_at_least_one_pair_no_three_of_a_kind = 6426 / 13997 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3033_303397


namespace NUMINAMATH_CALUDE_binary_to_decimal_l3033_303365

theorem binary_to_decimal (b : List Bool) :
  (b.reverse.enum.map (λ (i, x) => if x then 2^i else 0)).sum = 45 :=
sorry

end NUMINAMATH_CALUDE_binary_to_decimal_l3033_303365


namespace NUMINAMATH_CALUDE_distribute_nine_to_three_l3033_303396

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 9 distinct objects into 3 distinct containers,
    with each container receiving at least one object, is 504 -/
theorem distribute_nine_to_three : distribute 9 3 = 504 := by sorry

end NUMINAMATH_CALUDE_distribute_nine_to_three_l3033_303396


namespace NUMINAMATH_CALUDE_height_comparison_l3033_303322

theorem height_comparison (p q : ℝ) (h : p = 0.6 * q) :
  (q - p) / p = 2/3 := by sorry

end NUMINAMATH_CALUDE_height_comparison_l3033_303322


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_B_or_C_given_A_l3033_303323

/-- The probability of selecting at least one of boy B and girl C, given boy A is already selected -/
theorem prob_at_least_one_of_B_or_C_given_A (total_boys : Nat) (total_girls : Nat) 
  (representatives : Nat) (h1 : total_boys = 5) (h2 : total_girls = 2) (h3 : representatives = 3) :
  let remaining_boys := total_boys - 1
  let remaining_total := total_boys + total_girls - 1
  let total_ways := Nat.choose remaining_total (representatives - 1)
  let ways_without_B_or_C := Nat.choose (remaining_boys - 1) (representatives - 1) + 
                             Nat.choose (remaining_boys - 1) (representatives - 2) * total_girls
  (1 : ℚ) - (ways_without_B_or_C : ℚ) / total_ways = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_B_or_C_given_A_l3033_303323


namespace NUMINAMATH_CALUDE_increasing_function_implies_a_geq_5_l3033_303399

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem increasing_function_implies_a_geq_5 (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y < 4 → f a x < f a y) →
  a ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_increasing_function_implies_a_geq_5_l3033_303399


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3033_303389

theorem cylinder_minus_cones_volume (r h₁ h₂ : ℝ) (hr : r = 10) (hh₁ : h₁ = 15) (hh₂ : h₂ = 30) :
  let v_cyl := π * r^2 * h₂
  let v_cone := (1/3) * π * r^2 * h₁
  v_cyl - 2 * v_cone = 2000 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3033_303389


namespace NUMINAMATH_CALUDE_correct_operation_l3033_303302

theorem correct_operation (a : ℝ) : 2 * a^2 * a = 2 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3033_303302


namespace NUMINAMATH_CALUDE_board_cut_theorem_l3033_303369

theorem board_cut_theorem (total_length : ℝ) (shorter_length : ℝ) :
  total_length = 20 ∧
  shorter_length > 0 ∧
  shorter_length < total_length ∧
  2 * shorter_length = (total_length - shorter_length) + 4 →
  shorter_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l3033_303369


namespace NUMINAMATH_CALUDE_may_birth_percentage_l3033_303385

def total_mathematicians : ℕ := 120
def may_births : ℕ := 15

theorem may_birth_percentage :
  (may_births : ℚ) / total_mathematicians * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_may_birth_percentage_l3033_303385


namespace NUMINAMATH_CALUDE_wednesday_distance_l3033_303375

/-- Represents the distance Mona biked on each day of the week -/
structure BikeDistance where
  monday : ℕ
  wednesday : ℕ
  saturday : ℕ

/-- Defines the conditions of Mona's biking schedule -/
def validBikeSchedule (d : BikeDistance) : Prop :=
  d.monday + d.wednesday + d.saturday = 30 ∧
  d.monday = 6 ∧
  d.saturday = 2 * d.monday

theorem wednesday_distance (d : BikeDistance) (h : validBikeSchedule d) : d.wednesday = 12 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_distance_l3033_303375


namespace NUMINAMATH_CALUDE_fraction_sum_l3033_303324

theorem fraction_sum (a b : ℝ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3033_303324


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3033_303309

/-- A rectangle with an inscribed ellipse -/
structure RectangleWithEllipse where
  -- Rectangle dimensions
  x : ℝ
  y : ℝ
  -- Ellipse semi-major and semi-minor axes
  a : ℝ
  b : ℝ
  -- Conditions
  rectangle_area : x * y = 4024
  ellipse_area : π * a * b = 4024 * π
  foci_distance : x^2 + y^2 = 4 * (a^2 - b^2)
  major_axis : x + y = 2 * a

/-- The perimeter of a rectangle with an inscribed ellipse is 8√2012 -/
theorem rectangle_perimeter (r : RectangleWithEllipse) : r.x + r.y = 8 * Real.sqrt 2012 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3033_303309


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3033_303347

def y := 2^(3^5 * 4^4 * 5^7 * 6^5 * 7^3 * 8^6 * 9^10)

theorem smallest_multiplier_for_perfect_square (k : ℕ) : 
  k > 0 ∧ (∃ m : ℕ, k * y = m^2) ∧ (∀ l < k, l > 0 → ¬∃ m : ℕ, l * y = m^2) ↔ k = 70 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3033_303347


namespace NUMINAMATH_CALUDE_joan_snow_volume_l3033_303329

/-- The volume of snow on a rectangular driveway -/
def snow_volume (length width depth : ℚ) : ℚ :=
  length * width * depth

/-- Proof that the volume of snow on Joan's driveway is 90 cubic feet -/
theorem joan_snow_volume :
  snow_volume 40 3 (3/4) = 90 := by
  sorry

end NUMINAMATH_CALUDE_joan_snow_volume_l3033_303329


namespace NUMINAMATH_CALUDE_stream_speed_l3033_303328

/-- Given a boat traveling downstream, prove the speed of the stream. -/
theorem stream_speed 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 25) 
  (h2 : downstream_distance = 90) 
  (h3 : downstream_time = 3) : 
  ∃ stream_speed : ℝ, 
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧ 
    stream_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3033_303328


namespace NUMINAMATH_CALUDE_min_value_expression_l3033_303380

theorem min_value_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x * y = 1) :
  ∃ (t : ℝ), t = 25 ∧ ∀ (z : ℝ), (3 * x^3 + 125 * y^3) / (x - y) ≥ z := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3033_303380


namespace NUMINAMATH_CALUDE_common_chord_circle_equation_l3033_303306

-- Define the two circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 2*y - 13 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 + 12*x + 16*y - 25 = 0

-- Define the result circle
def result_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 25

-- Theorem statement
theorem common_chord_circle_equation :
  ∀ x y : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_C1 x₁ y₁ ∧ circle_C2 x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 > 0 ∧
    (x - (x₁ + x₂)/2)^2 + (y - (y₁ + y₂)/2)^2 = ((x₁ - x₂)^2 + (y₁ - y₂)^2) / 4) →
  result_circle x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_circle_equation_l3033_303306


namespace NUMINAMATH_CALUDE_pascals_triangle_15_numbers_4th_entry_l3033_303357

theorem pascals_triangle_15_numbers_4th_entry : 
  let n : ℕ := 14  -- The row number (15 numbers, so it's the 14th row)
  let k : ℕ := 4   -- The position of the number we're looking for
  Nat.choose (n - 1) (k - 1) = 286 := by
sorry

end NUMINAMATH_CALUDE_pascals_triangle_15_numbers_4th_entry_l3033_303357


namespace NUMINAMATH_CALUDE_girl_travel_distance_l3033_303337

/-- 
Given a constant speed and time, calculates the distance traveled.
-/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- 
Theorem: A girl traveling at 4 m/s for 32 seconds covers a distance of 128 meters.
-/
theorem girl_travel_distance : 
  distance_traveled 4 32 = 128 := by
  sorry

end NUMINAMATH_CALUDE_girl_travel_distance_l3033_303337


namespace NUMINAMATH_CALUDE_circle_properties_l3033_303381

/-- The circle equation --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 6*y - 12 = 0

/-- The center of the circle --/
def circle_center : ℝ × ℝ := (-2, 3)

/-- The radius of the circle --/
def circle_radius : ℝ := 5

/-- Theorem stating that the given equation represents a circle with the specified center and radius --/
theorem circle_properties :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3033_303381


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3033_303343

/-- Hyperbola with given properties and intersecting circle -/
structure HyperbolaWithCircle where
  b : ℝ
  h_b_pos : b > 0
  hyperbola : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 - y^2/b^2 = 1
  asymptote : ℝ → ℝ := fun x ↦ b * x
  circle : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 + y^2 = 1
  intersection_area : ℝ := b

/-- The eccentricity of a hyperbola with the given properties is √3 -/
theorem hyperbola_eccentricity (h : HyperbolaWithCircle) : 
  ∃ (e : ℝ), e = Real.sqrt 3 ∧ e^2 = 1 + 1/h.b^2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3033_303343


namespace NUMINAMATH_CALUDE_coconut_trips_l3033_303307

/-- The number of trips needed to move coconuts -/
def num_trips (total_coconuts : ℕ) (barbie_capacity : ℕ) (bruno_capacity : ℕ) : ℕ :=
  total_coconuts / (barbie_capacity + bruno_capacity)

/-- Theorem stating that 12 trips are needed to move 144 coconuts -/
theorem coconut_trips : num_trips 144 4 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_coconut_trips_l3033_303307
