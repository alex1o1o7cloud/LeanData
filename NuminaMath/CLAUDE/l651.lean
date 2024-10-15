import Mathlib

namespace NUMINAMATH_CALUDE_function_properties_l651_65156

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_properties (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_neg : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧ (f 2 = f 0) := by sorry

end NUMINAMATH_CALUDE_function_properties_l651_65156


namespace NUMINAMATH_CALUDE_bargain_bin_book_count_l651_65138

/-- Calculates the final number of books in a bargain bin after selling and adding books. -/
def final_book_count (initial : ℕ) (sold : ℕ) (added : ℕ) : ℕ :=
  initial - sold + added

/-- Proves that the final number of books in the bin is correct for the given scenario. -/
theorem bargain_bin_book_count :
  final_book_count 4 3 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bargain_bin_book_count_l651_65138


namespace NUMINAMATH_CALUDE_hyunji_pencils_l651_65125

/-- Given an initial number of pencils, the number given away, and the number received,
    calculate the final number of pencils. -/
def final_pencils (initial given_away received : ℕ) : ℕ :=
  initial - given_away + received

/-- Theorem stating that with 20 initial pencils, giving away 7 and receiving 5
    results in 18 pencils. -/
theorem hyunji_pencils : final_pencils 20 7 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_hyunji_pencils_l651_65125


namespace NUMINAMATH_CALUDE_mixed_doubles_groupings_l651_65112

theorem mixed_doubles_groupings (male_players : Nat) (female_players : Nat) :
  male_players = 5 → female_players = 3 →
  (Nat.choose male_players 2) * (Nat.choose female_players 2) * (Nat.factorial 2) = 60 :=
by sorry

end NUMINAMATH_CALUDE_mixed_doubles_groupings_l651_65112


namespace NUMINAMATH_CALUDE_wait_probability_is_two_thirds_l651_65185

/-- The duration of the red light in seconds -/
def red_light_duration : ℕ := 30

/-- The minimum waiting time in seconds -/
def min_wait_time : ℕ := 10

/-- The probability of waiting at least 'min_wait_time' seconds for the green light -/
def wait_probability : ℚ := (red_light_duration - min_wait_time) / red_light_duration

theorem wait_probability_is_two_thirds : 
  wait_probability = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_wait_probability_is_two_thirds_l651_65185


namespace NUMINAMATH_CALUDE_art_museum_pictures_l651_65135

theorem art_museum_pictures : ∃ (P : ℕ), P > 0 ∧ P % 2 = 1 ∧ (P + 1) % 2 = 0 ∧ ∀ (Q : ℕ), (Q > 0 ∧ Q % 2 = 1 ∧ (Q + 1) % 2 = 0) → P ≤ Q :=
by sorry

end NUMINAMATH_CALUDE_art_museum_pictures_l651_65135


namespace NUMINAMATH_CALUDE_functional_equation_solution_l651_65144

/-- The functional equation that f must satisfy -/
def functional_equation (f : ℝ → ℝ) (α β : ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x * f y = y^α * f (x/2) + x^β * f (y/2)

/-- The theorem stating the possible forms of f -/
theorem functional_equation_solution (f : ℝ → ℝ) (α β : ℝ) :
  functional_equation f α β →
  (∃ c : ℝ, c = 2^(1-α) ∧ ∀ x, x > 0 → f x = c * x^α) ∨
  (∀ x, x > 0 → f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l651_65144


namespace NUMINAMATH_CALUDE_fat_per_cup_of_rice_l651_65198

/-- Amount of rice eaten in the morning -/
def morning_rice : ℕ := 3

/-- Amount of rice eaten in the afternoon -/
def afternoon_rice : ℕ := 2

/-- Amount of rice eaten in the evening -/
def evening_rice : ℕ := 5

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Total fat intake from rice in a week (in grams) -/
def weekly_fat_intake : ℕ := 700

/-- Calculate the amount of fat in a cup of rice -/
theorem fat_per_cup_of_rice : 
  (weekly_fat_intake : ℚ) / ((morning_rice + afternoon_rice + evening_rice) * days_in_week) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fat_per_cup_of_rice_l651_65198


namespace NUMINAMATH_CALUDE_roots_condition_implies_a_equals_neg_nine_l651_65157

/-- The polynomial p(x) = x³ - 6x² + ax + a, where a is a parameter --/
def p (a : ℝ) (x : ℝ) : ℝ := x^3 - 6*x^2 + a*x + a

/-- The condition that the sum of cubes of the roots minus 3 is zero --/
def sum_of_cubes_minus_3_is_zero (x₁ x₂ x₃ : ℝ) : Prop :=
  (x₁ - 3)^3 + (x₂ - 3)^3 + (x₃ - 3)^3 = 0

theorem roots_condition_implies_a_equals_neg_nine (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 
    (∀ x : ℝ, p a x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    sum_of_cubes_minus_3_is_zero x₁ x₂ x₃) →
  a = -9 :=
sorry

end NUMINAMATH_CALUDE_roots_condition_implies_a_equals_neg_nine_l651_65157


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l651_65139

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : a = 12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l651_65139


namespace NUMINAMATH_CALUDE_gem_stone_necklaces_sold_megan_sold_three_gem_stone_necklaces_l651_65158

/-- The number of gem stone necklaces sold at a garage sale -/
theorem gem_stone_necklaces_sold (bead_necklaces : ℕ) (price_per_necklace : ℕ) (total_earnings : ℕ) : ℕ :=
  let gem_stone_necklaces := (total_earnings - bead_necklaces * price_per_necklace) / price_per_necklace
  gem_stone_necklaces

/-- Proof that Megan sold 3 gem stone necklaces -/
theorem megan_sold_three_gem_stone_necklaces : 
  gem_stone_necklaces_sold 7 9 90 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gem_stone_necklaces_sold_megan_sold_three_gem_stone_necklaces_l651_65158


namespace NUMINAMATH_CALUDE_a_squared_gt_a_necessary_not_sufficient_l651_65154

theorem a_squared_gt_a_necessary_not_sufficient :
  (∀ a : ℝ, a > 1 → a^2 > a) ∧
  (∃ a : ℝ, a^2 > a ∧ ¬(a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_a_squared_gt_a_necessary_not_sufficient_l651_65154


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l651_65170

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions, its 12th term equals 14. -/
theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 2) :
  a 12 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l651_65170


namespace NUMINAMATH_CALUDE_bus_fraction_is_two_thirds_l651_65180

def total_distance : ℝ := 30.000000000000007

theorem bus_fraction_is_two_thirds :
  let foot_distance := (1 / 5 : ℝ) * total_distance
  let car_distance := 4
  let bus_distance := total_distance - (foot_distance + car_distance)
  bus_distance / total_distance = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_bus_fraction_is_two_thirds_l651_65180


namespace NUMINAMATH_CALUDE_high_confidence_possible_no_cases_l651_65188

/-- Represents the confidence level of the relationship between smoking and lung cancer -/
def confidence_level : ℝ := 0.99

/-- Represents a sample of smokers -/
def sample_size : ℕ := 100

/-- Represents the possibility of having no lung cancer cases in a sample -/
def possible_no_cases : Prop := true

/-- Theorem stating that despite high confidence in the smoking-lung cancer relationship,
    it's possible to have a sample with no lung cancer cases -/
theorem high_confidence_possible_no_cases :
  confidence_level > 0.99 → possible_no_cases := by sorry

end NUMINAMATH_CALUDE_high_confidence_possible_no_cases_l651_65188


namespace NUMINAMATH_CALUDE_equation_solution_l651_65164

theorem equation_solution : ∃! x : ℝ, 13 + Real.sqrt (-4 + 5 * x * 3) = 14 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l651_65164


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l651_65169

/-- The average age of a cricket team given specific conditions -/
theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (average_age : ℝ),
  team_size = 11 →
  captain_age = 24 →
  wicket_keeper_age_diff = 7 →
  (team_size : ℝ) * average_age = 
    (captain_age : ℝ) + (captain_age + wicket_keeper_age_diff : ℝ) + 
    ((team_size - 2 : ℝ) * (average_age - 1)) →
  average_age = 23 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l651_65169


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l651_65145

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (cube_side_length : ℝ) 
  (h_side_length : cube_side_length = 10) : ℝ :=
by
  -- The volume of the tetrahedron formed by alternately colored vertices
  -- of a cube with side length 10 units is 1000/3 cubic units
  sorry

#check tetrahedron_volume_in_cube

end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l651_65145


namespace NUMINAMATH_CALUDE_horse_rider_ratio_l651_65171

theorem horse_rider_ratio (total_horses : ℕ) (total_legs_walking : ℕ) 
  (h1 : total_horses = 10)
  (h2 : total_legs_walking = 50) :
  (total_horses - (total_legs_walking / 6)) / total_horses = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_horse_rider_ratio_l651_65171


namespace NUMINAMATH_CALUDE_sandy_book_purchase_l651_65168

/-- The number of books Sandy bought from the first shop -/
def books_first_shop : ℕ := 65

/-- The amount Sandy spent at the first shop -/
def amount_first_shop : ℚ := 1380

/-- The amount Sandy spent at the second shop -/
def amount_second_shop : ℚ := 900

/-- The average price Sandy paid per book -/
def average_price : ℚ := 19

/-- The number of books Sandy bought from the second shop -/
def books_second_shop : ℕ := 55

theorem sandy_book_purchase :
  (amount_first_shop + amount_second_shop) / (books_first_shop + books_second_shop : ℚ) = average_price :=
by sorry

end NUMINAMATH_CALUDE_sandy_book_purchase_l651_65168


namespace NUMINAMATH_CALUDE_max_members_is_414_l651_65108

/-- The number of members in the dance group. -/
def m : ℕ := 414

/-- Represents the condition that when arranged in a square formation, there are 11 members left over. -/
def square_formation_condition (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k^2 + 11

/-- Represents the condition that when arranged in a formation with 5 more rows than columns, there are no members left over. -/
def rectangular_formation_condition (m : ℕ) : Prop :=
  ∃ n : ℕ, m = n * (n + 5)

/-- Theorem stating that 414 is the maximum number of members satisfying both conditions. -/
theorem max_members_is_414 :
  square_formation_condition m ∧
  rectangular_formation_condition m ∧
  ∀ x > m, ¬(square_formation_condition x ∧ rectangular_formation_condition x) :=
by sorry

end NUMINAMATH_CALUDE_max_members_is_414_l651_65108


namespace NUMINAMATH_CALUDE_job_completion_time_specific_job_completion_time_l651_65161

/-- 
If one person completes a job in a days and another person completes the same job in b days,
then together they will complete the job in (a * b) / (a + b) days.
-/
theorem job_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let combined_time := (a * b) / (a + b)
  combined_time = (a⁻¹ + b⁻¹)⁻¹ :=
by sorry

/--
If one person completes a job in 8 days and another person completes the same job in 24 days,
then together they will complete the job in 6 days.
-/
theorem specific_job_completion_time :
  let a := 8
  let b := 24
  let combined_time := (a * b) / (a + b)
  combined_time = 6 :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_specific_job_completion_time_l651_65161


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l651_65175

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l651_65175


namespace NUMINAMATH_CALUDE_average_of_last_three_l651_65163

theorem average_of_last_three (A B C D : ℝ) : 
  A = 33 →
  D = 18 →
  (A + B + C) / 3 = 20 →
  (B + C + D) / 3 = 15 := by
sorry

end NUMINAMATH_CALUDE_average_of_last_three_l651_65163


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l651_65178

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 75 → b = 100 → c^2 = a^2 + b^2 → c = 125 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l651_65178


namespace NUMINAMATH_CALUDE_largest_sum_of_3digit_numbers_l651_65124

def digits : Finset Nat := {1, 2, 3, 7, 8, 9}

def is_valid_pair (a b : Nat) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ b ≥ 100 ∧ b < 1000 ∧
  (∃ (d1 d2 d3 d4 d5 d6 : Nat),
    {d1, d2, d3, d4, d5, d6} = digits ∧
    a = 100 * d1 + 10 * d2 + d3 ∧
    b = 100 * d4 + 10 * d5 + d6)

def sum_of_pair (a b : Nat) : Nat := a + b

theorem largest_sum_of_3digit_numbers :
  (∃ (a b : Nat), is_valid_pair a b ∧
    ∀ (x y : Nat), is_valid_pair x y → sum_of_pair x y ≤ sum_of_pair a b) ∧
  (∀ (a b : Nat), is_valid_pair a b → sum_of_pair a b ≤ 1803) ∧
  (∃ (a b : Nat), is_valid_pair a b ∧ sum_of_pair a b = 1803) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_of_3digit_numbers_l651_65124


namespace NUMINAMATH_CALUDE_infinite_series_sum_l651_65179

/-- The sum of the infinite series ∑(n=1 to ∞) (4n^2 - 2n + 1) / 3^n is equal to 5 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (4 * n^2 - 2 * n + 1) / 3^n) = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l651_65179


namespace NUMINAMATH_CALUDE_complement_of_A_when_a_is_one_range_of_a_given_subset_l651_65162

-- Define set A
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 8*a^2 ≤ 0}

-- Part I
theorem complement_of_A_when_a_is_one :
  Set.compl (A 1) = {x | x < -2 ∨ x > 4} := by sorry

-- Part II
theorem range_of_a_given_subset (a : ℝ) (h1 : a > 0) (h2 : Set.Ioo (-1 : ℝ) 1 ⊆ A a) :
  a ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_when_a_is_one_range_of_a_given_subset_l651_65162


namespace NUMINAMATH_CALUDE_pencil_multiple_l651_65177

theorem pencil_multiple (reeta_pencils : ℕ) (total_pencils : ℕ) (anika_pencils : ℕ → ℕ) :
  reeta_pencils = 20 →
  total_pencils = 64 →
  (∀ M : ℕ, anika_pencils M = 20 * M + 4) →
  ∃ M : ℕ, M = 2 ∧ anika_pencils M + reeta_pencils = total_pencils :=
by sorry

end NUMINAMATH_CALUDE_pencil_multiple_l651_65177


namespace NUMINAMATH_CALUDE_greatest_power_of_three_in_19_factorial_l651_65134

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Counts the number of factors of 3 in n! -/
def count_factors_of_three (n : ℕ) : ℕ :=
  if n < 3 then 0
  else (n / 3) + count_factors_of_three (n / 3)

theorem greatest_power_of_three_in_19_factorial :
  ∀ n : ℕ, 3^n ∣ factorial 19 ↔ n ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_in_19_factorial_l651_65134


namespace NUMINAMATH_CALUDE_quadratic_discriminant_zero_l651_65181

theorem quadratic_discriminant_zero (b : ℝ) : 
  (∀ x, 3 * x^2 + 5 * b * x + 7 = 0 → (5 * b)^2 - 4 * 3 * 7 = 0) → 
  b = 2 * Real.sqrt 21 ∨ b = -2 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_zero_l651_65181


namespace NUMINAMATH_CALUDE_total_heads_count_l651_65165

/-- The number of feet per hen -/
def henFeet : ℕ := 2

/-- The number of feet per cow -/
def cowFeet : ℕ := 4

/-- Theorem: Given a group of hens and cows, if the total number of feet is 140
    and there are 26 hens, then the total number of heads is 48. -/
theorem total_heads_count (totalFeet : ℕ) (henCount : ℕ) : 
  totalFeet = 140 → henCount = 26 → henCount * henFeet + (totalFeet - henCount * henFeet) / cowFeet = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_heads_count_l651_65165


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l651_65126

-- Define the arithmetic operations
def calculation1 : ℤ := 36 * 17 + 129
def calculation2 : ℤ := 320 * (300 - 294)
def calculation3 : ℤ := 25 * 5 * 4
def calculation4 : ℚ := 18.45 - 25.6 - 24.4

-- Theorem statements
theorem arithmetic_calculations :
  (calculation1 = 741) ∧
  (calculation2 = 1920) ∧
  (calculation3 = 500) ∧
  (calculation4 = -31.55) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l651_65126


namespace NUMINAMATH_CALUDE_cookie_rows_per_tray_l651_65121

/-- Given the total number of cookies, cookies per row, and number of baking trays,
    calculate the number of rows of cookies on each baking tray. -/
def rows_per_tray (total_cookies : ℕ) (cookies_per_row : ℕ) (num_trays : ℕ) : ℕ :=
  (total_cookies / cookies_per_row) / num_trays

/-- Theorem stating that with 120 total cookies, 6 cookies per row, and 4 baking trays,
    there are 5 rows of cookies on each baking tray. -/
theorem cookie_rows_per_tray :
  rows_per_tray 120 6 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_rows_per_tray_l651_65121


namespace NUMINAMATH_CALUDE_circle_C_properties_line_l_property_circle_E_fixed_points_l651_65184

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 4)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - 2*y + 4 = 0

-- Define the circle E
def circle_E (x y y1 y2 : ℝ) : Prop :=
  x^2 + y^2 - 12*x - (y1 + y2)*y - 64 = 0

theorem circle_C_properties :
  (circle_C 0 0) ∧ 
  (circle_C 6 0) ∧ 
  (∃ x : ℝ, circle_C x 1) :=
sorry

theorem line_l_property (a b : ℝ) :
  line_l a b ↔ 
  ∃ t : ℝ, 
    (t - 3)^2 + (b + 4)^2 = 25 ∧
    ((a - t)^2 + (b - 1)^2) = ((a - 2)^2 + (b + 2)^2) :=
sorry

theorem circle_E_fixed_points (y1 y2 : ℝ) :
  (y1 * y2 = -100) →
  (circle_E 16 0 y1 y2) ∧
  (circle_E (-4) 0 y1 y2) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_line_l_property_circle_E_fixed_points_l651_65184


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l651_65160

theorem simplify_and_evaluate (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) :
  ((4 - x) / (x - 1) - x) / ((x - 2) / (x - 1)) = -2 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l651_65160


namespace NUMINAMATH_CALUDE_quarter_circle_arcs_sum_l651_65143

/-- The sum of the lengths of n quarter-circle arcs, each constructed on a segment of length D/n 
    (where D is the diameter of a large circle), approaches πD/8 as n approaches infinity. -/
theorem quarter_circle_arcs_sum (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * (D / n) / 4) - π * D / 8| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circle_arcs_sum_l651_65143


namespace NUMINAMATH_CALUDE_average_of_s_and_t_l651_65140

theorem average_of_s_and_t (s t : ℝ) : 
  (1 + 3 + 7 + s + t) / 5 = 12 → (s + t) / 2 = 24.5 := by
sorry

end NUMINAMATH_CALUDE_average_of_s_and_t_l651_65140


namespace NUMINAMATH_CALUDE_one_third_vector_AB_l651_65116

/-- Given two vectors OA and OB in 2D space, prove that 1/3 of vector AB equals the specified result. -/
theorem one_third_vector_AB (OA OB : ℝ × ℝ) : 
  OA = (4, 8) → OB = (-7, -2) → (1 / 3 : ℝ) • (OB - OA) = (-11/3, -10/3) := by
  sorry

end NUMINAMATH_CALUDE_one_third_vector_AB_l651_65116


namespace NUMINAMATH_CALUDE_stock_certificate_tearing_impossible_2002_pieces_l651_65182

theorem stock_certificate_tearing (n : ℕ) : n > 0 → (∃ k : ℕ, n = 1 + 7 * k) ↔ n % 7 = 1 :=
by sorry

theorem impossible_2002_pieces : ¬(∃ k : ℕ, 2002 = 1 + 7 * k) :=
by sorry

end NUMINAMATH_CALUDE_stock_certificate_tearing_impossible_2002_pieces_l651_65182


namespace NUMINAMATH_CALUDE_right_triangle_median_length_l651_65105

theorem right_triangle_median_length (DE DF EF : ℝ) :
  DE = 15 →
  DF = 9 →
  EF = 12 →
  DE^2 = DF^2 + EF^2 →
  (DE / 2 : ℝ) = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_median_length_l651_65105


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l651_65148

-- Define a random variable following normal distribution
def normalDistribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def probability (ξ : normalDistribution 4 σ) (event : Set ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_symmetry 
  (σ : ℝ) 
  (ξ : normalDistribution 4 σ) 
  (h : probability ξ {x | x > 8} = 0.4) : 
  probability ξ {x | x < 0} = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l651_65148


namespace NUMINAMATH_CALUDE_original_group_size_l651_65130

theorem original_group_size (initial_avg : ℝ) (new_boy1 new_boy2 new_boy3 : ℝ) (new_avg : ℝ) :
  initial_avg = 35 →
  new_boy1 = 40 →
  new_boy2 = 45 →
  new_boy3 = 50 →
  new_avg = 36 →
  ∃ n : ℕ,
    n * initial_avg + new_boy1 + new_boy2 + new_boy3 = (n + 3) * new_avg ∧
    n = 27 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l651_65130


namespace NUMINAMATH_CALUDE_sunzi_problem_correct_l651_65187

/-- Represents the problem from "The Mathematical Classic of Sunzi" -/
structure SunziProblem where
  x : ℕ  -- Total number of people
  y : ℕ  -- Total number of carriages

/-- Checks if the given numbers satisfy the conditions of the problem -/
def is_valid_solution (p : SunziProblem) : Prop :=
  (p.x / 3 : ℚ) = p.y - 2 ∧ (p.x - 9) / 2 = p.y

/-- The system of equations correctly represents the Sunzi problem -/
theorem sunzi_problem_correct (p : SunziProblem) : 
  is_valid_solution p ↔ 
    (∃ (empty_carriages : ℕ), p.y = p.x / 3 + empty_carriages ∧ empty_carriages = 2) ∧
    (∃ (walking_people : ℕ), p.y = (p.x - walking_people) / 2 ∧ walking_people = 9) :=
sorry

end NUMINAMATH_CALUDE_sunzi_problem_correct_l651_65187


namespace NUMINAMATH_CALUDE_division_result_l651_65172

theorem division_result : (0.08 : ℝ) / 0.002 = 40 := by sorry

end NUMINAMATH_CALUDE_division_result_l651_65172


namespace NUMINAMATH_CALUDE_max_value_constraint_l651_65151

theorem max_value_constraint (x y : ℝ) (h : 2 * x^2 + 3 * y^2 ≤ 12) :
  |x + 2*y| ≤ Real.sqrt 22 ∧ ∃ x y : ℝ, 2 * x^2 + 3 * y^2 = 12 ∧ |x + 2*y| = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l651_65151


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l651_65115

theorem no_real_roots_for_nonzero_k (k : ℝ) (hk : k ≠ 0) :
  ∀ x : ℝ, x^2 + k*x + 2*k^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l651_65115


namespace NUMINAMATH_CALUDE_johann_manipulation_l651_65132

theorem johann_manipulation (x y k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hk : k > 1) : 
  x * k - y / k > x - y := by
  sorry

end NUMINAMATH_CALUDE_johann_manipulation_l651_65132


namespace NUMINAMATH_CALUDE_base5_412_to_base7_l651_65113

/-- Converts a base 5 number to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 7 --/
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 7) ((m % 7) :: acc)
    go n []

theorem base5_412_to_base7 :
  decimalToBase7 (base5ToDecimal [2, 1, 4]) = [2, 1, 2] :=
sorry

end NUMINAMATH_CALUDE_base5_412_to_base7_l651_65113


namespace NUMINAMATH_CALUDE_a_equals_one_m_geq_two_l651_65189

/-- The function f defined as f(x) = |x + 2a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

/-- Theorem stating that given the conditions, a must equal 1 -/
theorem a_equals_one (a : ℝ) : 
  (∀ x, f a x < 4 - 2*a ↔ -4 < x ∧ x < 0) → a = 1 := by sorry

/-- The function g defined as g(x) = |x + 2| -/
def g (x : ℝ) : ℝ := |x + 2|

/-- Theorem stating that given the conditions, m must be greater than or equal to 2 -/
theorem m_geq_two (m : ℝ) :
  (∀ x, g x - g (-2*x) ≤ x + m) → m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_a_equals_one_m_geq_two_l651_65189


namespace NUMINAMATH_CALUDE_greatest_common_factor_36_54_81_l651_65190

theorem greatest_common_factor_36_54_81 : Nat.gcd 36 (Nat.gcd 54 81) = 9 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_36_54_81_l651_65190


namespace NUMINAMATH_CALUDE_centroid_count_l651_65117

/-- A point on the perimeter of the square -/
structure PerimeterPoint where
  x : ℚ
  y : ℚ
  on_perimeter : (x = 0 ∨ x = 12) ∨ (y = 0 ∨ y = 12)
  valid_coord : (0 ≤ x ∧ x ≤ 12) ∧ (0 ≤ y ∧ y ≤ 12)

/-- The set of 48 equally spaced points on the perimeter -/
def perimeter_points : Finset PerimeterPoint :=
  sorry

/-- Predicate to check if two points are consecutive on the perimeter -/
def are_consecutive (p q : PerimeterPoint) : Prop :=
  sorry

/-- The centroid of a triangle given by three points -/
def centroid (p q r : PerimeterPoint) : ℚ × ℚ :=
  ((p.x + q.x + r.x) / 3, (p.y + q.y + r.y) / 3)

/-- The set of all possible centroids -/
def possible_centroids : Finset (ℚ × ℚ) :=
  sorry

theorem centroid_count :
  ∀ p q r : PerimeterPoint,
    p ∈ perimeter_points →
    q ∈ perimeter_points →
    r ∈ perimeter_points →
    ¬(are_consecutive p q ∨ are_consecutive q r ∨ are_consecutive r p) →
    (Finset.card possible_centroids = 1156) :=
  sorry

end NUMINAMATH_CALUDE_centroid_count_l651_65117


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l651_65111

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x| + |x + 1|

-- Theorem for the solution set of f(x) > 3
theorem solution_set_f : 
  {x : ℝ | f x > 3} = {x : ℝ | x > 1 ∨ x < -2} := by sorry

-- Theorem for the range of m
theorem range_of_m : 
  (∀ x : ℝ, m^2 + 3*m + 2*f x ≥ 0) → (m ≥ -1 ∨ m ≤ -2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l651_65111


namespace NUMINAMATH_CALUDE_A_empty_A_singleton_A_at_most_one_A_element_when_zero_A_element_when_nine_eighths_l651_65149

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

-- Theorem 1: A is empty iff a > 9/8
theorem A_empty (a : ℝ) : A a = ∅ ↔ a > 9/8 := by sorry

-- Theorem 2: A contains exactly one element iff a = 0 or a = 9/8
theorem A_singleton (a : ℝ) : (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 9/8 := by sorry

-- Theorem 3: A contains at most one element iff a = 0 or a ≥ 9/8
theorem A_at_most_one (a : ℝ) : (∀ x y, x ∈ A a → y ∈ A a → x = y) ↔ a = 0 ∨ a ≥ 9/8 := by sorry

-- Additional theorems for specific elements when A is a singleton
theorem A_element_when_zero : (∀ x, x ∈ A 0 ↔ x = 2/3) := by sorry

theorem A_element_when_nine_eighths : (∀ x, x ∈ A (9/8) ↔ x = 4/3) := by sorry

end NUMINAMATH_CALUDE_A_empty_A_singleton_A_at_most_one_A_element_when_zero_A_element_when_nine_eighths_l651_65149


namespace NUMINAMATH_CALUDE_imaginary_power_2016_l651_65191

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_2016 : i ^ 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_2016_l651_65191


namespace NUMINAMATH_CALUDE_inequality_proof_l651_65150

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1/a) + (1/b) + (1/c) ≤ (a^8 + b^8 + c^8) / (a^3 * b^3 * c^3) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l651_65150


namespace NUMINAMATH_CALUDE_strawberry_plants_l651_65127

theorem strawberry_plants (initial : ℕ) : 
  (((initial * 2) * 2) * 2) - 4 = 20 → initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_plants_l651_65127


namespace NUMINAMATH_CALUDE_trajectory_of_point_l651_65166

/-- The trajectory of a point P satisfying |PF₁| + |PF₂| = 8, where F₁ and F₂ are fixed points -/
theorem trajectory_of_point (P : ℝ × ℝ) : 
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  (dist P F₁ + dist P F₂ = 8) → 
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂ :=
by sorry


end NUMINAMATH_CALUDE_trajectory_of_point_l651_65166


namespace NUMINAMATH_CALUDE_polynomial_three_distinct_roots_l651_65152

theorem polynomial_three_distinct_roots : 
  ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∀ (x : ℝ), (x - 4) * (x^2 + 4*x + 3) = 0 ↔ x = a ∨ x = b ∨ x = c :=
by sorry

end NUMINAMATH_CALUDE_polynomial_three_distinct_roots_l651_65152


namespace NUMINAMATH_CALUDE_sum_of_odd_coefficients_l651_65107

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (3*x - 2)^6 = a₀ + a₁*(2*x - 1) + a₂*(2*x - 1)^2 + a₃*(2*x - 1)^3 + 
                      a₄*(2*x - 1)^4 + a₅*(2*x - 1)^5 + a₆*(2*x - 1)^6) →
  a₁ + a₃ + a₅ = -63/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_coefficients_l651_65107


namespace NUMINAMATH_CALUDE_sandy_worked_five_days_l651_65122

/-- The number of days Sandy worked -/
def days_worked (total_hours : ℕ) (hours_per_day : ℕ) : ℚ :=
  total_hours / hours_per_day

/-- Proof that Sandy worked 5 days -/
theorem sandy_worked_five_days (total_hours : ℕ) (hours_per_day : ℕ) 
  (h1 : total_hours = 45)
  (h2 : hours_per_day = 9) :
  days_worked total_hours hours_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_sandy_worked_five_days_l651_65122


namespace NUMINAMATH_CALUDE_sum_factorials_mod_30_l651_65195

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem sum_factorials_mod_30 :
  sum_factorials 10 % 30 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_factorials_mod_30_l651_65195


namespace NUMINAMATH_CALUDE_valid_arrangements_ten_four_l651_65114

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k people in a row within a group -/
def groupArrangements (k : ℕ) : ℕ := Nat.factorial k

/-- The number of ways to arrange n people in a row, where k specific people are not allowed to sit in k consecutive seats -/
def validArrangements (n k : ℕ) : ℕ :=
  totalArrangements n - totalArrangements (n - k + 1) * groupArrangements k

theorem valid_arrangements_ten_four :
  validArrangements 10 4 = 3507840 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_ten_four_l651_65114


namespace NUMINAMATH_CALUDE_race_outcomes_l651_65197

/-- The number of participants in the race -/
def num_participants : Nat := 6

/-- The number of podium positions (1st, 2nd, 3rd) -/
def num_podium_positions : Nat := 3

/-- Represents whether a participant can finish in a specific position -/
def can_finish (participant : Nat) (position : Nat) : Prop :=
  ¬(participant = num_participants ∧ position = num_podium_positions)

/-- The number of valid race outcomes -/
def num_valid_outcomes : Nat := 120

theorem race_outcomes :
  (∀ (p₁ p₂ p₃ : Nat), p₁ ≤ num_participants → p₂ ≤ num_participants → p₃ ≤ num_participants →
    p₁ ≠ p₂ → p₁ ≠ p₃ → p₂ ≠ p₃ →
    can_finish p₁ 1 → can_finish p₂ 2 → can_finish p₃ 3 →
    ∃! (outcome : Nat), outcome = num_valid_outcomes) :=
by sorry

end NUMINAMATH_CALUDE_race_outcomes_l651_65197


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficients_l651_65146

theorem polynomial_factor_coefficients : 
  ∃ (a b : ℤ), 
    (∃ (d : ℤ), 3 * X ^ 4 + b * X ^ 3 + 45 * X ^ 2 - 21 * X + 8 = 
      (2 * X ^ 2 - 3 * X + 2) * (a * X ^ 2 + d * X + 4)) ∧ 
    a = 3 ∧ 
    b = -27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficients_l651_65146


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l651_65199

/-- Represents the number of communities --/
def num_communities : ℕ := 3

/-- Represents the number of students --/
def num_students : ℕ := 4

/-- Represents the total number of arrangements without restrictions --/
def total_arrangements : ℕ := (num_students.choose 2) * (num_communities.factorial)

/-- Represents the number of arrangements where two specific students are in the same community --/
def same_community_arrangements : ℕ := num_communities.factorial

/-- The main theorem stating the number of valid arrangements --/
theorem valid_arrangements_count : 
  total_arrangements - same_community_arrangements = 30 :=
sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l651_65199


namespace NUMINAMATH_CALUDE_trip_time_difference_l651_65133

theorem trip_time_difference (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ) :
  speed = 60 → distance1 = 540 → distance2 = 570 →
  (distance2 - distance1) / speed * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l651_65133


namespace NUMINAMATH_CALUDE_orange_count_correct_l651_65101

/-- The number of oranges in the box after adding and removing specified quantities -/
def final_oranges (initial added removed : ℝ) : ℝ :=
  initial + added - removed

/-- Theorem stating that the final number of oranges in the box is correct -/
theorem orange_count_correct (initial added removed : ℝ) :
  final_oranges initial added removed = initial + added - removed := by
  sorry

end NUMINAMATH_CALUDE_orange_count_correct_l651_65101


namespace NUMINAMATH_CALUDE_christmas_play_volunteers_l651_65176

theorem christmas_play_volunteers 
  (total_needed : ℕ) 
  (num_classes : ℕ) 
  (teachers_volunteered : ℕ) 
  (more_needed : ℕ) 
  (h1 : total_needed = 50) 
  (h2 : num_classes = 6) 
  (h3 : teachers_volunteered = 13) 
  (h4 : more_needed = 7) :
  (total_needed - teachers_volunteered - more_needed) / num_classes = 5 := by
  sorry

end NUMINAMATH_CALUDE_christmas_play_volunteers_l651_65176


namespace NUMINAMATH_CALUDE_train_journey_duration_l651_65102

-- Define the time type
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

-- Define the function to calculate time difference
def timeDifference (t1 t2 : Time) : Time :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  let diffMinutes := totalMinutes2 - totalMinutes1
  { hours := diffMinutes / 60, minutes := diffMinutes % 60 }

-- Theorem statement
theorem train_journey_duration :
  let departureTime := { hours := 9, minutes := 20 : Time }
  let arrivalTime := { hours := 11, minutes := 30 : Time }
  timeDifference departureTime arrivalTime = { hours := 2, minutes := 10 : Time } := by
  sorry


end NUMINAMATH_CALUDE_train_journey_duration_l651_65102


namespace NUMINAMATH_CALUDE_tetrahedron_projection_areas_l651_65123

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the area of the orthogonal projection of a triangle on the xOy plane -/
def projectionAreaXOY (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the area of the orthogonal projection of a triangle on the yOz plane -/
def projectionAreaYOZ (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the area of the orthogonal projection of a triangle on the zOx plane -/
def projectionAreaZOX (p1 p2 p3 : Point3D) : ℝ := sorry

theorem tetrahedron_projection_areas :
  let A : Point3D := ⟨2, 0, 0⟩
  let B : Point3D := ⟨2, 2, 0⟩
  let C : Point3D := ⟨0, 2, 0⟩
  let D : Point3D := ⟨1, 1, Real.sqrt 2⟩
  let S₁ := projectionAreaXOY A B C + projectionAreaXOY A B D + projectionAreaXOY A C D + projectionAreaXOY B C D
  let S₂ := projectionAreaYOZ A B C + projectionAreaYOZ A B D + projectionAreaYOZ A C D + projectionAreaYOZ B C D
  let S₃ := projectionAreaZOX A B C + projectionAreaZOX A B D + projectionAreaZOX A C D + projectionAreaZOX B C D
  S₃ = S₂ ∧ S₃ ≠ S₁ := by sorry

end NUMINAMATH_CALUDE_tetrahedron_projection_areas_l651_65123


namespace NUMINAMATH_CALUDE_B_max_at_181_l651_65131

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence B_k as defined in the problem -/
def B (k : ℕ) : ℝ := (binomial 2000 k) * (0.1 ^ k)

/-- The theorem stating that B_k is maximum when k = 181 -/
theorem B_max_at_181 : ∀ k ∈ Finset.range 2001, B 181 ≥ B k := by sorry

end NUMINAMATH_CALUDE_B_max_at_181_l651_65131


namespace NUMINAMATH_CALUDE_function_value_order_l651_65153

noncomputable def f (x : ℝ) := Real.log (abs (x - 2)) + x^2 - 4*x

theorem function_value_order :
  let a := f (Real.log 9 / Real.log 2)
  let b := f (Real.log 18 / Real.log 4)
  let c := f 1
  a > c ∧ c > b :=
by sorry

end NUMINAMATH_CALUDE_function_value_order_l651_65153


namespace NUMINAMATH_CALUDE_at_least_two_equations_have_solutions_l651_65186

-- Define the functions
def f₁ (a b c x : ℝ) : ℝ := (x - b) * (x - c) - (x - a)
def f₂ (a b c x : ℝ) : ℝ := (x - c) * (x - a) - (x - b)
def f₃ (a b c x : ℝ) : ℝ := (x - a) * (x - b) - (x - c)

-- Define the theorem
theorem at_least_two_equations_have_solutions (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x : ℝ, f₁ a b c x = 0) ∧ (∃ x : ℝ, f₂ a b c x = 0) ∨
  (∃ x : ℝ, f₁ a b c x = 0) ∧ (∃ x : ℝ, f₃ a b c x = 0) ∨
  (∃ x : ℝ, f₂ a b c x = 0) ∧ (∃ x : ℝ, f₃ a b c x = 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_equations_have_solutions_l651_65186


namespace NUMINAMATH_CALUDE_fib_sum_squares_l651_65167

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: Sum of squares of consecutive Fibonacci numbers -/
theorem fib_sum_squares (n : ℕ) : (fib n)^2 + (fib (n + 1))^2 = fib (2 * n + 2) := by
  sorry

end NUMINAMATH_CALUDE_fib_sum_squares_l651_65167


namespace NUMINAMATH_CALUDE_reciprocal_in_fourth_quadrant_l651_65141

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- Define the fourth quadrant
def fourth_quadrant (w : ℂ) : Prop :=
  w.re > 0 ∧ w.im < 0

-- Theorem statement
theorem reciprocal_in_fourth_quadrant :
  fourth_quadrant (z⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_in_fourth_quadrant_l651_65141


namespace NUMINAMATH_CALUDE_total_snowballs_l651_65196

def lucy_snowballs : ℕ := 19
def peter_snowballs : ℕ := 47
def charlie_snowballs : ℕ := lucy_snowballs + 31

theorem total_snowballs : lucy_snowballs + charlie_snowballs + peter_snowballs = 116 := by
  sorry

end NUMINAMATH_CALUDE_total_snowballs_l651_65196


namespace NUMINAMATH_CALUDE_flour_weight_acceptable_l651_65142

/-- A weight is acceptable if it falls within the labeled range -/
def is_acceptable (labeled_weight : ℝ) (tolerance : ℝ) (actual_weight : ℝ) : Prop :=
  actual_weight ≥ labeled_weight - tolerance ∧ actual_weight ≤ labeled_weight + tolerance

/-- Theorem stating that 99.80 kg is acceptable for a bag labeled as 100 ± 0.25 kg -/
theorem flour_weight_acceptable :
  is_acceptable 100 0.25 99.80 := by
  sorry

end NUMINAMATH_CALUDE_flour_weight_acceptable_l651_65142


namespace NUMINAMATH_CALUDE_sum_of_fraction_parts_l651_65136

/-- The decimal representation of the number we're considering -/
def repeating_decimal : ℚ := 0.45454545

/-- Expresses the repeating decimal as a fraction -/
def as_fraction (x : ℚ) : ℚ := (100 * x - x) / 99

/-- Reduces a fraction to its lowest terms -/
def reduce_fraction (x : ℚ) : ℚ := x

theorem sum_of_fraction_parts : 
  (reduce_fraction (as_fraction repeating_decimal)).num +
  (reduce_fraction (as_fraction repeating_decimal)).den = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fraction_parts_l651_65136


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_2_10_l651_65109

-- Define the function f(x) = x^2 + 3x
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 2*x + 3

-- Theorem statement
theorem tangent_slope_at_point_2_10 :
  f_derivative 2 = 7 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_2_10_l651_65109


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l651_65100

theorem simplify_sqrt_expression (x : ℝ) (h : x < 1) :
  (x - 1) * Real.sqrt (-1 / (x - 1)) = -Real.sqrt (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l651_65100


namespace NUMINAMATH_CALUDE_aquarium_original_price_l651_65128

/-- Proves that the original price of an aquarium is $120 given the conditions of the problem -/
theorem aquarium_original_price (P : ℝ) : 
  (0.5 * P + 0.05 * (0.5 * P) = 63) → P = 120 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_original_price_l651_65128


namespace NUMINAMATH_CALUDE_number_problem_l651_65137

theorem number_problem (x : ℝ) : (0.16 * (0.40 * x) = 6) → x = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l651_65137


namespace NUMINAMATH_CALUDE_min_value_product_l651_65194

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1 / 2) :
  (x + y) * (2 * y + 3 * z) * (x * z + 2) ≥ 4 * Real.sqrt 6 ∧
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' * y' * z' = 1 / 2 ∧
    (x' + y') * (2 * y' + 3 * z') * (x' * z' + 2) = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l651_65194


namespace NUMINAMATH_CALUDE_investment_proof_l651_65129

/-- The compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_proof :
  let initial_investment : ℝ := 1000
  let interest_rate : ℝ := 0.08
  let time_period : ℕ := 6
  let final_balance : ℝ := 1586.87
  (compound_interest initial_investment interest_rate time_period) = final_balance := by
  sorry

end NUMINAMATH_CALUDE_investment_proof_l651_65129


namespace NUMINAMATH_CALUDE_no_valid_grid_l651_65159

/-- Represents a 4x4 grid with some initial values -/
structure Grid :=
  (a11 : ℝ) (a12 : ℝ) (a13 : ℝ) (a14 : ℝ)
  (a21 : ℝ) (a22 : ℝ) (a23 : ℝ) (a24 : ℝ)
  (a31 : ℝ) (a32 : ℝ) (a33 : ℝ) (a34 : ℝ)
  (a41 : ℝ) (a42 : ℝ) (a43 : ℝ) (a44 : ℝ)

/-- Checks if a sequence of 4 numbers forms an arithmetic progression -/
def isArithmeticSequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

/-- Defines the conditions for the grid based on the problem statement -/
def validGrid (g : Grid) : Prop :=
  g.a12 = 9 ∧ g.a21 = 1 ∧ g.a34 = 5 ∧ g.a43 = 8 ∧
  isArithmeticSequence g.a11 g.a12 g.a13 g.a14 ∧
  isArithmeticSequence g.a21 g.a22 g.a23 g.a24 ∧
  isArithmeticSequence g.a31 g.a32 g.a33 g.a34 ∧
  isArithmeticSequence g.a41 g.a42 g.a43 g.a44 ∧
  isArithmeticSequence g.a11 g.a21 g.a31 g.a41 ∧
  isArithmeticSequence g.a12 g.a22 g.a32 g.a42 ∧
  isArithmeticSequence g.a13 g.a23 g.a33 g.a43 ∧
  isArithmeticSequence g.a14 g.a24 g.a34 g.a44

/-- The main theorem stating that no valid grid exists -/
theorem no_valid_grid : ¬ ∃ (g : Grid), validGrid g := by
  sorry

end NUMINAMATH_CALUDE_no_valid_grid_l651_65159


namespace NUMINAMATH_CALUDE_sum_is_five_digits_l651_65120

/-- Represents a digit (1-9) -/
def Digit := Fin 9

/-- The sum of 10765, AB4, and CB is always a 5-digit number -/
theorem sum_is_five_digits (A B C : Digit) (h : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  let AB4 := 100 * A.val + 10 * B.val + 4
  let CB := 10 * C.val + B.val
  let sum := 10765 + AB4 + CB
  9999 < sum ∧ sum < 100000 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_five_digits_l651_65120


namespace NUMINAMATH_CALUDE_product_positive_l651_65155

theorem product_positive (a b : ℝ) (h1 : a > b) (h2 : b > 0) : b * (a - b) > 0 := by
  sorry

end NUMINAMATH_CALUDE_product_positive_l651_65155


namespace NUMINAMATH_CALUDE_unique_initial_pair_l651_65104

def arithmetic_mean_operation (a b : ℕ) : ℕ × ℕ :=
  if (a + b) % 2 = 0 then
    let mean := (a + b) / 2
    if mean < a then (mean, a) else (b, mean)
  else
    (a, b)

def perform_operations (n : ℕ) (pair : ℕ × ℕ) : ℕ × ℕ :=
  match n with
  | 0 => pair
  | n + 1 => perform_operations n (arithmetic_mean_operation pair.1 pair.2)

theorem unique_initial_pair :
  ∀ x : ℕ,
    x < 2015 →
    x ≠ 991 →
    ∃ i : ℕ,
      i ≤ 10 ∧
      (perform_operations i (x, 2015)).1 = (perform_operations i (x, 2015)).2 :=
sorry

end NUMINAMATH_CALUDE_unique_initial_pair_l651_65104


namespace NUMINAMATH_CALUDE_first_company_manager_percentage_l651_65119

/-- Represents a company's workforce composition -/
structure Company where
  total : ℝ
  managers : ℝ

/-- The merged company resulting from two companies -/
def MergedCompany (c1 c2 : Company) : Company where
  total := c1.total + c2.total
  managers := c1.managers + c2.managers

theorem first_company_manager_percentage (c1 c2 : Company) :
  let merged := MergedCompany c1 c2
  (c1.total = 0.25 * merged.total) →
  (merged.managers = 0.25 * merged.total) →
  (c1.managers / c1.total = 0.25) := by
  sorry

end NUMINAMATH_CALUDE_first_company_manager_percentage_l651_65119


namespace NUMINAMATH_CALUDE_eighteenth_digit_is_five_l651_65173

/-- The decimal expansion of 10000/9899 -/
def decimal_expansion : ℕ → ℕ
| 0 => 1  -- integer part
| 1 => 0  -- first decimal digit
| 2 => 1  -- second decimal digit
| n + 3 => (decimal_expansion (n + 1) + decimal_expansion (n + 2)) % 10

/-- The 18th digit after the decimal point in 10000/9899 is 5 -/
theorem eighteenth_digit_is_five : decimal_expansion 18 = 5 := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_digit_is_five_l651_65173


namespace NUMINAMATH_CALUDE_intersection_right_angle_coordinates_l651_65103

-- Define the line and parabola
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define points A and B as intersections
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ parabola A.1 A.2 ∧
  line B.1 B.2 ∧ parabola B.1 B.2 ∧
  A ≠ B

-- Define point C on the parabola
def point_on_parabola (C : ℝ × ℝ) : Prop :=
  parabola C.1 C.2

-- Define the right angle condition
def right_angle (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem intersection_right_angle_coordinates :
  ∀ A B C : ℝ × ℝ,
  intersection_points A B →
  point_on_parabola C →
  right_angle A B C →
  (C = (1, -2) ∨ C = (9, -6)) :=
sorry

end NUMINAMATH_CALUDE_intersection_right_angle_coordinates_l651_65103


namespace NUMINAMATH_CALUDE_remainder_17_49_mod_5_l651_65110

theorem remainder_17_49_mod_5 : 17^49 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_49_mod_5_l651_65110


namespace NUMINAMATH_CALUDE_expression_equals_two_l651_65193

theorem expression_equals_two : (Real.sqrt 3)^2 + (4 - Real.pi)^0 - |(-3)| + Real.sqrt 2 * Real.cos (π / 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l651_65193


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l651_65106

-- Problem 1
theorem problem_1 : (1) - 2^2 + (π - 3)^0 + 0.5^(-1) = -1 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : (x - 2*y) * (x^2 + 2*x*y + 4*y^2) = x^3 - 8*y^3 := by sorry

-- Problem 3
theorem problem_3 (a : ℝ) : a * a^2 * a^3 + (-2*a^3)^2 - a^8 / a^2 = 4*a^6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l651_65106


namespace NUMINAMATH_CALUDE_percentage_of_lower_grades_with_cars_l651_65174

theorem percentage_of_lower_grades_with_cars
  (total_students : ℕ)
  (seniors : ℕ)
  (lower_grades : ℕ)
  (senior_car_percentage : ℚ)
  (total_car_percentage : ℚ)
  (h1 : total_students = 1200)
  (h2 : seniors = 300)
  (h3 : lower_grades = 900)
  (h4 : seniors + lower_grades = total_students)
  (h5 : senior_car_percentage = 1/2)
  (h6 : total_car_percentage = 1/5)
  : (total_car_percentage * total_students - senior_car_percentage * seniors) / lower_grades = 1/10 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_lower_grades_with_cars_l651_65174


namespace NUMINAMATH_CALUDE_largest_divisible_power_of_three_l651_65118

theorem largest_divisible_power_of_three : ∃! n : ℕ, 
  (∀ k : ℕ, k ≤ n → (4^27000 - 82) % 3^k = 0) ∧
  (∀ m : ℕ, m > n → (4^27000 - 82) % 3^m ≠ 0) ∧
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_power_of_three_l651_65118


namespace NUMINAMATH_CALUDE_third_to_fourth_l651_65183

/-- An angle is in the third quadrant if it's between 180° and 270° -/
def is_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 180 + 360 * k < α ∧ α < 270 + 360 * k

/-- An angle is in the fourth quadrant if it's between 270° and 360° -/
def is_fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 270 + 360 * k < α ∧ α < 360 + 360 * k

theorem third_to_fourth (α : ℝ) (h : is_third_quadrant α) :
  is_fourth_quadrant (180 - α) :=
by sorry

end NUMINAMATH_CALUDE_third_to_fourth_l651_65183


namespace NUMINAMATH_CALUDE_complex_number_equality_l651_65192

theorem complex_number_equality (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l651_65192


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_4_l651_65147

theorem greatest_three_digit_divisible_by_3_6_4 : ∃ n : ℕ, 
  n = 984 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  n % 3 = 0 ∧ n % 6 = 0 ∧ n % 4 = 0 ∧
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 6 = 0 ∧ m % 4 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_4_l651_65147
