import Mathlib

namespace hiking_trip_days_l4136_413688

/-- Represents the hiking trip scenario -/
structure HikingTrip where
  rateUp : ℝ
  rateDown : ℝ
  distanceDown : ℝ
  days : ℝ

/-- The hiking trip satisfies the given conditions -/
def validHikingTrip (trip : HikingTrip) : Prop :=
  trip.rateUp = 6 ∧
  trip.rateDown = 1.5 * trip.rateUp ∧
  trip.distanceDown = 18 ∧
  trip.rateUp * trip.days = trip.rateDown * trip.days

/-- The number of days for the hiking trip is 2 -/
theorem hiking_trip_days (trip : HikingTrip) (h : validHikingTrip trip) : trip.days = 2 := by
  sorry


end hiking_trip_days_l4136_413688


namespace algebra_test_correct_percentage_l4136_413621

theorem algebra_test_correct_percentage (x : ℕ) (h : x > 0) :
  let total_problems := 5 * x
  let missed_problems := x
  let correct_problems := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) * 100 = 80 := by
  sorry

end algebra_test_correct_percentage_l4136_413621


namespace smallest_fraction_l4136_413648

theorem smallest_fraction (x : ℝ) (hx : x = 9) : 
  min ((x - 3) / 8) (min (8 / x) (min (8 / (x + 2)) (min (8 / (x - 2)) ((x + 3) / 8)))) = (x - 3) / 8 := by
  sorry

end smallest_fraction_l4136_413648


namespace cubic_roots_negative_real_parts_l4136_413691

theorem cubic_roots_negative_real_parts
  (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℂ, a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃ = 0 → (x.re < 0)) ↔
  ((a₀ > 0 ∧ a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0) ∨ (a₀ < 0 ∧ a₁ < 0 ∧ a₂ < 0 ∧ a₃ < 0)) ∧
  a₁ * a₂ - a₀ * a₃ > 0 :=
by sorry

end cubic_roots_negative_real_parts_l4136_413691


namespace order_of_logarithmic_fractions_l4136_413619

theorem order_of_logarithmic_fractions :
  let a := (Real.log 2) / 2
  let b := (Real.log 3) / 3
  let c := 1 / Real.exp 1
  a < b ∧ b < c := by sorry

end order_of_logarithmic_fractions_l4136_413619


namespace no_intersection_intersection_count_is_zero_l4136_413633

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|4 * x - 1|

-- Theorem statement
theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
by
  sorry

-- Define the number of intersection points
def intersection_count : ℕ := 0

-- Theorem to prove the number of intersection points is 0
theorem intersection_count_is_zero :
  intersection_count = 0 :=
by
  sorry

end no_intersection_intersection_count_is_zero_l4136_413633


namespace iced_cube_theorem_l4136_413659

/-- Represents a cube with icing on some faces -/
structure IcedCube :=
  (size : ℕ)
  (has_top_icing : Bool)
  (has_lateral_icing : Bool)
  (has_bottom_icing : Bool)

/-- Counts the number of subcubes with icing on exactly two sides -/
def count_two_sided_iced_subcubes (cube : IcedCube) : ℕ :=
  sorry

/-- The main theorem about the 5x5x5 iced cube -/
theorem iced_cube_theorem :
  let cake : IcedCube := {
    size := 5,
    has_top_icing := true,
    has_lateral_icing := true,
    has_bottom_icing := false
  }
  count_two_sided_iced_subcubes cake = 32 :=
sorry

end iced_cube_theorem_l4136_413659


namespace correct_completion_for_two_viewers_l4136_413656

/-- Represents the options for completing the sentence --/
inductive SentenceCompletion
  | NoneOfThem
  | BothOfThem
  | NoneOfWhom
  | NeitherOfWhom

/-- Represents a person who looked at the house --/
structure HouseViewer where
  wantsToBuy : Bool

/-- The correct sentence completion given two house viewers --/
def correctCompletion (viewer1 viewer2 : HouseViewer) : SentenceCompletion :=
  if !viewer1.wantsToBuy ∧ !viewer2.wantsToBuy then
    SentenceCompletion.NeitherOfWhom
  else
    SentenceCompletion.BothOfThem  -- This else case is not actually used in our theorem

theorem correct_completion_for_two_viewers (viewer1 viewer2 : HouseViewer) 
  (h1 : ¬viewer1.wantsToBuy) (h2 : ¬viewer2.wantsToBuy) :
  correctCompletion viewer1 viewer2 = SentenceCompletion.NeitherOfWhom :=
by sorry

end correct_completion_for_two_viewers_l4136_413656


namespace box_length_given_cube_fill_l4136_413672

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents the properties of cubes filling the box -/
structure CubeFill where
  sideLength : ℕ
  count : ℕ

/-- Theorem stating the relationship between box dimensions and cube fill -/
theorem box_length_given_cube_fill 
  (box : BoxDimensions) 
  (cube : CubeFill) 
  (h1 : box.width = 20) 
  (h2 : box.depth = 10) 
  (h3 : cube.count = 56) 
  (h4 : box.length * box.width * box.depth = cube.count * cube.sideLength ^ 3) 
  (h5 : cube.sideLength ∣ box.width ∧ cube.sideLength ∣ box.depth) :
  box.length = 280 := by
  sorry

#check box_length_given_cube_fill

end box_length_given_cube_fill_l4136_413672


namespace system_of_equations_l4136_413637

theorem system_of_equations (a b : ℝ) 
  (eq1 : 2 * a - b = 12) 
  (eq2 : a + 2 * b = 8) : 
  3 * a + b = 20 := by
  sorry

end system_of_equations_l4136_413637


namespace solution_values_l4136_413661

-- Define the equations
def equation_1 (a x : ℝ) : Prop := a * x + 3 = 2 * (x - a)
def equation_2 (x : ℝ) : Prop := |x - 2| - 3 = 0

-- Theorem statement
theorem solution_values (a x : ℝ) :
  equation_1 a x ∧ equation_2 x → a = -5 ∨ a = 1 := by
  sorry

end solution_values_l4136_413661


namespace greatest_integer_fraction_inequality_l4136_413675

theorem greatest_integer_fraction_inequality :
  ∀ x : ℤ, (7 : ℚ) / 9 > (x : ℚ) / 13 ↔ x ≤ 10 :=
sorry

end greatest_integer_fraction_inequality_l4136_413675


namespace cycling_time_problem_l4136_413662

theorem cycling_time_problem (total_distance : ℝ) (total_time : ℝ) (initial_speed : ℝ) (reduced_speed : ℝ)
  (h1 : total_distance = 140)
  (h2 : total_time = 7)
  (h3 : initial_speed = 25)
  (h4 : reduced_speed = 15) :
  ∃ (energetic_time : ℝ), 
    energetic_time * initial_speed + (total_time - energetic_time) * reduced_speed = total_distance ∧
    energetic_time = 7/2 := by
  sorry

end cycling_time_problem_l4136_413662


namespace jeds_speed_jeds_speed_is_89_l4136_413600

def speed_limit : ℕ := 50
def speeding_fine_per_mph : ℕ := 16
def red_light_fine : ℕ := 75
def cellphone_fine : ℕ := 120
def parking_fine : ℕ := 50
def total_fine : ℕ := 1046
def red_light_violations : ℕ := 2
def parking_violations : ℕ := 3

theorem jeds_speed : ℕ :=
  let non_speeding_fines := red_light_fine * red_light_violations + 
                            cellphone_fine + 
                            parking_fine * parking_violations
  let speeding_fine := total_fine - non_speeding_fines
  let mph_over_limit := speeding_fine / speeding_fine_per_mph
  speed_limit + mph_over_limit

#check jeds_speed

theorem jeds_speed_is_89 : jeds_speed = 89 := by
  sorry

end jeds_speed_jeds_speed_is_89_l4136_413600


namespace book_selection_theorem_l4136_413618

/-- The number of ways to select 5 books from 10 books with specific conditions -/
def select_books (n : ℕ) (k : ℕ) (adjacent_pairs : ℕ) (remaining : ℕ) : ℕ :=
  adjacent_pairs * Nat.choose remaining (k - 2)

/-- Theorem stating the number of ways to select 5 books from 10 books 
    where order doesn't matter and two of the selected books must be adjacent -/
theorem book_selection_theorem :
  select_books 10 5 9 8 = 504 := by
  sorry

end book_selection_theorem_l4136_413618


namespace grandma_last_birthday_age_l4136_413634

/-- Represents Grandma's age in various units -/
structure GrandmaAge where
  years : Nat
  months : Nat
  weeks : Nat
  days : Nat

/-- Calculates Grandma's age on her last birthday given her current age -/
def lastBirthdayAge (age : GrandmaAge) : Nat :=
  age.years + (age.months / 12) + 1

/-- Theorem stating that Grandma's age on her last birthday was 65 years -/
theorem grandma_last_birthday_age :
  let currentAge : GrandmaAge := { years := 60, months := 50, weeks := 40, days := 30 }
  lastBirthdayAge currentAge = 65 := by
  sorry

#eval lastBirthdayAge { years := 60, months := 50, weeks := 40, days := 30 }

end grandma_last_birthday_age_l4136_413634


namespace dot_product_problem_l4136_413609

theorem dot_product_problem (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  (a.1 + b.1, a.2 + b.2) = (-1, 1) →
  a.1 * b.1 + a.2 * b.2 = -4 :=
by
  sorry

end dot_product_problem_l4136_413609


namespace min_distance_to_midpoint_l4136_413671

/-- Given a line segment AB with length 4 and a point P satisfying |PA| - |PB| = 3,
    where O is the midpoint of AB, the minimum value of |OP| is 3/2. -/
theorem min_distance_to_midpoint (A B P O : EuclideanSpace ℝ (Fin 2)) :
  dist A B = 4 →
  O = midpoint ℝ A B →
  dist P A - dist P B = 3 →
  ∃ (min_dist : ℝ), min_dist = 3/2 ∧ ∀ Q, dist P A - dist Q B = 3 → dist O Q ≥ min_dist :=
by sorry

end min_distance_to_midpoint_l4136_413671


namespace probability_both_defective_six_two_two_l4136_413698

/-- The probability of both selected products being defective, given that one is defective -/
def probability_both_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ :=
  if total ≥ defective ∧ total ≥ selected ∧ selected > 0 then
    (defective.choose (selected - 1)) / (total.choose 1 * (total - 1).choose (selected - 1))
  else
    0

theorem probability_both_defective_six_two_two :
  probability_both_defective 6 2 2 = 1 / 15 := by
  sorry

end probability_both_defective_six_two_two_l4136_413698


namespace committee_selection_l4136_413613

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) : 
  Nat.choose n k = 792 :=
by sorry

end committee_selection_l4136_413613


namespace three_digit_number_proof_l4136_413602

theorem three_digit_number_proof (a : Nat) (h1 : a < 10) : 
  (100 * a + 10 * a + 5) % 9 = 8 → 100 * a + 10 * a + 5 = 665 := by
  sorry

end three_digit_number_proof_l4136_413602


namespace nathan_blankets_l4136_413676

-- Define the warmth provided by each blanket
def warmth_per_blanket : ℕ := 3

-- Define the total warmth provided by the blankets Nathan used
def total_warmth : ℕ := 21

-- Define the number of blankets Nathan used (half of the total)
def blankets_used : ℕ := total_warmth / warmth_per_blanket

-- Define the total number of blankets in Nathan's closet
def total_blankets : ℕ := 2 * blankets_used

-- Theorem stating that the total number of blankets is 14
theorem nathan_blankets : total_blankets = 14 := by
  sorry

end nathan_blankets_l4136_413676


namespace min_value_of_function_min_value_achieved_l4136_413645

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  (x^2 + x + 1) / (x - 1) ≥ 3 + 2 * Real.sqrt 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 1) :
  ∃ x₀ > 1, (x₀^2 + x₀ + 1) / (x₀ - 1) = 3 + 2 * Real.sqrt 3 :=
sorry

end min_value_of_function_min_value_achieved_l4136_413645


namespace max_k_value_l4136_413641

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 1

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := Real.log x + k * x

theorem max_k_value :
  (∃ k : ℝ, ∀ x : ℝ, x > 0 → f x ≥ g k x) →
  (∀ k : ℝ, (∀ x : ℝ, x > 0 → f x ≥ g k x) → k ≤ 1) ∧
  (∀ x : ℝ, x > 0 → f x ≥ g 1 x) :=
sorry

end max_k_value_l4136_413641


namespace fermats_little_theorem_l4136_413693

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Nat.Prime p) :
  a^p ≡ a [MOD p] := by
  sorry

end fermats_little_theorem_l4136_413693


namespace vincent_songs_before_camp_l4136_413685

/-- The number of songs Vincent knows now -/
def total_songs : ℕ := 74

/-- The number of songs Vincent learned at summer camp -/
def learned_at_camp : ℕ := 18

/-- The number of songs Vincent knew before summer camp -/
def songs_before_camp : ℕ := total_songs - learned_at_camp

theorem vincent_songs_before_camp :
  songs_before_camp = 56 :=
sorry

end vincent_songs_before_camp_l4136_413685


namespace not_right_angled_triangle_l4136_413610

theorem not_right_angled_triangle : ∃ (a b c : ℝ),
  ((a = 30 ∧ b = 60 ∧ c = 90) → a^2 + b^2 ≠ c^2) ∧
  ((a = 3*Real.sqrt 2 ∧ b = 4*Real.sqrt 2 ∧ c = 5*Real.sqrt 2) → a^2 + b^2 = c^2) ∧
  ((a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3) → a^2 + b^2 = c^2) ∧
  ((a = 5 ∧ b = 12 ∧ c = 13) → a^2 + b^2 = c^2) :=
by sorry

end not_right_angled_triangle_l4136_413610


namespace reciprocal_of_negative_two_l4136_413692

theorem reciprocal_of_negative_two :
  ∃ (x : ℚ), x * (-2) = 1 ∧ x = -1/2 := by sorry

end reciprocal_of_negative_two_l4136_413692


namespace birthday_height_calculation_l4136_413624

/-- Given an initial height and a growth rate, calculates the new height -/
def new_height (initial_height : ℝ) (growth_rate : ℝ) : ℝ :=
  initial_height * (1 + growth_rate)

/-- Proves that given an initial height of 119.7 cm and a growth rate of 5%,
    the new height is 125.685 cm -/
theorem birthday_height_calculation :
  new_height 119.7 0.05 = 125.685 := by
  sorry

end birthday_height_calculation_l4136_413624


namespace events_mutually_exclusive_not_complementary_l4136_413699

/-- A bag containing red and white balls -/
structure Bag where
  red : Nat
  white : Nat

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite (b : Bag) : Prop := sorry

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite (b : Bag) : Prop := sorry

/-- Two events are mutually exclusive -/
def mutuallyExclusive (e1 e2 : Prop) : Prop := ¬(e1 ∧ e2)

/-- Two events are complementary -/
def complementary (e1 e2 : Prop) : Prop := (e1 ∨ e2) ∧ mutuallyExclusive e1 e2

/-- The main theorem -/
theorem events_mutually_exclusive_not_complementary (b : Bag) 
  (h : b.red = 2 ∧ b.white = 2) : 
  mutuallyExclusive (exactlyOneWhite b) (exactlyTwoWhite b) ∧ 
  ¬complementary (exactlyOneWhite b) (exactlyTwoWhite b) := by
  sorry

end events_mutually_exclusive_not_complementary_l4136_413699


namespace inequality_range_l4136_413608

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 3 * a ≥ 0) ↔ a ≥ 1/2 := by
  sorry

end inequality_range_l4136_413608


namespace angle_B_is_45_degrees_l4136_413667

theorem angle_B_is_45_degrees 
  (A B C : Real) 
  (a b c : Real) 
  (triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_angle_correspondence : a = BC ∧ b = AC ∧ c = AB) 
  (equation : a^2 + c^2 = b^2 + Real.sqrt 2 * a * c) : 
  B = 45 * π / 180 := by
sorry

end angle_B_is_45_degrees_l4136_413667


namespace cubic_and_sixth_degree_polynomial_roots_l4136_413652

theorem cubic_and_sixth_degree_polynomial_roots : ∀ s : ℂ,
  s^3 - 2*s^2 + s - 1 = 0 → s^6 - 16*s - 8 = 0 := by
  sorry

end cubic_and_sixth_degree_polynomial_roots_l4136_413652


namespace negation_of_forall_x_squared_gt_one_l4136_413625

theorem negation_of_forall_x_squared_gt_one :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x₀ : ℝ, x₀^2 ≤ 1) := by
  sorry

end negation_of_forall_x_squared_gt_one_l4136_413625


namespace perpendicular_slope_l4136_413636

/-- The slope of a line perpendicular to a line passing through two given points -/
theorem perpendicular_slope (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  (- 1 / m) = 4 / 3 →
  x₁ = 3 ∧ y₁ = -7 ∧ x₂ = -5 ∧ y₂ = -1 :=
by sorry

end perpendicular_slope_l4136_413636


namespace subsets_with_three_adjacent_chairs_l4136_413660

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets of n chairs
    arranged in a circle that contain at least three adjacent chairs -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of subsets of 12 chairs arranged
    in a circle that contain at least three adjacent chairs is 1634 -/
theorem subsets_with_three_adjacent_chairs :
  subsets_with_adjacent_chairs n = 1634 := by
  sorry

end subsets_with_three_adjacent_chairs_l4136_413660


namespace sum_product_equality_l4136_413681

theorem sum_product_equality : 1.25 * 67.875 + 125 * 6.7875 + 1250 * 0.053375 = 1000 := by
  sorry

end sum_product_equality_l4136_413681


namespace profit_percentage_l4136_413680

/-- Given that the cost price of 25 articles equals the selling price of 18 articles,
    prove that the profit percentage is 700/18. -/
theorem profit_percentage (C S : ℝ) (h : 25 * C = 18 * S) :
  (S - C) / C * 100 = 700 / 18 := by
  sorry

end profit_percentage_l4136_413680


namespace wire_cutting_l4136_413629

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 90 →
  ratio = 2 / 7 →
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 20 :=
by sorry

end wire_cutting_l4136_413629


namespace total_time_theorem_l4136_413665

/-- The time Carlotta spends practicing for each minute of singing -/
def practice_time : ℕ := 3

/-- The time Carlotta spends throwing tantrums for each minute of singing -/
def tantrum_time : ℕ := 5

/-- The length of the final stage performance in minutes -/
def performance_length : ℕ := 6

/-- The total time spent per minute of singing -/
def total_time_per_minute : ℕ := 1 + practice_time + tantrum_time

/-- Theorem: The total combined amount of time Carlotta spends practicing, 
    throwing tantrums, and singing in the final stage performance is 54 minutes -/
theorem total_time_theorem : performance_length * total_time_per_minute = 54 := by
  sorry

end total_time_theorem_l4136_413665


namespace megacorp_mining_earnings_l4136_413615

/-- MegaCorp's daily earnings from mining -/
def daily_mining_earnings : ℝ := 67111111.11

/-- MegaCorp's daily earnings from oil refining -/
def daily_oil_earnings : ℝ := 5000000

/-- MegaCorp's monthly expenses -/
def monthly_expenses : ℝ := 30000000

/-- MegaCorp's fine -/
def fine : ℝ := 25600000

/-- The fine percentage of annual profits -/
def fine_percentage : ℝ := 0.01

/-- Number of days in a month (approximation) -/
def days_in_month : ℝ := 30

/-- Number of months in a year -/
def months_in_year : ℝ := 12

theorem megacorp_mining_earnings :
  fine = fine_percentage * months_in_year * (days_in_month * (daily_mining_earnings + daily_oil_earnings) - monthly_expenses) :=
by sorry

end megacorp_mining_earnings_l4136_413615


namespace special_permutations_count_l4136_413694

/-- The number of permutations of n distinct elements where a₁ is not in the 1st position,
    a₂ is not in the 2nd position, and a₃ is not in the 3rd position. -/
def special_permutations (n : ℕ) : ℕ :=
  (n^3 - 6*n^2 + 14*n - 13) * Nat.factorial (n - 3)

/-- Theorem stating that for n ≥ 3, the number of permutations of n distinct elements
    where a₁ is not in the 1st position, a₂ is not in the 2nd position, and a₃ is not
    in the 3rd position is equal to (n³ - 6n² + 14n - 13) * (n-3)! -/
theorem special_permutations_count (n : ℕ) (h : n ≥ 3) :
  special_permutations n = (n^3 - 6*n^2 + 14*n - 13) * Nat.factorial (n - 3) := by
  sorry

end special_permutations_count_l4136_413694


namespace square_area_ratio_l4136_413622

theorem square_area_ratio (x : ℝ) (hx : x > 0) : 
  (x^2) / ((3*x)^2 + (6*x)^2) = 1/45 := by
  sorry

end square_area_ratio_l4136_413622


namespace forty_percent_changed_ratings_l4136_413640

/-- Represents the survey results for parents' ratings of online class experience -/
structure SurveyResults where
  total_parents : ℕ
  upgrade_percent : ℚ
  maintain_percent : ℚ
  downgrade_percent : ℚ

/-- Calculates the percentage of parents who changed their ratings -/
def changed_ratings_percentage (results : SurveyResults) : ℚ :=
  (results.upgrade_percent + results.downgrade_percent) * 100

/-- Theorem stating that given the survey conditions, 40% of parents changed their ratings -/
theorem forty_percent_changed_ratings (results : SurveyResults) 
  (h1 : results.total_parents = 120)
  (h2 : results.upgrade_percent = 30 / 100)
  (h3 : results.maintain_percent = 60 / 100)
  (h4 : results.downgrade_percent = 10 / 100)
  (h5 : results.upgrade_percent + results.maintain_percent + results.downgrade_percent = 1) :
  changed_ratings_percentage results = 40 := by
  sorry

end forty_percent_changed_ratings_l4136_413640


namespace campground_distance_l4136_413650

theorem campground_distance (speed1 speed2 speed3 time1 time2 time3 : ℝ) 
  (h1 : speed1 = 60)
  (h2 : speed2 = 50)
  (h3 : speed3 = 55)
  (h4 : time1 = 2)
  (h5 : time2 = 3)
  (h6 : time3 = 4) :
  speed1 * time1 + speed2 * time2 + speed3 * time3 = 490 :=
by
  sorry

end campground_distance_l4136_413650


namespace sample_size_C_l4136_413639

def total_students : ℕ := 150 + 150 + 400 + 300
def students_in_C : ℕ := 400
def total_survey_size : ℕ := 40

theorem sample_size_C : 
  (students_in_C * total_survey_size) / total_students = 16 :=
sorry

end sample_size_C_l4136_413639


namespace distance_between_points_l4136_413606

/-- The distance between points A and B given specific square dimensions -/
theorem distance_between_points (small_perimeter : ℝ) (large_area : ℝ) : 
  small_perimeter = 8 → large_area = 25 → ∃ (dist : ℝ), dist^2 = 58 := by
  sorry

end distance_between_points_l4136_413606


namespace parabola_point_value_l4136_413657

/-- Prove that for a parabola y = x^2 + (a+1)x + a passing through (-1, m), m must equal 0 -/
theorem parabola_point_value (a m : ℝ) : 
  ((-1)^2 + (a + 1)*(-1) + a = m) → m = 0 := by
  sorry

end parabola_point_value_l4136_413657


namespace art_of_passing_through_walls_l4136_413647

theorem art_of_passing_through_walls (n : ℕ) : 
  (8 * Real.sqrt (8 / n) = Real.sqrt (8 * 8 / n)) ↔ n = 63 :=
sorry

end art_of_passing_through_walls_l4136_413647


namespace roots_sum_powers_l4136_413695

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 4*a + 5 = 0 → b^2 - 4*b + 5 = 0 → a^3 + a^4*b^2 + a^2*b^4 + b^3 = 154 := by
  sorry

end roots_sum_powers_l4136_413695


namespace hyperbola_asymptotes_l4136_413678

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 3)^2 / 49 - (y + 2)^2 / 36 = 1

-- Define the asymptote function
def asymptote (m c x : ℝ) (y : ℝ) : Prop :=
  y = m * x + c

-- Theorem statement
theorem hyperbola_asymptotes :
  ∃ (m₁ m₂ c : ℝ),
    m₁ = 6/7 ∧ m₂ = -6/7 ∧ c = -32/7 ∧
    (∀ (x y : ℝ), hyperbola x y →
      (asymptote m₁ c x y ∨ asymptote m₂ c x y)) :=
sorry

end hyperbola_asymptotes_l4136_413678


namespace estimate_boys_in_grade_l4136_413607

theorem estimate_boys_in_grade (total_students : ℕ) (sample_size : ℕ) (girls_in_sample : ℕ) 
  (h1 : total_students = 1200)
  (h2 : sample_size = 20)
  (h3 : girls_in_sample = 8) :
  total_students - (girls_in_sample * total_students / sample_size) = 720 := by
  sorry

end estimate_boys_in_grade_l4136_413607


namespace XAXAXA_divisible_by_seven_l4136_413663

/-- Given two digits X and A, XAXAXA is the six-digit number formed by repeating XA three times -/
def XAXAXA (X A : ℕ) : ℕ :=
  100000 * X + 10000 * A + 1000 * X + 100 * A + 10 * X + A

/-- Theorem: For any two digits X and A, XAXAXA is divisible by 7 -/
theorem XAXAXA_divisible_by_seven (X A : ℕ) (hX : X < 10) (hA : A < 10) :
  ∃ k, XAXAXA X A = 7 * k :=
sorry

end XAXAXA_divisible_by_seven_l4136_413663


namespace smallest_cube_ending_392_l4136_413635

theorem smallest_cube_ending_392 : 
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 392 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 392 → n ≤ m :=
by
  use 22
  sorry

end smallest_cube_ending_392_l4136_413635


namespace second_company_base_rate_l4136_413697

/-- The base rate of United Telephone in dollars -/
def united_base_rate : ℝ := 8.00

/-- The per-minute rate of United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The per-minute rate of the second company in dollars -/
def second_per_minute : ℝ := 0.20

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℝ := 80

/-- The base rate of the second company in dollars -/
def second_base_rate : ℝ := 12.00

/-- Theorem stating that the base rate of the second company is $12.00 -/
theorem second_company_base_rate :
  united_base_rate + united_per_minute * equal_minutes =
  second_base_rate + second_per_minute * equal_minutes :=
by sorry

end second_company_base_rate_l4136_413697


namespace smallest_number_l4136_413673

def jungkook_number : ℚ := 6 / 3
def yoongi_number : ℚ := 4
def yuna_number : ℚ := 5

theorem smallest_number : 
  jungkook_number ≤ yoongi_number ∧ jungkook_number ≤ yuna_number :=
by sorry

end smallest_number_l4136_413673


namespace circle_center_and_chord_length_l4136_413677

/-- Definition of the circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- Definition of the line y = x -/
def line_y_eq_x (x y : ℝ) : Prop := y = x

theorem circle_center_and_chord_length :
  ∃ (center_x center_y : ℝ) (chord_length : ℝ),
    (∀ x y, circle_C x y ↔ (x - center_x)^2 + (y - center_y)^2 = 1) ∧
    center_x = 1 ∧
    center_y = 0 ∧
    chord_length = Real.sqrt 2 ∧
    chord_length^2 = 2 * (1 - (1 / Real.sqrt 2)^2) :=
sorry

end circle_center_and_chord_length_l4136_413677


namespace paco_cookie_consumption_l4136_413670

/-- Represents the number of sweet cookies Paco ate -/
def sweet_cookies_eaten : ℕ := 15

/-- Represents the initial number of sweet cookies Paco had -/
def initial_sweet_cookies : ℕ := 40

/-- Represents the initial number of salty cookies Paco had -/
def initial_salty_cookies : ℕ := 25

/-- Represents the number of salty cookies Paco ate -/
def salty_cookies_eaten : ℕ := 28

theorem paco_cookie_consumption :
  sweet_cookies_eaten = 15 ∧
  initial_sweet_cookies = 40 ∧
  initial_salty_cookies = 25 ∧
  salty_cookies_eaten = 28 ∧
  salty_cookies_eaten = sweet_cookies_eaten + 13 :=
by sorry

end paco_cookie_consumption_l4136_413670


namespace water_tank_capacity_l4136_413632

/-- Represents a cylindrical water tank --/
structure WaterTank where
  capacity : ℝ
  initialWater : ℝ
  finalWater : ℝ

/-- Proves that the water tank has a capacity of 75 liters --/
theorem water_tank_capacity (tank : WaterTank)
  (h1 : tank.initialWater / tank.capacity = 1 / 3)
  (h2 : (tank.initialWater + 5) / tank.capacity = 2 / 5) :
  tank.capacity = 75 := by
  sorry

end water_tank_capacity_l4136_413632


namespace sqrt_simplification_l4136_413682

theorem sqrt_simplification :
  Real.sqrt (49 - 20 * Real.sqrt 3) = 5 - 2 * Real.sqrt 3 := by
  sorry

end sqrt_simplification_l4136_413682


namespace smallest_n_divisible_by_23_l4136_413689

theorem smallest_n_divisible_by_23 :
  ∃ (n : ℕ), (n^3 + 12*n^2 + 15*n + 180) % 23 = 0 ∧
  ∀ (m : ℕ), m < n → (m^3 + 12*m^2 + 15*m + 180) % 23 ≠ 0 :=
by
  use 10
  sorry

end smallest_n_divisible_by_23_l4136_413689


namespace inequality_proof_l4136_413658

theorem inequality_proof (x a : ℝ) (h : x < a ∧ a < 0) : x^3 > a*x ∧ a*x < 0 := by
  sorry

end inequality_proof_l4136_413658


namespace abs_ratio_eq_sqrt_seven_halves_l4136_413674

theorem abs_ratio_eq_sqrt_seven_halves (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 5*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (7/2) := by
  sorry

end abs_ratio_eq_sqrt_seven_halves_l4136_413674


namespace complex_fraction_evaluation_l4136_413649

theorem complex_fraction_evaluation :
  (2 : ℂ) / (Complex.I * (3 - Complex.I)) = (1 / 5 : ℂ) - (3 / 5 : ℂ) * Complex.I :=
by sorry

end complex_fraction_evaluation_l4136_413649


namespace power_of_product_l4136_413653

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by
  sorry

end power_of_product_l4136_413653


namespace sum_value_theorem_l4136_413620

theorem sum_value_theorem (a b c : ℚ) (h1 : |a + 1| + (b - 2)^2 = 0) (h2 : |c| = 3) :
  a + b + 2*c = 7 ∨ a + b + 2*c = -5 := by
  sorry

end sum_value_theorem_l4136_413620


namespace unique_prime_perfect_power_l4136_413638

theorem unique_prime_perfect_power : 
  ∃! p : ℕ, p.Prime ∧ p ≤ 1000 ∧ ∃ m n : ℕ, n ≥ 2 ∧ 2 * p + 1 = m^n ∧ p = 13 := by
  sorry

end unique_prime_perfect_power_l4136_413638


namespace direction_vector_y_component_l4136_413611

/-- Given a line passing through two points, prove that if its direction vector
    has a specific form, then the y-component of the direction vector is 4.5. -/
theorem direction_vector_y_component
  (p1 : ℝ × ℝ)
  (p2 : ℝ × ℝ)
  (h1 : p1 = (1, -1))
  (h2 : p2 = (5, 5))
  (direction_vector : ℝ × ℝ)
  (h3 : direction_vector.1 = 3)
  (h4 : ∃ (t : ℝ), t • (p2 - p1) = direction_vector) :
  direction_vector.2 = 4.5 := by
sorry

end direction_vector_y_component_l4136_413611


namespace bad_carrots_count_l4136_413654

theorem bad_carrots_count (olivia_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  olivia_carrots = 20 → 
  mom_carrots = 14 → 
  good_carrots = 19 → 
  olivia_carrots + mom_carrots - good_carrots = 15 := by
sorry

end bad_carrots_count_l4136_413654


namespace cube_coloring_theorem_l4136_413616

/-- Represents a point in the cube --/
inductive CubePoint
| Center
| FaceCenter
| Vertex
| EdgeCenter

/-- Represents a color --/
inductive Color
| Blue
| Red

/-- Represents a straight line in the cube --/
structure Line where
  points : List CubePoint
  aligned : points.length = 3

/-- A coloring of the cube points --/
def Coloring := CubePoint → Color

/-- The set of all points in the cube --/
def cubePoints : List CubePoint := 
  [CubePoint.Center] ++ 
  List.replicate 6 CubePoint.FaceCenter ++
  List.replicate 8 CubePoint.Vertex ++
  List.replicate 12 CubePoint.EdgeCenter

/-- Theorem: For any coloring of the cube points, there exists a line with three points of the same color --/
theorem cube_coloring_theorem :
  ∀ (coloring : Coloring),
  ∃ (line : Line),
  ∀ (p : CubePoint),
  p ∈ line.points → coloring p = coloring (line.points.get ⟨0, by sorry⟩) :=
by sorry

end cube_coloring_theorem_l4136_413616


namespace equilateral_triangle_division_l4136_413651

/-- A type representing a polygon with a given number of sides -/
def Polygon (n : ℕ) := Unit

/-- A type representing an equilateral triangle -/
def EquilateralTriangle := Unit

/-- A function that divides an equilateral triangle into two polygons -/
def divide (t : EquilateralTriangle) : Polygon 2020 × Polygon 2021 := sorry

/-- Theorem stating that an equilateral triangle can be divided into a 2020-gon and a 2021-gon -/
theorem equilateral_triangle_division :
  ∃ (t : EquilateralTriangle), ∃ (p : Polygon 2020 × Polygon 2021), divide t = p := by sorry

end equilateral_triangle_division_l4136_413651


namespace tangent_line_cubic_l4136_413684

/-- The equation of the tangent line to y = x³ at (1, 1) is 3x - y - 2 = 0 -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3) → -- The curve equation
  (x = 1 ∧ y = 1) → -- The point (1, 1) on the curve
  (3*x - y - 2 = 0) -- The equation of the tangent line
:= by sorry

end tangent_line_cubic_l4136_413684


namespace target_hit_probability_l4136_413664

theorem target_hit_probability (prob_A prob_B : ℝ) 
  (h_prob_A : prob_A = 0.6) 
  (h_prob_B : prob_B = 0.5) : 
  let prob_hit_atleast_once := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B + prob_A * prob_B
  prob_A * (1 - prob_B) / prob_hit_atleast_once + prob_A * prob_B / prob_hit_atleast_once = 3/4 := by
  sorry

end target_hit_probability_l4136_413664


namespace triangle_proof_l4136_413626

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle --/
def TriangleConditions (t : Triangle) : Prop :=
  (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C ∧
  t.b = Real.sqrt 7 ∧
  t.a + t.c = 4

theorem triangle_proof (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 4 ∧ 
  (1 / 2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 2) / 4 :=
by sorry

end triangle_proof_l4136_413626


namespace two_digit_number_50th_power_l4136_413679

theorem two_digit_number_50th_power (log2 log3 log11 : ℝ) 
  (h_log2 : log2 = 0.3010)
  (h_log3 : log3 = 0.4771)
  (h_log11 : log11 = 1.0414) :
  ∃! P : ℕ, 
    10 ≤ P ∧ P < 100 ∧ 
    (10^68 : ℝ) ≤ (P^50 : ℝ) ∧ (P^50 : ℝ) < (10^69 : ℝ) ∧
    P = 23 := by
  sorry

end two_digit_number_50th_power_l4136_413679


namespace sally_orange_balloons_l4136_413696

/-- The number of orange balloons Sally has after losing some -/
def remaining_orange_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Sally has 7 orange balloons after losing 2 -/
theorem sally_orange_balloons :
  remaining_orange_balloons 9 2 = 7 := by
  sorry

end sally_orange_balloons_l4136_413696


namespace proper_subset_of_A_l4136_413605

def A : Set ℝ := { x | x^2 < 5*x }

theorem proper_subset_of_A : Set.Subset (Set.Ioo 1 5) A ∧ (Set.Ioo 1 5) ≠ A := by sorry

end proper_subset_of_A_l4136_413605


namespace binomial_expansion_sum_l4136_413612

theorem binomial_expansion_sum (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (a - x)^8 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₅ = 56 →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 256 := by
sorry

end binomial_expansion_sum_l4136_413612


namespace doll_collection_l4136_413630

theorem doll_collection (jazmin_dolls geraldine_dolls : ℕ) 
  (h1 : jazmin_dolls = 1209) 
  (h2 : geraldine_dolls = 2186) : 
  jazmin_dolls + geraldine_dolls = 3395 := by
  sorry

end doll_collection_l4136_413630


namespace cos_24_minus_cos_48_l4136_413631

theorem cos_24_minus_cos_48 : Real.cos (24 * Real.pi / 180) - Real.cos (48 * Real.pi / 180) = 1 / 2 := by
  sorry

end cos_24_minus_cos_48_l4136_413631


namespace proposition_contrapositive_equivalence_l4136_413628

theorem proposition_contrapositive_equivalence (P Q : Prop) :
  (P → Q) ↔ (¬Q → ¬P) := by sorry

end proposition_contrapositive_equivalence_l4136_413628


namespace max_distinct_angles_for_ten_points_l4136_413686

/-- The number of points on the circle -/
def n : ℕ := 10

/-- The maximum number of distinct inscribed angle values -/
def max_distinct_angles : ℕ := 80

/-- A function that calculates the number of distinct inscribed angle values
    given the number of points on a circle -/
noncomputable def distinct_angles (points : ℕ) : ℕ := sorry

/-- Theorem stating that the maximum number of distinct inscribed angle values
    for 10 points on a circle is 80 -/
theorem max_distinct_angles_for_ten_points :
  distinct_angles n = max_distinct_angles :=
sorry

end max_distinct_angles_for_ten_points_l4136_413686


namespace triangle_not_unique_l4136_413642

/-- A triangle is defined by three side lengths -/
structure Triangle :=
  (a b c : ℝ)

/-- Predicate to check if three real numbers can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- Given one side and the sum of the other two sides, 
    the triangle is not uniquely determined -/
theorem triangle_not_unique (s : ℝ) (sum : ℝ) :
  ∃ (t1 t2 : Triangle),
    t1 ≠ t2 ∧
    t1.a = s ∧
    t2.a = s ∧
    t1.b + t1.c = sum ∧
    t2.b + t2.c = sum ∧
    is_triangle t1.a t1.b t1.c ∧
    is_triangle t2.a t2.b t2.c :=
  sorry


end triangle_not_unique_l4136_413642


namespace gcd_459_357_l4136_413683

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l4136_413683


namespace rectangular_field_area_l4136_413604

/-- Proves that a rectangular field with width one-third of length and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_field_area (width length : ℝ) (h1 : width = length / 3) (h2 : 2 * (width + length) = 72) :
  width * length = 243 := by
  sorry

end rectangular_field_area_l4136_413604


namespace find_c_l4136_413601

theorem find_c (a b c : ℝ) (x : ℝ) 
  (eq : (x + a) * (x + b) = x^2 + c*x + 12)
  (h1 : b = 4)
  (h2 : a + b = 6) : 
  c = 6 := by
sorry

end find_c_l4136_413601


namespace sum_of_100th_bracket_l4136_413614

def sequence_start : ℕ := 3

def cycle_length : ℕ := 4

def numbers_per_cycle : ℕ := 10

def target_bracket : ℕ := 100

theorem sum_of_100th_bracket :
  let total_numbers := (target_bracket - 1) / cycle_length * numbers_per_cycle
  let last_number := sequence_start + 2 * (total_numbers - 1)
  let bracket_numbers := [last_number - 6, last_number - 4, last_number - 2, last_number]
  List.sum bracket_numbers = 1992 := by
sorry

end sum_of_100th_bracket_l4136_413614


namespace stamp_theorem_l4136_413669

/-- Represents the ability to form a value using given stamp denominations -/
def can_form (n : ℕ) (k : ℕ) : Prop :=
  ∃ (a b : ℕ), k = a * n + b * (n + 2)

/-- Theorem stating that for n = 3, any value k ≥ 8 can be formed using stamps of denominations 3 and 5 -/
theorem stamp_theorem :
  ∀ k : ℕ, k ≥ 8 → can_form 3 k :=
by sorry

end stamp_theorem_l4136_413669


namespace pairing_fraction_l4136_413603

/-- Represents the number of students in each grade --/
structure Students where
  seventh : ℕ
  tenth : ℕ

/-- Represents the pairing between seventh and tenth graders --/
def Pairing (s : Students) :=
  (s.tenth / 4 : ℚ) = (s.seventh / 3 : ℚ)

/-- Calculates the fraction of students with partners --/
def fractionWithPartners (s : Students) : ℚ :=
  (s.tenth / 4 + s.seventh / 3) / (s.tenth + s.seventh)

theorem pairing_fraction (s : Students) (h : Pairing s) :
  fractionWithPartners s = 2 / 7 := by
  sorry


end pairing_fraction_l4136_413603


namespace square_inequality_l4136_413655

theorem square_inequality (a x y : ℝ) :
  (2 ≤ x ∧ x ≤ 3) ∧ (3 ≤ y ∧ y ≤ 4) →
  ((3 * x - 2 * y - a) * (3 * x - 2 * y - a^2) ≤ 0 ↔ a ≤ -4) :=
by sorry

end square_inequality_l4136_413655


namespace least_x_72_implies_n_8_l4136_413668

theorem least_x_72_implies_n_8 (x : ℕ+) (p : ℕ) (n : ℕ+) :
  Nat.Prime p →
  (∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ (x : ℚ) / (n * p : ℚ) = q) →
  (∀ y : ℕ+, y < x → ¬∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ (y : ℚ) / (n * p : ℚ) = q) →
  x = 72 →
  n = 8 :=
by sorry

end least_x_72_implies_n_8_l4136_413668


namespace expected_red_lights_value_l4136_413643

/-- The number of traffic posts -/
def n : ℕ := 3

/-- The probability of encountering a red light at each post -/
def p : ℝ := 0.4

/-- The expected number of red lights encountered -/
def expected_red_lights : ℝ := n * p

/-- Theorem: The expected number of red lights encountered is 1.2 -/
theorem expected_red_lights_value : expected_red_lights = 1.2 := by
  sorry

end expected_red_lights_value_l4136_413643


namespace matrix_power_50_l4136_413627

/-- Given a 2x2 matrix C, prove that its 50th power is equal to a specific matrix. -/
theorem matrix_power_50 (C : Matrix (Fin 2) (Fin 2) ℤ) : 
  C = !![5, 2; -16, -6] → C^50 = !![-299, -100; 800, 249] := by
  sorry

end matrix_power_50_l4136_413627


namespace triangle_side_length_l4136_413617

theorem triangle_side_length (A B C a b c : ℝ) : 
  A + C = 2 * B → 
  a + c = 8 → 
  a * c = 15 → 
  b = Real.sqrt 19 := by
  sorry

end triangle_side_length_l4136_413617


namespace three_conditions_theorem_l4136_413666

def condition1 (a b : ℕ) : Prop := (a^2 + 6*a + 8) % b = 0

def condition2 (a b : ℕ) : Prop := a^2 + a*b - 6*b^2 - 15*b - 9 = 0

def condition3 (a b : ℕ) : Prop := (a + 2*b + 2) % 4 = 0

def condition4 (a b : ℕ) : Prop := Nat.Prime (a + 6*b + 2)

def satisfiesThreeConditions (a b : ℕ) : Prop :=
  (condition1 a b ∧ condition2 a b ∧ condition3 a b) ∨
  (condition1 a b ∧ condition2 a b ∧ condition4 a b) ∨
  (condition1 a b ∧ condition3 a b ∧ condition4 a b) ∨
  (condition2 a b ∧ condition3 a b ∧ condition4 a b)

theorem three_conditions_theorem :
  ∀ a b : ℕ, satisfiesThreeConditions a b ↔ ((a = 5 ∧ b = 1) ∨ (a = 17 ∧ b = 7)) :=
sorry

end three_conditions_theorem_l4136_413666


namespace grandpa_xiaoqiang_age_relation_l4136_413623

theorem grandpa_xiaoqiang_age_relation (x : ℕ) : 
  66 - x = 7 * (12 - x) ↔ 
  (∃ (grandpa_age xiaoqiang_age : ℕ), 
    grandpa_age = 66 ∧ 
    xiaoqiang_age = 12 ∧ 
    grandpa_age - x = 7 * (xiaoqiang_age - x)) :=
by sorry

end grandpa_xiaoqiang_age_relation_l4136_413623


namespace smallest_sum_after_slice_l4136_413644

-- Define the structure of a die
structure Die :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)

-- Define the cube structure
structure Cube :=
  (dice : Fin 27 → Die)

-- Define the function to calculate the sum of visible faces
def sum_visible_faces (c : Cube) : Nat :=
  -- Implementation details omitted
  sorry

-- Main theorem
theorem smallest_sum_after_slice (c : Cube) : sum_visible_faces c ≥ 98 :=
  sorry

end smallest_sum_after_slice_l4136_413644


namespace equation_solution_l4136_413687

theorem equation_solution : ∃ x : ℚ, (27 / 4 : ℚ) * x - 18 = 3 * x + 27 ∧ x = 12 := by
  sorry

end equation_solution_l4136_413687


namespace emilees_earnings_l4136_413690

/-- Given the earnings and work conditions of Jermaine, Terrence, and Emilee, prove Emilee's earnings. -/
theorem emilees_earnings 
  (total_earnings : ℝ)
  (j_hours r_j : ℝ)
  (t_hours r_t : ℝ)
  (e_hours r_e : ℝ)
  (h1 : total_earnings = 90)
  (h2 : r_j * j_hours = r_t * t_hours + 5)
  (h3 : r_t * t_hours = 30)
  (h4 : total_earnings = r_j * j_hours + r_t * t_hours + r_e * e_hours) :
  r_e * e_hours = 25 := by
  sorry

end emilees_earnings_l4136_413690


namespace diagonals_150_sided_polygon_l4136_413646

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals in a 150-sided polygon is 11025 -/
theorem diagonals_150_sided_polygon :
  num_diagonals 150 = 11025 := by
  sorry

end diagonals_150_sided_polygon_l4136_413646
