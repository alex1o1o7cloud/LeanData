import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_area_l689_68919

/-- The area of a rectangle with length 15 cm and width 0.9 times its length is 202.5 cm². -/
theorem rectangle_area : 
  let length : ℝ := 15
  let width : ℝ := 0.9 * length
  length * width = 202.5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l689_68919


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l689_68942

-- Define the quadratic function
def f (x : ℝ) := x^2 + x - 2

-- Define the solution set
def solution_set := {x : ℝ | x < -2 ∨ x > 1}

-- Theorem stating that the solution set is correct
theorem quadratic_inequality_solution :
  ∀ x : ℝ, f x > 0 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l689_68942


namespace NUMINAMATH_CALUDE_ollie_fewer_than_angus_l689_68951

/-- The number of fish caught by Patrick -/
def patrick_fish : ℕ := 8

/-- The number of fish caught by Ollie -/
def ollie_fish : ℕ := 5

/-- The number of fish caught by Angus -/
def angus_fish : ℕ := patrick_fish + 4

/-- The difference between Angus's and Ollie's fish catch -/
def fish_difference : ℕ := angus_fish - ollie_fish

theorem ollie_fewer_than_angus : fish_difference = 7 := by sorry

end NUMINAMATH_CALUDE_ollie_fewer_than_angus_l689_68951


namespace NUMINAMATH_CALUDE_initial_puppies_count_l689_68993

/-- The number of puppies Alyssa had initially -/
def initial_puppies : ℕ := 7

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 5

/-- The number of puppies Alyssa has left -/
def puppies_left : ℕ := 2

/-- Theorem stating that the initial number of puppies equals the sum of puppies given away and puppies left -/
theorem initial_puppies_count : initial_puppies = puppies_given_away + puppies_left := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l689_68993


namespace NUMINAMATH_CALUDE_orange_put_back_l689_68910

theorem orange_put_back (apple_price orange_price : ℚ)
  (total_fruit : ℕ) (initial_avg_price final_avg_price : ℚ)
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruit = 10)
  (h4 : initial_avg_price = 54/100)
  (h5 : final_avg_price = 45/100) :
  ∃ (oranges_to_put_back : ℕ),
    oranges_to_put_back = 6 ∧
    ∃ (initial_apples initial_oranges : ℕ),
      initial_apples + initial_oranges = total_fruit ∧
      (initial_apples * apple_price + initial_oranges * orange_price) / total_fruit = initial_avg_price ∧
      ∃ (final_oranges : ℕ),
        final_oranges = initial_oranges - oranges_to_put_back ∧
        (initial_apples * apple_price + final_oranges * orange_price) / (initial_apples + final_oranges) = final_avg_price :=
by
  sorry

end NUMINAMATH_CALUDE_orange_put_back_l689_68910


namespace NUMINAMATH_CALUDE_remainder_theorem_l689_68948

-- Define the polynomial P(x) = x^100 - 2x^51 + 1
def P (x : ℝ) : ℝ := x^100 - 2*x^51 + 1

-- Define the divisor polynomial D(x) = x^2 - 1
def D (x : ℝ) : ℝ := x^2 - 1

-- Define the remainder polynomial R(x) = -2x + 2
def R (x : ℝ) : ℝ := -2*x + 2

-- Theorem statement
theorem remainder_theorem : 
  ∃ (Q : ℝ → ℝ), ∀ x, P x = Q x * D x + R x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l689_68948


namespace NUMINAMATH_CALUDE_typhoon_fallen_trees_l689_68924

/-- Represents the number of trees that fell during a typhoon --/
structure FallenTrees where
  narra : ℕ
  mahogany : ℕ

/-- Represents the initial and final state of trees on the farm --/
structure FarmState where
  initialNarra : ℕ
  initialMahogany : ℕ
  finalTotal : ℕ

def replantedTrees (fallen : FallenTrees) : ℕ :=
  2 * fallen.narra + 3 * fallen.mahogany

theorem typhoon_fallen_trees (farm : FarmState) 
  (h1 : farm.initialNarra = 30)
  (h2 : farm.initialMahogany = 50)
  (h3 : farm.finalTotal = 88) :
  ∃ (fallen : FallenTrees),
    fallen.mahogany = fallen.narra + 1 ∧
    farm.finalTotal = 
      farm.initialNarra + farm.initialMahogany - 
      (fallen.narra + fallen.mahogany) + 
      replantedTrees fallen ∧
    fallen.narra + fallen.mahogany = 5 :=
  sorry


end NUMINAMATH_CALUDE_typhoon_fallen_trees_l689_68924


namespace NUMINAMATH_CALUDE_winter_sports_camp_l689_68937

theorem winter_sports_camp (total_students : ℕ) (boys girls : ℕ) (pine_students oak_students : ℕ)
  (seventh_grade eighth_grade : ℕ) (pine_girls : ℕ) :
  total_students = 120 →
  boys = 70 →
  girls = 50 →
  pine_students = 70 →
  oak_students = 50 →
  seventh_grade = 60 →
  eighth_grade = 60 →
  pine_girls = 30 →
  pine_students / 2 = seventh_grade →
  ∃ (oak_eighth_boys : ℕ), oak_eighth_boys = 15 :=
by sorry

end NUMINAMATH_CALUDE_winter_sports_camp_l689_68937


namespace NUMINAMATH_CALUDE_train_speed_l689_68906

/-- Proves that a train with given length, crossing a platform of given length in a given time, has a specific speed in km/h -/
theorem train_speed (train_length platform_length : Real) (crossing_time : Real) : 
  train_length = 450 ∧ 
  platform_length = 250.056 ∧ 
  crossing_time = 20 →
  (train_length + platform_length) / crossing_time * 3.6 = 126.01008 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l689_68906


namespace NUMINAMATH_CALUDE_sum_of_fractions_l689_68967

theorem sum_of_fractions : (2 : ℚ) / 7 + 8 / 10 = 38 / 35 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l689_68967


namespace NUMINAMATH_CALUDE_circle_center_on_line_ab_range_l689_68986

theorem circle_center_on_line_ab_range :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ a*x - b*y + 1 = 0) →
  a*b ≤ 1/8 ∧ ∀ ε > 0, ∃ (a' b' : ℝ), a'*b' < -ε :=
by sorry

end NUMINAMATH_CALUDE_circle_center_on_line_ab_range_l689_68986


namespace NUMINAMATH_CALUDE_dance_team_size_l689_68962

theorem dance_team_size (initial_size : ℕ) (quit : ℕ) (new_members : ℕ) : 
  initial_size = 25 → quit = 8 → new_members = 13 → 
  initial_size - quit + new_members = 30 := by
  sorry

end NUMINAMATH_CALUDE_dance_team_size_l689_68962


namespace NUMINAMATH_CALUDE_root_difference_quadratic_equation_l689_68920

theorem root_difference_quadratic_equation : ∃ (x y : ℝ), 
  (x^2 + 40*x + 300 = -48) ∧ 
  (y^2 + 40*y + 300 = -48) ∧ 
  x ≠ y ∧ 
  |x - y| = 16 := by
sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_equation_l689_68920


namespace NUMINAMATH_CALUDE_sum_of_numbers_l689_68956

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_prod : x * y = 12) (h_recip : 1 / x = 3 * (1 / y)) : x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l689_68956


namespace NUMINAMATH_CALUDE_birds_on_fence_l689_68935

theorem birds_on_fence (initial_birds : ℕ) (new_birds : ℕ) : 
  initial_birds = 1 → new_birds = 4 → initial_birds + new_birds = 5 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l689_68935


namespace NUMINAMATH_CALUDE_initial_column_size_l689_68996

/-- The number of people in each column initially -/
def people_per_column : ℕ := 30

/-- The total number of people -/
def total_people : ℕ := people_per_column * 16

/-- The number of columns formed when 48 people stand in each column -/
def columns_with_48 : ℕ := total_people / 48

theorem initial_column_size :
  (total_people = people_per_column * 16) ∧
  (total_people = 48 * 10) ∧
  (columns_with_48 = 10) →
  people_per_column = 30 := by
  sorry

end NUMINAMATH_CALUDE_initial_column_size_l689_68996


namespace NUMINAMATH_CALUDE_woodworker_productivity_increase_l689_68990

/-- Woodworker's productivity increase problem -/
theorem woodworker_productivity_increase
  (normal_days : ℕ)
  (normal_parts : ℕ)
  (new_days : ℕ)
  (extra_parts : ℕ)
  (h1 : normal_days = 24)
  (h2 : normal_parts = 360)
  (h3 : new_days = 22)
  (h4 : extra_parts = 80) :
  (normal_parts + extra_parts) / new_days - normal_parts / normal_days = 5 :=
by sorry

end NUMINAMATH_CALUDE_woodworker_productivity_increase_l689_68990


namespace NUMINAMATH_CALUDE_trig_identity_l689_68925

theorem trig_identity (α : Real) (h : Real.sin (π/3 - α) = 1/3) :
  Real.cos (π/3 + 2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l689_68925


namespace NUMINAMATH_CALUDE_michael_basketball_points_l689_68952

theorem michael_basketball_points :
  ∀ (junior_points : ℝ),
    (junior_points + (junior_points * 1.2) = 572) →
    junior_points = 260 := by
  sorry

end NUMINAMATH_CALUDE_michael_basketball_points_l689_68952


namespace NUMINAMATH_CALUDE_total_payment_for_bikes_l689_68987

-- Define the payment for painting a bike
def paint_payment : ℕ := 5

-- Define the additional payment for selling a bike
def sell_additional : ℕ := 8

-- Define the number of bikes
def num_bikes : ℕ := 8

-- Theorem to prove
theorem total_payment_for_bikes : 
  (paint_payment + (paint_payment + sell_additional)) * num_bikes = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_for_bikes_l689_68987


namespace NUMINAMATH_CALUDE_math_team_selection_l689_68902

theorem math_team_selection (boys : ℕ) (girls : ℕ) (team_size : ℕ) : 
  boys = 10 → girls = 12 → team_size = 8 → 
  Nat.choose (boys + girls) team_size = 319770 := by
sorry

end NUMINAMATH_CALUDE_math_team_selection_l689_68902


namespace NUMINAMATH_CALUDE_least_multiple_17_above_500_l689_68978

theorem least_multiple_17_above_500 : ∃ (n : ℕ), n * 17 = 510 ∧ 
  510 > 500 ∧ 
  (∀ m : ℕ, m * 17 > 500 → m * 17 ≥ 510) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_17_above_500_l689_68978


namespace NUMINAMATH_CALUDE_football_field_theorem_l689_68974

/-- Represents a rectangular football field -/
structure FootballField where
  length : ℝ  -- length in centimeters
  width : ℝ   -- width in meters
  perimeter_condition : 2 * (length / 100 + width) > 350
  area_condition : (length / 100) * width < 7560

/-- Checks if a field meets international match requirements -/
def is_international_match_compliant (field : FootballField) : Prop :=
  100 ≤ field.length / 100 ∧ field.length / 100 ≤ 110 ∧
  64 ≤ field.width ∧ field.width ≤ 75

theorem football_field_theorem (field : FootballField) 
  (h_width : field.width = 70) :
  (10.5 < field.length / 100 ∧ field.length / 100 < 108) ∧
  is_international_match_compliant field := by
  sorry

end NUMINAMATH_CALUDE_football_field_theorem_l689_68974


namespace NUMINAMATH_CALUDE_age_difference_l689_68966

theorem age_difference (a b c d : ℕ) : 
  (a + b = b + c + 13) →
  (b + d = c + d + 7) →
  (a + d = 2 * c - 12) →
  (a = c + 13) :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l689_68966


namespace NUMINAMATH_CALUDE_prime_product_l689_68953

theorem prime_product (p q : ℕ) : 
  Prime p ∧ Prime q ∧ Prime (q^2 - p^2) → p * q = 6 :=
by sorry

end NUMINAMATH_CALUDE_prime_product_l689_68953


namespace NUMINAMATH_CALUDE_triangle_formation_l689_68912

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  can_form_triangle 3 6 8 ∧
  can_form_triangle 3 8 9 ∧
  ¬(can_form_triangle 3 6 9) ∧
  can_form_triangle 6 8 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l689_68912


namespace NUMINAMATH_CALUDE_harry_seed_cost_l689_68992

/-- The cost of seeds for Harry's garden --/
def seedCost (pumpkinPrice tomatoPrice pepperPrice : ℚ)
             (pumpkinQty tomatoQty pepperQty : ℕ) : ℚ :=
  pumpkinPrice * pumpkinQty + tomatoPrice * tomatoQty + pepperPrice * pepperQty

/-- Theorem stating the total cost of seeds for Harry --/
theorem harry_seed_cost :
  seedCost (5/2) (3/2) (9/10) 3 4 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_harry_seed_cost_l689_68992


namespace NUMINAMATH_CALUDE_total_animals_equals_total_humps_l689_68911

/-- Represents the composition of a herd of animals -/
structure Herd where
  horses : ℕ
  twoHumpedCamels : ℕ
  oneHumpedCamels : ℕ

/-- Calculates the total number of humps in the herd -/
def totalHumps (h : Herd) : ℕ :=
  2 * h.twoHumpedCamels + h.oneHumpedCamels

/-- Calculates the total number of animals in the herd -/
def totalAnimals (h : Herd) : ℕ :=
  h.horses + h.twoHumpedCamels + h.oneHumpedCamels

/-- Theorem stating that the total number of animals equals the total number of humps
    under specific conditions -/
theorem total_animals_equals_total_humps (h : Herd) :
  h.horses = h.twoHumpedCamels →
  totalHumps h = 200 →
  totalAnimals h = 200 := by
  sorry

#check total_animals_equals_total_humps

end NUMINAMATH_CALUDE_total_animals_equals_total_humps_l689_68911


namespace NUMINAMATH_CALUDE_sequence_a_property_l689_68994

def sequence_a (n : ℕ) : ℚ :=
  3 / (15 * n - 14)

theorem sequence_a_property :
  (sequence_a 1 = 3) ∧
  (∀ n : ℕ, n > 0 → 1 / (sequence_a (n + 1) + 1) - 1 / (sequence_a n) = 5) :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_property_l689_68994


namespace NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_x_l689_68958

theorem negation_of_forall_x_squared_gt_x :
  (¬ ∀ x : ℕ, x^2 > x) ↔ (∃ x₀ : ℕ, x₀^2 ≤ x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_x_l689_68958


namespace NUMINAMATH_CALUDE_line_through_points_l689_68961

/-- A line passing through (0, -2) and (1, 0) also passes through (7, b). Prove b = 12. -/
theorem line_through_points (b : ℝ) : 
  (∃ m c : ℝ, (0 = m * 0 + c ∧ -2 = m * 0 + c) ∧ 
              (0 = m * 1 + c) ∧ 
              (b = m * 7 + c)) → 
  b = 12 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l689_68961


namespace NUMINAMATH_CALUDE_cookies_theorem_l689_68960

/-- The number of cookies Mona brought -/
def mona_cookies : ℕ := 20

/-- The number of cookies Jasmine brought -/
def jasmine_cookies : ℕ := mona_cookies - 5

/-- The number of cookies Rachel brought -/
def rachel_cookies : ℕ := jasmine_cookies + 10

/-- The total number of cookies brought by Mona, Jasmine, and Rachel -/
def total_cookies : ℕ := mona_cookies + jasmine_cookies + rachel_cookies

theorem cookies_theorem : total_cookies = 60 := by
  sorry

end NUMINAMATH_CALUDE_cookies_theorem_l689_68960


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l689_68915

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- Theorem 1: Solution set for a = 2
theorem solution_set_a_2 : 
  {x : ℝ | f 2 x ≤ 1} = {x : ℝ | -1/3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2: Range of a for the inequality to hold
theorem range_of_a : 
  {a : ℝ | ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l689_68915


namespace NUMINAMATH_CALUDE_initial_kids_count_l689_68933

/-- The number of kids still awake after the first round of napping -/
def kids_after_first_round (initial : ℕ) : ℕ := initial / 2

/-- The number of kids still awake after the second round of napping -/
def kids_after_second_round (initial : ℕ) : ℕ := kids_after_first_round initial / 2

/-- Theorem stating that the initial number of kids ready for a nap is 20 -/
theorem initial_kids_count : ∃ (initial : ℕ), 
  kids_after_second_round initial = 5 ∧ initial = 20 := by
  sorry

#check initial_kids_count

end NUMINAMATH_CALUDE_initial_kids_count_l689_68933


namespace NUMINAMATH_CALUDE_sequence_property_l689_68931

def is_arithmetic_progression (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_progression (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

def has_common_difference (a b c d : ℝ) (diff : ℝ) : Prop :=
  b - a = diff ∧ c - b = diff ∧ d - c = diff

theorem sequence_property (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ ∧ a₅ < a₆ ∧ a₆ < a₇ ∧ a₇ < a₈ →
  ((has_common_difference a₁ a₂ a₃ a₄ 4 ∧ has_common_difference a₅ a₆ a₇ a₈ 36) ∨
   (has_common_difference a₂ a₃ a₄ a₅ 4 ∧ has_common_difference a₅ a₆ a₇ a₈ 36) ∨
   (has_common_difference a₁ a₂ a₃ a₄ 4 ∧ has_common_difference a₄ a₅ a₆ a₇ 36) ∨
   (has_common_difference a₂ a₃ a₄ a₅ 4 ∧ has_common_difference a₄ a₅ a₆ a₇ 36) ∨
   (has_common_difference a₁ a₂ a₃ a₄ 36 ∧ has_common_difference a₅ a₆ a₇ a₈ 4)) →
  (is_geometric_progression a₂ a₃ a₄ a₅ ∨ is_geometric_progression a₃ a₄ a₅ a₆ ∨
   is_geometric_progression a₄ a₅ a₆ a₇) →
  a₈ = 126 ∨ a₈ = 6 :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l689_68931


namespace NUMINAMATH_CALUDE_two_letter_selection_count_l689_68959

def word : String := "УЧЕБНИК"

def is_vowel (c : Char) : Bool :=
  c = 'У' || c = 'Е' || c = 'И'

def is_consonant (c : Char) : Bool :=
  c = 'Ч' || c = 'Б' || c = 'Н' || c = 'К'

def count_vowels (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

def count_consonants (s : String) : Nat :=
  s.toList.filter is_consonant |>.length

theorem two_letter_selection_count :
  count_vowels word * count_consonants word = 12 :=
by sorry

end NUMINAMATH_CALUDE_two_letter_selection_count_l689_68959


namespace NUMINAMATH_CALUDE_base8_sum_3_to_100_l689_68946

/-- Converts a base 8 number to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 8 -/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSequenceSum (first last n : ℕ) : ℕ :=
  n * (first + last) / 2

theorem base8_sum_3_to_100 :
  let first := 3
  let last := base8ToBase10 100
  let n := last - first + 1
  base10ToBase8 (arithmeticSequenceSum first last n) = 4035 := by
  sorry

end NUMINAMATH_CALUDE_base8_sum_3_to_100_l689_68946


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l689_68922

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a > b → a - b > -2) ∧ ¬(a - b > -2 → a > b) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l689_68922


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l689_68970

def i : ℂ := Complex.I

theorem sum_of_powers_of_i : 
  (Finset.range 2015).sum (λ n => i ^ n) = i :=
sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l689_68970


namespace NUMINAMATH_CALUDE_min_points_on_circle_l689_68914

theorem min_points_on_circle (circle_length : ℕ) (h : circle_length = 1956) :
  let min_points := 1304
  ∀ n : ℕ, n < min_points →
    ¬(∀ p : ℕ, p < n →
      (∃! q : ℕ, q < n ∧ (q - p) % circle_length = 1) ∧
      (∃! r : ℕ, r < n ∧ (r - p) % circle_length = 2)) ∧
  (∀ p : ℕ, p < min_points →
    (∃! q : ℕ, q < min_points ∧ (q - p) % circle_length = 1) ∧
    (∃! r : ℕ, r < min_points ∧ (r - p) % circle_length = 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_points_on_circle_l689_68914


namespace NUMINAMATH_CALUDE_shortening_powers_l689_68945

def is_power (n : ℕ) (k : ℕ) : Prop :=
  ∃ m : ℕ, n = m^k

def shorten (n : ℕ) : ℕ :=
  n / 10

theorem shortening_powers (n : ℕ) :
  n > 1000000 →
  is_power (shorten n) 2 →
  is_power (shorten (shorten n)) 3 →
  is_power (shorten (shorten (shorten n))) 4 →
  is_power (shorten (shorten (shorten (shorten n)))) 5 →
  is_power (shorten (shorten (shorten (shorten (shorten n))))) 6 :=
by sorry

end NUMINAMATH_CALUDE_shortening_powers_l689_68945


namespace NUMINAMATH_CALUDE_tim_cabinet_price_l689_68901

/-- The price Tim paid for a cabinet after discount -/
theorem tim_cabinet_price (original_price : ℝ) (discount_percentage : ℝ) 
  (h1 : original_price = 1200)
  (h2 : discount_percentage = 15) : 
  original_price * (1 - discount_percentage / 100) = 1020 := by
  sorry

end NUMINAMATH_CALUDE_tim_cabinet_price_l689_68901


namespace NUMINAMATH_CALUDE_smallest_positive_integer_3003m_66666n_l689_68998

theorem smallest_positive_integer_3003m_66666n : 
  (∃ (k : ℕ), k > 0 ∧ ∀ (x : ℕ), x > 0 → (∃ (m n : ℤ), x = 3003 * m + 66666 * n) → k ≤ x) ∧
  (∃ (m n : ℤ), 3 = 3003 * m + 66666 * n) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_3003m_66666n_l689_68998


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_nonnegative_l689_68980

-- Define the function f
def f (x a b : ℝ) : ℝ := x^2 + |x - a| + b

-- State the theorem
theorem decreasing_function_implies_a_nonnegative 
  (h : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ ≤ 0 → f x₂ a b ≤ f x₁ a b) : 
  a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_nonnegative_l689_68980


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l689_68926

theorem smallest_number_divisible (n : ℕ) : n ≥ 62 →
  (∃ (k : ℕ), n - 8 = 18 * k ∧ n - 8 ≥ 44) →
  (∀ (m : ℕ), m < n →
    ¬(∃ (l : ℕ), m - 8 = 18 * l ∧ m - 8 ≥ 44)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l689_68926


namespace NUMINAMATH_CALUDE_max_value_of_b_l689_68973

theorem max_value_of_b (a b : ℤ) : 
  (a + b)^2 + a*(a + b) + b = 0 → b ≤ 9 := by sorry

end NUMINAMATH_CALUDE_max_value_of_b_l689_68973


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_2y_l689_68988

theorem max_value_of_x_plus_2y (x y : ℝ) (h : x^2 - x*y + y^2 = 1) :
  x + 2*y ≤ 2 * Real.sqrt 21 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_2y_l689_68988


namespace NUMINAMATH_CALUDE_nancy_history_marks_l689_68979

def american_literature : ℕ := 66
def home_economics : ℕ := 52
def physical_education : ℕ := 68
def art : ℕ := 89
def average_marks : ℕ := 70
def total_subjects : ℕ := 5

theorem nancy_history_marks :
  ∃ history : ℕ,
    history = average_marks * total_subjects - (american_literature + home_economics + physical_education + art) ∧
    history = 75 := by
  sorry

end NUMINAMATH_CALUDE_nancy_history_marks_l689_68979


namespace NUMINAMATH_CALUDE_repeating_decimal_35_eq_fraction_l689_68944

/-- Represents a repeating decimal where the digits 35 repeat infinitely after the decimal point. -/
def repeating_decimal_35 : ℚ :=
  35 / 99

/-- The theorem states that the repeating decimal 0.353535... is equal to the fraction 35/99. -/
theorem repeating_decimal_35_eq_fraction :
  repeating_decimal_35 = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_35_eq_fraction_l689_68944


namespace NUMINAMATH_CALUDE_skincare_fraction_is_two_fifths_l689_68907

/-- Represents Susie's babysitting and spending scenario -/
structure BabysittingScenario where
  hours_per_day : ℕ
  rate_per_hour : ℕ
  days_per_week : ℕ
  makeup_fraction : ℚ
  money_left : ℕ

/-- Calculates the fraction spent on skincare products given a babysitting scenario -/
def skincare_fraction (scenario : BabysittingScenario) : ℚ :=
  -- Definition to be proved
  2 / 5

/-- Theorem stating that given the specific scenario, the fraction spent on skincare is 2/5 -/
theorem skincare_fraction_is_two_fifths :
  let scenario : BabysittingScenario := {
    hours_per_day := 3,
    rate_per_hour := 10,
    days_per_week := 7,
    makeup_fraction := 3 / 10,
    money_left := 63
  }
  skincare_fraction scenario = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_skincare_fraction_is_two_fifths_l689_68907


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l689_68900

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 - 2*(m-2)*x₁ + m^2 = 0 ∧ 
   x₂^2 - 2*(m-2)*x₂ + m^2 = 0) → 
  m < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l689_68900


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l689_68965

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 23 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 23 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l689_68965


namespace NUMINAMATH_CALUDE_current_books_count_l689_68903

/-- The number of books in a library over time -/
def library_books (initial_old_books : ℕ) (bought_two_years_ago : ℕ) (bought_last_year : ℕ) (donated_this_year : ℕ) : ℕ :=
  initial_old_books + bought_two_years_ago + bought_last_year - donated_this_year

/-- Theorem: The current number of books in the library is 1000 -/
theorem current_books_count :
  let initial_old_books : ℕ := 500
  let bought_two_years_ago : ℕ := 300
  let bought_last_year : ℕ := bought_two_years_ago + 100
  let donated_this_year : ℕ := 200
  library_books initial_old_books bought_two_years_ago bought_last_year donated_this_year = 1000 := by
  sorry

end NUMINAMATH_CALUDE_current_books_count_l689_68903


namespace NUMINAMATH_CALUDE_toy_cost_l689_68929

/-- Given Roger's initial amount, the cost of a game, and the number of toys he can buy,
    prove that each toy costs $7. -/
theorem toy_cost (initial_amount : ℕ) (game_cost : ℕ) (num_toys : ℕ) :
  initial_amount = 68 →
  game_cost = 47 →
  num_toys = 3 →
  (initial_amount - game_cost) / num_toys = 7 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_l689_68929


namespace NUMINAMATH_CALUDE_existence_of_number_with_prime_multiples_l689_68950

theorem existence_of_number_with_prime_multiples : ∃ x : ℝ, 
  (∃ p : ℕ, Nat.Prime p ∧ (10 : ℝ) * x = p) ∧ 
  (∃ q : ℕ, Nat.Prime q ∧ (15 : ℝ) * x = q) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_number_with_prime_multiples_l689_68950


namespace NUMINAMATH_CALUDE_enthalpy_change_reaction_l689_68975

/-- Standard enthalpy of formation for Na₂O (s) in kJ/mol -/
def ΔH_f_Na2O : ℝ := -416

/-- Standard enthalpy of formation for H₂O (l) in kJ/mol -/
def ΔH_f_H2O : ℝ := -286

/-- Standard enthalpy of formation for NaOH (s) in kJ/mol -/
def ΔH_f_NaOH : ℝ := -427.8

/-- Standard enthalpy change of the reaction Na₂O + H₂O → 2NaOH at 298 K -/
def ΔH_reaction : ℝ := 2 * ΔH_f_NaOH - (ΔH_f_Na2O + ΔH_f_H2O)

theorem enthalpy_change_reaction :
  ΔH_reaction = -153.6 := by sorry

end NUMINAMATH_CALUDE_enthalpy_change_reaction_l689_68975


namespace NUMINAMATH_CALUDE_square_edge_sum_l689_68954

theorem square_edge_sum (u v w x : ℕ+) : 
  u * x + u * v + v * w + w * x = 15 → u + v + w + x = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_square_edge_sum_l689_68954


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l689_68997

theorem absolute_value_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l689_68997


namespace NUMINAMATH_CALUDE_target_hit_probability_l689_68913

theorem target_hit_probability (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) 
  (h_B : p_B = 1/3) 
  (h_C : p_C = 1/4) : 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l689_68913


namespace NUMINAMATH_CALUDE_smallest_acute_angle_in_right_triangle_l689_68971

theorem smallest_acute_angle_in_right_triangle (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 90 → (a / b) = (3 / 2) → min a b = 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_acute_angle_in_right_triangle_l689_68971


namespace NUMINAMATH_CALUDE_certain_value_proof_l689_68995

theorem certain_value_proof (x : ℤ) : 
  (∀ n : ℤ, 101 * n^2 ≤ x → n ≤ 10) ∧ 
  (∃ n : ℤ, n = 10 ∧ 101 * n^2 ≤ x) →
  x = 10100 :=
by sorry

end NUMINAMATH_CALUDE_certain_value_proof_l689_68995


namespace NUMINAMATH_CALUDE_investment_rate_proof_l689_68940

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_proof (principal : ℝ) (time : ℝ) (rate : ℝ) :
  principal = 7000 →
  time = 2 →
  simpleInterest principal rate time = simpleInterest principal 0.12 time + 420 →
  rate = 0.15 := by
sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l689_68940


namespace NUMINAMATH_CALUDE_lottery_winning_probability_l689_68905

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def winnerBallsDrawn : ℕ := 6

def winningProbability : ℚ :=
  1 / (megaBallCount * (winnerBallCount.choose winnerBallsDrawn))

theorem lottery_winning_probability :
  winningProbability = 1 / 476721000 := by sorry

end NUMINAMATH_CALUDE_lottery_winning_probability_l689_68905


namespace NUMINAMATH_CALUDE_max_sales_revenue_l689_68918

/-- Sales volume function -/
def f (t : ℕ) : ℝ := -2 * t + 200

/-- Price function -/
def g (t : ℕ) : ℝ :=
  if t ≤ 30 then 0.5 * t + 30 else 40

/-- Daily sales revenue function -/
def S (t : ℕ) : ℝ := f t * g t

/-- The maximum daily sales revenue occurs at t = 20 and is equal to 6400 -/
theorem max_sales_revenue :
  ∃ (t : ℕ), t ∈ Finset.range 50 ∧
  S t = 6400 ∧
  ∀ (t' : ℕ), t' ∈ Finset.range 50 → S t' ≤ S t :=
by sorry

end NUMINAMATH_CALUDE_max_sales_revenue_l689_68918


namespace NUMINAMATH_CALUDE_exist_50_integers_with_equal_sum_l689_68981

/-- Sum of digits function -/
def S (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + S (n / 10)

/-- Theorem statement -/
theorem exist_50_integers_with_equal_sum :
  ∃ (n : Fin 50 → ℕ), (∀ i j, i < j → n i < n j) ∧
    (∀ i j, i < j → n i + S (n i) = n j + S (n j)) :=
sorry

end NUMINAMATH_CALUDE_exist_50_integers_with_equal_sum_l689_68981


namespace NUMINAMATH_CALUDE_investment_final_values_l689_68991

/-- Calculates the final value of an investment after two years --/
def final_value (initial : ℝ) (year1_change : ℝ) (year1_dividend : ℝ) (year2_change : ℝ) : ℝ :=
  (initial * (1 + year1_change) + initial * year1_dividend) * (1 + year2_change)

/-- Proves that the final values of investments D, E, and F are correct --/
theorem investment_final_values :
  let d := final_value 100 0 0.1 0.05
  let e := final_value 100 0.3 0 (-0.1)
  let f := final_value 100 (-0.1) 0 0.2
  d = 115.5 ∧ e = 117 ∧ f = 108 :=
by sorry

#eval final_value 100 0 0.1 0.05
#eval final_value 100 0.3 0 (-0.1)
#eval final_value 100 (-0.1) 0 0.2

end NUMINAMATH_CALUDE_investment_final_values_l689_68991


namespace NUMINAMATH_CALUDE_questionnaire_C_count_l689_68923

/-- Represents the system sampling method described in the problem -/
def SystemSampling (totalPopulation sampleSize firstDrawn : ℕ) : 
  List ℕ := sorry

/-- Counts the number of elements in a list that fall within a given range -/
def CountInRange (list : List ℕ) (lower upper : ℕ) : ℕ := sorry

theorem questionnaire_C_count :
  let totalPopulation : ℕ := 960
  let sampleSize : ℕ := 32
  let firstDrawn : ℕ := 5
  let sample := SystemSampling totalPopulation sampleSize firstDrawn
  CountInRange sample 751 960 = 7 := by sorry

end NUMINAMATH_CALUDE_questionnaire_C_count_l689_68923


namespace NUMINAMATH_CALUDE_dmv_waiting_time_l689_68932

theorem dmv_waiting_time (x : ℝ) : 
  x + (4 * x + 14) = 114 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_dmv_waiting_time_l689_68932


namespace NUMINAMATH_CALUDE_cube_painting_problem_l689_68976

theorem cube_painting_problem (n : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 3 / 4 → 
  n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_painting_problem_l689_68976


namespace NUMINAMATH_CALUDE_sine_product_upper_bound_sine_product_upper_bound_achievable_l689_68930

/-- Given points A, B, and C in a coordinate plane, where A = (-8, 0), B = (8, 0), and C = (t, 6) for some real number t, the product of sines of angles CAB and CBA is at most 3/8. -/
theorem sine_product_upper_bound (t : ℝ) :
  let A : ℝ × ℝ := (-8, 0)
  let B : ℝ × ℝ := (8, 0)
  let C : ℝ × ℝ := (t, 6)
  let angle_CAB := Real.arctan ((C.2 - A.2) / (C.1 - A.1)) - Real.arctan ((B.2 - A.2) / (B.1 - A.1))
  let angle_CBA := Real.arctan ((C.2 - B.2) / (C.1 - B.1)) - Real.arctan ((A.2 - B.2) / (A.1 - B.1))
  Real.sin angle_CAB * Real.sin angle_CBA ≤ 3/8 :=
by sorry

/-- The upper bound 3/8 for the product of sines is achievable. -/
theorem sine_product_upper_bound_achievable :
  ∃ t : ℝ,
  let A : ℝ × ℝ := (-8, 0)
  let B : ℝ × ℝ := (8, 0)
  let C : ℝ × ℝ := (t, 6)
  let angle_CAB := Real.arctan ((C.2 - A.2) / (C.1 - A.1)) - Real.arctan ((B.2 - A.2) / (B.1 - A.1))
  let angle_CBA := Real.arctan ((C.2 - B.2) / (C.1 - B.1)) - Real.arctan ((A.2 - B.2) / (A.1 - B.1))
  Real.sin angle_CAB * Real.sin angle_CBA = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_sine_product_upper_bound_sine_product_upper_bound_achievable_l689_68930


namespace NUMINAMATH_CALUDE_intersection_perpendicular_implies_k_l689_68909

/-- The line l: kx - y - 2 = 0 intersects the circle O: x^2 + y^2 = 4 at points A and B. -/
def intersects (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  k * A.1 - A.2 - 2 = 0 ∧
  k * B.1 - B.2 - 2 = 0 ∧
  A.1^2 + A.2^2 = 4 ∧
  B.1^2 + B.2^2 = 4

/-- The dot product of OA and OB is zero. -/
def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

/-- Main theorem: If the line intersects the circle and the intersection points are perpendicular from the origin, then k = ±1. -/
theorem intersection_perpendicular_implies_k (k : ℝ) (A B : ℝ × ℝ) 
  (h_intersects : intersects k A B) (h_perp : perpendicular A B) : k = 1 ∨ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_implies_k_l689_68909


namespace NUMINAMATH_CALUDE_gaspard_empty_bags_iff_even_sum_l689_68968

/-- Represents the state of the bags -/
structure BagState where
  m : ℕ
  n : ℕ

/-- Defines the allowed operations on the bags -/
inductive Operation
  | RemoveEqual : ℕ → Operation
  | TripleOne : Bool → Operation

/-- Applies an operation to a bag state -/
def applyOperation (state : BagState) (op : Operation) : BagState :=
  match op with
  | Operation.RemoveEqual k => ⟨state.m - k, state.n - k⟩
  | Operation.TripleOne true => ⟨3 * state.m, state.n⟩
  | Operation.TripleOne false => ⟨state.m, 3 * state.n⟩

/-- Defines when a bag state is empty -/
def isEmptyState (state : BagState) : Prop :=
  state.m = 0 ∧ state.n = 0

/-- Defines when a sequence of operations can empty the bags -/
def canEmpty (initialState : BagState) : Prop :=
  ∃ (ops : List Operation), isEmptyState (ops.foldl applyOperation initialState)

/-- The main theorem: Gaspard can empty both bags iff m + n is even -/
theorem gaspard_empty_bags_iff_even_sum (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) :
    canEmpty ⟨m, n⟩ ↔ Even (m + n) := by
  sorry


end NUMINAMATH_CALUDE_gaspard_empty_bags_iff_even_sum_l689_68968


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l689_68927

/-- Calculates the cost of plastering a rectangular tank's walls and bottom. -/
def plasteringCost (length width depth : ℝ) (costPerSquareMeter : ℝ) : ℝ :=
  let bottomArea := length * width
  let wallArea := 2 * (length * depth + width * depth)
  let totalArea := bottomArea + wallArea
  totalArea * costPerSquareMeter

/-- Theorem stating the cost of plastering a specific tank. -/
theorem tank_plastering_cost :
  let length : ℝ := 25
  let width : ℝ := 12
  let depth : ℝ := 6
  let costPerSquareMeter : ℝ := 0.75  -- 75 paise = 0.75 rupees
  plasteringCost length width depth costPerSquareMeter = 558 := by
  sorry

#eval plasteringCost 25 12 6 0.75

end NUMINAMATH_CALUDE_tank_plastering_cost_l689_68927


namespace NUMINAMATH_CALUDE_square_area_five_parts_l689_68941

/-- Given a square divided into five equal areas with side AB of length 3.6 cm,
    the total area of the square is 1156 square centimeters. -/
theorem square_area_five_parts (s : ℝ) (h1 : s > 0) :
  let ab : ℝ := 3.6
  let area : ℝ := s^2
  (∃ (x : ℝ), ab = s * x ∧ x > 0 ∧ x < 1 ∧ 5 * (s * x)^2 = area) →
  area = 1156 := by
sorry

end NUMINAMATH_CALUDE_square_area_five_parts_l689_68941


namespace NUMINAMATH_CALUDE_cookie_problem_l689_68904

theorem cookie_problem (C : ℕ) : C ≥ 187 ∧ (3 : ℚ) / 70 * C = 8 → C = 187 :=
by
  sorry

#check cookie_problem

end NUMINAMATH_CALUDE_cookie_problem_l689_68904


namespace NUMINAMATH_CALUDE_equality_iff_inequality_l689_68957

theorem equality_iff_inequality (x : ℝ) : (x - 2) * (x - 3) ≤ 0 ↔ |x - 2| + |x - 3| = 1 := by
  sorry

end NUMINAMATH_CALUDE_equality_iff_inequality_l689_68957


namespace NUMINAMATH_CALUDE_steven_peach_count_l689_68982

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := 9

/-- The difference in peaches between Steven and Jake -/
def steven_jake_diff : ℕ := 7

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := jake_peaches + steven_jake_diff

theorem steven_peach_count : steven_peaches = 16 := by
  sorry

end NUMINAMATH_CALUDE_steven_peach_count_l689_68982


namespace NUMINAMATH_CALUDE_square_area_l689_68977

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the square
def square (p1 p2 : Point2D) : ℝ := 
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2

-- Theorem statement
theorem square_area (p1 p2 : Point2D) (h : p1 = ⟨1, 2⟩ ∧ p2 = ⟨4, 6⟩) : 
  square p1 p2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l689_68977


namespace NUMINAMATH_CALUDE_factorization_proof_l689_68949

theorem factorization_proof (x : ℝ) : 75 * x^11 + 225 * x^22 = 75 * x^11 * (1 + 3 * x^11) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l689_68949


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l689_68928

/-- Represents a quadrilateral with diagonals intersecting at a point -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem quadrilateral_diagonal_length 
  (ABCD : Quadrilateral) 
  (h1 : distance ABCD.B ABCD.O = 3)
  (h2 : distance ABCD.O ABCD.D = 9)
  (h3 : distance ABCD.A ABCD.O = 5)
  (h4 : distance ABCD.O ABCD.C = 2)
  (h5 : distance ABCD.A ABCD.B = 7) :
  distance ABCD.A ABCD.D = Real.sqrt 151 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l689_68928


namespace NUMINAMATH_CALUDE_only_cylinder_has_quadrilateral_cross_section_l689_68917

-- Define the types of solids
inductive Solid
  | Cone
  | Cylinder
  | Sphere

-- Define a function that determines if a solid can have a quadrilateral cross-section
def canHaveQuadrilateralCrossSection (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => True
  | _ => False

-- Theorem statement
theorem only_cylinder_has_quadrilateral_cross_section :
  ∀ s : Solid, canHaveQuadrilateralCrossSection s ↔ s = Solid.Cylinder :=
by
  sorry


end NUMINAMATH_CALUDE_only_cylinder_has_quadrilateral_cross_section_l689_68917


namespace NUMINAMATH_CALUDE_bowling_ball_weight_is_18_l689_68934

-- Define the weight of one bowling ball
def bowling_ball_weight : ℝ := sorry

-- Define the weight of one kayak
def kayak_weight : ℝ := sorry

-- Theorem to prove the weight of one bowling ball
theorem bowling_ball_weight_is_18 :
  (10 * bowling_ball_weight = 6 * kayak_weight) →
  (3 * kayak_weight = 90) →
  bowling_ball_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_is_18_l689_68934


namespace NUMINAMATH_CALUDE_sam_initial_pennies_l689_68936

/-- The number of pennies Sam spent -/
def pennies_spent : ℕ := 93

/-- The number of pennies Sam has left -/
def pennies_left : ℕ := 5

/-- The initial number of pennies in Sam's bank -/
def initial_pennies : ℕ := pennies_spent + pennies_left

theorem sam_initial_pennies : initial_pennies = 98 := by
  sorry

end NUMINAMATH_CALUDE_sam_initial_pennies_l689_68936


namespace NUMINAMATH_CALUDE_probability_identical_cubes_value_l689_68964

/-- Represents a cube with 8 faces, each face can be painted with one of three colors -/
structure Cube :=
  (faces : Fin 8 → Fin 3)

/-- The total number of ways to paint two cubes -/
def total_paintings : ℕ := 3^8 * 3^8

/-- The number of ways to paint two cubes so they look identical after rotation -/
def identical_paintings : ℕ := 831

/-- The probability that two cubes look identical after painting and possible rotations -/
def probability_identical_cubes : ℚ :=
  identical_paintings / total_paintings

theorem probability_identical_cubes_value :
  probability_identical_cubes = 831 / 43046721 :=
sorry

end NUMINAMATH_CALUDE_probability_identical_cubes_value_l689_68964


namespace NUMINAMATH_CALUDE_card_area_theorem_l689_68972

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem card_area_theorem (original : Rectangle) 
  (h1 : original.length = 3 ∧ original.width = 7)
  (h2 : ∃ (shortened : Rectangle), 
    (shortened.length = original.length ∧ shortened.width = original.width - 2) ∨
    (shortened.length = original.length - 2 ∧ shortened.width = original.width) ∧
    area shortened = 15) :
  ∃ (other_shortened : Rectangle),
    (other_shortened.length = original.length - 2 ∧ other_shortened.width = original.width) ∨
    (other_shortened.length = original.length ∧ other_shortened.width = original.width - 2) ∧
    area other_shortened ≠ area shortened ∧
    area other_shortened = 7 := by
  sorry

end NUMINAMATH_CALUDE_card_area_theorem_l689_68972


namespace NUMINAMATH_CALUDE_time_difference_per_mile_l689_68938

-- Define the given conditions
def young_girl_distance : ℝ := 18  -- miles
def young_girl_time : ℝ := 135     -- minutes (2 hours and 15 minutes)
def current_distance : ℝ := 12     -- miles
def current_time : ℝ := 300        -- minutes (5 hours)

-- Define the theorem
theorem time_difference_per_mile : 
  (current_time / current_distance) - (young_girl_time / young_girl_distance) = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_per_mile_l689_68938


namespace NUMINAMATH_CALUDE_total_collection_theorem_l689_68984

def group_size : ℕ := 77

def member_contribution (n : ℕ) : ℕ := n

def total_collection_paise (n : ℕ) : ℕ := n * member_contribution n

def paise_to_rupees (p : ℕ) : ℚ := p / 100

theorem total_collection_theorem :
  paise_to_rupees (total_collection_paise group_size) = 59.29 := by
  sorry

end NUMINAMATH_CALUDE_total_collection_theorem_l689_68984


namespace NUMINAMATH_CALUDE_black_balls_probability_l689_68921

theorem black_balls_probability 
  (m₁ m₂ k₁ k₂ : ℕ) 
  (h_total : m₁ + m₂ = 25)
  (h_white_prob : (k₁ : ℝ) / m₁ * (k₂ : ℝ) / m₂ = 0.54)
  : ((m₁ - k₁ : ℝ) / m₁) * ((m₂ - k₂ : ℝ) / m₂) = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_black_balls_probability_l689_68921


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l689_68989

theorem isosceles_triangle_vertex_angle (α β : ℝ) : 
  α = 50 → -- base angle is 50°
  β = 180 - 2*α → -- vertex angle formula
  β = 80 -- vertex angle is 80°
:= by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l689_68989


namespace NUMINAMATH_CALUDE_numbered_cube_consecutive_pairs_l689_68969

/-- Represents a cube with numbers on its faces -/
structure NumberedCube where
  numbers : Fin 6 → ℕ
  distinct : ∀ i j, i ≠ j → numbers i ≠ numbers j

/-- Checks if two faces are adjacent on a cube -/
def adjacent (f1 f2 : Fin 6) : Prop := sorry

/-- Checks if two numbers are consecutive -/
def consecutive (n1 n2 : ℕ) : Prop := n2 = n1 + 1 ∨ n1 = n2 + 1

/-- Theorem: A cube numbered with consecutive integers from 1 to 6 
    has at least two pairs of adjacent faces with consecutive numbers -/
theorem numbered_cube_consecutive_pairs (c : NumberedCube) 
  (h_range : ∀ i, c.numbers i ∈ Finset.range 6) : 
  ∃ (f1 f2 f3 f4 : Fin 6), f1 ≠ f2 ∧ f3 ≠ f4 ∧ (f1, f2) ≠ (f3, f4) ∧ 
    adjacent f1 f2 ∧ adjacent f3 f4 ∧ 
    consecutive (c.numbers f1) (c.numbers f2) ∧ 
    consecutive (c.numbers f3) (c.numbers f4) := by
  sorry

end NUMINAMATH_CALUDE_numbered_cube_consecutive_pairs_l689_68969


namespace NUMINAMATH_CALUDE_peace_treaty_day_l689_68983

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem peace_treaty_day (startDay : DayOfWeek) (daysPassed : Nat) :
  startDay = DayOfWeek.Monday ∧ daysPassed = 893 →
  advanceDay startDay daysPassed = DayOfWeek.Saturday :=
by
  sorry -- Proof omitted as per instructions


end NUMINAMATH_CALUDE_peace_treaty_day_l689_68983


namespace NUMINAMATH_CALUDE_bob_improvement_percentage_l689_68963

/-- The percentage improvement needed to match a target time -/
def percentage_improvement (current_time target_time : ℕ) : ℚ :=
  (current_time - target_time : ℚ) / current_time * 100

/-- Bob's current mile time in seconds -/
def bob_time : ℕ := 640

/-- Bob's sister's mile time in seconds -/
def sister_time : ℕ := 320

/-- Theorem: Bob needs to improve his time by 50% to match his sister's time -/
theorem bob_improvement_percentage :
  percentage_improvement bob_time sister_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_bob_improvement_percentage_l689_68963


namespace NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l689_68916

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := -x^2 + 4*x + 5 < 0
def inequality2 (x : ℝ) : Prop := 2*x^2 - 5*x + 2 ≤ 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | x < -1 ∨ x > 5}
def solution_set2 : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}

-- Theorem statements
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1 := by sorry

theorem inequality2_solution : 
  ∀ x : ℝ, inequality2 x ↔ x ∈ solution_set2 := by sorry

end NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l689_68916


namespace NUMINAMATH_CALUDE_parking_theorem_l689_68943

/-- Represents the number of empty parking spaces -/
def total_spaces : ℕ := 10

/-- Represents the number of cars to be parked -/
def num_cars : ℕ := 3

/-- Represents the number of empty spaces required between cars -/
def spaces_between : ℕ := 1

/-- Calculates the number of parking arrangements given the constraints -/
def parking_arrangements (total : ℕ) (cars : ℕ) (spaces : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of parking arrangements is 40 -/
theorem parking_theorem : 
  parking_arrangements total_spaces num_cars spaces_between = 40 :=
sorry

end NUMINAMATH_CALUDE_parking_theorem_l689_68943


namespace NUMINAMATH_CALUDE_probability_three_two_correct_l689_68939

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def different_numbers : ℕ := 10

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The probability of drawing exactly 3 slips with one number and 2 slips with another number -/
def probability_three_two : ℚ := 75 / 35313

theorem probability_three_two_correct :
  probability_three_two = (different_numbers.choose 2 * slips_per_number.choose 3 * slips_per_number.choose 2) / total_slips.choose drawn_slips :=
by sorry

end NUMINAMATH_CALUDE_probability_three_two_correct_l689_68939


namespace NUMINAMATH_CALUDE_S_union_T_eq_S_l689_68908

def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 > 0}
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

theorem S_union_T_eq_S : S ∪ T = S := by sorry

end NUMINAMATH_CALUDE_S_union_T_eq_S_l689_68908


namespace NUMINAMATH_CALUDE_scaled_determinant_l689_68985

theorem scaled_determinant (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 12 →
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 108 := by
  sorry

end NUMINAMATH_CALUDE_scaled_determinant_l689_68985


namespace NUMINAMATH_CALUDE_price_difference_is_24_l689_68947

/-- The original price of the smartphone --/
def original_price : ℚ := 800

/-- The single discount rate offered by the first store --/
def single_discount_rate : ℚ := 25 / 100

/-- The first discount rate offered by the second store --/
def first_discount_rate : ℚ := 20 / 100

/-- The second discount rate offered by the second store --/
def second_discount_rate : ℚ := 10 / 100

/-- The price after applying a single discount --/
def price_after_single_discount : ℚ := original_price * (1 - single_discount_rate)

/-- The price after applying two successive discounts --/
def price_after_successive_discounts : ℚ := 
  original_price * (1 - first_discount_rate) * (1 - second_discount_rate)

/-- Theorem stating that the difference between the two final prices is $24 --/
theorem price_difference_is_24 : 
  price_after_single_discount - price_after_successive_discounts = 24 := by
  sorry


end NUMINAMATH_CALUDE_price_difference_is_24_l689_68947


namespace NUMINAMATH_CALUDE_polynomial_remainder_l689_68999

theorem polynomial_remainder (x : ℝ) : 
  (x^4 - x + 1) % (x + 3) = 85 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l689_68999


namespace NUMINAMATH_CALUDE_work_rate_problem_l689_68955

/-- Proves that given the work rates of A and B, and the combined work rate of A, B, and C,
    we can determine how long it takes C to do the work alone. -/
theorem work_rate_problem (a b c : ℝ) (h1 : a = 6) (h2 : b = 5) 
  (h3 : 1 / a + 1 / b + 1 / c = 1 / (10 / 9)) : c = 15 / 8 := by
  sorry

#eval (15 : ℚ) / 8  -- To show that 15/8 = 1.875

end NUMINAMATH_CALUDE_work_rate_problem_l689_68955
