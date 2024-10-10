import Mathlib

namespace angle_range_in_scalene_triangle_l2907_290727

-- Define a scalene triangle
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the theorem
theorem angle_range_in_scalene_triangle (t : ScaleneTriangle) 
  (h_longest : t.a ≥ t.b ∧ t.a ≥ t.c) 
  (h_inequality : t.a^2 < t.b^2 + t.c^2) :
  let A := Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))
  60 * π / 180 < A ∧ A < 90 * π / 180 := by
  sorry

end angle_range_in_scalene_triangle_l2907_290727


namespace square_root_five_minus_one_squared_plus_two_times_minus_four_equals_zero_l2907_290768

theorem square_root_five_minus_one_squared_plus_two_times_minus_four_equals_zero :
  ∀ x : ℝ, x = Real.sqrt 5 - 1 → x^2 + 2*x - 4 = 0 := by
sorry

end square_root_five_minus_one_squared_plus_two_times_minus_four_equals_zero_l2907_290768


namespace sarah_amount_l2907_290750

-- Define the total amount Bridge and Sarah have
def total : ℕ := 300

-- Define the difference between Bridget's and Sarah's amounts
def difference : ℕ := 50

-- Theorem to prove
theorem sarah_amount : ∃ (s : ℕ), s + (s + difference) = total ∧ s = 125 := by
  sorry

end sarah_amount_l2907_290750


namespace unique_prime_sum_and_diff_l2907_290764

theorem unique_prime_sum_and_diff : 
  ∃! p : ℕ, Prime p ∧ 
  (∃ a b : ℕ, Prime a ∧ Prime b ∧ p = a + b) ∧ 
  (∃ c d : ℕ, Prime c ∧ Prime d ∧ p = c - d) ∧ 
  p = 5 := by sorry

end unique_prime_sum_and_diff_l2907_290764


namespace arthur_muffins_l2907_290707

theorem arthur_muffins (initial_muffins : ℕ) (multiplier : ℚ) : 
  initial_muffins = 80 →
  multiplier = 5/2 →
  (multiplier * initial_muffins : ℚ) - initial_muffins = 120 :=
by sorry

end arthur_muffins_l2907_290707


namespace remainder_theorem_l2907_290785

theorem remainder_theorem (n m p q r : ℤ)
  (hn : n % 18 = 10)
  (hm : m % 27 = 16)
  (hp : p % 6 = 4)
  (hq : q % 12 = 8)
  (hr : r % 3 = 2) :
  ((3*n + 2*m) - (p + q) / r) % 9 = 2 := by
  sorry

end remainder_theorem_l2907_290785


namespace circle_line_regions_l2907_290798

/-- Represents a configuration of concentric circles and intersecting lines. -/
structure CircleLineConfiguration where
  n : ℕ  -- number of concentric circles
  k : ℕ  -- number of lines through point A
  m : ℕ  -- number of lines through point B

/-- Calculates the maximum number of regions formed by the configuration. -/
def max_regions (config : CircleLineConfiguration) : ℕ :=
  (config.k + 1) * (config.m + 1) * config.n

/-- Calculates the minimum number of regions formed by the configuration. -/
def min_regions (config : CircleLineConfiguration) : ℕ :=
  (config.k + config.m + 1) + config.n - 1

/-- Theorem stating the maximum and minimum number of regions formed. -/
theorem circle_line_regions (config : CircleLineConfiguration) :
  (max_regions config = (config.k + 1) * (config.m + 1) * config.n) ∧
  (min_regions config = (config.k + config.m + 1) + config.n - 1) :=
sorry

end circle_line_regions_l2907_290798


namespace sphere_radius_is_correct_l2907_290758

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  r_bottom : ℝ
  r_top : ℝ
  sphere_radius : ℝ
  is_tangent : Bool

/-- The specific truncated cone with tangent sphere from the problem -/
def problem_cone : TruncatedConeWithSphere :=
  { r_bottom := 20
  , r_top := 5
  , sphere_radius := 10
  , is_tangent := true }

/-- Theorem stating that the sphere radius is correct -/
theorem sphere_radius_is_correct (c : TruncatedConeWithSphere) :
  c.r_bottom = 20 ∧ c.r_top = 5 ∧ c.is_tangent = true → c.sphere_radius = 10 := by
  sorry

end sphere_radius_is_correct_l2907_290758


namespace special_quadrilateral_area_l2907_290778

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the properties of the quadrilateral
def has_inscribed_circle (Q : Quadrilateral) : Prop := sorry
def has_circumscribed_circle (Q : Quadrilateral) : Prop := sorry
def perpendicular_diagonals (Q : Quadrilateral) : Prop := sorry
def circumradius (Q : Quadrilateral) : ℝ := sorry
def side_length_relation (Q : Quadrilateral) : Prop := sorry

-- Define the area of the quadrilateral
def area (Q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem special_quadrilateral_area 
  (Q : Quadrilateral) 
  (h1 : has_inscribed_circle Q)
  (h2 : has_circumscribed_circle Q)
  (h3 : perpendicular_diagonals Q)
  (h4 : circumradius Q = R)
  (h5 : side_length_relation Q) :
  area Q = (8 * R^2) / 5 := by sorry

end special_quadrilateral_area_l2907_290778


namespace find_number_l2907_290754

theorem find_number (x : ℝ) : (0.05 * x = 0.2 * 650 + 190) → x = 6400 := by
  sorry

end find_number_l2907_290754


namespace ab_minus_bc_plus_ac_equals_seven_l2907_290701

theorem ab_minus_bc_plus_ac_equals_seven 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 14) 
  (h2 : a = b + c) : 
  a*b - b*c + a*c = 7 := by
sorry

end ab_minus_bc_plus_ac_equals_seven_l2907_290701


namespace better_deal_gives_three_contacts_per_dollar_l2907_290715

/-- Represents a box of contacts with a given number of contacts and price --/
structure ContactBox where
  contacts : ℕ
  price : ℚ

/-- Calculates the number of contacts per dollar for a given box --/
def contactsPerDollar (box : ContactBox) : ℚ :=
  box.contacts / box.price

theorem better_deal_gives_three_contacts_per_dollar
  (box1 box2 : ContactBox)
  (h1 : box1 = ⟨50, 25⟩)
  (h2 : box2 = ⟨99, 33⟩)
  (h3 : contactsPerDollar box2 > contactsPerDollar box1) :
  contactsPerDollar box2 = 3 := by
  sorry

#check better_deal_gives_three_contacts_per_dollar

end better_deal_gives_three_contacts_per_dollar_l2907_290715


namespace stone_skipping_l2907_290700

/-- Represents the number of skips for each throw --/
structure Throws where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- Defines the conditions of the stone-skipping problem --/
def validThrows (t : Throws) : Prop :=
  t.second = t.first + 2 ∧
  t.third = 2 * t.second ∧
  t.fourth = t.third - 3 ∧
  t.fifth = 8 ∧
  t.first + t.second + t.third + t.fourth + t.fifth = 33

/-- The theorem to be proved --/
theorem stone_skipping (t : Throws) (h : validThrows t) : 
  t.fifth - t.fourth = 1 := by
  sorry

end stone_skipping_l2907_290700


namespace andrena_debelyn_difference_l2907_290723

/-- The number of dolls each person has after the gift exchange --/
structure DollCounts where
  debelyn : ℕ
  christel : ℕ
  andrena : ℕ

/-- Calculate the final doll counts based on initial counts and gifts --/
def finalCounts (debelynInitial christelInitial : ℕ) : DollCounts :=
  let debelynFinal := debelynInitial - 2
  let christelFinal := christelInitial - 5
  let andrenaFinal := christelFinal + 2
  { debelyn := debelynFinal
  , christel := christelFinal
  , andrena := andrenaFinal }

/-- Theorem stating the difference in doll counts between Andrena and Debelyn --/
theorem andrena_debelyn_difference : 
  let counts := finalCounts 20 24
  counts.andrena - counts.debelyn = 3 := by sorry

end andrena_debelyn_difference_l2907_290723


namespace x_gt_2_sufficient_not_necessary_for_x_gt_1_l2907_290767

theorem x_gt_2_sufficient_not_necessary_for_x_gt_1 :
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) :=
by sorry

end x_gt_2_sufficient_not_necessary_for_x_gt_1_l2907_290767


namespace complex_magnitude_problem_l2907_290721

theorem complex_magnitude_problem (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) :
  Complex.abs (z + Complex.I) = 2 := by
sorry

end complex_magnitude_problem_l2907_290721


namespace soda_duration_problem_l2907_290746

/-- Given the number of soda and water bottles, and the daily consumption ratio,
    calculate the number of days the soda bottles will last. -/
def sodaDuration (sodaCount waterCount : ℕ) (sodaRatio waterRatio : ℕ) : ℕ :=
  min (sodaCount / sodaRatio) (waterCount / waterRatio)

/-- Theorem stating that with 360 soda bottles and 162 water bottles,
    consumed in a 3:2 ratio, the soda bottles will last for 81 days. -/
theorem soda_duration_problem :
  sodaDuration 360 162 3 2 = 81 := by
  sorry

end soda_duration_problem_l2907_290746


namespace exponential_inequality_l2907_290740

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2^a + 2*a = 2^b + 3*b) : a > b := by
  sorry

end exponential_inequality_l2907_290740


namespace function_inequality_implies_parameter_bound_l2907_290720

theorem function_inequality_implies_parameter_bound 
  (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 4, f x ∈ Set.Icc 1 2) →
  (∀ x ∈ Set.Icc 0 4, f x ^ 2 - a * f x + 2 < 0) →
  a > 3 := by
sorry

end function_inequality_implies_parameter_bound_l2907_290720


namespace malik_yards_per_game_l2907_290710

-- Define the number of games
def num_games : ℕ := 4

-- Define Josiah's yards per game
def josiah_yards_per_game : ℕ := 22

-- Define Darnell's average yards per game
def darnell_avg_yards : ℕ := 11

-- Define the total yards run by all three athletes
def total_yards : ℕ := 204

-- Theorem to prove
theorem malik_yards_per_game :
  ∃ (malik_yards : ℕ),
    malik_yards * num_games + 
    josiah_yards_per_game * num_games + 
    darnell_avg_yards * num_games = 
    total_yards ∧ 
    malik_yards = 18 := by
  sorry

end malik_yards_per_game_l2907_290710


namespace triangle_probability_l2907_290716

theorem triangle_probability (total_figures : ℕ) (triangle_count : ℕ) 
  (h1 : total_figures = 8) (h2 : triangle_count = 3) :
  (triangle_count : ℚ) / total_figures = 3 / 8 := by
  sorry

end triangle_probability_l2907_290716


namespace largest_prime_divisor_l2907_290799

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The number given in the problem in base 5 --/
def problemNumber : List Nat := [3, 3, 0, 4, 2, 0, 3, 1, 2]

/-- The decimal representation of the problem number --/
def decimalNumber : Nat := base5ToDecimal problemNumber

/-- Checks if a number is prime --/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- Theorem stating the largest prime divisor of the problem number --/
theorem largest_prime_divisor :
  ∃ (p : Nat), p = 11019 ∧ 
    isPrime p ∧ 
    (decimalNumber % p = 0) ∧
    (∀ q : Nat, isPrime q → decimalNumber % q = 0 → q ≤ p) :=
by sorry


end largest_prime_divisor_l2907_290799


namespace nickels_count_l2907_290749

/-- Proves that given 70 coins consisting of nickels and dimes with a total value of $5.55, the number of nickels is 29. -/
theorem nickels_count (total_coins : ℕ) (total_value : ℚ) (nickels : ℕ) (dimes : ℕ) :
  total_coins = 70 →
  total_value = 555/100 →
  total_coins = nickels + dimes →
  total_value = (5/100 : ℚ) * nickels + (10/100 : ℚ) * dimes →
  nickels = 29 := by
sorry

end nickels_count_l2907_290749


namespace equal_diagonals_implies_quad_or_pent_l2907_290737

/-- A convex n-gon with n ≥ 4 -/
structure ConvexNGon where
  n : ℕ
  convex : n ≥ 4

/-- The property that all diagonals of a polygon are equal -/
def all_diagonals_equal (F : ConvexNGon) : Prop := sorry

/-- The property that a polygon is a quadrilateral -/
def is_quadrilateral (F : ConvexNGon) : Prop := F.n = 4

/-- The property that a polygon is a pentagon -/
def is_pentagon (F : ConvexNGon) : Prop := F.n = 5

/-- Theorem: If all diagonals of a convex n-gon (n ≥ 4) are equal, 
    then it is either a quadrilateral or a pentagon -/
theorem equal_diagonals_implies_quad_or_pent (F : ConvexNGon) :
  all_diagonals_equal F → is_quadrilateral F ∨ is_pentagon F := by sorry

end equal_diagonals_implies_quad_or_pent_l2907_290737


namespace shaded_area_circles_l2907_290755

/-- Given a larger circle of radius 8 and two smaller circles touching the larger circle
    and each other at the center of the larger circle, the area of the shaded region
    (the area of the larger circle minus the areas of the two smaller circles) is 32π. -/
theorem shaded_area_circles (r : ℝ) (h : r = 8) : 
  r^2 * π - 2 * (r/2)^2 * π = 32 * π := by sorry

end shaded_area_circles_l2907_290755


namespace polygon_similarity_nesting_l2907_290731

-- Define polygons
variable (Polygon : Type)

-- Define similarity relation between polygons
variable (similar : Polygon → Polygon → Prop)

-- Define nesting relation between polygons
variable (nesting : Polygon → Polygon → Prop)

-- Main theorem
theorem polygon_similarity_nesting 
  (p q : Polygon) : 
  (¬ similar p q) ↔ 
  (∃ r : Polygon, similar r q ∧ ¬ nesting r p) :=
sorry

end polygon_similarity_nesting_l2907_290731


namespace quadratic_equation_solution_l2907_290756

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : d^2 + c*d + d = 0) : c = 1 ∧ d = -2 := by
  sorry

end quadratic_equation_solution_l2907_290756


namespace original_speed_B_l2907_290706

/-- Two people traveling towards each other -/
structure TravelScenario where
  speed_A : ℝ
  speed_B : ℝ

/-- The condition that the meeting point remains the same after speed changes -/
def meeting_point_unchanged (s : TravelScenario) : Prop :=
  s.speed_A / s.speed_B = (5/4 * s.speed_A) / (s.speed_B + 10)

/-- The theorem stating that if the meeting point is unchanged, B's original speed is 40 km/h -/
theorem original_speed_B (s : TravelScenario) :
  meeting_point_unchanged s → s.speed_B = 40 := by
  sorry

end original_speed_B_l2907_290706


namespace museum_visitors_theorem_l2907_290708

/-- Represents the inverse proportional relationship between visitors and ticket price -/
def inverse_proportion (v t k : ℝ) : Prop := v * t = k

/-- Given conditions of the problem -/
def museum_conditions (v₁ v₂ t₁ t₂ k : ℝ) : Prop :=
  t₁ = 20 ∧ v₁ = 150 ∧ t₂ = 30 ∧
  inverse_proportion v₁ t₁ k ∧
  inverse_proportion v₂ t₂ k

/-- Theorem statement -/
theorem museum_visitors_theorem (v₁ v₂ t₁ t₂ k : ℝ) :
  museum_conditions v₁ v₂ t₁ t₂ k → v₂ = 100 := by
  sorry

end museum_visitors_theorem_l2907_290708


namespace sqrt_calculation_l2907_290794

theorem sqrt_calculation :
  (Real.sqrt 80 - Real.sqrt 20 + Real.sqrt 5 = 3 * Real.sqrt 5) ∧
  (2 * Real.sqrt 6 * 3 * Real.sqrt (1/2) / Real.sqrt 3 = 6) := by
  sorry

end sqrt_calculation_l2907_290794


namespace general_equation_proof_l2907_290726

theorem general_equation_proof (n : ℝ) (h1 : n ≠ 4) (h2 : n ≠ 8) :
  n / (n - 4) + (8 - n) / ((8 - n) - 4) = 2 := by
  sorry

end general_equation_proof_l2907_290726


namespace number_of_digits_c_l2907_290709

theorem number_of_digits_c (a b c : ℕ) : 
  a < b → b < c → 
  (b + a) % (b - a) = 0 → 
  (c + b) % (c - b) = 0 → 
  a ≥ 10^2010 → a < 10^2011 →
  b ≥ 10^2011 → b < 10^2012 →
  c ≥ 10^4 ∧ c < 10^5 := by sorry

end number_of_digits_c_l2907_290709


namespace absolute_value_difference_l2907_290713

theorem absolute_value_difference (a b : ℝ) : 
  (|a| = 2) → (|b| = 5) → (a < b) → ((a - b = -3) ∨ (a - b = -7)) := by
  sorry

end absolute_value_difference_l2907_290713


namespace jennifer_grooming_time_l2907_290722

/-- Calculates the total grooming time in hours for a given number of dogs, 
    grooming time per dog, and number of days. -/
def totalGroomingTime (numDogs : ℕ) (groomTimePerDog : ℕ) (numDays : ℕ) : ℚ :=
  (numDogs * groomTimePerDog * numDays : ℚ) / 60

/-- Proves that Jennifer spends 20 hours grooming her dogs in 30 days. -/
theorem jennifer_grooming_time :
  totalGroomingTime 2 20 30 = 20 := by
  sorry

end jennifer_grooming_time_l2907_290722


namespace no_solution_implies_a_equals_one_l2907_290771

/-- If the system of equations ax + y = 1 and x + y = 2 has no solution, then a = 1 -/
theorem no_solution_implies_a_equals_one (a : ℝ) : 
  (∀ x y : ℝ, ¬(ax + y = 1 ∧ x + y = 2)) → a = 1 := by
  sorry

end no_solution_implies_a_equals_one_l2907_290771


namespace library_checkout_false_implication_l2907_290725

-- Define the universe of books in the library
variable (Book : Type)

-- Define a predicate for books available for checkout
variable (available_for_checkout : Book → Prop)

-- The main theorem
theorem library_checkout_false_implication 
  (h : ¬ ∀ (b : Book), available_for_checkout b) :
  (∃ (b : Book), ¬ available_for_checkout b) ∧ 
  (¬ ∀ (b : Book), available_for_checkout b) := by
  sorry

end library_checkout_false_implication_l2907_290725


namespace rain_probability_tel_aviv_l2907_290763

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : 
  binomial_probability 6 4 0.5 = 0.234375 := by sorry

end rain_probability_tel_aviv_l2907_290763


namespace sum_in_M_alpha_sum_l2907_290759

/-- The set of functions f(x) that satisfy the condition:
    For all x₁, x₂ ∈ ℝ and x₂ > x₁, -α(x₂ - x₁) < f(x₂) - f(x₁) < α(x₂ - x₁) -/
def M_alpha (α : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₂ > x₁ → -α * (x₂ - x₁) < f x₂ - f x₁ ∧ f x₂ - f x₁ < α * (x₂ - x₁)

/-- Theorem: If f ∈ Mα₁ and g ∈ Mα₂, then f + g ∈ Mα₁+α₂ -/
theorem sum_in_M_alpha_sum (α₁ α₂ : ℝ) (f g : ℝ → ℝ) 
  (hα₁ : α₁ > 0) (hα₂ : α₂ > 0)
  (hf : M_alpha α₁ f) (hg : M_alpha α₂ g) : 
  M_alpha (α₁ + α₂) (fun x ↦ f x + g x) :=
by sorry

end sum_in_M_alpha_sum_l2907_290759


namespace stewart_farm_horse_food_consumption_l2907_290760

/-- Calculates the total daily horse food consumption on the Stewart farm -/
theorem stewart_farm_horse_food_consumption
  (sheep_to_horse_ratio : ℚ)
  (sheep_count : ℕ)
  (food_per_horse : ℕ) :
  sheep_to_horse_ratio = 5 / 7 →
  sheep_count = 40 →
  food_per_horse = 230 →
  (sheep_count * (7 / 5) : ℚ).num * food_per_horse = 12880 := by
  sorry

#eval (40 * (7 / 5) : ℚ).num * 230

end stewart_farm_horse_food_consumption_l2907_290760


namespace cycle_price_proof_l2907_290724

/-- Proves that given a cycle sold at a 10% loss for Rs. 1620, its original price was Rs. 1800. -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1620)
  (h2 : loss_percentage = 10) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1800 :=
by
  sorry

end cycle_price_proof_l2907_290724


namespace circle_ratio_l2907_290718

theorem circle_ratio (α : Real) (r R x : Real) : 
  r > 0 → R > 0 → x > 0 → r < R →
  (R - r) = (R + r) * Real.sin α →
  x = (r * R) / ((Real.sqrt r + Real.sqrt R)^2) →
  (r / x) = 2 * (1 + Real.cos α) / (1 + Real.sin α) := by
sorry

end circle_ratio_l2907_290718


namespace right_triangle_longer_leg_l2907_290792

theorem right_triangle_longer_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- b is the longer leg
  b ≤ c →            -- Longer leg is shorter than hypotenuse
  b = 60 :=          -- Conclusion: longer leg is 60
by
  sorry

#check right_triangle_longer_leg

end right_triangle_longer_leg_l2907_290792


namespace mailbox_probability_l2907_290773

-- Define the number of mailboxes
def num_mailboxes : ℕ := 2

-- Define the number of letters
def num_letters : ℕ := 3

-- Define the function to calculate the total number of ways to distribute letters
def total_ways : ℕ := 2^num_letters

-- Define the function to calculate the number of favorable ways
def favorable_ways : ℕ := (num_letters.choose (num_letters - 1)) * (num_mailboxes^(num_mailboxes - 1))

-- Define the probability
def probability : ℚ := favorable_ways / total_ways

-- Theorem statement
theorem mailbox_probability : probability = 3/4 := by sorry

end mailbox_probability_l2907_290773


namespace circle_center_l2907_290779

/-- The center of a circle described by the equation x^2 - 6x + y^2 + 2y = 20 is (3, -1) -/
theorem circle_center (x y : ℝ) : 
  (x^2 - 6*x + y^2 + 2*y = 20) → 
  ∃ (h : ℝ) (k : ℝ) (r : ℝ), 
    h = 3 ∧ k = -1 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end circle_center_l2907_290779


namespace complex_number_location_l2907_290751

theorem complex_number_location (z : ℂ) (h : (3 - 2*I)*z = 4 + 3*I) :
  0 < z.re ∧ 0 < z.im :=
sorry

end complex_number_location_l2907_290751


namespace probability_is_half_l2907_290796

/-- Represents a circular field with 6 equally spaced roads -/
structure CircularField :=
  (radius : ℝ)
  (num_roads : Nat)
  (road_angle : ℝ)

/-- Represents a geologist's position after traveling -/
structure GeologistPosition :=
  (road : Nat)
  (distance : ℝ)

/-- Calculates the distance between two geologists -/
def distance_between (field : CircularField) (pos1 pos2 : GeologistPosition) : ℝ :=
  sorry

/-- Determines if two roads are neighboring -/
def are_neighboring (field : CircularField) (road1 road2 : Nat) : Bool :=
  sorry

/-- Calculates the probability of two geologists being more than 8 km apart -/
def probability_more_than_8km (field : CircularField) (speed : ℝ) (time : ℝ) : ℝ :=
  sorry

/-- Main theorem: Probability of geologists being more than 8 km apart is 0.5 -/
theorem probability_is_half (field : CircularField) :
  probability_more_than_8km field 5 1 = 0.5 := by
  sorry

end probability_is_half_l2907_290796


namespace sum_of_first_10_common_elements_l2907_290704

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 20 and common ratio 2 -/
def geometric_progression (k : ℕ) : ℕ := 20 * 2^k

/-- Common elements between the arithmetic and geometric progressions -/
def common_elements (n : ℕ) : Prop :=
  ∃ k : ℕ, arithmetic_progression n = geometric_progression k

/-- The sum of the first 10 common elements -/
def sum_of_common_elements : ℕ := 13981000

theorem sum_of_first_10_common_elements :
  sum_of_common_elements = 13981000 :=
sorry

end sum_of_first_10_common_elements_l2907_290704


namespace arithmetic_square_root_of_one_fourth_l2907_290777

theorem arithmetic_square_root_of_one_fourth (x : ℝ) : x = Real.sqrt (1/4) → x = 1/2 := by
  sorry

end arithmetic_square_root_of_one_fourth_l2907_290777


namespace opposite_expressions_theorem_l2907_290780

theorem opposite_expressions_theorem (a : ℚ) : 
  (3 * a + 1 = -(3 * (a - 1))) → a = 1/3 := by
sorry

end opposite_expressions_theorem_l2907_290780


namespace smallest_n_square_and_cube_l2907_290795

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 4 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 5 * x = y^2) → (∃ (z : ℕ), 4 * x = z^3) → x ≥ n) ∧
  n = 1080 := by
sorry

end smallest_n_square_and_cube_l2907_290795


namespace inequality_solution_system_of_equations_solution_l2907_290776

-- Part 1: Inequality
theorem inequality_solution (x : ℝ) :
  (5 * x - 12 ≤ 2 * (4 * x - 3)) ↔ (x ≥ -2) := by sorry

-- Part 2: System of equations
theorem system_of_equations_solution (x y : ℝ) :
  (x - y = 5 ∧ 2 * x + y = 4) → (x = 3 ∧ y = -2) := by sorry

end inequality_solution_system_of_equations_solution_l2907_290776


namespace plan2_cheaper_l2907_290734

/-- Represents a payment plan with number of installments and months between payments -/
structure PaymentPlan where
  installments : ℕ
  months_between : ℕ

/-- Calculates the total payment amount for a given payment plan -/
def totalPayment (price : ℝ) (rate : ℝ) (plan : PaymentPlan) : ℝ :=
  price * (1 + rate) ^ (plan.installments * plan.months_between)

theorem plan2_cheaper (price : ℝ) (rate : ℝ) (plan1 plan2 : PaymentPlan) :
  price > 0 →
  rate > 0 →
  plan1.installments = 3 →
  plan1.months_between = 4 →
  plan2.installments = 12 →
  plan2.months_between = 1 →
  totalPayment price rate plan2 ≤ totalPayment price rate plan1 := by
  sorry

#check plan2_cheaper

end plan2_cheaper_l2907_290734


namespace overlap_implies_ratio_l2907_290742

/-- Two overlapping rectangles with dimensions p and q -/
def overlap_rectangles (p q : ℝ) : Prop :=
  ∃ (overlap_area total_area : ℝ),
    overlap_area = q^2 ∧
    total_area = 2*p*q - q^2 ∧
    overlap_area = (1/4) * total_area

/-- The ratio of p to q is 5:2 -/
def ratio_is_5_2 (p q : ℝ) : Prop :=
  p / q = 5/2

/-- Theorem: If two rectangles of dimensions p and q overlap such that
    the overlap area is one-quarter of the total area, then p:q = 5:2 -/
theorem overlap_implies_ratio (p q : ℝ) (h : q ≠ 0) :
  overlap_rectangles p q → ratio_is_5_2 p q :=
by
  sorry


end overlap_implies_ratio_l2907_290742


namespace inequality_system_solution_l2907_290797

theorem inequality_system_solution (x : ℝ) :
  (4 * x + 6 > 1 - x) ∧ (3 * (x - 1) ≤ x + 5) → -1 < x ∧ x ≤ 4 := by
  sorry

end inequality_system_solution_l2907_290797


namespace flu_free_inhabitants_l2907_290702

theorem flu_free_inhabitants (total_population : ℕ) (flu_percentage : ℚ) : 
  total_population = 14000000 →
  flu_percentage = 15 / 10000 →
  (total_population : ℚ) - (flu_percentage * total_population) = 13979000 := by
  sorry

end flu_free_inhabitants_l2907_290702


namespace bounds_on_ratio_of_squares_l2907_290793

theorem bounds_on_ratio_of_squares (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m M : ℝ), 
    (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → m ≤ |a + b|^2 / (|a|^2 + |b|^2) ∧ |a + b|^2 / (|a|^2 + |b|^2) ≤ M) ∧
    (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ |c + d|^2 / (|c|^2 + |d|^2) = m) ∧
    (∃ e f : ℝ, e ≠ 0 ∧ f ≠ 0 ∧ |e + f|^2 / (|e|^2 + |f|^2) = M) ∧
    M - m = 2 :=
by sorry

end bounds_on_ratio_of_squares_l2907_290793


namespace product_of_sums_l2907_290732

theorem product_of_sums (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  a * b + a + b = 99 ∧ b * c + b + c = 99 ∧ c * a + c + a = 99 →
  (a + 1) * (b + 1) * (c + 1) = 1000 := by
sorry

end product_of_sums_l2907_290732


namespace inequality_solution_implies_m_less_than_four_l2907_290729

-- Define the system of inequalities
def has_solution (m : ℝ) : Prop :=
  ∃ x : ℝ, (2 * x - 6 + m < 0) ∧ (4 * x - m > 0)

-- State the theorem
theorem inequality_solution_implies_m_less_than_four :
  ∀ m : ℝ, has_solution m → m < 4 := by
  sorry

end inequality_solution_implies_m_less_than_four_l2907_290729


namespace line_plane_perpendicularity_l2907_290781

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel m n → 
  contains β n → 
  perpendicularPlanes α β :=
sorry

end line_plane_perpendicularity_l2907_290781


namespace painter_paintings_l2907_290705

/-- Given a painter who makes a certain number of paintings per day and already has some paintings,
    calculate the total number of paintings after a given number of days. -/
def total_paintings (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  paintings_per_day * days + initial_paintings

/-- Theorem: A painter who makes 2 paintings per day and already has 20 paintings
    will have 80 paintings in total after 30 days. -/
theorem painter_paintings : total_paintings 2 20 30 = 80 := by
  sorry

end painter_paintings_l2907_290705


namespace rectangle_area_l2907_290738

/-- The area of a rectangle with perimeter 90 feet and length three times the width is 380.15625 square feet. -/
theorem rectangle_area (w : ℝ) (l : ℝ) (h1 : 2 * l + 2 * w = 90) (h2 : l = 3 * w) :
  l * w = 380.15625 := by
sorry

end rectangle_area_l2907_290738


namespace vector_problem_l2907_290769

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b (y : ℝ) : ℝ × ℝ := (Real.cos y, Real.sin y)
noncomputable def c (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) (c x)

noncomputable def g (m x : ℝ) : ℝ := f (x + m)

theorem vector_problem (x y : ℝ) 
  (h : ‖a x - b y‖ = 2 * Real.sqrt 5 / 5) : 
  (Real.cos (x - y) = 3 / 5) ∧ 
  (∃ (m : ℝ), m > 0 ∧ m = Real.pi / 4 ∧ 
    ∀ (n : ℝ), n > 0 → (∀ (t : ℝ), g n t = g n (-t)) → m ≤ n) := by
  sorry

end vector_problem_l2907_290769


namespace total_writing_time_l2907_290711

theorem total_writing_time :
  let woody_time : ℝ := 18 -- Woody's writing time in months
  let ivanka_time : ℝ := woody_time + 3 -- Ivanka's writing time
  let alice_time : ℝ := woody_time / 2 -- Alice's writing time
  let tom_time : ℝ := alice_time * 2 -- Tom's writing time
  ivanka_time + woody_time + alice_time + tom_time = 66 := by
  sorry

end total_writing_time_l2907_290711


namespace num_terms_eq_508020_l2907_290745

/-- The number of terms in the simplified expression of (x+y+z+w)^2008 + (x-y-z-w)^2008 -/
def num_terms : ℕ :=
  let n := 2008
  let sum := (n / 2 + 1)^2 - (n / 2) * (n / 2 + 1) / 2
  sum

/-- Theorem stating that the number of terms in the simplified expression
    of (x+y+z+w)^2008 + (x-y-z-w)^2008 is equal to 508020 -/
theorem num_terms_eq_508020 : num_terms = 508020 := by
  sorry

end num_terms_eq_508020_l2907_290745


namespace sequence_with_differences_two_or_five_l2907_290789

theorem sequence_with_differences_two_or_five :
  ∃ (p : Fin 101 → Fin 101), Function.Bijective p ∧
    (∀ i : Fin 100, (p (i + 1) : ℕ) - (p i : ℕ) = 2 ∨ (p (i + 1) : ℕ) - (p i : ℕ) = 5 ∨
                    (p i : ℕ) - (p (i + 1) : ℕ) = 2 ∨ (p i : ℕ) - (p (i + 1) : ℕ) = 5) :=
by sorry


end sequence_with_differences_two_or_five_l2907_290789


namespace initial_friends_correct_l2907_290790

/-- The number of friends initially playing the game -/
def initial_friends : ℕ := 8

/-- The number of additional players who joined -/
def additional_players : ℕ := 2

/-- The number of lives each player has -/
def lives_per_player : ℕ := 6

/-- The total number of lives after new players joined -/
def total_lives : ℕ := 60

/-- Theorem stating that the initial number of friends is correct -/
theorem initial_friends_correct :
  initial_friends * lives_per_player + additional_players * lives_per_player = total_lives :=
by sorry

end initial_friends_correct_l2907_290790


namespace square_exterior_points_diagonal_l2907_290757

-- Define the square ABCD
def square_side_length : ℝ := 15

-- Define the lengths BG, DH, AG, and CH
def BG : ℝ := 7
def DH : ℝ := 7
def AG : ℝ := 17
def CH : ℝ := 17

-- Define the theorem
theorem square_exterior_points_diagonal (A B C D G H : ℝ × ℝ) :
  let AB := square_side_length
  let AD := square_side_length
  (B.1 - G.1)^2 + (B.2 - G.2)^2 = BG^2 →
  (D.1 - H.1)^2 + (D.2 - H.2)^2 = DH^2 →
  (A.1 - G.1)^2 + (A.2 - G.2)^2 = AG^2 →
  (C.1 - H.1)^2 + (C.2 - H.2)^2 = CH^2 →
  (G.1 - H.1)^2 + (G.2 - H.2)^2 = 98 :=
by sorry


end square_exterior_points_diagonal_l2907_290757


namespace three_eighths_decimal_l2907_290736

theorem three_eighths_decimal : (3 : ℚ) / 8 = 0.375 := by
  sorry

end three_eighths_decimal_l2907_290736


namespace expand_and_compare_l2907_290744

theorem expand_and_compare (m n : ℝ) :
  (∀ x : ℝ, (x + 2) * (x + 3) = x^2 + m*x + n) → m = 5 ∧ n = 6 := by
  sorry

end expand_and_compare_l2907_290744


namespace exponent_equation_solution_l2907_290782

theorem exponent_equation_solution (a b : ℝ) (m n : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^m * b * b^n)^3 = a^6 * b^15 → m = 2 ∧ n = 4 := by
  sorry

end exponent_equation_solution_l2907_290782


namespace sally_fries_theorem_l2907_290743

def sally_fries_problem (sally_initial : ℕ) (mark_total : ℕ) (jessica_total : ℕ) : Prop :=
  let mark_share := mark_total / 3
  let jessica_share := jessica_total / 2
  sally_initial + mark_share + jessica_share = 38

theorem sally_fries_theorem :
  sally_fries_problem 14 36 24 :=
by
  sorry

end sally_fries_theorem_l2907_290743


namespace unique_n_with_special_divisors_l2907_290712

def isDivisor (d n : ℕ) : Prop := d ∣ n

def divisors (n : ℕ) : Set ℕ := {d : ℕ | isDivisor d n}

theorem unique_n_with_special_divisors :
  ∃! n : ℕ, n > 0 ∧
  ∃ (d₂ d₃ : ℕ), d₂ ∈ divisors n ∧ d₃ ∈ divisors n ∧
  1 < d₂ ∧ d₂ < d₃ ∧
  n = d₂^2 + d₃^3 ∧
  ∀ d ∈ divisors n, d = 1 ∨ d ≥ d₂ :=
by
  sorry

end unique_n_with_special_divisors_l2907_290712


namespace increasing_linear_function_positive_slope_l2907_290717

/-- A linear function f(x) = mx + b -/
def LinearFunction (m b : ℝ) : ℝ → ℝ := fun x ↦ m * x + b

/-- A function is increasing if for any x₁ < x₂, f(x₁) < f(x₂) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

theorem increasing_linear_function_positive_slope (m b : ℝ) :
  IsIncreasing (LinearFunction m b) → m > 0 := by
  sorry

end increasing_linear_function_positive_slope_l2907_290717


namespace solutions_of_fourth_power_equation_l2907_290787

theorem solutions_of_fourth_power_equation :
  let S : Set ℂ := {x | x^4 - 16 = 0}
  S = {2, -2, Complex.I * 2, -Complex.I * 2} := by
  sorry

end solutions_of_fourth_power_equation_l2907_290787


namespace probability_even_product_l2907_290791

/-- Spinner C with numbers 1 through 6 -/
def spinner_C : Finset ℕ := Finset.range 6

/-- Spinner D with numbers 1 through 4 -/
def spinner_D : Finset ℕ := Finset.range 4

/-- Function to check if a number is even -/
def is_even (n : ℕ) : Bool := n % 2 = 0

/-- Function to check if the product of two numbers is even -/
def product_is_even (x y : ℕ) : Bool := is_even (x * y)

/-- Total number of possible outcomes -/
def total_outcomes : ℕ := (Finset.card spinner_C) * (Finset.card spinner_D)

/-- Number of outcomes where the product is even -/
def even_product_outcomes : ℕ := Finset.card (Finset.filter (λ (pair : ℕ × ℕ) => product_is_even pair.1 pair.2) (spinner_C.product spinner_D))

/-- Theorem stating the probability of getting an even product -/
theorem probability_even_product :
  (even_product_outcomes : ℚ) / total_outcomes = 3 / 4 := by sorry

end probability_even_product_l2907_290791


namespace elliptical_cylinder_stability_l2907_290752

/-- A cylinder with an elliptical cross-section -/
structure EllipticalCylinder where
  a : ℝ
  b : ℝ
  h : a > b

/-- Stability condition for an elliptical cylinder -/
def is_stable (c : EllipticalCylinder) : Prop :=
  c.b / c.a < 1 / Real.sqrt 2

/-- Theorem: An elliptical cylinder is in stable equilibrium iff b/a < 1/√2 -/
theorem elliptical_cylinder_stability (c : EllipticalCylinder) :
  is_stable c ↔ c.b / c.a < 1 / Real.sqrt 2 := by sorry

end elliptical_cylinder_stability_l2907_290752


namespace polygon_properties_l2907_290783

/-- Represents a convex polygon with properties as described in the problem -/
structure ConvexPolygon where
  n : ℕ                             -- number of sides
  interior_angle_sum : ℝ             -- sum of interior angles minus one unknown angle
  triangle_area : ℝ                  -- area of triangle formed by three adjacent vertices
  triangle_side : ℝ                  -- length of one side of the triangle
  triangle_opposite_angle : ℝ        -- angle opposite to the known side in the triangle

/-- The theorem to be proved -/
theorem polygon_properties (p : ConvexPolygon) 
  (h1 : p.interior_angle_sum = 3240)
  (h2 : p.triangle_area = 150)
  (h3 : p.triangle_side = 15)
  (h4 : p.triangle_opposite_angle = 60) :
  p.n = 20 ∧ (180 * (p.n - 2) - p.interior_angle_sum = 0) := by
  sorry


end polygon_properties_l2907_290783


namespace simplify_expression_l2907_290775

theorem simplify_expression (b : ℝ) (h : b ≠ 1) :
  1 - 1 / (2 + b / (1 - b)) = 1 / (2 - b) := by sorry

end simplify_expression_l2907_290775


namespace adam_figurines_l2907_290761

/-- The number of figurines that can be made from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be made from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be made from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of blocks of basswood Adam owns -/
def basswood_blocks : ℕ := 15

/-- The number of blocks of butternut wood Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of blocks of Aspen wood Adam owns -/
def aspen_blocks : ℕ := 20

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 
  basswood_blocks * basswood_figurines + 
  butternut_blocks * butternut_figurines + 
  aspen_blocks * aspen_figurines

theorem adam_figurines : total_figurines = 245 := by
  sorry

end adam_figurines_l2907_290761


namespace quadratic_equation_coefficients_l2907_290788

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
    (∀ x, 3 * x^2 - 1 = 5 * x ↔ a * x^2 + b * x + c = 0) →
    a = 3 ∧ b = -5 ∧ c = -1 := by
  sorry

end quadratic_equation_coefficients_l2907_290788


namespace domain_subset_theorem_l2907_290735

theorem domain_subset_theorem (a : ℝ) : 
  (Set.Ioo a (a + 1) ⊆ Set.Ioo (-1) 1) ↔ a ∈ Set.Icc (-1) 0 := by
sorry

end domain_subset_theorem_l2907_290735


namespace danny_carpooling_l2907_290739

/-- Given Danny's carpooling route, prove the distance to the first friend's house -/
theorem danny_carpooling (x : ℝ) :
  x > 0 ∧ 
  (x / 2 > 0) ∧ 
  (3 * (x + x / 2) = 36) →
  x = 8 :=
by sorry

end danny_carpooling_l2907_290739


namespace max_overtakes_l2907_290753

/-- Represents a team in the relay race -/
structure Team :=
  (members : Nat)
  (segments : Nat)

/-- Represents the relay race setup -/
structure RelayRace :=
  (team1 : Team)
  (team2 : Team)
  (simultaneous_start : Bool)
  (instantaneous_exchange : Bool)

/-- Defines what constitutes an overtake in the race -/
def is_valid_overtake (race : RelayRace) (position : Nat) : Prop :=
  position > 0 ∧ position < race.team1.segments ∧ position < race.team2.segments

/-- The main theorem stating the maximum number of overtakes -/
theorem max_overtakes (race : RelayRace) : 
  race.team1.members = 20 →
  race.team2.members = 20 →
  race.team1.segments = 20 →
  race.team2.segments = 20 →
  race.simultaneous_start = true →
  race.instantaneous_exchange = true →
  ∃ (n : Nat), n = 38 ∧ 
    (∀ (m : Nat), (∃ (valid_overtakes : List Nat), 
      (∀ o ∈ valid_overtakes, is_valid_overtake race o) ∧ 
      valid_overtakes.length = m) → m ≤ n) :=
sorry


end max_overtakes_l2907_290753


namespace place_letters_in_mailboxes_l2907_290703

theorem place_letters_in_mailboxes :
  let n_letters : ℕ := 3
  let n_mailboxes : ℕ := 5
  (n_letters > 0) → (n_mailboxes > 0) →
  (number_of_ways : ℕ := n_mailboxes ^ n_letters) →
  number_of_ways = 125 := by
  sorry

end place_letters_in_mailboxes_l2907_290703


namespace intersection_point_l2907_290770

theorem intersection_point (x y : ℚ) :
  (8 * x - 5 * y = 10) ∧ (9 * x + 4 * y = 20) ↔ x = 140 / 77 ∧ y = 70 / 77 := by
  sorry

end intersection_point_l2907_290770


namespace pure_imaginary_condition_l2907_290741

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I * ((2 : ℂ) + a * Complex.I) * ((1 : ℂ) - Complex.I)).re = 0 → 
  a = -2 := by
  sorry

end pure_imaginary_condition_l2907_290741


namespace min_distance_complex_main_theorem_l2907_290748

-- Define inductive reasoning
def inductiveReasoning : String := "reasoning from specific to general"

-- Define deductive reasoning
def deductiveReasoning : String := "reasoning from general to specific"

-- Theorem for the complex number part
theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

-- Main theorem combining all parts
theorem main_theorem :
  inductiveReasoning = "reasoning from specific to general" ∧
  deductiveReasoning = "reasoning from general to specific" ∧
  ∀ (z : ℂ), Complex.abs (z + 2 - 2*I) = 1 →
    ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

end min_distance_complex_main_theorem_l2907_290748


namespace cubic_fraction_equals_five_l2907_290772

theorem cubic_fraction_equals_five :
  let a : ℚ := 3
  let b : ℚ := 2
  (a^3 + b^3) / (a^2 - 2*a*b + b^2 + a*b) = 5 := by sorry

end cubic_fraction_equals_five_l2907_290772


namespace simplify_and_evaluate_l2907_290719

theorem simplify_and_evaluate (m : ℝ) (h : m = 2 - Real.sqrt 2) :
  (3 / (m + 1) + 1 - m) / ((m + 2) / (m + 1)) = Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l2907_290719


namespace parallel_lines_condition_l2907_290765

/-- Two lines L₁ and L₂ are defined as follows:
    L₁: ax + 3y + 1 = 0
    L₂: 2x + (a+1)y + 1 = 0
    This theorem proves that if L₁ and L₂ are parallel, then a = -3. -/
theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, ax + 3*y + 1 = 0 ↔ 2*x + (a+1)*y + 1 = 0) →
  a = -3 :=
by sorry

end parallel_lines_condition_l2907_290765


namespace parallel_transitivity_l2907_290747

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary properties for a line in 3D space
  -- This is a simplified representation
  mk :: 

-- Define parallelism for lines in 3D space
def parallel (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be parallel
  sorry

-- State the theorem
theorem parallel_transitivity (a b c : Line3D) :
  parallel a c → parallel b c → parallel a b := by
  sorry

end parallel_transitivity_l2907_290747


namespace a_eq_2_sufficient_not_necessary_for_abs_a_eq_2_l2907_290774

theorem a_eq_2_sufficient_not_necessary_for_abs_a_eq_2 :
  (∃ a : ℝ, a = 2 → |a| = 2) ∧ 
  (∃ a : ℝ, |a| = 2 ∧ a ≠ 2) :=
by sorry

end a_eq_2_sufficient_not_necessary_for_abs_a_eq_2_l2907_290774


namespace negation_abc_zero_l2907_290762

theorem negation_abc_zero (a b c : ℝ) : (a = 0 ∨ b = 0 ∨ c = 0) → a * b * c = 0 := by
  sorry

end negation_abc_zero_l2907_290762


namespace juniors_score_l2907_290766

/-- Given a class with juniors and seniors, prove the juniors' score -/
theorem juniors_score (n : ℕ) (junior_score : ℝ) :
  n > 0 →
  (0.2 * n : ℝ) * junior_score + (0.8 * n : ℝ) * 80 = n * 82 →
  junior_score = 90 :=
by
  sorry

end juniors_score_l2907_290766


namespace quadratic_inequality_l2907_290728

/-- A quadratic function with positive leading coefficient and symmetry about x = 2 -/
def symmetric_quadratic (a b c : ℝ) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality 
  (a b c : ℝ) 
  (ha : a > 0)
  (h_sym : ∀ x, symmetric_quadratic a b c (x + 2) = symmetric_quadratic a b c (2 - x)) :
  symmetric_quadratic a b c (Real.sqrt 2 / 2) > symmetric_quadratic a b c Real.pi :=
by
  sorry

end quadratic_inequality_l2907_290728


namespace harry_sister_stamp_ratio_l2907_290730

/-- Proves the ratio of Harry's stamps to his sister's stamps -/
theorem harry_sister_stamp_ratio :
  let total_stamps : ℕ := 240
  let sister_stamps : ℕ := 60
  let harry_stamps : ℕ := total_stamps - sister_stamps
  (harry_stamps : ℚ) / sister_stamps = 3 := by
  sorry

end harry_sister_stamp_ratio_l2907_290730


namespace product_remainder_zero_l2907_290733

theorem product_remainder_zero (n : ℕ) :
  (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 0 :=
by
  sorry

end product_remainder_zero_l2907_290733


namespace pool_capacity_l2907_290784

theorem pool_capacity (fill_time_both : ℝ) (fill_time_first : ℝ) (additional_rate : ℝ) :
  fill_time_both = 48 →
  fill_time_first = 120 →
  additional_rate = 50 →
  ∃ (capacity : ℝ),
    capacity = 12000 ∧
    capacity / fill_time_both = capacity / fill_time_first + (capacity / fill_time_first + additional_rate) :=
by sorry

end pool_capacity_l2907_290784


namespace ab_length_l2907_290714

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
def collinear (A B C D : ℝ × ℝ) : Prop := sorry
def distance (P Q : ℝ × ℝ) : ℝ := sorry
def perimeter (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem ab_length
  (h_collinear : collinear A B C D)
  (h_ab_cd : distance A B = distance C D)
  (h_bc : distance B C = 8)
  (h_be : distance B E = 12)
  (h_ce : distance C E = 12)
  (h_perimeter : perimeter A E D = 3 * perimeter B E C) :
  distance A B = 18 := by sorry

end ab_length_l2907_290714


namespace pet_shop_total_cost_l2907_290786

/-- The cost of purchasing all pets in a pet shop given specific conditions. -/
theorem pet_shop_total_cost :
  let num_puppies : ℕ := 2
  let num_kittens : ℕ := 2
  let num_parakeets : ℕ := 3
  let parakeet_cost : ℕ := 10
  let puppy_cost : ℕ := 3 * parakeet_cost
  let kitten_cost : ℕ := 2 * parakeet_cost
  num_puppies * puppy_cost + num_kittens * kitten_cost + num_parakeets * parakeet_cost = 130 := by
sorry

end pet_shop_total_cost_l2907_290786
