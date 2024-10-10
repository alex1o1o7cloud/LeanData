import Mathlib

namespace smallest_common_multiple_of_8_and_6_l237_23727

theorem smallest_common_multiple_of_8_and_6 :
  ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ (m : ℕ), m > 0 ∧ 8 ∣ m ∧ 6 ∣ m → n ≤ m :=
by
  use 24
  sorry

end smallest_common_multiple_of_8_and_6_l237_23727


namespace squirrel_mushroom_collection_l237_23756

/-- Represents the number of mushrooms in each clearing --/
def MushroomSequence : Type := List Nat

/-- The total number of mushrooms collected by the squirrel --/
def TotalMushrooms : Nat := 60

/-- The number of clearings visited by the squirrel --/
def NumberOfClearings : Nat := 10

/-- Checks if a given sequence is valid according to the problem conditions --/
def IsValidSequence (seq : MushroomSequence) : Prop :=
  seq.length = NumberOfClearings ∧
  seq.sum = TotalMushrooms ∧
  seq.all (· > 0)

/-- The correct sequence of mushrooms collected in each clearing --/
def CorrectSequence : MushroomSequence := [5, 2, 11, 8, 2, 12, 3, 7, 2, 8]

/-- Theorem stating that the CorrectSequence is a valid solution to the problem --/
theorem squirrel_mushroom_collection :
  IsValidSequence CorrectSequence :=
sorry

end squirrel_mushroom_collection_l237_23756


namespace garden_area_bounds_l237_23757

/-- Represents a rectangular garden with given constraints -/
structure Garden where
  wall : ℝ
  fence : ℝ
  minParallelSide : ℝ

/-- The area of the garden as a function of the length perpendicular to the wall -/
def Garden.area (g : Garden) (x : ℝ) : ℝ :=
  x * (g.fence - 2 * x)

/-- Theorem stating the maximum and minimum areas of the garden -/
theorem garden_area_bounds (g : Garden) 
  (h_wall : g.wall = 12)
  (h_fence : g.fence = 40)
  (h_minSide : g.minParallelSide = 6) :
  (∃ x : ℝ, g.area x ≤ 168 ∧ 
   ∀ y : ℝ, g.minParallelSide ≤ g.fence - 2 * y → g.area y ≤ g.area x) ∧
  (∃ x : ℝ, g.area x ≥ 102 ∧ 
   ∀ y : ℝ, g.minParallelSide ≤ g.fence - 2 * y → g.area y ≥ g.area x) :=
sorry

end garden_area_bounds_l237_23757


namespace warehouse_capacity_prove_warehouse_capacity_l237_23764

/-- The total capacity of a grain storage warehouse --/
theorem warehouse_capacity : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_bins twenty_ton_bins twenty_ton_capacity fifteen_ton_capacity =>
    total_bins = 30 ∧
    twenty_ton_bins = 12 ∧
    twenty_ton_capacity = 20 ∧
    fifteen_ton_capacity = 15 →
    (twenty_ton_bins * twenty_ton_capacity) +
    ((total_bins - twenty_ton_bins) * fifteen_ton_capacity) = 510

/-- Proof of the warehouse capacity theorem --/
theorem prove_warehouse_capacity :
  warehouse_capacity 30 12 20 15 := by
  sorry

end warehouse_capacity_prove_warehouse_capacity_l237_23764


namespace value_range_of_f_l237_23735

def f (x : ℝ) := 3 * x - 1

theorem value_range_of_f :
  Set.Icc (-16 : ℝ) 5 = Set.image f (Set.Ico (-5 : ℝ) 2) := by sorry

end value_range_of_f_l237_23735


namespace weighted_am_gm_inequality_l237_23717

theorem weighted_am_gm_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  0.2 * a + 0.3 * b + 0.5 * c ≥ (a * b * c) ^ (1/3) := by
  sorry

end weighted_am_gm_inequality_l237_23717


namespace complex_modulus_range_l237_23784

theorem complex_modulus_range (z k : ℂ) (h1 : Complex.abs z = Complex.abs (1 + k * z)) (h2 : Complex.abs k < 1) :
  1 / (Complex.abs k + 1) ≤ Complex.abs z ∧ Complex.abs z ≤ 1 / (1 - Complex.abs k) :=
by sorry

end complex_modulus_range_l237_23784


namespace apple_sales_proof_l237_23704

/-- The number of kilograms of apples sold in the first hour -/
def first_hour_sales : ℝ := 10

/-- The number of kilograms of apples sold in the second hour -/
def second_hour_sales : ℝ := 2

/-- The average number of kilograms of apples sold per hour over two hours -/
def average_sales : ℝ := 6

theorem apple_sales_proof :
  first_hour_sales = 10 :=
by
  have h1 : average_sales = (first_hour_sales + second_hour_sales) / 2 :=
    sorry
  -- Proof goes here
  sorry

end apple_sales_proof_l237_23704


namespace oplus_composition_l237_23754

/-- Definition of the ⊕ operation -/
def oplus (x y : ℝ) : ℝ := x^2 + y

/-- Theorem stating that h ⊕ (h ⊕ h) = 2h^2 + h -/
theorem oplus_composition (h : ℝ) : oplus h (oplus h h) = 2 * h^2 + h := by
  sorry

end oplus_composition_l237_23754


namespace common_factor_of_polynomial_l237_23775

theorem common_factor_of_polynomial (x : ℝ) :
  ∃ (k : ℝ), 2*x^2 - 8*x = 2*x*k :=
by
  sorry

end common_factor_of_polynomial_l237_23775


namespace ap_special_condition_l237_23734

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  first : ℝ
  diff : ℝ

/-- The nth term of an arithmetic progression -/
def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.first + (n - 1 : ℝ) * ap.diff

theorem ap_special_condition (ap : ArithmeticProgression) :
  nthTerm ap 4 + nthTerm ap 20 = nthTerm ap 8 + nthTerm ap 15 + nthTerm ap 12 →
  ap.first = 10 * ap.diff := by
  sorry

end ap_special_condition_l237_23734


namespace roses_cut_equals_difference_l237_23744

/-- The number of roses Jessica cut from her garden -/
def roses_cut : ℕ := sorry

/-- The initial number of roses in the vase -/
def initial_roses : ℕ := 7

/-- The final number of roses in the vase -/
def final_roses : ℕ := 23

/-- Theorem stating that the number of roses Jessica cut is equal to the difference between the final and initial number of roses in the vase -/
theorem roses_cut_equals_difference : roses_cut = final_roses - initial_roses := by sorry

end roses_cut_equals_difference_l237_23744


namespace fraction_transformation_l237_23774

theorem fraction_transformation (x : ℤ) : 
  x = 437 → (537 - x : ℚ) / (463 + x) = 1 / 9 := by
  sorry

end fraction_transformation_l237_23774


namespace multiplication_puzzle_l237_23761

theorem multiplication_puzzle : ∃ (a b : Nat), 
  a < 10000 ∧ 
  b < 1000 ∧ 
  a / 1000 = 3 ∧ 
  a % 100 = 20 ∧
  b / 100 = 3 ∧
  (a * (b % 10)) % 10000 = 9060 ∧
  ((a * (b / 10)) / 10000) * 10000 + ((a * (b / 10)) % 10000) = 62510 ∧
  a * b = 1157940830 := by
  sorry

end multiplication_puzzle_l237_23761


namespace fraction_inequality_l237_23747

theorem fraction_inequality (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 3 → (4 * x + 3 ≤ 9 - 3 * x ↔ -1 ≤ x ∧ x ≤ 6/7) := by sorry

end fraction_inequality_l237_23747


namespace perpendicular_bisector_c_value_l237_23709

/-- The line x + y = c is a perpendicular bisector of the line segment from (2,4) to (6,8) -/
def is_perpendicular_bisector (c : ℝ) : Prop :=
  let midpoint := ((2 + 6) / 2, (4 + 8) / 2)
  (midpoint.1 + midpoint.2 = c) ∧
  (∀ (x y : ℝ), x + y = c → (x - 2)^2 + (y - 4)^2 = (x - 6)^2 + (y - 8)^2)

/-- If the line x + y = c is a perpendicular bisector of the line segment from (2,4) to (6,8), then c = 10 -/
theorem perpendicular_bisector_c_value :
  ∃ c, is_perpendicular_bisector c → c = 10 := by
  sorry

end perpendicular_bisector_c_value_l237_23709


namespace queue_adjustment_ways_l237_23755

theorem queue_adjustment_ways (n m k : ℕ) (hn : n = 10) (hm : m = 3) (hk : k = 2) :
  (Nat.choose (n - m) k) * (m + 1) * (m + 2) = 420 := by
  sorry

end queue_adjustment_ways_l237_23755


namespace product_of_reciprocals_l237_23745

theorem product_of_reciprocals (a b : ℝ) : 
  a = 1 / (2 - Real.sqrt 3) → 
  b = 1 / (2 + Real.sqrt 3) → 
  a * b = 1 := by
sorry

end product_of_reciprocals_l237_23745


namespace russom_subway_tickets_l237_23751

theorem russom_subway_tickets (bus_tickets : ℕ) (max_envelopes : ℕ) (subway_tickets : ℕ) : 
  bus_tickets = 18 →
  max_envelopes = 6 →
  bus_tickets % max_envelopes = 0 →
  subway_tickets % max_envelopes = 0 →
  subway_tickets > 0 →
  ∀ n : ℕ, n < subway_tickets → n % max_envelopes ≠ 0 ∨ n = 0 →
  subway_tickets = 6 :=
by sorry

end russom_subway_tickets_l237_23751


namespace function_range_l237_23701

/-- The function y = (3 * sin(x) + 1) / (sin(x) + 2) has a range of [-2, 4/3] -/
theorem function_range (x : ℝ) : 
  let y := (3 * Real.sin x + 1) / (Real.sin x + 2)
  ∃ (a b : ℝ), a = -2 ∧ b = 4/3 ∧ a ≤ y ∧ y ≤ b ∧
  (∃ (x₁ x₂ : ℝ), 
    (3 * Real.sin x₁ + 1) / (Real.sin x₁ + 2) = a ∧
    (3 * Real.sin x₂ + 1) / (Real.sin x₂ + 2) = b) :=
by sorry

end function_range_l237_23701


namespace equality_proof_l237_23739

theorem equality_proof (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end equality_proof_l237_23739


namespace product_inequality_l237_23736

theorem product_inequality (a b c d : ℝ) 
  (sum_zero : a + b + c = 0)
  (d_def : d = max (abs a) (max (abs b) (abs c))) :
  abs ((1 + a) * (1 + b) * (1 + c)) ≥ 1 - d^2 :=
by sorry

end product_inequality_l237_23736


namespace decagon_division_impossible_l237_23778

/-- Represents a division of a polygon into colored triangles -/
structure TriangleDivision (n : ℕ) where
  black_sides : ℕ
  white_sides : ℕ
  valid_division : black_sides - white_sides = n

/-- Checks if a number is divisible by 3 -/
def divisible_by_three (n : ℕ) : Prop := ∃ k, n = 3 * k

theorem decagon_division_impossible :
  ¬ ∃ (d : TriangleDivision 10),
    divisible_by_three d.black_sides ∧
    divisible_by_three d.white_sides :=
sorry

end decagon_division_impossible_l237_23778


namespace angle_between_asymptotes_l237_23772

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = Real.sqrt 3 * x
def asymptote2 (x y : ℝ) : Prop := y = -Real.sqrt 3 * x

-- Theorem statement
theorem angle_between_asymptotes :
  ∃ (θ : ℝ), θ = 60 * π / 180 ∧
  (∀ (x y : ℝ), hyperbola x y → 
    (asymptote1 x y ∨ asymptote2 x y) →
    ∃ (x1 y1 x2 y2 : ℝ), 
      asymptote1 x1 y1 ∧ asymptote2 x2 y2 ∧
      Real.cos θ = (x1 * x2 + y1 * y2) / 
        (Real.sqrt (x1^2 + y1^2) * Real.sqrt (x2^2 + y2^2))) :=
by sorry

end angle_between_asymptotes_l237_23772


namespace gold_coins_percentage_l237_23776

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beads_percent : ℝ
  marbles_percent : ℝ
  silver_coins_percent : ℝ
  gold_coins_percent : ℝ

/-- Theorem stating the percentage of gold coins in the urn --/
theorem gold_coins_percentage (u : UrnComposition) 
  (beads_cond : u.beads_percent = 0.3)
  (marbles_cond : u.marbles_percent = 0.1)
  (silver_coins_cond : u.silver_coins_percent = 0.45 * (1 - u.beads_percent - u.marbles_percent))
  (total_cond : u.beads_percent + u.marbles_percent + u.silver_coins_percent + u.gold_coins_percent = 1) :
  u.gold_coins_percent = 0.33 := by
  sorry

#check gold_coins_percentage

end gold_coins_percentage_l237_23776


namespace age_double_time_l237_23798

/-- Given Julio's current age is 42 and James' current age is 8,
    this theorem proves that it will take 26 years for Julio's age to be twice James' age. -/
theorem age_double_time (julio_age : ℕ) (james_age : ℕ) (h1 : julio_age = 42) (h2 : james_age = 8) :
  ∃ (years : ℕ), julio_age + years = 2 * (james_age + years) ∧ years = 26 := by
  sorry

end age_double_time_l237_23798


namespace combined_discount_optimal_l237_23700

/-- Represents the cost calculation for a clothing purchase with discount options -/
def ClothingPurchase (x : ℕ) : Prop :=
  x > 30 ∧
  let jacket_price : ℕ := 100
  let tshirt_price : ℕ := 60
  let option1_cost : ℕ := 3000 + 60 * (x - 30)
  let option2_cost : ℕ := 2400 + 48 * x
  let combined_cost : ℕ := 3000 + 48 * (x - 30)
  combined_cost ≤ min option1_cost option2_cost

/-- Theorem stating that the combined discount strategy is optimal for any valid x -/
theorem combined_discount_optimal (x : ℕ) : ClothingPurchase x := by
  sorry

end combined_discount_optimal_l237_23700


namespace three_digit_primes_with_digit_product_189_l237_23753

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

def target_set : Set ℕ := {379, 397, 739, 937}

theorem three_digit_primes_with_digit_product_189 :
  ∀ n : ℕ, is_three_digit n ∧ Nat.Prime n ∧ digit_product n = 189 ↔ n ∈ target_set :=
sorry

end three_digit_primes_with_digit_product_189_l237_23753


namespace randy_practice_hours_l237_23705

/-- Calculates the number of hours per day Randy needs to practice piano to become an expert --/
def hours_per_day_to_expert (current_age : ℕ) (target_age : ℕ) (practice_days_per_week : ℕ) (vacation_weeks : ℕ) (hours_to_expert : ℕ) : ℚ :=
  let years_to_practice := target_age - current_age
  let weeks_per_year := 52
  let practice_weeks := weeks_per_year - vacation_weeks
  let practice_days_per_year := practice_weeks * practice_days_per_week
  let total_practice_days := years_to_practice * practice_days_per_year
  hours_to_expert / total_practice_days

/-- Theorem stating that Randy needs to practice 5 hours per day to become a piano expert --/
theorem randy_practice_hours :
  hours_per_day_to_expert 12 20 5 2 10000 = 5 := by
  sorry

end randy_practice_hours_l237_23705


namespace intersection_A_B_l237_23720

-- Define set A
def A : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (x + 1)}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Icc 0 2 := by sorry

end intersection_A_B_l237_23720


namespace triangle_trigonometric_identities_l237_23786

theorem triangle_trigonometric_identities (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a = 2 * Real.sin A) →
  (b = 2 * Real.sin B) →
  (c = 2 * Real.sin C) →
  (((a^2 * Real.sin (B - C)) / (Real.sin B * Real.sin C) +
    (b^2 * Real.sin (C - A)) / (Real.sin C * Real.sin A) +
    (c^2 * Real.sin (A - B)) / (Real.sin A * Real.sin B) = 0) ∧
   ((a^2 * Real.sin (B - C)) / (Real.sin B + Real.sin C) +
    (b^2 * Real.sin (C - A)) / (Real.sin C + Real.sin A) +
    (c^2 * Real.sin (A - B)) / (Real.sin A + Real.sin B) = 0)) :=
by sorry

end triangle_trigonometric_identities_l237_23786


namespace point_in_second_quadrant_l237_23742

/-- A point in the second quadrant with given distances to the axes has specific coordinates -/
theorem point_in_second_quadrant (P : ℝ × ℝ) : 
  P.1 < 0 ∧ P.2 > 0 ∧  -- P is in the second quadrant
  |P.2| = 5 ∧          -- distance to x-axis is 5
  |P.1| = 3            -- distance to y-axis is 3
  → P = (-3, 5) := by
sorry

end point_in_second_quadrant_l237_23742


namespace symmetric_line_equation_l237_23788

/-- Given two lines in a plane, this function returns the line that is symmetric to the first line with respect to the second line. -/
def symmetricLine (l1 l2 : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The line y = 2x + 1 -/
def line1 : ℝ → ℝ → Prop :=
  fun x y ↦ y = 2 * x + 1

/-- The line y = x - 2 -/
def line2 : ℝ → ℝ → Prop :=
  fun x y ↦ y = x - 2

/-- The line x - 2y - 7 = 0 -/
def lineL : ℝ → ℝ → Prop :=
  fun x y ↦ x - 2 * y - 7 = 0

theorem symmetric_line_equation :
  symmetricLine line1 line2 = lineL :=
sorry

end symmetric_line_equation_l237_23788


namespace indigo_restaurant_rating_l237_23707

/-- Calculates the average star rating for a restaurant given the number of reviews for each star rating. -/
def averageStarRating (five_star : ℕ) (four_star : ℕ) (three_star : ℕ) (two_star : ℕ) : ℚ :=
  let total_stars := 5 * five_star + 4 * four_star + 3 * three_star + 2 * two_star
  let total_reviews := five_star + four_star + three_star + two_star
  (total_stars : ℚ) / total_reviews

/-- The average star rating for Indigo Restaurant is 4 stars. -/
theorem indigo_restaurant_rating :
  averageStarRating 6 7 4 1 = 4 := by
  sorry


end indigo_restaurant_rating_l237_23707


namespace meal_combinations_count_l237_23718

/-- Represents the number of main dishes available -/
def num_main_dishes : ℕ := 2

/-- Represents the number of stir-fry dishes available -/
def num_stir_fry_dishes : ℕ := 4

/-- Calculates the total number of meal combinations -/
def total_combinations : ℕ := num_main_dishes * num_stir_fry_dishes

/-- Theorem stating that the total number of meal combinations is 8 -/
theorem meal_combinations_count : total_combinations = 8 := by
  sorry

end meal_combinations_count_l237_23718


namespace weaving_increase_proof_l237_23719

/-- Represents the daily increase in cloth production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the number of days in a month -/
def days_in_month : ℕ := 30

/-- Represents the initial daily production in meters -/
def initial_production : ℚ := 5

/-- Represents the total production in meters for the month -/
def total_production : ℚ := 390

/-- Theorem stating that given the initial conditions, the daily increase in production is 16/29 meters -/
theorem weaving_increase_proof :
  initial_production * days_in_month + 
  (days_in_month * (days_in_month - 1) / 2) * daily_increase = 
  total_production :=
by
  sorry


end weaving_increase_proof_l237_23719


namespace arithmetic_mean_problem_l237_23789

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 17 + (2 * x) + 15 + (2 * x + 6) + (3 * x - 5)) / 6 = 30 → x = 137 / 8 := by
  sorry

end arithmetic_mean_problem_l237_23789


namespace speaking_orders_eq_600_l237_23723

-- Define the total number of people in the group
def total_people : ℕ := 7

-- Define the number of people to be selected
def selected_people : ℕ := 4

-- Function to calculate the number of speaking orders
def speaking_orders : ℕ :=
  -- Case 1: Only one of leader or deputy participates
  (2 * (total_people - 2).choose (selected_people - 1) * (selected_people).factorial) +
  -- Case 2: Both leader and deputy participate (not adjacent)
  ((total_people - 2).choose (selected_people - 2) * selected_people.factorial -
   (total_people - 2).choose (selected_people - 2) * 2 * (selected_people - 1).factorial)

-- Theorem statement
theorem speaking_orders_eq_600 :
  speaking_orders = 600 :=
sorry

end speaking_orders_eq_600_l237_23723


namespace fraction_equation_solution_l237_23769

theorem fraction_equation_solution :
  ∃ x : ℝ, x ≠ 1 ∧ (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := by
  sorry

end fraction_equation_solution_l237_23769


namespace first_term_of_geometric_sequence_l237_23749

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = s n * r

/-- Given a geometric sequence where the 6th term is 32 and the 7th term is 64, the first term is 1. -/
theorem first_term_of_geometric_sequence
  (s : ℕ → ℝ)
  (h_geometric : IsGeometricSequence s)
  (h_6th : s 6 = 32)
  (h_7th : s 7 = 64) :
  s 1 = 1 := by
  sorry


end first_term_of_geometric_sequence_l237_23749


namespace unique_integer_solution_range_l237_23785

open Real

theorem unique_integer_solution_range (a : ℝ) : 
  (∃! (x : ℤ), (log (20 - 5 * (x : ℝ)^2) > log (a - (x : ℝ)) + 1)) ↔ 
  (2 ≤ a ∧ a < 5/2) :=
by sorry

end unique_integer_solution_range_l237_23785


namespace red_window_exchange_equations_l237_23713

/-- Represents the relationship between online and offline booth transactions -/
theorem red_window_exchange_equations 
  (x y : ℝ)  -- Total transaction amounts for online (x) and offline (y) booths
  (online_booths : ℕ := 44)  -- Number of online booths
  (offline_booths : ℕ := 71)  -- Number of offline booths
  (h1 : y - 7 * x = 1.8)  -- Relationship between total transaction amounts
  (h2 : y / offline_booths - x / online_booths = 0.3)  -- Difference in average transaction amounts
  : ∃ (system : ℝ × ℝ → Prop), 
    system (x, y) ∧ 
    (∀ (a b : ℝ), system (a, b) ↔ (b - 7 * a = 1.8 ∧ b / offline_booths - a / online_booths = 0.3)) :=
by
  sorry


end red_window_exchange_equations_l237_23713


namespace system_solutions_l237_23715

def has_solution (a : ℝ) : Prop :=
  ∃ (x y : ℝ), y > 0 ∧ x ≥ 0 ∧ y - 2 = a * (x - 4) ∧ 2 * x / (|y| + y) = Real.sqrt x

theorem system_solutions (a : ℝ) :
  (a ≤ 0 ∨ a = 1/4) →
    (has_solution a ∧
     ∃ (x y : ℝ), (x = 0 ∧ y = 2 - 4*a) ∨ (x = 4 ∧ y = 2)) ∧
  ((0 < a ∧ a < 1/4) ∨ (1/4 < a ∧ a < 1/2)) →
    (has_solution a ∧
     ∃ (x y : ℝ), (x = 0 ∧ y = 2 - 4*a) ∨ (x = 4 ∧ y = 2) ∨ (x = ((1-2*a)/a)^2 ∧ y = (1-2*a)/a)) ∧
  (a ≥ 1/2) →
    (has_solution a ∧
     ∃ (x y : ℝ), x = 4 ∧ y = 2) :=
by sorry


end system_solutions_l237_23715


namespace smallest_number_l237_23711

def number_set : Finset ℤ := {0, -3, 2, -2}

theorem smallest_number : 
  ∀ x ∈ number_set, -3 ≤ x :=
by
  sorry

end smallest_number_l237_23711


namespace parallelogram_segment_sum_l237_23708

/-- A grid of equilateral triangles -/
structure TriangularGrid where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A parallelogram on the triangular grid -/
structure Parallelogram (grid : TriangularGrid) where
  vertices : Fin 4 → ℕ × ℕ  -- Grid coordinates of the vertices
  area : ℝ

/-- The possible sums of lengths of grid segments inside the parallelogram -/
def possible_segment_sums (grid : TriangularGrid) (p : Parallelogram grid) : Set ℝ :=
  {3, 4, 5, 6}

theorem parallelogram_segment_sum 
  (grid : TriangularGrid) 
  (p : Parallelogram grid) 
  (h_side_length : grid.side_length = 1) 
  (h_area : p.area = Real.sqrt 3) :
  ∃ (sum : ℝ), sum ∈ possible_segment_sums grid p :=
sorry

end parallelogram_segment_sum_l237_23708


namespace insurance_coverage_percentage_l237_23741

theorem insurance_coverage_percentage 
  (total_cost : ℝ) 
  (out_of_pocket : ℝ) 
  (h1 : total_cost = 500) 
  (h2 : out_of_pocket = 200) : 
  (total_cost - out_of_pocket) / total_cost * 100 = 60 := by
sorry

end insurance_coverage_percentage_l237_23741


namespace copenhagen_aarhus_distance_l237_23777

/-- The distance between two city centers with a detour -/
def distance_with_detour (map_distance : ℝ) (scale : ℝ) (detour_increase : ℝ) : ℝ :=
  map_distance * scale * (1 + detour_increase)

/-- Theorem: The distance between Copenhagen and Aarhus is 420 km -/
theorem copenhagen_aarhus_distance :
  distance_with_detour 35 10 0.2 = 420 := by
  sorry

end copenhagen_aarhus_distance_l237_23777


namespace not_perfect_power_of_ten_sixes_and_zeros_l237_23748

def is_composed_of_ten_sixes_and_zeros (n : ℕ) : Prop :=
  ∃ k, n = 6666666666 * 10^k

theorem not_perfect_power_of_ten_sixes_and_zeros (n : ℕ) 
  (h : is_composed_of_ten_sixes_and_zeros n) : 
  ¬ ∃ (a b : ℕ), b > 1 ∧ n = a^b :=
sorry

end not_perfect_power_of_ten_sixes_and_zeros_l237_23748


namespace triangle_and_division_counts_l237_23752

/-- The number of non-congruent triangles formed by m equally spaced points on a circle -/
def num_triangles (m : ℕ) : ℕ :=
  let k := m / 6
  match m % 6 with
  | 0 => 3*k^2 - 3*k + 1
  | 1 => 3*k^2 - 2*k
  | 2 => 3*k^2 - k
  | 3 => 3*k^2
  | 4 => 3*k^2 + k
  | 5 => 3*k^2 + 2*k
  | _ => 0  -- This case should never occur

/-- The number of ways to divide m identical items into 3 groups -/
def num_divisions (m : ℕ) : ℕ :=
  let k := m / 6
  match m % 6 with
  | 0 => 3*k^2
  | 1 => 3*k^2 + k
  | 2 => 3*k^2 + 2*k
  | 3 => 3*k^2 + 3*k + 1
  | 4 => 3*k^2 + 4*k + 1
  | 5 => 3*k^2 + 5*k + 2
  | _ => 0  -- This case should never occur

theorem triangle_and_division_counts (m : ℕ) (h : m ≥ 3) :
  (num_triangles m = num_triangles m) ∧ (num_divisions m = num_divisions m) :=
sorry

end triangle_and_division_counts_l237_23752


namespace left_handed_women_percentage_l237_23721

/-- Represents the population distribution in Smithtown -/
structure SmithtownPopulation where
  right_handed : ℕ
  left_handed : ℕ
  men : ℕ
  women : ℕ

/-- Conditions for Smithtown's population distribution -/
def valid_distribution (p : SmithtownPopulation) : Prop :=
  p.right_handed = 3 * p.left_handed ∧
  p.men = 3 * p.women / 2 ∧
  p.right_handed + p.left_handed = p.men + p.women ∧
  p.right_handed ≥ p.men

/-- Theorem: In a valid Smithtown population distribution, 
    left-handed women constitute 25% of the total population -/
theorem left_handed_women_percentage 
  (p : SmithtownPopulation) 
  (h : valid_distribution p) : 
  (p.left_handed : ℚ) / (p.right_handed + p.left_handed : ℚ) = 1/4 := by
  sorry

end left_handed_women_percentage_l237_23721


namespace coordinates_wrt_y_axis_l237_23737

/-- Given a point A with coordinates (3,-1) in the standard coordinate system,
    its coordinates with respect to the y-axis are (-3,-1). -/
theorem coordinates_wrt_y_axis :
  let A : ℝ × ℝ := (3, -1)
  let A_y_axis : ℝ × ℝ := (-3, -1)
  A_y_axis = (- A.1, A.2) :=
by sorry

end coordinates_wrt_y_axis_l237_23737


namespace fraction_inequality_l237_23733

theorem fraction_inequality (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hab : a > b) :
  b / a < (b + x) / (a + x) := by
  sorry

end fraction_inequality_l237_23733


namespace correct_factorization_l237_23792

theorem correct_factorization (x y : ℝ) : x^3 + 4*x^2*y + 4*x*y^2 = x * (x + 2*y)^2 := by
  sorry

end correct_factorization_l237_23792


namespace cuboid_sum_of_edges_l237_23770

/-- Represents the dimensions of a rectangular cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Checks if the dimensions form a geometric progression -/
def isGeometricProgression (d : CuboidDimensions) : Prop :=
  ∃ q : ℝ, q > 0 ∧ d.length = q * d.width ∧ d.width = q * d.height

/-- Calculates the volume of a rectangular cuboid -/
def volume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the surface area of a rectangular cuboid -/
def surfaceArea (d : CuboidDimensions) : ℝ :=
  2 * (d.length * d.width + d.width * d.height + d.height * d.length)

/-- Calculates the sum of all edges of a rectangular cuboid -/
def sumOfEdges (d : CuboidDimensions) : ℝ :=
  4 * (d.length + d.width + d.height)

/-- Theorem: For a rectangular cuboid with volume 8, surface area 32, and dimensions 
    forming a geometric progression, the sum of all edges is 32 -/
theorem cuboid_sum_of_edges : 
  ∀ d : CuboidDimensions, 
    volume d = 8 → 
    surfaceArea d = 32 → 
    isGeometricProgression d → 
    sumOfEdges d = 32 := by
  sorry

end cuboid_sum_of_edges_l237_23770


namespace compute_F_3_f_5_l237_23768

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 2
def F (a b : ℝ) : ℝ := b^3 - a

-- State the theorem
theorem compute_F_3_f_5 : F 3 (f 5) = 24 := by sorry

end compute_F_3_f_5_l237_23768


namespace triangle_construction_valid_l237_23706

/-- A triangle can be constructed with perimeter k, one side c, and angle difference δ
    between angles opposite the other two sides if and only if 2c < k. -/
theorem triangle_construction_valid (k c : ℝ) (δ : ℝ) :
  (∃ (a b : ℝ) (α β γ : ℝ),
    a + b + c = k ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    α + β + γ = π ∧
    α - β = δ ∧
    0 < α ∧ α < π ∧
    0 < β ∧ β < π ∧
    0 < γ ∧ γ < π) ↔
  2 * c < k :=
by sorry


end triangle_construction_valid_l237_23706


namespace eva_total_marks_2019_l237_23759

/-- Eva's marks in different subjects and semesters -/
structure EvaMarks where
  maths_second : ℕ
  arts_second : ℕ
  science_second : ℕ
  maths_first : ℕ
  arts_first : ℕ
  science_first : ℕ

/-- Calculate the total marks for Eva in 2019 -/
def total_marks (marks : EvaMarks) : ℕ :=
  marks.maths_first + marks.arts_first + marks.science_first +
  marks.maths_second + marks.arts_second + marks.science_second

/-- Theorem stating Eva's total marks in 2019 -/
theorem eva_total_marks_2019 (marks : EvaMarks)
  (h1 : marks.maths_second = 80)
  (h2 : marks.arts_second = 90)
  (h3 : marks.science_second = 90)
  (h4 : marks.maths_first = marks.maths_second + 10)
  (h5 : marks.arts_first = marks.arts_second - 15)
  (h6 : marks.science_first = marks.science_second - marks.science_second / 3) :
  total_marks marks = 485 := by
  sorry


end eva_total_marks_2019_l237_23759


namespace quadratic_roots_problem_l237_23730

theorem quadratic_roots_problem (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + 2*k*x + k^2 = x + 1 ↔ x = x₁ ∨ x = x₂) ∧
    (3*x₁ - x₂)*(x₁ - 3*x₂) = 19) →
  k = 0 ∨ k = -3 :=
by sorry

end quadratic_roots_problem_l237_23730


namespace yellow_marbles_count_l237_23763

/-- Given a total number of marbles, with blue marbles being three times
    the number of red marbles, and a specific number of red marbles,
    prove the number of yellow marbles. -/
theorem yellow_marbles_count
  (total : ℕ)
  (red : ℕ)
  (h1 : total = 85)
  (h2 : red = 14) :
  total - (red + 3 * red) = 29 := by
  sorry

end yellow_marbles_count_l237_23763


namespace nine_rings_puzzle_l237_23725

def min_moves : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => min_moves (n + 2) + 2 * min_moves (n + 1) + 1

theorem nine_rings_puzzle : min_moves 7 = 85 := by
  sorry

end nine_rings_puzzle_l237_23725


namespace johns_father_age_multiple_l237_23796

/-- 
Given John's age, the sum of John and his father's ages, and the relationship between
John's father's age and John's age, this theorem proves the multiple of John's age
that represents his father's age without the additional 32 years.
-/
theorem johns_father_age_multiple 
  (john_age : ℕ)
  (sum_ages : ℕ)
  (father_age_relation : ℕ → ℕ)
  (h1 : john_age = 15)
  (h2 : sum_ages = 77)
  (h3 : father_age_relation m = m * john_age + 32)
  (h4 : sum_ages = john_age + father_age_relation m) :
  m = 2 := by
  sorry

end johns_father_age_multiple_l237_23796


namespace basketball_team_selection_l237_23732

theorem basketball_team_selection (total_players : Nat) (twins : Nat) (lineup_size : Nat) : 
  total_players = 12 →
  twins = 2 →
  lineup_size = 5 →
  (twins * (total_players - twins).choose (lineup_size - 1)) = 420 :=
by
  sorry

end basketball_team_selection_l237_23732


namespace existence_of_point_l237_23782

theorem existence_of_point (f : ℝ → ℝ) (h_pos : ∀ x, f x > 0) (h_nondec : ∀ x y, x ≤ y → f x ≤ f y) :
  ∃ a : ℝ, f (a + 1 / f a) < 2 * f a := by
  sorry

end existence_of_point_l237_23782


namespace total_cost_15_pencils_9_notebooks_l237_23729

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- The first given condition: 8 pencils and 5 notebooks cost $3.90 -/
axiom first_condition : 8 * pencil_cost + 5 * notebook_cost = 3.90

/-- The second given condition: 6 pencils and 4 notebooks cost $2.96 -/
axiom second_condition : 6 * pencil_cost + 4 * notebook_cost = 2.96

/-- The theorem to be proved -/
theorem total_cost_15_pencils_9_notebooks : 
  15 * pencil_cost + 9 * notebook_cost = 7.26 := by sorry

end total_cost_15_pencils_9_notebooks_l237_23729


namespace function_value_at_ten_l237_23793

theorem function_value_at_ten (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (2*x + y) + 7*x*y + 3*y^2 = f (3*x - y) + 3*x^2 + 2) : 
  f 10 = -123 := by
sorry

end function_value_at_ten_l237_23793


namespace multiply_826446281_by_11_twice_l237_23762

theorem multiply_826446281_by_11_twice :
  826446281 * 11 * 11 = 100000000001 := by
  sorry

end multiply_826446281_by_11_twice_l237_23762


namespace third_smallest_four_digit_in_pascal_l237_23758

/-- Represents a row in Pascal's triangle -/
def PascalRow := List Nat

/-- Generates the next row of Pascal's triangle given the current row -/
def nextPascalRow (row : PascalRow) : PascalRow :=
  sorry

/-- Checks if a number is a four-digit number -/
def isFourDigit (n : Nat) : Bool :=
  1000 ≤ n ∧ n ≤ 9999

/-- Finds the nth four-digit number in Pascal's triangle -/
def nthFourDigitInPascal (n : Nat) : Nat :=
  sorry

theorem third_smallest_four_digit_in_pascal :
  nthFourDigitInPascal 3 = 1002 :=
sorry

end third_smallest_four_digit_in_pascal_l237_23758


namespace b_investment_is_8000_l237_23703

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Calculates b's investment in the partnership. -/
def calculate_b_investment (p : Partnership) : ℕ :=
  (22 * p.c_investment - 9 * (p.a_investment + p.c_investment)) / 9

/-- Theorem stating that given the conditions, b's investment is 8000. -/
theorem b_investment_is_8000 (p : Partnership)
  (h1 : p.a_investment = 5000)
  (h2 : p.c_investment = 9000)
  (h3 : p.total_profit = 88000)
  (h4 : p.c_profit = 36000)
  : calculate_b_investment p = 8000 := by
  sorry

#eval calculate_b_investment ⟨5000, 0, 9000, 88000, 36000⟩

end b_investment_is_8000_l237_23703


namespace chess_game_probability_l237_23779

/-- The probability of player A winning a chess game -/
def prob_A_win : ℝ := 0.3

/-- The probability of the chess game ending in a draw -/
def prob_draw : ℝ := 0.5

/-- The probability of player B not losing the chess game -/
def prob_B_not_lose : ℝ := 1 - prob_A_win

theorem chess_game_probability : prob_B_not_lose = 0.7 := by
  sorry

end chess_game_probability_l237_23779


namespace comprehensive_formula_l237_23791

theorem comprehensive_formula (h1 : 12 * 5 = 60) (h2 : 60 - 42 = 18) :
  12 * 5 - 42 = 18 := by
  sorry

end comprehensive_formula_l237_23791


namespace triangle_properties_l237_23794

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  (t.b - t.c)^2 = t.a^2 - t.b * t.c ∧
  t.a = 3 ∧
  Real.sin t.C = 2 * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.A = π / 2 ∧ (1/2 * t.b * t.c * Real.sin t.A) = 1 := by
  sorry

end triangle_properties_l237_23794


namespace lowest_energy_point_min_energy_at_two_l237_23790

/-- Represents the energy function for an athlete during a 4-hour training session. -/
noncomputable def Q (t : ℝ) : ℝ :=
  if 0 < t ∧ t ≤ 1 then
    10000 - 3600 * t
  else if 1 < t ∧ t ≤ 4 then
    400 + 1200 * t + 4800 / t
  else
    0

/-- Theorem stating that the athlete's energy reaches its lowest point at t = 2 hours with a value of 5200kJ. -/
theorem lowest_energy_point :
  ∀ t : ℝ, 0 < t ∧ t ≤ 4 → Q t ≥ 5200 ∧ Q 2 = 5200 := by sorry

/-- Corollary stating that the minimum energy occurs at t = 2. -/
theorem min_energy_at_two :
  ∀ t : ℝ, 0 < t ∧ t ≤ 4 → Q 2 ≤ Q t := by sorry

end lowest_energy_point_min_energy_at_two_l237_23790


namespace quadratic_equation_completing_square_l237_23702

theorem quadratic_equation_completing_square (x : ℝ) :
  ∃ (q t : ℝ), (16 * x^2 - 32 * x - 512 = 0) ↔ ((x + q)^2 = t) ∧ t = 1 := by
  sorry

end quadratic_equation_completing_square_l237_23702


namespace quadratic_coefficient_l237_23726

theorem quadratic_coefficient (a : ℚ) : 
  (∀ x, (x + 4)^2 * a = (x + 4)^2 * (-8/9)) → 
  a = -8/9 := by
  sorry

end quadratic_coefficient_l237_23726


namespace problem_solution_l237_23787

theorem problem_solution (m : ℝ) (h : m + 1/m = 10) :
  m^2 + 1/m^2 + m^3 + 1/m^3 + 4 = 1072 := by
  sorry

end problem_solution_l237_23787


namespace ellipse_slope_theorem_l237_23740

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define a point on the ellipse
def point_on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

-- Define a line with negative slope
def negative_slope_line (k t : ℝ) (x y : ℝ) : Prop := y = k*x + t ∧ k < 0

-- Define the centroid condition
def centroid_origin (a b c : ℝ × ℝ) : Prop :=
  a.1 + b.1 + c.1 = 0 ∧ a.2 + b.2 + c.2 = 0

-- Define the area ratio condition
def area_ratio_condition (b m a c o : ℝ × ℝ) : Prop :=
  2 * (b.1 - m.1) * (a.2 - m.2) = 3 * (c.1 - m.1) * (o.2 - m.2)

-- Main theorem
theorem ellipse_slope_theorem (a b c m : ℝ × ℝ) (k t : ℝ) :
  point_on_ellipse a ∧ point_on_ellipse b ∧ point_on_ellipse c ∧
  negative_slope_line k t b.1 b.2 ∧
  negative_slope_line k t c.1 c.2 ∧
  m.1 = 0 ∧
  centroid_origin a b c ∧
  area_ratio_condition b m a c (0, 0) →
  k = -3*Real.sqrt 3/2 ∨ k = -Real.sqrt 3/6 := by sorry

end ellipse_slope_theorem_l237_23740


namespace quadratic_roots_count_l237_23797

/-- The number of real roots of the quadratic function y = x^2 + x - 1 is 2 -/
theorem quadratic_roots_count : 
  let f : ℝ → ℝ := fun x ↦ x^2 + x - 1
  (∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0) ∧ 
  (∀ (x y z : ℝ), f x = 0 → f y = 0 → f z = 0 → x = y ∨ x = z ∨ y = z) := by
  sorry

end quadratic_roots_count_l237_23797


namespace four_integer_pairs_satisfying_equation_l237_23765

theorem four_integer_pairs_satisfying_equation :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 1) ∧
    s.card = 4 := by
  sorry

end four_integer_pairs_satisfying_equation_l237_23765


namespace hot_dog_problem_l237_23716

theorem hot_dog_problem (cost_per_hot_dog : ℕ) (total_paid : ℕ) (h1 : cost_per_hot_dog = 50) (h2 : total_paid = 300) :
  total_paid / cost_per_hot_dog = 6 := by
sorry

end hot_dog_problem_l237_23716


namespace towel_rate_calculation_l237_23783

/-- Given the prices and quantities of towels, calculates the unknown rate. -/
def unknown_towel_rate (qty1 qty2 qty3 : ℕ) (price1 price2 avg_price : ℚ) : ℚ :=
  ((qty1 + qty2 + qty3 : ℚ) * avg_price - qty1 * price1 - qty2 * price2) / qty3

/-- Theorem stating that under the given conditions, the unknown rate is 325. -/
theorem towel_rate_calculation :
  let qty1 := 3
  let qty2 := 5
  let qty3 := 2
  let price1 := 100
  let price2 := 150
  let avg_price := 170
  unknown_towel_rate qty1 qty2 qty3 price1 price2 avg_price = 325 := by
sorry

end towel_rate_calculation_l237_23783


namespace expression_simplification_l237_23724

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 3) :
  (3 * x) / (x^2 - 9) * (1 - 3 / x) - 2 / (x + 3) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l237_23724


namespace trip_cost_calculation_l237_23738

def initial_odometer : ℕ := 85300
def final_odometer : ℕ := 85335
def fuel_efficiency : ℚ := 25
def gas_price : ℚ := 21/5  -- $4.20 represented as a rational number

def trip_cost : ℚ :=
  (final_odometer - initial_odometer : ℚ) / fuel_efficiency * gas_price

theorem trip_cost_calculation :
  trip_cost = 588/100 := by sorry

end trip_cost_calculation_l237_23738


namespace fraction_equality_l237_23781

theorem fraction_equality (a b : ℝ) (h : a / b = 3 / 5) : (2 * a + 3 * b) / a = 7 := by
  sorry

end fraction_equality_l237_23781


namespace parabola_focus_focus_of_y_eq_4x_squared_l237_23722

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1 / (4 * a))
  ∀ x y : ℝ, y = a * x^2 → (x - f.1)^2 + (y - f.2)^2 = (y - f.2 + 1 / (4 * a))^2 :=
sorry

/-- The focus of the parabola y = 4x^2 is at (0, 1/16) -/
theorem focus_of_y_eq_4x_squared :
  let f : ℝ × ℝ := (0, 1/16)
  ∀ x y : ℝ, y = 4 * x^2 → (x - f.1)^2 + (y - f.2)^2 = (y - f.2 + 1/16)^2 :=
sorry

end parabola_focus_focus_of_y_eq_4x_squared_l237_23722


namespace problem_solution_l237_23731

noncomputable def g (θ : Real) (x : Real) : Real := x * Real.sin θ - Real.log x - Real.sin θ

noncomputable def f (θ : Real) (x : Real) : Real := g θ x + (2*x - 1) / (2*x^2)

theorem problem_solution (θ : Real) (h1 : θ ∈ Set.Ioo 0 Real.pi) 
  (h2 : ∀ x ≥ 1, Monotone (g θ)) : 
  (θ = Real.pi/2) ∧ 
  (∀ x ∈ Set.Icc 1 2, f θ x > (deriv (f θ)) x + 1/2) ∧
  (∀ k > 1, ∃ x > 0, Real.exp x - x - 1 < k * g θ (x+1)) :=
sorry

end problem_solution_l237_23731


namespace perpendicular_lines_a_value_l237_23750

/-- Given a line l passing through points (a-2, -1) and (-a-2, 1), 
    and perpendicular to the line 2x+3y+1=0, prove that a = -2/3 -/
theorem perpendicular_lines_a_value (a : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (a - 2 + t*(-2*a), -1 + t*2)}
  let slope_l := (1 - (-1)) / ((-a - 2) - (a - 2))
  let slope_other := -2 / 3
  (∀ p ∈ l, 2 * p.1 + 3 * p.2 + 1 ≠ 0) → 
  (slope_l * slope_other = -1) →
  a = -2/3 := by
sorry

end perpendicular_lines_a_value_l237_23750


namespace cubic_equation_one_real_root_l237_23780

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, x^3 - 6*x^2 + 9*x - 10 = 0 :=
sorry

end cubic_equation_one_real_root_l237_23780


namespace line_circle_intersection_l237_23767

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
def distPointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Predicate to check if a line intersects a circle -/
def intersects (c : Circle) (l : Line) : Prop :=
  distPointToLine c.center l < c.radius

theorem line_circle_intersection (c : Circle) (l : Line) :
  distPointToLine c.center l < c.radius → intersects c l := by
  sorry

end line_circle_intersection_l237_23767


namespace inequality_theorem_l237_23766

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hy₁ : x₁ * y₁ - z₁^2 > 0) (hy₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) :=
by sorry

end inequality_theorem_l237_23766


namespace merchant_markup_problem_l237_23746

theorem merchant_markup_problem (markup_percentage : ℝ) : 
  (∀ cost_price : ℝ, cost_price > 0 →
    let marked_price := cost_price * (1 + markup_percentage / 100)
    let discounted_price := marked_price * (1 - 25 / 100)
    let profit_percentage := (discounted_price - cost_price) / cost_price * 100
    profit_percentage = 20) →
  markup_percentage = 60 := by
sorry

end merchant_markup_problem_l237_23746


namespace unique_positive_solution_l237_23799

theorem unique_positive_solution : 
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x := by
sorry

end unique_positive_solution_l237_23799


namespace min_value_of_exponential_sum_l237_23710

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + y = 3) :
  2^x + 2^y ≥ 4 * Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 3 ∧ 2^x₀ + 2^y₀ = 4 * Real.sqrt 2 :=
by sorry

end min_value_of_exponential_sum_l237_23710


namespace no_solution_implies_a_leq_8_l237_23771

theorem no_solution_implies_a_leq_8 (a : ℝ) :
  (∀ x : ℝ, ¬(abs (x - 5) + abs (x + 3) < a)) → a ≤ 8 := by
  sorry

end no_solution_implies_a_leq_8_l237_23771


namespace soccer_goals_average_l237_23743

theorem soccer_goals_average : 
  let players_with_3_goals : ℕ := 2
  let players_with_4_goals : ℕ := 3
  let players_with_5_goals : ℕ := 1
  let players_with_6_goals : ℕ := 1
  let total_goals : ℕ := 3 * players_with_3_goals + 4 * players_with_4_goals + 
                          5 * players_with_5_goals + 6 * players_with_6_goals
  let total_players : ℕ := players_with_3_goals + players_with_4_goals + 
                           players_with_5_goals + players_with_6_goals
  (total_goals : ℚ) / total_players = 29 / 7 := by
  sorry

end soccer_goals_average_l237_23743


namespace composite_number_impossibility_l237_23712

theorem composite_number_impossibility (n a q : ℕ) (h_n : n > 1) (h_q_prime : Nat.Prime q) 
  (h_q_div : q ∣ (n - 1)) (h_q_sqrt : q > Nat.sqrt n - 1) (h_n_div : n ∣ (a^(n-1) - 1)) 
  (h_gcd : Nat.gcd (a^((n-1)/q) - 1) n = 1) : 
  Nat.Prime n := by
sorry

end composite_number_impossibility_l237_23712


namespace soccer_leagues_games_l237_23773

/-- Calculate the number of games in a round-robin tournament -/
def gamesInLeague (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of games played across three leagues -/
def totalGames (a b c : ℕ) : ℕ := gamesInLeague a + gamesInLeague b + gamesInLeague c

theorem soccer_leagues_games :
  totalGames 20 25 30 = 925 := by
  sorry

end soccer_leagues_games_l237_23773


namespace square_pyramid_sum_l237_23760

/-- A square pyramid is a polyhedron with a square base and triangular lateral faces -/
structure SquarePyramid where
  base : Nat
  lateral_faces : Nat
  base_edges : Nat
  lateral_edges : Nat
  base_vertices : Nat
  apex : Nat

/-- Properties of a square pyramid -/
def square_pyramid : SquarePyramid :=
  { base := 1
  , lateral_faces := 4
  , base_edges := 4
  , lateral_edges := 4
  , base_vertices := 4
  , apex := 1 }

/-- The sum of faces, edges, and vertices of a square pyramid is 18 -/
theorem square_pyramid_sum :
  (square_pyramid.base + square_pyramid.lateral_faces) +
  (square_pyramid.base_edges + square_pyramid.lateral_edges) +
  (square_pyramid.base_vertices + square_pyramid.apex) = 18 := by
  sorry

end square_pyramid_sum_l237_23760


namespace x_divisibility_l237_23714

def x : ℕ := 128 + 192 + 256 + 320 + 576 + 704 + 6464

theorem x_divisibility :
  (∃ k : ℕ, x = 8 * k) ∧
  (∃ k : ℕ, x = 16 * k) ∧
  (∃ k : ℕ, x = 32 * k) ∧
  (∃ k : ℕ, x = 64 * k) :=
by sorry

end x_divisibility_l237_23714


namespace geometric_sequence_solution_l237_23728

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x else Real.log x / x

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_solution (a : ℕ → ℝ) (r : ℝ) :
  is_geometric_sequence a r →
  r > 0 →
  a 3 * a 4 * a 5 = 1 →
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1 →
  a 1 = Real.exp 2 := by
  sorry

end geometric_sequence_solution_l237_23728


namespace third_job_hourly_rate_l237_23795

-- Define the problem parameters
def total_earnings : ℝ := 430
def first_job_hours : ℝ := 15
def first_job_rate : ℝ := 8
def second_job_sales : ℝ := 1000
def second_job_commission_rate : ℝ := 0.1
def third_job_hours : ℝ := 12
def tax_deduction : ℝ := 50

-- Define the theorem
theorem third_job_hourly_rate :
  let first_job_earnings := first_job_hours * first_job_rate
  let second_job_earnings := second_job_sales * second_job_commission_rate
  let combined_wages := first_job_earnings + second_job_earnings
  let combined_wages_after_tax := combined_wages - tax_deduction
  let third_job_earnings := total_earnings - combined_wages_after_tax
  third_job_earnings / third_job_hours = 21.67 := by
  sorry

end third_job_hourly_rate_l237_23795
