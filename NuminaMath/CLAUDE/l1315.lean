import Mathlib

namespace NUMINAMATH_CALUDE_three_fourths_of_hundred_l1315_131569

theorem three_fourths_of_hundred : (3 / 4 : ℚ) * 100 = 75 := by sorry

end NUMINAMATH_CALUDE_three_fourths_of_hundred_l1315_131569


namespace NUMINAMATH_CALUDE_product_from_gcd_lcm_l1315_131567

theorem product_from_gcd_lcm (a b : ℤ) : 
  Int.gcd a b = 8 → Int.lcm a b = 48 → a * b = 384 := by
  sorry

end NUMINAMATH_CALUDE_product_from_gcd_lcm_l1315_131567


namespace NUMINAMATH_CALUDE_exists_three_numbers_sum_geq_54_l1315_131549

theorem exists_three_numbers_sum_geq_54 
  (S : Finset ℕ) 
  (distinct : S.card = 10) 
  (sum_gt_144 : S.sum id > 144) : 
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c ≥ 54 :=
by sorry

end NUMINAMATH_CALUDE_exists_three_numbers_sum_geq_54_l1315_131549


namespace NUMINAMATH_CALUDE_function_range_l1315_131506

/-- Given a function f(x) = x + 4a/x - a, where a < 0, 
    if f(x) < 0 for all x in (0, 1], then a ≤ -1/3 -/
theorem function_range (a : ℝ) (h1 : a < 0) :
  (∀ x ∈ Set.Ioo 0 1, x + 4 * a / x - a < 0) →
  a ≤ -1/3 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1315_131506


namespace NUMINAMATH_CALUDE_a_range_for_region_above_l1315_131532

/-- The inequality represents the region above the line -/
def represents_region_above (a : ℝ) : Prop :=
  ∀ x y : ℝ, 3 * a * x + (a^2 - 3 * a + 2) * y - 9 < 0 ↔ 
    y > (9 - 3 * a * x) / (a^2 - 3 * a + 2)

/-- The theorem stating the range of a -/
theorem a_range_for_region_above : 
  ∀ a : ℝ, represents_region_above a ↔ 1 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_a_range_for_region_above_l1315_131532


namespace NUMINAMATH_CALUDE_min_value_theorem_l1315_131512

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * x^2 + 1 / x^2 ≥ 2 * Real.sqrt 3 ∧ ∃ y > 0, 3 * y^2 + 1 / y^2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1315_131512


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1315_131584

theorem cubic_equation_roots (a b : ℝ) :
  (∃ x y z : ℕ+, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    (∀ t : ℝ, t^3 - 8*t^2 + a*t - b = 0 ↔ (t = x ∨ t = y ∨ t = z))) →
  a + b = 31 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1315_131584


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_sqrt_prime_l1315_131544

theorem sqrt_sum_eq_sqrt_prime (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ, Real.sqrt x + Real.sqrt y = Real.sqrt p ↔ (x = 0 ∧ y = p) ∨ (x = p ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_sqrt_prime_l1315_131544


namespace NUMINAMATH_CALUDE_john_jury_duty_days_l1315_131539

def jury_duty_days (jury_selection_days : ℕ) 
                   (trial_duration_multiplier : ℕ) 
                   (trial_extra_hours_per_day : ℕ) 
                   (deliberation_equivalent_full_days : ℕ) 
                   (deliberation_hours_per_day : ℕ) : ℕ :=
  let trial_days := jury_selection_days * trial_duration_multiplier
  let trial_extra_days := (trial_days * trial_extra_hours_per_day) / 24
  let deliberation_days := 
    (deliberation_equivalent_full_days * 24 + deliberation_hours_per_day - 1) / deliberation_hours_per_day
  jury_selection_days + trial_days + trial_extra_days + deliberation_days

theorem john_jury_duty_days : 
  jury_duty_days 2 4 3 6 14 = 22 := by sorry

end NUMINAMATH_CALUDE_john_jury_duty_days_l1315_131539


namespace NUMINAMATH_CALUDE_andy_diana_weight_l1315_131528

theorem andy_diana_weight (a b c d : ℝ) 
  (h1 : a + b = 300)
  (h2 : b + c = 280)
  (h3 : c + d = 310) :
  a + d = 330 := by
  sorry

end NUMINAMATH_CALUDE_andy_diana_weight_l1315_131528


namespace NUMINAMATH_CALUDE_total_days_2005_to_2010_l1315_131543

def is_leap_year (year : ℕ) : Bool := year = 2008

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def year_range : List ℕ := [2005, 2006, 2007, 2008, 2009, 2010]

theorem total_days_2005_to_2010 :
  (year_range.map days_in_year).sum = 2191 := by
  sorry

end NUMINAMATH_CALUDE_total_days_2005_to_2010_l1315_131543


namespace NUMINAMATH_CALUDE_train_speed_l1315_131511

theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 160)
  (h2 : bridge_length = 215)
  (h3 : crossing_time = 30) : 
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1315_131511


namespace NUMINAMATH_CALUDE_book_ratio_problem_l1315_131585

theorem book_ratio_problem (darla_books katie_books gary_books : ℕ) : 
  darla_books = 6 →
  gary_books = 5 * (darla_books + katie_books) →
  darla_books + katie_books + gary_books = 54 →
  katie_books = darla_books / 2 := by
  sorry

end NUMINAMATH_CALUDE_book_ratio_problem_l1315_131585


namespace NUMINAMATH_CALUDE_daily_harvest_l1315_131534

/-- The number of sections in the orchard -/
def sections : ℕ := 8

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := sections * sacks_per_section

theorem daily_harvest : total_sacks = 360 := by
  sorry

end NUMINAMATH_CALUDE_daily_harvest_l1315_131534


namespace NUMINAMATH_CALUDE_acme_vowel_soup_words_l1315_131580

/-- Represents the count of each vowel in the alphabet soup -/
structure VowelCount where
  a : Nat
  e : Nat
  i : Nat
  o : Nat
  u : Nat

/-- The modified Acme alphabet soup recipe -/
def acmeVowelSoup : VowelCount :=
  { a := 4, e := 6, i := 5, o := 3, u := 2 }

/-- The length of words to be formed -/
def wordLength : Nat := 5

/-- Calculates the number of five-letter words that can be formed from the given vowel counts -/
def countWords (vc : VowelCount) (len : Nat) : Nat :=
  sorry

/-- Theorem stating that the number of five-letter words from Acme Vowel Soup is 1125 -/
theorem acme_vowel_soup_words :
  countWords acmeVowelSoup wordLength = 1125 := by
  sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_words_l1315_131580


namespace NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l1315_131517

theorem arccos_of_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l1315_131517


namespace NUMINAMATH_CALUDE_tree_distance_l1315_131557

/-- Given two buildings 220 meters apart with 10 trees planted at equal intervals,
    the distance between the 1st tree and the 6th tree is 100 meters. -/
theorem tree_distance (building_distance : ℝ) (num_trees : ℕ) 
  (h1 : building_distance = 220)
  (h2 : num_trees = 10) : 
  let interval := building_distance / (num_trees + 1)
  (6 - 1) * interval = 100 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l1315_131557


namespace NUMINAMATH_CALUDE_simplify_expression_l1315_131594

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 81) - Real.sqrt (8 + 1/2))^2 = (35 - 6 * Real.sqrt 34) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1315_131594


namespace NUMINAMATH_CALUDE_helen_cookies_l1315_131591

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 435

/-- The number of cookies Helen baked this morning -/
def cookies_today : ℕ := 139

/-- The total number of cookies Helen baked -/
def total_cookies : ℕ := cookies_yesterday + cookies_today

/-- Theorem stating that the total number of cookies Helen baked is 574 -/
theorem helen_cookies : total_cookies = 574 := by sorry

end NUMINAMATH_CALUDE_helen_cookies_l1315_131591


namespace NUMINAMATH_CALUDE_martha_improvement_l1315_131504

/-- Represents Martha's running performance at a given time --/
structure Performance where
  laps : ℕ
  time : ℕ
  
/-- Calculates the lap time in seconds given a Performance --/
def lapTime (p : Performance) : ℚ :=
  (p.time * 60) / p.laps

/-- Martha's initial performance --/
def initialPerformance : Performance := ⟨15, 30⟩

/-- Martha's performance after two months --/
def finalPerformance : Performance := ⟨20, 27⟩

/-- Theorem stating the improvement in Martha's lap time --/
theorem martha_improvement :
  lapTime initialPerformance - lapTime finalPerformance = 39 := by
  sorry

end NUMINAMATH_CALUDE_martha_improvement_l1315_131504


namespace NUMINAMATH_CALUDE_hiking_rate_up_l1315_131583

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  rate_up : ℝ
  days_up : ℝ
  route_down_length : ℝ
  rate_down_multiplier : ℝ

/-- The hiking scenario satisfies the given conditions -/
def satisfies_conditions (h : HikingScenario) : Prop :=
  h.days_up = 2 ∧ 
  h.route_down_length = 15 ∧
  h.rate_down_multiplier = 1.5 ∧
  h.rate_up * h.days_up = h.route_down_length / h.rate_down_multiplier

/-- Theorem stating that the rate up the mountain is 5 miles per day -/
theorem hiking_rate_up (h : HikingScenario) 
  (hc : satisfies_conditions h) : h.rate_up = 5 := by
  sorry

#check hiking_rate_up

end NUMINAMATH_CALUDE_hiking_rate_up_l1315_131583


namespace NUMINAMATH_CALUDE_ratio_problem_l1315_131535

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1315_131535


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1315_131518

theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1315_131518


namespace NUMINAMATH_CALUDE_oplus_one_three_l1315_131558

def oplus (x y : ℤ) : ℤ := -3 * x + 4 * y

theorem oplus_one_three : oplus 1 3 = 9 := by sorry

end NUMINAMATH_CALUDE_oplus_one_three_l1315_131558


namespace NUMINAMATH_CALUDE_scientific_notation_exponent_l1315_131500

theorem scientific_notation_exponent (n : ℤ) : 12368000 = 1.2368 * (10 : ℝ) ^ n → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_exponent_l1315_131500


namespace NUMINAMATH_CALUDE_marcos_salary_calculation_l1315_131513

theorem marcos_salary_calculation (initial_salary : ℝ) : 
  initial_salary = 2500 →
  let salary_after_first_raise := initial_salary * 1.15
  let salary_after_second_raise := salary_after_first_raise * 1.10
  let final_salary := salary_after_second_raise * 0.85
  final_salary = 2688.125 := by
sorry

end NUMINAMATH_CALUDE_marcos_salary_calculation_l1315_131513


namespace NUMINAMATH_CALUDE_marathon_water_bottles_l1315_131522

theorem marathon_water_bottles (runners : ℕ) (bottles_per_runner : ℕ) (available_bottles : ℕ) : 
  runners = 14 → bottles_per_runner = 5 → available_bottles = 68 → 
  (runners * bottles_per_runner - available_bottles) = 2 := by
sorry

end NUMINAMATH_CALUDE_marathon_water_bottles_l1315_131522


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l1315_131571

theorem no_positive_integer_solution :
  ¬ ∃ (a b : ℕ+), 4 * (a^2 + a) = b^2 + b := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l1315_131571


namespace NUMINAMATH_CALUDE_hilt_trip_distance_l1315_131565

/-- Calculates the distance traveled given initial and final odometer readings -/
def distance_traveled (initial_reading final_reading : ℝ) : ℝ :=
  final_reading - initial_reading

theorem hilt_trip_distance :
  let initial_reading : ℝ := 212.3
  let final_reading : ℝ := 372
  distance_traveled initial_reading final_reading = 159.7 := by
  sorry

end NUMINAMATH_CALUDE_hilt_trip_distance_l1315_131565


namespace NUMINAMATH_CALUDE_distinct_shapes_count_is_31_l1315_131524

/-- Represents a convex-shaped paper made of four 1×1 squares -/
structure ConvexPaper :=
  (squares : Fin 4 → (Fin 1 × Fin 1))

/-- Represents a 5×6 grid paper -/
structure GridPaper :=
  (grid : Fin 5 → Fin 6 → Bool)

/-- Represents a placement of the convex paper on the grid paper -/
structure Placement :=
  (position : Fin 5 × Fin 6)
  (orientation : Fin 4)

/-- Checks if a placement is valid (all squares of convex paper overlap with grid squares) -/
def isValidPlacement (cp : ConvexPaper) (gp : GridPaper) (p : Placement) : Prop :=
  sorry

/-- Checks if two placements are rotationally equivalent -/
def areRotationallyEquivalent (p1 p2 : Placement) : Prop :=
  sorry

/-- The number of distinct shapes that can be formed -/
def distinctShapesCount (cp : ConvexPaper) (gp : GridPaper) : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct shapes is 31 -/
theorem distinct_shapes_count_is_31 (cp : ConvexPaper) (gp : GridPaper) :
  distinctShapesCount cp gp = 31 :=
  sorry

end NUMINAMATH_CALUDE_distinct_shapes_count_is_31_l1315_131524


namespace NUMINAMATH_CALUDE_ghost_castle_windows_l1315_131577

theorem ghost_castle_windows (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ghost_castle_windows_l1315_131577


namespace NUMINAMATH_CALUDE_cost_of_eight_books_l1315_131574

theorem cost_of_eight_books (cost_of_two : ℝ) (h : cost_of_two = 34) :
  8 * (cost_of_two / 2) = 136 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_eight_books_l1315_131574


namespace NUMINAMATH_CALUDE_desired_outcome_probability_l1315_131509

/-- Represents a die with a fixed number of sides --/
structure Die :=
  (sides : Nat)
  (values : Fin sides → Nat)

/-- Carla's die always shows 7 --/
def carla_die : Die :=
  { sides := 6,
    values := λ _ => 7 }

/-- Derek's die has numbers from 2 to 7 --/
def derek_die : Die :=
  { sides := 6,
    values := λ i => i.val + 2 }

/-- Emily's die has four faces showing 3 and two faces showing 8 --/
def emily_die : Die :=
  { sides := 6,
    values := λ i => if i.val < 4 then 3 else 8 }

/-- The probability of the desired outcome --/
def probability : Rat :=
  8 / 27

/-- Theorem stating the probability of the desired outcome --/
theorem desired_outcome_probability :
  (∀ (c : Fin carla_die.sides) (d : Fin derek_die.sides) (e : Fin emily_die.sides),
    (carla_die.values c > derek_die.values d ∧
     carla_die.values c > emily_die.values e ∧
     derek_die.values d + emily_die.values e < 10) →
    probability = 8 / 27) :=
by
  sorry

end NUMINAMATH_CALUDE_desired_outcome_probability_l1315_131509


namespace NUMINAMATH_CALUDE_area_increase_bound_l1315_131540

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  -- Add necessary fields here
  isConvex : Bool

/-- The perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- The area of a polygon -/
def area (p : ConvexPolygon) : ℝ := sorry

/-- The polygon resulting from moving all sides of p outward by distance h -/
def expandedPolygon (p : ConvexPolygon) (h : ℝ) : ConvexPolygon := sorry

theorem area_increase_bound (p : ConvexPolygon) (h : ℝ) (h_pos : h > 0) :
  area (expandedPolygon p h) - area p > perimeter p * h + π * h^2 := by
  sorry

end NUMINAMATH_CALUDE_area_increase_bound_l1315_131540


namespace NUMINAMATH_CALUDE_binomial_coefficient_16_11_l1315_131578

theorem binomial_coefficient_16_11 
  (h1 : Nat.choose 15 9 = 5005)
  (h2 : Nat.choose 15 10 = 3003)
  (h3 : Nat.choose 17 11 = 12376) :
  Nat.choose 16 11 = 4368 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_16_11_l1315_131578


namespace NUMINAMATH_CALUDE_wedge_volume_l1315_131555

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (angle : ℝ) : 
  d = 16 →
  angle = 60 →
  (π * (d / 2)^2 * d) / 2 = 512 * π :=
by sorry

end NUMINAMATH_CALUDE_wedge_volume_l1315_131555


namespace NUMINAMATH_CALUDE_arctan_sum_two_five_l1315_131552

theorem arctan_sum_two_five : Real.arctan (2/5) + Real.arctan (5/2) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_two_five_l1315_131552


namespace NUMINAMATH_CALUDE_total_subjects_l1315_131538

theorem total_subjects (avg_all : ℝ) (avg_first_five : ℝ) (last_subject : ℝ) 
  (h1 : avg_all = 78)
  (h2 : avg_first_five = 74)
  (h3 : last_subject = 98) :
  ∃ n : ℕ, n = 6 ∧ 
    n * avg_all = (n - 1) * avg_first_five + last_subject :=
by sorry

end NUMINAMATH_CALUDE_total_subjects_l1315_131538


namespace NUMINAMATH_CALUDE_baker_pastries_l1315_131563

theorem baker_pastries (cakes : ℕ) (pastry_difference : ℕ) : 
  cakes = 19 → pastry_difference = 112 → cakes + pastry_difference = 131 := by
  sorry

end NUMINAMATH_CALUDE_baker_pastries_l1315_131563


namespace NUMINAMATH_CALUDE_set_operations_l1315_131516

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem set_operations :
  (Set.univ \ (A ∩ B) = {x | x < 3 ∨ x ≥ 7}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 7) ∨ x ≥ 10}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1315_131516


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1315_131531

/-- If the quadratic equation x^2 - 3x + 2m = 0 has real roots, then m ≤ 9/8 -/
theorem quadratic_real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + 2*m = 0) → m ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1315_131531


namespace NUMINAMATH_CALUDE_simplify_expression_l1315_131593

theorem simplify_expression (x y : ℝ) : (x - y)^3 / (x - y)^2 * (y - x) = -(x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1315_131593


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l1315_131573

/-- The function g(x) as defined in the problem -/
def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

/-- Theorem stating that √21/7 is the greatest root of g(x) -/
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 21 / 7 ∧ g r = 0 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l1315_131573


namespace NUMINAMATH_CALUDE_navigation_time_is_21_days_l1315_131572

/-- Represents the timeline of a cargo shipment from Shanghai to Vancouver --/
structure CargoShipment where
  /-- Number of days for the ship to navigate from Shanghai to Vancouver --/
  navigationDays : ℕ
  /-- Number of days for customs and regulatory processes in Vancouver --/
  customsDays : ℕ
  /-- Number of days from port to warehouse --/
  portToWarehouseDays : ℕ
  /-- Number of days since the ship departed --/
  daysSinceDeparture : ℕ
  /-- Number of days until expected arrival at the warehouse --/
  daysUntilArrival : ℕ

/-- The theorem stating that the navigation time is 21 days --/
theorem navigation_time_is_21_days (shipment : CargoShipment)
  (h1 : shipment.customsDays = 4)
  (h2 : shipment.portToWarehouseDays = 7)
  (h3 : shipment.daysSinceDeparture = 30)
  (h4 : shipment.daysUntilArrival = 2)
  (h5 : shipment.navigationDays + shipment.customsDays + shipment.portToWarehouseDays =
        shipment.daysSinceDeparture + shipment.daysUntilArrival) :
  shipment.navigationDays = 21 := by
  sorry

end NUMINAMATH_CALUDE_navigation_time_is_21_days_l1315_131572


namespace NUMINAMATH_CALUDE_fraction_value_l1315_131586

theorem fraction_value (p q : ℚ) (x : ℚ) 
  (h1 : p / q = 4 / 5)
  (h2 : x + (2 * q - p) / (2 * q + p) = 4) :
  x = 25 / 7 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l1315_131586


namespace NUMINAMATH_CALUDE_sum_to_60_l1315_131529

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of integers from 1 to 60 is equal to 1830 -/
theorem sum_to_60 : sum_to_n 60 = 1830 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_60_l1315_131529


namespace NUMINAMATH_CALUDE_susan_board_game_l1315_131597

theorem susan_board_game (total_spaces : ℕ) (first_move : ℕ) (second_move : ℕ) (third_move : ℕ) (spaces_to_win : ℕ) :
  total_spaces = 48 →
  first_move = 8 →
  second_move = 2 →
  third_move = 6 →
  spaces_to_win = 37 →
  ∃ (spaces_moved_back : ℕ),
    first_move + second_move + third_move - spaces_moved_back = total_spaces - spaces_to_win ∧
    spaces_moved_back = 6 :=
by sorry

end NUMINAMATH_CALUDE_susan_board_game_l1315_131597


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l1315_131525

theorem roots_sum_of_squares (r s : ℝ) : 
  r^2 - 5*r + 3 = 0 → s^2 - 5*s + 3 = 0 → r^2 + s^2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l1315_131525


namespace NUMINAMATH_CALUDE_cs_consecutive_probability_l1315_131553

/-- The number of people sitting at the table -/
def total_people : ℕ := 12

/-- The number of computer scientists -/
def num_cs : ℕ := 5

/-- The number of chemistry majors -/
def num_chem : ℕ := 4

/-- The number of history majors -/
def num_hist : ℕ := 3

/-- The probability of all computer scientists sitting consecutively -/
def prob_cs_consecutive : ℚ := 1 / 66

theorem cs_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let consecutive_arrangements := Nat.factorial (num_cs) * Nat.factorial (total_people - num_cs)
  (consecutive_arrangements : ℚ) / total_arrangements = prob_cs_consecutive :=
sorry

end NUMINAMATH_CALUDE_cs_consecutive_probability_l1315_131553


namespace NUMINAMATH_CALUDE_coin_flip_expected_earnings_l1315_131510

/-- Represents the possible outcomes of the coin flip -/
inductive CoinOutcome
| A
| B
| C
| Disappear

/-- The probability of each outcome -/
def probability (outcome : CoinOutcome) : ℚ :=
  match outcome with
  | CoinOutcome.A => 1/4
  | CoinOutcome.B => 1/4
  | CoinOutcome.C => 1/3
  | CoinOutcome.Disappear => 1/6

/-- The payout for each outcome -/
def payout (outcome : CoinOutcome) : ℤ :=
  match outcome with
  | CoinOutcome.A => 2
  | CoinOutcome.B => -1
  | CoinOutcome.C => 4
  | CoinOutcome.Disappear => -3

/-- The expected earnings from flipping the coin -/
def expected_earnings : ℚ :=
  (probability CoinOutcome.A * payout CoinOutcome.A) +
  (probability CoinOutcome.B * payout CoinOutcome.B) +
  (probability CoinOutcome.C * payout CoinOutcome.C) +
  (probability CoinOutcome.Disappear * payout CoinOutcome.Disappear)

theorem coin_flip_expected_earnings :
  expected_earnings = 13/12 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_expected_earnings_l1315_131510


namespace NUMINAMATH_CALUDE_ellipse_sum_l1315_131556

theorem ellipse_sum (h k a b : ℝ) : 
  (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) →
  (h = 3 ∧ k = -5) →
  (a = 7 ∧ b = 4) →
  h + k + a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l1315_131556


namespace NUMINAMATH_CALUDE_parabolas_coincide_l1315_131598

/-- Represents a parabola with leading coefficient 1 -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  k : ℝ
  b : ℝ

/-- Returns the length of the segment intercepted by a line on a parabola -/
noncomputable def interceptLength (para : Parabola) (l : Line) : ℝ :=
  Real.sqrt ((para.p - l.k)^2 - 4*(para.q - l.b))

/-- Two lines are non-parallel if their slopes are different -/
def nonParallel (l₁ l₂ : Line) : Prop :=
  l₁.k ≠ l₂.k

theorem parabolas_coincide
  (Γ₁ Γ₂ : Parabola)
  (l₁ l₂ : Line)
  (h_nonparallel : nonParallel l₁ l₂)
  (h_equal_length₁ : interceptLength Γ₁ l₁ = interceptLength Γ₂ l₁)
  (h_equal_length₂ : interceptLength Γ₁ l₂ = interceptLength Γ₂ l₂) :
  Γ₁ = Γ₂ := by
  sorry

end NUMINAMATH_CALUDE_parabolas_coincide_l1315_131598


namespace NUMINAMATH_CALUDE_average_of_c_and_d_l1315_131568

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 8 + c + d) / 5 = 18 → (c + d) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_of_c_and_d_l1315_131568


namespace NUMINAMATH_CALUDE_arcsin_sufficient_not_necessary_l1315_131546

theorem arcsin_sufficient_not_necessary :
  (∃ α : ℝ, α = Real.arcsin (1/3) ∧ Real.sin α = 1/3) ∧
  (∃ β : ℝ, Real.sin β = 1/3 ∧ β ≠ Real.arcsin (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sufficient_not_necessary_l1315_131546


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1315_131519

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ

/-- Represents the number of triangles formed by connecting a point on a side to all vertices. -/
def triangles_formed (p : Polygon) : ℕ := p.sides - 1

/-- The polygon in our problem. -/
def our_polygon : Polygon :=
  { sides := 2024 }

/-- The theorem stating our problem. -/
theorem polygon_sides_count : triangles_formed our_polygon = 2023 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l1315_131519


namespace NUMINAMATH_CALUDE_friday_five_times_in_june_l1315_131551

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Represents a month with its dates -/
structure Month :=
  (dates : List Date)
  (numDays : Nat)

def May : Month := sorry
def June : Month := sorry

/-- Counts the occurrences of a specific day of the week in a month -/
def countDayInMonth (d : DayOfWeek) (m : Month) : Nat := sorry

/-- Checks if a month has exactly five occurrences of a specific day of the week -/
def hasFiveOccurrences (d : DayOfWeek) (m : Month) : Prop :=
  countDayInMonth d m = 5

theorem friday_five_times_in_june 
  (h1 : hasFiveOccurrences DayOfWeek.Tuesday May)
  (h2 : May.numDays = 31)
  (h3 : June.numDays = 31) :
  hasFiveOccurrences DayOfWeek.Friday June := by
  sorry

end NUMINAMATH_CALUDE_friday_five_times_in_june_l1315_131551


namespace NUMINAMATH_CALUDE_roots_magnitude_l1315_131542

theorem roots_magnitude (q : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + q*r₁ - 10 = 0 → 
  r₂^2 + q*r₂ - 10 = 0 → 
  (|r₁| > 4 ∨ |r₂| > 4) :=
by
  sorry

end NUMINAMATH_CALUDE_roots_magnitude_l1315_131542


namespace NUMINAMATH_CALUDE_grid_square_covers_at_least_four_l1315_131576

/-- A square on a grid -/
structure GridSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The area of the square is four times the unit area -/
  area_is_four : side^2 = 4

/-- The minimum number of grid points covered by a grid square -/
def min_covered_points (s : GridSquare) : ℕ := 4

/-- Theorem: A GridSquare covers at least 4 grid points -/
theorem grid_square_covers_at_least_four (s : GridSquare) :
  ∃ (n : ℕ), n ≥ 4 ∧ n = min_covered_points s :=
sorry

end NUMINAMATH_CALUDE_grid_square_covers_at_least_four_l1315_131576


namespace NUMINAMATH_CALUDE_min_value_f_over_x_range_of_a_l1315_131581

-- Part 1
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

theorem min_value_f_over_x (x : ℝ) (hx : x > 0) :
  ∃ (y : ℝ), y = (f 2 x) / x ∧ ∀ (z : ℝ), z > 0 → (f 2 z) / z ≥ y ∧ y = -2 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x ≤ a) ↔ a ∈ Set.Ici (3/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_f_over_x_range_of_a_l1315_131581


namespace NUMINAMATH_CALUDE_additional_volunteers_needed_l1315_131561

def volunteers_needed : ℕ := 50
def math_classes : ℕ := 6
def students_per_class : ℕ := 5
def teachers_volunteered : ℕ := 13

theorem additional_volunteers_needed :
  volunteers_needed - (math_classes * students_per_class + teachers_volunteered) = 7 := by
  sorry

end NUMINAMATH_CALUDE_additional_volunteers_needed_l1315_131561


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1315_131508

theorem solution_set_inequality (x : ℝ) :
  x * (2 - x) ≤ 0 ↔ x ≤ 0 ∨ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1315_131508


namespace NUMINAMATH_CALUDE_b_contribution_l1315_131521

def a_investment : ℕ := 3500
def a_months : ℕ := 12
def b_months : ℕ := 7
def a_share : ℕ := 2
def b_share : ℕ := 3

theorem b_contribution (x : ℕ) : 
  (a_investment * a_months) / (x * b_months) = a_share / b_share → 
  x = 9000 := by
  sorry

end NUMINAMATH_CALUDE_b_contribution_l1315_131521


namespace NUMINAMATH_CALUDE_B_wins_4_probability_C_wins_3_probability_l1315_131559

-- Define the players
inductive Player : Type
| A : Player
| B : Player
| C : Player

-- Define the win probabilities
def winProb (winner loser : Player) : ℝ :=
  match winner, loser with
  | Player.A, Player.B => 0.4
  | Player.B, Player.C => 0.5
  | Player.C, Player.A => 0.6
  | _, _ => 0 -- For other combinations, set probability to 0

-- Define the probability of B winning 4 consecutive matches
def prob_B_wins_4 : ℝ :=
  (1 - winProb Player.A Player.B) * 
  (winProb Player.B Player.C) * 
  (1 - winProb Player.A Player.B) * 
  (winProb Player.B Player.C)

-- Define the probability of C winning 3 consecutive matches
def prob_C_wins_3 : ℝ :=
  ((1 - winProb Player.A Player.B) * (1 - winProb Player.B Player.C) * 
   (winProb Player.C Player.A) * (1 - winProb Player.B Player.C)) +
  ((winProb Player.A Player.B) * (winProb Player.C Player.A) * 
   (1 - winProb Player.B Player.C) * (winProb Player.C Player.A))

-- Theorem statements
theorem B_wins_4_probability : prob_B_wins_4 = 0.09 := by sorry

theorem C_wins_3_probability : prob_C_wins_3 = 0.162 := by sorry

end NUMINAMATH_CALUDE_B_wins_4_probability_C_wins_3_probability_l1315_131559


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l1315_131507

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 3.4)
  (h2 : (c + d) / 2 = 3.85)
  (h3 : (e + f) / 2 = 4.45) :
  (a + b + c + d + e + f) / 6 = 3.9 := by
  sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l1315_131507


namespace NUMINAMATH_CALUDE_bird_migration_distance_l1315_131541

/-- Calculates the total distance traveled by migrating birds -/
theorem bird_migration_distance 
  (num_birds : ℕ) 
  (distance_jim_disney : ℝ) 
  (distance_disney_london : ℝ) : 
  num_birds = 20 → 
  distance_jim_disney = 50 → 
  distance_disney_london = 60 → 
  (num_birds : ℝ) * (distance_jim_disney + distance_disney_london) = 2200 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_distance_l1315_131541


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l1315_131590

theorem gcd_from_lcm_and_ratio (C D : ℕ+) 
  (h_lcm : Nat.lcm C D = 250)
  (h_ratio : C * 5 = D * 2) :
  Nat.gcd C D = 5 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l1315_131590


namespace NUMINAMATH_CALUDE_min_value_T_l1315_131595

theorem min_value_T (p : ℝ) (h1 : 0 < p) (h2 : p < 15) :
  ∃ (min_T : ℝ), min_T = 15 ∧
  ∀ x : ℝ, p ≤ x → x ≤ 15 →
    |x - p| + |x - 15| + |x - (15 + p)| ≥ min_T :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_T_l1315_131595


namespace NUMINAMATH_CALUDE_bobs_head_start_l1315_131530

/-- Proves that Bob's head-start is 1 mile given the conditions -/
theorem bobs_head_start (bob_speed jim_speed : ℝ) (catch_time : ℝ) (head_start : ℝ) : 
  bob_speed = 6 → 
  jim_speed = 9 → 
  catch_time = 20 / 60 →
  head_start + bob_speed * catch_time = jim_speed * catch_time →
  head_start = 1 := by
sorry

end NUMINAMATH_CALUDE_bobs_head_start_l1315_131530


namespace NUMINAMATH_CALUDE_quadratic_equation_pairs_l1315_131527

theorem quadratic_equation_pairs : 
  let count := Finset.filter (fun p : ℕ × ℕ => 
    let b := p.1
    let c := p.2
    b > 0 ∧ c > 0 ∧ 
    b + c ≤ 10 ∧
    b^2 - 4*c = 0 ∧
    c^2 - 4*b ≤ 0) (Finset.product (Finset.range 11) (Finset.range 11))
  count.card = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_pairs_l1315_131527


namespace NUMINAMATH_CALUDE_unique_solution_exists_l1315_131547

theorem unique_solution_exists (m n : ℕ) : 
  (∃! (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + m * b = n ∧ a + b = m * c) ↔ 
  (m > 1 ∧ (n - 1) % (m - 1) = 0 ∧ ¬∃k, n = m ^ k) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l1315_131547


namespace NUMINAMATH_CALUDE_bella_stamps_l1315_131523

theorem bella_stamps (snowflake : ℕ) (truck : ℕ) (rose : ℕ) (butterfly : ℕ) 
  (h1 : snowflake = 15)
  (h2 : truck = snowflake + 11)
  (h3 : rose = truck - 17)
  (h4 : butterfly = 2 * rose) :
  snowflake + truck + rose + butterfly = 68 := by
  sorry

end NUMINAMATH_CALUDE_bella_stamps_l1315_131523


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1315_131592

/-- Definition of a one-variable quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 2 * (x - x^2) - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1315_131592


namespace NUMINAMATH_CALUDE_m_range_l1315_131562

def p (m : ℝ) : Prop := ∀ x, |x - m| + |x - 1| > 1

def q (m : ℝ) : Prop := ∀ x > 0, (fun x => Real.log x / Real.log (3 + m)) x > 0

theorem m_range : 
  (∃ m : ℝ, (¬(p m ∧ q m)) ∧ (p m ∨ q m)) ↔ 
  (∃ m : ℝ, (-3 < m ∧ m < -2) ∨ (0 ≤ m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l1315_131562


namespace NUMINAMATH_CALUDE_inequality_proof_l1315_131564

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hab : a ≤ b) (hbc : b ≤ c) : 
  (a*x + b*y + c*z) * (x/a + y/b + z/c) ≤ (x+y+z)^2 * (a+c)^2 / (4*a*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1315_131564


namespace NUMINAMATH_CALUDE_test_points_calculation_l1315_131554

theorem test_points_calculation (total_problems : ℕ) (computation_problems : ℕ) 
  (computation_points : ℕ) (word_points : ℕ) :
  total_problems = 30 →
  computation_problems = 20 →
  computation_points = 3 →
  word_points = 5 →
  (computation_problems * computation_points) + 
  ((total_problems - computation_problems) * word_points) = 110 := by
sorry

end NUMINAMATH_CALUDE_test_points_calculation_l1315_131554


namespace NUMINAMATH_CALUDE_divisibility_of_f_l1315_131520

def f (x : ℕ) : ℕ := x^3 + 17

theorem divisibility_of_f (n : ℕ) (hn : n ≥ 2) :
  ∃ x : ℕ, (3^n ∣ f x) ∧ ¬(3^(n+1) ∣ f x) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_f_l1315_131520


namespace NUMINAMATH_CALUDE_square_field_area_proof_l1315_131588

/-- The time taken to cross the square field diagonally in hours -/
def crossing_time : ℝ := 6.0008333333333335

/-- The speed of the person crossing the field in km/hr -/
def crossing_speed : ℝ := 1.2

/-- The area of the square field in square meters -/
def field_area : ℝ := 25939744.8

/-- Theorem stating that the area of the square field is approximately 25939744.8 square meters -/
theorem square_field_area_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |field_area - (crossing_speed * 1000 * crossing_time)^2 / 2| < ε :=
sorry

end NUMINAMATH_CALUDE_square_field_area_proof_l1315_131588


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1315_131566

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + 3*a - 4 = 0) → (b^2 + 3*b - 4 = 0) → (a^2 + 4*a + b - 3 = -2) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1315_131566


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1315_131550

def first_ten_integers : Finset ℕ := Finset.range 10

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ i ∈ first_ten_integers, i ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ i ∈ first_ten_integers, i ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1315_131550


namespace NUMINAMATH_CALUDE_solve_for_a_l1315_131596

theorem solve_for_a (x a : ℝ) (h : 2 * x - a = -5) (hx : x = 5) : a = 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1315_131596


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1315_131570

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 1} = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1315_131570


namespace NUMINAMATH_CALUDE_percentage_ratio_proof_l1315_131548

theorem percentage_ratio_proof (P Q M N R : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.3 * P)
  (hN : N = 0.6 * P)
  (hR : R = 0.2 * P)
  (hP : P ≠ 0) : (M + R) / N = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_ratio_proof_l1315_131548


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l1315_131536

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculates the total cost of fencing for a rectangular plot. -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot. -/
theorem fencing_cost_calculation (plot : RectangularPlot)
  (h1 : plot.length = 61)
  (h2 : plot.breadth = plot.length - 22)
  (h3 : plot.fencing_cost_per_meter = 26.50) :
  total_fencing_cost plot = 5300 := by
  sorry

#eval total_fencing_cost { length := 61, breadth := 39, fencing_cost_per_meter := 26.50 }

end NUMINAMATH_CALUDE_fencing_cost_calculation_l1315_131536


namespace NUMINAMATH_CALUDE_julias_change_julias_change_is_eight_l1315_131537

/-- Calculates Julia's change after purchasing Snickers and M&M's -/
theorem julias_change (snickers_price : ℝ) (snickers_quantity : ℕ) (mms_quantity : ℕ) 
  (payment : ℝ) : ℝ :=
  let mms_price := 2 * snickers_price
  let total_cost := snickers_price * snickers_quantity + mms_price * mms_quantity
  payment - total_cost

/-- Proves that Julia's change is $8 given the specific conditions -/
theorem julias_change_is_eight :
  julias_change 1.5 2 3 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_julias_change_julias_change_is_eight_l1315_131537


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_one_fifth_l1315_131545

theorem sqrt_expression_equals_one_fifth :
  (Real.sqrt 3 + Real.sqrt 2) ^ (2 * (Real.log (Real.sqrt 5) / Real.log (Real.sqrt 3 - Real.sqrt 2))) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_one_fifth_l1315_131545


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1315_131587

theorem unique_solution_for_equation :
  ∃! (n k : ℕ), n > 0 ∧ k > 0 ∧ (n + 1)^n = 2 * n^k + 3 * n + 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1315_131587


namespace NUMINAMATH_CALUDE_union_covers_reals_l1315_131579

open Set Real

theorem union_covers_reals (a : ℝ) : 
  let S : Set ℝ := {x | |x - 2| > 3}
  let T : Set ℝ := {x | a < x ∧ x < a + 8}
  (S ∪ T = univ) → (-3 < a ∧ a < -1) :=
by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l1315_131579


namespace NUMINAMATH_CALUDE_triangle_inequality_l1315_131589

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (a ≥ b ∧ b ≥ c → A ≥ B ∧ B ≥ C) →
  (a * A + b * B + c * C) / (a + b + c) ≥ π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1315_131589


namespace NUMINAMATH_CALUDE_only_j_has_inverse_l1315_131515

-- Define the types for our functions
def Function : Type := ℝ → ℝ

-- Define properties for each function
def is_parabola_upward (f : Function) : Prop := sorry

def is_discontinuous_two_segments (f : Function) : Prop := sorry

def is_horizontal_line (f : Function) : Prop := sorry

def is_sine_function (f : Function) : Prop := sorry

def is_linear_positive_slope (f : Function) : Prop := sorry

-- Define what it means for a function to have an inverse
def has_inverse (f : Function) : Prop := sorry

-- State the theorem
theorem only_j_has_inverse 
  (F G H I J : Function)
  (hF : is_parabola_upward F)
  (hG : is_discontinuous_two_segments G)
  (hH : is_horizontal_line H)
  (hI : is_sine_function I)
  (hJ : is_linear_positive_slope J) :
  (¬ has_inverse F) ∧ 
  (¬ has_inverse G) ∧ 
  (¬ has_inverse H) ∧ 
  (¬ has_inverse I) ∧ 
  has_inverse J :=
sorry

end NUMINAMATH_CALUDE_only_j_has_inverse_l1315_131515


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l1315_131503

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x - 6 = 0
def equation2 (x : ℝ) : Prop := x/(x-1) - 1 = 3/(x^2-1)

-- Theorem for the first equation
theorem equation1_solutions :
  ∃ x1 x2 : ℝ, 
    (x1 = 2 + Real.sqrt 10 ∧ equation1 x1) ∧
    (x2 = 2 - Real.sqrt 10 ∧ equation1 x2) :=
sorry

-- Theorem for the second equation
theorem equation2_solution :
  ∃ x : ℝ, x = 2 ∧ equation2 x :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l1315_131503


namespace NUMINAMATH_CALUDE_inequality_proof_l1315_131560

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (a + 2*b + c)^2 / (2*b^2 + (c + a)^2) +
  (a + b + 2*c)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1315_131560


namespace NUMINAMATH_CALUDE_zack_group_size_l1315_131526

/-- Proves that Zack tutors students in groups of 10, given the problem conditions -/
theorem zack_group_size :
  ∀ (x : ℕ),
  (∃ (n : ℕ), x * n = 70) →  -- Zack tutors 70 students in total
  (∃ (m : ℕ), 10 * m = 70) →  -- Karen tutors 70 students in total
  x = 10 := by sorry

end NUMINAMATH_CALUDE_zack_group_size_l1315_131526


namespace NUMINAMATH_CALUDE_largest_c_value_l1315_131514

theorem largest_c_value (c : ℝ) : (3 * c + 4) * (c - 2) = 9 * c → c ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_c_value_l1315_131514


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l1315_131502

theorem reciprocal_of_negative_one_third :
  ∀ x : ℚ, x * (-1/3) = 1 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l1315_131502


namespace NUMINAMATH_CALUDE_sweater_shirt_price_difference_l1315_131599

theorem sweater_shirt_price_difference : 
  let shirt_total : ℕ := 360
  let shirt_count : ℕ := 20
  let sweater_total : ℕ := 900
  let sweater_count : ℕ := 45
  let shirt_avg : ℚ := shirt_total / shirt_count
  let sweater_avg : ℚ := sweater_total / sweater_count
  sweater_avg - shirt_avg = 2 := by
sorry

end NUMINAMATH_CALUDE_sweater_shirt_price_difference_l1315_131599


namespace NUMINAMATH_CALUDE_train_length_l1315_131575

/-- The length of a train passing a bridge -/
theorem train_length (v : ℝ) (t : ℝ) (b : ℝ) : v = 72 * 1000 / 3600 → t = 25 → b = 140 → v * t - b = 360 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1315_131575


namespace NUMINAMATH_CALUDE_only_345_right_triangle_l1315_131533

/-- A function that checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that only (3, 4, 5) forms a right-angled triangle among the given sets --/
theorem only_345_right_triangle :
  ¬ is_right_triangle 2 4 4 ∧
  ¬ is_right_triangle (Real.sqrt 3) 2 2 ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 5 12 14 :=
by sorry

end NUMINAMATH_CALUDE_only_345_right_triangle_l1315_131533


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_zero_l1315_131501

/-- Given a quadratic equation (k-1)x^2 + x - k^2 = 0 with a root x = 1, prove that k = 0 -/
theorem quadratic_root_implies_k_zero (k : ℝ) : 
  ((k - 1) * 1^2 + 1 - k^2 = 0) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_zero_l1315_131501


namespace NUMINAMATH_CALUDE_largest_number_comparison_l1315_131582

theorem largest_number_comparison :
  (1/2 : ℝ) > (37.5/100 : ℝ) ∧ (1/2 : ℝ) > (7/22 : ℝ) ∧ (1/2 : ℝ) > (π/10 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_comparison_l1315_131582


namespace NUMINAMATH_CALUDE_intersection_M_N_l1315_131505

def M : Set ℤ := {-2, -1, 0, 1}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1315_131505
