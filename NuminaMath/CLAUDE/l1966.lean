import Mathlib

namespace NUMINAMATH_CALUDE_reorganize_books_leftover_l1966_196694

/-- The number of books left over when reorganizing boxes -/
def books_left_over (initial_boxes : ℕ) (books_per_initial_box : ℕ) (books_per_new_box : ℕ) : ℕ :=
  (initial_boxes * books_per_initial_box) % books_per_new_box

/-- Theorem stating the number of books left over in the specific scenario -/
theorem reorganize_books_leftover :
  books_left_over 2020 42 45 = 30 := by
  sorry

end NUMINAMATH_CALUDE_reorganize_books_leftover_l1966_196694


namespace NUMINAMATH_CALUDE_f_upper_bound_l1966_196662

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin x else -x^2 - 1

theorem f_upper_bound (k : ℝ) :
  (∀ x, f x ≤ k * x) ↔ 1 ≤ k ∧ k ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_f_upper_bound_l1966_196662


namespace NUMINAMATH_CALUDE_travel_ways_theorem_l1966_196617

/-- The number of ways to travel between two cities using exactly k buses -/
def travel_ways (n k : ℕ) : ℚ :=
  ((n - 1)^k - (-1)^k) / n

/-- Theorem stating the number of ways to travel between two cities using exactly k buses -/
theorem travel_ways_theorem (n k : ℕ) (hn : n ≥ 2) (hk : k ≥ 1) :
  travel_ways n k = ((n - 1)^k - (-1)^k) / n :=
by
  sorry

#check travel_ways_theorem

end NUMINAMATH_CALUDE_travel_ways_theorem_l1966_196617


namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_t_bound_l1966_196643

-- Define the function f(x)
def f (t : ℝ) (x : ℝ) : ℝ := x^3 - t*x^2 + 3*x

-- Define the derivative of f(x)
def f_derivative (t : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*t*x + 3

-- State the theorem
theorem monotonic_decreasing_implies_t_bound :
  ∀ t : ℝ, (∀ x ∈ Set.Icc 1 4, f_derivative t x ≤ 0) →
  t ≥ 51/8 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_t_bound_l1966_196643


namespace NUMINAMATH_CALUDE_four_team_tournament_handshakes_l1966_196626

/-- The number of handshakes in a tournament with teams of two -/
def tournament_handshakes (num_teams : ℕ) : ℕ :=
  let total_people := num_teams * 2
  let handshakes_per_person := total_people - 2
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a tournament with 4 teams of 2 people each, where each person
    shakes hands with everyone except their partner and themselves,
    the total number of handshakes is 24. -/
theorem four_team_tournament_handshakes :
  tournament_handshakes 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_team_tournament_handshakes_l1966_196626


namespace NUMINAMATH_CALUDE_gcd_32_24_l1966_196639

theorem gcd_32_24 : Nat.gcd 32 24 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_32_24_l1966_196639


namespace NUMINAMATH_CALUDE_sum_of_roots_l1966_196613

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a = 1) 
  (hb : b^3 - 3*b^2 + 5*b = 5) : 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1966_196613


namespace NUMINAMATH_CALUDE_people_on_stairs_l1966_196615

-- Define the number of people and steps
def num_people : ℕ := 3
def num_steps : ℕ := 7

-- Define a function to calculate the number of arrangements
def arrange_people (people : ℕ) (steps : ℕ) : ℕ :=
  -- Number of ways with each person on a different step
  (steps.choose people) * (people.factorial) +
  -- Number of ways with one step having 2 people and another having 1 person
  (people.choose 2) * (steps.choose 2) * 2

-- State the theorem
theorem people_on_stairs :
  arrange_people num_people num_steps = 336 := by
  sorry

end NUMINAMATH_CALUDE_people_on_stairs_l1966_196615


namespace NUMINAMATH_CALUDE_earbuds_cost_after_tax_l1966_196676

/-- The total amount paid after tax for an item with a given cost and tax rate. -/
def total_amount_after_tax (cost : ℝ) (tax_rate : ℝ) : ℝ :=
  cost * (1 + tax_rate)

/-- Theorem stating that the total amount paid after tax for an item costing $200 with a 15% tax rate is $230. -/
theorem earbuds_cost_after_tax :
  total_amount_after_tax 200 0.15 = 230 := by
  sorry

end NUMINAMATH_CALUDE_earbuds_cost_after_tax_l1966_196676


namespace NUMINAMATH_CALUDE_trig_identity_l1966_196625

theorem trig_identity : 
  (2 * Real.sin (10 * π / 180) - Real.cos (20 * π / 180)) / Real.cos (70 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1966_196625


namespace NUMINAMATH_CALUDE_munchausen_polygon_theorem_l1966_196600

/-- A polygon in a 2D plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.length ≥ 3

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- A line in a 2D plane -/
structure Line where
  point1 : Point
  point2 : Point

/-- Checks if a point is inside a polygon -/
def is_inside (p : Point) (poly : Polygon) : Prop := sorry

/-- Counts the number of regions a line divides a polygon into -/
def count_regions (l : Line) (poly : Polygon) : ℕ := sorry

/-- Theorem: There exists a polygon and a point inside it such that 
    any line passing through this point divides the polygon into 
    exactly three smaller polygons -/
theorem munchausen_polygon_theorem : 
  ∃ (poly : Polygon) (p : Point), 
    is_inside p poly ∧ 
    ∀ (l : Line), l.point1 = p ∨ l.point2 = p → count_regions l poly = 3 := by
  sorry

end NUMINAMATH_CALUDE_munchausen_polygon_theorem_l1966_196600


namespace NUMINAMATH_CALUDE_regular_discount_is_30_percent_l1966_196635

/-- The regular discount range for pet food at a store -/
def regular_discount_range : ℝ := sorry

/-- The additional sale discount percentage -/
def additional_sale_discount : ℝ := 0.20

/-- The manufacturer's suggested retail price (MSRP) for a container of pet food -/
def msrp : ℝ := 35.00

/-- The lowest possible price after the additional sale discount -/
def lowest_sale_price : ℝ := 19.60

/-- Theorem stating that the regular discount range is 30% -/
theorem regular_discount_is_30_percent :
  regular_discount_range = 0.30 := by sorry

end NUMINAMATH_CALUDE_regular_discount_is_30_percent_l1966_196635


namespace NUMINAMATH_CALUDE_quadrilateral_qt_length_l1966_196637

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the conditions
def is_convex (quad : Quadrilateral) : Prop := sorry

def is_perpendicular (A B C D : ℝ × ℝ) : Prop := sorry

def distance (A B : ℝ × ℝ) : ℝ := sorry

def line_through (A B : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

def intersect_point (l₁ l₂ : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- State the theorem
theorem quadrilateral_qt_length 
  (quad : Quadrilateral)
  (h_convex : is_convex quad)
  (h_perp_rs_pq : is_perpendicular quad.R quad.S quad.P quad.Q)
  (h_perp_pq_rs : is_perpendicular quad.P quad.Q quad.R quad.S)
  (h_rs_length : distance quad.R quad.S = 39)
  (h_pq_length : distance quad.P quad.Q = 52)
  (h_t : ∃ T : ℝ × ℝ, T ∈ line_through quad.Q (intersect_point (line_through quad.P quad.S) (line_through quad.Q quad.Q)) ∧
                       T = intersect_point (line_through quad.P quad.Q) (line_through quad.Q quad.Q))
  (h_pt_length : ∀ T : ℝ × ℝ, T ∈ line_through quad.P quad.Q → distance quad.P T = 13) :
  ∃ T : ℝ × ℝ, T ∈ line_through quad.P quad.Q ∧ distance quad.Q T = 195 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_qt_length_l1966_196637


namespace NUMINAMATH_CALUDE_periodic_function_value_l1966_196660

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2 * Real.pi) = f x

theorem periodic_function_value 
  (f : ℝ → ℝ) 
  (h1 : periodic_function f) 
  (h2 : f 0 = 0) : 
  f (4 * Real.pi) = 0 := by
sorry

end NUMINAMATH_CALUDE_periodic_function_value_l1966_196660


namespace NUMINAMATH_CALUDE_stratified_sampling_results_l1966_196636

theorem stratified_sampling_results (total_sample : ℕ) (junior_pop : ℕ) (senior_pop : ℕ)
  (h1 : total_sample = 60)
  (h2 : junior_pop = 400)
  (h3 : senior_pop = 200) :
  (Nat.choose junior_pop ((junior_pop * total_sample) / (junior_pop + senior_pop))) *
  (Nat.choose senior_pop ((senior_pop * total_sample) / (junior_pop + senior_pop))) =
  (Nat.choose 400 40) * (Nat.choose 200 20) :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_results_l1966_196636


namespace NUMINAMATH_CALUDE_equation_solution_l1966_196606

theorem equation_solution : ∃! x : ℝ, 5 * (3 * x + 2) - 2 = -2 * (1 - 7 * x) ∧ x = -10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1966_196606


namespace NUMINAMATH_CALUDE_village_foods_tomato_sales_l1966_196603

theorem village_foods_tomato_sales (customers : ℕ) (lettuce_per_customer : ℕ) 
  (lettuce_price : ℚ) (tomato_price : ℚ) (total_sales : ℚ) 
  (h1 : customers = 500)
  (h2 : lettuce_per_customer = 2)
  (h3 : lettuce_price = 1)
  (h4 : tomato_price = 1/2)
  (h5 : total_sales = 2000) :
  (total_sales - (↑customers * ↑lettuce_per_customer * lettuce_price)) / (↑customers * tomato_price) = 4 := by
sorry

end NUMINAMATH_CALUDE_village_foods_tomato_sales_l1966_196603


namespace NUMINAMATH_CALUDE_angle_measure_when_complement_and_supplement_are_complementary_l1966_196601

theorem angle_measure_when_complement_and_supplement_are_complementary :
  ∀ x : ℝ,
  (90 - x) + (180 - x) = 90 →
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_when_complement_and_supplement_are_complementary_l1966_196601


namespace NUMINAMATH_CALUDE_farmer_plan_proof_l1966_196671

/-- The number of cows a farmer plans to add to their farm -/
def planned_cows : ℕ := 3

/-- The initial number of animals on the farm -/
def initial_animals : ℕ := 11

/-- The number of pigs and goats the farmer plans to add -/
def planned_pigs_and_goats : ℕ := 7

/-- The total number of animals after all additions -/
def total_animals : ℕ := 21

theorem farmer_plan_proof :
  initial_animals + planned_pigs_and_goats + planned_cows = total_animals :=
by sorry

end NUMINAMATH_CALUDE_farmer_plan_proof_l1966_196671


namespace NUMINAMATH_CALUDE_four_is_17th_term_terms_before_4_l1966_196681

/-- An arithmetic sequence with first term 100 and common difference -6 -/
def arithmeticSequence (n : ℕ) : ℤ := 100 - 6 * (n - 1)

/-- The position of 4 in the sequence -/
def positionOf4 : ℕ := 17

theorem four_is_17th_term :
  arithmeticSequence positionOf4 = 4 ∧ 
  ∀ k : ℕ, k < positionOf4 → arithmeticSequence k > 4 :=
sorry

theorem terms_before_4 : 
  positionOf4 - 1 = 16 :=
sorry

end NUMINAMATH_CALUDE_four_is_17th_term_terms_before_4_l1966_196681


namespace NUMINAMATH_CALUDE_panda_pregnancy_percentage_l1966_196612

/-- The number of pandas in the zoo -/
def total_pandas : ℕ := 16

/-- The number of panda babies born -/
def babies_born : ℕ := 2

/-- The number of panda couples in the zoo -/
def total_couples : ℕ := total_pandas / 2

/-- The number of couples that got pregnant -/
def pregnant_couples : ℕ := babies_born

/-- The percentage of panda couples that get pregnant after mating -/
def pregnancy_percentage : ℚ := pregnant_couples / total_couples * 100

theorem panda_pregnancy_percentage :
  pregnancy_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_panda_pregnancy_percentage_l1966_196612


namespace NUMINAMATH_CALUDE_positive_sum_of_odd_monotone_increasing_l1966_196667

def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem positive_sum_of_odd_monotone_increasing (f : ℝ → ℝ) (a : ℕ → ℝ) :
  is_monotone_increasing f →
  is_odd_function f →
  is_arithmetic_sequence a →
  a 3 > 0 →
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_positive_sum_of_odd_monotone_increasing_l1966_196667


namespace NUMINAMATH_CALUDE_binary_digits_difference_l1966_196661

theorem binary_digits_difference : ∃ n m : ℕ, 
  (2^n ≤ 300 ∧ 300 < 2^(n+1)) ∧ 
  (2^m ≤ 1400 ∧ 1400 < 2^(m+1)) ∧ 
  m - n = 2 := by
sorry

end NUMINAMATH_CALUDE_binary_digits_difference_l1966_196661


namespace NUMINAMATH_CALUDE_sin_theta_for_point_neg_two_three_l1966_196655

theorem sin_theta_for_point_neg_two_three (θ : Real) :
  (∃ (t : Real), t > 0 ∧ t * Real.cos θ = -2 ∧ t * Real.sin θ = 3) →
  Real.sin θ = 3 / Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_sin_theta_for_point_neg_two_three_l1966_196655


namespace NUMINAMATH_CALUDE_sales_increase_percentage_l1966_196665

theorem sales_increase_percentage (original_price : ℝ) (original_quantity : ℝ) 
  (discount_rate : ℝ) (income_increase_rate : ℝ) (new_quantity : ℝ)
  (h1 : discount_rate = 0.1)
  (h2 : income_increase_rate = 0.125)
  (h3 : original_price * original_quantity * (1 + income_increase_rate) = 
        original_price * (1 - discount_rate) * new_quantity) :
  new_quantity / original_quantity - 1 = 0.25 := by
sorry

end NUMINAMATH_CALUDE_sales_increase_percentage_l1966_196665


namespace NUMINAMATH_CALUDE_john_chips_consumption_l1966_196652

/-- The number of bags of chips John eats for dinner -/
def dinner_chips : ℕ := 1

/-- The number of bags of chips John eats after dinner -/
def after_dinner_chips : ℕ := 2 * dinner_chips

/-- The total number of bags of chips John eats -/
def total_chips : ℕ := dinner_chips + after_dinner_chips

theorem john_chips_consumption :
  total_chips = 3 :=
sorry

end NUMINAMATH_CALUDE_john_chips_consumption_l1966_196652


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l1966_196664

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 35 →
  n2 = 45 →
  avg1 = 40 →
  avg2 = 60 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = ((n1 + n2) : ℚ) * (51.25 : ℚ) :=
by
  sorry

#eval (35 * 40 + 45 * 60) / (35 + 45) -- Should evaluate to 51.25

end NUMINAMATH_CALUDE_average_marks_combined_classes_l1966_196664


namespace NUMINAMATH_CALUDE_carls_paintable_area_l1966_196634

/-- Calculates the total paintable area in square feet for a given number of bedrooms --/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area_per_room := wall_area - unpaintable_area
  num_bedrooms * paintable_area_per_room

/-- The total paintable area for Carl's bedrooms is 1552 square feet --/
theorem carls_paintable_area :
  total_paintable_area 4 15 11 9 80 = 1552 := by
sorry

end NUMINAMATH_CALUDE_carls_paintable_area_l1966_196634


namespace NUMINAMATH_CALUDE_volume_of_112_ounces_l1966_196622

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℚ
  /-- Assumption: k is positive -/
  k_pos : k > 0

/-- Volume of the substance given its weight -/
def volume (s : Substance) (weight : ℚ) : ℚ :=
  s.k * weight

theorem volume_of_112_ounces (s : Substance) 
  (h : volume s 63 = 27) : volume s 112 = 48 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_112_ounces_l1966_196622


namespace NUMINAMATH_CALUDE_smallest_n_with_hex_digit_greater_than_9_l1966_196646

/-- Sum of digits in base b representation of n -/
def sumDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- f(n) is the sum of digits in base-five representation of n -/
def f (n : ℕ) : ℕ := sumDigits n 5

/-- g(n) is the sum of digits in base-nine representation of f(n) -/
def g (n : ℕ) : ℕ := sumDigits (f n) 9

/-- Converts a natural number to its base-sixteen representation -/
def toBase16 (n : ℕ) : List ℕ := sorry

/-- Checks if a list of digits contains only elements from 0 to 9 -/
def onlyDecimalDigits (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d ≤ 9

theorem smallest_n_with_hex_digit_greater_than_9 :
  (∀ m < 621, onlyDecimalDigits (toBase16 (g m))) ∧
  ¬onlyDecimalDigits (toBase16 (g 621)) := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_hex_digit_greater_than_9_l1966_196646


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1966_196638

theorem arithmetic_calculation : (7.356 - 1.092) + 3.5 = 9.764 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1966_196638


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_l1966_196666

/-- Theorem: Percentage of ryegrass in a mixture of seed mixtures X and Y -/
theorem ryegrass_percentage_in_mixture (x_ryegrass : ℝ) (y_ryegrass : ℝ) (x_weight : ℝ) :
  x_ryegrass = 0.40 →
  y_ryegrass = 0.25 →
  x_weight = 0.8667 →
  x_ryegrass * x_weight + y_ryegrass * (1 - x_weight) = 0.380005 :=
by sorry

end NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_l1966_196666


namespace NUMINAMATH_CALUDE_max_sum_is_33_l1966_196607

def numbers : List ℕ := [2, 5, 8, 11, 14]

structure LShape :=
  (a b c d e : ℕ)
  (in_numbers : {a, b, c, d, e} ⊆ numbers.toFinset)
  (horizontal_eq_vertical : a + b + e = a + c + e)

def sum (l : LShape) : ℕ := l.a + l.b + l.e

theorem max_sum_is_33 :
  ∃ (l : LShape), sum l = 33 ∧ ∀ (l' : LShape), sum l' ≤ 33 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_33_l1966_196607


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1966_196618

open Set

def R : Set ℝ := univ

def A : Set ℝ := {x | x > 0}

def B : Set ℝ := {x | x^2 - x - 2 > 0}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (R \ B) = Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1966_196618


namespace NUMINAMATH_CALUDE_selection_theorem_l1966_196610

def boys : ℕ := 4
def girls : ℕ := 5
def total_selection : ℕ := 5

theorem selection_theorem :
  -- Condition 1
  (Nat.choose boys 2 * Nat.choose (girls - 1) 2 = 36) ∧
  -- Condition 2
  (Nat.choose (boys - 1 + girls - 1) (total_selection - 1) +
   Nat.choose (boys - 1 + girls) total_selection = 91) := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l1966_196610


namespace NUMINAMATH_CALUDE_campers_fed_l1966_196653

/-- Represents the types of fish caught -/
inductive FishType
  | Trout
  | Bass
  | Salmon

/-- Represents the catch of fish -/
structure Catch where
  troutWeight : ℕ
  bassCount : ℕ
  bassWeight : ℕ
  salmonCount : ℕ
  salmonWeight : ℕ

/-- Calculates the total weight of fish caught -/
def totalWeight (c : Catch) : ℕ :=
  c.troutWeight + c.bassCount * c.bassWeight + c.salmonCount * c.salmonWeight

/-- Calculates the number of campers that can be fed -/
def campersCanFeed (c : Catch) (poundsPerPerson : ℕ) : ℕ :=
  totalWeight c / poundsPerPerson

/-- Theorem stating the number of campers that can be fed -/
theorem campers_fed (c : Catch) (poundsPerPerson : ℕ) :
  c.troutWeight = 8 ∧ 
  c.bassCount = 6 ∧ 
  c.bassWeight = 2 ∧ 
  c.salmonCount = 2 ∧ 
  c.salmonWeight = 12 ∧ 
  poundsPerPerson = 2 → 
  campersCanFeed c poundsPerPerson = 22 := by
  sorry

end NUMINAMATH_CALUDE_campers_fed_l1966_196653


namespace NUMINAMATH_CALUDE_andrew_donation_start_age_l1966_196668

/-- Proves that Andrew started donating at age 10 given the conditions --/
theorem andrew_donation_start_age 
  (current_age : ℕ) 
  (total_donation : ℕ) 
  (yearly_donation : ℕ) 
  (h1 : current_age = 29) 
  (h2 : total_donation = 133) 
  (h3 : yearly_donation = 7) : 
  current_age - (total_donation / yearly_donation) = 10 := by
  sorry

end NUMINAMATH_CALUDE_andrew_donation_start_age_l1966_196668


namespace NUMINAMATH_CALUDE_female_salmon_count_l1966_196691

theorem female_salmon_count (total : Nat) (male : Nat) (h1 : total = 971639) (h2 : male = 712261) :
  total - male = 259378 := by
  sorry

end NUMINAMATH_CALUDE_female_salmon_count_l1966_196691


namespace NUMINAMATH_CALUDE_eric_blue_marbles_l1966_196651

theorem eric_blue_marbles :
  let total_marbles : ℕ := 20
  let white_marbles : ℕ := 12
  let green_marbles : ℕ := 2
  let blue_marbles : ℕ := total_marbles - white_marbles - green_marbles
  blue_marbles = 6 := by sorry

end NUMINAMATH_CALUDE_eric_blue_marbles_l1966_196651


namespace NUMINAMATH_CALUDE_largest_valid_B_l1966_196632

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 100000
  let d2 := (n / 10000) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 100) % 10
  let d5 := (n / 10) % 10
  let d6 := n % 10
  d1 + d2 + d3 + d4 + d5 + d6

def is_valid_B (B : ℕ) : Prop :=
  B < 10 ∧ 
  is_divisible_by_3 (sum_of_digits (400000 + B * 10000 + 4832)) ∧
  is_divisible_by_4 (last_two_digits (400000 + B * 10000 + 4832))

theorem largest_valid_B :
  ∀ B, is_valid_B B → B ≤ 9 ∧ is_valid_B 9 := by sorry

end NUMINAMATH_CALUDE_largest_valid_B_l1966_196632


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_zero_l1966_196658

theorem sum_of_x_and_y_is_zero (x y : ℝ) 
  (h : (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = 1) : 
  x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_zero_l1966_196658


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1966_196645

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ ∃ r : ℝ, r > 0 ∧ ∀ k : ℕ, a (k + 1) = r * a k

-- Define the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  (a 1)^2 - 10*(a 1) + 16 = 0 →
  (a 19)^2 - 10*(a 19) + 16 = 0 →
  a 8 * a 10 * a 12 = 64 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1966_196645


namespace NUMINAMATH_CALUDE_cylinder_cone_hemisphere_volume_l1966_196644

/-- Given a cylinder with volume 72π cm³, prove that the combined volume of a cone 
    with the same height as the cylinder and a hemisphere with the same radius 
    as the cylinder is equal to 72π cm³. -/
theorem cylinder_cone_hemisphere_volume 
  (r : ℝ) 
  (h : ℝ) 
  (cylinder_volume : π * r^2 * h = 72 * π) : 
  (1/3) * π * r^2 * h + (2/3) * π * r^3 = 72 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_hemisphere_volume_l1966_196644


namespace NUMINAMATH_CALUDE_hyperbola_center_is_3_4_l1966_196678

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x + 16 * y^2 - 128 * y - 896 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 4)

/-- Theorem: The center of the hyperbola is (3, 4) -/
theorem hyperbola_center_is_3_4 :
  ∀ (x y : ℝ), hyperbola_equation x y →
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (dx dy : ℝ),
  dx^2 + dy^2 < ε^2 →
  hyperbola_equation (hyperbola_center.1 + dx) (hyperbola_center.2 + dy) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_3_4_l1966_196678


namespace NUMINAMATH_CALUDE_largest_quantity_l1966_196695

theorem largest_quantity (A B C : ℚ) : 
  A = 2020/2019 + 2020/2021 →
  B = 2021/2022 + 2023/2022 →
  C = 2022/2021 + 2022/2023 →
  A > B ∧ A > C :=
by sorry

end NUMINAMATH_CALUDE_largest_quantity_l1966_196695


namespace NUMINAMATH_CALUDE_prob_consonant_correct_l1966_196649

/-- The word from which letters are selected -/
def word : String := "barkhint"

/-- The number of letters in the word -/
def word_length : Nat := word.length

/-- The number of vowels in the word -/
def vowel_count : Nat := (word.toList.filter (fun c => c ∈ ['a', 'e', 'i', 'o', 'u'])).length

/-- The probability of selecting at least one consonant when choosing two letters at random -/
def prob_at_least_one_consonant : ℚ := 27 / 28

/-- Theorem stating that the probability of selecting at least one consonant
    when choosing two letters at random from the word "barkhint" is 27/28 -/
theorem prob_consonant_correct :
  prob_at_least_one_consonant = 1 - (vowel_count / word_length) * ((vowel_count - 1) / (word_length - 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_prob_consonant_correct_l1966_196649


namespace NUMINAMATH_CALUDE_no_real_roots_composition_l1966_196647

/-- Given a quadratic function f(x) = ax^2 + bx + c where a ≠ 0,
    if f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_real_roots_composition (a b c : ℝ) (ha : a ≠ 0) :
  ((b - 1)^2 - 4*a*c < 0) →
  (a^2 * ((b + 1)^2 - 4*(a*c + b + 1)) < 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_composition_l1966_196647


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l1966_196657

/-- Proves that a rectangular garden with length three times its width and area 588 square meters has a width of 14 meters. -/
theorem rectangular_garden_width :
  ∀ (width length area : ℝ),
    length = 3 * width →
    area = length * width →
    area = 588 →
    width = 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l1966_196657


namespace NUMINAMATH_CALUDE_second_machine_rate_l1966_196633

/-- Represents a copy machine with a constant rate of copies per minute -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Represents a system of two copy machines -/
structure TwoMachineSystem where
  machine1 : CopyMachine
  machine2 : CopyMachine

/-- Calculates the total copies made by a system in a given time -/
def total_copies (system : TwoMachineSystem) (minutes : ℕ) : ℕ :=
  (system.machine1.copies_per_minute + system.machine2.copies_per_minute) * minutes

/-- Theorem: Given the conditions, the second machine makes 65 copies per minute -/
theorem second_machine_rate (system : TwoMachineSystem) :
  system.machine1.copies_per_minute = 35 →
  total_copies system 30 = 3000 →
  system.machine2.copies_per_minute = 65 := by
  sorry

#check second_machine_rate

end NUMINAMATH_CALUDE_second_machine_rate_l1966_196633


namespace NUMINAMATH_CALUDE_large_rectangle_length_is_40_l1966_196674

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Represents a configuration of rectangles -/
structure RectangleConfiguration where
  smallRectangle : Rectangle
  count : ℕ

/-- Calculates the length of the longer side of a large rectangle 
    formed by arranging smaller rectangles -/
def largeRectangleLength (config : RectangleConfiguration) : ℝ :=
  config.smallRectangle.width * 2 + config.smallRectangle.length

theorem large_rectangle_length_is_40 :
  ∀ (config : RectangleConfiguration),
    config.smallRectangle.width = 10 →
    config.count = 4 →
    config.smallRectangle.length = 2 * config.smallRectangle.width →
    largeRectangleLength config = 40 := by
  sorry

#check large_rectangle_length_is_40

end NUMINAMATH_CALUDE_large_rectangle_length_is_40_l1966_196674


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1966_196680

theorem simplify_square_roots : 
  (Real.sqrt 392 / Real.sqrt 56) - (Real.sqrt 252 / Real.sqrt 63) = Real.sqrt 7 - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1966_196680


namespace NUMINAMATH_CALUDE_sum_of_symmetric_points_l1966_196682

/-- Two points M(a, -3) and N(4, b) are symmetric with respect to the origin -/
def symmetric_points (a b : ℝ) : Prop :=
  (a = -4) ∧ (b = 3)

/-- Theorem: If M(a, -3) and N(4, b) are symmetric with respect to the origin, then a + b = -1 -/
theorem sum_of_symmetric_points (a b : ℝ) (h : symmetric_points a b) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_symmetric_points_l1966_196682


namespace NUMINAMATH_CALUDE_second_player_wins_l1966_196669

/-- Represents a position on an 8x8 chessboard -/
def Position := Fin 8 × Fin 8

/-- Represents a knight's move -/
def KnightMove := List (Int × Int)

/-- The list of possible knight moves -/
def knightMoves : KnightMove :=
  [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]

/-- Checks if a position is valid on the 8x8 board -/
def isValidPosition (p : Position) : Bool := true

/-- Checks if two positions are a knight's move apart -/
def isKnightMove (p1 p2 : Position) : Bool := sorry

/-- Represents the state of the game -/
structure GameState where
  placedKnights : List Position
  currentPlayer : Nat

/-- Checks if a move is legal given the current game state -/
def isLegalMove (state : GameState) (move : Position) : Bool := sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Option Position

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (player : Nat) (strat : Strategy) : Prop := sorry

/-- The main theorem: the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strat : Strategy), isWinningStrategy 1 strat := sorry

end NUMINAMATH_CALUDE_second_player_wins_l1966_196669


namespace NUMINAMATH_CALUDE_min_value_problem_l1966_196604

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 3/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 3/b = 1 → 2*x + 3*y ≤ 2*a + 3*b ∧ 
  ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ 2/c + 3/d = 1 ∧ 2*c + 3*d = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1966_196604


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1966_196692

theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ x y : ℝ, a^2 * x + y + 7 = 0 ∧ x - 2 * a * y + 1 = 0) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (a^2 * x₁ + y₁ + 7 = 0 ∧ x₁ - 2 * a * y₁ + 1 = 0) →
    (a^2 * x₂ + y₂ + 7 = 0 ∧ x₂ - 2 * a * y₂ + 1 = 0) →
    (x₂ - x₁) * (a^2 * (x₂ - x₁) + (y₂ - y₁)) = 0) →
  a = 0 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1966_196692


namespace NUMINAMATH_CALUDE_third_circle_radius_l1966_196663

theorem third_circle_radius (r1 r2 r3 : ℝ) : 
  r1 = 15 → r2 = 25 → π * r3^2 = π * (r2^2 - r1^2) → r3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l1966_196663


namespace NUMINAMATH_CALUDE_book_selection_ways_l1966_196688

-- Define the number of books on the shelf
def n : ℕ := 10

-- Define the number of books to be selected
def k : ℕ := 5

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem book_selection_ways : combination n k = 252 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_ways_l1966_196688


namespace NUMINAMATH_CALUDE_ellipse_max_y_coordinate_l1966_196628

theorem ellipse_max_y_coordinate :
  ∀ x y : ℝ, x^2/25 + (y-3)^2/9 = 1 → y ≤ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_max_y_coordinate_l1966_196628


namespace NUMINAMATH_CALUDE_power_of_product_l1966_196670

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1966_196670


namespace NUMINAMATH_CALUDE_game_total_score_l1966_196629

/-- Represents the scores of three teams in a soccer game -/
structure GameScores where
  teamA_first : ℕ
  teamB_first : ℕ
  teamC_first : ℕ
  teamA_second : ℕ
  teamB_second : ℕ
  teamC_second : ℕ

/-- Calculates the total score of all teams -/
def totalScore (scores : GameScores) : ℕ :=
  scores.teamA_first + scores.teamB_first + scores.teamC_first +
  scores.teamA_second + scores.teamB_second + scores.teamC_second

/-- Theorem stating the total score of the game -/
theorem game_total_score :
  ∀ (scores : GameScores),
  scores.teamA_first = 8 →
  scores.teamB_first = scores.teamA_first / 2 →
  scores.teamC_first = 2 * scores.teamB_first →
  scores.teamA_second = scores.teamC_first →
  scores.teamB_second = scores.teamA_first →
  scores.teamC_second = scores.teamB_second + 3 →
  totalScore scores = 47 := by
  sorry


end NUMINAMATH_CALUDE_game_total_score_l1966_196629


namespace NUMINAMATH_CALUDE_hawk_pregnancies_l1966_196696

theorem hawk_pregnancies (num_kettles : ℕ) (babies_per_pregnancy : ℕ) 
  (survival_rate : ℚ) (total_expected_babies : ℕ) :
  num_kettles = 6 →
  babies_per_pregnancy = 4 →
  survival_rate = 3/4 →
  total_expected_babies = 270 →
  (total_expected_babies : ℚ) / (num_kettles * babies_per_pregnancy * survival_rate) = 15 := by
  sorry

end NUMINAMATH_CALUDE_hawk_pregnancies_l1966_196696


namespace NUMINAMATH_CALUDE_circle_equation_l1966_196693

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : ℝ := x - y - 1
def line2 (x y : ℝ) : ℝ := 4*x + 3*y + 14
def line3 (x y : ℝ) : ℝ := 3*x + 4*y + 10

-- State the theorem
theorem circle_equation (C : Circle) :
  (∀ x y, line1 x y = 0 → x = C.center.1 ∧ y = C.center.2) →
  (∃ x y, line2 x y = 0 ∧ (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2) →
  (∃ x1 y1 x2 y2, line3 x1 y1 = 0 ∧ line3 x2 y2 = 0 ∧
    (x1 - C.center.1)^2 + (y1 - C.center.2)^2 = C.radius^2 ∧
    (x2 - C.center.1)^2 + (y2 - C.center.2)^2 = C.radius^2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 36) →
  C.center = (2, 1) ∧ C.radius = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1966_196693


namespace NUMINAMATH_CALUDE_divisibility_theorems_l1966_196621

def divisible_by_7 (n : Int) : Prop := ∃ k : Int, n = 7 * k

theorem divisibility_theorems :
  (∀ a b : Int, divisible_by_7 a ∧ divisible_by_7 b → divisible_by_7 (a + b)) ∧
  (∀ a b : Int, ¬divisible_by_7 (a + b) → ¬divisible_by_7 a ∨ ¬divisible_by_7 b) ∧
  ¬(∀ a b : Int, ¬divisible_by_7 a ∧ ¬divisible_by_7 b → ¬divisible_by_7 (a + b)) ∧
  ¬(∀ a b : Int, divisible_by_7 a ∨ divisible_by_7 b → divisible_by_7 (a + b)) ∧
  ¬(∀ a b : Int, divisible_by_7 (a + b) → divisible_by_7 a ∧ divisible_by_7 b) ∧
  ¬(∀ a b : Int, ¬divisible_by_7 (a + b) → ¬divisible_by_7 a ∧ ¬divisible_by_7 b) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_theorems_l1966_196621


namespace NUMINAMATH_CALUDE_mary_travel_time_l1966_196623

/-- Represents the time in minutes for Mary's travel process -/
def travel_process (uber_to_house bag_check waiting_for_boarding : ℕ) : ℕ :=
  let uber_to_airport := 5 * uber_to_house
  let security := 3 * bag_check
  let waiting_for_takeoff := 2 * waiting_for_boarding
  uber_to_house + uber_to_airport + bag_check + security + waiting_for_boarding + waiting_for_takeoff

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℕ) : ℚ :=
  minutes / 60

theorem mary_travel_time :
  minutes_to_hours (travel_process 10 15 20) = 3 := by
  sorry

end NUMINAMATH_CALUDE_mary_travel_time_l1966_196623


namespace NUMINAMATH_CALUDE_matchstick_100th_stage_l1966_196675

/-- Represents the number of matchsticks in each stage of the geometric shape construction -/
def matchstick_sequence : ℕ → ℕ
  | 0 => 4  -- First stage (index 0) has 4 matchsticks
  | n + 1 => matchstick_sequence n + 5  -- Each subsequent stage adds 5 matchsticks

/-- Theorem stating that the 100th stage (index 99) requires 499 matchsticks -/
theorem matchstick_100th_stage : matchstick_sequence 99 = 499 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_100th_stage_l1966_196675


namespace NUMINAMATH_CALUDE_increasing_cubic_range_l1966_196656

/-- A cubic function f(x) = x³ + ax² + 7ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 7*a*x

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 7*a

theorem increasing_cubic_range (a : ℝ) :
  (∀ x : ℝ, (f_deriv a x) ≥ 0) → 0 ≤ a ∧ a ≤ 21 :=
sorry

end NUMINAMATH_CALUDE_increasing_cubic_range_l1966_196656


namespace NUMINAMATH_CALUDE_hash_three_six_l1966_196630

/-- Custom operation # defined for any two real numbers -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem stating that 3 # 6 = 48 -/
theorem hash_three_six : hash 3 6 = 48 := by sorry

end NUMINAMATH_CALUDE_hash_three_six_l1966_196630


namespace NUMINAMATH_CALUDE_smallest_possible_a_for_parabola_l1966_196616

theorem smallest_possible_a_for_parabola :
  ∀ (a b c : ℚ),
    a > 0 →
    (∃ (n : ℤ), 2 * a + b + 3 * c = n) →
    (∀ (x : ℚ), a * (x - 3/5)^2 - 13/5 = a * x^2 + b * x + c) →
    (∀ (a' : ℚ), a' > 0 ∧ 
      (∃ (b' c' : ℚ) (n' : ℤ), 2 * a' + b' + 3 * c' = n' ∧
        (∀ (x : ℚ), a' * (x - 3/5)^2 - 13/5 = a' * x^2 + b' * x + c')) →
      a ≤ a') →
    a = 45/19 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_a_for_parabola_l1966_196616


namespace NUMINAMATH_CALUDE_f_eight_equals_zero_l1966_196673

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_two_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

theorem f_eight_equals_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_sym : has_period_two_symmetry f) :
  f 8 = 0 :=
sorry

end NUMINAMATH_CALUDE_f_eight_equals_zero_l1966_196673


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l1966_196687

-- Define set A
def A : Set ℝ := {x | |x - 1| < 1}

-- Define set B
def B : Set ℝ := {x | x ≤ 2}

-- Theorem statement
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l1966_196687


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_squared_l1966_196641

-- Define the parabolas and their properties
def Parabola : Type := ℝ × ℝ → Prop

-- Define the focus and directrix of a parabola
structure ParabolaProperties where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ

-- Define the two parabolas
def P₁ : Parabola := sorry
def P₂ : Parabola := sorry

-- Define the properties of the two parabolas
def P₁_props : ParabolaProperties := sorry
def P₂_props : ParabolaProperties := sorry

-- Define the condition that the foci and directrices are parallel
def parallel_condition (P₁_props P₂_props : ParabolaProperties) : Prop := sorry

-- Define the condition that each focus lies on the other parabola
def focus_on_parabola (P : Parabola) (F : ℝ × ℝ) : Prop := sorry

-- Define the distance between foci
def foci_distance (P₁_props P₂_props : ParabolaProperties) : ℝ := sorry

-- Define the intersection points of the parabolas
def intersection_points (P₁ P₂ : Parabola) : Set (ℝ × ℝ) := sorry

-- Main theorem
theorem parabola_intersection_distance_squared 
  (P₁ P₂ : Parabola) 
  (P₁_props P₂_props : ParabolaProperties) 
  (h₁ : parallel_condition P₁_props P₂_props)
  (h₂ : focus_on_parabola P₂ P₁_props.focus)
  (h₃ : focus_on_parabola P₁ P₂_props.focus)
  (h₄ : foci_distance P₁_props P₂_props = 1)
  (h₅ : ∃ A B, A ∈ intersection_points P₁ P₂ ∧ B ∈ intersection_points P₁ P₂ ∧ A ≠ B) :
  ∃ A B, A ∈ intersection_points P₁ P₂ ∧ B ∈ intersection_points P₁ P₂ ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 15/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_squared_l1966_196641


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1966_196679

theorem trigonometric_identities :
  (Real.sin (62 * π / 180) * Real.cos (32 * π / 180) - Real.cos (62 * π / 180) * Real.sin (32 * π / 180) = 1/2) ∧
  (Real.sin (75 * π / 180) * Real.cos (75 * π / 180) ≠ Real.sqrt 3 / 4) ∧
  ((1 + Real.tan (75 * π / 180)) / (1 - Real.tan (75 * π / 180)) ≠ Real.sqrt 3) ∧
  (Real.sin (50 * π / 180) * (Real.sqrt 3 * Real.sin (10 * π / 180) + Real.cos (10 * π / 180)) / Real.cos (10 * π / 180) = 1) := by
  sorry

#check trigonometric_identities

end NUMINAMATH_CALUDE_trigonometric_identities_l1966_196679


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1966_196677

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 1 → a + Real.log b > b + Real.log a) ∧
  (∃ a b : ℝ, a + Real.log b > b + Real.log a ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1966_196677


namespace NUMINAMATH_CALUDE_money_distribution_l1966_196659

theorem money_distribution (A B C : ℤ) 
  (total : A + B + C = 300)
  (ac_sum : A + C = 200)
  (bc_sum : B + C = 350) :
  C = 250 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1966_196659


namespace NUMINAMATH_CALUDE_mary_lamb_count_l1966_196619

/-- Calculates the final number of lambs Mary has given the initial conditions. -/
def final_lamb_count (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) 
  (traded_lambs : ℕ) (extra_lambs : ℕ) : ℕ :=
  initial_lambs + lambs_with_babies * babies_per_lamb - traded_lambs + extra_lambs

/-- Proves that Mary ends up with 14 lambs given the initial conditions. -/
theorem mary_lamb_count : 
  final_lamb_count 6 2 2 3 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_mary_lamb_count_l1966_196619


namespace NUMINAMATH_CALUDE_power_function_unique_m_l1966_196685

/-- A function f: ℝ → ℝ is increasing on (0, +∞) if for all x₁, x₂ ∈ (0, +∞),
    x₁ < x₂ implies f(x₁) < f(x₂) -/
def IncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f x₁ < f x₂

/-- A function f: ℝ → ℝ is a power function if there exist constants a and b
    such that f(x) = a * x^b for all x > 0 -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, x > 0 → f x = a * x^b

theorem power_function_unique_m :
  ∃! m : ℝ, IsPowerFunction (fun x ↦ (m^2 - m - 1) * x^(m^2 - 3*m - 3)) ∧
            IncreasingOn (fun x ↦ (m^2 - m - 1) * x^(m^2 - 3*m - 3)) ∧
            m = -1 :=
sorry

end NUMINAMATH_CALUDE_power_function_unique_m_l1966_196685


namespace NUMINAMATH_CALUDE_anti_terrorism_drill_mode_l1966_196699

/-- The mode of a binomial distribution with parameters n and p -/
def binomial_mode (n : ℕ) (p : ℝ) : Set ℕ :=
  {k : ℕ | k ≤ n ∧ (k.pred : ℝ) < n * p ∧ n * p ≤ k}

theorem anti_terrorism_drill_mode :
  binomial_mode 99 0.8 = {79, 80} := by
  sorry

end NUMINAMATH_CALUDE_anti_terrorism_drill_mode_l1966_196699


namespace NUMINAMATH_CALUDE_probability_skew_edges_probability_skew_edges_proof_l1966_196654

/-- A cube with edge length 1 -/
structure Cube where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- The number of edges in a cube -/
def num_edges : ℕ := 12

/-- The number of edges remaining after choosing one edge -/
def remaining_edges : ℕ := 11

/-- The number of edges parallel to a chosen edge -/
def parallel_edges : ℕ := 3

/-- The number of edges perpendicular to a chosen edge -/
def perpendicular_edges : ℕ := 4

/-- The number of edges skew to a chosen edge -/
def skew_edges : ℕ := 4

/-- The probability that two randomly chosen edges of a cube lie on skew lines -/
theorem probability_skew_edges (c : Cube) : ℚ :=
  4 / 11

/-- Proof that the probability of two randomly chosen edges of a cube lying on skew lines is 4/11 -/
theorem probability_skew_edges_proof (c : Cube) :
  probability_skew_edges c = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_skew_edges_probability_skew_edges_proof_l1966_196654


namespace NUMINAMATH_CALUDE_twenty_seven_binary_l1966_196602

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Convert a list of booleans to a natural number in base 2 -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem twenty_seven_binary :
  toBinary 27 = [true, true, false, true, true] :=
sorry

end NUMINAMATH_CALUDE_twenty_seven_binary_l1966_196602


namespace NUMINAMATH_CALUDE_problem_1_l1966_196605

theorem problem_1 : -9 + 5 - (-12) + (-3) = 8 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1966_196605


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_zero_l1966_196631

theorem tan_alpha_2_implies_expression_zero (α : Real) (h : Real.tan α = 2) :
  2 * (Real.sin α)^2 - 3 * Real.sin α * Real.cos α - 2 * (Real.cos α)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_zero_l1966_196631


namespace NUMINAMATH_CALUDE_john_tax_payment_l1966_196640

/-- Calculates the total tax payment given earnings, deductions, and tax rates -/
def calculate_tax (earnings deductions : ℕ) (low_rate high_rate : ℚ) : ℚ :=
  let taxable_income := earnings - deductions
  let low_bracket := min taxable_income 20000
  let high_bracket := taxable_income - low_bracket
  (low_bracket : ℚ) * low_rate + (high_bracket : ℚ) * high_rate

/-- Theorem stating that John's tax payment is $12,000 -/
theorem john_tax_payment :
  calculate_tax 100000 30000 (1/10) (1/5) = 12000 := by
  sorry

end NUMINAMATH_CALUDE_john_tax_payment_l1966_196640


namespace NUMINAMATH_CALUDE_chessboard_cut_theorem_l1966_196609

/-- Represents a 2×1 domino on a chessboard -/
structure Domino :=
  (x : ℕ) (y : ℕ)

/-- Represents a chessboard configuration -/
structure Chessboard :=
  (n : ℕ)
  (dominoes : List Domino)

/-- Checks if a given line cuts through any domino -/
def line_cuts_domino (board : Chessboard) (line : ℕ) (is_vertical : Bool) : Prop :=
  sorry

/-- Checks if there exists a line that doesn't cut any domino -/
def exists_uncut_line (board : Chessboard) : Prop :=
  sorry

/-- The main theorem -/
theorem chessboard_cut_theorem (board : Chessboard) :
  (board.dominoes.length = 2 * board.n^2) →
  (exists_uncut_line board ↔ board.n = 1 ∨ board.n = 2) :=
sorry

end NUMINAMATH_CALUDE_chessboard_cut_theorem_l1966_196609


namespace NUMINAMATH_CALUDE_sum_product_zero_l1966_196690

theorem sum_product_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : a*c + b*d = 0) : 
  a*b + c*d = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_product_zero_l1966_196690


namespace NUMINAMATH_CALUDE_blair_received_15_bars_l1966_196686

/-- Represents the distribution of gold bars among three people -/
structure GoldDistribution where
  total_bars : ℕ
  total_weight : ℝ
  brennan_bars : ℕ
  maya_bars : ℕ
  blair_bars : ℕ
  brennan_weight_percent : ℝ
  maya_weight_percent : ℝ

/-- The conditions of the gold bar distribution problem -/
def gold_distribution_conditions (d : GoldDistribution) : Prop :=
  d.total_bars = d.brennan_bars + d.maya_bars + d.blair_bars ∧
  d.brennan_bars = 24 ∧
  d.maya_bars = 13 ∧
  d.brennan_weight_percent = 45 ∧
  d.maya_weight_percent = 26 ∧
  d.brennan_weight_percent + d.maya_weight_percent < 100

/-- Theorem stating that Blair received 15 gold bars -/
theorem blair_received_15_bars (d : GoldDistribution) 
  (h : gold_distribution_conditions d) : d.blair_bars = 15 := by
  sorry

end NUMINAMATH_CALUDE_blair_received_15_bars_l1966_196686


namespace NUMINAMATH_CALUDE_inequality_range_l1966_196614

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1966_196614


namespace NUMINAMATH_CALUDE_average_difference_l1966_196608

/-- The total number of students in the school -/
def total_students : ℕ := 120

/-- The class sizes -/
def class_sizes : List ℕ := [60, 30, 20, 5, 5]

/-- The number of teachers -/
def total_teachers : ℕ := 6

/-- The number of teaching teachers -/
def teaching_teachers : ℕ := 5

/-- Average class size from teaching teachers' perspective -/
def t : ℚ := (List.sum class_sizes) / teaching_teachers

/-- Average class size from students' perspective -/
def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes)) / total_students

theorem average_difference : t - s = -17.25 := by sorry

end NUMINAMATH_CALUDE_average_difference_l1966_196608


namespace NUMINAMATH_CALUDE_current_rate_l1966_196650

/-- Calculates the rate of the current given a man's rowing speeds -/
theorem current_rate (downstream_speed upstream_speed still_water_speed : ℝ) 
  (h1 : downstream_speed = 32)
  (h2 : upstream_speed = 17)
  (h3 : still_water_speed = 24.5)
  : (downstream_speed - still_water_speed) = 7.5 := by
  sorry

#check current_rate

end NUMINAMATH_CALUDE_current_rate_l1966_196650


namespace NUMINAMATH_CALUDE_rectangle_combination_perimeter_l1966_196683

theorem rectangle_combination_perimeter : ∀ (l w : ℝ),
  l = 4 ∧ w = 2 →
  ∃ (new_l new_w : ℝ),
    ((new_l = l + l ∧ new_w = w) ∨ (new_l = l ∧ new_w = w + w)) ∧
    (2 * (new_l + new_w) = 20 ∨ 2 * (new_l + new_w) = 16) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_combination_perimeter_l1966_196683


namespace NUMINAMATH_CALUDE_tank_capacity_l1966_196611

theorem tank_capacity : ∃ (x : ℝ), x > 0 ∧ (3/4 * x - 1/3 * x = 18) ∧ x = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1966_196611


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l1966_196672

/-- An arithmetic sequence is a sequence where the difference between 
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_15th_term 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_5 : a 5 = 5) 
  (h_10 : a 10 = 15) : 
  a 15 = 25 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l1966_196672


namespace NUMINAMATH_CALUDE_outfit_combinations_l1966_196620

def num_shirts : ℕ := 5
def num_pants : ℕ := 3
def num_hats : ℕ := 2

theorem outfit_combinations : num_shirts * num_pants * num_hats = 30 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1966_196620


namespace NUMINAMATH_CALUDE_expression_value_l1966_196642

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : a ≠ 0) 
  (h3 : c * d = 1) 
  (h4 : |m| = 3) : 
  m^2 - (-1) + |a + b| - c * d * m = 7 ∨ m^2 - (-1) + |a + b| - c * d * m = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1966_196642


namespace NUMINAMATH_CALUDE_billie_whipped_cream_cans_l1966_196648

/-- The number of cans of whipped cream needed to cover the remaining pies after Tiffany eats some -/
def whipped_cream_cans_needed (pies_per_day : ℕ) (days : ℕ) (cream_cans_per_pie : ℕ) (pies_eaten : ℕ) : ℕ :=
  (pies_per_day * days - pies_eaten) * cream_cans_per_pie

/-- Theorem stating the number of whipped cream cans Billie needs to buy -/
theorem billie_whipped_cream_cans : 
  whipped_cream_cans_needed 3 11 2 4 = 58 := by
  sorry

end NUMINAMATH_CALUDE_billie_whipped_cream_cans_l1966_196648


namespace NUMINAMATH_CALUDE_solve_equation_l1966_196697

theorem solve_equation (x : ℝ) : 3 * x + 1 = -(5 - 2 * x) → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1966_196697


namespace NUMINAMATH_CALUDE_inequality_proof_l1966_196689

theorem inequality_proof (a b c d e f : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f) : 
  (((a * b * c) / (a + b + d)) ^ (1/3 : ℝ)) + (((d * e * f) / (c + e + f)) ^ (1/3 : ℝ)) < 
  ((a + b + d) * (c + e + f)) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1966_196689


namespace NUMINAMATH_CALUDE_abs_inequality_l1966_196684

theorem abs_inequality (x y : ℝ) 
  (h1 : |x + y + 1| ≤ 1/3) 
  (h2 : |y - 1/3| ≤ 2/3) : 
  |2/3 * x + 1| ≥ 4/9 := by
sorry

end NUMINAMATH_CALUDE_abs_inequality_l1966_196684


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_90_l1966_196624

theorem thirty_percent_less_than_90 (x : ℝ) : x + (1/4) * x = 63 ↔ x = 50.4 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_90_l1966_196624


namespace NUMINAMATH_CALUDE_square_sum_equals_eight_l1966_196698

theorem square_sum_equals_eight : (-2)^2 + 2^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_eight_l1966_196698


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l1966_196627

theorem at_least_one_real_root (m : ℝ) : 
  ¬(∀ x : ℝ, x^2 - 5*x + m ≠ 0 ∧ 2*x^2 + x + 6 - m ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l1966_196627
