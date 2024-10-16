import Mathlib

namespace NUMINAMATH_CALUDE_lateral_surface_is_parallelogram_l3406_340611

-- Define the types for our geometric objects
inductive PrismType
| Right
| Oblique

-- Define the shapes we're considering
inductive Shape
| Rectangle
| Parallelogram

-- Define a function that returns the possible shapes of a prism's lateral surface
def lateralSurfaceShape (p : PrismType) : Set Shape :=
  match p with
  | PrismType.Right => {Shape.Rectangle}
  | PrismType.Oblique => {Shape.Rectangle, Shape.Parallelogram}

-- Theorem statement
theorem lateral_surface_is_parallelogram :
  ∀ (p : PrismType), ∃ (s : Shape), s ∈ lateralSurfaceShape p → s = Shape.Parallelogram := by
  sorry

#check lateral_surface_is_parallelogram

end NUMINAMATH_CALUDE_lateral_surface_is_parallelogram_l3406_340611


namespace NUMINAMATH_CALUDE_sum_15_with_9_dice_l3406_340607

/-- The number of ways to distribute n indistinguishable objects among k distinct containers,
    with no container receiving more than m objects. -/
def distribute (n k m : ℕ) : ℕ := sorry

/-- The number of ways to throw 9 fair 6-sided dice such that their sum is 15. -/
def ways_to_sum_15 : ℕ := distribute 6 9 5

theorem sum_15_with_9_dice : ways_to_sum_15 = 3003 := by sorry

end NUMINAMATH_CALUDE_sum_15_with_9_dice_l3406_340607


namespace NUMINAMATH_CALUDE_apples_fit_count_l3406_340651

def jack_basket_full_capacity : ℕ := 12
def jack_basket_space_left : ℕ := 4
def jill_basket_capacity_ratio : ℕ := 2

def jack_current_apples : ℕ := jack_basket_full_capacity - jack_basket_space_left
def jill_basket_capacity : ℕ := jill_basket_capacity_ratio * jack_basket_full_capacity

theorem apples_fit_count : (jill_basket_capacity / jack_current_apples) = 3 := by
  sorry

end NUMINAMATH_CALUDE_apples_fit_count_l3406_340651


namespace NUMINAMATH_CALUDE_percent_of_a_l3406_340636

theorem percent_of_a (a b c : ℝ) (h1 : b = 0.35 * a) (h2 : c = 0.4 * b) : c = 0.14 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_l3406_340636


namespace NUMINAMATH_CALUDE_pastries_cakes_difference_l3406_340658

/-- The number of cakes made by the baker -/
def cakes_made : ℕ := 105

/-- The number of pastries made by the baker -/
def pastries_made : ℕ := 275

/-- The number of pastries sold by the baker -/
def pastries_sold : ℕ := 214

/-- The number of cakes sold by the baker -/
def cakes_sold : ℕ := 163

/-- Theorem stating the difference between pastries and cakes sold -/
theorem pastries_cakes_difference :
  pastries_sold - cakes_sold = 51 := by sorry

end NUMINAMATH_CALUDE_pastries_cakes_difference_l3406_340658


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3406_340646

structure IsoscelesTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  isIsosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3)
  sumIs180 : angle1 + angle2 + angle3 = 180

theorem isosceles_triangle_base_angle 
  (triangle : IsoscelesTriangle) 
  (has80DegreeAngle : triangle.angle1 = 80 ∨ triangle.angle2 = 80 ∨ triangle.angle3 = 80) :
  (∃ baseAngle : ℝ, (baseAngle = 80 ∨ baseAngle = 50) ∧ 
   ((triangle.angle1 = baseAngle ∧ triangle.angle2 = baseAngle) ∨
    (triangle.angle1 = baseAngle ∧ triangle.angle3 = baseAngle) ∨
    (triangle.angle2 = baseAngle ∧ triangle.angle3 = baseAngle))) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3406_340646


namespace NUMINAMATH_CALUDE_complement_union_A_B_l3406_340655

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

-- Define set B
def B : Set ℕ := {x ∈ U | ∃ a ∈ A, x = 2*a}

-- Theorem to prove
theorem complement_union_A_B : (U \ (A ∪ B)) = {0, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l3406_340655


namespace NUMINAMATH_CALUDE_multiples_of_eight_range_l3406_340619

theorem multiples_of_eight_range (end_num : ℕ) (num_multiples : ℚ) : 
  end_num = 200 →
  num_multiples = 13.5 →
  ∃ (start_num : ℕ), 
    start_num = 84 ∧
    (end_num - start_num) / 8 + 1 = num_multiples ∧
    start_num ≤ end_num ∧
    ∀ n : ℕ, start_num ≤ n ∧ n ≤ end_num → (n - start_num) % 8 = 0 → n ≤ end_num :=
by sorry


end NUMINAMATH_CALUDE_multiples_of_eight_range_l3406_340619


namespace NUMINAMATH_CALUDE_office_printing_calculation_l3406_340648

/-- Calculate the number of one-page documents printed per day -/
def documents_per_day (packs : ℕ) (sheets_per_pack : ℕ) (days : ℕ) : ℕ :=
  (packs * sheets_per_pack) / days

theorem office_printing_calculation :
  documents_per_day 2 240 6 = 80 := by
  sorry

end NUMINAMATH_CALUDE_office_printing_calculation_l3406_340648


namespace NUMINAMATH_CALUDE_log_identity_l3406_340623

theorem log_identity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hcb : c > b) 
  (h_pythagorean : a^2 + b^2 = c^2) : 
  Real.log a / Real.log (c + b) + Real.log a / Real.log (c - b) = 
  2 * (Real.log a / Real.log (c + b)) * (Real.log a / Real.log (c - b)) := by
sorry

end NUMINAMATH_CALUDE_log_identity_l3406_340623


namespace NUMINAMATH_CALUDE_kate_wand_sale_l3406_340673

/-- The amount of money Kate collected after selling magic wands -/
def kateCollected (numBought : ℕ) (numSold : ℕ) (costPerWand : ℕ) (markup : ℕ) : ℕ :=
  numSold * (costPerWand + markup)

/-- Theorem stating how much money Kate collected from selling magic wands -/
theorem kate_wand_sale :
  kateCollected 3 2 60 5 = 130 := by
  sorry

end NUMINAMATH_CALUDE_kate_wand_sale_l3406_340673


namespace NUMINAMATH_CALUDE_logarithm_inequality_l3406_340693

theorem logarithm_inequality (t : ℝ) (x y z : ℝ) 
  (ht : t > 1)
  (hx : x = Real.log t / Real.log 2)
  (hy : y = Real.log t / Real.log 3)
  (hz : z = Real.log t / Real.log 5) :
  3 * y < 2 * x ∧ 2 * x < 5 * z := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l3406_340693


namespace NUMINAMATH_CALUDE_license_plate_count_is_9750000_l3406_340604

/-- The number of possible distinct license plates -/
def license_plate_count : ℕ :=
  (Nat.choose 6 2) * 26 * 25 * (10^4)

/-- Theorem stating the number of distinct license plates -/
theorem license_plate_count_is_9750000 :
  license_plate_count = 9750000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_is_9750000_l3406_340604


namespace NUMINAMATH_CALUDE_range_of_x_l3406_340689

def p (x : ℝ) : Prop := (x + 2) * (x - 2) ≤ 0
def q (x : ℝ) : Prop := x^2 - 3*x - 4 ≤ 0

theorem range_of_x : 
  (∀ x : ℝ, ¬(p x ∧ q x)) → 
  (∀ x : ℝ, p x ∨ q x) → 
  {x : ℝ | p x ∨ q x} = {x : ℝ | -2 ≤ x ∧ x < -1} ∪ {x : ℝ | 2 < x ∧ x ≤ 4} :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l3406_340689


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3406_340697

theorem inequality_and_equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  1 + a^2017 + b^2017 ≥ a^10 * b^7 + a^7 * b^2000 + a^2000 * b^10 ∧
  (1 + a^2017 + b^2017 = a^10 * b^7 + a^7 * b^2000 + a^2000 * b^10 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3406_340697


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3406_340660

/-- Given a > 0 and a ≠ 1, prove that if f(x) = a^x is decreasing on ℝ, 
    then g(x) = (2-a)x^3 is increasing on ℝ, but not necessarily vice versa. -/
theorem sufficient_but_not_necessary 
  (a : ℝ) 
  (ha_pos : a > 0) 
  (ha_neq_one : a ≠ 1) 
  (f : ℝ → ℝ) 
  (hf : f = fun x ↦ a^x) 
  (g : ℝ → ℝ) 
  (hg : g = fun x ↦ (2-a)*x^3) : 
  (∀ x y, x < y → f x > f y) → 
  (∀ x y, x < y → g x < g y) ∧ 
  ¬(∀ x y, x < y → g x < g y → ∀ x y, x < y → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3406_340660


namespace NUMINAMATH_CALUDE_money_division_l3406_340675

/-- The problem of dividing money among three people -/
theorem money_division (total : ℚ) (c_share : ℚ) (b_ratio : ℚ) :
  total = 328 →
  c_share = 64 →
  b_ratio = 65 / 100 →
  ∃ (a_share : ℚ),
    a_share + b_ratio * a_share + c_share = total ∧
    (c_share * 100) / a_share = 40 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l3406_340675


namespace NUMINAMATH_CALUDE_johns_computer_purchase_cost_l3406_340608

theorem johns_computer_purchase_cost
  (computer_cost : ℝ)
  (peripherals_cost_ratio : ℝ)
  (original_video_card_cost : ℝ)
  (upgraded_video_card_cost_ratio : ℝ)
  (video_card_discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (h1 : computer_cost = 1500)
  (h2 : peripherals_cost_ratio = 1 / 4)
  (h3 : original_video_card_cost = 300)
  (h4 : upgraded_video_card_cost_ratio = 2.5)
  (h5 : video_card_discount_rate = 0.12)
  (h6 : sales_tax_rate = 0.05) :
  let peripherals_cost := computer_cost * peripherals_cost_ratio
  let upgraded_video_card_cost := original_video_card_cost * upgraded_video_card_cost_ratio
  let video_card_discount := upgraded_video_card_cost * video_card_discount_rate
  let final_video_card_cost := upgraded_video_card_cost - video_card_discount
  let sales_tax := peripherals_cost * sales_tax_rate
  let total_cost := computer_cost + peripherals_cost + final_video_card_cost + sales_tax
  total_cost = 2553.75 :=
by sorry

end NUMINAMATH_CALUDE_johns_computer_purchase_cost_l3406_340608


namespace NUMINAMATH_CALUDE_complex_number_problem_l3406_340620

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  z₁ = 1 + Complex.I * Real.sqrt 3 →
  Complex.abs z₂ = 2 →
  ∃ (r : ℝ), r > 0 ∧ z₁ * z₂ = r →
  z₂ = 1 - Complex.I * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3406_340620


namespace NUMINAMATH_CALUDE_josephine_milk_sales_l3406_340665

/-- Given the conditions of Josephine's milk sales, prove that the amount in each of the two unknown containers is 0.75 liters. -/
theorem josephine_milk_sales (total_milk : ℝ) (big_containers : ℕ) (small_containers : ℕ) (unknown_containers : ℕ)
  (big_container_capacity : ℝ) (small_container_capacity : ℝ)
  (h_total : total_milk = 10)
  (h_big : big_containers = 3)
  (h_small : small_containers = 5)
  (h_unknown : unknown_containers = 2)
  (h_big_capacity : big_container_capacity = 2)
  (h_small_capacity : small_container_capacity = 0.5) :
  (total_milk - (big_containers * big_container_capacity + small_containers * small_container_capacity)) / unknown_containers = 0.75 := by
sorry

end NUMINAMATH_CALUDE_josephine_milk_sales_l3406_340665


namespace NUMINAMATH_CALUDE_binomial_9_choose_5_l3406_340657

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_choose_5_l3406_340657


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3406_340656

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 2 ∧ b = 4 ∧ c = 4 →  -- Two sides are 4, one side is 2
  a + b + c = 10 :=        -- The perimeter is 10
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3406_340656


namespace NUMINAMATH_CALUDE_intersecting_circles_distance_l3406_340618

theorem intersecting_circles_distance (R r d : ℝ) : 
  R > 0 → r > 0 → R > r → 
  (∃ (x y : ℝ × ℝ), (x.1 - y.1)^2 + (x.2 - y.2)^2 = d^2 ∧ 
    ∃ (p : ℝ × ℝ), (p.1 - x.1)^2 + (p.2 - x.2)^2 = R^2 ∧ 
                   (p.1 - y.1)^2 + (p.2 - y.2)^2 = r^2) →
  R - r < d ∧ d < R + r :=
by sorry

end NUMINAMATH_CALUDE_intersecting_circles_distance_l3406_340618


namespace NUMINAMATH_CALUDE_specific_pairing_probability_l3406_340668

/-- Represents a classroom with male and female students -/
structure Classroom where
  female_count : ℕ
  male_count : ℕ

/-- Represents a pairing of students -/
structure Pairing where
  classroom : Classroom
  is_opposite_gender : Bool

/-- Calculates the probability of a specific pairing -/
def probability_of_specific_pairing (c : Classroom) (p : Pairing) : ℚ :=
  1 / c.male_count

/-- Theorem: The probability of a specific female-male pairing in a classroom
    with 20 female students and 18 male students is 1/18 -/
theorem specific_pairing_probability :
  let c : Classroom := { female_count := 20, male_count := 18 }
  let p : Pairing := { classroom := c, is_opposite_gender := true }
  probability_of_specific_pairing c p = 1 / 18 := by
    sorry

end NUMINAMATH_CALUDE_specific_pairing_probability_l3406_340668


namespace NUMINAMATH_CALUDE_total_money_l3406_340602

theorem total_money (mark_money : Rat) (carolyn_money : Rat) (jack_money : Rat)
  (h1 : mark_money = 4 / 5)
  (h2 : carolyn_money = 2 / 5)
  (h3 : jack_money = 1 / 2) :
  mark_money + carolyn_money + jack_money = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3406_340602


namespace NUMINAMATH_CALUDE_total_wheels_in_both_garages_l3406_340600

-- Define the types of vehicles
inductive Vehicle
| Bicycle
| Tricycle
| Unicycle
| Quadracycle

-- Define the garage contents
def first_garage : List (Vehicle × Nat) :=
  [(Vehicle.Bicycle, 5), (Vehicle.Tricycle, 6), (Vehicle.Unicycle, 9), (Vehicle.Quadracycle, 3)]

def second_garage : List (Vehicle × Nat) :=
  [(Vehicle.Bicycle, 2), (Vehicle.Tricycle, 1), (Vehicle.Unicycle, 3), (Vehicle.Quadracycle, 4)]

-- Define the number of wheels for each vehicle type
def wheels_per_vehicle (v : Vehicle) : Nat :=
  match v with
  | Vehicle.Bicycle => 2
  | Vehicle.Tricycle => 3
  | Vehicle.Unicycle => 1
  | Vehicle.Quadracycle => 4

-- Define the number of missing wheels in the second garage
def missing_wheels : Nat := 3

-- Function to calculate total wheels in a garage
def total_wheels_in_garage (garage : List (Vehicle × Nat)) : Nat :=
  garage.foldl (fun acc (v, count) => acc + wheels_per_vehicle v * count) 0

-- Theorem statement
theorem total_wheels_in_both_garages :
  total_wheels_in_garage first_garage +
  total_wheels_in_garage second_garage - missing_wheels = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_in_both_garages_l3406_340600


namespace NUMINAMATH_CALUDE_sum_equals_fraction_l3406_340695

def binomial_coefficient (n k : ℕ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def sum_expression : ℚ :=
  Finset.sum (Finset.range 8) (fun i =>
    let n := i + 3
    (binomial_coefficient n 2) / ((binomial_coefficient n 3) * (binomial_coefficient (n + 1) 3)))

theorem sum_equals_fraction :
  sum_expression = 164 / 165 :=
sorry

end NUMINAMATH_CALUDE_sum_equals_fraction_l3406_340695


namespace NUMINAMATH_CALUDE_captain_age_is_your_age_l3406_340671

/-- Represents the age of a person in years -/
def Age : Type := ℕ

/-- Represents a person -/
structure Person where
  age : Age

/-- Represents the captain of the steamboat -/
def Captain : Person := sorry

/-- Represents you -/
def You : Person := sorry

/-- The theorem states that the captain's age is equal to your age -/
theorem captain_age_is_your_age : Captain.age = You.age := by sorry

end NUMINAMATH_CALUDE_captain_age_is_your_age_l3406_340671


namespace NUMINAMATH_CALUDE_initial_population_proof_l3406_340603

def population_change (initial : ℕ) : ℕ := 
  let after_first_year := initial * 125 / 100
  (after_first_year * 70) / 100

theorem initial_population_proof : 
  ∃ (P : ℕ), population_change P = 363650 ∧ P = 415600 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_proof_l3406_340603


namespace NUMINAMATH_CALUDE_northern_walks_of_length_6_l3406_340626

/-- A northern walk is a path on a grid with the following properties:
  1. It starts at the origin.
  2. Each step is 1 unit north, east, or west.
  3. It never revisits a point.
  4. It has a specified length. -/
def NorthernWalk (length : ℕ) : Type := Unit

/-- Count the number of northern walks of a given length. -/
def countNorthernWalks (length : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 239 northern walks of length 6. -/
theorem northern_walks_of_length_6 : countNorthernWalks 6 = 239 := by sorry

end NUMINAMATH_CALUDE_northern_walks_of_length_6_l3406_340626


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3406_340601

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3406_340601


namespace NUMINAMATH_CALUDE_problem_solution_l3406_340682

theorem problem_solution : 
  let P : ℕ := 2007 / 5
  let Q : ℕ := P / 4
  let Y : ℕ := 2 * (P - Q)
  Y = 602 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3406_340682


namespace NUMINAMATH_CALUDE_sector_central_angle_l3406_340659

theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = (2/5) * Real.pi) :
  (2 * area) / (r^2) = Real.pi / 5 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3406_340659


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3406_340613

/-- Given a geometric sequence {a_n}, prove that if a_2 * a_6 = 36, then a_4 = ±6 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_prod : a 2 * a 6 = 36) : a 4 = 6 ∨ a 4 = -6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3406_340613


namespace NUMINAMATH_CALUDE_optimal_chair_removal_l3406_340644

/-- Proves that removing 36 chairs from 156 chairs initially arranged in rows of 12
    results in the optimal number of chairs for the assembly. -/
theorem optimal_chair_removal :
  let initial_chairs : ℕ := 156
  let chairs_per_row : ℕ := 12
  let students_attending : ℕ := 100
  let chairs_to_remove : ℕ := 36
  let remaining_chairs : ℕ := initial_chairs - chairs_to_remove
  
  (remaining_chairs ≥ students_attending) ∧
  (remaining_chairs % chairs_per_row = 0) ∧
  (remaining_chairs % 10 = 0) ∧
  (∀ n : ℕ, n < chairs_to_remove →
    ¬((initial_chairs - n ≥ students_attending) ∧
      (initial_chairs - n) % chairs_per_row = 0 ∧
      (initial_chairs - n) % 10 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_chair_removal_l3406_340644


namespace NUMINAMATH_CALUDE_complex_roots_count_l3406_340627

theorem complex_roots_count (z : ℂ) : 
  let θ := Complex.arg z
  (Complex.abs z = 1) →
  ((z ^ (7 * 6 * 5 * 4 * 3 * 2 * 1) - z ^ (6 * 5 * 4 * 3 * 2 * 1)).im = 0) →
  ((z ^ (6 * 5 * 4 * 3 * 2 * 1) - z ^ (5 * 4 * 3 * 2 * 1)).im = 0) →
  (0 ≤ θ) →
  (θ < 2 * Real.pi) →
  (Real.cos (4320 * θ) = 0 ∨ Real.sin (3360 * θ) = 0) →
  (Real.cos (420 * θ) = 0 ∨ Real.sin (300 * θ) = 0) →
  Nat := by sorry

end NUMINAMATH_CALUDE_complex_roots_count_l3406_340627


namespace NUMINAMATH_CALUDE_final_sum_is_eight_times_original_l3406_340664

theorem final_sum_is_eight_times_original (S a b : ℝ) (h : a + b = S) :
  (2 * (4 * a)) + (2 * (4 * b)) = 8 * S := by
  sorry

end NUMINAMATH_CALUDE_final_sum_is_eight_times_original_l3406_340664


namespace NUMINAMATH_CALUDE_divisibility_by_2008_l3406_340638

theorem divisibility_by_2008 (k m : ℕ) (h1 : ∃ (u : ℕ), k = 25 * (2 * u + 1)) (h2 : ∃ (v : ℕ), m = 25 * v) :
  2008 ∣ (2^k + 4^m) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_2008_l3406_340638


namespace NUMINAMATH_CALUDE_distance_after_5_hours_l3406_340669

/-- The distance between two people after walking in opposite directions for a given time -/
def distance_between (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem: The distance between two people walking in opposite directions for 5 hours,
    with speeds of 5 km/hr and 10 km/hr respectively, is 75 km -/
theorem distance_after_5_hours :
  distance_between 5 10 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_5_hours_l3406_340669


namespace NUMINAMATH_CALUDE_larger_number_proof_l3406_340643

theorem larger_number_proof (x y : ℝ) (h1 : x > y) (h2 : x - y = 3) (h3 : x^2 - y^2 = 39) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3406_340643


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l3406_340663

theorem cycle_gain_percent (cost_price selling_price : ℝ) (h1 : cost_price = 1500) (h2 : selling_price = 1620) :
  (selling_price - cost_price) / cost_price * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l3406_340663


namespace NUMINAMATH_CALUDE_operation_state_theorem_l3406_340609

/-- Represents the state of a student operating a computer -/
def OperationState := Fin 5 → Fin 5 → Bool

/-- The given condition that the product of diagonal elements is 0 -/
def DiagonalProductZero (a : OperationState) : Prop :=
  (a 0 0) && (a 1 1) && (a 2 2) && (a 3 3) && (a 4 4) = false

/-- At least one student is not operating their own computer -/
def AtLeastOneNotOwnComputer (a : OperationState) : Prop :=
  ∃ i : Fin 5, a i i = false

/-- 
If the product of diagonal elements in the operation state matrix is 0,
then at least one student is not operating their own computer.
-/
theorem operation_state_theorem (a : OperationState) :
  DiagonalProductZero a → AtLeastOneNotOwnComputer a := by
  sorry

end NUMINAMATH_CALUDE_operation_state_theorem_l3406_340609


namespace NUMINAMATH_CALUDE_fraction_ordering_l3406_340630

theorem fraction_ordering : 6/29 < 8/25 ∧ 8/25 < 10/31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l3406_340630


namespace NUMINAMATH_CALUDE_fifteen_equidistant_planes_spheres_l3406_340680

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of 5 points in 3D space -/
def FivePoints := Fin 5 → Point3D

/-- Predicate to check if 5 points lie on the same plane -/
def lieOnSamePlane (points : FivePoints) : Prop := sorry

/-- Predicate to check if 5 points lie on the same sphere -/
def lieOnSameSphere (points : FivePoints) : Prop := sorry

/-- Count of equidistant planes or spheres from 5 points -/
def countEquidistantPlanesSpheres (points : FivePoints) : ℕ := sorry

/-- Theorem stating that there are exactly 15 equidistant planes or spheres -/
theorem fifteen_equidistant_planes_spheres (points : FivePoints) 
  (h1 : ¬ lieOnSamePlane points) (h2 : ¬ lieOnSameSphere points) :
  countEquidistantPlanesSpheres points = 15 := by sorry

end NUMINAMATH_CALUDE_fifteen_equidistant_planes_spheres_l3406_340680


namespace NUMINAMATH_CALUDE_flour_difference_l3406_340677

theorem flour_difference : (7 : ℚ) / 8 - (5 : ℚ) / 6 = (1 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_flour_difference_l3406_340677


namespace NUMINAMATH_CALUDE_ratio_of_sum_equals_three_times_difference_l3406_340640

theorem ratio_of_sum_equals_three_times_difference
  (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0) (h4 : x + y = 3 * (x - y)) :
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sum_equals_three_times_difference_l3406_340640


namespace NUMINAMATH_CALUDE_bank_transfer_balance_l3406_340654

theorem bank_transfer_balance (initial_balance first_transfer second_transfer service_charge_rate : ℝ) 
  (h1 : initial_balance = 400)
  (h2 : first_transfer = 90)
  (h3 : second_transfer = 60)
  (h4 : service_charge_rate = 0.02)
  : initial_balance - (first_transfer + first_transfer * service_charge_rate + second_transfer * service_charge_rate) = 307 := by
  sorry

end NUMINAMATH_CALUDE_bank_transfer_balance_l3406_340654


namespace NUMINAMATH_CALUDE_find_n_l3406_340681

theorem find_n : ∃ n : ℤ, 3^4 - 13 = 4^3 + n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3406_340681


namespace NUMINAMATH_CALUDE_isabellas_hair_length_l3406_340612

/-- Given Isabella's initial hair length and growth, calculate her final hair length -/
theorem isabellas_hair_length 
  (initial_length : ℕ) 
  (growth : ℕ) 
  (h1 : initial_length = 18) 
  (h2 : growth = 6) : 
  initial_length + growth = 24 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_length_l3406_340612


namespace NUMINAMATH_CALUDE_line_intersections_l3406_340642

/-- The line equation y = -2x + 4 -/
def line_equation (x y : ℝ) : Prop := y = -2 * x + 4

/-- The point (x, y) lies on the x-axis -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The point (x, y) lies on the y-axis -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The line y = -2x + 4 intersects the x-axis at (2, 0) and the y-axis at (0, 4) -/
theorem line_intersections :
  (∃ (x y : ℝ), line_equation x y ∧ on_x_axis x y ∧ x = 2 ∧ y = 0) ∧
  (∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = 4) :=
sorry

end NUMINAMATH_CALUDE_line_intersections_l3406_340642


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3406_340653

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3406_340653


namespace NUMINAMATH_CALUDE_train_speed_problem_l3406_340670

theorem train_speed_problem (length1 length2 speed1 time : ℝ) 
  (h1 : length1 = 500)
  (h2 : length2 = 750)
  (h3 : speed1 = 60)
  (h4 : time = 44.99640028797697) : 
  ∃ speed2 : ℝ, 
    speed2 = 40 ∧ 
    (length1 + length2) / 1000 = (speed1 + speed2) * (time / 3600) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3406_340670


namespace NUMINAMATH_CALUDE_total_spending_l3406_340692

/-- The amount Ben spends -/
def ben_spent : ℝ := 50

/-- The amount David spends -/
def david_spent : ℝ := 37.5

/-- The difference in spending between Ben and David -/
def spending_difference : ℝ := 12.5

/-- The difference in cost per item between Ben and David -/
def cost_difference_per_item : ℝ := 0.25

theorem total_spending :
  ben_spent + david_spent = 87.5 ∧
  ben_spent - david_spent = spending_difference ∧
  ben_spent / david_spent = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_total_spending_l3406_340692


namespace NUMINAMATH_CALUDE_watch_correction_theorem_l3406_340641

/-- Represents the time difference between two dates in hours -/
def timeDifference : ℚ := 189

/-- Represents the daily loss rate of the watch in minutes per day -/
def dailyLossRate : ℚ := 13 / 4

/-- Calculates the positive correction in minutes to be added to the watch -/
def watchCorrection (timeDiff : ℚ) (lossRate : ℚ) : ℚ :=
  timeDiff * (lossRate / 24)

/-- Theorem stating that the watch correction is 2457/96 minutes -/
theorem watch_correction_theorem :
  watchCorrection timeDifference dailyLossRate = 2457 / 96 := by
  sorry

#eval watchCorrection timeDifference dailyLossRate

end NUMINAMATH_CALUDE_watch_correction_theorem_l3406_340641


namespace NUMINAMATH_CALUDE_snowdrift_depth_l3406_340684

theorem snowdrift_depth (initial_depth melted_depth third_day_depth fourth_day_depth final_depth : ℝ) :
  melted_depth = initial_depth / 2 →
  third_day_depth = melted_depth + 6 →
  fourth_day_depth = third_day_depth + 18 →
  final_depth = 34 →
  initial_depth = 20 :=
by sorry

end NUMINAMATH_CALUDE_snowdrift_depth_l3406_340684


namespace NUMINAMATH_CALUDE_island_with_2008_roads_sum_of_roads_formula_l3406_340666

def number_of_roads (n : ℕ) : ℕ := 55 + n.choose 2

def sum_of_roads (n : ℕ) : ℕ := 55 * n + (n + 1).choose 3

theorem island_with_2008_roads : ∃ n : ℕ, n > 0 ∧ number_of_roads n = 2008 := by sorry

theorem sum_of_roads_formula (n : ℕ) (h : n > 0) : 
  (Finset.range n).sum (λ k => number_of_roads (k + 1)) = sum_of_roads n := by sorry

end NUMINAMATH_CALUDE_island_with_2008_roads_sum_of_roads_formula_l3406_340666


namespace NUMINAMATH_CALUDE_f_monotone_increasing_intervals_l3406_340625

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

theorem f_monotone_increasing_intervals :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π - π / 12) (k * π + 5 * π / 12)) := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_intervals_l3406_340625


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3406_340678

theorem arithmetic_equality : 1357 + 3571 + 5713 - 7135 = 3506 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3406_340678


namespace NUMINAMATH_CALUDE_multiple_of_17_binary_properties_l3406_340633

/-- A function that returns the number of 1's in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- A function that returns the number of 0's in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem multiple_of_17_binary_properties (n : ℕ) 
  (h1 : n % 17 = 0) 
  (h2 : count_ones n = 3) : 
  (count_zeros n ≥ 6) ∧ 
  (count_zeros n = 7 → Even n) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_17_binary_properties_l3406_340633


namespace NUMINAMATH_CALUDE_shape_selections_count_l3406_340606

/-- Represents a regular hexagonal grid --/
structure HexagonalGrid :=
  (size : ℕ)

/-- Represents a shape that can be selected from the grid --/
structure Shape :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the number of ways to select a shape from a hexagonal grid --/
def selectionsCount (grid : HexagonalGrid) (shape : Shape) : ℕ :=
  sorry

/-- The number of distinct rotations for a shape in a hexagonal grid --/
def rotationsCount : ℕ := 3

/-- Theorem stating that there are 72 ways to select the given shape from the hexagonal grid --/
theorem shape_selections_count :
  ∀ (grid : HexagonalGrid) (shape : Shape),
  grid.size = 5 →  -- Assuming the grid size is 5 based on the problem description
  shape.width = 2 →  -- Assuming the shape width is 2 based on diagram b
  shape.height = 2 →  -- Assuming the shape height is 2 based on diagram b
  selectionsCount grid shape * rotationsCount = 72 :=
sorry

end NUMINAMATH_CALUDE_shape_selections_count_l3406_340606


namespace NUMINAMATH_CALUDE_lowest_two_digit_product_12_l3406_340617

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem lowest_two_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_lowest_two_digit_product_12_l3406_340617


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3406_340662

noncomputable def trapezoidArea (h α β : ℝ) : ℝ :=
  2 * h^2 * (Real.tan β + Real.tan α)

theorem isosceles_trapezoid_area
  (h α β : ℝ)
  (h_pos : h > 0)
  (α_pos : α > 0)
  (β_pos : β > 0)
  (α_lt_90 : α < π / 2)
  (β_lt_90 : β < π / 2)
  (h_eq : h = 2)
  (α_eq : α = 15 * π / 180)
  (β_eq : β = 75 * π / 180) :
  trapezoidArea h α β = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3406_340662


namespace NUMINAMATH_CALUDE_histogram_approximates_density_curve_l3406_340639

/-- Represents a sample frequency distribution histogram --/
structure SampleHistogram where
  sampleSize : ℕ
  groupInterval : ℝ
  distribution : ℝ → ℝ

/-- Represents a population density curve --/
def PopulationDensityCurve := ℝ → ℝ

/-- Measures the difference between a histogram and a density curve --/
def difference (h : SampleHistogram) (p : PopulationDensityCurve) : ℝ := sorry

theorem histogram_approximates_density_curve
  (h : ℕ → SampleHistogram)
  (p : PopulationDensityCurve)
  (hsize : ∀ ε > 0, ∃ N, ∀ n ≥ N, (h n).sampleSize > 1 / ε)
  (hinterval : ∀ ε > 0, ∃ N, ∀ n ≥ N, (h n).groupInterval < ε) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, difference (h n) p < ε :=
sorry

end NUMINAMATH_CALUDE_histogram_approximates_density_curve_l3406_340639


namespace NUMINAMATH_CALUDE_system_solution_l3406_340688

theorem system_solution (x y a : ℝ) : 
  (4 * x + y = a ∧ 2 * x + 5 * y = 3 * a ∧ x = 2) → a = 18 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3406_340688


namespace NUMINAMATH_CALUDE_parabola_points_order_l3406_340691

theorem parabola_points_order : 
  let y₁ : ℝ := -1/2 * (-2)^2 + 2 * (-2)
  let y₂ : ℝ := -1/2 * (-1)^2 + 2 * (-1)
  let y₃ : ℝ := -1/2 * 8^2 + 2 * 8
  y₃ < y₁ ∧ y₁ < y₂ := by sorry

end NUMINAMATH_CALUDE_parabola_points_order_l3406_340691


namespace NUMINAMATH_CALUDE_markers_in_packages_l3406_340634

theorem markers_in_packages (total_markers : ℕ) (markers_per_package : ℕ) (h1 : total_markers = 40) (h2 : markers_per_package = 5) :
  total_markers / markers_per_package = 8 :=
by sorry

end NUMINAMATH_CALUDE_markers_in_packages_l3406_340634


namespace NUMINAMATH_CALUDE_min_value_theorem_l3406_340674

theorem min_value_theorem (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + y = 1) :
  ∀ z : ℝ, z = (1 / (x + 1)) + (4 / y) → z ≥ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3406_340674


namespace NUMINAMATH_CALUDE_finite_rational_points_with_finite_orbit_l3406_340683

noncomputable def f (C : ℚ) (x : ℚ) : ℚ := x^2 - C

def has_finite_orbit (C : ℚ) (x : ℚ) : Prop :=
  ∃ (S : Finset ℚ), ∀ n : ℕ, (f C)^[n] x ∈ S

theorem finite_rational_points_with_finite_orbit (C : ℚ) :
  {x : ℚ | has_finite_orbit C x}.Finite :=
sorry

end NUMINAMATH_CALUDE_finite_rational_points_with_finite_orbit_l3406_340683


namespace NUMINAMATH_CALUDE_linear_system_solution_l3406_340624

theorem linear_system_solution (u v : ℚ) 
  (eq1 : 6 * u - 7 * v = 32)
  (eq2 : 3 * u + 5 * v = 1) : 
  2 * u + 3 * v = 64 / 51 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l3406_340624


namespace NUMINAMATH_CALUDE_min_value_theorem_l3406_340635

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + 2*y = 3) :
  (x^2 + 3*y) / (x*y) ≥ 2*Real.sqrt 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3406_340635


namespace NUMINAMATH_CALUDE_reciprocal_of_two_l3406_340647

theorem reciprocal_of_two : (2⁻¹ : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_two_l3406_340647


namespace NUMINAMATH_CALUDE_puppy_food_consumption_l3406_340616

/-- Represents the feeding schedule for a puppy over 4 weeks plus one day -/
structure PuppyFeeding where
  first_two_weeks : ℚ  -- Amount of food per day in first two weeks
  second_two_weeks : ℚ  -- Amount of food per day in second two weeks
  today : ℚ  -- Amount of food given today

/-- Calculates the total amount of food eaten by the puppy over 4 weeks plus one day -/
def total_food (feeding : PuppyFeeding) : ℚ :=
  feeding.first_two_weeks * 14 + feeding.second_two_weeks * 14 + feeding.today

/-- Theorem stating that the puppy will eat 25 cups of food over 4 weeks plus one day -/
theorem puppy_food_consumption :
  let feeding := PuppyFeeding.mk (3/4) 1 (1/2)
  total_food feeding = 25 := by sorry

end NUMINAMATH_CALUDE_puppy_food_consumption_l3406_340616


namespace NUMINAMATH_CALUDE_count_triples_product_million_l3406_340652

theorem count_triples_product_million : 
  (Finset.filter (fun (triple : ℕ × ℕ × ℕ) => triple.1 * triple.2.1 * triple.2.2 = 10^6) (Finset.product (Finset.range (10^6 + 1)) (Finset.product (Finset.range (10^6 + 1)) (Finset.range (10^6 + 1))))).card = 784 := by
  sorry

end NUMINAMATH_CALUDE_count_triples_product_million_l3406_340652


namespace NUMINAMATH_CALUDE_student_arrangement_l3406_340679

/-- Number of ways to arrange n distinct objects from m objects -/
def arrangement (m n : ℕ) : ℕ := sorry

/-- The number of students -/
def total_students : ℕ := 14

/-- The number of female students -/
def female_students : ℕ := 6

/-- The number of male students -/
def male_students : ℕ := 8

/-- The number of female students that must be grouped together -/
def grouped_females : ℕ := 4

/-- The number of gaps after arranging male students and grouped females -/
def gaps : ℕ := male_students + 1

theorem student_arrangement :
  arrangement male_students male_students *
  arrangement gaps (female_students - grouped_females) *
  arrangement grouped_females grouped_females =
  arrangement total_students total_students := by sorry

end NUMINAMATH_CALUDE_student_arrangement_l3406_340679


namespace NUMINAMATH_CALUDE_prank_combinations_l3406_340698

/-- The number of people available for the prank on each day of the week -/
def available_people : Fin 5 → ℕ
  | 0 => 2  -- Monday
  | 1 => 3  -- Tuesday
  | 2 => 6  -- Wednesday
  | 3 => 4  -- Thursday
  | 4 => 3  -- Friday

/-- The total number of different combinations of people for the prank across the week -/
def total_combinations : ℕ := (List.range 5).map available_people |>.prod

theorem prank_combinations :
  total_combinations = 432 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l3406_340698


namespace NUMINAMATH_CALUDE_co_molecular_weight_l3406_340699

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (carbon_count oxygen_count : ℕ) : ℝ :=
  carbon_count * carbon_weight + oxygen_count * oxygen_weight

/-- Theorem: The molecular weight of CO is 28.01 g/mol -/
theorem co_molecular_weight :
  molecular_weight 1 1 = 28.01 := by sorry

end NUMINAMATH_CALUDE_co_molecular_weight_l3406_340699


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3406_340615

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - x - 2 < 0} = Set.Ioo (-1 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3406_340615


namespace NUMINAMATH_CALUDE_egg_collection_ratio_l3406_340650

/-- 
Given:
- Benjamin collects 6 dozen eggs
- Trisha collects 4 dozen less than Benjamin
- The total eggs collected by all three is 26 dozen

Prove that the ratio of Carla's eggs to Benjamin's eggs is 3:1
-/
theorem egg_collection_ratio : 
  let benjamin_eggs : ℕ := 6
  let trisha_eggs : ℕ := benjamin_eggs - 4
  let total_eggs : ℕ := 26
  let carla_eggs : ℕ := total_eggs - benjamin_eggs - trisha_eggs
  (carla_eggs : ℚ) / benjamin_eggs = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_egg_collection_ratio_l3406_340650


namespace NUMINAMATH_CALUDE_clinton_belts_l3406_340649

/-- Proves that Clinton has 7 belts given the conditions -/
theorem clinton_belts :
  ∀ (shoes belts : ℕ),
  shoes = 14 →
  shoes = 2 * belts →
  belts = 7 := by
  sorry

end NUMINAMATH_CALUDE_clinton_belts_l3406_340649


namespace NUMINAMATH_CALUDE_f_strictly_decreasing_iff_a_in_range_l3406_340690

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 4) * x + 3 * a

theorem f_strictly_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) * (x₁ - x₂) < 0) ↔
  0 < a ∧ a ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_f_strictly_decreasing_iff_a_in_range_l3406_340690


namespace NUMINAMATH_CALUDE_expression_evaluation_l3406_340661

theorem expression_evaluation : 2 + 3 * 4 - 1^2 + 6 / 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3406_340661


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3406_340632

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 5 → x > 3) ∧ 
  (∃ x : ℝ, x > 3 ∧ ¬(x > 5)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3406_340632


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l3406_340605

def daily_incomes : List ℝ := [250, 400, 750, 400, 500]

theorem cab_driver_average_income :
  (daily_incomes.sum / daily_incomes.length : ℝ) = 460 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l3406_340605


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3406_340686

/-- A cubic function with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

/-- Theorem stating the properties of the cubic function f -/
theorem cubic_function_properties :
  ∃ (a b : ℝ),
    (f' a b (-2/3) = 0 ∧ f' a b 1 = 0) ∧ 
    (a = -1/2 ∧ b = -2) ∧
    (∃ (t : ℝ), 
      (t = 0 ∧ 2 * 0 + f a b 0 - 1 = 0) ∨
      (t = 1/4 ∧ 33 * (1/4) + 16 * (f a b (1/4)) - 16 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3406_340686


namespace NUMINAMATH_CALUDE_milk_for_nine_cookies_l3406_340628

-- Define the relationship between cookies and quarts of milk
def milk_for_cookies (cookies : ℕ) : ℚ :=
  (3 : ℚ) * cookies / 18

-- Define the conversion from quarts to pints
def quarts_to_pints (quarts : ℚ) : ℚ :=
  2 * quarts

theorem milk_for_nine_cookies :
  quarts_to_pints (milk_for_cookies 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_milk_for_nine_cookies_l3406_340628


namespace NUMINAMATH_CALUDE_chord_midpoint_theorem_l3406_340629

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the chord equation
def chord_equation (x y : ℝ) : Prop := x + 4*y - 5 = 0

-- Theorem statement
theorem chord_midpoint_theorem :
  ∀ (A B : ℝ × ℝ),
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 →
  (A.1 + B.1) / 2 = P.1 ∧ (A.2 + B.2) / 2 = P.2 →
  ∀ (x y : ℝ), chord_equation x y ↔ ∃ t : ℝ, (x, y) = (1 - t, 1 + t/4) :=
sorry

end NUMINAMATH_CALUDE_chord_midpoint_theorem_l3406_340629


namespace NUMINAMATH_CALUDE_continuous_function_zero_on_interval_l3406_340687

theorem continuous_function_zero_on_interval
  (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_eq : ∀ x, f (2 * x^2 - 1) = 2 * x * f x) :
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_zero_on_interval_l3406_340687


namespace NUMINAMATH_CALUDE_harry_apples_l3406_340637

/-- Harry's apple problem -/
theorem harry_apples : ∀ (initial_apples bought_apples friends apples_per_friend : ℕ),
  initial_apples = 79 →
  bought_apples = 5 →
  friends = 7 →
  apples_per_friend = 3 →
  initial_apples + bought_apples - friends * apples_per_friend = 63 := by
  sorry

end NUMINAMATH_CALUDE_harry_apples_l3406_340637


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l3406_340631

theorem min_value_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.log x - a * Real.exp x - b + 1 ≤ 0) →
  ∃ m : ℝ, m = 0 ∧ (∀ a' b' : ℝ, (∀ x : ℝ, x > 0 → Real.log x - a' * Real.exp x - b' + 1 ≤ 0) → a' + b' ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l3406_340631


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l3406_340622

theorem quadratic_root_implies_a (a : ℝ) :
  (∃ x : ℝ, x^2 - a = 0) ∧ (2^2 - a = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l3406_340622


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3406_340685

/-- A rectangular prism with different length, width, and height. -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height
  different_dimensions : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- The number of faces in a rectangular prism. -/
def faces_count : ℕ := 6

/-- The number of edges in a rectangular prism. -/
def edges_count : ℕ := 12

/-- The number of diagonals in a rectangular prism. -/
def diagonals_count (rp : RectangularPrism) : ℕ := 16

/-- Theorem: A rectangular prism with different length, width, and height has exactly 16 diagonals. -/
theorem rectangular_prism_diagonals (rp : RectangularPrism) : 
  diagonals_count rp = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3406_340685


namespace NUMINAMATH_CALUDE_rectangle_circles_l3406_340694

theorem rectangle_circles (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circles_l3406_340694


namespace NUMINAMATH_CALUDE_complement_of_B_in_A_l3406_340676

def A : Set ℕ := {2, 3, 4}
def B (a : ℕ) : Set ℕ := {a + 2, a}

theorem complement_of_B_in_A (a : ℕ) (h : A ∩ B a = B a) : 
  (A \ B a) = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_B_in_A_l3406_340676


namespace NUMINAMATH_CALUDE_divisor_sum_840_l3406_340667

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem divisor_sum_840 (i j : ℕ) (h : i > 0 ∧ j > 0) :
  sum_of_divisors (2^i * 3^j) = 840 → i + j = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_840_l3406_340667


namespace NUMINAMATH_CALUDE_absolute_value_fraction_sum_not_one_l3406_340610

theorem absolute_value_fraction_sum_not_one (a b : ℝ) (h : a * b ≠ 0) :
  |a| / a + |b| / b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_sum_not_one_l3406_340610


namespace NUMINAMATH_CALUDE_rotation_of_vector_l3406_340645

/-- Given points O and P in a 2D Cartesian plane, and Q obtained by rotating OP counterclockwise by 3π/4, 
    prove that Q has the specified coordinates. -/
theorem rotation_of_vector (O P Q : ℝ × ℝ) : 
  O = (0, 0) → 
  P = (6, 8) → 
  Q = (Real.cos (3 * Real.pi / 4) * (P.1 - O.1) - Real.sin (3 * Real.pi / 4) * (P.2 - O.2) + O.1,
       Real.sin (3 * Real.pi / 4) * (P.1 - O.1) + Real.cos (3 * Real.pi / 4) * (P.2 - O.2) + O.2) →
  Q = (-7 * Real.sqrt 2, -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_rotation_of_vector_l3406_340645


namespace NUMINAMATH_CALUDE_circle_equation_range_l3406_340672

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + m*x - 2*y + 3 = 0

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  m < -2 * Real.sqrt 2 ∨ m > 2 * Real.sqrt 2

-- Theorem statement
theorem circle_equation_range :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) ↔ m_range m :=
sorry

end NUMINAMATH_CALUDE_circle_equation_range_l3406_340672


namespace NUMINAMATH_CALUDE_remainder_theorem_l3406_340696

theorem remainder_theorem : (9 * 7^18 + 2^18) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3406_340696


namespace NUMINAMATH_CALUDE_rectangular_field_width_l3406_340614

/-- The width of a rectangular field given its length-to-width ratio and perimeter -/
def field_width (length_width_ratio : ℚ) (perimeter : ℚ) : ℚ :=
  perimeter / (2 * (length_width_ratio + 1))

theorem rectangular_field_width :
  field_width (7/5) 288 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l3406_340614


namespace NUMINAMATH_CALUDE_third_circle_radius_l3406_340621

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) : 
  r₁ = 25 → r₂ = 40 → 
  π * r₃^2 = (π * r₂^2 - π * r₁^2) / 2 →
  r₃ = 15 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_third_circle_radius_l3406_340621
