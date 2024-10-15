import Mathlib

namespace NUMINAMATH_CALUDE_ab_nonpositive_l1818_181854

theorem ab_nonpositive (a b : ℝ) (h : 2011 * a + 2012 * b = 0) : a * b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_nonpositive_l1818_181854


namespace NUMINAMATH_CALUDE_power_two_ge_square_l1818_181860

theorem power_two_ge_square (n : ℕ) : 2^n ≥ n^2 ↔ n ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_power_two_ge_square_l1818_181860


namespace NUMINAMATH_CALUDE_total_sugar_third_layer_is_correct_l1818_181855

/-- The amount of sugar needed for the smallest layer of the cake -/
def smallest_layer_sugar : ℝ := 2

/-- The size multiplier for the second layer compared to the first -/
def second_layer_multiplier : ℝ := 1.5

/-- The size multiplier for the third layer compared to the second -/
def third_layer_multiplier : ℝ := 2.5

/-- The percentage of sugar loss while baking each layer -/
def sugar_loss_percentage : ℝ := 0.15

/-- Calculates the total cups of sugar needed for the third layer -/
def total_sugar_third_layer : ℝ :=
  smallest_layer_sugar * second_layer_multiplier * third_layer_multiplier * (1 + sugar_loss_percentage)

/-- Theorem stating that the total sugar needed for the third layer is 8.625 cups -/
theorem total_sugar_third_layer_is_correct :
  total_sugar_third_layer = 8.625 := by
  sorry

end NUMINAMATH_CALUDE_total_sugar_third_layer_is_correct_l1818_181855


namespace NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l1818_181861

/-- Given a curve C in the Cartesian coordinate system with polar equation ρ = 2cosθ - 4sinθ,
    prove that its Cartesian equation is (x - 2)² - 15y² = 68 - (y + 8)² -/
theorem polar_to_cartesian_equivalence (x y ρ θ : ℝ) :
  ρ = 2 * Real.cos θ - 4 * Real.sin θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  (x - 2)^2 - 15 * y^2 = 68 - (y + 8)^2 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l1818_181861


namespace NUMINAMATH_CALUDE_star_op_identity_l1818_181824

/-- Define the * operation on ordered pairs of real numbers -/
def star_op (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, a * d + b * c)

/-- Theorem: If (a, b) * (x, y) = (a, b) and a² ≠ b², then (x, y) = (1, 0) -/
theorem star_op_identity {a b x y : ℝ} (h : a^2 ≠ b^2) :
  star_op a b x y = (a, b) → (x, y) = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_star_op_identity_l1818_181824


namespace NUMINAMATH_CALUDE_base8_573_equals_379_l1818_181835

/-- Converts a base-8 number to base 10 --/
def base8_to_base10 (a b c : ℕ) : ℕ := a * 8^2 + b * 8^1 + c * 8^0

/-- The base-8 number 573₈ is equal to 379 in base 10 --/
theorem base8_573_equals_379 : base8_to_base10 5 7 3 = 379 := by
  sorry

end NUMINAMATH_CALUDE_base8_573_equals_379_l1818_181835


namespace NUMINAMATH_CALUDE_sqrt_twelve_less_than_four_l1818_181801

theorem sqrt_twelve_less_than_four : Real.sqrt 12 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_less_than_four_l1818_181801


namespace NUMINAMATH_CALUDE_distance_to_work_is_27_l1818_181813

/-- The distance to work in miles -/
def distance_to_work : ℝ := 27

/-- The total commute time in hours -/
def total_commute_time : ℝ := 1.5

/-- The average speed to work in mph -/
def speed_to_work : ℝ := 45

/-- The average speed from work in mph -/
def speed_from_work : ℝ := 30

/-- Theorem stating that the distance to work is 27 miles -/
theorem distance_to_work_is_27 :
  distance_to_work = 27 ∧
  total_commute_time = distance_to_work / speed_to_work + distance_to_work / speed_from_work :=
by sorry

end NUMINAMATH_CALUDE_distance_to_work_is_27_l1818_181813


namespace NUMINAMATH_CALUDE_money_percentage_difference_l1818_181827

/-- The problem statement about Kim, Sal, and Phil's money --/
theorem money_percentage_difference 
  (sal_phil_total : ℝ)
  (kim_money : ℝ)
  (sal_percent_less : ℝ)
  (h1 : sal_phil_total = 1.80)
  (h2 : kim_money = 1.12)
  (h3 : sal_percent_less = 20) :
  let phil_money := sal_phil_total / (2 - sal_percent_less / 100)
  let sal_money := phil_money * (1 - sal_percent_less / 100)
  let percentage_difference := (kim_money - sal_money) / sal_money * 100
  percentage_difference = 40 := by
sorry

end NUMINAMATH_CALUDE_money_percentage_difference_l1818_181827


namespace NUMINAMATH_CALUDE_cycle_cost_price_l1818_181853

-- Define the cost price and selling price
def cost_price : ℝ := 1600
def selling_price : ℝ := 1360

-- Define the loss percentage
def loss_percentage : ℝ := 15

-- Theorem statement
theorem cycle_cost_price : 
  selling_price = cost_price * (1 - loss_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_cycle_cost_price_l1818_181853


namespace NUMINAMATH_CALUDE_loaf_has_twelve_slices_l1818_181847

/-- Represents a household with bread consumption patterns. -/
structure Household where
  members : ℕ
  breakfast_slices : ℕ
  snack_slices : ℕ
  loaves : ℕ
  days : ℕ

/-- Calculates the number of slices in a loaf of bread for a given household. -/
def slices_per_loaf (h : Household) : ℕ :=
  (h.members * (h.breakfast_slices + h.snack_slices) * h.days) / h.loaves

/-- Theorem stating that for the given household, a loaf of bread contains 12 slices. -/
theorem loaf_has_twelve_slices : 
  slices_per_loaf { members := 4, breakfast_slices := 3, snack_slices := 2, loaves := 5, days := 3 } = 12 := by
  sorry

end NUMINAMATH_CALUDE_loaf_has_twelve_slices_l1818_181847


namespace NUMINAMATH_CALUDE_complement_event_probability_formula_l1818_181830

/-- The probability of the complement event Ā occurring k times in n trials, 
    given that the probability of event A is p -/
def complementEventProbability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * (1 - p) ^ k * p ^ (n - k)

/-- Theorem stating that the probability of the complement event Ā occurring k times 
    in n trials is equal to ⁽ᵏⁿ)(1-p)ᵏp⁽ⁿ⁻ᵏ⁾, given that the probability of event A is p -/
theorem complement_event_probability_formula (n k : ℕ) (p : ℝ) 
    (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : k ≤ n) : 
  complementEventProbability n k p = Nat.choose n k * (1 - p) ^ k * p ^ (n - k) := by
  sorry

#check complement_event_probability_formula

end NUMINAMATH_CALUDE_complement_event_probability_formula_l1818_181830


namespace NUMINAMATH_CALUDE_triangle_angle_sum_impossibility_l1818_181817

theorem triangle_angle_sum_impossibility (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬((α + β < 120 ∧ β + γ < 120 ∧ γ + α < 120) ∨
    (α + β > 120 ∧ β + γ > 120 ∧ γ + α > 120)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_impossibility_l1818_181817


namespace NUMINAMATH_CALUDE_walk_distance_proof_l1818_181866

/-- Calculates the distance traveled given a constant speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that walking at 4 miles per hour for 2 hours results in a distance of 8 miles -/
theorem walk_distance_proof :
  let speed : ℝ := 4
  let time : ℝ := 2
  distance_traveled speed time = 8 := by sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l1818_181866


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1818_181815

/-- The volume of a cone formed by rolling up a half-sector of a circle -/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let slant_height := r
  let base_circumference := r * π
  let base_radius := base_circumference / (2 * π)
  let cone_height := Real.sqrt (slant_height ^ 2 - base_radius ^ 2)
  (1 / 3) * π * base_radius ^ 2 * cone_height = 9 * π * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1818_181815


namespace NUMINAMATH_CALUDE_min_value_of_f_l1818_181818

def f (x : ℝ) := x^2 + 2

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1818_181818


namespace NUMINAMATH_CALUDE_no_k_exists_for_prime_and_binomial_cong_l1818_181822

theorem no_k_exists_for_prime_and_binomial_cong (k : ℕ+) (p : ℕ) : 
  p = 6 * k + 1 → 
  Nat.Prime p → 
  (Nat.choose (3 * k) k : ZMod p) = 1 → 
  False := by sorry

end NUMINAMATH_CALUDE_no_k_exists_for_prime_and_binomial_cong_l1818_181822


namespace NUMINAMATH_CALUDE_total_cost_equals_12_46_l1818_181897

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 249/100

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 187/100

/-- The number of sandwiches -/
def num_sandwiches : ℕ := 2

/-- The number of sodas -/
def num_sodas : ℕ := 4

/-- The total cost of the order -/
def total_cost : ℚ := num_sandwiches * sandwich_cost + num_sodas * soda_cost

theorem total_cost_equals_12_46 : total_cost = 1246/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_equals_12_46_l1818_181897


namespace NUMINAMATH_CALUDE_cube_face_sum_l1818_181887

/-- Represents the six positive integers on the faces of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- The sum of the products of the three numbers adjacent to each vertex -/
def vertexSum (faces : CubeFaces) : ℕ :=
  faces.a * faces.b * faces.c +
  faces.a * faces.e * faces.c +
  faces.a * faces.b * faces.f +
  faces.a * faces.e * faces.f +
  faces.d * faces.b * faces.c +
  faces.d * faces.e * faces.c +
  faces.d * faces.b * faces.f +
  faces.d * faces.e * faces.f

/-- The sum of the numbers on the faces -/
def faceSum (faces : CubeFaces) : ℕ :=
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f

theorem cube_face_sum (faces : CubeFaces) :
  vertexSum faces = 1386 → faceSum faces = 38 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_sum_l1818_181887


namespace NUMINAMATH_CALUDE_trick_or_treat_total_l1818_181877

/-- Calculates the total number of treats received by children while trick-or-treating. -/
def total_treats (num_children : ℕ) (hours_out : ℕ) (houses_per_hour : ℕ) (treats_per_child_per_house : ℕ) : ℕ :=
  num_children * hours_out * houses_per_hour * treats_per_child_per_house

/-- Proves that given the specific conditions, the total number of treats is 180. -/
theorem trick_or_treat_total (num_children : ℕ) (hours_out : ℕ) (houses_per_hour : ℕ) (treats_per_child_per_house : ℕ) 
    (h1 : num_children = 3)
    (h2 : hours_out = 4)
    (h3 : houses_per_hour = 5)
    (h4 : treats_per_child_per_house = 3) :
  total_treats num_children hours_out houses_per_hour treats_per_child_per_house = 180 := by
  sorry

end NUMINAMATH_CALUDE_trick_or_treat_total_l1818_181877


namespace NUMINAMATH_CALUDE_hair_cut_total_l1818_181807

theorem hair_cut_total (day1 : Float) (day2 : Float) (h1 : day1 = 0.38) (h2 : day2 = 0.5) :
  day1 + day2 = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_total_l1818_181807


namespace NUMINAMATH_CALUDE_quadratic_condition_for_x_equals_one_l1818_181840

theorem quadratic_condition_for_x_equals_one :
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧
  ¬(∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_for_x_equals_one_l1818_181840


namespace NUMINAMATH_CALUDE_symmetrical_line_over_x_axis_l1818_181864

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- Reflects a line over the X axis --/
def reflect_over_x_axis (l : Line) : Line :=
  { a := l.a,
    b := -l.b,
    c := -l.c,
    eq := sorry }

theorem symmetrical_line_over_x_axis :
  let original_line : Line := { a := 1, b := -2, c := 3, eq := sorry }
  let reflected_line := reflect_over_x_axis original_line
  reflected_line.a = 1 ∧ reflected_line.b = 2 ∧ reflected_line.c = -3 :=
by sorry

end NUMINAMATH_CALUDE_symmetrical_line_over_x_axis_l1818_181864


namespace NUMINAMATH_CALUDE_henrys_cd_collection_l1818_181834

theorem henrys_cd_collection (country rock classical : ℕ) : 
  country = rock + 3 →
  rock = 2 * classical →
  country = 23 →
  classical = 10 := by
sorry

end NUMINAMATH_CALUDE_henrys_cd_collection_l1818_181834


namespace NUMINAMATH_CALUDE_problems_solved_l1818_181857

theorem problems_solved (first last : ℕ) (h : first = 70 ∧ last = 125) : last - first + 1 = 56 := by
  sorry

end NUMINAMATH_CALUDE_problems_solved_l1818_181857


namespace NUMINAMATH_CALUDE_sin_210_degrees_l1818_181829

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l1818_181829


namespace NUMINAMATH_CALUDE_equation_condition_l1818_181886

theorem equation_condition (a d e : ℕ) : 
  (0 < a ∧ a < 10) → (0 < d ∧ d < 10) → (0 < e ∧ e < 10) →
  ((10 * a + d) * (10 * a + e) = 100 * a^2 + 110 * a + d * e ↔ d + e = 11) :=
by sorry

end NUMINAMATH_CALUDE_equation_condition_l1818_181886


namespace NUMINAMATH_CALUDE_theater_admission_revenue_l1818_181825

/-- Calculates the total amount collected from theater admissions --/
def total_amount_collected (adult_price child_price : ℚ) (total_attendance children_attendance : ℕ) : ℚ :=
  let adults_attendance := total_attendance - children_attendance
  let adult_revenue := adult_price * adults_attendance
  let child_revenue := child_price * children_attendance
  adult_revenue + child_revenue

/-- Theorem stating that the total amount collected is $140 given the specified conditions --/
theorem theater_admission_revenue :
  total_amount_collected (60/100) (25/100) 280 80 = 140 := by
  sorry

end NUMINAMATH_CALUDE_theater_admission_revenue_l1818_181825


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1818_181839

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1818_181839


namespace NUMINAMATH_CALUDE_square_field_area_l1818_181876

theorem square_field_area (diagonal : ℝ) (h : diagonal = 16) : 
  let side := diagonal / Real.sqrt 2
  let area := side ^ 2
  area = 128 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l1818_181876


namespace NUMINAMATH_CALUDE_kiran_currency_notes_l1818_181869

/-- Represents the currency denominations in Rupees --/
inductive Denomination
  | fifty : Denomination
  | hundred : Denomination

/-- Represents the total amount and number of notes for each denomination --/
structure CurrencyNotes where
  total_amount : ℕ
  fifty_amount : ℕ
  fifty_count : ℕ
  hundred_count : ℕ

/-- Calculates the total number of currency notes --/
def total_notes (c : CurrencyNotes) : ℕ :=
  c.fifty_count + c.hundred_count

/-- Theorem stating that given the conditions, Kiran has 85 currency notes in total --/
theorem kiran_currency_notes :
  ∀ (c : CurrencyNotes),
    c.total_amount = 5000 →
    c.fifty_amount = 3500 →
    c.fifty_count = c.fifty_amount / 50 →
    c.hundred_count = (c.total_amount - c.fifty_amount) / 100 →
    total_notes c = 85 := by
  sorry

end NUMINAMATH_CALUDE_kiran_currency_notes_l1818_181869


namespace NUMINAMATH_CALUDE_f_750_value_l1818_181806

/-- A function satisfying f(xy) = f(x)/y for positive reals -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 ∧ y > 0 → f (x * y) = f x / y

theorem f_750_value (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 1000 = 4) :
  f 750 = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_f_750_value_l1818_181806


namespace NUMINAMATH_CALUDE_ali_seashells_l1818_181885

theorem ali_seashells (initial : ℕ) (given_to_friends : ℕ) (left_after_selling : ℕ) :
  initial = 180 →
  given_to_friends = 40 →
  left_after_selling = 55 →
  ∃ (given_to_brothers : ℕ),
    given_to_brothers = 30 ∧
    2 * left_after_selling = initial - given_to_friends - given_to_brothers :=
by sorry

end NUMINAMATH_CALUDE_ali_seashells_l1818_181885


namespace NUMINAMATH_CALUDE_inequality_and_nonexistence_l1818_181881

theorem inequality_and_nonexistence (x y z : ℝ) :
  (x^2 + 2*y^2 + 3*z^2 ≥ Real.sqrt 3 * (x*y + y*z + z*x)) ∧
  (∀ k > Real.sqrt 3, ∃ x y z : ℝ, x^2 + 2*y^2 + 3*z^2 < k*(x*y + y*z + z*x)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_nonexistence_l1818_181881


namespace NUMINAMATH_CALUDE_equation_solutions_l1818_181814

theorem equation_solutions :
  ∀ a b : ℕ, a^2 = b * (b + 7) → (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1818_181814


namespace NUMINAMATH_CALUDE_fixed_point_on_symmetric_line_l1818_181837

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define symmetry about a point
def symmetric_about (l1 l2 : Line) (p : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), (y = l1.slope * (x - 4)) →
    ∃ (x' y' : ℝ), (y' = l2.slope * x' + l2.intercept) ∧
      (x + x') / 2 = p.1 ∧ (y + y') / 2 = p.2

-- Define a point being on a line
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

-- Theorem statement
theorem fixed_point_on_symmetric_line (k : ℝ) :
  ∀ (l2 : Line), symmetric_about ⟨k, -4*k⟩ l2 (2, 1) →
    point_on_line (0, 2) l2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_symmetric_line_l1818_181837


namespace NUMINAMATH_CALUDE_divisibility_in_sequence_l1818_181836

theorem divisibility_in_sequence (a : ℕ → ℕ) 
  (h : ∀ n ∈ Finset.range 3029, 2 * a (n + 2) = a (n + 1) + 4 * a n) :
  ∃ i ∈ Finset.range 3031, 2^2020 ∣ a i := by
sorry

end NUMINAMATH_CALUDE_divisibility_in_sequence_l1818_181836


namespace NUMINAMATH_CALUDE_probability_x_lt_2y_is_one_sixth_l1818_181858

/-- A rectangle in the 2D plane -/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  x_min_le_x_max : x_min ≤ x_max
  y_min_le_y_max : y_min ≤ y_max

/-- The probability that a randomly chosen point (x,y) from the given rectangle satisfies x < 2y -/
def probability_x_lt_2y (r : Rectangle) : ℝ :=
  sorry

/-- The specific rectangle with vertices (0,0), (6,0), (6,1), and (0,1) -/
def specific_rectangle : Rectangle :=
  { x_min := 0
  , x_max := 6
  , y_min := 0
  , y_max := 1
  , x_min_le_x_max := by norm_num
  , y_min_le_y_max := by norm_num
  }

/-- Theorem stating that the probability of x < 2y for a randomly chosen point
    in the specific rectangle is 1/6 -/
theorem probability_x_lt_2y_is_one_sixth :
  probability_x_lt_2y specific_rectangle = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_lt_2y_is_one_sixth_l1818_181858


namespace NUMINAMATH_CALUDE_inequality_proof_l1818_181896

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1818_181896


namespace NUMINAMATH_CALUDE_power_mod_eleven_l1818_181812

theorem power_mod_eleven : 5^2023 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l1818_181812


namespace NUMINAMATH_CALUDE_sum_of_digits_of_second_smallest_divisible_by_all_less_than_8_l1818_181800

def is_divisible_by_all_less_than_8 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 8 → n % k = 0

def second_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  P n ∧ ∃ m : ℕ, P m ∧ m < n ∧ ∀ k : ℕ, P k → k = m ∨ n ≤ k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_second_smallest_divisible_by_all_less_than_8 :
  ∃ N : ℕ, second_smallest is_divisible_by_all_less_than_8 N ∧ sum_of_digits N = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_second_smallest_divisible_by_all_less_than_8_l1818_181800


namespace NUMINAMATH_CALUDE_calculate_total_profit_l1818_181831

/-- Given the investments of three partners and the profit share of one partner,
    calculate the total profit of the business. -/
theorem calculate_total_profit
  (a_investment b_investment c_investment : ℕ)
  (c_profit_share : ℕ)
  (h1 : a_investment = 5000)
  (h2 : b_investment = 15000)
  (h3 : c_investment = 30000)
  (h4 : c_profit_share = 3000) :
  (a_investment + b_investment + c_investment) * c_profit_share
  / c_investment = 5000 :=
sorry

end NUMINAMATH_CALUDE_calculate_total_profit_l1818_181831


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_m_range_l1818_181821

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 - 4*m*x + 2*m + 6 = 0}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_nonempty_implies_m_range (m : ℝ) :
  (A m ∩ B).Nonempty → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_m_range_l1818_181821


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1818_181863

theorem unique_triple_solution : 
  ∃! (x y z : ℕ+), 
    (¬(3 ∣ z ∧ y ∣ z)) ∧ 
    (Nat.Prime y) ∧ 
    (x^3 - y^3 = z^2) ∧
    x = 8 ∧ y = 7 ∧ z = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1818_181863


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1818_181895

/-- Given two vectors a and b in ℝ², where a = (2, m) and b = (l, -2),
    if a is parallel to a + 2b, then m = -4. -/
theorem parallel_vectors_m_value
  (m l : ℝ)
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h1 : a = (2, m))
  (h2 : b = (l, -2))
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ a = k • (a + 2 • b)) :
  m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1818_181895


namespace NUMINAMATH_CALUDE_refrigerator_temp_difference_l1818_181820

/-- The temperature difference between two compartments in a refrigerator -/
def temperature_difference (refrigeration_temp freezer_temp : ℤ) : ℤ :=
  refrigeration_temp - freezer_temp

/-- Theorem stating the temperature difference between specific compartments -/
theorem refrigerator_temp_difference :
  temperature_difference 3 (-10) = 13 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_temp_difference_l1818_181820


namespace NUMINAMATH_CALUDE_magic_square_x_value_l1818_181889

/-- Represents a 3x3 multiplicative magic square --/
structure MagicSquare where
  a11 : ℝ
  a12 : ℝ
  a13 : ℝ
  a21 : ℝ
  a22 : ℝ
  a23 : ℝ
  a31 : ℝ
  a32 : ℝ
  a33 : ℝ
  positive : ∀ i j, (i, j) ∈ [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)] → 
    match (i, j) with
    | (1, 1) => a11 > 0
    | (1, 2) => a12 > 0
    | (1, 3) => a13 > 0
    | (2, 1) => a21 > 0
    | (2, 2) => a22 > 0
    | (2, 3) => a23 > 0
    | (3, 1) => a31 > 0
    | (3, 2) => a32 > 0
    | (3, 3) => a33 > 0
    | _ => False
  magic : a11 * a12 * a13 = a21 * a22 * a23 ∧
          a11 * a12 * a13 = a31 * a32 * a33 ∧
          a11 * a12 * a13 = a11 * a21 * a31 ∧
          a11 * a12 * a13 = a12 * a22 * a32 ∧
          a11 * a12 * a13 = a13 * a23 * a33 ∧
          a11 * a12 * a13 = a11 * a22 * a33 ∧
          a11 * a12 * a13 = a13 * a22 * a31

theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.a11 = 5)
  (h2 : ms.a21 = 4)
  (h3 : ms.a33 = 20) :
  ms.a12 = 100 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_x_value_l1818_181889


namespace NUMINAMATH_CALUDE_group_transfer_equation_l1818_181842

/-- 
Given two groups of people, with 22 in the first group and 26 in the second group,
this theorem proves the equation for the number of people that should be transferred
from the second group to the first group so that the first group has twice the number
of people as the second group.
-/
theorem group_transfer_equation (x : ℤ) : (22 + x = 2 * (26 - x)) ↔ 
  (22 + x = 2 * (26 - x) ∧ 
   22 + x > 0 ∧ 
   26 - x > 0) := by
  sorry

end NUMINAMATH_CALUDE_group_transfer_equation_l1818_181842


namespace NUMINAMATH_CALUDE_job_completion_time_l1818_181816

/-- Given that m people can complete a job in d days, 
    prove that (m + r) people can complete the same job in md / (m + r) days. -/
theorem job_completion_time 
  (m d r : ℕ) (m_pos : m > 0) (d_pos : d > 0) (r_pos : r > 0) : 
  let n := (m * d) / (m + r)
  ∃ (W : ℝ), W > 0 ∧ W / (m * d) = W / ((m + r) * n) :=
sorry

end NUMINAMATH_CALUDE_job_completion_time_l1818_181816


namespace NUMINAMATH_CALUDE_fifty_cent_items_count_l1818_181892

theorem fifty_cent_items_count (a b c : ℕ) : 
  a + b + c = 50 →
  50 * a + 400 * b + 500 * c = 10000 →
  a = 40 :=
by sorry

end NUMINAMATH_CALUDE_fifty_cent_items_count_l1818_181892


namespace NUMINAMATH_CALUDE_seed_germination_probability_l1818_181872

/-- The probability of success in a single trial -/
def p : ℝ := 0.9

/-- The probability of failure in a single trial -/
def q : ℝ := 1 - p

/-- The number of trials -/
def n : ℕ := 4

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Probability of exactly k successes in n trials -/
def P (k : ℕ) : ℝ := (binomial n k : ℝ) * p^k * q^(n - k)

theorem seed_germination_probability :
  (P 3 = 0.2916) ∧ (P 3 + P 4 = 0.9477) := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_probability_l1818_181872


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l1818_181852

theorem reciprocal_of_sum : (1 / (1/2 + 1/3) : ℚ) = 6/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l1818_181852


namespace NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_150_greatest_common_multiple_10_15_under_150_is_120_l1818_181868

theorem greatest_common_multiple_10_15_under_150 : ℕ → Prop :=
  fun n =>
    (∃ k : ℕ, n = 10 * k) ∧
    (∃ k : ℕ, n = 15 * k) ∧
    n < 150 ∧
    ∀ m : ℕ, (∃ k : ℕ, m = 10 * k) ∧ (∃ k : ℕ, m = 15 * k) ∧ m < 150 → m ≤ n →
    n = 120

-- The proof goes here
theorem greatest_common_multiple_10_15_under_150_is_120 :
  greatest_common_multiple_10_15_under_150 120 :=
sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_150_greatest_common_multiple_10_15_under_150_is_120_l1818_181868


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1818_181865

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1818_181865


namespace NUMINAMATH_CALUDE_divisibility_by_three_l1818_181867

theorem divisibility_by_three (a : ℤ) : ¬(3 ∣ a) → (3 ∣ (5 * a^2 + 1)) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l1818_181867


namespace NUMINAMATH_CALUDE_alice_bob_meet_l1818_181871

/-- The number of points on the circle -/
def n : ℕ := 15

/-- Alice's movement per turn (clockwise) -/
def alice_move : ℕ := 7

/-- Bob's movement per turn (counterclockwise) -/
def bob_move : ℕ := 11

/-- The starting position for both Alice and Bob -/
def start_pos : ℕ := n

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 5

/-- The position on the circle after a given number of clockwise moves -/
def position_after_moves (start : ℕ) (moves : ℕ) : ℕ :=
  (start + moves - 1) % n + 1

theorem alice_bob_meet :
  position_after_moves start_pos (meeting_turns * alice_move) =
  position_after_moves start_pos (meeting_turns * (n - bob_move)) :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l1818_181871


namespace NUMINAMATH_CALUDE_sum_even_integers_200_to_400_l1818_181805

def even_integers_between (a b : ℕ) : List ℕ :=
  (List.range (b - a + 1)).map (fun i => a + 2 * i)

theorem sum_even_integers_200_to_400 :
  (even_integers_between 200 400).sum = 30100 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_integers_200_to_400_l1818_181805


namespace NUMINAMATH_CALUDE_equation_system_solution_l1818_181833

/-- Given a system of equations, prove the values of x and y, and the expression for 2p + q -/
theorem equation_system_solution (p q r x y : ℚ) 
  (eq1 : p / q = 6 / 7)
  (eq2 : p / r = 8 / 9)
  (eq3 : q / r = x / y) :
  x = 28 ∧ y = 27 ∧ 2 * p + q = 19 / 6 * p := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l1818_181833


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l1818_181884

/-- A quadratic function with specific properties -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_symmetry (d e f : ℝ) :
  (∀ x : ℝ, p d e f (10.5 + x) = p d e f (10.5 - x)) →  -- axis of symmetry at x = 10.5
  p d e f 3 = -5 →                                      -- passes through (3, -5)
  p d e f 12 = -5 :=                                    -- conclusion: p(12) = -5
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l1818_181884


namespace NUMINAMATH_CALUDE_determinant_zero_implies_ratio_four_l1818_181803

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_zero_implies_ratio_four (θ : ℝ) : 
  determinant (Real.sin θ) 2 (Real.cos θ) 3 = 0 → 
  (3 * Real.sin θ + 2 * Real.cos θ) / (3 * Real.sin θ - Real.cos θ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_determinant_zero_implies_ratio_four_l1818_181803


namespace NUMINAMATH_CALUDE_problem_statement_l1818_181838

open Real

-- Define the propositions
def p : Prop := ∀ x, cos (2*x - π/5) = cos (2*(x - π/5))

def q : Prop := ∀ α, tan α = 2 → (cos α)^2 - 2*(sin α)^2 = -7/4 * sin (2*α)

-- State the theorem
theorem problem_statement : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1818_181838


namespace NUMINAMATH_CALUDE_non_constant_geometric_sequence_exists_l1818_181862

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is non-constant -/
def NonConstant (a : ℕ → ℝ) : Prop :=
  ∃ m n : ℕ, a m ≠ a n

theorem non_constant_geometric_sequence_exists :
  ∃ a : ℕ → ℝ, GeometricSequence a ∧ NonConstant a ∧
  ∃ r s : ℕ, r ≠ s ∧ a r = a s :=
by sorry

end NUMINAMATH_CALUDE_non_constant_geometric_sequence_exists_l1818_181862


namespace NUMINAMATH_CALUDE_triangle_tangent_circles_intersection_l1818_181883

/-- Triangle ABC with side lengths AB=8, BC=9, CA=10 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB_length : dist A B = 8)
  (BC_length : dist B C = 9)
  (CA_length : dist C A = 10)

/-- Circle passing through a point and tangent to a line at another point -/
structure TangentCircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (passes_through : ℝ × ℝ)
  (tangent_point : ℝ × ℝ)
  (tangent_line : ℝ × ℝ → ℝ × ℝ → Prop)

/-- The intersection point of two circles -/
def CircleIntersection (ω₁ ω₂ : TangentCircle) : ℝ × ℝ := sorry

/-- The theorem to be proved -/
theorem triangle_tangent_circles_intersection
  (abc : Triangle)
  (ω₁ : TangentCircle)
  (ω₂ : TangentCircle)
  (h₁ : ω₁.passes_through = abc.B ∧ ω₁.tangent_point = abc.A ∧ ω₁.tangent_line abc.A abc.C)
  (h₂ : ω₂.passes_through = abc.C ∧ ω₂.tangent_point = abc.A ∧ ω₂.tangent_line abc.A abc.B)
  (K : ℝ × ℝ)
  (hK : K = CircleIntersection ω₁ ω₂ ∧ K ≠ abc.A) :
  dist abc.A K = 10 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_circles_intersection_l1818_181883


namespace NUMINAMATH_CALUDE_shopping_expense_percentage_l1818_181844

theorem shopping_expense_percentage (T : ℝ) (O : ℝ) : 
  T > 0 →
  0.50 * T + 0.20 * T + O * T / 100 = T →
  0.04 * (0.50 * T) + 0 * (0.20 * T) + 0.08 * (O * T / 100) = 0.044 * T →
  O = 30 := by
sorry

end NUMINAMATH_CALUDE_shopping_expense_percentage_l1818_181844


namespace NUMINAMATH_CALUDE_divisibility_by_9_52B7_l1818_181874

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def number_52B7 (B : ℕ) : ℕ := 5000 + 200 + B * 10 + 7

theorem divisibility_by_9_52B7 :
  ∀ B : ℕ, B < 10 → (is_divisible_by_9 (number_52B7 B) ↔ B = 4) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_9_52B7_l1818_181874


namespace NUMINAMATH_CALUDE_swimmers_speed_l1818_181888

/-- Proves that a person's swimming speed in still water is 4 km/h given the conditions -/
theorem swimmers_speed (water_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : water_speed = 2)
  (h2 : distance = 16) (h3 : time = 8) : ∃ v : ℝ, v = 4 ∧ distance = (v - water_speed) * time :=
by
  sorry

end NUMINAMATH_CALUDE_swimmers_speed_l1818_181888


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1818_181851

theorem sum_of_fractions_equals_one 
  (a b c x y z : ℝ) 
  (h1 : 13 * x + b * y + c * z = 0)
  (h2 : a * x + 23 * y + c * z = 0)
  (h3 : a * x + b * y + 42 * z = 0)
  (h4 : a ≠ 13)
  (h5 : x ≠ 0) :
  a / (a - 13) + b / (b - 23) + c / (c - 42) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1818_181851


namespace NUMINAMATH_CALUDE_children_ages_l1818_181846

theorem children_ages (total_age_first_birth total_age_third_birth total_age_children : ℕ)
  (h1 : total_age_first_birth = 45)
  (h2 : total_age_third_birth = 70)
  (h3 : total_age_children = 14) :
  ∃ (age1 age2 age3 : ℕ),
    age1 = 8 ∧ age2 = 5 ∧ age3 = 1 ∧
    age1 + age2 + age3 = total_age_children :=
by
  sorry


end NUMINAMATH_CALUDE_children_ages_l1818_181846


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1818_181899

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence : arithmetic_sequence 3 4 30 = 119 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1818_181899


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1818_181880

theorem arithmetic_calculation : 5 * (7 + 3) - 10 * 2 + 36 / 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1818_181880


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1818_181875

theorem cone_lateral_surface_area (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = 120) :
  (θ / 360) * π * r^2 = 12 * π :=
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1818_181875


namespace NUMINAMATH_CALUDE_line_equation_l1818_181802

/-- A line L in R² -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (a * X + b * Y + c = 0)

/-- Point in R² -/
structure Point where
  x : ℝ
  y : ℝ

def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def intersects_at (l1 l2 : Line) (x : ℝ) : Prop :=
  ∃ y : ℝ, l1.a * x + l1.b * y + l1.c = 0 ∧ l2.a * x + l2.b * y + l2.c = 0

theorem line_equation (l : Line) (p : Point) (l2 l3 : Line) :
  passes_through l { x := 1, y := 5 } →
  perpendicular l { a := 2, b := -5, c := 3, eq := sorry } →
  intersects_at l { a := 3, b := 1, c := -1, eq := sorry } (-1) →
  l = { a := 5, b := 2, c := -15, eq := sorry } :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l1818_181802


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1818_181898

theorem trigonometric_identities (α : ℝ) (h1 : α ∈ Set.Ioo (π/2) π) (h2 : Real.sin α = 3/5) :
  (Real.tan (α - π/4) = -7) ∧
  ((Real.sin (2*α) - Real.cos α) / (1 + Real.cos (2*α)) = -1/8) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1818_181898


namespace NUMINAMATH_CALUDE_velocity_of_point_C_l1818_181893

/-- Given the equation relating distances and time, prove the velocity of point C. -/
theorem velocity_of_point_C 
  (a R L T : ℝ) 
  (x : ℝ) 
  (h : (a * T) / (a * T - R) = (L + x) / x) :
  (x / T) = a * L / R :=
sorry

end NUMINAMATH_CALUDE_velocity_of_point_C_l1818_181893


namespace NUMINAMATH_CALUDE_max_intersections_fifth_degree_polynomials_l1818_181823

-- Define a fifth degree polynomial with leading coefficient 1
def FifthDegreePolynomial (a b c d e : ℝ) : ℝ → ℝ := λ x => x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

-- Theorem statement
theorem max_intersections_fifth_degree_polynomials 
  (p q : ℝ → ℝ) 
  (hp : ∃ a b c d e, p = FifthDegreePolynomial a b c d e) 
  (hq : ∃ a' b' c' d' e', q = FifthDegreePolynomial a' b' c' d' e') 
  (hpq_diff : p ≠ q) :
  ∃ S : Finset ℝ, (∀ x ∈ S, p x = q x) ∧ S.card ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_fifth_degree_polynomials_l1818_181823


namespace NUMINAMATH_CALUDE_popcorn_buckets_needed_l1818_181809

/-- The number of popcorn buckets needed by a movie theater -/
theorem popcorn_buckets_needed (packages : ℕ) (buckets_per_package : ℕ) 
  (h1 : packages = 54)
  (h2 : buckets_per_package = 8) :
  packages * buckets_per_package = 432 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_buckets_needed_l1818_181809


namespace NUMINAMATH_CALUDE_concert_cost_for_two_l1818_181808

def concert_cost (ticket_price : ℝ) (processing_fee_rate : ℝ) (parking_fee : ℝ) (entrance_fee : ℝ) (num_people : ℕ) : ℝ :=
  let total_ticket_price := ticket_price * num_people
  let processing_fee := total_ticket_price * processing_fee_rate
  let total_entrance_fee := entrance_fee * num_people
  total_ticket_price + processing_fee + parking_fee + total_entrance_fee

theorem concert_cost_for_two :
  concert_cost 50 0.15 10 5 2 = 135 := by
  sorry

end NUMINAMATH_CALUDE_concert_cost_for_two_l1818_181808


namespace NUMINAMATH_CALUDE_diamond_three_four_l1818_181804

/-- The diamond operation defined for real numbers -/
def diamond (a b : ℝ) : ℝ := 4*a + 5*b - a^2*b

/-- Theorem stating that 3 ⋄ 4 = -4 -/
theorem diamond_three_four : diamond 3 4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l1818_181804


namespace NUMINAMATH_CALUDE_problem_statement_l1818_181832

theorem problem_statement : ¬(
  (∀ (p q : Prop), (p → ¬p) ↔ (q → ¬p)) ∧
  ((∀ x : ℝ, x ∈ Set.Icc 0 1 → Real.exp x ≥ 1) ∧ 
   (∃ x : ℝ, x^2 + x + 1 < 0)) ∧
  (¬∀ (a b m : ℝ), a * m^2 < b * m^2 → a < b) ∧
  (∀ (a b : ℝ), (a + b) / 2 ≥ Real.sqrt (a * b) → (a > 0 ∧ b > 0))
) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1818_181832


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l1818_181879

theorem number_exceeding_percentage : ∃ x : ℝ, x = 75 ∧ x = 0.16 * x + 63 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l1818_181879


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_odd_integers_with_product_110895_l1818_181826

-- Define a function that generates five consecutive odd integers
def fiveConsecutiveOddIntegers (x : ℤ) : List ℤ :=
  [x - 4, x - 2, x, x + 2, x + 4]

-- Theorem statement
theorem largest_of_five_consecutive_odd_integers_with_product_110895 :
  ∃ x : ℤ, 
    (fiveConsecutiveOddIntegers x).prod = 110895 ∧
    (fiveConsecutiveOddIntegers x).all (λ i => i % 2 ≠ 0) ∧
    (fiveConsecutiveOddIntegers x).maximum? = some 17 :=
by sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_odd_integers_with_product_110895_l1818_181826


namespace NUMINAMATH_CALUDE_irrational_among_options_l1818_181843

theorem irrational_among_options : 
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 5 = (a : ℚ) / b) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (3.14 : ℝ) = (a : ℚ) / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (22 : ℚ) / 7 = (a : ℚ) / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 4 = (a : ℚ) / b) :=
by
  sorry

end NUMINAMATH_CALUDE_irrational_among_options_l1818_181843


namespace NUMINAMATH_CALUDE_partnership_theorem_l1818_181810

def partnership (total_capital : ℝ) (total_profit : ℝ) (a_profit : ℝ) : Prop :=
  let b_share : ℝ := (1 / 4) * total_capital
  let c_share : ℝ := (1 / 5) * total_capital
  let d_share : ℝ := total_capital - (b_share + c_share + (83 / 249) * total_capital)
  let a_share : ℝ := (83 / 249) * total_capital
  (a_profit / total_profit = 83 / 249) ∧
  (b_share + c_share + d_share + a_share = total_capital) ∧
  (d_share ≥ 0)

theorem partnership_theorem (total_capital : ℝ) (total_profit : ℝ) (a_profit : ℝ) 
  (h1 : total_capital > 0)
  (h2 : total_profit = 2490)
  (h3 : a_profit = 830) :
  partnership total_capital total_profit a_profit :=
by
  sorry

end NUMINAMATH_CALUDE_partnership_theorem_l1818_181810


namespace NUMINAMATH_CALUDE_function_property_l1818_181894

/-- A function satisfying f(x) + 3f(1 - x) = 4x^3 for all real x has f(4) = -72.5 -/
theorem function_property (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x^3) : 
  f 4 = -72.5 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1818_181894


namespace NUMINAMATH_CALUDE_average_goat_price_l1818_181873

/-- Given the number of goats and hens, their total cost, and the average cost of a hen,
    calculate the average cost of a goat. -/
theorem average_goat_price
  (num_goats : ℕ)
  (num_hens : ℕ)
  (total_cost : ℕ)
  (avg_hen_price : ℕ)
  (h1 : num_goats = 5)
  (h2 : num_hens = 10)
  (h3 : total_cost = 2500)
  (h4 : avg_hen_price = 50) :
  (total_cost - num_hens * avg_hen_price) / num_goats = 400 := by
  sorry

#check average_goat_price

end NUMINAMATH_CALUDE_average_goat_price_l1818_181873


namespace NUMINAMATH_CALUDE_entrance_exam_questions_entrance_exam_questions_is_70_l1818_181811

/-- Proves that the total number of questions in an entrance exam is 70,
    given the specified scoring system and student performance. -/
theorem entrance_exam_questions : ℕ :=
  let correct_marks : ℕ := 3
  let wrong_marks : ℤ := -1
  let total_score : ℤ := 38
  let correct_answers : ℕ := 27
  let total_questions : ℕ := 70
  
  have h1 : (correct_answers : ℤ) * correct_marks + 
            (total_questions - correct_answers : ℤ) * wrong_marks = total_score := by sorry
  
  total_questions

/-- The proof that the number of questions in the entrance exam is 70. -/
theorem entrance_exam_questions_is_70 : entrance_exam_questions = 70 := by sorry

end NUMINAMATH_CALUDE_entrance_exam_questions_entrance_exam_questions_is_70_l1818_181811


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1818_181828

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin α = 3/5) : 
  Real.tan (α + π/4) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1818_181828


namespace NUMINAMATH_CALUDE_representation_2015_l1818_181841

theorem representation_2015 : ∃ (a b c : ℤ), 
  a + b + c = 2015 ∧ 
  Nat.Prime a.natAbs ∧ 
  ∃ (k : ℤ), b = 3 * k ∧
  400 < c ∧ c < 500 ∧
  ¬∃ (m : ℤ), c = 3 * m := by
  sorry

end NUMINAMATH_CALUDE_representation_2015_l1818_181841


namespace NUMINAMATH_CALUDE_money_division_l1818_181891

theorem money_division (a b c : ℕ) (h1 : a = b / 2) (h2 : b = c / 2) (h3 : c = 224) :
  a + b + c = 392 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l1818_181891


namespace NUMINAMATH_CALUDE_bus_students_l1818_181856

theorem bus_students (initial : Real) (got_on : Real) (total : Real) : 
  initial = 10.0 → got_on = 3.0 → total = initial + got_on → total = 13.0 := by
  sorry

end NUMINAMATH_CALUDE_bus_students_l1818_181856


namespace NUMINAMATH_CALUDE_raphael_manny_ratio_l1818_181848

/-- Represents the number of lasagna pieces each person eats -/
structure LasagnaPieces where
  manny : ℕ
  lisa : ℕ
  raphael : ℕ
  aaron : ℕ
  kai : ℕ

/-- The properties of the lasagna distribution -/
def LasagnaDistribution (p : LasagnaPieces) : Prop :=
  p.manny = 1 ∧
  p.aaron = 0 ∧
  p.kai = 2 * p.manny ∧
  p.lisa = 2 + (p.raphael - 1) ∧
  p.manny + p.lisa + p.raphael + p.aaron + p.kai = 6

theorem raphael_manny_ratio (p : LasagnaPieces) 
  (h : LasagnaDistribution p) : p.raphael = p.manny := by
  sorry

end NUMINAMATH_CALUDE_raphael_manny_ratio_l1818_181848


namespace NUMINAMATH_CALUDE_range_of_a_for_positive_solutions_l1818_181890

theorem range_of_a_for_positive_solutions (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (1/4)^x + (1/2)^(x-1) + a = 0) ↔ -3 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_positive_solutions_l1818_181890


namespace NUMINAMATH_CALUDE_length_of_AB_l1818_181859

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 3)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ 
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB : 
  ∀ A B : ℝ × ℝ, intersection_points A B → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l1818_181859


namespace NUMINAMATH_CALUDE_sphere_volume_surface_area_equality_l1818_181819

theorem sphere_volume_surface_area_equality (r : ℝ) (h : r > 0) :
  (4 / 3 : ℝ) * Real.pi * r^3 = 36 * Real.pi → 4 * Real.pi * r^2 = 36 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_surface_area_equality_l1818_181819


namespace NUMINAMATH_CALUDE_condition_C_necessary_for_A_l1818_181878

-- Define the conditions as propositions
variable (A B C D : Prop)

-- Define the relationship between the conditions
variable (h : (C → D) → (A → B))

-- Theorem to prove
theorem condition_C_necessary_for_A (h : (C → D) → (A → B)) : A → C :=
  sorry

end NUMINAMATH_CALUDE_condition_C_necessary_for_A_l1818_181878


namespace NUMINAMATH_CALUDE_polynomial_non_negative_l1818_181849

theorem polynomial_non_negative (x : ℝ) : x^4 - x^3 + 3*x^2 - 2*x + 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_non_negative_l1818_181849


namespace NUMINAMATH_CALUDE_oxygen_atoms_count_l1818_181845

-- Define the atomic weights
def carbon_weight : ℕ := 12
def hydrogen_weight : ℕ := 1
def oxygen_weight : ℕ := 16

-- Define the number of Carbon and Hydrogen atoms
def carbon_atoms : ℕ := 2
def hydrogen_atoms : ℕ := 4

-- Define the total molecular weight of the compound
def total_weight : ℕ := 60

-- Theorem to prove
theorem oxygen_atoms_count :
  let carbon_hydrogen_weight := carbon_atoms * carbon_weight + hydrogen_atoms * hydrogen_weight
  let oxygen_weight_total := total_weight - carbon_hydrogen_weight
  oxygen_weight_total / oxygen_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_oxygen_atoms_count_l1818_181845


namespace NUMINAMATH_CALUDE_new_person_weight_is_85_l1818_181870

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + n * avg_increase

/-- Theorem stating that the weight of the new person is 85 kg -/
theorem new_person_weight_is_85 :
  new_person_weight 8 2.5 65 = 85 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_85_l1818_181870


namespace NUMINAMATH_CALUDE_toothpick_grid_15_8_l1818_181882

/-- Calculates the number of toothpicks needed for a rectangular grid with diagonals -/
def toothpick_count (height width : ℕ) : ℕ :=
  let horizontal := (height + 1) * width
  let vertical := (width + 1) * height
  let diagonal := height * width
  horizontal + vertical + diagonal

/-- Theorem stating the correct number of toothpicks for a 15x8 grid with diagonals -/
theorem toothpick_grid_15_8 :
  toothpick_count 15 8 = 383 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grid_15_8_l1818_181882


namespace NUMINAMATH_CALUDE_factorial_product_not_perfect_square_l1818_181850

theorem factorial_product_not_perfect_square (n : ℕ) (hn : n ≥ 100) :
  ¬ ∃ m : ℕ, n.factorial * (n + 1).factorial = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_product_not_perfect_square_l1818_181850
