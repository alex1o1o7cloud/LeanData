import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_f_l2656_265697

/-- The function f(x,y) represents the given expression -/
def f (x y : ℝ) : ℝ := x^2 + y^2 - 8*x + 6*y + 20

theorem min_value_of_f :
  (∀ x y : ℝ, f x y ≥ -5) ∧ (∃ x y : ℝ, f x y = -5) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2656_265697


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l2656_265604

/-- The number of minutes between 1:00 PM and 6:20 PM -/
def time_interval : ℕ := 320

/-- The interval in minutes at which Megan eats a Popsicle -/
def popsicle_interval : ℕ := 20

/-- The number of Popsicles Megan consumes -/
def popsicles_consumed : ℕ := time_interval / popsicle_interval

theorem megan_popsicle_consumption :
  popsicles_consumed = 16 :=
by sorry

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l2656_265604


namespace NUMINAMATH_CALUDE_no_seven_divisible_ones_five_l2656_265600

theorem no_seven_divisible_ones_five : ¬ ∃ (n : ℕ), (
  let num := (10^(n+1) - 10) / 9 + 5
  (num % 7 = 0) ∧ (num > 0)
) := by
  sorry

end NUMINAMATH_CALUDE_no_seven_divisible_ones_five_l2656_265600


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_sum_l2656_265687

/-- Given two arithmetic sequences {a_n} and {b_n} with the sum of their first n terms
    denoted as (A_n, B_n), where A_n / B_n = (5n + 12) / (2n + 3) for all n,
    prove that a_5 / b_5 + a_7 / b_12 = 30/7. -/
theorem arithmetic_sequence_ratio_sum (a b : ℕ → ℚ) (A B : ℕ → ℚ) :
  (∀ n, A n / B n = (5 * n + 12) / (2 * n + 3)) →
  (∀ n, A n = n * (a 1 + a n) / 2) →
  (∀ n, B n = n * (b 1 + b n) / 2) →
  a 5 / b 5 + a 7 / b 12 = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_sum_l2656_265687


namespace NUMINAMATH_CALUDE_adrian_holidays_l2656_265662

/-- The number of days Adrian takes off each month -/
def days_off_per_month : ℕ := 4

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Adrian takes in a year -/
def total_holidays : ℕ := days_off_per_month * months_in_year

theorem adrian_holidays : total_holidays = 48 := by
  sorry

end NUMINAMATH_CALUDE_adrian_holidays_l2656_265662


namespace NUMINAMATH_CALUDE_no_integer_solution_l2656_265653

theorem no_integer_solution : ¬∃ (m n : ℤ), m^2 = n^2 + 1954 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2656_265653


namespace NUMINAMATH_CALUDE_garden_area_this_year_l2656_265625

/-- Represents the garden and its contents over two years --/
structure Garden where
  cabbage_area : ℝ  -- Area taken by one cabbage
  tomato_area : ℝ   -- Area taken by one tomato plant
  last_year_cabbages : ℕ
  last_year_tomatoes : ℕ
  cabbage_increase : ℕ
  tomato_decrease : ℕ

/-- Calculates the total area of the garden --/
def garden_area (g : Garden) : ℝ :=
  let this_year_cabbages := g.last_year_cabbages + g.cabbage_increase
  let this_year_tomatoes := max (g.last_year_tomatoes - g.tomato_decrease) 0
  g.cabbage_area * this_year_cabbages + g.tomato_area * this_year_tomatoes

/-- The theorem stating the area of the garden this year --/
theorem garden_area_this_year (g : Garden) 
  (h1 : g.cabbage_area = 1)
  (h2 : g.tomato_area = 0.5)
  (h3 : g.last_year_cabbages = 72)
  (h4 : g.last_year_tomatoes = 36)
  (h5 : g.cabbage_increase = 193)
  (h6 : g.tomato_decrease = 50) :
  garden_area g = 265 := by
  sorry

#eval garden_area { 
  cabbage_area := 1, 
  tomato_area := 0.5, 
  last_year_cabbages := 72, 
  last_year_tomatoes := 36, 
  cabbage_increase := 193, 
  tomato_decrease := 50 
}

end NUMINAMATH_CALUDE_garden_area_this_year_l2656_265625


namespace NUMINAMATH_CALUDE_or_and_not_implies_false_and_true_l2656_265611

theorem or_and_not_implies_false_and_true (p q : Prop) :
  (p ∨ q) → (¬p) → (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_or_and_not_implies_false_and_true_l2656_265611


namespace NUMINAMATH_CALUDE_tree_height_average_l2656_265641

def tree_heights (n : ℕ) : Type := Fin n → ℕ

def valid_heights (h : tree_heights 7) : Prop :=
  h 1 = 16 ∧
  ∀ i : Fin 6, (h i = 2 * h i.succ ∨ 2 * h i = h i.succ)

def average_height (h : tree_heights 7) : ℚ :=
  (h 0 + h 1 + h 2 + h 3 + h 4 + h 5 + h 6 : ℚ) / 7

theorem tree_height_average (h : tree_heights 7) 
  (hvalid : valid_heights h) : average_height h = 145.1 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_average_l2656_265641


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2656_265668

theorem smallest_candy_count (x : ℕ) : 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 8 = 3 ∧ 
  x % 9 = 7 ∧
  (∀ y : ℕ, y > 0 → y % 6 = 5 → y % 8 = 3 → y % 9 = 7 → x ≤ y) → 
  x = 203 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2656_265668


namespace NUMINAMATH_CALUDE_motorcycles_in_anytown_l2656_265674

/-- Given the ratio of vehicles in Anytown and the number of sedans, 
    prove the number of motorcycles. -/
theorem motorcycles_in_anytown 
  (truck_ratio : ℕ) 
  (sedan_ratio : ℕ) 
  (motorcycle_ratio : ℕ) 
  (num_sedans : ℕ) 
  (h1 : truck_ratio = 3)
  (h2 : sedan_ratio = 7)
  (h3 : motorcycle_ratio = 2)
  (h4 : num_sedans = 9100) : 
  (num_sedans / sedan_ratio) * motorcycle_ratio = 2600 := by
  sorry

end NUMINAMATH_CALUDE_motorcycles_in_anytown_l2656_265674


namespace NUMINAMATH_CALUDE_power_of_point_theorem_l2656_265644

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a plane -/
def Point := ℝ × ℝ

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Determine if a point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop := sorry

theorem power_of_point_theorem 
  (c : Circle) (A B C D E : Point) 
  (hB : onCircle B c) (hC : onCircle C c) (hD : onCircle D c) (hE : onCircle E c)
  (hAB : distance A B = 7)
  (hBC : distance B C = 7)
  (hAD : distance A D = 10) :
  distance D E = 0.2 := by sorry

end NUMINAMATH_CALUDE_power_of_point_theorem_l2656_265644


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2656_265688

/-- Given vectors a and b in ℝ², prove that if k*a + b is perpendicular to a - 2*b, then k = 2 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 2))
  (h2 : b = (2, -1))
  (h3 : (k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 2 * b.1, a.2 - 2 * b.2) = 0) :
  k = 2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2656_265688


namespace NUMINAMATH_CALUDE_salt_solution_volume_l2656_265637

/-- Proves that for a salt solution with given conditions, the initial volume is 56 gallons -/
theorem salt_solution_volume 
  (initial_concentration : ℝ) 
  (final_concentration : ℝ) 
  (added_water : ℝ) 
  (h1 : initial_concentration = 0.10)
  (h2 : final_concentration = 0.08)
  (h3 : added_water = 14) :
  ∃ (initial_volume : ℝ), 
    initial_volume * initial_concentration = 
    (initial_volume + added_water) * final_concentration ∧ 
    initial_volume = 56 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l2656_265637


namespace NUMINAMATH_CALUDE_range_of_a_l2656_265635

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on [0, +∞) if f(x) ≥ f(y) for all 0 ≤ x ≤ y -/
def IsDecreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≥ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
    (h_even : IsEven f)
    (h_decreasing : IsDecreasingOnNonnegative f)
    (h_inequality : ∀ x, x ∈ Set.Ici 1 ∩ Set.Iio 3 → 
      f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) :
    a ∈ Set.Icc (Real.exp (-1)) ((2 + Real.log 3) / 3) := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l2656_265635


namespace NUMINAMATH_CALUDE_roots_of_quadratic_with_absolute_value_l2656_265601

theorem roots_of_quadratic_with_absolute_value
  (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (n : ℕ), n ≤ 4 ∧
  ∃ (roots : Finset ℂ), roots.card = n ∧
  ∀ z ∈ roots, a * z^2 + b * Complex.abs z + c = 0 :=
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_with_absolute_value_l2656_265601


namespace NUMINAMATH_CALUDE_rectangular_box_area_product_l2656_265639

theorem rectangular_box_area_product (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) :
  (x * z) * (x * y) * (y * z) = (x * y * z)^2 := by sorry

end NUMINAMATH_CALUDE_rectangular_box_area_product_l2656_265639


namespace NUMINAMATH_CALUDE_pizza_eating_l2656_265627

theorem pizza_eating (n : ℕ) (initial_pizza : ℚ) : 
  initial_pizza = 1 →
  (let eat_fraction := 1/3
   let remaining_fraction := 1 - eat_fraction
   let total_eaten := (1 - remaining_fraction^n) / (1 - remaining_fraction)
   n = 6 →
   total_eaten = 665/729) := by
sorry

end NUMINAMATH_CALUDE_pizza_eating_l2656_265627


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_schools_l2656_265680

theorem stratified_sampling_middle_schools 
  (total_schools : ℕ) 
  (high_schools : ℕ) 
  (middle_schools : ℕ) 
  (elementary_schools : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_schools = high_schools + middle_schools + elementary_schools)
  (h2 : total_schools = 100)
  (h3 : high_schools = 10)
  (h4 : middle_schools = 30)
  (h5 : elementary_schools = 60)
  (h6 : sample_size = 20) :
  (middle_schools : ℚ) * sample_size / total_schools = 6 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_schools_l2656_265680


namespace NUMINAMATH_CALUDE_complex_power_sum_l2656_265630

open Complex

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^1500 + (1 / z)^1500 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2656_265630


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2656_265632

theorem inequality_solution_range (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x : ℝ, |x - 4| + |x + 3| < a) : a > 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2656_265632


namespace NUMINAMATH_CALUDE_inequality_proof_l2656_265681

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a * b + b * c + c * a = 1) : 
  (3 / Real.sqrt (a^2 + 1)) + (4 / Real.sqrt (b^2 + 1)) + (12 / Real.sqrt (c^2 + 1)) < 39/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2656_265681


namespace NUMINAMATH_CALUDE_sqrt_fraction_equals_seven_fifths_l2656_265602

theorem sqrt_fraction_equals_seven_fifths :
  (Real.sqrt 64 + Real.sqrt 36) / Real.sqrt (64 + 36) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equals_seven_fifths_l2656_265602


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_rotated_square_l2656_265670

theorem lateral_surface_area_of_rotated_square (Q : ℝ) (h : Q > 0) :
  let side_length := Real.sqrt Q
  let radius := side_length
  let height := side_length
  let lateral_surface_area := 2 * Real.pi * radius * height
  lateral_surface_area = 2 * Real.pi * Q :=
by sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_rotated_square_l2656_265670


namespace NUMINAMATH_CALUDE_right_triangle_area_l2656_265621

/-- The area of a right triangle with hypotenuse 12 inches and one angle of 30° is 18√3 square inches. -/
theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 12 → θ = 30 * π / 180 → area = 18 * Real.sqrt 3 → 
  area = (1/2) * h * h * Real.sin θ * Real.cos θ :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2656_265621


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2656_265661

theorem divisibility_equivalence (a b : ℤ) : 
  (13 ∣ (2*a + 3*b)) ↔ (13 ∣ (2*b - 3*a)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2656_265661


namespace NUMINAMATH_CALUDE_prove_certain_number_l2656_265618

def w : ℕ := 468
def certain_number : ℕ := 2028

theorem prove_certain_number :
  (∃ (n : ℕ), (2^4 ∣ n * w) ∧ (3^3 ∣ n * w) ∧ (13^3 ∣ n * w)) ∧
  (∀ (x : ℕ), x < w → ¬(∃ (m : ℕ), (2^4 ∣ m * x) ∧ (3^3 ∣ m * x) ∧ (13^3 ∣ m * x))) →
  certain_number * w % 2^4 = 0 ∧
  certain_number * w % 3^3 = 0 ∧
  certain_number * w % 13^3 = 0 ∧
  (∀ (y : ℕ), y < certain_number →
    (y * w % 2^4 ≠ 0 ∨ y * w % 3^3 ≠ 0 ∨ y * w % 13^3 ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_prove_certain_number_l2656_265618


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2656_265692

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 96 → volume = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2656_265692


namespace NUMINAMATH_CALUDE_fred_baseball_cards_l2656_265609

def final_baseball_cards (initial : ℕ) (sold : ℕ) (traded : ℕ) (bought : ℕ) : ℕ :=
  initial - sold - traded + bought

theorem fred_baseball_cards : final_baseball_cards 25 7 3 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fred_baseball_cards_l2656_265609


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l2656_265699

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l2656_265699


namespace NUMINAMATH_CALUDE_cube_sum_equals_negative_eighteen_l2656_265606

theorem cube_sum_equals_negative_eighteen
  (a b c : ℝ)
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h : (a^3 + 6) / a = (b^3 + 6) / b ∧ (b^3 + 6) / b = (c^3 + 6) / c) :
  a^3 + b^3 + c^3 = -18 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_equals_negative_eighteen_l2656_265606


namespace NUMINAMATH_CALUDE_monomial_properties_l2656_265675

/-- Represents a monomial with coefficient and exponents for variables a, b, and c -/
structure Monomial where
  coeff : ℤ
  a_exp : ℕ
  b_exp : ℕ
  c_exp : ℕ

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ :=
  m.a_exp + m.b_exp + m.c_exp

/-- The given monomial -7a³b⁴c -/
def given_monomial : Monomial :=
  { coeff := -7
    a_exp := 3
    b_exp := 4
    c_exp := 1 }

theorem monomial_properties :
  given_monomial.coeff = -7 ∧ degree given_monomial = 8 := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l2656_265675


namespace NUMINAMATH_CALUDE_inequality_proof_l2656_265679

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2656_265679


namespace NUMINAMATH_CALUDE_periodic_function_value_l2656_265685

-- Define a periodic function with period 2
def isPeriodic2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f x

-- Theorem statement
theorem periodic_function_value (f : ℝ → ℝ) (h1 : isPeriodic2 f) (h2 : f 2 = 2) :
  f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l2656_265685


namespace NUMINAMATH_CALUDE_cubic_sum_values_l2656_265646

def N (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z],
    ![y, z, x],
    ![z, x, y]]

theorem cubic_sum_values (x y z : ℂ) :
  N x y z ^ 2 = 1 →
  x * y * z = 2 →
  x^3 + y^3 + z^3 = 5 ∨ x^3 + y^3 + z^3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_values_l2656_265646


namespace NUMINAMATH_CALUDE_total_word_count_180_to_220_l2656_265608

/-- Represents the word count for a number in the range [180, 220] -/
def word_count (n : ℕ) : ℕ :=
  if n = 180 then 3
  else if n ≥ 190 ∧ n ≤ 220 then 2
  else 3

/-- The sum of word counts for numbers in the range [a, b] -/
def sum_word_counts (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).sum (λ i => word_count (a + i))

theorem total_word_count_180_to_220 :
  sum_word_counts 180 220 = 99 := by
  sorry

end NUMINAMATH_CALUDE_total_word_count_180_to_220_l2656_265608


namespace NUMINAMATH_CALUDE_last_number_theorem_l2656_265671

theorem last_number_theorem (a b c d : ℝ) 
  (h1 : (a + b + c) / 3 = 6)
  (h2 : (b + c + d) / 3 = 5)
  (h3 : a + d = 11) :
  d = 4 := by
sorry

end NUMINAMATH_CALUDE_last_number_theorem_l2656_265671


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2656_265656

/-- Given a circle D with equation x^2 - 4y - 34 = -y^2 + 12x + 74,
    prove that its center (c,d) and radius r satisfy c + d + r = 4 + 2√17 -/
theorem circle_center_radius_sum (x y c d r : ℝ) : 
  (∀ x y, x^2 - 4*y - 34 = -y^2 + 12*x + 74) →
  ((x - c)^2 + (y - d)^2 = r^2) →
  (c + d + r = 4 + 2 * Real.sqrt 17) := by
  sorry


end NUMINAMATH_CALUDE_circle_center_radius_sum_l2656_265656


namespace NUMINAMATH_CALUDE_same_color_probability_l2656_265636

/-- The probability of drawing balls of the same color from two bags -/
theorem same_color_probability (bagA_white bagA_red bagB_white bagB_red : ℕ) :
  bagA_white = 8 →
  bagA_red = 4 →
  bagB_white = 6 →
  bagB_red = 6 →
  (bagA_white / (bagA_white + bagA_red : ℚ)) * (bagB_white / (bagB_white + bagB_red : ℚ)) +
  (bagA_red / (bagA_white + bagA_red : ℚ)) * (bagB_red / (bagB_white + bagB_red : ℚ)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l2656_265636


namespace NUMINAMATH_CALUDE_product_of_powers_equals_square_l2656_265638

theorem product_of_powers_equals_square : (1889568 : ℕ)^2 = 3^8 * 3^12 * 2^5 * 2^10 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_square_l2656_265638


namespace NUMINAMATH_CALUDE_square_intersection_perimeter_ratio_l2656_265654

/-- Given a square with vertices at (-2b, -2b), (2b, -2b), (-2b, 2b), and (2b, 2b),
    intersected by the line y = bx, the ratio of the perimeter of one of the
    resulting quadrilaterals to b is equal to 12 + 4√2. -/
theorem square_intersection_perimeter_ratio (b : ℝ) (b_pos : b > 0) :
  let square_vertices := [(-2*b, -2*b), (2*b, -2*b), (-2*b, 2*b), (2*b, 2*b)]
  let intersecting_line := fun x => b * x
  let quadrilateral_perimeter := 12 * b + 4 * b * Real.sqrt 2
  quadrilateral_perimeter / b = 12 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_intersection_perimeter_ratio_l2656_265654


namespace NUMINAMATH_CALUDE_product_with_9999_l2656_265698

theorem product_with_9999 : ∃ x : ℕ, x * 9999 = 4691100843 ∧ x = 469143 := by
  sorry

end NUMINAMATH_CALUDE_product_with_9999_l2656_265698


namespace NUMINAMATH_CALUDE_factor_sum_l2656_265684

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 4) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 15 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l2656_265684


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l2656_265666

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_plane_to : Plane → Plane → Prop)
variable (perpendicular_line_to_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two planes parallel to the same plane are parallel
theorem planes_parallel_to_same_plane_are_parallel
  (P Q R : Plane)
  (h1 : parallel_plane_to P R)
  (h2 : parallel_plane_to Q R) :
  parallel_planes P Q :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_same_plane_are_parallel
  (l1 l2 : Line) (P : Plane)
  (h1 : perpendicular_line_to_plane l1 P)
  (h2 : perpendicular_line_to_plane l2 P) :
  parallel_lines l1 l2 :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l2656_265666


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2656_265634

theorem problem_1 (a b : ℚ) (h1 : a = -1/2) (h2 : b = -1) :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) + a * b^2 = -3/4 := by
  sorry

theorem problem_2 (x y : ℝ) (h : |2*x - 1| + (3*y + 2)^2 = 0) :
  5 * x^2 - (2*x*y - 3 * (1/3 * x*y + 2) + 5 * x^2) = 19/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2656_265634


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2656_265669

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  a = Real.sqrt 3 →
  S = Real.sqrt 3 / 2 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  S = 1/2 * a * b * Real.sin C →
  A = π/3 ∧ b + c = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2656_265669


namespace NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l2656_265682

-- Define the complex polynomials
variable (P Q : ℂ → ℂ)

-- Define what it means for a function to be a polynomial
def IsPolynomial (f : ℂ → ℂ) : Prop := sorry

-- Define what it means for a function to be even
def IsEven (f : ℂ → ℂ) : Prop := ∀ z, f (-z) = f z

-- State the theorem
theorem even_polynomial_iff_product_with_negation :
  (IsPolynomial P ∧ IsEven P) ↔ 
  (∃ Q, IsPolynomial Q ∧ ∀ z, P z = Q z * Q (-z)) := by sorry

end NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l2656_265682


namespace NUMINAMATH_CALUDE_cubic_quadratic_system_solution_l2656_265648

theorem cubic_quadratic_system_solution :
  ∀ a b c : ℕ,
    a^3 - b^3 - c^3 = 3*a*b*c →
    a^2 = 2*(a + b + c) →
    ((a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 2 ∧ c = 2) ∨ (a = 4 ∧ b = 3 ∧ c = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_quadratic_system_solution_l2656_265648


namespace NUMINAMATH_CALUDE_unique_prime_sum_diff_l2656_265695

theorem unique_prime_sum_diff : ∃! p : ℕ, 
  Prime p ∧ 
  (∃ q s r t : ℕ, Prime q ∧ Prime s ∧ Prime r ∧ Prime t ∧ 
    p = q + s ∧ p = r - t) ∧ 
  p = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_diff_l2656_265695


namespace NUMINAMATH_CALUDE_tablet_consumption_time_l2656_265615

theorem tablet_consumption_time (num_tablets : ℕ) (interval : ℕ) : num_tablets = 10 ∧ interval = 25 → (num_tablets - 1) * interval = 225 := by
  sorry

end NUMINAMATH_CALUDE_tablet_consumption_time_l2656_265615


namespace NUMINAMATH_CALUDE_sibling_pairs_count_l2656_265645

theorem sibling_pairs_count 
  (business_students : ℕ) 
  (law_students : ℕ) 
  (sibling_pair_probability : ℝ) 
  (h1 : business_students = 500) 
  (h2 : law_students = 800) 
  (h3 : sibling_pair_probability = 7.500000000000001e-05) : 
  ℕ := 
by
  sorry

#check sibling_pairs_count

end NUMINAMATH_CALUDE_sibling_pairs_count_l2656_265645


namespace NUMINAMATH_CALUDE_min_value_expression_l2656_265651

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1/2) :
  a^2 + 4*a*b + 9*b^2 + 8*b*c + 3*c^2 ≥ 13.5 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 1/2 ∧
    a₀^2 + 4*a₀*b₀ + 9*b₀^2 + 8*b₀*c₀ + 3*c₀^2 = 13.5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2656_265651


namespace NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l2656_265690

/-- Given two cyclists, Alberto and Bjorn, with different speeds, 
    prove the difference in distance traveled after a certain time. -/
theorem alberto_bjorn_distance_difference 
  (alberto_speed : ℝ) 
  (bjorn_speed : ℝ) 
  (time : ℝ) 
  (h1 : alberto_speed = 18) 
  (h2 : bjorn_speed = 17) 
  (h3 : time = 5) : 
  alberto_speed * time - bjorn_speed * time = 5 := by
  sorry

#check alberto_bjorn_distance_difference

end NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l2656_265690


namespace NUMINAMATH_CALUDE_congruence_solution_l2656_265607

theorem congruence_solution (y : ℤ) : 
  (10 * y + 3) % 18 = 7 % 18 → y % 9 = 4 % 9 := by
sorry

end NUMINAMATH_CALUDE_congruence_solution_l2656_265607


namespace NUMINAMATH_CALUDE_sam_dimes_l2656_265624

/-- The number of dimes Sam has after receiving more from his dad -/
def total_dimes (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Proof that Sam has 16 dimes after receiving more from his dad -/
theorem sam_dimes : total_dimes 9 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sam_dimes_l2656_265624


namespace NUMINAMATH_CALUDE_fifth_week_consumption_l2656_265628

/-- Represents the vegetable consumption for a day -/
structure VegetableConsumption where
  asparagus : Float
  broccoli : Float
  cauliflower : Float
  spinach : Float
  kale : Float
  zucchini : Float
  carrots : Float

/-- Calculates the total vegetable consumption for a day -/
def totalConsumption (vc : VegetableConsumption) : Float :=
  vc.asparagus + vc.broccoli + vc.cauliflower + vc.spinach + vc.kale + vc.zucchini + vc.carrots

/-- Initial weekday consumption -/
def initialWeekday : VegetableConsumption := {
  asparagus := 0.25, broccoli := 0.25, cauliflower := 0.5,
  spinach := 0, kale := 0, zucchini := 0, carrots := 0
}

/-- Initial weekend consumption -/
def initialWeekend : VegetableConsumption := {
  asparagus := 0.3, broccoli := 0.4, cauliflower := 0.6,
  spinach := 0, kale := 0, zucchini := 0, carrots := 0
}

/-- Updated weekday consumption -/
def updatedWeekday : VegetableConsumption := {
  asparagus := initialWeekday.asparagus * 2,
  broccoli := initialWeekday.broccoli * 3,
  cauliflower := initialWeekday.cauliflower * 1.75,
  spinach := 0.5,
  kale := 0, zucchini := 0, carrots := 0
}

/-- Updated Saturday consumption -/
def updatedSaturday : VegetableConsumption := {
  asparagus := initialWeekend.asparagus,
  broccoli := initialWeekend.broccoli,
  cauliflower := initialWeekend.cauliflower,
  spinach := 0,
  kale := 1,
  zucchini := 0.3,
  carrots := 0
}

/-- Updated Sunday consumption -/
def updatedSunday : VegetableConsumption := {
  asparagus := initialWeekend.asparagus,
  broccoli := initialWeekend.broccoli,
  cauliflower := initialWeekend.cauliflower,
  spinach := 0,
  kale := 0,
  zucchini := 0,
  carrots := 0.5
}

/-- Theorem: The total vegetable consumption in the fifth week is 17.225 pounds -/
theorem fifth_week_consumption :
  5 * totalConsumption updatedWeekday +
  totalConsumption updatedSaturday +
  totalConsumption updatedSunday = 17.225 := by
  sorry

end NUMINAMATH_CALUDE_fifth_week_consumption_l2656_265628


namespace NUMINAMATH_CALUDE_price_change_l2656_265689

/-- Calculates the final price of an item after three price changes -/
theorem price_change (initial_price : ℝ) : 
  initial_price = 320 → 
  (initial_price * 1.15 * 0.9 * 1.25) = 414 := by
  sorry


end NUMINAMATH_CALUDE_price_change_l2656_265689


namespace NUMINAMATH_CALUDE_equation_solution_l2656_265672

theorem equation_solution (M N : ℕ) 
  (h1 : (4 : ℚ) / 7 = M / 63)
  (h2 : (4 : ℚ) / 7 = 84 / N) : 
  M + N = 183 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2656_265672


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2656_265691

theorem complex_number_quadrant : ∃ (a b : ℝ), a < 0 ∧ b < 0 ∧ (1 - I) / (1 + 2*I) = a + b*I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2656_265691


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l2656_265631

/-- The number of people that can be seated in one seat of the Ferris wheel -/
def people_per_seat : ℕ := 9

/-- The number of seats on the Ferris wheel -/
def number_of_seats : ℕ := 2

/-- The total number of people that can ride the Ferris wheel at the same time -/
def total_riders : ℕ := people_per_seat * number_of_seats

theorem ferris_wheel_capacity : total_riders = 18 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l2656_265631


namespace NUMINAMATH_CALUDE_rachel_chocolate_sales_l2656_265617

theorem rachel_chocolate_sales (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : 
  total_bars = 13 → price_per_bar = 2 → unsold_bars = 4 → 
  (total_bars - unsold_bars) * price_per_bar = 18 := by
sorry

end NUMINAMATH_CALUDE_rachel_chocolate_sales_l2656_265617


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l2656_265626

theorem root_equation_implies_expression_value (m : ℝ) : 
  m^2 - 4*m - 2 = 0 → 2*m^2 - 8*m = 4 := by
sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l2656_265626


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l2656_265642

/-- The number of ways to distribute n distinguishable objects into k boxes,
    where m boxes are distinguishable and (k-m) boxes are indistinguishable. -/
def distribution_count (n k m : ℕ) : ℕ :=
  k^n - (k-m)^n + ((k-m)^n / 2)

/-- The number of ways to place 5 distinguishable balls into 3 boxes,
    where one box is distinguishable (red) and the other two are indistinguishable. -/
theorem five_balls_three_boxes :
  distribution_count 5 3 1 = 227 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l2656_265642


namespace NUMINAMATH_CALUDE_total_arrangements_eq_5760_l2656_265647

/-- The number of ways to arrange n distinct objects taken k at a time -/
def arrangements (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The total number of students -/
def total_students : ℕ := 8

/-- The number of students in each row -/
def students_per_row : ℕ := 4

/-- The number of students with fixed positions (A, B, and C) -/
def fixed_students : ℕ := 3

/-- The number of ways to arrange A and B in the front row -/
def front_row_arrangements : ℕ := arrangements students_per_row 2

/-- The number of ways to arrange C in the back row -/
def back_row_arrangements : ℕ := arrangements students_per_row 1

/-- The number of ways to arrange the remaining students -/
def remaining_arrangements : ℕ := arrangements (total_students - fixed_students) (total_students - fixed_students)

theorem total_arrangements_eq_5760 :
  front_row_arrangements * back_row_arrangements * remaining_arrangements = 5760 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_eq_5760_l2656_265647


namespace NUMINAMATH_CALUDE_wilson_hamburgers_l2656_265620

/-- The number of hamburgers Wilson bought -/
def num_hamburgers : ℕ := 2

/-- The price of each hamburger in dollars -/
def hamburger_price : ℕ := 5

/-- The number of cola bottles -/
def num_cola : ℕ := 3

/-- The price of each cola bottle in dollars -/
def cola_price : ℕ := 2

/-- The discount amount in dollars -/
def discount : ℕ := 4

/-- The total amount Wilson paid in dollars -/
def total_paid : ℕ := 12

theorem wilson_hamburgers :
  num_hamburgers * hamburger_price + num_cola * cola_price - discount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_wilson_hamburgers_l2656_265620


namespace NUMINAMATH_CALUDE_katy_made_65_brownies_l2656_265650

/-- The number of brownies Katy made and ate over four days --/
def brownies_problem (total : ℕ) : Prop :=
  ∃ (mon tue wed thu_before thu_after : ℕ),
    -- Monday's consumption
    mon = 5 ∧
    -- Tuesday's consumption
    tue = 2 * mon ∧
    -- Wednesday's consumption
    wed = 3 * tue ∧
    -- Remaining brownies before sharing on Thursday
    thu_before = total - (mon + tue + wed) ∧
    -- Brownies left after sharing on Thursday
    thu_after = thu_before / 2 ∧
    -- Brownies left after sharing equals Tuesday's consumption
    thu_after = tue ∧
    -- All brownies are gone after Thursday
    mon + tue + wed + thu_before = total

/-- The total number of brownies Katy made is 65 --/
theorem katy_made_65_brownies : brownies_problem 65 := by
  sorry

end NUMINAMATH_CALUDE_katy_made_65_brownies_l2656_265650


namespace NUMINAMATH_CALUDE_subset_families_inequality_l2656_265665

/-- Given an n-element set X and two families of subsets 𝓐 and 𝓑 of X, 
    where each subset in 𝓐 cannot be compared with every subset in 𝓑, 
    prove that √|𝓐| + √|𝓑| ≤ 2^(7/2). -/
theorem subset_families_inequality (n : ℕ) (X : Finset (Finset ℕ)) 
  (𝓐 𝓑 : Finset (Finset ℕ)) : 
  (∀ A ∈ 𝓐, ∀ B ∈ 𝓑, ¬(A ⊆ B ∨ B ⊆ A)) →
  X.card = n →
  (∀ A ∈ 𝓐, A ∈ X) →
  (∀ B ∈ 𝓑, B ∈ X) →
  Real.sqrt (𝓐.card : ℝ) + Real.sqrt (𝓑.card : ℝ) ≤ 2^(7/2) :=
by sorry

end NUMINAMATH_CALUDE_subset_families_inequality_l2656_265665


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2656_265643

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (3, 2)
  are_parallel a b → m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2656_265643


namespace NUMINAMATH_CALUDE_second_workshop_production_l2656_265678

/-- Represents the production and sampling data for three workshops -/
structure WorkshopData where
  total_production : ℕ
  sample_1 : ℕ
  sample_2 : ℕ
  sample_3 : ℕ

/-- Checks if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- Calculates the production of the second workshop based on sampling data -/
def productionOfSecondWorkshop (data : WorkshopData) : ℕ :=
  data.sample_2 * data.total_production / (data.sample_1 + data.sample_2 + data.sample_3)

/-- Theorem stating the production of the second workshop is 1200 given the conditions -/
theorem second_workshop_production
  (data : WorkshopData)
  (h_total : data.total_production = 3600)
  (h_arithmetic : isArithmeticSequence data.sample_1 data.sample_2 data.sample_3) :
  productionOfSecondWorkshop data = 1200 := by
  sorry


end NUMINAMATH_CALUDE_second_workshop_production_l2656_265678


namespace NUMINAMATH_CALUDE_solve_equation_l2656_265683

theorem solve_equation (x : ℝ) (h : 61 + 5 * 12 / (x / 3) = 62) : x = 180 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2656_265683


namespace NUMINAMATH_CALUDE_y_satisfies_conditions_l2656_265614

/-- The function we want to prove satisfies the given conditions -/
def y (t : ℝ) : ℝ := t^3 - t^2 + t + 19

/-- The derivative of y(t) -/
def y_derivative (t : ℝ) : ℝ := 3*t^2 - 2*t + 1

theorem y_satisfies_conditions :
  (∀ t, (deriv y) t = y_derivative t) ∧ y 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_y_satisfies_conditions_l2656_265614


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l2656_265667

theorem bus_seating_capacity : ∀ (x : ℕ),
  (4 * x + 30 = 5 * x - 10) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l2656_265667


namespace NUMINAMATH_CALUDE_a_plus_2b_plus_3c_equals_35_l2656_265613

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem a_plus_2b_plus_3c_equals_35 :
  (∀ x, f (x + 2) = 5 * x^2 + 2 * x + 6) →
  (∃ a b c, ∀ x, f x = a * x^2 + b * x + c) →
  (∃ a b c, (∀ x, f x = a * x^2 + b * x + c) ∧ a + 2 * b + 3 * c = 35) :=
by sorry

end NUMINAMATH_CALUDE_a_plus_2b_plus_3c_equals_35_l2656_265613


namespace NUMINAMATH_CALUDE_coffee_beans_remaining_l2656_265660

theorem coffee_beans_remaining (jar_weight empty_weight full_weight remaining_weight : ℝ)
  (h1 : empty_weight = 0.2 * full_weight)
  (h2 : remaining_weight = 0.6 * full_weight)
  (h3 : empty_weight > 0)
  (h4 : full_weight > empty_weight) : 
  let beans_weight := full_weight - empty_weight
  let defective_weight := 0.1 * beans_weight
  let remaining_beans := remaining_weight - empty_weight
  (remaining_beans - defective_weight) / (beans_weight - defective_weight) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_coffee_beans_remaining_l2656_265660


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_lines_l2656_265603

/-- Given two intersecting lines and a perpendicular line, prove the equations of a line through the intersection point and its symmetric line. -/
theorem intersection_and_perpendicular_lines
  (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) (perp_line : ℝ → ℝ → Prop)
  (h1 : ∀ x y, line1 x y ↔ x - 2*y + 4 = 0)
  (h2 : ∀ x y, line2 x y ↔ x + y - 2 = 0)
  (h3 : ∀ x y, perp_line x y ↔ 5*x + 3*y - 6 = 0)
  (P : ℝ × ℝ) (hP : line1 P.1 P.2 ∧ line2 P.1 P.2)
  (l : ℝ → ℝ → Prop) (hl : l P.1 P.2)
  (hperp : ∀ x y, l x y → (5 * (y - P.2) = -3 * (x - P.1))) :
  (∀ x y, l x y ↔ 3*x - 5*y + 10 = 0) ∧
  (∀ x y, (3*x - 5*y - 10 = 0) ↔ (l (-x) (-y))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_lines_l2656_265603


namespace NUMINAMATH_CALUDE_raft_capacity_raft_problem_l2656_265696

/-- Calculates the number of people that can fit on a raft given certain conditions -/
theorem raft_capacity (max_capacity : ℕ) (reduction_full : ℕ) (life_jackets_needed : ℕ) : ℕ :=
  let capacity_with_jackets := max_capacity - reduction_full
  let jacket_to_person_ratio := capacity_with_jackets / reduction_full
  let reduction := life_jackets_needed / jacket_to_person_ratio
  max_capacity - reduction

/-- Proves that 17 people can fit on the raft under given conditions -/
theorem raft_problem : raft_capacity 21 7 8 = 17 := by
  sorry

end NUMINAMATH_CALUDE_raft_capacity_raft_problem_l2656_265696


namespace NUMINAMATH_CALUDE_sum_of_digits_square_999999999_l2656_265652

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def repeated_nines (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem sum_of_digits_square_999999999 :
  digit_sum ((repeated_nines 9)^2) = 81 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_999999999_l2656_265652


namespace NUMINAMATH_CALUDE_opposite_sides_range_l2656_265633

def line_equation (x y a : ℝ) : ℝ := x - y + a

theorem opposite_sides_range (a : ℝ) : 
  (line_equation 0 0 a) * (line_equation 1 (-1) a) < 0 ↔ -2 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l2656_265633


namespace NUMINAMATH_CALUDE_faye_candy_count_l2656_265663

/-- Calculates the final number of candy pieces Faye has after eating some and receiving more. -/
def final_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) : ℕ :=
  initial - eaten + received

/-- Proves that Faye ends up with 62 pieces of candy given the initial conditions. -/
theorem faye_candy_count :
  final_candy_count 47 25 40 = 62 := by
  sorry

end NUMINAMATH_CALUDE_faye_candy_count_l2656_265663


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2656_265693

def circle_equation (x y : ℝ) : Prop :=
  x^2 + 2*x - 4*y - 7 = -y^2 + 8*x

def center_and_radius_sum (c d s : ℝ) : ℝ :=
  c + d + s

theorem circle_center_radius_sum :
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    center_and_radius_sum c d s = 5 + 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2656_265693


namespace NUMINAMATH_CALUDE_total_cost_proof_l2656_265610

def sandwich_price : ℚ := 245/100
def soda_price : ℚ := 87/100
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4

theorem total_cost_proof : 
  (num_sandwiches : ℚ) * sandwich_price + (num_sodas : ℚ) * soda_price = 838/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l2656_265610


namespace NUMINAMATH_CALUDE_largest_consecutive_nonprime_less_than_30_l2656_265640

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem largest_consecutive_nonprime_less_than_30 :
  ∃ (n : ℕ),
    isTwoDigit n ∧
    n < 30 ∧
    (∀ k : ℕ, k ∈ List.range 5 → ¬isPrime (n - k)) ∧
    (∀ m : ℕ, m > n → ¬(isTwoDigit m ∧ m < 30 ∧ (∀ k : ℕ, k ∈ List.range 5 → ¬isPrime (m - k)))) ∧
    n = 28 := by sorry

end NUMINAMATH_CALUDE_largest_consecutive_nonprime_less_than_30_l2656_265640


namespace NUMINAMATH_CALUDE_total_points_theorem_l2656_265664

/-- The total points scored by Zach and Ben in a football game -/
def total_points (zach_points ben_points : Float) : Float :=
  zach_points + ben_points

/-- Theorem stating that the total points scored by Zach and Ben is 63.0 -/
theorem total_points_theorem (zach_points ben_points : Float)
  (h1 : zach_points = 42.0)
  (h2 : ben_points = 21.0) :
  total_points zach_points ben_points = 63.0 := by
  sorry

end NUMINAMATH_CALUDE_total_points_theorem_l2656_265664


namespace NUMINAMATH_CALUDE_f_three_intersections_iff_a_in_range_l2656_265623

/-- The function f(x) = √(ax + 4) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a * x + 4)

/-- The inverse function of f -/
noncomputable def f_inv (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) / a

/-- Predicate for f and f_inv having exactly three distinct intersection points -/
def has_three_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = f_inv a x₁ ∧ f a x₂ = f_inv a x₂ ∧ f a x₃ = f_inv a x₃ ∧
    ∀ x : ℝ, f a x = f_inv a x → x = x₁ ∨ x = x₂ ∨ x = x₃

theorem f_three_intersections_iff_a_in_range (a : ℝ) :
  a ≠ 0 → (has_three_intersections a ↔ -4 * Real.sqrt 3 / 3 < a ∧ a ≤ -2) :=
sorry

end NUMINAMATH_CALUDE_f_three_intersections_iff_a_in_range_l2656_265623


namespace NUMINAMATH_CALUDE_union_determines_m_l2656_265658

def A (m : ℝ) : Set ℝ := {2, m}
def B (m : ℝ) : Set ℝ := {1, m^2}

theorem union_determines_m :
  ∀ m : ℝ, A m ∪ B m = {1, 2, 3, 9} → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_determines_m_l2656_265658


namespace NUMINAMATH_CALUDE_rectangle_length_from_perimeter_and_width_l2656_265677

/-- The perimeter of a rectangle is twice the sum of its length and width -/
def rectangle_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Given a rectangle with perimeter 100 cm and width 20 cm, its length is 30 cm -/
theorem rectangle_length_from_perimeter_and_width :
  ∃ (length : ℝ), 
    rectangle_perimeter length 20 = 100 ∧ length = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_from_perimeter_and_width_l2656_265677


namespace NUMINAMATH_CALUDE_children_who_got_on_bus_stop_l2656_265619

/-- The number of children who got on the bus at a stop -/
def children_who_got_on (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Proof that 38 children got on the bus at the stop -/
theorem children_who_got_on_bus_stop : children_who_got_on 26 64 = 38 := by
  sorry

end NUMINAMATH_CALUDE_children_who_got_on_bus_stop_l2656_265619


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2656_265649

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a + 2) * (a + 1) * (a - 1) = a^3 - 6 → 
  a^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2656_265649


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_is_162_l2656_265673

theorem smallest_number_of_eggs : ℕ → Prop :=
  fun n =>
    (n > 150) ∧
    (∃ c : ℕ, c > 0 ∧ n = 15 * c - 3) ∧
    (∀ m : ℕ, m > 150 ∧ (∃ d : ℕ, d > 0 ∧ m = 15 * d - 3) → m ≥ n) →
    n = 162

theorem smallest_number_of_eggs_is_162 : smallest_number_of_eggs 162 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_is_162_l2656_265673


namespace NUMINAMATH_CALUDE_parabola_vertex_l2656_265694

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x - 2)^2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 0)

/-- Theorem: The vertex coordinates of the parabola y = (x-2)² are (2, 0) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2656_265694


namespace NUMINAMATH_CALUDE_halloween_candy_theorem_l2656_265612

/-- The amount of remaining candy after Halloween night -/
def remaining_candy (debby_candy sister_candy brother_candy eaten : ℕ) : ℕ :=
  debby_candy + sister_candy + brother_candy - eaten

/-- Theorem stating the remaining candy after Halloween night -/
theorem halloween_candy_theorem :
  remaining_candy 32 42 48 56 = 66 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_theorem_l2656_265612


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2656_265655

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 9*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2656_265655


namespace NUMINAMATH_CALUDE_cubic_properties_l2656_265605

theorem cubic_properties :
  (∀ x : ℝ, x^3 > 0 → x > 0) ∧
  (∀ x : ℝ, x < 1 → x^3 < x) :=
by sorry

end NUMINAMATH_CALUDE_cubic_properties_l2656_265605


namespace NUMINAMATH_CALUDE_k_range_theorem_l2656_265659

/-- A function f is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x < f y

/-- A function f has a maximum value of a and a minimum value of b on the interval [0, k] -/
def HasMaxMinOn (f : ℝ → ℝ) (a b k : ℝ) : Prop :=
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ k → f x ≤ a) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ f x = a) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ k → b ≤ f x) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ f x = b)

theorem k_range_theorem (k : ℝ) :
  let p := IsIncreasing (λ x : ℝ => k * x + 1)
  let q := HasMaxMinOn (λ x : ℝ => x^2 - 2*x + 3) 3 2 k
  (¬(p ∧ q)) ∧ (p ∨ q) → k ∈ Set.Ioo 0 1 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_k_range_theorem_l2656_265659


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l2656_265676

theorem coefficient_x_squared_expansion :
  let f : Polynomial ℤ := (X + 1)^5 * (2*X + 1)
  (f.coeff 2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l2656_265676


namespace NUMINAMATH_CALUDE_infinite_essentially_different_solutions_l2656_265629

/-- Two integer triples are essentially different if they are not scalar multiples of each other -/
def EssentiallyDifferent (a b c a₁ b₁ c₁ : ℤ) : Prop :=
  ∀ r : ℚ, ¬(a₁ = r * a ∧ b₁ = r * b ∧ c₁ = r * c)

/-- The set of solutions for the equation x^2 = y^2 + k·z^2 -/
def SolutionSet (k : ℕ) : Set (ℤ × ℤ × ℤ) :=
  {(x, y, z) | x^2 = y^2 + k * z^2}

/-- The theorem stating that there are infinitely many essentially different solutions -/
theorem infinite_essentially_different_solutions (k : ℕ) :
  ∃ S : Set (ℤ × ℤ × ℤ),
    (∀ (x y z : ℤ), (x, y, z) ∈ S → (x, y, z) ∈ SolutionSet k) ∧
    (∀ (x y z x₁ y₁ z₁ : ℤ), (x, y, z) ∈ S → (x₁, y₁, z₁) ∈ S → (x, y, z) ≠ (x₁, y₁, z₁) →
      EssentiallyDifferent x y z x₁ y₁ z₁) ∧
    Set.Infinite S :=
  sorry

end NUMINAMATH_CALUDE_infinite_essentially_different_solutions_l2656_265629


namespace NUMINAMATH_CALUDE_min_a_bound_l2656_265616

theorem min_a_bound (a : ℝ) : (∀ x : ℝ, x > 0 → x / (x^2 + 3*x + 1) ≤ a) ↔ a ≥ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_min_a_bound_l2656_265616


namespace NUMINAMATH_CALUDE_camping_hike_distance_l2656_265686

theorem camping_hike_distance (total_distance stream_to_meadow meadow_to_campsite : ℝ)
  (h1 : total_distance = 0.7)
  (h2 : stream_to_meadow = 0.4)
  (h3 : meadow_to_campsite = 0.1) :
  total_distance - (stream_to_meadow + meadow_to_campsite) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_camping_hike_distance_l2656_265686


namespace NUMINAMATH_CALUDE_digit_2500_is_8_l2656_265657

/-- The number of digits in the representation of positive integers from 1 to n -/
def digitCount (n : ℕ) : ℕ := sorry

/-- The nth digit in the concatenation of integers from 1 to 1099 -/
def nthDigit (n : ℕ) : ℕ := sorry

theorem digit_2500_is_8 : nthDigit 2500 = 8 := by sorry

end NUMINAMATH_CALUDE_digit_2500_is_8_l2656_265657


namespace NUMINAMATH_CALUDE_sum_of_w_and_z_l2656_265622

theorem sum_of_w_and_z (w x y z : ℤ) 
  (eq1 : w + x = 45)
  (eq2 : x + y = 51)
  (eq3 : y + z = 28) :
  w + z = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_w_and_z_l2656_265622
