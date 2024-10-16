import Mathlib

namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l819_81928

theorem parallel_vectors_k_value (k : ℝ) : 
  let a : Fin 2 → ℝ := ![3*k + 1, 2]
  let b : Fin 2 → ℝ := ![k, 1]
  (∃ (c : ℝ), a = c • b) → k = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l819_81928


namespace NUMINAMATH_CALUDE_max_rectangle_area_l819_81954

/-- Given 40 feet of fencing for a rectangular pen, the maximum area enclosed is 100 square feet. -/
theorem max_rectangle_area (fencing : ℝ) (h : fencing = 40) :
  ∃ (length width : ℝ),
    length > 0 ∧ width > 0 ∧
    2 * (length + width) = fencing ∧
    ∀ (l w : ℝ), l > 0 → w > 0 → 2 * (l + w) = fencing → l * w ≤ length * width ∧
    length * width = 100 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l819_81954


namespace NUMINAMATH_CALUDE_hari_contribution_is_8280_l819_81952

/-- Represents the business partnership between Praveen and Hari -/
structure Partnership where
  praveen_initial : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's contribution to the partnership -/
def hari_contribution (p : Partnership) : ℕ :=
  (p.praveen_initial * p.praveen_months * p.profit_ratio_hari) / (p.hari_months * p.profit_ratio_praveen)

/-- Theorem stating Hari's contribution in the given scenario -/
theorem hari_contribution_is_8280 :
  let p : Partnership := {
    praveen_initial := 3220,
    praveen_months := 12,
    hari_months := 7,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  hari_contribution p = 8280 := by
  sorry

end NUMINAMATH_CALUDE_hari_contribution_is_8280_l819_81952


namespace NUMINAMATH_CALUDE_grey_cats_count_l819_81908

/-- The number of grey cats in a house after a series of events -/
def grey_cats_after_events : ℕ :=
  let initial_total : ℕ := 16
  let initial_white : ℕ := 2
  let initial_black : ℕ := (25 * initial_total) / 100
  let black_after_leaving : ℕ := initial_black / 2
  let white_after_arrival : ℕ := initial_white + 2
  let initial_grey : ℕ := initial_total - initial_white - initial_black
  initial_grey + 1

/-- Theorem stating the number of grey cats after the events -/
theorem grey_cats_count : grey_cats_after_events = 11 := by
  sorry

end NUMINAMATH_CALUDE_grey_cats_count_l819_81908


namespace NUMINAMATH_CALUDE_line_through_point_with_slope_l819_81948

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Creates a Line from a point and a slope -/
def lineFromPointSlope (x y m : ℝ) : Line :=
  { slope := m, yIntercept := y - m * x }

/-- The equation of a line in the form y = mx + b -/
def lineEquation (l : Line) (x : ℝ) : ℝ := l.slope * x + l.yIntercept

theorem line_through_point_with_slope (x₀ y₀ m : ℝ) :
  let l := lineFromPointSlope x₀ y₀ m
  ∀ x, lineEquation l x = m * (x - x₀) + y₀ := by sorry

end NUMINAMATH_CALUDE_line_through_point_with_slope_l819_81948


namespace NUMINAMATH_CALUDE_draw_jack_queen_king_of_hearts_probability_l819_81975

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (jacks : Nat)
  (queens : Nat)
  (king_of_hearts : Nat)

/-- The probability of drawing a specific sequence of cards from a deck -/
def draw_probability (d : Deck) : ℚ :=
  (d.jacks : ℚ) / d.total_cards *
  (d.queens : ℚ) / (d.total_cards - 1) *
  (d.king_of_hearts : ℚ) / (d.total_cards - 2)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  ⟨52, 4, 4, 1⟩

theorem draw_jack_queen_king_of_hearts_probability :
  draw_probability standard_deck = 4 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_draw_jack_queen_king_of_hearts_probability_l819_81975


namespace NUMINAMATH_CALUDE_number_equation_solution_l819_81969

theorem number_equation_solution : ∃ x : ℝ, (7 * x = 3 * x + 12) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l819_81969


namespace NUMINAMATH_CALUDE_all_props_true_l819_81966

-- Define the original proposition
def original_prop (x y : ℝ) : Prop := (x * y = 0) → (x = 0 ∨ y = 0)

-- Define the inverse proposition
def inverse_prop (x y : ℝ) : Prop := (x = 0 ∨ y = 0) → (x * y = 0)

-- Define the negation proposition
def negation_prop (x y : ℝ) : Prop := (x * y ≠ 0) → (x ≠ 0 ∧ y ≠ 0)

-- Define the contrapositive proposition
def contrapositive_prop (x y : ℝ) : Prop := (x ≠ 0 ∧ y ≠ 0) → (x * y ≠ 0)

-- Theorem stating that all three derived propositions are true
theorem all_props_true : 
  (∀ x y : ℝ, inverse_prop x y) ∧ 
  (∀ x y : ℝ, negation_prop x y) ∧ 
  (∀ x y : ℝ, contrapositive_prop x y) :=
sorry

end NUMINAMATH_CALUDE_all_props_true_l819_81966


namespace NUMINAMATH_CALUDE_number_of_men_is_group_size_l819_81950

/-- Represents the number of men, women, and boys -/
def group_size : ℕ := 8

/-- Represents the total earnings of all people -/
def total_earnings : ℕ := 105

/-- Represents the wage of each man -/
def men_wage : ℕ := 7

/-- Theorem stating that the number of men is equal to the group size -/
theorem number_of_men_is_group_size :
  ∃ (women_wage boy_wage : ℚ),
    group_size * men_wage + group_size * women_wage + group_size * boy_wage = total_earnings →
    group_size = group_size := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_is_group_size_l819_81950


namespace NUMINAMATH_CALUDE_median_pets_is_three_l819_81945

/-- Represents the distribution of pet ownership --/
def PetDistribution : List (ℕ × ℕ) :=
  [(2, 5), (3, 6), (4, 1), (5, 4), (6, 3)]

/-- The total number of individuals in the survey --/
def TotalIndividuals : ℕ := 19

/-- Calculates the median position for an odd number of data points --/
def MedianPosition (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- Finds the median number of pets owned given the distribution --/
def MedianPets (dist : List (ℕ × ℕ)) (total : ℕ) : ℕ :=
  sorry -- Proof to be implemented

theorem median_pets_is_three :
  MedianPets PetDistribution TotalIndividuals = 3 :=
by sorry

end NUMINAMATH_CALUDE_median_pets_is_three_l819_81945


namespace NUMINAMATH_CALUDE_dart_probability_l819_81955

theorem dart_probability (square_side : Real) (circle_area : Real) 
  (h1 : square_side = 1)
  (h2 : circle_area = Real.pi / 4) :
  1 - (circle_area / (square_side * square_side)) = 1 - Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_dart_probability_l819_81955


namespace NUMINAMATH_CALUDE_root_power_equality_l819_81937

theorem root_power_equality (x₀ : ℝ) (h : x₀^11 + x₀^7 + x₀^3 = 1) :
  x₀^4 + x₀^3 - 1 = x₀^15 := by
  sorry

end NUMINAMATH_CALUDE_root_power_equality_l819_81937


namespace NUMINAMATH_CALUDE_coefficient_of_x_l819_81920

theorem coefficient_of_x (x y : ℝ) (some : ℝ) 
  (h1 : x / (2 * y) = 3 / 2)
  (h2 : (some * x + 5 * y) / (x - 2 * y) = 26) :
  some = 7 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l819_81920


namespace NUMINAMATH_CALUDE_face_mask_profit_l819_81921

/-- Calculates the total profit from selling face masks given specific conditions --/
theorem face_mask_profit : 
  let original_price : ℝ := 10
  let discount1 : ℝ := 0.2
  let discount2 : ℝ := 0.3
  let discount3 : ℝ := 0.4
  let packs1 : ℕ := 20
  let packs2 : ℕ := 30
  let packs3 : ℕ := 40
  let masks_per_pack : ℕ := 5
  let sell_price1 : ℝ := 0.75
  let sell_price2 : ℝ := 0.85
  let sell_price3 : ℝ := 0.95

  let cost1 : ℝ := original_price * (1 - discount1)
  let cost2 : ℝ := original_price * (1 - discount2)
  let cost3 : ℝ := original_price * (1 - discount3)

  let total_cost : ℝ := cost1 + cost2 + cost3

  let revenue1 : ℝ := (packs1 * masks_per_pack : ℝ) * sell_price1
  let revenue2 : ℝ := (packs2 * masks_per_pack : ℝ) * sell_price2
  let revenue3 : ℝ := (packs3 * masks_per_pack : ℝ) * sell_price3

  let total_revenue : ℝ := revenue1 + revenue2 + revenue3

  let total_profit : ℝ := total_revenue - total_cost

  total_profit = 371.5 := by sorry

end NUMINAMATH_CALUDE_face_mask_profit_l819_81921


namespace NUMINAMATH_CALUDE_inequality_solution_set_l819_81983

theorem inequality_solution_set (x : ℝ) :
  (1 / x < 1 / 2) ↔ x ∈ (Set.Ioi 2 ∪ Set.Iio 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l819_81983


namespace NUMINAMATH_CALUDE_only_four_and_eight_satisfy_l819_81956

/-- A natural number is a proper divisor of another natural number if it divides the number, is greater than 1, and is not equal to the number itself. -/
def IsProperDivisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d > 1 ∧ d ≠ n

/-- The set of proper divisors of a natural number. -/
def ProperDivisors (n : ℕ) : Set ℕ :=
  {d | IsProperDivisor d n}

/-- The property that all proper divisors of n, when increased by 1, form the set of proper divisors of m. -/
def SatisfiesProperty (n m : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), f = (· + 1) ∧
  (ProperDivisors m) = f '' (ProperDivisors n)

/-- The theorem stating that only 4 and 8 satisfy the given property. -/
theorem only_four_and_eight_satisfy :
  ∀ n : ℕ, (∃ m : ℕ, SatisfiesProperty n m) ↔ n = 4 ∨ n = 8 := by
  sorry


end NUMINAMATH_CALUDE_only_four_and_eight_satisfy_l819_81956


namespace NUMINAMATH_CALUDE_baseball_card_packs_l819_81995

/-- The number of people buying baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person bought -/
def cards_per_person : ℕ := 540

/-- The number of cards in each pack -/
def cards_per_pack : ℕ := 20

/-- The total number of packs of baseball cards -/
def total_packs : ℕ := (num_people * cards_per_person) / cards_per_pack

theorem baseball_card_packs :
  total_packs = 108 := by sorry

end NUMINAMATH_CALUDE_baseball_card_packs_l819_81995


namespace NUMINAMATH_CALUDE_partnership_annual_gain_l819_81940

/-- Represents the annual gain of a partnership given the following conditions:
    - A invests x at the beginning of the year
    - B invests 2x after 6 months
    - C invests 3x after 8 months
    - A's share is 6200
    - Profit is divided based on investment amount and time
-/
theorem partnership_annual_gain (x : ℝ) (total_gain : ℝ) : 
  x > 0 →
  (x * 12) / (x * 12 + 2 * x * 6 + 3 * x * 4) = 6200 / total_gain →
  total_gain = 18600 := by
  sorry

#check partnership_annual_gain

end NUMINAMATH_CALUDE_partnership_annual_gain_l819_81940


namespace NUMINAMATH_CALUDE_lemonade_scaling_l819_81934

/-- Lemonade recipe and scaling -/
theorem lemonade_scaling (lemons : ℕ) (sugar : ℚ) :
  (30 : ℚ) / 40 = lemons / 10 →
  (2 : ℚ) / 5 = sugar / 10 →
  lemons = 8 ∧ sugar = 4 := by
  sorry

#check lemonade_scaling

end NUMINAMATH_CALUDE_lemonade_scaling_l819_81934


namespace NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l819_81909

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l819_81909


namespace NUMINAMATH_CALUDE_bird_migration_l819_81919

theorem bird_migration (total : ℕ) (to_asia : ℕ) (difference : ℕ) : ℕ :=
  let to_africa := to_asia + difference
  to_africa

#check bird_migration 8 31 11 = 42

end NUMINAMATH_CALUDE_bird_migration_l819_81919


namespace NUMINAMATH_CALUDE_system_negative_solution_l819_81918

theorem system_negative_solution (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧
    a * x + b * y = c ∧
    b * x + c * y = a ∧
    c * x + a * y = b) ↔
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_system_negative_solution_l819_81918


namespace NUMINAMATH_CALUDE_inequalities_hold_l819_81946

theorem inequalities_hold (a b c x y z : ℝ) 
  (hx : x^2 < a^2) (hy : y^2 < b^2) (hz : z^2 < c^2) :
  (x^2*y^2 + y^2*z^2 + z^2*x^2 < a^2*b^2 + b^2*c^2 + c^2*a^2) ∧
  (x^4 + y^4 + z^4 < a^4 + b^4 + c^4) ∧
  (x^2*y^2*z^2 < a^2*b^2*c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l819_81946


namespace NUMINAMATH_CALUDE_angles_between_plane_and_legs_l819_81926

/-- Given a right triangle with an acute angle α and a plane through the smallest median
    forming an angle β with the triangle's plane, this theorem states the angles between
    the plane and the legs of the triangle. -/
theorem angles_between_plane_and_legs (α β : Real) 
  (h_acute : 0 < α ∧ α < Real.pi / 2)
  (h_right_triangle : True)  -- Placeholder for the right triangle condition
  (h_smallest_median : True) -- Placeholder for the smallest median condition
  (h_plane_angle : True)     -- Placeholder for the plane angle condition
  : ∃ (γ θ : Real),
    γ = Real.arcsin (Real.sin β * Real.cos α) ∧
    θ = Real.arcsin (Real.sin β * Real.sin α) :=
by sorry

end NUMINAMATH_CALUDE_angles_between_plane_and_legs_l819_81926


namespace NUMINAMATH_CALUDE_greatest_common_length_l819_81964

def cm_to_inch (cm : ℚ) : ℚ := cm / 2.54

def round_to_nearest_int (x : ℚ) : ℤ := 
  if x - x.floor < 0.5 then x.floor else x.ceil

def length1 : ℚ := 700
def length2 : ℚ := 385
def length3 : ℚ := 1295
def length4 : ℚ := 1545
def length5 : ℚ := 2663

theorem greatest_common_length : 
  Int.gcd 
    (round_to_nearest_int (cm_to_inch length1))
    (Int.gcd 
      (round_to_nearest_int (cm_to_inch length2))
      (Int.gcd 
        (round_to_nearest_int (cm_to_inch length3))
        (Int.gcd 
          (round_to_nearest_int (cm_to_inch length4))
          (round_to_nearest_int (cm_to_inch length5))))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_length_l819_81964


namespace NUMINAMATH_CALUDE_books_taken_out_monday_l819_81959

/-- The number of books taken out on Monday from a library -/
def books_taken_out (initial_books : ℕ) (books_returned : ℕ) (final_books : ℕ) : ℕ :=
  initial_books + books_returned - final_books

/-- Theorem stating that 124 books were taken out on Monday -/
theorem books_taken_out_monday : books_taken_out 336 22 234 = 124 := by
  sorry

end NUMINAMATH_CALUDE_books_taken_out_monday_l819_81959


namespace NUMINAMATH_CALUDE_rhombuses_in_five_by_five_grid_l819_81905

/-- Represents a grid of equilateral triangles -/
structure TriangleGrid where
  rows : Nat
  cols : Nat

/-- Calculates the number of rhombuses in a triangle grid -/
def count_rhombuses (grid : TriangleGrid) : Nat :=
  sorry

/-- Theorem stating that a 5x5 grid of equilateral triangles contains 30 rhombuses -/
theorem rhombuses_in_five_by_five_grid :
  let grid : TriangleGrid := { rows := 5, cols := 5 }
  count_rhombuses grid = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombuses_in_five_by_five_grid_l819_81905


namespace NUMINAMATH_CALUDE_max_savings_is_596_l819_81960

def plane_cost : ℚ := 600
def boat_cost : ℚ := 254
def helicopter_cost : ℚ := 850
def jetski_cost : ℚ := 175
def paragliding_cost : ℚ := 95

def total_jetski_paragliding_cost : ℚ := jetski_cost + 2 * paragliding_cost

def max_savings : ℚ := max (plane_cost - boat_cost) 
                         (max (helicopter_cost - boat_cost) 
                              (total_jetski_paragliding_cost - boat_cost))

theorem max_savings_is_596 : max_savings = 596 := by
  sorry

end NUMINAMATH_CALUDE_max_savings_is_596_l819_81960


namespace NUMINAMATH_CALUDE_jung_mi_number_problem_l819_81901

theorem jung_mi_number_problem :
  ∃ x : ℚ, (-4/5) * (x + (-2/3)) = -1/2 ∧ x = 31/24 := by
  sorry

end NUMINAMATH_CALUDE_jung_mi_number_problem_l819_81901


namespace NUMINAMATH_CALUDE_average_of_four_l819_81912

theorem average_of_four (total : ℕ) (avg_all : ℚ) (avg_two : ℚ) :
  total = 6 →
  avg_all = 8 →
  avg_two = 14 →
  (total * avg_all - 2 * avg_two) / (total - 2) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_average_of_four_l819_81912


namespace NUMINAMATH_CALUDE_oranges_taken_l819_81994

/-- Given a basket of oranges, prove that the number of oranges taken is 5 -/
theorem oranges_taken (original : ℕ) (remaining : ℕ) (taken : ℕ) : 
  original = 8 → remaining = 3 → taken = original - remaining → taken = 5 := by
  sorry

end NUMINAMATH_CALUDE_oranges_taken_l819_81994


namespace NUMINAMATH_CALUDE_beautiful_point_coordinates_l819_81991

/-- A point (x, y) is "beautiful" if x + y = x * y -/
def is_beautiful_point (x y : ℝ) : Prop := x + y = x * y

/-- The distance of a point (x, y) from the y-axis is |x| -/
def distance_from_y_axis (x : ℝ) : ℝ := |x|

theorem beautiful_point_coordinates :
  ∀ x y : ℝ, is_beautiful_point x y → distance_from_y_axis x = 2 →
  ((x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2/3)) :=
by sorry

end NUMINAMATH_CALUDE_beautiful_point_coordinates_l819_81991


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l819_81947

/-- A quadratic function is a function of the form f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem unique_quadratic_function (f : ℝ → ℝ) 
  (hf : QuadraticFunction f)
  (h0 : f 0 = -5)
  (h1 : f (-1) = -4)
  (h2 : f 2 = -5) :
  ∀ x, f x = (1/3) * x^2 - (2/3) * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_l819_81947


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l819_81968

theorem sunzi_wood_measurement (x y : ℝ) : 
  (x - y = 4.5 ∧ (1/2) * x + 1 = y) ↔ 
  (∃ (rope_length wood_length : ℝ),
    rope_length = x ∧
    wood_length = y ∧
    rope_length - wood_length = 4.5 ∧
    (1/2) * rope_length + 1 = wood_length) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l819_81968


namespace NUMINAMATH_CALUDE_quadratic_inequality_l819_81992

/-- A quadratic function with axis of symmetry at x = 1 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The axis of symmetry of f is at x = 1 -/
axiom axis_of_symmetry (b c : ℝ) : ∀ x, f b c (1 + x) = f b c (1 - x)

/-- The inequality f(1) < f(2) < f(-1) holds for the quadratic function f -/
theorem quadratic_inequality (b c : ℝ) : f b c 1 < f b c 2 ∧ f b c 2 < f b c (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l819_81992


namespace NUMINAMATH_CALUDE_two_digit_reverse_square_diff_l819_81963

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem two_digit_reverse_square_diff (n : ℕ) : 
  is_two_digit n ∧ 
  is_two_digit (reverse_digits n) ∧ 
  is_perfect_square (n * n - (reverse_digits n) * (reverse_digits n)) →
  n = 56 ∨ n = 65 := by sorry

end NUMINAMATH_CALUDE_two_digit_reverse_square_diff_l819_81963


namespace NUMINAMATH_CALUDE_jung_age_l819_81914

/-- Proves Jung's age given the ages of Li and Zhang and their relationships -/
theorem jung_age (li_age : ℕ) (zhang_age : ℕ) (jung_age : ℕ)
  (h1 : zhang_age = 2 * li_age)
  (h2 : li_age = 12)
  (h3 : jung_age = zhang_age + 2) :
  jung_age = 26 := by
sorry

end NUMINAMATH_CALUDE_jung_age_l819_81914


namespace NUMINAMATH_CALUDE_apple_count_l819_81943

theorem apple_count (red_apples green_apples total_apples : ℕ) : 
  red_apples = 16 →
  green_apples = red_apples + 12 →
  total_apples = red_apples + green_apples →
  total_apples = 44 := by
  sorry

end NUMINAMATH_CALUDE_apple_count_l819_81943


namespace NUMINAMATH_CALUDE_inequality_solution_set_l819_81927

theorem inequality_solution_set (x : ℝ) : (x + 1) / (x - 1) ≤ 0 ↔ x ∈ Set.Icc (-1) 1 \ {1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l819_81927


namespace NUMINAMATH_CALUDE_reciprocal_product_equals_19901_l819_81906

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a n / (1 + (n + 1) * a n * a (n + 1))

-- State the theorem
theorem reciprocal_product_equals_19901 :
  1 / (a 190 * a 200) = 19901 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_product_equals_19901_l819_81906


namespace NUMINAMATH_CALUDE_exponent_rule_multiplication_l819_81961

theorem exponent_rule_multiplication (a : ℝ) : a^4 * a^6 = a^10 := by sorry

end NUMINAMATH_CALUDE_exponent_rule_multiplication_l819_81961


namespace NUMINAMATH_CALUDE_scientific_notation_of_316000000_l819_81902

/-- Definition of scientific notation -/
def is_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ ∃ (x : ℝ), x = a * (10 : ℝ) ^ n

/-- The number to be represented in scientific notation -/
def number : ℝ := 316000000

/-- Theorem stating that 316000000 in scientific notation is 3.16 × 10^8 -/
theorem scientific_notation_of_316000000 :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n ∧ number = a * (10 : ℝ) ^ n ∧ a = 3.16 ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_316000000_l819_81902


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l819_81930

/-- Represents the rates of biking, jogging, and swimming in km/h -/
structure Rates where
  biking : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The sum of activities for Ed -/
def ed_sum (r : Rates) : ℕ := 3 * r.biking + 2 * r.jogging + 3 * r.swimming

/-- The sum of activities for Sue -/
def sue_sum (r : Rates) : ℕ := 5 * r.biking + 3 * r.jogging + 2 * r.swimming

/-- The sum of squares of the rates -/
def sum_of_squares (r : Rates) : ℕ := r.biking^2 + r.jogging^2 + r.swimming^2

theorem rates_sum_of_squares : 
  ∃ r : Rates, ed_sum r = 82 ∧ sue_sum r = 99 ∧ sum_of_squares r = 314 := by
  sorry

end NUMINAMATH_CALUDE_rates_sum_of_squares_l819_81930


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_l819_81972

theorem complex_magnitude_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_l819_81972


namespace NUMINAMATH_CALUDE_triangle_theorem_l819_81942

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def given_conditions (t : Triangle) : Prop :=
  t.b = 3 ∧
  t.b * t.c * Real.cos t.A = -6 ∧
  1/2 * t.b * t.c * Real.sin t.A = 3

-- State the theorem
theorem triangle_theorem (t : Triangle) :
  given_conditions t →
  t.A = Real.pi * 3/4 ∧ t.a = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l819_81942


namespace NUMINAMATH_CALUDE_chord_line_equation_l819_81917

/-- The equation of a line containing a chord of an ellipse --/
theorem chord_line_equation (x y : ℝ) :
  let ellipse := fun (x y : ℝ) ↦ x^2 / 16 + y^2 / 9 = 1
  let midpoint := (2, (3 : ℝ) / 2)
  let chord_line := fun (x y : ℝ) ↦ 3 * x + 4 * y - 12 = 0
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧
    ellipse x₂ y₂ ∧
    ((x₁ + x₂) / 2, (y₁ + y₂) / 2) = midpoint ∧
    (∀ x y, chord_line x y ↔ ∃ t, x = (1 - t) * x₁ + t * x₂ ∧ y = (1 - t) * y₁ + t * y₂) :=
by
  sorry

end NUMINAMATH_CALUDE_chord_line_equation_l819_81917


namespace NUMINAMATH_CALUDE_base_7_digits_of_1234_l819_81932

theorem base_7_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_7_digits_of_1234_l819_81932


namespace NUMINAMATH_CALUDE_special_numbers_theorem_l819_81941

def is_special_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 1000 * a + 100 * (b - a) + n % 100 ∧ 
                10 ≤ a ∧ a < 100 ∧ 
                0 ≤ b - a ∧ b - a < 100 ∧
                n = (a + (n % 100))^2

theorem special_numbers_theorem : 
  {n : ℕ | is_special_number n} = {3025, 2025, 9801} := by sorry

end NUMINAMATH_CALUDE_special_numbers_theorem_l819_81941


namespace NUMINAMATH_CALUDE_distance_height_relation_l819_81986

/-- An equilateral triangle with an arbitrary line in its plane -/
structure TriangleWithLine where
  /-- The side length of the equilateral triangle -/
  side : ℝ
  /-- The height of the equilateral triangle -/
  height : ℝ
  /-- The distance from the first vertex to the line -/
  m : ℝ
  /-- The distance from the second vertex to the line -/
  n : ℝ
  /-- The distance from the third vertex to the line -/
  p : ℝ
  /-- The side length is positive -/
  side_pos : 0 < side
  /-- The height is related to the side length as in an equilateral triangle -/
  height_eq : height = (Real.sqrt 3 / 2) * side

/-- The main theorem stating the relationship between distances and height -/
theorem distance_height_relation (t : TriangleWithLine) :
  (t.m - t.n)^2 + (t.n - t.p)^2 + (t.p - t.m)^2 = 2 * t.height^2 := by
  sorry

end NUMINAMATH_CALUDE_distance_height_relation_l819_81986


namespace NUMINAMATH_CALUDE_evaluate_expression_l819_81989

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  3 * x^y + 4 * y^x = 59 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l819_81989


namespace NUMINAMATH_CALUDE_final_values_l819_81933

def program_execution (a b : Int) : Int × Int :=
  let a' := a + b
  let b' := a' - b
  (a', b')

theorem final_values : program_execution 1 3 = (4, 1) := by
  sorry

end NUMINAMATH_CALUDE_final_values_l819_81933


namespace NUMINAMATH_CALUDE_sqrt_64_times_sqrt_25_l819_81957

theorem sqrt_64_times_sqrt_25 : Real.sqrt (64 * Real.sqrt 25) = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_64_times_sqrt_25_l819_81957


namespace NUMINAMATH_CALUDE_perfect_cube_property_l819_81951

theorem perfect_cube_property (x y : ℕ+) (h : ∃ k : ℕ+, x * y^2 = k^3) :
  ∃ m : ℕ+, x^2 * y = m^3 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_property_l819_81951


namespace NUMINAMATH_CALUDE_min_gennadys_correct_l819_81916

/-- Represents the number of people with a specific name at the festival -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat
  gennadies : Nat

/-- Checks if the given name counts satisfy the festival conditions -/
def satisfiesConditions (counts : NameCount) : Prop :=
  counts.alexanders = 45 ∧
  counts.borises = 122 ∧
  counts.vasilies = 27 ∧
  counts.alexanders + counts.borises + counts.vasilies + counts.gennadies - 1 ≥ counts.borises

/-- The minimum number of Gennadys required for the festival -/
def minGennadys : Nat := 49

/-- Theorem stating that the minimum number of Gennadys is correct -/
theorem min_gennadys_correct :
  (∀ counts : NameCount, satisfiesConditions counts → counts.gennadies ≥ minGennadys) ∧
  (∃ counts : NameCount, satisfiesConditions counts ∧ counts.gennadies = minGennadys) := by
  sorry

#check min_gennadys_correct

end NUMINAMATH_CALUDE_min_gennadys_correct_l819_81916


namespace NUMINAMATH_CALUDE_recurring_decimal_product_l819_81990

/-- Represents a recurring decimal with a single digit repeating -/
def recurring_decimal_single (n : ℕ) : ℚ :=
  n / 9

/-- Represents a recurring decimal with two digits repeating -/
def recurring_decimal_double (n : ℕ) : ℚ :=
  n / 99

/-- The product of 0.1̅ and 0.23̅ is equal to 23/891 -/
theorem recurring_decimal_product :
  (recurring_decimal_single 1) * (recurring_decimal_double 23) = 23 / 891 := by
  sorry

#eval (1 / 9 : ℚ) * (23 / 99 : ℚ) == 23 / 891  -- For verification

end NUMINAMATH_CALUDE_recurring_decimal_product_l819_81990


namespace NUMINAMATH_CALUDE_solution_set_f_min_value_2a_plus_b_min_value_2a_plus_b_is_9_8_l819_81903

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + 1 + |3 - x|

-- Theorem for the solution set of f(x) ≤ 6
theorem solution_set_f (x : ℝ) (h : x ≥ -1) :
  f x ≤ 6 ↔ -1 ≤ x ∧ x ≤ 4 := by sorry

-- Define n as the minimum value of f(x)
def n : ℝ := 4

-- Theorem for the minimum value of 2a + b
theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * n * a * b = a + 2 * b) :
  2 * a + b ≥ 9/8 := by sorry

-- Theorem stating that 9/8 is indeed the minimum value
theorem min_value_2a_plus_b_is_9_8 :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * n * a * b = a + 2 * b ∧ 2 * a + b = 9/8 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_min_value_2a_plus_b_min_value_2a_plus_b_is_9_8_l819_81903


namespace NUMINAMATH_CALUDE_rectangle_area_l819_81987

/-- The area of a rectangle formed by three identical smaller rectangles -/
theorem rectangle_area (shorter_side : ℝ) (h : shorter_side = 7) : 
  let longer_side := 3 * shorter_side
  let large_rectangle_length := 3 * shorter_side
  let large_rectangle_width := longer_side
  large_rectangle_length * large_rectangle_width = 441 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l819_81987


namespace NUMINAMATH_CALUDE_vertex_in_fourth_quadrant_l819_81939

-- Define the line y = x + m
def line (x m : ℝ) : ℝ := x + m

-- Define the parabola y = (x + m)^2 - 1
def parabola (x m : ℝ) : ℝ := (x + m)^2 - 1

-- Define what it means for a line to pass through the first, third, and fourth quadrants
def passes_through_134 (m : ℝ) : Prop :=
  ∃ (x1 x3 x4 : ℝ), 
    (x1 > 0 ∧ line x1 m > 0) ∧
    (x3 < 0 ∧ line x3 m < 0) ∧
    (x4 > 0 ∧ line x4 m < 0)

-- Define the fourth quadrant
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem vertex_in_fourth_quadrant (m : ℝ) :
  passes_through_134 m → in_fourth_quadrant (-m) (-1) :=
sorry

end NUMINAMATH_CALUDE_vertex_in_fourth_quadrant_l819_81939


namespace NUMINAMATH_CALUDE_sqrt_calculations_l819_81965

theorem sqrt_calculations :
  (∀ (x y : ℝ), x > 0 → y > 0 → ∃ (z : ℝ), z > 0 ∧ z * z = x * y) →
  (∀ (x y : ℝ), x > 0 → y > 0 → ∃ (z : ℝ), z > 0 ∧ z * z = x / y) →
  (∃ (sqrt10 sqrt2 sqrt15 sqrt3 sqrt5 sqrt27 sqrt12 sqrt_third : ℝ),
    sqrt10 > 0 ∧ sqrt10 * sqrt10 = 10 ∧
    sqrt2 > 0 ∧ sqrt2 * sqrt2 = 2 ∧
    sqrt15 > 0 ∧ sqrt15 * sqrt15 = 15 ∧
    sqrt3 > 0 ∧ sqrt3 * sqrt3 = 3 ∧
    sqrt5 > 0 ∧ sqrt5 * sqrt5 = 5 ∧
    sqrt27 > 0 ∧ sqrt27 * sqrt27 = 27 ∧
    sqrt12 > 0 ∧ sqrt12 * sqrt12 = 12 ∧
    sqrt_third > 0 ∧ sqrt_third * sqrt_third = 1/3 ∧
    sqrt10 * sqrt2 + sqrt15 / sqrt3 = 3 * sqrt5 ∧
    sqrt27 - (sqrt12 - sqrt_third) = 4/3 * sqrt3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l819_81965


namespace NUMINAMATH_CALUDE_stacy_has_32_berries_l819_81988

/-- The number of berries Skylar has -/
def skylar_berries : ℕ := 20

/-- The number of berries Steve has -/
def steve_berries : ℕ := skylar_berries / 2

/-- The number of berries Stacy has -/
def stacy_berries : ℕ := 3 * steve_berries + 2

/-- Theorem stating that Stacy has 32 berries -/
theorem stacy_has_32_berries : stacy_berries = 32 := by
  sorry

end NUMINAMATH_CALUDE_stacy_has_32_berries_l819_81988


namespace NUMINAMATH_CALUDE_count_triples_sum_8_l819_81962

/-- The number of ordered triples of natural numbers that sum to a given natural number. -/
def count_triples (n : ℕ) : ℕ := Nat.choose (n + 2) 2

/-- Theorem stating that the number of ordered triples (A, B, C) of natural numbers
    that satisfy A + B + C = 8 is equal to 21. -/
theorem count_triples_sum_8 : count_triples 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_count_triples_sum_8_l819_81962


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_range_of_b_when_a_zero_sum_of_squares_greater_than_e_l819_81998

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - a * Real.sin x
def g (x : ℝ) : ℝ := b * Real.sqrt x

-- Tangent line equation
theorem tangent_line_at_zero :
  ∃ m c : ℝ, ∀ x : ℝ, m * x + c = (1 - a) * x + 1 :=
sorry

-- Range of b when a = 0
theorem range_of_b_when_a_zero (h : a = 0) :
  ∃ x > 0, f x = g x ↔ b ≥ Real.sqrt (2 * Real.exp 1) :=
sorry

-- Proof of a^2 + b^2 > e
theorem sum_of_squares_greater_than_e (h : ∃ x > 0, f x = g x) :
  a^2 + b^2 > Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_range_of_b_when_a_zero_sum_of_squares_greater_than_e_l819_81998


namespace NUMINAMATH_CALUDE_simplify_expression_l819_81938

theorem simplify_expression : 
  (Real.sqrt 8 + Real.sqrt 12) - (2 * Real.sqrt 3 - Real.sqrt 2) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l819_81938


namespace NUMINAMATH_CALUDE_simplify_fraction_sum_l819_81915

theorem simplify_fraction_sum (a b c d : ℕ) : 
  a = 75 → b = 135 → 
  (∃ (k : ℕ), k * c = a ∧ k * d = b) → 
  (∀ (m : ℕ), m * c = a ∧ m * d = b → m ≤ k) →
  c + d = 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_sum_l819_81915


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l819_81931

theorem exam_maximum_marks :
  ∀ (total_marks : ℕ) (passing_percentage : ℚ) (student_score : ℕ) (failing_margin : ℕ),
    passing_percentage = 1/4 →
    student_score = 185 →
    failing_margin = 25 →
    (passing_percentage * total_marks : ℚ) = (student_score + failing_margin) →
    total_marks = 840 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l819_81931


namespace NUMINAMATH_CALUDE_rate_of_discount_l819_81925

/-- Calculate the rate of discount given the marked price and selling price -/
theorem rate_of_discount (marked_price selling_price : ℝ) :
  marked_price = 200 →
  selling_price = 120 →
  (marked_price - selling_price) / marked_price * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_rate_of_discount_l819_81925


namespace NUMINAMATH_CALUDE_automobile_distance_l819_81935

/-- Proves that an automobile with given acceleration travels a specific distance. -/
theorem automobile_distance (a : ℝ) : 
  let acceleration := a / 12 -- feet per second squared
  let time := 2 * 60 -- 2 minutes in seconds
  let distance_feet := (1 / 2) * acceleration * time^2
  let distance_yards := distance_feet / 3
  distance_yards = 200 * a := by sorry

end NUMINAMATH_CALUDE_automobile_distance_l819_81935


namespace NUMINAMATH_CALUDE_walnut_problem_l819_81910

/-- Calculates the final number of walnuts in the main burrow after the actions of three squirrels. -/
def final_walnut_count (initial : ℕ) (boy_gather boy_drop boy_hide : ℕ)
  (girl_bring girl_eat girl_give girl_lose girl_knock : ℕ)
  (third_gather third_drop third_hide third_return third_give : ℕ) : ℕ :=
  initial + boy_gather - boy_drop - boy_hide +
  girl_bring - girl_eat - girl_give - girl_lose - girl_knock +
  third_return

/-- The final number of walnuts in the main burrow is 44. -/
theorem walnut_problem :
  final_walnut_count 30 20 4 8 15 5 4 3 2 10 1 3 6 1 = 44 := by
  sorry

end NUMINAMATH_CALUDE_walnut_problem_l819_81910


namespace NUMINAMATH_CALUDE_infinitely_many_special_numbers_l819_81984

/-- Sum of digits of a natural number's decimal representation -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for special numbers -/
def is_special (m : ℕ) : Prop :=
  ∀ n : ℕ, m ≠ n + sum_of_digits n

/-- Theorem stating that there are infinitely many special numbers -/
theorem infinitely_many_special_numbers :
  ∀ k : ℕ, ∃ S : Finset ℕ, (∀ m ∈ S, is_special m) ∧ S.card > k :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_numbers_l819_81984


namespace NUMINAMATH_CALUDE_shopkeeper_oranges_l819_81993

/-- The number of oranges bought by a shopkeeper -/
def oranges : ℕ := sorry

/-- The number of bananas bought by the shopkeeper -/
def bananas : ℕ := 400

/-- The percentage of oranges that are not rotten -/
def good_orange_percentage : ℚ := 85 / 100

/-- The percentage of bananas that are not rotten -/
def good_banana_percentage : ℚ := 95 / 100

/-- The overall percentage of fruits in good condition -/
def total_good_percentage : ℚ := 89 / 100

theorem shopkeeper_oranges :
  (good_orange_percentage * oranges + good_banana_percentage * bananas) / (oranges + bananas) = total_good_percentage ∧
  oranges = 600 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_oranges_l819_81993


namespace NUMINAMATH_CALUDE_ellipse_properties_l819_81976

/-- Prove that an ellipse with given properties has specific semi-major and semi-minor axes -/
theorem ellipse_properties (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (∀ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c > 0 ∧ c^2 = m^2 - n^2) →  -- Ellipse property: c^2 = a^2 - b^2
  (2 : ℝ) = m - (m^2 - n^2).sqrt →  -- Right focus at (2, 0)
  (1 / 2 : ℝ) = (m^2 - n^2).sqrt / m →  -- Eccentricity is 1/2
  m^2 = 16 ∧ n^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l819_81976


namespace NUMINAMATH_CALUDE_tic_tac_toe_rounds_difference_l819_81953

theorem tic_tac_toe_rounds_difference 
  (total_rounds : ℕ) 
  (william_wins : ℕ) 
  (h1 : total_rounds = 15) 
  (h2 : william_wins = 10) 
  (h3 : william_wins > total_rounds - william_wins) : 
  william_wins - (total_rounds - william_wins) = 5 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_rounds_difference_l819_81953


namespace NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l819_81929

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Check if four points form a rectangle --/
def isRectangle (r : Rectangle) : Prop :=
  let (x1, y1) := r.v1
  let (x2, y2) := r.v2
  let (x3, y3) := r.v3
  let (x4, y4) := r.v4
  (x1 = x3 ∧ x2 = x4 ∧ y1 = y2 ∧ y3 = y4) ∨
  (x1 = x2 ∧ x3 = x4 ∧ y1 = y3 ∧ y2 = y4)

/-- The main theorem --/
theorem fourth_vertex_of_rectangle :
  ∀ (r : Rectangle),
    r.v1 = (1, 1) →
    r.v2 = (5, 1) →
    r.v3 = (1, 7) →
    isRectangle r →
    r.v4 = (5, 7) := by
  sorry


end NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l819_81929


namespace NUMINAMATH_CALUDE_cube_projection_sum_squares_zero_l819_81982

/-- Represents a vertex of a cube -/
structure CubeVertex where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the orthogonal projection of a cube vertex onto a complex plane -/
def project (v : CubeVertex) : ℂ :=
  Complex.mk v.x v.y

/-- Given four vertices of a cube where three are adjacent to the fourth,
    and their orthogonal projections onto a complex plane,
    the sum of the squares of the projected complex numbers is zero. -/
theorem cube_projection_sum_squares_zero
  (V V₁ V₂ V₃ : CubeVertex)
  (adj₁ : V₁.x = V.x ∨ V₁.y = V.y ∨ V₁.z = V.z)
  (adj₂ : V₂.x = V.x ∨ V₂.y = V.y ∨ V₂.z = V.z)
  (adj₃ : V₃.x = V.x ∨ V₃.y = V.y ∨ V₃.z = V.z)
  (origin_proj : project V = 0)
  : (project V₁)^2 + (project V₂)^2 + (project V₃)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_projection_sum_squares_zero_l819_81982


namespace NUMINAMATH_CALUDE_egyptian_pi_approximation_l819_81977

theorem egyptian_pi_approximation (d : ℝ) (h : d > 0) :
  (π * d^2 / 4 = (8 * d / 9)^2) → π = 256 / 81 := by
  sorry

end NUMINAMATH_CALUDE_egyptian_pi_approximation_l819_81977


namespace NUMINAMATH_CALUDE_frances_towel_weight_frances_towel_weight_is_192_ounces_l819_81936

/-- Calculates the weight of Frances's towels in ounces given the conditions of the problem. -/
theorem frances_towel_weight (mary_towels : ℕ) (total_weight : ℕ) (mary_frances_ratio : ℕ) : ℕ :=
  let frances_towels := mary_towels / mary_frances_ratio
  let total_towels := mary_towels + frances_towels
  let weight_per_towel := total_weight / total_towels
  let frances_weight_pounds := weight_per_towel * frances_towels
  frances_weight_pounds * 16

/-- Proves that Frances's towels weigh 192 ounces given the conditions of the problem. -/
theorem frances_towel_weight_is_192_ounces : frances_towel_weight 24 60 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_frances_towel_weight_frances_towel_weight_is_192_ounces_l819_81936


namespace NUMINAMATH_CALUDE_fraction_problem_l819_81911

theorem fraction_problem (A B x : ℝ) : 
  A + B = 27 → 
  B = 15 → 
  0.5 * A + x * B = 11 → 
  x = 1/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l819_81911


namespace NUMINAMATH_CALUDE_donna_dog_walking_rate_l819_81924

def dog_walking_hours : ℕ := 2 * 7
def card_shop_earnings : ℚ := 2 * 5 * 12.5
def babysitting_earnings : ℚ := 4 * 10
def total_earnings : ℚ := 305

theorem donna_dog_walking_rate : 
  ∃ (rate : ℚ), rate * dog_walking_hours + card_shop_earnings + babysitting_earnings = total_earnings ∧ rate = 10 := by
sorry

end NUMINAMATH_CALUDE_donna_dog_walking_rate_l819_81924


namespace NUMINAMATH_CALUDE_expression_equality_l819_81958

theorem expression_equality (n : ℕ) (h : n ≥ 1) :
  (5^(n+1) + 6^(n+2))^2 - (5^(n+1) - 6^(n+2))^2 = 144 * 30^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l819_81958


namespace NUMINAMATH_CALUDE_tan_theta_value_l819_81996

theorem tan_theta_value (θ : Real) (x y : Real) : 
  x = -Real.sqrt 3 / 2 ∧ y = 1 / 2 → Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l819_81996


namespace NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l819_81974

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem product_of_symmetric_complex_numbers :
  ∀ z₁ z₂ : ℂ, symmetric_wrt_imaginary_axis z₁ z₂ → z₁ = 2 + I → z₁ * z₂ = -5 := by
  sorry

#check product_of_symmetric_complex_numbers

end NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l819_81974


namespace NUMINAMATH_CALUDE_current_price_calculation_l819_81907

/-- The current unit price after price adjustments -/
def current_price (x : ℝ) : ℝ := (1 - 0.25) * (x + 10)

/-- Theorem stating that the current price calculation is correct -/
theorem current_price_calculation (x : ℝ) : 
  current_price x = (1 - 0.25) * (x + 10) := by
  sorry

end NUMINAMATH_CALUDE_current_price_calculation_l819_81907


namespace NUMINAMATH_CALUDE_group1_larger_than_group2_l819_81980

/-- A point on a circle -/
structure CirclePoint where
  angle : ℝ

/-- A convex polygon formed by points on a circle -/
structure ConvexPolygon where
  vertices : List CirclePoint
  is_convex : Bool

/-- The set of n points on the circle -/
def circle_points (n : ℕ) : List CirclePoint :=
  sorry

/-- Group 1: Polygons that include A₁ as a vertex -/
def group1 (n : ℕ) : List ConvexPolygon :=
  sorry

/-- Group 2: Polygons that do not include A₁ as a vertex -/
def group2 (n : ℕ) : List ConvexPolygon :=
  sorry

/-- Theorem: Group 1 contains more polygons than Group 2 -/
theorem group1_larger_than_group2 (n : ℕ) : 
  (group1 n).length > (group2 n).length :=
  sorry

end NUMINAMATH_CALUDE_group1_larger_than_group2_l819_81980


namespace NUMINAMATH_CALUDE_frustum_cone_volume_l819_81949

theorem frustum_cone_volume (frustum_volume : ℝ) (base_area_ratio : ℝ) (cone_volume : ℝ) :
  frustum_volume = 78 →
  base_area_ratio = 9 →
  (cone_volume - frustum_volume) / cone_volume = (1 / base_area_ratio.sqrt)^3 →
  cone_volume = 81 := by
sorry

end NUMINAMATH_CALUDE_frustum_cone_volume_l819_81949


namespace NUMINAMATH_CALUDE_line_segment_length_l819_81997

/-- The hyperbola C with equation x^2 - y^2/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The line l with equation y = 2√3x + m -/
def line (x y m : ℝ) : Prop := y = 2 * Real.sqrt 3 * x + m

/-- The right vertex of the hyperbola -/
def right_vertex (x y : ℝ) : Prop := hyperbola x y ∧ x > 0 ∧ y = 0

/-- The asymptotes of the hyperbola -/
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- The intersection points of the line and the asymptotes -/
def intersection_points (x₁ y₁ x₂ y₂ m : ℝ) : Prop :=
  line x₁ y₁ m ∧ asymptote x₁ y₁ ∧
  line x₂ y₂ m ∧ asymptote x₂ y₂ ∧
  x₁ ≠ x₂

/-- The theorem statement -/
theorem line_segment_length 
  (x y m x₁ y₁ x₂ y₂ : ℝ) :
  right_vertex x y →
  line x y m →
  intersection_points x₁ y₁ x₂ y₂ m →
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 * Real.sqrt 13 / 3 :=
sorry

end NUMINAMATH_CALUDE_line_segment_length_l819_81997


namespace NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l819_81967

-- Define the universal set U as ℝ
def U := ℝ

-- Define set M
def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := {x : ℝ | x ≥ 1}

-- Theorem statement
theorem intersection_of_M_and_complement_of_N :
  M ∩ (Set.univ \ N) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l819_81967


namespace NUMINAMATH_CALUDE_parabola_c_value_l819_81971

/-- A parabola passing through two points with its vertex on a line -/
theorem parabola_c_value (b c : ℝ) : 
  (∀ x, (x^2 + b*x + c) = 10 → x = 2 ∨ x = -2) →  -- parabola passes through (2, 10) and (-2, 6)
  (∃ x, x^2 + b*x + c = 6 ∧ x = -2) →             -- parabola passes through (-2, 6)
  (∃ x, x^2 + b*x + c = -x + 4) →                 -- vertex lies on y = -x + 4
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l819_81971


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l819_81999

theorem at_least_one_equation_has_two_roots (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (4 * b^2 - 4 * a * c > 0) ∨ 
  (4 * c^2 - 4 * a * b > 0) ∨ 
  (4 * a^2 - 4 * b * c > 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l819_81999


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l819_81985

/-- Given two points P and Q in a Cartesian coordinate system that are symmetric
    with respect to the x-axis, prove that the sum of their x-coordinate and
    the y-coordinate (before the shift) is 3. -/
theorem symmetric_points_sum (a b : ℝ) : 
  (a - 3 = 2) →  -- x-coordinates are equal
  (1 = -(b + 1)) →  -- y-coordinates are opposites
  a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l819_81985


namespace NUMINAMATH_CALUDE_right_triangle_from_equations_l819_81979

theorem right_triangle_from_equations (a b c x : ℝ) :
  (∃ α : ℝ, α^2 + 2*a*α + b^2 = 0 ∧ α^2 + 2*c*α - b^2 = 0) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  a^2 = b^2 + c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_from_equations_l819_81979


namespace NUMINAMATH_CALUDE_alyssas_attended_games_l819_81944

theorem alyssas_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 31) 
  (h2 : missed_games = 18) : 
  total_games - missed_games = 13 := by
  sorry

end NUMINAMATH_CALUDE_alyssas_attended_games_l819_81944


namespace NUMINAMATH_CALUDE_rational_coloring_exists_l819_81913

theorem rational_coloring_exists : ∃ (f : ℚ → Bool), 
  (∀ x : ℚ, x ≠ 0 → f x ≠ f (-x)) ∧ 
  (∀ x : ℚ, x ≠ 1/2 → f x ≠ f (1 - x)) ∧ 
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → f x ≠ f (1 / x)) := by
  sorry

end NUMINAMATH_CALUDE_rational_coloring_exists_l819_81913


namespace NUMINAMATH_CALUDE_time_difference_l819_81923

def brian_time : ℕ := 96
def todd_time : ℕ := 88

theorem time_difference : brian_time - todd_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_l819_81923


namespace NUMINAMATH_CALUDE_sequences_theorem_l819_81970

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = 2

/-- Sequence b satisfying b_{n+1} - b_n = a_n -/
def b_sequence (a b : ℕ → ℤ) : Prop :=
  ∀ n, b (n + 1) - b n = a n

/-- Main theorem about sequences a and b -/
theorem sequences_theorem (a b : ℕ → ℤ) 
  (h1 : arithmetic_sequence a)
  (h2 : b_sequence a b)
  (h3 : b 2 = -18)
  (h4 : b 3 = -24) :
  (∀ n, a n = 2 * n - 10) ∧
  (b 5 = -30 ∧ b 6 = -30 ∧ ∀ n, b n ≥ -30) :=
sorry

end NUMINAMATH_CALUDE_sequences_theorem_l819_81970


namespace NUMINAMATH_CALUDE_x_equation_implies_a_plus_b_l819_81904

theorem x_equation_implies_a_plus_b (x : ℝ) (a b : ℕ+) :
  x^2 + 5*x + 5/x + 1/x^2 = 34 →
  x = a + Real.sqrt b →
  (a : ℝ) + b = 5 := by sorry

end NUMINAMATH_CALUDE_x_equation_implies_a_plus_b_l819_81904


namespace NUMINAMATH_CALUDE_randy_tower_blocks_l819_81900

/-- 
Given:
- Randy has 90 blocks in total
- He uses 89 blocks to build a house
- He uses some blocks to build a tower
- He used 26 more blocks for the house than for the tower

Prove that Randy used 63 blocks to build the tower.
-/
theorem randy_tower_blocks : 
  ∀ (total house tower : ℕ),
  total = 90 →
  house = 89 →
  house = tower + 26 →
  tower = 63 := by
sorry

end NUMINAMATH_CALUDE_randy_tower_blocks_l819_81900


namespace NUMINAMATH_CALUDE_min_fraction_value_l819_81978

theorem min_fraction_value (x y : ℝ) (hx : 3 ≤ x ∧ x ≤ 5) (hy : -5 ≤ y ∧ y ≤ -3) :
  (x + y) / x ≥ 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_value_l819_81978


namespace NUMINAMATH_CALUDE_triangle_abc_isosceles_l819_81973

/-- Given points A(3,5), B(-6,-2), and C(0,-6), prove that AB = AC -/
theorem triangle_abc_isosceles (A B C : ℝ × ℝ) : 
  A = (3, 5) → B = (-6, -2) → C = (0, -6) → 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 := by
  sorry

#check triangle_abc_isosceles

end NUMINAMATH_CALUDE_triangle_abc_isosceles_l819_81973


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l819_81922

theorem sum_of_a_and_b (a b : ℝ) 
  (h1 : (a + Real.sqrt b) + (a - Real.sqrt b) = -6)
  (h2 : (a + Real.sqrt b) * (a - Real.sqrt b) = 4) :
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l819_81922


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l819_81981

theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x : ℝ, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → 
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l819_81981
