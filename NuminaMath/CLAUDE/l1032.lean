import Mathlib

namespace NUMINAMATH_CALUDE_count_theorem_l1032_103264

/-- Count of numbers between 100 and 799 with digits in strictly increasing order -/
def strictlyIncreasingCount : Nat := Nat.choose 7 3

/-- Count of numbers between 100 and 799 with last two digits equal -/
def lastTwoEqualCount : Nat := Nat.choose 7 2

/-- Total count of numbers between 100 and 799 with digits in strictly increasing order or equal to the last digit -/
def totalCount : Nat := strictlyIncreasingCount + lastTwoEqualCount

theorem count_theorem : totalCount = 56 := by sorry

end NUMINAMATH_CALUDE_count_theorem_l1032_103264


namespace NUMINAMATH_CALUDE_agathas_bike_purchase_l1032_103224

/-- Agatha's bike purchase problem -/
theorem agathas_bike_purchase (frame_cost seat_handlebar_cost front_wheel_cost remaining_money : ℕ) 
  (h1 : frame_cost = 15)
  (h2 : front_wheel_cost = 25)
  (h3 : remaining_money = 20) :
  frame_cost + front_wheel_cost + remaining_money = 60 := by
  sorry

#check agathas_bike_purchase

end NUMINAMATH_CALUDE_agathas_bike_purchase_l1032_103224


namespace NUMINAMATH_CALUDE_perpendicular_sum_l1032_103277

/-- Given vectors a and b in ℝ², if a + b is perpendicular to a, then the second component of b is -4. -/
theorem perpendicular_sum (a b : ℝ × ℝ) (h : a.1 = 1 ∧ a.2 = 3 ∧ b.2 = -2) :
  (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 0 → b.1 = -4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_sum_l1032_103277


namespace NUMINAMATH_CALUDE_quadratic_equations_roots_l1032_103237

theorem quadratic_equations_roots :
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = -5 ∧ x₁^2 + 6*x₁ + 5 = 0 ∧ x₂^2 + 6*x₂ + 5 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 2 + Real.sqrt 5 ∧ y₂ = 2 - Real.sqrt 5 ∧ y₁^2 - 4*y₁ - 1 = 0 ∧ y₂^2 - 4*y₂ - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_roots_l1032_103237


namespace NUMINAMATH_CALUDE_olly_ferrets_l1032_103209

theorem olly_ferrets (num_dogs : ℕ) (num_cats : ℕ) (total_shoes : ℕ) :
  num_dogs = 3 →
  num_cats = 2 →
  total_shoes = 24 →
  ∃ (num_ferrets : ℕ),
    num_ferrets * 4 + num_dogs * 4 + num_cats * 4 = total_shoes ∧
    num_ferrets = 1 :=
by sorry

end NUMINAMATH_CALUDE_olly_ferrets_l1032_103209


namespace NUMINAMATH_CALUDE_max_value_rational_function_l1032_103261

theorem max_value_rational_function : 
  ∃ (n : ℕ), n = 97 ∧ 
  (∀ x : ℝ, (4 * x^2 + 12 * x + 29) / (4 * x^2 + 12 * x + 5) ≤ n) ∧
  (∀ m : ℕ, m > n → ∃ x : ℝ, (4 * x^2 + 12 * x + 29) / (4 * x^2 + 12 * x + 5) < m) :=
by sorry

end NUMINAMATH_CALUDE_max_value_rational_function_l1032_103261


namespace NUMINAMATH_CALUDE_faraway_impossible_totals_l1032_103285

/-- Represents the number of creatures in Faraway village -/
structure FarawayVillage where
  horses : ℕ
  goats : ℕ

/-- The total number of creatures in Faraway village -/
def total_creatures (v : FarawayVillage) : ℕ :=
  21 * v.horses + 6 * v.goats

/-- Theorem stating that 74 and 89 cannot be the total number of creatures -/
theorem faraway_impossible_totals :
  ¬ ∃ (v : FarawayVillage), total_creatures v = 74 ∨ total_creatures v = 89 := by
  sorry

end NUMINAMATH_CALUDE_faraway_impossible_totals_l1032_103285


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1032_103210

/-- The ellipse defined by x^2/9 + y^2/4 = 1 -/
def ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) + (p.2^2 / 4) = 1}

/-- The length of the major axis of the ellipse -/
def major_axis_length : ℝ := 6

/-- Theorem: The length of the major axis of the ellipse defined by x^2/9 + y^2/4 = 1 is 6 -/
theorem ellipse_major_axis_length : 
  ∀ p ∈ ellipse, major_axis_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1032_103210


namespace NUMINAMATH_CALUDE_jacob_calorie_limit_l1032_103297

/-- Jacob's calorie intake and limit problem -/
theorem jacob_calorie_limit :
  ∀ (breakfast lunch dinner total_eaten planned_limit : ℕ),
    breakfast = 400 →
    lunch = 900 →
    dinner = 1100 →
    total_eaten = breakfast + lunch + dinner →
    total_eaten = planned_limit + 600 →
    planned_limit = 1800 := by
  sorry

end NUMINAMATH_CALUDE_jacob_calorie_limit_l1032_103297


namespace NUMINAMATH_CALUDE_rumor_day_seven_l1032_103246

def rumor_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

theorem rumor_day_seven :
  (∀ k < 7, rumor_spread k < 3280) ∧ rumor_spread 7 ≥ 3280 := by
  sorry

end NUMINAMATH_CALUDE_rumor_day_seven_l1032_103246


namespace NUMINAMATH_CALUDE_max_ratio_squared_l1032_103214

theorem max_ratio_squared (a b x z : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b)
  (hx : 0 ≤ x) (hxa : x < a) (hz : 0 ≤ z) (hzb : z < b)
  (heq : a^2 + z^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b - z)^2) :
  (a / b)^2 ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l1032_103214


namespace NUMINAMATH_CALUDE_single_room_cost_l1032_103213

/-- Proves that the cost of each single room is $35 given the hotel booking information -/
theorem single_room_cost (total_rooms : ℕ) (double_room_cost : ℕ) (total_revenue : ℕ) (double_rooms : ℕ)
  (h1 : total_rooms = 260)
  (h2 : double_room_cost = 60)
  (h3 : total_revenue = 14000)
  (h4 : double_rooms = 196) :
  (total_revenue - double_rooms * double_room_cost) / (total_rooms - double_rooms) = 35 := by
  sorry

#check single_room_cost

end NUMINAMATH_CALUDE_single_room_cost_l1032_103213


namespace NUMINAMATH_CALUDE_f_cos_x_equals_two_plus_cos_two_x_l1032_103266

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_cos_x_equals_two_plus_cos_two_x (x : ℝ) : 
  (∀ y : ℝ, f (Real.sin y) = 2 - Real.cos (2 * y)) → 
  f (Real.cos x) = 2 + Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_f_cos_x_equals_two_plus_cos_two_x_l1032_103266


namespace NUMINAMATH_CALUDE_largest_special_square_proof_l1032_103290

/-- A number is a perfect square if it's the square of an integer -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- Remove the last two digits of a natural number -/
def remove_last_two_digits (n : ℕ) : ℕ := n / 100

/-- The largest perfect square satisfying the given conditions -/
def largest_special_square : ℕ := 1681

theorem largest_special_square_proof :
  (is_perfect_square largest_special_square) ∧ 
  (is_perfect_square (remove_last_two_digits largest_special_square)) ∧ 
  (largest_special_square % 100 ≠ 0) ∧
  (∀ n : ℕ, n > largest_special_square → 
    ¬(is_perfect_square n ∧ 
      is_perfect_square (remove_last_two_digits n) ∧ 
      n % 100 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_largest_special_square_proof_l1032_103290


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1032_103258

theorem quadratic_inequality (a b c : ℝ) (h : (a + b + c) * c ≤ 0) : b^2 ≥ 4*a*c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1032_103258


namespace NUMINAMATH_CALUDE_toy_store_shelves_l1032_103281

/-- The number of shelves needed to display bears in a toy store. -/
def shelves_needed (initial_stock new_shipment bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Theorem stating that the toy store used 2 shelves to display the bears. -/
theorem toy_store_shelves :
  shelves_needed 5 7 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l1032_103281


namespace NUMINAMATH_CALUDE_rectangle_area_l1032_103223

/-- The area of a rectangle with length 20 cm and width 25 cm is 500 cm² -/
theorem rectangle_area : 
  ∀ (rectangle : Set ℝ) (length width area : ℝ),
  length = 20 →
  width = 25 →
  area = length * width →
  area = 500 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1032_103223


namespace NUMINAMATH_CALUDE_system_of_inequalities_l1032_103217

theorem system_of_inequalities (x : ℝ) :
  (2 + x < 6 - 3 * x) ∧ (x ≤ (4 + x) / 2) → x < 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l1032_103217


namespace NUMINAMATH_CALUDE_percentage_of_singles_is_70_percent_l1032_103221

def total_hits : ℕ := 50
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 10

def singles : ℕ := total_hits - (home_runs + triples + doubles)

theorem percentage_of_singles_is_70_percent :
  (singles : ℚ) / total_hits * 100 = 70 := by sorry

end NUMINAMATH_CALUDE_percentage_of_singles_is_70_percent_l1032_103221


namespace NUMINAMATH_CALUDE_student_heights_average_l1032_103287

theorem student_heights_average :
  ∀ (h1 h2 h3 h4 : ℝ),
    h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4 →
    max h1 (max h2 (max h3 h4)) = 152 →
    min h1 (min h2 (min h3 h4)) = 137 →
    ∃ (avg : ℝ), avg = 145 ∧ (h1 + h2 + h3 + h4) / 4 = avg :=
by sorry

end NUMINAMATH_CALUDE_student_heights_average_l1032_103287


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1032_103239

theorem quadratic_root_range (m : ℝ) (α β : ℝ) : 
  (∃ x, x^2 - 2*(m-1)*x + (m-1) = 0) ∧ 
  (α^2 - 2*(m-1)*α + (m-1) = 0) ∧ 
  (β^2 - 2*(m-1)*β + (m-1) = 0) ∧ 
  (0 < α) ∧ (α < 1) ∧ (1 < β) ∧ (β < 2) →
  (2 < m) ∧ (m < 7/3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1032_103239


namespace NUMINAMATH_CALUDE_certain_number_operations_l1032_103242

theorem certain_number_operations (x : ℝ) : 
  (((x + 5) * 2) / 5) - 5 = 62.5 / 2 → x = 85.625 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_operations_l1032_103242


namespace NUMINAMATH_CALUDE_ice_cream_bill_calculation_l1032_103250

/-- The final bill for four ice cream sundaes with a 20% tip -/
def final_bill (sundae1 sundae2 sundae3 sundae4 : ℝ) (tip_percentage : ℝ) : ℝ :=
  let total_cost := sundae1 + sundae2 + sundae3 + sundae4
  let tip := tip_percentage * total_cost
  total_cost + tip

/-- Theorem stating that the final bill for the given sundae prices and tip percentage is $42.00 -/
theorem ice_cream_bill_calculation :
  final_bill 7.50 10.00 8.50 9.00 0.20 = 42.00 := by
  sorry


end NUMINAMATH_CALUDE_ice_cream_bill_calculation_l1032_103250


namespace NUMINAMATH_CALUDE_inequality_proof_l1032_103205

theorem inequality_proof (x a : ℝ) (h : x > a ∧ a > 0) : x^2 > x*a ∧ x*a > a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1032_103205


namespace NUMINAMATH_CALUDE_expenditure_problem_l1032_103278

theorem expenditure_problem (initial_amount : ℝ) : 
  let remaining_after_clothes := (2/3) * initial_amount
  let remaining_after_food := (4/5) * remaining_after_clothes
  let remaining_after_travel := (3/4) * remaining_after_food
  let remaining_after_entertainment := (5/7) * remaining_after_travel
  let final_remaining := (5/6) * remaining_after_entertainment
  final_remaining = 200 → initial_amount = 840 := by
sorry

end NUMINAMATH_CALUDE_expenditure_problem_l1032_103278


namespace NUMINAMATH_CALUDE_andrei_apple_spending_l1032_103255

/-- Calculates Andrei's monthly spending on apples given the original price, price increase percentage, discount percentage, and amount bought per month. -/
def andreiMonthlySpending (originalPrice : ℚ) (priceIncrease : ℚ) (discount : ℚ) (kgPerMonth : ℚ) : ℚ :=
  let newPrice := originalPrice * (1 + priceIncrease / 100)
  let discountedPrice := newPrice * (1 - discount / 100)
  discountedPrice * kgPerMonth

/-- Theorem stating that Andrei's monthly spending on apples is 99 rubles under the given conditions. -/
theorem andrei_apple_spending :
  andreiMonthlySpending 50 10 10 2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_andrei_apple_spending_l1032_103255


namespace NUMINAMATH_CALUDE_no_natural_solution_l1032_103228

theorem no_natural_solution (x y z : ℕ) : (x : ℚ) / y + (y : ℚ) / z + (z : ℚ) / x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l1032_103228


namespace NUMINAMATH_CALUDE_similar_triangles_segment_length_l1032_103293

/-- Two triangles are similar if they have the same shape but not necessarily the same size. -/
def SimilarTriangles (P Q R X Y Z : ℝ × ℝ) : Prop := sorry

theorem similar_triangles_segment_length 
  (P Q R X Y Z : ℝ × ℝ) 
  (h_similar : SimilarTriangles P Q R X Y Z)
  (h_PQ : dist P Q = 8)
  (h_QR : dist Q R = 16)
  (h_YZ : dist Y Z = 24) :
  dist X Y = 12 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_segment_length_l1032_103293


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_squared_l1032_103265

theorem isosceles_right_triangle_hypotenuse_squared 
  (u v w : ℂ) (s t : ℝ) (k : ℝ) : 
  (∀ z : ℂ, z^3 + 2*z^2 + s*z + t = 0 ↔ z = u ∨ z = v ∨ z = w) → 
  Complex.abs u^2 + Complex.abs v^2 + Complex.abs w^2 = 350 →
  ∃ (x y : ℝ), 
    (Complex.abs (u - v))^2 = x^2 + y^2 ∧ 
    (Complex.abs (v - w))^2 = x^2 + y^2 ∧
    (Complex.abs (w - u))^2 = x^2 + y^2 ∧
    k^2 = (Complex.abs (w - u))^2 →
  k^2 = 525 := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_squared_l1032_103265


namespace NUMINAMATH_CALUDE_james_february_cost_l1032_103238

/-- Calculates the total cost for a streaming service based on the given parameters. -/
def streaming_cost (base_cost : ℝ) (free_hours : ℕ) (extra_hour_cost : ℝ) 
                   (movie_rental_cost : ℝ) (hours_streamed : ℕ) (movies_rented : ℕ) : ℝ :=
  let extra_hours := max (hours_streamed - free_hours) 0
  base_cost + (extra_hours : ℝ) * extra_hour_cost + (movies_rented : ℝ) * movie_rental_cost

/-- Theorem stating that James' streaming cost in February is $24. -/
theorem james_february_cost :
  streaming_cost 15 50 2 0.1 53 30 = 24 := by
  sorry

end NUMINAMATH_CALUDE_james_february_cost_l1032_103238


namespace NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l1032_103227

/-- 
Given a bucket that weighs p kilograms when three-fourths full of water
and q kilograms when one-third full of water, this theorem proves that
the weight of the bucket when full is (8p - 3q) / 5 kilograms.
-/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let three_fourths_weight := p
  let one_third_weight := q
  let full_weight := (8 * p - 3 * q) / 5
  full_weight

/-- The proof of the bucket_weight theorem. -/
theorem bucket_weight_proof (p q : ℝ) : 
  bucket_weight p q = (8 * p - 3 * q) / 5 := by
  sorry

end NUMINAMATH_CALUDE_bucket_weight_bucket_weight_proof_l1032_103227


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1032_103245

theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 4 * z^2 - 5) * (4 * z^4 - 3 * z^2 + 2) =
  12 * z^7 + 16 * z^6 - 9 * z^5 - 32 * z^4 + 6 * z^3 + 23 * z^2 - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1032_103245


namespace NUMINAMATH_CALUDE_field_trip_students_l1032_103288

theorem field_trip_students (van_capacity : ℕ) (num_vans : ℕ) (num_adults : ℕ) 
  (h1 : van_capacity = 9)
  (h2 : num_vans = 6)
  (h3 : num_adults = 14) :
  num_vans * van_capacity - num_adults = 40 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l1032_103288


namespace NUMINAMATH_CALUDE_comic_book_pages_l1032_103294

theorem comic_book_pages (total_frames : Nat) (frames_per_page : Nat) 
  (h1 : total_frames = 143)
  (h2 : frames_per_page = 11) :
  (total_frames / frames_per_page = 13) ∧ (total_frames % frames_per_page = 0) := by
  sorry

end NUMINAMATH_CALUDE_comic_book_pages_l1032_103294


namespace NUMINAMATH_CALUDE_dice_surface_area_l1032_103243

/-- The surface area of a cube with edge length 20 centimeters is 2400 square centimeters. -/
theorem dice_surface_area :
  let edge_length : ℝ := 20
  let surface_area : ℝ := 6 * edge_length ^ 2
  surface_area = 2400 := by sorry

end NUMINAMATH_CALUDE_dice_surface_area_l1032_103243


namespace NUMINAMATH_CALUDE_valid_numbers_l1032_103202

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  let x := n / 100
  let y := (n / 10) % 10
  let z := n % 10
  x + y + z = (10 * x + y) - (10 * y + z)

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {209, 428, 647, 866, 214, 433, 652, 871} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1032_103202


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1032_103251

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x + 2| → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1032_103251


namespace NUMINAMATH_CALUDE_mod_fifteen_equivalence_l1032_103272

theorem mod_fifteen_equivalence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 15879 [MOD 15] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_fifteen_equivalence_l1032_103272


namespace NUMINAMATH_CALUDE_parabola_point_distance_l1032_103234

/-- For a parabola y^2 = 2x, the x-coordinate of a point on the parabola
    that is at a distance of 3 from its focus is 5/2. -/
theorem parabola_point_distance (x y : ℝ) : 
  y^2 = 2*x →  -- parabola equation
  (x + 1/2)^2 + y^2 = 3^2 →  -- distance from focus is 3
  x = 5/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l1032_103234


namespace NUMINAMATH_CALUDE_max_ab_value_l1032_103225

theorem max_ab_value (a b : ℝ) (h : ∀ x : ℝ, Real.exp (x + 1) ≥ a * x + b) :
  (∀ c d : ℝ, (∀ x : ℝ, Real.exp (x + 1) ≥ c * x + d) → a * b ≥ c * d) ∧
  a * b = Real.exp 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l1032_103225


namespace NUMINAMATH_CALUDE_roots_sequence_property_l1032_103232

/-- Given x₁ and x₂ are roots of x² - 6x + 1 = 0, prove that for all natural numbers n,
    aₙ = x₁ⁿ + x₂ⁿ is an integer and not a multiple of 5. -/
theorem roots_sequence_property (x₁ x₂ : ℝ) (h : x₁^2 - 6*x₁ + 1 = 0 ∧ x₂^2 - 6*x₂ + 1 = 0) :
  ∀ n : ℕ, ∃ k : ℤ, (x₁^n + x₂^n = k) ∧ ¬(5 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_roots_sequence_property_l1032_103232


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1032_103247

theorem sufficient_condition_for_inequality (x : ℝ) : 
  1 < x ∧ x < 2 → (x + 1) / (x - 1) > 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1032_103247


namespace NUMINAMATH_CALUDE_parallel_lines_not_always_equal_l1032_103262

-- Define a line in a plane
structure Line :=
  (extends_infinitely : Bool)
  (can_be_measured : Bool)

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.extends_infinitely ∧ l2.extends_infinitely ∧ ¬l1.can_be_measured ∧ ¬l2.can_be_measured

-- Theorem: Two parallel lines are not always equal
theorem parallel_lines_not_always_equal :
  ∃ l1 l2 : Line, parallel l1 l2 ∧ l1 ≠ l2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_not_always_equal_l1032_103262


namespace NUMINAMATH_CALUDE_pyramid_volume_with_conditions_l1032_103263

/-- The volume of a right pyramid with a hexagonal base -/
noncomputable def pyramidVolume (totalSurfaceArea : ℝ) (triangularFaceRatio : ℝ) : ℝ :=
  let hexagonalBaseArea := totalSurfaceArea / 3
  let sideLength := Real.sqrt (320 / (3 * Real.sqrt 3))
  let triangularHeight := 160 / sideLength
  let pyramidHeight := Real.sqrt (triangularHeight^2 - (sideLength / 2)^2)
  (1 / 3) * hexagonalBaseArea * pyramidHeight

/-- Theorem: The volume of the pyramid with given conditions -/
theorem pyramid_volume_with_conditions :
  ∃ (V : ℝ), pyramidVolume 720 (1/3) = V :=
sorry

end NUMINAMATH_CALUDE_pyramid_volume_with_conditions_l1032_103263


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_solution_a_l1032_103289

/-- Proves that the percentage of alcohol in Solution A is 27% given the specified conditions. -/
theorem alcohol_percentage_in_solution_a : ∀ x : ℝ,
  -- Solution A has 6 liters of water and x% of alcohol
  -- Solution B has 9 liters of a solution containing 57% alcohol
  -- After mixing, the new mixture has 45% alcohol concentration
  (6 * x + 9 * 0.57 = 15 * 0.45) →
  x = 0.27 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_solution_a_l1032_103289


namespace NUMINAMATH_CALUDE_student_in_all_clubs_l1032_103299

theorem student_in_all_clubs (n : ℕ) (F G C : Finset (Fin n)) :
  n = 30 →
  F.card = 22 →
  G.card = 21 →
  C.card = 18 →
  ∃ s, s ∈ F ∩ G ∩ C :=
by
  sorry

end NUMINAMATH_CALUDE_student_in_all_clubs_l1032_103299


namespace NUMINAMATH_CALUDE_derivative_negative_two_exp_times_sin_l1032_103233

theorem derivative_negative_two_exp_times_sin (x : ℝ) :
  deriv (λ x => -2 * Real.exp x * Real.sin x) x = -2 * Real.exp x * (Real.sin x + Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_negative_two_exp_times_sin_l1032_103233


namespace NUMINAMATH_CALUDE_total_cantaloupes_l1032_103208

def keith_cantaloupes : ℝ := 29.5
def fred_cantaloupes : ℝ := 16.25
def jason_cantaloupes : ℝ := 20.75
def olivia_cantaloupes : ℝ := 12.5
def emily_cantaloupes : ℝ := 15.8

theorem total_cantaloupes : 
  keith_cantaloupes + fred_cantaloupes + jason_cantaloupes + olivia_cantaloupes + emily_cantaloupes = 94.8 := by
  sorry

end NUMINAMATH_CALUDE_total_cantaloupes_l1032_103208


namespace NUMINAMATH_CALUDE_document_word_count_l1032_103230

/-- Calculates the approximate total number of words in a document -/
def approx_total_words (num_pages : ℕ) (avg_words_per_page : ℕ) : ℕ :=
  num_pages * avg_words_per_page

/-- Theorem stating that a document with 8 pages and an average of 605 words per page has approximately 4800 words in total -/
theorem document_word_count : approx_total_words 8 605 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_document_word_count_l1032_103230


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_five_l1032_103284

theorem four_digit_divisible_by_five (n : ℕ) : 
  (5000 ≤ n ∧ n ≤ 5999) ∧ (n % 5 = 0) → 
  (Finset.filter (λ x : ℕ => (5000 ≤ x ∧ x ≤ 5999) ∧ (x % 5 = 0)) (Finset.range 10000)).card = 200 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_five_l1032_103284


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1032_103269

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 2*x^2 + 4*x - 1) % (x - 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1032_103269


namespace NUMINAMATH_CALUDE_apple_harvest_per_section_l1032_103274

theorem apple_harvest_per_section 
  (total_sections : ℕ) 
  (total_sacks : ℕ) 
  (h1 : total_sections = 8) 
  (h2 : total_sacks = 360) : 
  total_sacks / total_sections = 45 := by
  sorry

end NUMINAMATH_CALUDE_apple_harvest_per_section_l1032_103274


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1032_103229

theorem complex_equation_solution :
  ∃ (z : ℂ), 3 - 2 * Complex.I * z = 5 + 3 * Complex.I * z ∧ z = (2 * Complex.I) / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1032_103229


namespace NUMINAMATH_CALUDE_inequality_solution_l1032_103244

theorem inequality_solution (y : ℝ) : 
  (7/30 : ℝ) + |y - 3/10| < 11/30 ↔ 1/6 < y ∧ y < 1/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1032_103244


namespace NUMINAMATH_CALUDE_area_of_region_l1032_103201

-- Define the region
def region (x y : ℝ) : Prop :=
  Real.sqrt (Real.arcsin y) ≤ Real.sqrt (Real.arccos x) ∧ 
  -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1

-- State the theorem
theorem area_of_region : 
  MeasureTheory.volume {p : ℝ × ℝ | region p.1 p.2} = 1 + π / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l1032_103201


namespace NUMINAMATH_CALUDE_remainder_of_product_of_nines_l1032_103295

def product_of_nines (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * (10^(i+1) - 1)) 1

theorem remainder_of_product_of_nines :
  product_of_nines 999 % 1000 = 109 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_of_nines_l1032_103295


namespace NUMINAMATH_CALUDE_radical_expression_simplification_l1032_103253

theorem radical_expression_simplification
  (a b x : ℝ) 
  (h1 : a < b) 
  (h2 : -b ≤ x) 
  (h3 : x ≤ -a) :
  Real.sqrt (-(x + a)^3 * (x + b)) = -(x + a) * Real.sqrt (-(x + a) * (x + b)) :=
by sorry

end NUMINAMATH_CALUDE_radical_expression_simplification_l1032_103253


namespace NUMINAMATH_CALUDE_smallest_positive_angle_with_same_terminal_side_l1032_103204

theorem smallest_positive_angle_with_same_terminal_side (angle : ℝ) : 
  angle = 1000 →
  (∃ (k : ℤ), angle = 280 + 360 * k) →
  (∀ (x : ℝ), 0 ≤ x ∧ x < 360 ∧ (∃ (m : ℤ), angle = x + 360 * m) → x ≥ 280) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_with_same_terminal_side_l1032_103204


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l1032_103248

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  GeometricSequence a →
  a 4 - a 2 = 6 →
  a 5 - a 1 = 15 →
  a 3 = 4 ∨ a 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l1032_103248


namespace NUMINAMATH_CALUDE_square_root_of_nine_l1032_103268

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {-3, 3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l1032_103268


namespace NUMINAMATH_CALUDE_expression_value_l1032_103200

theorem expression_value : (85 + 32 / 113) * 113 = 9635 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1032_103200


namespace NUMINAMATH_CALUDE_sin_cos_sum_47_43_l1032_103249

theorem sin_cos_sum_47_43 : Real.sin (47 * π / 180) * Real.cos (43 * π / 180) + Real.cos (47 * π / 180) * Real.sin (43 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_47_43_l1032_103249


namespace NUMINAMATH_CALUDE_office_payroll_is_75000_l1032_103226

/-- Calculates the total monthly payroll for office workers given the following conditions:
  * There are 15 factory workers with a total monthly payroll of $30,000
  * There are 30 office workers
  * The average monthly salary of an office worker exceeds that of a factory worker by $500
-/
def office_workers_payroll (
  factory_workers : ℕ)
  (factory_payroll : ℕ)
  (office_workers : ℕ)
  (salary_difference : ℕ) : ℕ :=
  let factory_avg_salary := factory_payroll / factory_workers
  let office_avg_salary := factory_avg_salary + salary_difference
  office_workers * office_avg_salary

/-- Theorem stating that the total monthly payroll for office workers is $75,000 -/
theorem office_payroll_is_75000 :
  office_workers_payroll 15 30000 30 500 = 75000 := by
  sorry

end NUMINAMATH_CALUDE_office_payroll_is_75000_l1032_103226


namespace NUMINAMATH_CALUDE_sum_of_digits_successor_l1032_103273

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: If S(n) = 4387, then S(n+1) = 4388 -/
theorem sum_of_digits_successor (n : ℕ) (h : sum_of_digits n = 4387) : 
  sum_of_digits (n + 1) = 4388 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_successor_l1032_103273


namespace NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_root_l1032_103286

theorem arithmetic_sequence_quadratic_root (x y z : ℝ) : 
  (∃ d : ℝ, y = x + d ∧ z = x + 2*d) →  -- arithmetic sequence
  x ≤ y ∧ y ≤ z ∧ z ≤ 10 →             -- ordering condition
  (∃! r : ℝ, z*r^2 + y*r + x = 0) →    -- quadratic has exactly one root
  (∃ r : ℝ, z*r^2 + y*r + x = 0 ∧ r = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_root_l1032_103286


namespace NUMINAMATH_CALUDE_equation_solution_l1032_103211

theorem equation_solution (x : ℝ) : 
  x ≠ 1 → -x^2 = (2*x + 4)/(x - 1) → x = -2 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1032_103211


namespace NUMINAMATH_CALUDE_double_price_profit_percentage_l1032_103275

theorem double_price_profit_percentage (cost : ℝ) (initial_profit_percentage : ℝ) 
  (initial_selling_price : ℝ) (new_selling_price : ℝ) (new_profit_percentage : ℝ) :
  initial_profit_percentage = 20 →
  initial_selling_price = cost * (1 + initial_profit_percentage / 100) →
  new_selling_price = 2 * initial_selling_price →
  new_profit_percentage = ((new_selling_price - cost) / cost) * 100 →
  new_profit_percentage = 140 :=
by sorry

end NUMINAMATH_CALUDE_double_price_profit_percentage_l1032_103275


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1032_103206

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 9x + 4 is 1 -/
theorem quadratic_discriminant : discriminant 5 (-9) 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1032_103206


namespace NUMINAMATH_CALUDE_equal_cost_at_60_messages_l1032_103207

/-- Represents a text messaging plan with a per-message cost and a monthly fee. -/
structure TextPlan where
  perMessageCost : ℚ
  monthlyFee : ℚ

/-- Calculates the total cost for a given number of messages under a specific plan. -/
def totalCost (plan : TextPlan) (messages : ℚ) : ℚ :=
  plan.perMessageCost * messages + plan.monthlyFee

/-- The number of messages at which all plans have the same cost. -/
def equalCostMessages (planA planB planC : TextPlan) : ℚ :=
  60

theorem equal_cost_at_60_messages (planA planB planC : TextPlan) 
    (hA : planA = ⟨0.25, 9⟩) 
    (hB : planB = ⟨0.40, 0⟩)
    (hC : planC = ⟨0.20, 12⟩) : 
    let messages := equalCostMessages planA planB planC
    totalCost planA messages = totalCost planB messages ∧ 
    totalCost planA messages = totalCost planC messages :=
  sorry

end NUMINAMATH_CALUDE_equal_cost_at_60_messages_l1032_103207


namespace NUMINAMATH_CALUDE_quadratic_sum_l1032_103236

/-- Given a quadratic expression x^2 - 20x + 49, prove that when written in the form (x+b)^2 + c,
    the sum of b and c is equal to -61. -/
theorem quadratic_sum (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 49 = (x + b)^2 + c) → b + c = -61 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1032_103236


namespace NUMINAMATH_CALUDE_questionnaires_from_unit_D_l1032_103257

/-- Represents the number of questionnaires drawn from each unit -/
structure SampleDistribution where
  unitA : ℕ
  unitB : ℕ
  unitC : ℕ
  unitD : ℕ

/-- The sample distribution forms an arithmetic sequence -/
def is_arithmetic_sequence (s : SampleDistribution) : Prop :=
  s.unitB - s.unitA = s.unitC - s.unitB ∧ s.unitC - s.unitB = s.unitD - s.unitC

/-- The total sample size is 150 -/
def total_sample_size (s : SampleDistribution) : ℕ :=
  s.unitA + s.unitB + s.unitC + s.unitD

theorem questionnaires_from_unit_D 
  (s : SampleDistribution)
  (h1 : is_arithmetic_sequence s)
  (h2 : total_sample_size s = 150)
  (h3 : s.unitB = 30) :
  s.unitD = 60 := by
  sorry

end NUMINAMATH_CALUDE_questionnaires_from_unit_D_l1032_103257


namespace NUMINAMATH_CALUDE_p_current_age_l1032_103218

theorem p_current_age (p q : ℕ) : 
  (p - 3) / (q - 3) = 4 / 3 →
  (p + 6) / (q + 6) = 7 / 6 →
  p = 15 := by
sorry

end NUMINAMATH_CALUDE_p_current_age_l1032_103218


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1032_103231

theorem min_value_quadratic (x : ℝ) :
  ∃ (z_min : ℝ), ∀ (z : ℝ), z = 3 * x^2 + 18 * x + 11 → z ≥ z_min ∧ ∃ (x_min : ℝ), 3 * x_min^2 + 18 * x_min + 11 = z_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1032_103231


namespace NUMINAMATH_CALUDE_plant_pricing_theorem_l1032_103212

/-- Represents the selling price per plant as a function of the number of plants per pot -/
def selling_price_per_plant (x : ℝ) : ℝ := -0.3 * x + 4.5

/-- Represents the price per pot as a function of the number of plants per pot -/
def price_per_pot (x : ℝ) : ℝ := -0.3 * x^2 + 4.5 * x

/-- Represents the cultivation cost per pot as a function of the number of plants -/
def cultivation_cost (x : ℝ) : ℝ := 2 + 0.3 * x

theorem plant_pricing_theorem :
  ∀ x : ℝ,
  5 ≤ x → x ≤ 12 →
  (selling_price_per_plant x = -0.3 * x + 4.5) ∧
  (price_per_pot x = -0.3 * x^2 + 4.5 * x) ∧
  ((price_per_pot x = 16.2) → (x = 6 ∨ x = 9)) ∧
  (∃ x : ℝ, (x = 12 ∨ x = 15) ∧
    30 * (price_per_pot x) - 40 * (cultivation_cost x) = 100) :=
by sorry


end NUMINAMATH_CALUDE_plant_pricing_theorem_l1032_103212


namespace NUMINAMATH_CALUDE_fixed_fee_calculation_l1032_103222

theorem fixed_fee_calculation (feb_bill march_bill : ℝ) 
  (h : feb_bill = 18.72 ∧ march_bill = 33.78) :
  ∃ (fixed_fee hourly_rate : ℝ),
    fixed_fee + hourly_rate = feb_bill ∧
    fixed_fee + 3 * hourly_rate = march_bill ∧
    fixed_fee = 11.19 := by
sorry

end NUMINAMATH_CALUDE_fixed_fee_calculation_l1032_103222


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1032_103235

theorem express_y_in_terms_of_x (p : ℝ) (x y : ℝ) : 
  x = 3 + 2^p → y = 3 + 2^(-p) → y = (3*x - 8) / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1032_103235


namespace NUMINAMATH_CALUDE_equation_solution_l1032_103241

theorem equation_solution (x y : ℤ) (hy : y ≠ 0) :
  (2 : ℝ) ^ ((x - y : ℝ) / y) - (3 / 2 : ℝ) * y = 1 ↔
  ∃ n : ℕ, x = ((2 * n + 1) * (2 ^ (2 * n + 1) - 2)) / 3 ∧
           y = (2 ^ (2 * n + 1) - 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1032_103241


namespace NUMINAMATH_CALUDE_log_equation_solution_l1032_103216

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 + Real.log x / Real.log 27 = 7 →
  x = 3 ^ (42 / 11) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1032_103216


namespace NUMINAMATH_CALUDE_vector_addition_l1032_103254

theorem vector_addition : 
  let v1 : Fin 2 → ℝ := ![3, -7]
  let v2 : Fin 2 → ℝ := ![-6, 11]
  v1 + v2 = ![(-3), 4] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l1032_103254


namespace NUMINAMATH_CALUDE_no_square_divisible_by_six_between_50_and_120_l1032_103279

theorem no_square_divisible_by_six_between_50_and_120 : ¬ ∃ x : ℕ,
  (∃ y : ℕ, x = y^2) ∧ 
  (∃ z : ℕ, x = 6 * z) ∧ 
  50 < x ∧ x < 120 := by
sorry

end NUMINAMATH_CALUDE_no_square_divisible_by_six_between_50_and_120_l1032_103279


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_equals_three_l1032_103256

theorem sqrt_equality_implies_t_equals_three (t : ℝ) : 
  Real.sqrt (2 * Real.sqrt (t - 2)) = (7 - t) ^ (1/4) → t = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_equals_three_l1032_103256


namespace NUMINAMATH_CALUDE_f_neg_one_eq_zero_iff_r_eq_neg_eight_l1032_103276

/-- A polynomial function f(x) with a parameter r -/
def f (r : ℝ) (x : ℝ) : ℝ := 3 * x^4 + x^3 + 2 * x^2 - 4 * x + r

/-- Theorem stating that f(-1) = 0 if and only if r = -8 -/
theorem f_neg_one_eq_zero_iff_r_eq_neg_eight :
  ∀ r : ℝ, f r (-1) = 0 ↔ r = -8 := by sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_zero_iff_r_eq_neg_eight_l1032_103276


namespace NUMINAMATH_CALUDE_periodic_decimal_as_fraction_l1032_103270

-- Define the periodic decimal expansion
def periodic_decimal : ℝ :=
  0.5123412341234123412341234123412341234

-- Theorem statement
theorem periodic_decimal_as_fraction :
  periodic_decimal = 51229 / 99990 := by
  sorry

end NUMINAMATH_CALUDE_periodic_decimal_as_fraction_l1032_103270


namespace NUMINAMATH_CALUDE_perfect_square_sum_l1032_103271

theorem perfect_square_sum (n : ℤ) (h1 : n > 1) (h2 : ∃ x : ℤ, 3*n + 1 = x^2) :
  ∃ a b c : ℤ, n + 1 = a^2 + b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l1032_103271


namespace NUMINAMATH_CALUDE_daves_age_ratio_l1032_103296

theorem daves_age_ratio (D N : ℚ) : 
  (D > 0) → 
  (N > 0) → 
  (∃ (a b c d : ℚ), a + b + c + d = D) → -- Combined ages of four children equal D
  (D - N = 3 * (D - 4 * N)) → -- N years ago, Dave's age was thrice the sum of children's ages
  D / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_daves_age_ratio_l1032_103296


namespace NUMINAMATH_CALUDE_base_with_final_digit_one_l1032_103215

theorem base_with_final_digit_one : 
  ∃! b : ℕ, 2 ≤ b ∧ b ≤ 15 ∧ 648 % b = 1 :=
by sorry

end NUMINAMATH_CALUDE_base_with_final_digit_one_l1032_103215


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1032_103282

/-- A quadratic function with vertex (1,16) and roots 8 units apart -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 15

/-- The function g(x) defined in terms of f(x) and a parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := (2 - 2*a)*x - f x

theorem quadratic_function_properties :
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ |x₁ - x₂| = 8) ∧
  (∀ x : ℝ, f x ≤ f 1) ∧
  f 1 = 16 ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, Monotone (g a)) ↔ a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1032_103282


namespace NUMINAMATH_CALUDE_largest_five_digit_distinct_odd_number_l1032_103203

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d < 10

def is_five_digit_number (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digits_are_distinct (n : ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 → i ≠ j →
    (n / 10^i) % 10 ≠ (n / 10^j) % 10

def all_digits_odd (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 5 → is_odd_digit ((n / 10^i) % 10)

theorem largest_five_digit_distinct_odd_number :
  ∀ n : ℕ, is_five_digit_number n → digits_are_distinct n → all_digits_odd n →
    n ≤ 97531 := by sorry

end NUMINAMATH_CALUDE_largest_five_digit_distinct_odd_number_l1032_103203


namespace NUMINAMATH_CALUDE_particle_movement_probability_l1032_103283

/-- The probability of a particle moving from (0, 0) to (2, 3) in 5 steps,
    where each step has an equal probability of 1/2 of moving right or up. -/
theorem particle_movement_probability :
  let n : ℕ := 5  -- Total number of steps
  let k : ℕ := 2  -- Number of steps to the right
  let p : ℚ := 1/2  -- Probability of moving right (or up)
  Nat.choose n k * p^n = (1/2)^5 := by
  sorry

end NUMINAMATH_CALUDE_particle_movement_probability_l1032_103283


namespace NUMINAMATH_CALUDE_polynomial_value_at_zero_l1032_103259

def is_valid_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ), 
    ∀ x, p x = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6

theorem polynomial_value_at_zero 
  (p : ℝ → ℝ) 
  (h_valid : is_valid_polynomial p) 
  (h_values : ∀ n : ℕ, n ≤ 6 → p (3^n) = (1 : ℝ) / (3^n)) :
  p 0 = 29523 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_zero_l1032_103259


namespace NUMINAMATH_CALUDE_solution_set_part1_min_value_part2_l1032_103298

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 2 x ≥ 1 - 2*x} = {x : ℝ | x ≥ -1} := by sorry

-- Part 2
theorem min_value_part2 (a m n : ℝ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) 
  (h4 : m^2 * n = a) (h5 : ∀ x, f a x + |x - 1| ≥ 3) :
  ∃ (x : ℝ), m + n ≥ x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_min_value_part2_l1032_103298


namespace NUMINAMATH_CALUDE_power_of_two_digit_sum_five_l1032_103240

def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem power_of_two_digit_sum_five (n : ℕ) : 
  sum_of_digits (2^n) = 5 ↔ n = 5 := by sorry

end NUMINAMATH_CALUDE_power_of_two_digit_sum_five_l1032_103240


namespace NUMINAMATH_CALUDE_loan_period_calculation_l1032_103260

/-- The time period (in years) for which A lent money to C -/
def loan_period_C : ℚ := 2/3

theorem loan_period_calculation (principal_B principal_C total_interest : ℚ) 
  (loan_period_B interest_rate : ℚ) :
  principal_B = 5000 →
  principal_C = 3000 →
  loan_period_B = 2 →
  interest_rate = 1/10 →
  total_interest = 2200 →
  principal_B * interest_rate * loan_period_B + 
  principal_C * interest_rate * loan_period_C = total_interest :=
by sorry

end NUMINAMATH_CALUDE_loan_period_calculation_l1032_103260


namespace NUMINAMATH_CALUDE_stock_price_uniqueness_l1032_103219

theorem stock_price_uniqueness (n : Nat) (k l : Nat) (h_n : 0 < n ∧ n < 100) :
  (1 + n / 100 : ℚ) ^ k * (1 - n / 100 : ℚ) ^ l ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_uniqueness_l1032_103219


namespace NUMINAMATH_CALUDE_prob_at_least_one_contract_l1032_103280

/-- The probability of getting at least one contract given specific probabilities for hardware and software contracts -/
theorem prob_at_least_one_contract 
  (p_hardware : ℝ) 
  (p_not_software : ℝ) 
  (p_both : ℝ) 
  (h1 : p_hardware = 4/5)
  (h2 : p_not_software = 3/5)
  (h3 : p_both = 0.3) :
  p_hardware + (1 - p_not_software) - p_both = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_contract_l1032_103280


namespace NUMINAMATH_CALUDE_geometric_series_product_l1032_103220

theorem geometric_series_product (x : ℝ) : 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/x)^n → x = 9 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_product_l1032_103220


namespace NUMINAMATH_CALUDE_max_tuesday_money_l1032_103252

/-- The amount of money Max's mom gave him on Tuesday -/
def tuesday_amount : ℝ := 8

/-- The amount of money Max's mom gave him on Wednesday -/
def wednesday_amount (t : ℝ) : ℝ := 5 * t

/-- The amount of money Max's mom gave him on Thursday -/
def thursday_amount (t : ℝ) : ℝ := wednesday_amount t + 9

theorem max_tuesday_money :
  ∃ t : ℝ, t = tuesday_amount ∧
    thursday_amount t = t + 41 :=
by sorry

end NUMINAMATH_CALUDE_max_tuesday_money_l1032_103252


namespace NUMINAMATH_CALUDE_triangle_side_length_l1032_103267

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1032_103267


namespace NUMINAMATH_CALUDE_BaCl2_mass_produced_l1032_103292

-- Define the molar masses
def molar_mass_BaCl2 : ℝ := 208.23

-- Define the initial amounts
def initial_BaCl2_moles : ℝ := 8
def initial_NaOH_moles : ℝ := 12

-- Define the stoichiometric ratios
def ratio_NaOH_to_BaCl2 : ℝ := 2
def ratio_BaOH2_to_BaCl2 : ℝ := 1

-- Define the theorem
theorem BaCl2_mass_produced : 
  let BaCl2_produced := min initial_BaCl2_moles (initial_NaOH_moles / ratio_NaOH_to_BaCl2)
  BaCl2_produced * molar_mass_BaCl2 = 1665.84 :=
by sorry

end NUMINAMATH_CALUDE_BaCl2_mass_produced_l1032_103292


namespace NUMINAMATH_CALUDE_a_less_than_two_l1032_103291

def A (a : ℝ) : Set ℝ := {x | x ≤ a}
def B : Set ℝ := Set.Iio 2

theorem a_less_than_two (a : ℝ) (h : A a ⊆ B) : a < 2 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_two_l1032_103291
