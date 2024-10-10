import Mathlib

namespace range_of_m_l2852_285206

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ m > 1 := by
sorry

end range_of_m_l2852_285206


namespace prime_factor_puzzle_l2852_285280

theorem prime_factor_puzzle (a b c d w x y z : ℕ) : 
  (Nat.Prime w) → 
  (Nat.Prime x) → 
  (Nat.Prime y) → 
  (Nat.Prime z) → 
  (w < x) → 
  (x < y) → 
  (y < z) → 
  ((w^a) * (x^b) * (y^c) * (z^d) = 660) → 
  ((a + b) - (c + d) = 1) → 
  d = 1 := by
sorry

end prime_factor_puzzle_l2852_285280


namespace emily_holidays_l2852_285268

/-- The number of days Emily takes off each month -/
def days_off_per_month : ℕ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Emily takes in a year -/
def total_holidays : ℕ := days_off_per_month * months_in_year

theorem emily_holidays : total_holidays = 24 := by
  sorry

end emily_holidays_l2852_285268


namespace fish_tagging_problem_l2852_285217

/-- The number of tagged fish in the second catch, given the conditions of the fish tagging problem. -/
def tagged_fish_in_second_catch (total_fish : ℕ) (initially_tagged : ℕ) (second_catch : ℕ) : ℕ :=
  (initially_tagged * second_catch) / total_fish

/-- Theorem stating that the number of tagged fish in the second catch is 2 under the given conditions. -/
theorem fish_tagging_problem (total_fish : ℕ) (initially_tagged : ℕ) (second_catch : ℕ)
  (h_total : total_fish = 1750)
  (h_tagged : initially_tagged = 70)
  (h_catch : second_catch = 50) :
  tagged_fish_in_second_catch total_fish initially_tagged second_catch = 2 :=
by sorry

end fish_tagging_problem_l2852_285217


namespace derivative_f_at_one_l2852_285273

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_f_at_one : 
  deriv f 1 = 2 := by sorry

end derivative_f_at_one_l2852_285273


namespace projection_onto_common_vector_l2852_285225

/-- Given two vectors v1 and v2 in ℝ², prove that their projection onto a common vector is v1 if v1 is orthogonal to (v2 - v1). -/
theorem projection_onto_common_vector (v1 v2 : ℝ × ℝ) :
  v1 = (3, 2) ∧ v2 = (1, 5) →
  let diff := (v2.1 - v1.1, v2.2 - v1.2)
  (v1.1 * diff.1 + v1.2 * diff.2 = 0) →
  ∃ (x y : ℝ), (∀ t : ℝ, (v1.1 + t * diff.1, v1.2 + t * diff.2) = (x, y)) :=
by sorry

#check projection_onto_common_vector

end projection_onto_common_vector_l2852_285225


namespace fraction_sum_equals_decimal_l2852_285296

theorem fraction_sum_equals_decimal : 
  2/5 + 3/25 + 4/125 + 1/625 = 0.5536 := by
sorry

end fraction_sum_equals_decimal_l2852_285296


namespace find_divisor_l2852_285203

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) (divisor : Nat) :
  dividend = quotient * divisor + remainder →
  dividend = 172 →
  quotient = 10 →
  remainder = 2 →
  divisor = 17 := by
sorry

end find_divisor_l2852_285203


namespace algebraic_expression_value_l2852_285245

theorem algebraic_expression_value (x : ℝ) :
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end algebraic_expression_value_l2852_285245


namespace quadratic_is_square_of_binomial_l2852_285226

/-- Given that ax^2 + 21x + 9 is the square of a binomial, prove that a = 49/4 -/
theorem quadratic_is_square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 21 * x + 9 = (r * x + s)^2) → 
  a = 49/4 := by
  sorry

end quadratic_is_square_of_binomial_l2852_285226


namespace machine_speed_ratio_l2852_285247

def machine_a_rate (parts_a : ℕ) (time_a : ℕ) : ℚ := parts_a / time_a
def machine_b_rate (parts_b : ℕ) (time_b : ℕ) : ℚ := parts_b / time_b

theorem machine_speed_ratio :
  let parts_a_100 : ℕ := 100
  let time_a_100 : ℕ := 40
  let parts_a_50 : ℕ := 50
  let time_a_50 : ℕ := 10
  let parts_b : ℕ := 100
  let time_b : ℕ := 40
  machine_a_rate parts_a_100 time_a_100 = machine_b_rate parts_b time_b →
  machine_a_rate parts_a_50 time_a_50 / machine_b_rate parts_b time_b = 2 := by
sorry

end machine_speed_ratio_l2852_285247


namespace star_37_25_l2852_285219

-- Define the star operation
def star (x y : ℝ) : ℝ := x * y + 3

-- State the theorem
theorem star_37_25 :
  (∀ (x : ℝ), x > 0 → star (star x 1) x = star x (star 1 x)) →
  star 1 1 = 4 →
  star 37 25 = 928 := by sorry

end star_37_25_l2852_285219


namespace first_year_after_2020_with_sum_of_digits_15_l2852_285282

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2020WithSumOfDigits15 (year : ℕ) : Prop :=
  year > 2020 ∧ 
  sumOfDigits year = 15 ∧
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 15

theorem first_year_after_2020_with_sum_of_digits_15 :
  isFirstYearAfter2020WithSumOfDigits15 2049 := by
  sorry

end first_year_after_2020_with_sum_of_digits_15_l2852_285282


namespace shirt_price_calculation_l2852_285290

/-- Calculates the final price of a shirt given its original cost, profit margin, and discount percentage. -/
def final_price (original_cost : ℝ) (profit_margin : ℝ) (discount : ℝ) : ℝ :=
  let selling_price := original_cost * (1 + profit_margin)
  selling_price * (1 - discount)

/-- Theorem stating that a shirt with an original cost of $20, a 30% profit margin, and a 50% discount has a final price of $13. -/
theorem shirt_price_calculation :
  final_price 20 0.3 0.5 = 13 := by
  sorry

#eval final_price 20 0.3 0.5

end shirt_price_calculation_l2852_285290


namespace roberto_outfits_l2852_285207

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem stating the number of outfits Roberto can create -/
theorem roberto_outfits :
  let trousers : ℕ := 6
  let shirts : ℕ := 7
  let jackets : ℕ := 4
  let shoes : ℕ := 2
  number_of_outfits trousers shirts jackets shoes = 336 := by
  sorry


end roberto_outfits_l2852_285207


namespace necessary_but_not_sufficient_condition_l2852_285240

def p (x : ℝ) : Prop := x^2 - 3*x + 2 < 0

theorem necessary_but_not_sufficient_condition :
  (∃ (a b : ℝ), (a = -1 ∧ b = 2 ∨ a = -2 ∧ b = 2) ∧
    (∀ x, p x → a < x ∧ x < b) ∧
    (∃ y, a < y ∧ y < b ∧ ¬(p y))) :=
sorry

end necessary_but_not_sufficient_condition_l2852_285240


namespace system_equivalence_l2852_285205

theorem system_equivalence (x y a b : ℝ) : 
  (2 * x + y = 5 ∧ a * x + 3 * y = -1) ∧
  (x - y = 1 ∧ 4 * x + b * y = 11) →
  a = -2 ∧ b = 3 := by
sorry

end system_equivalence_l2852_285205


namespace inequality_proof_l2852_285244

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^2 + 1 - 2*x*(x*y^2 - x + z + 1) ≥ 0 :=
by sorry

end inequality_proof_l2852_285244


namespace fraction_to_zero_power_l2852_285230

theorem fraction_to_zero_power :
  let a : ℤ := -573293
  let b : ℕ := 7903827
  (a : ℚ) / b ^ (0 : ℕ) = 1 :=
by sorry

end fraction_to_zero_power_l2852_285230


namespace library_books_remaining_l2852_285233

theorem library_books_remaining (initial_books : ℕ) 
  (day1_borrowers : ℕ) (books_per_borrower : ℕ) (day2_borrowed : ℕ) : 
  initial_books = 100 →
  day1_borrowers = 5 →
  books_per_borrower = 2 →
  day2_borrowed = 20 →
  initial_books - (day1_borrowers * books_per_borrower + day2_borrowed) = 70 :=
by sorry

end library_books_remaining_l2852_285233


namespace train_length_l2852_285283

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 25 → speed * time * (1000 / 3600) = 833.25 := by
  sorry

end train_length_l2852_285283


namespace solution_set_when_a_is_2_range_of_a_l2852_285265

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 1|

-- Theorem 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2
theorem range_of_a :
  ∀ a : ℝ, (∃ x₀ : ℝ, f a x₀ ≤ |2*a - 1|) → 0 ≤ a ∧ a ≤ 2 := by sorry

end solution_set_when_a_is_2_range_of_a_l2852_285265


namespace proposition_relationship_l2852_285237

theorem proposition_relationship (x y : ℝ) :
  (∀ x y, x + y ≠ 8 → (x ≠ 2 ∨ y ≠ 6)) ∧
  (∃ x y, (x ≠ 2 ∨ y ≠ 6) ∧ x + y = 8) :=
by sorry

end proposition_relationship_l2852_285237


namespace negation_of_cubic_greater_than_square_l2852_285257

theorem negation_of_cubic_greater_than_square :
  (¬ ∀ x : ℕ, x^3 > x^2) ↔ (∃ x : ℕ, x^3 ≤ x^2) := by sorry

end negation_of_cubic_greater_than_square_l2852_285257


namespace pencil_cost_l2852_285293

theorem pencil_cost (x y : ℚ) 
  (eq1 : 4 * x + 3 * y = 224)
  (eq2 : 2 * x + 5 * y = 154) : 
  y = 12 := by
sorry

end pencil_cost_l2852_285293


namespace probability_8_of_hearts_or_spade_l2852_285214

def standard_deck : ℕ := 52

def probability_8_of_hearts : ℚ := 1 / standard_deck

def probability_spade : ℚ := 1 / 4

theorem probability_8_of_hearts_or_spade :
  probability_8_of_hearts + probability_spade = 7 / 26 := by
  sorry

end probability_8_of_hearts_or_spade_l2852_285214


namespace specific_isosceles_triangle_area_l2852_285234

/-- Represents an isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  altitude : ℝ
  perimeter : ℝ
  leg_difference : ℝ

/-- Calculates the area of the isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles triangle -/
theorem specific_isosceles_triangle_area :
  let t : IsoscelesTriangle := {
    altitude := 10,
    perimeter := 40,
    leg_difference := 2
  }
  area t = 81.2 := by sorry

end specific_isosceles_triangle_area_l2852_285234


namespace max_value_of_exponential_difference_l2852_285238

theorem max_value_of_exponential_difference (x : ℝ) :
  ∃ (max : ℝ), max = 1/4 ∧ ∀ (y : ℝ), 5^y - 25^y ≤ max := by
  sorry

end max_value_of_exponential_difference_l2852_285238


namespace john_ultramarathon_distance_l2852_285255

/-- Calculates the total distance John can run after training -/
def johnRunningDistance (initialTime : ℝ) (timeIncrease : ℝ) (initialSpeed : ℝ) (speedIncrease : ℝ) : ℝ :=
  (initialTime * (1 + timeIncrease)) * (initialSpeed + speedIncrease)

theorem john_ultramarathon_distance :
  johnRunningDistance 8 0.75 8 4 = 168 := by
  sorry

end john_ultramarathon_distance_l2852_285255


namespace arithmetic_mean_difference_l2852_285231

theorem arithmetic_mean_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = (a + b + c) / 3 + 5)
  (h2 : (a + c) / 2 = (a + b + c) / 3 - 8) :
  (b + c) / 2 = (a + b + c) / 3 + 3 := by
  sorry

end arithmetic_mean_difference_l2852_285231


namespace hard_lens_price_l2852_285276

/-- Represents the price of contact lenses and sales information -/
structure LensSales where
  soft_price : ℕ
  hard_price : ℕ
  soft_count : ℕ
  hard_count : ℕ
  total_sales : ℕ

/-- Theorem stating the price of hard contact lenses -/
theorem hard_lens_price (sales : LensSales) : 
  sales.soft_price = 150 ∧ 
  sales.soft_count = sales.hard_count + 5 ∧
  sales.soft_count + sales.hard_count = 11 ∧
  sales.total_sales = sales.soft_price * sales.soft_count + sales.hard_price * sales.hard_count ∧
  sales.total_sales = 1455 →
  sales.hard_price = 85 := by
sorry

end hard_lens_price_l2852_285276


namespace theater_attendance_l2852_285266

/-- Calculates the total number of attendees given ticket prices and revenue --/
def total_attendees (adult_price child_price : ℕ) (total_revenue : ℕ) (num_children : ℕ) : ℕ :=
  let num_adults := (total_revenue - child_price * num_children) / adult_price
  num_adults + num_children

/-- Theorem stating that under the given conditions, the total number of attendees is 280 --/
theorem theater_attendance : total_attendees 60 25 14000 80 = 280 := by
  sorry

end theater_attendance_l2852_285266


namespace melanie_total_dimes_l2852_285252

def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4

theorem melanie_total_dimes : 
  initial_dimes + dimes_from_dad + dimes_from_mom = 19 := by
  sorry

end melanie_total_dimes_l2852_285252


namespace stating_triangle_division_theorem_l2852_285254

/-- 
Represents the number of parts a triangle is divided into when each vertex
is connected to n points on the opposite side, assuming no three lines intersect
at the same point.
-/
def triangle_division (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- 
Theorem stating that when each vertex of a triangle is connected by straight lines
to n points on the opposite side, and no three lines intersect at the same point,
the triangle is divided into 3n^2 + 3n + 1 parts.
-/
theorem triangle_division_theorem (n : ℕ) :
  triangle_division n = 3 * n^2 + 3 * n + 1 := by
  sorry

end stating_triangle_division_theorem_l2852_285254


namespace total_wait_days_l2852_285258

/-- The number of days Mark waits for his first vaccine appointment -/
def first_appointment_wait : ℕ := 4

/-- The number of days Mark waits for his second vaccine appointment -/
def second_appointment_wait : ℕ := 20

/-- The number of weeks Mark waits for the vaccine to be fully effective -/
def full_effectiveness_wait_weeks : ℕ := 2

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: The total number of days Mark waits is 38 -/
theorem total_wait_days : 
  first_appointment_wait + second_appointment_wait + (full_effectiveness_wait_weeks * days_per_week) = 38 := by
  sorry

end total_wait_days_l2852_285258


namespace distance_to_origin_l2852_285228

theorem distance_to_origin : ∃ (M : ℝ × ℝ), 
  M = (-5, 12) ∧ Real.sqrt ((-5)^2 + 12^2) = 13 := by
  sorry

end distance_to_origin_l2852_285228


namespace add_fractions_l2852_285236

theorem add_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end add_fractions_l2852_285236


namespace four_points_on_circle_l2852_285232

/-- A point in the 2D Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points lie on the same circle -/
def on_same_circle (A B C D : Point) : Prop :=
  ∃ (center : Point) (r : ℝ),
    (center.x - A.x)^2 + (center.y - A.y)^2 = r^2 ∧
    (center.x - B.x)^2 + (center.y - B.y)^2 = r^2 ∧
    (center.x - C.x)^2 + (center.y - C.y)^2 = r^2 ∧
    (center.x - D.x)^2 + (center.y - D.y)^2 = r^2

theorem four_points_on_circle :
  let A : Point := ⟨-1, 5⟩
  let B : Point := ⟨5, 5⟩
  let C : Point := ⟨-3, 1⟩
  let D : Point := ⟨6, -2⟩
  on_same_circle A B C D :=
by
  sorry

end four_points_on_circle_l2852_285232


namespace boat_speed_in_still_water_l2852_285291

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water :
  let current_speed : ℝ := 4
  let downstream_distance : ℝ := 5.133333333333334
  let downstream_time : ℝ := 14 / 60
  ∃ v : ℝ, v > 0 ∧ (v + current_speed) * downstream_time = downstream_distance ∧ v = 18 := by
  sorry

end boat_speed_in_still_water_l2852_285291


namespace sqrt_inequality_l2852_285270

theorem sqrt_inequality (x : ℝ) :
  3 - x ≥ 0 → x + 1 ≥ 0 →
  (Real.sqrt (3 - x) - Real.sqrt (x + 1) > 1 / 2 ↔ -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8) :=
by sorry

end sqrt_inequality_l2852_285270


namespace max_product_xyz_l2852_285209

theorem max_product_xyz (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0)
  (hsum : x + y + z = 1) (heq : x = y) (hbound : x ≤ z ∧ z ≤ 2*x) :
  ∃ (max_val : ℝ), ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 → 
    a + b + c = 1 → a = b → 
    a ≤ c → c ≤ 2*a → 
    a * b * c ≤ max_val ∧ 
    max_val = 1 / 27 :=
by sorry

end max_product_xyz_l2852_285209


namespace absolute_value_equation_solution_l2852_285256

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| + 3 * x = 14 :=
by
  -- The unique solution is x = 4.5
  use 4.5
  sorry

end absolute_value_equation_solution_l2852_285256


namespace trailing_zeros_of_product_factorials_mod_100_l2852_285288

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def product_of_factorials (n : ℕ) : ℕ := (List.range n).foldl (fun acc i => acc * factorial (i + 1)) 1

def trailing_zeros (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n.digits 10).reverse.takeWhile (· = 0) |>.length

theorem trailing_zeros_of_product_factorials_mod_100 :
  trailing_zeros (product_of_factorials 50) % 100 = 12 := by sorry

end trailing_zeros_of_product_factorials_mod_100_l2852_285288


namespace repeating_fraction_sixteen_equals_five_thirty_ninths_l2852_285289

/-- Given a positive integer k, this function returns the value of the infinite geometric series
    4/k + 5/k^2 + 4/k^3 + 5/k^4 + ... -/
def repeating_fraction (k : ℕ) : ℚ :=
  (4 * k + 5) / (k^2 - 1)

/-- The theorem states that for k = 16, the repeating fraction equals 5/39 -/
theorem repeating_fraction_sixteen_equals_five_thirty_ninths :
  repeating_fraction 16 = 5 / 39 := by
  sorry

end repeating_fraction_sixteen_equals_five_thirty_ninths_l2852_285289


namespace fraction_product_theorem_l2852_285286

theorem fraction_product_theorem : 
  (7 : ℚ) / 4 * 8 / 14 * 28 / 16 * 24 / 36 * 49 / 35 * 40 / 25 * 63 / 42 * 32 / 48 = 56 / 25 := by
  sorry

end fraction_product_theorem_l2852_285286


namespace inscribed_cylinder_height_l2852_285264

theorem inscribed_cylinder_height (r c h : ℝ) : 
  r > 0 → c > 0 → h > 0 →
  r = 8 → c = 3 →
  h^2 = r^2 - c^2 →
  h = Real.sqrt 55 := by
sorry

end inscribed_cylinder_height_l2852_285264


namespace quadratic_equal_roots_l2852_285239

theorem quadratic_equal_roots (m n : ℝ) : 
  (m = 2 ∧ n = 1) → 
  ∃ x : ℝ, x^2 - m*x + n = 0 ∧ 
  ∀ y : ℝ, y^2 - m*y + n = 0 → y = x :=
sorry

end quadratic_equal_roots_l2852_285239


namespace equal_intersection_areas_l2852_285263

/-- A tetrahedron with specific properties -/
structure Tetrahedron where
  opposite_edges_perpendicular : Bool
  vertical_segment : ℝ
  midplane_area : ℝ

/-- A sphere with a specific radius -/
structure Sphere where
  radius : ℝ

/-- The configuration of a tetrahedron and a sphere -/
structure Configuration where
  tetra : Tetrahedron
  sphere : Sphere
  radius_condition : sphere.radius^2 * π = tetra.midplane_area
  vertical_segment_condition : tetra.vertical_segment = 2 * sphere.radius

/-- The area of intersection of a shape with a plane -/
def intersection_area (height : ℝ) : Configuration → ℝ
  | _ => sorry

/-- The main theorem stating that the areas of intersection are equal for all heights -/
theorem equal_intersection_areas (config : Configuration) :
  ∀ h : ℝ, 0 ≤ h ∧ h ≤ config.tetra.vertical_segment →
    intersection_area h config = intersection_area (config.tetra.vertical_segment - h) config :=
  sorry

end equal_intersection_areas_l2852_285263


namespace harry_buckets_per_round_l2852_285272

theorem harry_buckets_per_round 
  (george_buckets : ℕ) 
  (total_buckets : ℕ) 
  (total_rounds : ℕ) 
  (h1 : george_buckets = 2)
  (h2 : total_buckets = 110)
  (h3 : total_rounds = 22) :
  (total_buckets - george_buckets * total_rounds) / total_rounds = 3 := by
  sorry

end harry_buckets_per_round_l2852_285272


namespace quadratic_properties_l2852_285277

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (x + a) * (x - a - 1)

-- State the theorem
theorem quadratic_properties (a : ℝ) (h_a : a > 0) :
  -- 1. Axis of symmetry
  (∃ (x : ℝ), x = 1/2 ∧ ∀ (y : ℝ), f a (x - y) = f a (x + y)) ∧
  -- 2. Vertex coordinates when maximum is 4
  (∃ (x_max : ℝ), x_max ∈ Set.Icc (-1) 3 ∧ 
    (∀ (x : ℝ), x ∈ Set.Icc (-1) 3 → f a x ≤ 4) ∧ 
    f a x_max = 4 →
    f a (1/2) = -9/4) ∧
  -- 3. Range of t
  (∀ (t x₁ x₂ y₁ y₂ : ℝ),
    y₁ ≠ y₂ ∧
    t < x₁ ∧ x₁ < t + 1 ∧
    t + 2 < x₂ ∧ x₂ < t + 3 ∧
    f a x₁ = y₁ ∧ f a x₂ = y₂ →
    t ≥ -1/2) :=
by sorry

end quadratic_properties_l2852_285277


namespace text_messages_difference_l2852_285243

theorem text_messages_difference (last_week : ℕ) (total : ℕ) : last_week = 111 → total = 283 → total - last_week - last_week = 61 := by
  sorry

end text_messages_difference_l2852_285243


namespace negation_of_existence_inequality_l2852_285287

theorem negation_of_existence_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end negation_of_existence_inequality_l2852_285287


namespace probability_at_least_six_heads_l2852_285278

/-- A sequence of 8 coin flips -/
def CoinFlipSequence := Fin 8 → Bool

/-- The total number of possible outcomes for 8 coin flips -/
def totalOutcomes : ℕ := 256

/-- Checks if a sequence has at least 6 consecutive heads -/
def hasAtLeastSixConsecutiveHeads (s : CoinFlipSequence) : Prop :=
  ∃ i, i + 5 < 8 ∧ (∀ j, i ≤ j ∧ j ≤ i + 5 → s j = true)

/-- The number of sequences with at least 6 consecutive heads -/
def favorableOutcomes : ℕ := 13

/-- The probability of getting at least 6 consecutive heads in 8 fair coin flips -/
def probabilityAtLeastSixHeads : ℚ := favorableOutcomes / totalOutcomes

theorem probability_at_least_six_heads :
  probabilityAtLeastSixHeads = 13 / 256 :=
sorry

end probability_at_least_six_heads_l2852_285278


namespace cubic_solution_sum_l2852_285246

theorem cubic_solution_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a = 12 →
  b^3 - 6*b^2 + 11*b = 12 →
  c^3 - 6*c^2 + 11*c = 12 →
  a * b / c + b * c / a + c * a / b = -23 / 12 :=
by sorry

end cubic_solution_sum_l2852_285246


namespace pants_count_l2852_285241

/-- Represents the number of each type of clothing item in a dresser -/
structure DresserContents where
  pants : ℕ
  shorts : ℕ
  shirts : ℕ

/-- The ratio of pants to shorts to shirts in the dresser -/
def clothingRatio : ℕ × ℕ × ℕ := (7, 7, 10)

/-- The number of shirts in the dresser -/
def shirtCount : ℕ := 20

/-- Checks if the given DresserContents satisfies the ratio condition -/
def satisfiesRatio (contents : DresserContents) : Prop :=
  contents.pants * clothingRatio.2.2 = contents.shirts * clothingRatio.1 ∧
  contents.shorts * clothingRatio.2.2 = contents.shirts * clothingRatio.2.1

theorem pants_count (contents : DresserContents) 
  (h_ratio : satisfiesRatio contents) 
  (h_shirts : contents.shirts = shirtCount) : 
  contents.pants = 14 := by
  sorry

end pants_count_l2852_285241


namespace yuna_candies_l2852_285284

theorem yuna_candies (initial_candies remaining_candies : ℕ) 
  (h1 : initial_candies = 23)
  (h2 : remaining_candies = 7) :
  initial_candies - remaining_candies = 16 := by
  sorry

end yuna_candies_l2852_285284


namespace painting_area_calculation_l2852_285262

theorem painting_area_calculation (price_per_sqft : ℝ) (total_cost : ℝ) (area : ℝ) :
  price_per_sqft = 15 →
  total_cost = 840 →
  area * price_per_sqft = total_cost →
  area = 56 := by
sorry

end painting_area_calculation_l2852_285262


namespace min_balls_for_five_same_color_l2852_285269

/-- Given a bag with 10 red balls, 10 yellow balls, and 10 white balls,
    the minimum number of balls that must be drawn to ensure
    at least 5 balls of the same color is 13. -/
theorem min_balls_for_five_same_color (red yellow white : ℕ) 
  (h_red : red = 10) (h_yellow : yellow = 10) (h_white : white = 10) :
  ∃ (n : ℕ), n = 13 ∧ 
  ∀ (m : ℕ), m < n → 
  ∃ (r y w : ℕ), r + y + w = m ∧ r < 5 ∧ y < 5 ∧ w < 5 :=
sorry

end min_balls_for_five_same_color_l2852_285269


namespace exam_count_proof_l2852_285267

theorem exam_count_proof (prev_avg : ℝ) (desired_avg : ℝ) (next_score : ℝ) :
  prev_avg = 84 →
  desired_avg = 86 →
  next_score = 100 →
  ∃ n : ℕ, n > 0 ∧ (n * desired_avg - (n - 1) * prev_avg = next_score) ∧ n = 8 := by
  sorry

end exam_count_proof_l2852_285267


namespace total_buttons_for_order_l2852_285260

/-- The number of shirts ordered for each type -/
def shirts_per_type : ℕ := 200

/-- The number of buttons on the first type of shirt -/
def buttons_type1 : ℕ := 3

/-- The number of buttons on the second type of shirt -/
def buttons_type2 : ℕ := 5

/-- Theorem: The total number of buttons needed for the order is 1600 -/
theorem total_buttons_for_order :
  shirts_per_type * buttons_type1 + shirts_per_type * buttons_type2 = 1600 := by
  sorry

end total_buttons_for_order_l2852_285260


namespace ratio_odd_even_divisors_l2852_285220

def M : ℕ := 42 * 43 * 75 * 196

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 14 := by sorry

end ratio_odd_even_divisors_l2852_285220


namespace student_average_greater_than_true_average_l2852_285253

theorem student_average_greater_than_true_average 
  (a b c : ℝ) (h : a < b ∧ b < c) : (a + b + c) / 2 > (a + b + c) / 3 := by
  sorry

end student_average_greater_than_true_average_l2852_285253


namespace base5_addition_puzzle_l2852_285215

/-- Converts a base 10 number to base 5 --/
def toBase5 (n : ℕ) : ℕ := sorry

/-- Represents a digit in base 5 --/
structure Digit5 where
  value : ℕ
  property : value < 5

theorem base5_addition_puzzle :
  ∀ (S H E : Digit5),
    S.value ≠ 0 ∧ H.value ≠ 0 ∧ E.value ≠ 0 →
    S.value ≠ H.value ∧ S.value ≠ E.value ∧ H.value ≠ E.value →
    (S.value * 25 + H.value * 5 + E.value) + (H.value * 5 + E.value) = 
    (S.value * 25 + E.value * 5 + S.value) →
    S.value = 4 ∧ H.value = 1 ∧ E.value = 2 ∧ 
    toBase5 (S.value + H.value + E.value) = 12 :=
by sorry

end base5_addition_puzzle_l2852_285215


namespace shopping_mall_purchase_l2852_285285

/-- Represents the shopping mall's purchase of products A and B -/
structure ProductPurchase where
  cost_price_A : ℝ
  selling_price_A : ℝ
  selling_price_B : ℝ
  profit_margin_B : ℝ
  total_units : ℕ
  total_cost : ℝ

/-- Theorem stating the correct number of units purchased for each product -/
theorem shopping_mall_purchase (p : ProductPurchase)
  (h1 : p.cost_price_A = 40)
  (h2 : p.selling_price_A = 60)
  (h3 : p.selling_price_B = 80)
  (h4 : p.profit_margin_B = 0.6)
  (h5 : p.total_units = 50)
  (h6 : p.total_cost = 2200) :
  ∃ (units_A units_B : ℕ),
    units_A + units_B = p.total_units ∧
    units_A * p.cost_price_A + units_B * (p.selling_price_B / (1 + p.profit_margin_B)) = p.total_cost ∧
    units_A = 30 ∧
    units_B = 20 := by
  sorry


end shopping_mall_purchase_l2852_285285


namespace farmer_additional_earnings_l2852_285227

/-- Represents the farmer's market transactions and wheelbarrow sale --/
def farmer_earnings (duck_price chicken_price : ℕ) (ducks_sold chickens_sold : ℕ) : ℕ :=
  let total_earnings := duck_price * ducks_sold + chicken_price * chickens_sold
  let wheelbarrow_cost := total_earnings / 2
  let wheelbarrow_sale_price := wheelbarrow_cost * 2
  wheelbarrow_sale_price - wheelbarrow_cost

/-- Proves that the farmer's additional earnings from selling the wheelbarrow is $30 --/
theorem farmer_additional_earnings :
  farmer_earnings 10 8 2 5 = 30 := by
  sorry

end farmer_additional_earnings_l2852_285227


namespace gcd_problem_l2852_285275

theorem gcd_problem : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 35 n = 7 ∧ n = 77 := by
  sorry

end gcd_problem_l2852_285275


namespace cubic_root_sum_l2852_285222

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a*b/c + b*c/a + c*a/b = 49/6 := by
sorry

end cubic_root_sum_l2852_285222


namespace greatest_common_divisor_of_differences_gcd_of_54_87_172_l2852_285235

theorem greatest_common_divisor_of_differences : Int → Int → Int → Prop :=
  fun a b c => 
    let diff1 := b - a
    let diff2 := c - b
    let diff3 := c - a
    Nat.gcd (Nat.gcd (Int.natAbs diff1) (Int.natAbs diff2)) (Int.natAbs diff3) = 1

theorem gcd_of_54_87_172 : greatest_common_divisor_of_differences 54 87 172 := by
  sorry

end greatest_common_divisor_of_differences_gcd_of_54_87_172_l2852_285235


namespace inequality_proof_l2852_285229

theorem inequality_proof (a b c x y z : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0)
  (h4 : x ≥ y) (h5 : y ≥ z) (h6 : z > 0) : 
  (a^2 * x^2) / ((b*y + c*z) * (b*z + c*y)) + 
  (b^2 * y^2) / ((a*x + c*z) * (a*z + c*x)) + 
  (c^2 * z^2) / ((a*x + b*y) * (a*y + b*x)) ≥ 3/4 := by
sorry

end inequality_proof_l2852_285229


namespace divisibility_of_fifth_powers_l2852_285210

theorem divisibility_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
sorry

end divisibility_of_fifth_powers_l2852_285210


namespace tonys_initial_money_is_87_l2852_285213

/-- Calculates Tony's initial amount of money -/
def tonys_initial_money (cheese_cost beef_cost beef_amount cheese_amount money_left : ℕ) : ℕ :=
  cheese_cost * cheese_amount + beef_cost * beef_amount + money_left

/-- Proves that Tony's initial amount of money was $87 -/
theorem tonys_initial_money_is_87 :
  tonys_initial_money 7 5 1 3 61 = 87 := by
  sorry

end tonys_initial_money_is_87_l2852_285213


namespace journey_equation_l2852_285218

/-- Given a journey with three parts, prove the relationship between total distance, 
    total time, speeds, and time spent on each part. -/
theorem journey_equation 
  (D T x y z t₁ t₂ t₃ : ℝ) 
  (h_total_time : T = t₁ + t₂ + t₃) 
  (h_total_distance : D = x * t₁ + y * t₂ + z * t₃) 
  (h_positive_speed : x > 0 ∧ y > 0 ∧ z > 0)
  (h_positive_time : t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0)
  (h_positive_total : D > 0 ∧ T > 0) :
  D = x * t₁ + y * (T - t₁ - t₃) + z * t₃ :=
sorry

end journey_equation_l2852_285218


namespace two_color_cubes_count_l2852_285292

/-- Represents a cube with painted stripes -/
structure StripedCube where
  edge_length : ℕ
  stripe_count : ℕ

/-- Counts the number of smaller cubes with exactly two faces painted with different colors -/
def count_two_color_cubes (cube : StripedCube) : ℕ :=
  sorry

/-- Theorem stating the correct number of two-color cubes for a 6x6x6 cube with three stripes -/
theorem two_color_cubes_count (cube : StripedCube) :
  cube.edge_length = 6 ∧ cube.stripe_count = 3 →
  count_two_color_cubes cube = 12 :=
by sorry

end two_color_cubes_count_l2852_285292


namespace triangle_area_l2852_285224

theorem triangle_area (base height : ℝ) (h1 : base = 6) (h2 : height = 8) :
  (1 / 2) * base * height = 24 := by
  sorry

end triangle_area_l2852_285224


namespace plane_speed_calculation_l2852_285248

theorem plane_speed_calculation (D : ℝ) (V : ℝ) (h1 : D = V * 5) (h2 : D = 720 * (5/3)) :
  V = 240 := by
sorry

end plane_speed_calculation_l2852_285248


namespace rationalize_denominator_l2852_285274

theorem rationalize_denominator :
  ∃ (a b : ℝ), a + b * Real.sqrt 3 = -Real.sqrt 3 - 2 ∧
  (a + b * Real.sqrt 3) * (Real.sqrt 3 - 2) = 1 := by
sorry

end rationalize_denominator_l2852_285274


namespace one_fourth_of_12_8_l2852_285211

theorem one_fourth_of_12_8 :
  let x : ℚ := 12.8 / 4
  x = 16 / 5 ∧ x = 3 + 1 / 5 :=
by sorry

end one_fourth_of_12_8_l2852_285211


namespace isosceles_triangle_side_length_l2852_285259

/-- An isosceles triangle with specific measurements -/
structure IsoscelesTriangle where
  -- Base of the triangle
  base : ℝ
  -- Median from one of the equal sides
  median : ℝ
  -- Length of the equal sides
  side : ℝ
  -- The base is 4√2 cm
  base_eq : base = 4 * Real.sqrt 2
  -- The median is 5 cm
  median_eq : median = 5
  -- The triangle is isosceles (implied by the structure)

/-- 
Theorem: In an isosceles triangle with a base of 4√2 cm and a median of 5 cm 
from one of the equal sides, the length of the equal sides is 6 cm.
-/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) : t.side = 6 := by
  sorry

end isosceles_triangle_side_length_l2852_285259


namespace jays_change_is_twenty_l2852_285298

/-- The change Jay received after purchasing items and paying with a fifty-dollar bill -/
def jays_change (book_price pen_price ruler_price paid_amount : ℕ) : ℕ :=
  paid_amount - (book_price + pen_price + ruler_price)

/-- Theorem stating that Jay's change is $20 given the specific prices and payment amount -/
theorem jays_change_is_twenty :
  jays_change 25 4 1 50 = 20 := by
  sorry

end jays_change_is_twenty_l2852_285298


namespace line_slope_range_l2852_285297

/-- The range of m for a line x - my + √3m = 0 with a point M satisfying certain conditions -/
theorem line_slope_range (m : ℝ) : 
  (∃ (x y : ℝ), x - m * y + Real.sqrt 3 * m = 0 ∧ 
    y^2 = 3 * x^2 - 3) →
  (m ≤ -Real.sqrt 6 / 6 ∨ m ≥ Real.sqrt 6 / 6) :=
by sorry

end line_slope_range_l2852_285297


namespace smallest_overlap_coffee_tea_l2852_285202

/-- The smallest possible percentage of adults who drink both coffee and tea,
    given that 50% drink coffee and 60% drink tea. -/
theorem smallest_overlap_coffee_tea : ℝ :=
  let coffee_drinkers : ℝ := 50
  let tea_drinkers : ℝ := 60
  let total_percentage : ℝ := 100
  min (coffee_drinkers + tea_drinkers - total_percentage) coffee_drinkers

end smallest_overlap_coffee_tea_l2852_285202


namespace harry_pencils_left_l2852_285299

/-- Calculates the number of pencils left with Harry given the initial conditions. -/
def pencils_left_with_harry (anna_pencils : ℕ) (harry_lost : ℕ) : ℕ :=
  2 * anna_pencils - harry_lost

/-- Proves that Harry has 81 pencils left given the initial conditions. -/
theorem harry_pencils_left :
  pencils_left_with_harry 50 19 = 81 := by
  sorry

#eval pencils_left_with_harry 50 19

end harry_pencils_left_l2852_285299


namespace min_t_value_l2852_285200

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2 * Real.sqrt 2) ^ 2 + (y - 1) ^ 2 = 1

-- Define points A and B
def point_A (t : ℝ) : ℝ × ℝ := (-t, 0)
def point_B (t : ℝ) : ℝ × ℝ := (t, 0)

-- Define the condition for point P
def point_P_condition (P : ℝ × ℝ) (t : ℝ) : Prop :=
  circle_C P.1 P.2 ∧
  let AP := (P.1 + t, P.2)
  let BP := (P.1 - t, P.2)
  AP.1 * BP.1 + AP.2 * BP.2 = 0

-- State the theorem
theorem min_t_value :
  ∀ t : ℝ, t > 0 →
  (∃ P : ℝ × ℝ, point_P_condition P t) →
  (∀ t' : ℝ, t' > 0 ∧ (∃ P : ℝ × ℝ, point_P_condition P t') → t' ≥ 2) :=
sorry

end min_t_value_l2852_285200


namespace trajectory_is_line_segment_l2852_285223

/-- The trajectory of a point P satisfying |PF₁| + |PF₂| = 8, where F₁ and F₂ are fixed points -/
theorem trajectory_is_line_segment (F₁ F₂ P : ℝ × ℝ) : 
  F₁ = (-4, 0) → 
  F₂ = (4, 0) → 
  dist P F₁ + dist P F₂ = 8 →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂ :=
sorry

end trajectory_is_line_segment_l2852_285223


namespace equation_solution_l2852_285204

theorem equation_solution : ∃! x : ℝ, (x + 1) / 2 = x - 2 := by sorry

end equation_solution_l2852_285204


namespace tim_tetrises_l2852_285281

/-- The number of tetrises Tim scored -/
def num_tetrises (single_points tetris_points num_singles total_points : ℕ) : ℕ :=
  (total_points - num_singles * single_points) / tetris_points

/-- Theorem: Tim scored 4 tetrises -/
theorem tim_tetrises :
  let single_points : ℕ := 1000
  let tetris_points : ℕ := 8 * single_points
  let num_singles : ℕ := 6
  let total_points : ℕ := 38000
  num_tetrises single_points tetris_points num_singles total_points = 4 := by
  sorry

end tim_tetrises_l2852_285281


namespace two_digit_divisible_number_exists_l2852_285250

theorem two_digit_divisible_number_exists : ∃ n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ 
  n % 8 = 0 ∧ n % 12 = 0 ∧ n % 18 = 0 ∧
  60 ≤ n ∧ n ≤ 79 := by
  sorry

end two_digit_divisible_number_exists_l2852_285250


namespace monkey_peaches_l2852_285212

theorem monkey_peaches (x : ℕ) : 
  (x / 2 - 12 + (x / 2 + 12) / 2 + 12 = x - 19) → x = 100 := by
  sorry

end monkey_peaches_l2852_285212


namespace total_marbles_l2852_285251

/-- The total number of marbles given the conditions of red, blue, and green marbles -/
theorem total_marbles (r : ℝ) (b : ℝ) (g : ℝ) : 
  r > 0 → 
  r = 1.5 * b → 
  g = 1.8 * r → 
  r + b + g = 3.467 * r := by
sorry


end total_marbles_l2852_285251


namespace fourth_root_of_unity_l2852_285216

theorem fourth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 7 ∧ 
  (Complex.tan (π / 4) + Complex.I) / (Complex.tan (π / 4) - Complex.I) = 
  Complex.exp (Complex.I * (2 * n * π / 8)) := by
  sorry

end fourth_root_of_unity_l2852_285216


namespace compound_line_chart_optimal_l2852_285279

/-- Represents different types of statistical charts -/
inductive StatisticalChart
  | Bar
  | Pie
  | Line
  | Scatter
  | CompoundLine

/-- Represents the requirements for the chart -/
structure ChartRequirements where
  numStudents : Nat
  showComparison : Bool
  showChangesOverTime : Bool

/-- Determines if a chart type is optimal for given requirements -/
def isOptimalChart (chart : StatisticalChart) (req : ChartRequirements) : Prop :=
  chart = StatisticalChart.CompoundLine ∧
  req.numStudents = 2 ∧
  req.showComparison = true ∧
  req.showChangesOverTime = true

/-- Theorem stating that a compound line chart is optimal for the given scenario -/
theorem compound_line_chart_optimal (req : ChartRequirements) :
  req.numStudents = 2 →
  req.showComparison = true →
  req.showChangesOverTime = true →
  isOptimalChart StatisticalChart.CompoundLine req :=
by sorry

end compound_line_chart_optimal_l2852_285279


namespace f_extrema_l2852_285249

def f (p q x : ℝ) : ℝ := x^3 - p*x^2 - q*x

theorem f_extrema (p q : ℝ) :
  (f p q 1 = 0) →
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, f p q x ≤ f p q x₁) ∧ (∀ x : ℝ, f p q x ≥ f p q x₂) ∧ 
                 (f p q x₁ = 4/27) ∧ (f p q x₂ = 0)) :=
by sorry

end f_extrema_l2852_285249


namespace gcd_problem_l2852_285261

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ a = 17 * k) :
  Nat.gcd (Int.natAbs (2 * a ^ 2 + 33 * a + 85)) (Int.natAbs (a + 17)) = 34 := by
  sorry

end gcd_problem_l2852_285261


namespace acute_angle_alpha_l2852_285208

theorem acute_angle_alpha (α : Real) (h : 0 < α ∧ α < Real.pi / 2) 
  (eq : Real.cos (Real.pi / 6) * Real.sin α = Real.sqrt 3 / 4) : 
  α = Real.pi / 6 := by
  sorry

end acute_angle_alpha_l2852_285208


namespace complex_modulus_evaluation_l2852_285295

theorem complex_modulus_evaluation :
  Complex.abs (3 - 5*I + (-2 + (3/4)*I)) = (Real.sqrt 305) / 4 := by
  sorry

end complex_modulus_evaluation_l2852_285295


namespace problem_stack_surface_area_l2852_285242

/-- Represents a solid formed by stacking unit cubes -/
structure CubeStack where
  base_length : ℕ
  base_width : ℕ
  base_height : ℕ
  top_cube : Bool

/-- Calculates the surface area of a CubeStack -/
def surface_area (stack : CubeStack) : ℕ :=
  sorry

/-- The specific cube stack described in the problem -/
def problem_stack : CubeStack :=
  { base_length := 3
  , base_width := 3
  , base_height := 1
  , top_cube := true }

/-- Theorem stating that the surface area of the problem_stack is 34 square units -/
theorem problem_stack_surface_area :
  surface_area problem_stack = 34 :=
sorry

end problem_stack_surface_area_l2852_285242


namespace least_sum_of_exponents_for_260_l2852_285294

/-- Given a natural number n, returns the sum of exponents in its binary representation -/
def sumOfExponents (n : ℕ) : ℕ := sorry

/-- Checks if a natural number n can be expressed as a sum of at least three distinct powers of 2 -/
def hasAtLeastThreeDistinctPowers (n : ℕ) : Prop := sorry

theorem least_sum_of_exponents_for_260 :
  ∀ k : ℕ, (hasAtLeastThreeDistinctPowers 260 ∧ sumOfExponents 260 = k) → k ≥ 10 :=
sorry

end least_sum_of_exponents_for_260_l2852_285294


namespace sum_E_equals_1600_l2852_285221

-- Define E(n) as the sum of even digits in n
def E (n : ℕ) : ℕ := sorry

-- Define the sum of E(n) from 1 to 200
def sum_E : ℕ := (Finset.range 200).sum (fun i => E (i + 1))

-- Theorem to prove
theorem sum_E_equals_1600 : sum_E = 1600 := by sorry

end sum_E_equals_1600_l2852_285221


namespace equiangular_hexagon_side_lengths_l2852_285271

/-- An equiangular hexagon is a hexagon where all internal angles are equal. -/
structure EquiangularHexagon where
  sides : Fin 6 → ℝ
  is_equiangular : True  -- This is a placeholder for the equiangular property

/-- The theorem stating the side lengths of a specific equiangular hexagon -/
theorem equiangular_hexagon_side_lengths 
  (h : EquiangularHexagon) 
  (h1 : h.sides 0 = 3)
  (h2 : h.sides 2 = 5)
  (h3 : h.sides 3 = 4)
  (h4 : h.sides 4 = 1) :
  h.sides 5 = 6 ∧ h.sides 1 = 2 := by
  sorry


end equiangular_hexagon_side_lengths_l2852_285271


namespace simplify_expression_l2852_285201

theorem simplify_expression (a : ℝ) : (1 : ℝ) * (2 * a) * (3 * a^2) * (4 * a^3) * (5 * a^4) = 120 * a^10 := by
  sorry

end simplify_expression_l2852_285201
