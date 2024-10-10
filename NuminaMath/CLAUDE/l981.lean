import Mathlib

namespace max_roses_for_680_l981_98158

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : ℚ  -- Price of a single rose
  oneDozen : ℚ    -- Price of one dozen roses
  twoDozen : ℚ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased with a given amount -/
def maxRoses (pricing : RosePricing) (amount : ℚ) : ℕ :=
  sorry

/-- Theorem: Given the specific pricing, the maximum number of roses for $680 is 325 -/
theorem max_roses_for_680 (pricing : RosePricing) 
  (h1 : pricing.individual = 230 / 100)
  (h2 : pricing.oneDozen = 36)
  (h3 : pricing.twoDozen = 50) :
  maxRoses pricing 680 = 325 :=
sorry

end max_roses_for_680_l981_98158


namespace two_equidistant_points_l981_98139

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle with center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Configuration of a circle and two parallel lines -/
structure CircleLineConfiguration where
  circle : Circle
  line1 : Line
  line2 : Line
  d : ℝ
  h : d > circle.radius

/-- A point is equidistant from a circle and two parallel lines -/
def isEquidistant (p : Point) (config : CircleLineConfiguration) : Prop :=
  sorry

/-- The number of equidistant points -/
def numEquidistantPoints (config : CircleLineConfiguration) : ℕ :=
  sorry

/-- Theorem: There are exactly 2 equidistant points -/
theorem two_equidistant_points (config : CircleLineConfiguration) :
  numEquidistantPoints config = 2 :=
sorry

end two_equidistant_points_l981_98139


namespace difference_of_sum_and_difference_of_squares_l981_98189

theorem difference_of_sum_and_difference_of_squares (x y : ℝ) 
  (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := by
  sorry

end difference_of_sum_and_difference_of_squares_l981_98189


namespace exact_rolls_probability_l981_98117

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The number of dice rolled -/
def dice : ℕ := 8

/-- The number of dice we want to show a specific number -/
def target : ℕ := 4

/-- The probability of rolling exactly 'target' number of twos 
    when rolling 'dice' number of 'sides'-sided dice -/
def probability : ℚ := 168070 / 16777216

theorem exact_rolls_probability : 
  (Nat.choose dice target * (1 / sides) ^ target * ((sides - 1) / sides) ^ (dice - target)) = probability := by
  sorry

end exact_rolls_probability_l981_98117


namespace ratio_x_to_y_l981_98124

theorem ratio_x_to_y (x y : ℝ) 
  (h1 : (3*x - 2*y) / (2*x + 3*y) = 5/4)
  (h2 : x + y = 5) : 
  x / y = 23/2 := by
sorry

end ratio_x_to_y_l981_98124


namespace square_side_length_l981_98168

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 8)
  (h2 : rectangle_length = 2)
  (h3 : rectangle_width > 0)
  (h4 : rectangle_length > 0) :
  ∃ (square_side : ℝ), 
    square_side > 0 ∧ 
    square_side * square_side = rectangle_width * rectangle_length ∧
    square_side = 4 := by
  sorry

end square_side_length_l981_98168


namespace teachers_count_l981_98178

/-- Represents the total number of faculty and students in the school -/
def total_population : ℕ := 2400

/-- Represents the total number of individuals in the sample -/
def sample_size : ℕ := 160

/-- Represents the number of students in the sample -/
def students_in_sample : ℕ := 150

/-- Calculates the number of teachers in the school -/
def number_of_teachers : ℕ :=
  total_population - (total_population * students_in_sample) / sample_size

theorem teachers_count : number_of_teachers = 150 := by
  sorry

end teachers_count_l981_98178


namespace cube_paper_expenditure_l981_98171

-- Define the parameters
def paper_cost_per_kg : ℚ := 60
def cube_edge_length : ℚ := 10
def area_covered_per_kg : ℚ := 20

-- Define the function to calculate the expenditure
def calculate_expenditure (edge_length area_per_kg cost_per_kg : ℚ) : ℚ :=
  6 * edge_length^2 / area_per_kg * cost_per_kg

-- State the theorem
theorem cube_paper_expenditure :
  calculate_expenditure cube_edge_length area_covered_per_kg paper_cost_per_kg = 1800 := by
  sorry

end cube_paper_expenditure_l981_98171


namespace inequality_conditions_l981_98105

theorem inequality_conditions (a b : ℝ) :
  ((b > 0 ∧ 0 > a) → (1 / a < 1 / b)) ∧
  ((0 > a ∧ a > b) → (1 / a < 1 / b)) ∧
  ((a > 0 ∧ 0 > b) → ¬(1 / a < 1 / b)) ∧
  ((a > b ∧ b > 0) → (1 / a < 1 / b)) := by
sorry

end inequality_conditions_l981_98105


namespace remainder_theorem_l981_98187

theorem remainder_theorem : (7 * 11^24 + 2^24) % 12 = 11 := by
  sorry

end remainder_theorem_l981_98187


namespace circle_on_parabola_circle_standard_equation_l981_98142

def parabola (x y : ℝ) : Prop := y^2 = 16 * x

def circle_equation (h k r x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem circle_on_parabola (h k : ℝ) :
  parabola h k →
  first_quadrant h k →
  circle_equation h k 6 0 0 →
  circle_equation h k 6 4 0 →
  h = 2 ∧ k = 4 * Real.sqrt 2 :=
sorry

theorem circle_standard_equation (h k : ℝ) :
  h = 2 →
  k = 4 * Real.sqrt 2 →
  ∀ x y : ℝ, circle_equation h k 6 x y ↔ circle_equation 2 (4 * Real.sqrt 2) 6 x y :=
sorry

end circle_on_parabola_circle_standard_equation_l981_98142


namespace book_pricing_problem_l981_98154

/-- Proves that the cost price is approximately 64% of the marked price
    given the conditions of the book pricing problem. -/
theorem book_pricing_problem (MP CP : ℝ) : 
  MP > 0 → -- Marked price is positive
  CP > 0 → -- Cost price is positive
  MP * 0.88 = 1.375 * CP → -- Condition after applying discount and gain
  ∃ ε > 0, |CP / MP - 0.64| < ε := by
sorry


end book_pricing_problem_l981_98154


namespace concert_seats_count_l981_98184

/-- Represents the concert ticket sales scenario -/
structure ConcertSales where
  main_price : ℕ  -- Price of main seat tickets
  back_price : ℕ  -- Price of back seat tickets
  total_revenue : ℕ  -- Total revenue from ticket sales
  back_seats_sold : ℕ  -- Number of back seat tickets sold

/-- Calculates the total number of seats in the arena -/
def total_seats (cs : ConcertSales) : ℕ :=
  let main_seats := (cs.total_revenue - cs.back_price * cs.back_seats_sold) / cs.main_price
  main_seats + cs.back_seats_sold

/-- Theorem stating that the total number of seats is 20,000 -/
theorem concert_seats_count (cs : ConcertSales) 
  (h1 : cs.main_price = 55)
  (h2 : cs.back_price = 45)
  (h3 : cs.total_revenue = 955000)
  (h4 : cs.back_seats_sold = 14500) : 
  total_seats cs = 20000 := by
  sorry

#eval total_seats ⟨55, 45, 955000, 14500⟩

end concert_seats_count_l981_98184


namespace greatest_common_factor_of_168_252_315_l981_98162

theorem greatest_common_factor_of_168_252_315 : Nat.gcd 168 (Nat.gcd 252 315) = 21 := by
  sorry

end greatest_common_factor_of_168_252_315_l981_98162


namespace product_sum_theorem_l981_98116

theorem product_sum_theorem (a b c : ℝ) (h : a * b * c = 1) :
  a / (a * b + a + 1) + b / (b * c + b + 1) + c / (c * a + c + 1) = 1 := by
  sorry

end product_sum_theorem_l981_98116


namespace negative_sixty_four_to_four_thirds_l981_98106

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end negative_sixty_four_to_four_thirds_l981_98106


namespace max_distance_to_point_l981_98129

/-- The maximum distance from a point on the curve y = √(2 - x^2) to (0, -1) -/
theorem max_distance_to_point (x : ℝ) : 
  let y : ℝ := Real.sqrt (2 - x^2)
  let d : ℝ := Real.sqrt (x^2 + (y + 1)^2)
  d ≤ 1 + Real.sqrt 2 :=
by sorry

end max_distance_to_point_l981_98129


namespace union_of_A_and_B_l981_98112

open Set

def A : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 3}

theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -3 < x ∧ x ≤ 3} := by
  sorry

end union_of_A_and_B_l981_98112


namespace functional_inequality_solution_l981_98182

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x > 0, f x > 0}

/-- The functional inequality condition -/
def SatisfiesInequality (f : PositiveRealFunction) : Prop :=
  ∀ x y, x > 0 → y > 0 → f.val (x * y) ≤ (x * f.val y + y * f.val x) / 2

/-- The theorem statement -/
theorem functional_inequality_solution :
  ∀ f : PositiveRealFunction, SatisfiesInequality f →
  ∃ a : ℝ, a > 0 ∧ ∀ x > 0, f.val x = a * x :=
sorry

end functional_inequality_solution_l981_98182


namespace min_value_at_eight_l981_98122

theorem min_value_at_eight (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 24 / n ≥ 17 / 3 ∧
  ∃ (m : ℕ), m > 0 ∧ (m : ℝ) / 3 + 24 / m = 17 / 3 :=
sorry

end min_value_at_eight_l981_98122


namespace min_reciprocal_sum_l981_98173

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b = 5) :
  (1/a + 1/b) ≥ 5 + 2*Real.sqrt 6 := by
  sorry

end min_reciprocal_sum_l981_98173


namespace sanchez_rope_purchase_sanchez_rope_purchase_l981_98159

theorem sanchez_rope_purchase (inches_per_foot : ℕ) (this_week_inches : ℕ) : ℕ :=
  let last_week_feet := (this_week_inches / inches_per_foot) + 4
  last_week_feet

#check sanchez_rope_purchase 12 96 = 12

/- Proof
theorem sanchez_rope_purchase (inches_per_foot : ℕ) (this_week_inches : ℕ) : ℕ :=
  let last_week_feet := (this_week_inches / inches_per_foot) + 4
  last_week_feet
sorry
-/

end sanchez_rope_purchase_sanchez_rope_purchase_l981_98159


namespace total_tea_gallons_l981_98199

-- Define the number of containers
def num_containers : ℕ := 80

-- Define the relationship between containers and pints
def containers_to_pints : ℚ := 7 / (7/2)

-- Define the conversion rate from pints to gallons
def pints_per_gallon : ℕ := 8

-- Theorem stating the total amount of tea in gallons
theorem total_tea_gallons : 
  (↑num_containers * containers_to_pints) / ↑pints_per_gallon = 20 := by
  sorry

end total_tea_gallons_l981_98199


namespace four_digit_permutations_l981_98160

/-- The number of distinct permutations of a multiset with repeated elements -/
def multinomial_coefficient (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.map Nat.factorial).prod

/-- The multiset representation of the given digits -/
def digit_multiset : List ℕ := [3, 3, 3, 5]

/-- The total number of digits -/
def total_digits : ℕ := digit_multiset.length

/-- The list of repetitions for each unique digit -/
def repetitions : List ℕ := [3, 1]

theorem four_digit_permutations :
  multinomial_coefficient total_digits repetitions = 4 := by
  sorry

end four_digit_permutations_l981_98160


namespace B_power_93_l981_98123

def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 0; 0, 0, -1; 0, 1, 0]

theorem B_power_93 : B^93 = B := by sorry

end B_power_93_l981_98123


namespace min_value_of_expression_l981_98164

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/y = 1) :
  ∃ (m : ℝ), m = 11 + 4 * Real.sqrt 6 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 → 1/a + 1/b = 1 → 
  3*a/(a-1) + 8*b/(b-1) ≥ m :=
sorry

end min_value_of_expression_l981_98164


namespace decimal_division_equivalence_l981_98151

theorem decimal_division_equivalence : 
  ∀ (a b : ℚ), a = 11.7 ∧ b = 2.6 → 
    (a / b = 117 / 26) ∧ (a / b = 4.5) := by
  sorry

end decimal_division_equivalence_l981_98151


namespace initial_water_percentage_l981_98118

theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_volume = 70)
  (h2 : added_water = 14)
  (h3 : final_water_percentage = 25)
  (h4 : (initial_volume * x / 100 + added_water) / (initial_volume + added_water) = final_water_percentage / 100) :
  x = 10 := by
  sorry

end initial_water_percentage_l981_98118


namespace sophia_book_reading_l981_98198

theorem sophia_book_reading (total_pages : ℕ) (pages_read : ℕ) :
  total_pages = 90 →
  pages_read = (total_pages - pages_read) + 30 →
  pages_read = (2 : ℚ) / 3 * total_pages :=
by
  sorry

end sophia_book_reading_l981_98198


namespace circle_properties_l981_98109

/-- Given a circle with equation x^2 - 8x - y^2 + 2y = 6, prove its properties. -/
theorem circle_properties :
  let E : Set (ℝ × ℝ) := {p | let (x, y) := p; x^2 - 8*x - y^2 + 2*y = 6}
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ E ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c = 4 ∧
    d = 1 ∧
    s^2 = 11 ∧
    c + d + s = 5 + Real.sqrt 11 :=
by sorry

end circle_properties_l981_98109


namespace units_digit_of_n_l981_98190

/-- Given two natural numbers m and n, returns true if m has a units digit of 9 -/
def hasUnitsDigitOf9 (m : ℕ) : Prop :=
  m % 10 = 9

/-- Given a natural number n, returns its units digit -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 31^6) (h2 : hasUnitsDigitOf9 m) :
  unitsDigit n = 2 := by
  sorry

end units_digit_of_n_l981_98190


namespace smallest_marble_count_l981_98121

theorem smallest_marble_count : ∃ m : ℕ, 
  m > 0 ∧ 
  m % 9 = 1 ∧ 
  m % 7 = 3 ∧ 
  (∀ n : ℕ, n > 0 ∧ n % 9 = 1 ∧ n % 7 = 3 → m ≤ n) ∧ 
  m = 10 := by
sorry

end smallest_marble_count_l981_98121


namespace smallest_multiple_l981_98130

theorem smallest_multiple (n : ℕ) : n = 2015 ↔ 
  n > 0 ∧ 
  31 ∣ n ∧ 
  n % 97 = 6 ∧ 
  ∀ m : ℕ, m > 0 → 31 ∣ m → m % 97 = 6 → n ≤ m :=
by sorry

end smallest_multiple_l981_98130


namespace largest_prime_factor_of_1717_l981_98125

theorem largest_prime_factor_of_1717 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 1717 ∧ ∀ (q : ℕ), q.Prime → q ∣ 1717 → q ≤ p :=
by sorry

end largest_prime_factor_of_1717_l981_98125


namespace total_money_l981_98186

theorem total_money (john alice bob : ℚ) (h1 : john = 5/8) (h2 : alice = 7/20) (h3 : bob = 1/4) :
  john + alice + bob = 1.225 := by
  sorry

end total_money_l981_98186


namespace sqrt_six_star_sqrt_six_l981_98152

-- Define the ¤ operation
def star (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt_six_star_sqrt_six : star (Real.sqrt 6) (Real.sqrt 6) = 24 := by
  sorry

end sqrt_six_star_sqrt_six_l981_98152


namespace rain_is_random_event_l981_98111

/-- An event is random if its probability is strictly between 0 and 1 -/
def is_random_event (p : ℝ) : Prop := 0 < p ∧ p < 1

/-- The probability of rain in Xiangyang tomorrow -/
def rain_probability : ℝ := 0.75

theorem rain_is_random_event : is_random_event rain_probability := by
  sorry

end rain_is_random_event_l981_98111


namespace red_bank_amount_when_equal_l981_98175

/-- Proves that the amount in the red coin bank is 12,500 won when both banks have equal amounts -/
theorem red_bank_amount_when_equal (red_initial : ℕ) (yellow_initial : ℕ) 
  (red_daily : ℕ) (yellow_daily : ℕ) :
  red_initial = 8000 →
  yellow_initial = 5000 →
  red_daily = 300 →
  yellow_daily = 500 →
  ∃ d : ℕ, red_initial + d * red_daily = yellow_initial + d * yellow_daily ∧
          red_initial + d * red_daily = 12500 :=
by sorry

end red_bank_amount_when_equal_l981_98175


namespace correct_sum_and_digit_sum_l981_98183

def num1 : ℕ := 943587
def num2 : ℕ := 329430
def incorrect_sum : ℕ := 1412017

def change_digit (n : ℕ) (d e : ℕ) : ℕ := 
  sorry

theorem correct_sum_and_digit_sum :
  ∃ (d e : ℕ),
    (change_digit num1 d e + change_digit num2 d e ≠ incorrect_sum) ∧
    (change_digit num1 d e + change_digit num2 d e = num1 + change_digit num2 d e) ∧
    (d + e = 7) :=
  sorry

end correct_sum_and_digit_sum_l981_98183


namespace windows_installed_proof_l981_98185

/-- Calculates the number of windows already installed given the total number of windows,
    time to install each window, and remaining installation time. -/
def windows_installed (total_windows : ℕ) (install_time_per_window : ℕ) (remaining_time : ℕ) : ℕ :=
  total_windows - (remaining_time / install_time_per_window)

/-- Proves that given the specific conditions, the number of windows already installed is 8. -/
theorem windows_installed_proof :
  windows_installed 14 8 48 = 8 := by
  sorry

end windows_installed_proof_l981_98185


namespace sine_tangent_sum_greater_than_2pi_l981_98128

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_angles : A + B + C = π

-- State the theorem
theorem sine_tangent_sum_greater_than_2pi (t : AcuteTriangle) :
  Real.sin t.A + Real.sin t.B + Real.sin t.C +
  Real.tan t.A + Real.tan t.B + Real.tan t.C > 2 * π :=
by sorry

end sine_tangent_sum_greater_than_2pi_l981_98128


namespace thirtieth_triangular_number_l981_98146

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end thirtieth_triangular_number_l981_98146


namespace cookies_left_l981_98150

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens John bought -/
def dozens_bought : ℕ := 2

/-- The number of cookies John ate -/
def cookies_eaten : ℕ := 3

/-- Theorem: John has 21 cookies left -/
theorem cookies_left : dozens_bought * dozen - cookies_eaten = 21 := by
  sorry

end cookies_left_l981_98150


namespace triangle_to_square_area_ratio_l981_98169

/-- The ratio of the area of a specific triangle to the area of a square -/
theorem triangle_to_square_area_ratio :
  let square_side : ℝ := 10
  let triangle_vertices : List (ℝ × ℝ) := [(2, 4), (4, 4), (4, 6)]
  let triangle_area := abs ((4 - 2) * (6 - 4) / 2)
  let square_area := square_side ^ 2
  triangle_area / square_area = 1 / 50 := by
  sorry

end triangle_to_square_area_ratio_l981_98169


namespace sea_lion_penguin_ratio_l981_98107

theorem sea_lion_penguin_ratio :
  let sea_lions : ℕ := 48
  let penguins : ℕ := sea_lions + 84
  (sea_lions : ℚ) / penguins = 4 / 11 := by
sorry

end sea_lion_penguin_ratio_l981_98107


namespace sum_perpendiculars_equals_altitude_l981_98165

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Checks if a triangle is isosceles with AB = AC -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  dist t.A t.B = dist t.A t.C

/-- Calculates the altitude of a triangle -/
noncomputable def Triangle.altitude (t : Triangle) : ℝ := sorry

/-- Calculates the perpendicular distance from a point to a line segment -/
noncomputable def perpendicularDistance (p : Point) (a b : Point) : ℝ := sorry

/-- Checks if a point is inside or on a triangle -/
def Triangle.containsPoint (t : Triangle) (p : Point) : Prop := sorry

/-- Theorem: Sum of perpendiculars equals altitude for isosceles triangle -/
theorem sum_perpendiculars_equals_altitude (t : Triangle) (p : Point) :
  t.isIsosceles →
  t.containsPoint p →
  perpendicularDistance p t.B t.C + 
  perpendicularDistance p t.C t.A + 
  perpendicularDistance p t.A t.B = 
  t.altitude := by sorry

end sum_perpendiculars_equals_altitude_l981_98165


namespace songs_added_l981_98100

/-- Calculates the number of new songs added to an mp3 player. -/
theorem songs_added (initial : ℕ) (deleted : ℕ) (final : ℕ) : 
  initial = 11 → deleted = 9 → final = 10 → final - (initial - deleted) = 8 := by
  sorry

end songs_added_l981_98100


namespace monotone_increasing_k_range_l981_98101

/-- A function f(x) = kx^2 - ln x is monotonically increasing in the interval (1, +∞) -/
def is_monotone_increasing (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → f x ≤ f y

/-- The range of k for which f(x) = kx^2 - ln x is monotonically increasing in (1, +∞) -/
theorem monotone_increasing_k_range (k : ℝ) :
  (is_monotone_increasing (fun x => k * x^2 - Real.log x) k) → k ≥ 1 := by
  sorry

end monotone_increasing_k_range_l981_98101


namespace f_lower_bound_f_inequality_solution_l981_98153

def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

theorem f_lower_bound : ∀ x : ℝ, f x ≥ 4 := by sorry

theorem f_inequality_solution : 
  ∀ x : ℝ, f x ≥ x^2 - 2*x + 4 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

end f_lower_bound_f_inequality_solution_l981_98153


namespace division_problem_l981_98193

theorem division_problem : (62976 : ℕ) / 512 = 123 := by sorry

end division_problem_l981_98193


namespace range_of_m_l981_98114

open Set

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
  x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ¬∃ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 = 0

-- Define the range of m
def range_m : Set ℝ := Ioc 1 2 ∪ Ici 3

-- Theorem statement
theorem range_of_m : 
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ 
  (∀ m : ℝ, m ∈ range_m ↔ (p m ∨ q m) ∧ ¬(p m ∧ q m)) :=
sorry

end range_of_m_l981_98114


namespace project_hours_proof_l981_98108

theorem project_hours_proof (kate mark pat : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : 3 * pat = mark)
  (h3 : mark = kate + 105) :
  kate + mark + pat = 189 := by
  sorry

end project_hours_proof_l981_98108


namespace f_monotonicity_and_extrema_l981_98113

noncomputable def f (a x : ℝ) : ℝ := (a + 1/a) * Real.log x + 1/x - x

theorem f_monotonicity_and_extrema :
  ∀ a : ℝ, a > 0 →
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 1/a ∧ 1/a < x₂ ∧ x₂ < 1 → f a x₁ < f a (1/a) ∧ f a (1/a) < f a x₂) ∧
    (f a (1/a) = -(a + 1/a) * Real.log a + a - 1/a) ∧
    (f a a = (a + 1/a) * Real.log a + 1/a - a) ∧
    (∀ x : ℝ, x > 0 → f a x ≥ f a (1/a) ∧ f a x ≤ f a a) := by
  sorry

end f_monotonicity_and_extrema_l981_98113


namespace square_perimeter_ratio_l981_98140

theorem square_perimeter_ratio (a b : ℝ) (h : a > 0) (k : b > 0) (area_ratio : a^2 / b^2 = 49 / 64) :
  a / b = 7 / 8 := by
  sorry

end square_perimeter_ratio_l981_98140


namespace parallel_lines_a_values_l981_98127

/-- Given two lines l₁ and l₂, if they are parallel, then a = -1 or a = 2 -/
theorem parallel_lines_a_values (a : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | (a - 1) * x + y + 3 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | 2 * x + a * y + 1 = 0}
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₂ → (a - 1) * (x₂ - x₁) = -(y₂ - y₁)) →
  a = -1 ∨ a = 2 := by
sorry

end parallel_lines_a_values_l981_98127


namespace rectangle_y_value_l981_98103

/-- Given a rectangle with vertices at (2, y), (10, y), (2, -1), and (10, -1),
    where y is negative and the area is 96 square units, prove that y = -13. -/
theorem rectangle_y_value (y : ℝ) : 
  y < 0 → -- y is negative
  (10 - 2) * |(-1) - y| = 96 → -- area of the rectangle is 96
  y = -13 := by sorry

end rectangle_y_value_l981_98103


namespace sufficient_but_not_necessary_l981_98126

-- Define propositions p and q
variable (p q : Prop)

-- Define the statement "p or q is false" is sufficient for "not p is true"
def is_sufficient : Prop :=
  (¬(p ∨ q)) → (¬p)

-- Define the statement "p or q is false" is not necessary for "not p is true"
def is_not_necessary : Prop :=
  ∃ (p q : Prop), (¬p) ∧ ¬(¬(p ∨ q))

-- The main theorem stating that "p or q is false" is sufficient but not necessary for "not p is true"
theorem sufficient_but_not_necessary :
  (is_sufficient p q) ∧ is_not_necessary :=
sorry

end sufficient_but_not_necessary_l981_98126


namespace function_value_at_two_l981_98102

theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = 3 * x - 2) :
  f 2 = -1/2 := by
  sorry

end function_value_at_two_l981_98102


namespace unique_two_digit_number_l981_98134

theorem unique_two_digit_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10 ≠ n % 10) ∧ 
  n^2 = (n / 10 + n % 10)^3 := by
  sorry

end unique_two_digit_number_l981_98134


namespace greatest_root_of_g_l981_98176

def g (x : ℝ) : ℝ := 20 * x^4 - 21 * x^2 + 5

theorem greatest_root_of_g :
  ∃ (r : ℝ), g r = 0 ∧ r = 1 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
by sorry

end greatest_root_of_g_l981_98176


namespace odd_even_sum_theorem_l981_98137

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_sum_theorem (f g : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_even : is_even_function g) 
  (h_diff : ∀ x, f x - g x = x^2 + 9*x + 12) : 
  ∀ x, f x + g x = -x^2 + 9*x - 12 :=
by sorry

end odd_even_sum_theorem_l981_98137


namespace m_range_l981_98167

-- Define the plane region
def plane_region (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |2 * p.1 + p.2 + m| < 3}

-- Theorem statement
theorem m_range (m : ℝ) :
  ((0, 0) ∈ plane_region m) ∧
  ((-1, 1) ∈ plane_region m) ↔
  -2 < m ∧ m < 3 :=
by
  sorry


end m_range_l981_98167


namespace platform_length_l981_98148

/-- Given a train and platform with specific properties, prove the length of the platform. -/
theorem platform_length (train_length : ℝ) (time_cross_platform : ℝ) (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 40)
  (h3 : time_cross_pole = 18) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 367 := by
sorry

end platform_length_l981_98148


namespace three_digit_sum_problem_l981_98136

theorem three_digit_sum_problem (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  122 * a + 212 * b + 221 * c = 2003 →
  100 * a + 10 * b + c = 345 :=
by sorry

end three_digit_sum_problem_l981_98136


namespace system_of_equations_sum_l981_98163

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end system_of_equations_sum_l981_98163


namespace rahul_deepak_age_ratio_l981_98145

theorem rahul_deepak_age_ratio : 
  ∀ (rahul_age deepak_age : ℕ),
    rahul_age + 6 = 18 →
    deepak_age = 9 →
    (rahul_age : ℚ) / deepak_age = 4 / 3 := by
  sorry

end rahul_deepak_age_ratio_l981_98145


namespace union_of_A_and_B_l981_98181

-- Define the sets A, B, and C
def A (x : ℝ) : Set ℝ := {2, -1, x^2 - x + 1}
def B (x y : ℝ) : Set ℝ := {2*y, -4, x + 4}
def C : Set ℝ := {-1}

-- State the theorem
theorem union_of_A_and_B (x y : ℝ) :
  (A x ∩ B x y = C) →
  (A x ∪ B x y = {2, -1, x^2 - x + 1, 2*y, -4, x + 4}) :=
by sorry

end union_of_A_and_B_l981_98181


namespace pr_less_than_qr_implies_p_less_than_q_l981_98174

theorem pr_less_than_qr_implies_p_less_than_q
  (r p q : ℝ) 
  (h1 : r < 0) 
  (h2 : p * q ≠ 0) 
  (h3 : p * r < q * r) : 
  p < q :=
by sorry

end pr_less_than_qr_implies_p_less_than_q_l981_98174


namespace sum_and_reciprocal_squared_l981_98143

theorem sum_and_reciprocal_squared (x N : ℝ) (h1 : x ≠ 0) (h2 : x + 1/x = N) (h3 : x^2 + 1/x^2 = 2) : N = 2 := by
  sorry

end sum_and_reciprocal_squared_l981_98143


namespace golf_distance_ratio_l981_98110

/-- Proves that the ratio of the distance traveled on the second turn to the distance traveled on the first turn is 1/2 in a golf scenario. -/
theorem golf_distance_ratio
  (total_distance : ℝ)
  (first_turn_distance : ℝ)
  (overshoot_distance : ℝ)
  (h1 : total_distance = 250)
  (h2 : first_turn_distance = 180)
  (h3 : overshoot_distance = 20)
  : (total_distance - first_turn_distance + overshoot_distance) / first_turn_distance = 1 / 2 := by
  sorry

end golf_distance_ratio_l981_98110


namespace buddy_baseball_cards_l981_98133

theorem buddy_baseball_cards (monday tuesday wednesday thursday : ℕ) : 
  tuesday = monday / 2 →
  wednesday = tuesday + 12 →
  thursday = wednesday + tuesday / 3 →
  thursday = 32 →
  monday = 40 :=
by
  sorry

end buddy_baseball_cards_l981_98133


namespace cube_difference_l981_98138

theorem cube_difference (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 - b^3 = 992 := by
sorry

end cube_difference_l981_98138


namespace glitched_clock_correct_time_fraction_l981_98131

/-- Represents a 12-hour digital clock with a glitch where '2' is displayed as '7' -/
structure GlitchedClock where
  /-- The number of hours in the clock cycle -/
  hours : Nat
  /-- The number of minutes in an hour -/
  minutes_per_hour : Nat
  /-- The digit that is erroneously displayed -/
  glitched_digit : Nat
  /-- The digit that replaces the glitched digit -/
  replacement_digit : Nat

/-- The fraction of the day that the glitched clock shows the correct time -/
def correct_time_fraction (clock : GlitchedClock) : ℚ :=
  sorry

/-- Theorem stating that the fraction of correct time for the given clock is 55/72 -/
theorem glitched_clock_correct_time_fraction :
  let clock : GlitchedClock := {
    hours := 12,
    minutes_per_hour := 60,
    glitched_digit := 2,
    replacement_digit := 7
  }
  correct_time_fraction clock = 55 / 72 := by
  sorry

end glitched_clock_correct_time_fraction_l981_98131


namespace kenny_basketball_time_l981_98194

/-- Represents Kenny's activities and their durations --/
structure KennyActivities where
  basketball : ℝ
  running : ℝ
  trumpet : ℝ
  swimming : ℝ
  studying : ℝ

/-- Theorem stating the duration of Kenny's basketball playing --/
theorem kenny_basketball_time (k : KennyActivities) 
  (h1 : k.running = 2 * k.basketball)
  (h2 : k.trumpet = 2 * k.running)
  (h3 : k.swimming = 2.5 * k.trumpet)
  (h4 : k.studying = 0.5 * k.swimming)
  (h5 : k.trumpet = 40) : 
  k.basketball = 10 := by
  sorry

end kenny_basketball_time_l981_98194


namespace remainder_relationship_l981_98191

theorem remainder_relationship (P P' D R R' C : ℕ) (h1 : P > P') (h2 : P % D = R) (h3 : P' % D = R') : 
  ∃ (s r : ℕ), ((P + C) * P') % D = s ∧ (P * P') % D = r ∧ 
  (∃ (C1 D1 : ℕ), s > r) ∧ (∃ (C2 D2 : ℕ), s < r) :=
sorry

end remainder_relationship_l981_98191


namespace apartment_doors_count_l981_98120

/-- Calculates the total number of doors needed for apartment buildings -/
def total_doors (num_buildings : ℕ) (floors_per_building : ℕ) (apartments_per_floor : ℕ) (doors_per_apartment : ℕ) : ℕ :=
  num_buildings * floors_per_building * apartments_per_floor * doors_per_apartment

/-- Proves that the total number of doors needed for the given apartment buildings is 1008 -/
theorem apartment_doors_count :
  total_doors 2 12 6 7 = 1008 := by
  sorry

end apartment_doors_count_l981_98120


namespace planes_intersect_necessary_not_sufficient_l981_98115

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the intersection relation between two planes
variable (planesIntersect : Plane → Plane → Prop)

-- Define the skew relation between two lines
variable (skewLines : Line → Line → Prop)

-- Define the theorem
theorem planes_intersect_necessary_not_sufficient
  (α β : Plane) (m n : Line)
  (h1 : ¬ planesIntersect α β)
  (h2 : perpendicular m α)
  (h3 : perpendicular n β) :
  (∀ α' β' m' n', planesIntersect α' β' → skewLines m' n' → perpendicular m' α' → perpendicular n' β' → True) ∧
  (∃ α' β' m' n', planesIntersect α' β' ∧ ¬ skewLines m' n' ∧ perpendicular m' α' ∧ perpendicular n' β') :=
sorry

end planes_intersect_necessary_not_sufficient_l981_98115


namespace oil_drilling_probability_l981_98196

/-- The probability of hitting an oil layer when drilling in a sea area -/
theorem oil_drilling_probability 
  (total_area : ℝ) 
  (oil_area : ℝ) 
  (h1 : total_area = 10000) 
  (h2 : oil_area = 40) : 
  oil_area / total_area = 1 / 250 := by
sorry

end oil_drilling_probability_l981_98196


namespace hot_dog_buns_per_package_l981_98157

/-- Proves that the number of hot dog buns in one package is 8, given the conditions of the problem -/
theorem hot_dog_buns_per_package 
  (total_packages : ℕ) 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (buns_per_student : ℕ) 
  (h1 : total_packages = 30)
  (h2 : num_classes = 4)
  (h3 : students_per_class = 30)
  (h4 : buns_per_student = 2) : 
  (num_classes * students_per_class * buns_per_student) / total_packages = 8 := by
  sorry

#check hot_dog_buns_per_package

end hot_dog_buns_per_package_l981_98157


namespace interesting_triple_existence_l981_98161

/-- Definition of an interesting triple -/
def is_interesting (a b c : ℕ) : Prop :=
  (c^2 + 1) ∣ ((a^2 + 1) * (b^2 + 1)) ∧
  ¬((c^2 + 1) ∣ (a^2 + 1)) ∧
  ¬((c^2 + 1) ∣ (b^2 + 1))

theorem interesting_triple_existence 
  (a b c : ℕ) 
  (h : is_interesting a b c) : 
  ∃ u v : ℕ, is_interesting u v c ∧ u * v < c^3 := by
  sorry

end interesting_triple_existence_l981_98161


namespace cafe_pricing_theorem_l981_98119

/-- Represents the pricing structure of a café -/
structure CafePrices where
  sandwich : ℝ
  coffee : ℝ
  pie : ℝ

/-- The café's pricing satisfies the given conditions -/
def satisfies_conditions (p : CafePrices) : Prop :=
  4 * p.sandwich + 9 * p.coffee + p.pie = 4.30 ∧
  7 * p.sandwich + 14 * p.coffee + p.pie = 7.00

/-- Calculates the total cost for a given order -/
def order_cost (p : CafePrices) (sandwiches coffees pies : ℕ) : ℝ :=
  p.sandwich * sandwiches + p.coffee * coffees + p.pie * pies

/-- Theorem stating that the cost of 11 sandwiches, 23 coffees, and 2 pies is $18.87 -/
theorem cafe_pricing_theorem (p : CafePrices) :
  satisfies_conditions p →
  order_cost p 11 23 2 = 18.87 := by
  sorry

end cafe_pricing_theorem_l981_98119


namespace divisibility_property_l981_98141

theorem divisibility_property (m n : ℕ) (h : 24 ∣ (m * n + 1)) : 24 ∣ (m + n) := by
  sorry

end divisibility_property_l981_98141


namespace infinitely_many_divisible_by_15_l981_98197

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem infinitely_many_divisible_by_15 :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ 15 ∣ v (15 * k) :=
sorry

end infinitely_many_divisible_by_15_l981_98197


namespace max_of_expression_l981_98192

theorem max_of_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 50) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ 18.124 ∧
  (x = 16 → Real.sqrt (x + 50) + Real.sqrt (20 - x) + 2 * Real.sqrt x = 18.124) :=
by sorry

end max_of_expression_l981_98192


namespace quadratic_inequality_empty_solution_set_l981_98155

theorem quadratic_inequality_empty_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) ↔ a ∈ Set.Icc (-4 : ℝ) 4 :=
sorry

end quadratic_inequality_empty_solution_set_l981_98155


namespace election_winner_votes_l981_98135

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) :
  winner_percentage = 55 / 100 →
  vote_difference = 100 →
  (winner_percentage * total_votes).num = 
    (1 - winner_percentage) * total_votes + vote_difference →
  (winner_percentage * total_votes).num = 550 := by
sorry

end election_winner_votes_l981_98135


namespace constant_triangle_sum_l981_98149

/-- Given a rectangle ABCD with width 'a' and height 'b', and a line 'r' parallel to AB
    intersecting diagonal AC at point (x₀, y₀), the sum of the areas of the two triangles
    formed by 'r' is constant and equal to (a*b)/2, regardless of the position of 'r'. -/
theorem constant_triangle_sum (a b x₀ : ℝ) (ha : a > 0) (hb : b > 0) (hx : 0 ≤ x₀ ∧ x₀ ≤ a) :
  let y₀ := (b / a) * x₀
  let area₁ := (1 / 2) * b * x₀
  let area₂ := (1 / 2) * b * (a - x₀)
  area₁ + area₂ = (a * b) / 2 :=
by sorry

end constant_triangle_sum_l981_98149


namespace markup_calculation_l981_98104

/-- Calculates the required markup given the purchase price, overhead percentage, and desired net profit. -/
def calculate_markup (purchase_price : ℝ) (overhead_percent : ℝ) (net_profit : ℝ) : ℝ :=
  purchase_price * overhead_percent + net_profit

/-- Theorem stating that the markup for the given conditions is $53.75 -/
theorem markup_calculation :
  let purchase_price : ℝ := 75
  let overhead_percent : ℝ := 0.45
  let net_profit : ℝ := 20
  calculate_markup purchase_price overhead_percent net_profit = 53.75 := by
  sorry

end markup_calculation_l981_98104


namespace glasses_cost_glasses_cost_proof_l981_98177

/-- Calculate the total cost of glasses after discounts -/
theorem glasses_cost (frame_cost lens_cost : ℝ) 
  (insurance_coverage : ℝ) (frame_coupon : ℝ) : ℝ :=
  let discounted_lens_cost := lens_cost * (1 - insurance_coverage)
  let discounted_frame_cost := frame_cost - frame_coupon
  discounted_lens_cost + discounted_frame_cost

/-- Prove that the total cost of glasses after discounts is $250 -/
theorem glasses_cost_proof :
  glasses_cost 200 500 0.8 50 = 250 := by
  sorry

end glasses_cost_glasses_cost_proof_l981_98177


namespace square_of_102_l981_98147

theorem square_of_102 : 102 * 102 = 10404 := by
  sorry

end square_of_102_l981_98147


namespace kim_gum_needs_l981_98188

/-- The number of cousins Kim has -/
def num_cousins : ℕ := 4

/-- The number of gum pieces Kim wants to give to each cousin -/
def gum_per_cousin : ℕ := 5

/-- The total number of gum pieces Kim needs -/
def total_gum : ℕ := num_cousins * gum_per_cousin

/-- Theorem stating that the total number of gum pieces Kim needs is 20 -/
theorem kim_gum_needs : total_gum = 20 := by sorry

end kim_gum_needs_l981_98188


namespace grade_10_sample_size_l981_98166

/-- Represents the number of students to be selected from a stratum in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (stratum_size : ℕ) (sample_size : ℕ) : ℕ :=
  (stratum_size * sample_size) / total_population

/-- The problem statement -/
theorem grade_10_sample_size :
  stratified_sample_size 4500 1200 150 = 40 := by
  sorry


end grade_10_sample_size_l981_98166


namespace power_product_equality_l981_98195

theorem power_product_equality : (81 : ℝ) ^ (1/4) * (81 : ℝ) ^ (1/5) = 3 * (3 ^ 4) ^ (1/5) := by sorry

end power_product_equality_l981_98195


namespace circle_points_count_l981_98144

def number_of_triangles (n : ℕ) : ℕ := n.choose 4

theorem circle_points_count : ∃ (n : ℕ), n > 3 ∧ number_of_triangles n = 126 ∧ n = 9 := by
  sorry

end circle_points_count_l981_98144


namespace even_sum_probability_l981_98170

/-- Represents a spinner with its possible outcomes -/
structure Spinner :=
  (outcomes : List ℕ)

/-- The probability of getting an even sum from spinning all three spinners -/
def probability_even_sum (s t u : Spinner) : ℚ :=
  sorry

/-- The spinners as defined in the problem -/
def spinner_s : Spinner := ⟨[1, 2, 4]⟩
def spinner_t : Spinner := ⟨[3, 3, 6]⟩
def spinner_u : Spinner := ⟨[2, 4, 6]⟩

/-- The main theorem to prove -/
theorem even_sum_probability :
  probability_even_sum spinner_s spinner_t spinner_u = 5/9 :=
sorry

end even_sum_probability_l981_98170


namespace girls_on_playground_l981_98156

theorem girls_on_playground (total_children boys : ℕ) 
  (h1 : total_children = 117)
  (h2 : boys = 40) :
  total_children - boys = 77 := by
sorry

end girls_on_playground_l981_98156


namespace petyas_fruits_l981_98179

theorem petyas_fruits (total : ℕ) (apples tangerines oranges : ℕ) : 
  total = 20 →
  apples + tangerines + oranges = total →
  tangerines * 6 = apples →
  apples > oranges →
  oranges = 6 :=
by sorry

end petyas_fruits_l981_98179


namespace melanie_dimes_problem_l981_98132

/-- Calculates the number of dimes Melanie's mother gave her -/
def mothers_dimes (initial : ℤ) (given_to_dad : ℤ) (final : ℤ) : ℤ :=
  final - (initial - given_to_dad)

theorem melanie_dimes_problem : mothers_dimes 7 8 3 = 4 := by
  sorry

end melanie_dimes_problem_l981_98132


namespace polynomial_factor_l981_98180

theorem polynomial_factor (x : ℝ) : ∃ (q : ℝ → ℝ), x^2 - 1 = (x + 1) * q x := by
  sorry

end polynomial_factor_l981_98180


namespace largest_negative_smallest_positive_smallest_abs_l981_98172

theorem largest_negative_smallest_positive_smallest_abs (a b c : ℤ) : 
  (∀ x : ℤ, x < 0 → x ≤ a) →  -- a is the largest negative integer
  (∀ x : ℤ, x > 0 → b ≤ x) →  -- b is the smallest positive integer
  (∀ x : ℤ, |c| ≤ |x|) →      -- c has the smallest absolute value
  a + c - b = -2 := by
sorry

end largest_negative_smallest_positive_smallest_abs_l981_98172
