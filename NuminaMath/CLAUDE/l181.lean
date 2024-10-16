import Mathlib

namespace NUMINAMATH_CALUDE_shoes_cost_proof_l181_18166

def budget : ℕ := 200
def shirt_cost : ℕ := 30
def pants_cost : ℕ := 46
def coat_cost : ℕ := 38
def socks_cost : ℕ := 11
def belt_cost : ℕ := 18
def remaining : ℕ := 16

theorem shoes_cost_proof :
  budget - (shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost) - remaining = 41 := by
  sorry

end NUMINAMATH_CALUDE_shoes_cost_proof_l181_18166


namespace NUMINAMATH_CALUDE_inequality_proof_l181_18170

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  x / (1 + x^2) + y / (1 + y^2) + z / (1 + z^2) ≤ 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l181_18170


namespace NUMINAMATH_CALUDE_city_mpg_is_14_l181_18174

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℝ
  city_miles_per_tankful : ℝ
  city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given the car's fuel efficiency data -/
def calculate_city_mpg (car : CarFuelEfficiency) : ℝ :=
  sorry

/-- Theorem stating that for a car with given fuel efficiency data, 
    the city miles per gallon is 14 -/
theorem city_mpg_is_14 (car : CarFuelEfficiency) 
  (h1 : car.highway_miles_per_tankful = 480)
  (h2 : car.city_miles_per_tankful = 336)
  (h3 : car.city_mpg_difference = 6) :
  calculate_city_mpg car = 14 := by
  sorry

end NUMINAMATH_CALUDE_city_mpg_is_14_l181_18174


namespace NUMINAMATH_CALUDE_sum_of_cubes_l181_18148

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l181_18148


namespace NUMINAMATH_CALUDE_exponential_inequality_l181_18142

theorem exponential_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  Real.exp a * Real.exp c > Real.exp b * Real.exp d := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l181_18142


namespace NUMINAMATH_CALUDE_mrs_excellent_class_size_l181_18179

/-- Represents the number of students in Mrs. Excellent's class -/
def total_students : ℕ := 29

/-- Represents the number of girls in the class -/
def girls : ℕ := 13

/-- Represents the number of boys in the class -/
def boys : ℕ := girls + 3

/-- Represents the total number of jellybeans Mrs. Excellent has -/
def total_jellybeans : ℕ := 450

/-- Represents the number of jellybeans left after distribution -/
def leftover_jellybeans : ℕ := 10

theorem mrs_excellent_class_size :
  (girls * girls + boys * boys + leftover_jellybeans = total_jellybeans) ∧
  (girls + boys = total_students) := by
  sorry

#check mrs_excellent_class_size

end NUMINAMATH_CALUDE_mrs_excellent_class_size_l181_18179


namespace NUMINAMATH_CALUDE_smallest_angle_tan_equation_l181_18122

theorem smallest_angle_tan_equation (x : Real) : 
  (x > 0) →
  (x < 9 * Real.pi / 180) →
  (Real.tan (4 * x) ≠ (Real.cos x - Real.sin x) / (Real.cos x + Real.sin x)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_tan_equation_l181_18122


namespace NUMINAMATH_CALUDE_seashells_to_glass_ratio_l181_18189

/-- Represents the number of treasures Simon collected -/
structure Treasures where
  sandDollars : ℕ
  glasspieces : ℕ
  seashells : ℕ

/-- The conditions of Simon's treasure collection -/
def simonsTreasures : Treasures where
  sandDollars := 10
  glasspieces := 3 * 10
  seashells := 3 * 10

/-- The total number of treasures Simon collected -/
def totalTreasures : ℕ := 190

/-- Theorem stating that the ratio of seashells to glass pieces is 1:1 -/
theorem seashells_to_glass_ratio (t : Treasures) 
  (h1 : t.sandDollars = simonsTreasures.sandDollars)
  (h2 : t.glasspieces = 3 * t.sandDollars)
  (h3 : t.seashells = t.glasspieces)
  (h4 : t.sandDollars + t.glasspieces + t.seashells = totalTreasures) :
  t.seashells = t.glasspieces := by
  sorry

#check seashells_to_glass_ratio

end NUMINAMATH_CALUDE_seashells_to_glass_ratio_l181_18189


namespace NUMINAMATH_CALUDE_ellipse_max_min_sum_l181_18102

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the function we want to maximize/minimize
def f (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem ellipse_max_min_sum :
  (∃ x y : ℝ, ellipse x y ∧ f x y = Real.sqrt 5) ∧
  (∃ x y : ℝ, ellipse x y ∧ f x y = -Real.sqrt 5) ∧
  (∀ x y : ℝ, ellipse x y → f x y ≤ Real.sqrt 5) ∧
  (∀ x y : ℝ, ellipse x y → f x y ≥ -Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_max_min_sum_l181_18102


namespace NUMINAMATH_CALUDE_certain_number_value_l181_18150

theorem certain_number_value (p q : ℕ) (x : ℚ) 
  (hp : p > 1) 
  (hq : q > 1) 
  (hx : x * (p + 1) = 28 * (q + 1)) 
  (hpq_min : ∀ (p' q' : ℕ), p' > 1 → q' > 1 → p' + q' < p + q → ¬∃ (x' : ℚ), x' * (p' + 1) = 28 * (q' + 1)) :
  x = 392 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_l181_18150


namespace NUMINAMATH_CALUDE_john_calorie_burn_l181_18106

/-- Represents the number of calories John burns per day -/
def calories_burned_per_day : ℕ := 2300

/-- Represents the number of calories John eats per day -/
def calories_eaten_per_day : ℕ := 1800

/-- Represents the number of calories needed to be burned to lose 1 pound -/
def calories_per_pound : ℕ := 4000

/-- Represents the number of days it takes John to lose 10 pounds -/
def days_to_lose_10_pounds : ℕ := 80

/-- Represents the number of pounds John wants to lose -/
def pounds_to_lose : ℕ := 10

theorem john_calorie_burn :
  calories_burned_per_day = 
    calories_eaten_per_day + 
    (pounds_to_lose * calories_per_pound) / days_to_lose_10_pounds :=
by
  sorry

end NUMINAMATH_CALUDE_john_calorie_burn_l181_18106


namespace NUMINAMATH_CALUDE_plane_perp_necessary_not_sufficient_l181_18131

/-- Two planes are perpendicular -/
def planes_perpendicular (α β : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (m : Line) (β : Plane) : Prop := sorry

/-- A line lies in a plane -/
def line_in_plane (m : Line) (α : Plane) : Prop := sorry

theorem plane_perp_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (h_different : α ≠ β)
  (h_m_in_α : line_in_plane m α) :
  (planes_perpendicular α β → line_perpendicular_to_plane m β) ∧
  ¬(line_perpendicular_to_plane m β → planes_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_plane_perp_necessary_not_sufficient_l181_18131


namespace NUMINAMATH_CALUDE_overlapped_area_of_reflected_triangle_l181_18113

/-- Given three points O, A, and B on a coordinate plane, and the reflection of triangle OAB
    along the line y = 6 creating triangle PQR, prove that the overlapped area is 8 square units. -/
theorem overlapped_area_of_reflected_triangle (O A B : ℝ × ℝ) : 
  O = (0, 0) →
  A = (12, 2) →
  B = (0, 8) →
  let P : ℝ × ℝ := (O.1, 12 - O.2)
  let Q : ℝ × ℝ := (A.1, 12 - A.2)
  let R : ℝ × ℝ := (B.1, 12 - B.2)
  let M : ℝ × ℝ := (4, 6)
  8 = (1/2) * |M.1 - B.1| * |B.2 - R.2| := by sorry

end NUMINAMATH_CALUDE_overlapped_area_of_reflected_triangle_l181_18113


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l181_18197

theorem average_of_four_numbers (p q r s : ℝ) 
  (h : (5 / 4) * (p + q + r + s) = 20) : 
  (p + q + r + s) / 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l181_18197


namespace NUMINAMATH_CALUDE_polynomial_value_l181_18137

theorem polynomial_value (x : ℝ) (h : x^2 + 2*x + 1 = 4) : 2*x^2 + 4*x + 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l181_18137


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l181_18147

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of the given expression -/
def expression : ℕ := 8 * 19 * 1981 - 8^3

theorem units_digit_of_expression :
  unitsDigit expression = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l181_18147


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l181_18192

theorem quadratic_inequality_solution_set (a b c : ℝ) :
  a > 0 → (∀ x, a * x^2 + b * x + c > 0) ↔ b^2 - 4*a*c < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l181_18192


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l181_18177

-- Define the polynomial, divisor, and quotient
def P (z : ℝ) : ℝ := 2*z^4 - 3*z^3 + 5*z^2 - 7*z + 6
def D (z : ℝ) : ℝ := 2*z - 3
def Q (z : ℝ) : ℝ := z^3 + z^2 - 4*z + 5

-- State the theorem
theorem polynomial_division_remainder :
  ∃ (R : ℝ), ∀ (z : ℝ), P z = D z * Q z + R :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l181_18177


namespace NUMINAMATH_CALUDE_tylers_age_l181_18139

theorem tylers_age :
  ∀ (tyler_age brother_age : ℕ),
  tyler_age = brother_age - 3 →
  tyler_age + brother_age = 11 →
  tyler_age = 4 := by
sorry

end NUMINAMATH_CALUDE_tylers_age_l181_18139


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l181_18173

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 4 / 3 →
  c / a = 2 →
  a * b * c = 1944 →
  max a (max b c) = 18 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l181_18173


namespace NUMINAMATH_CALUDE_correct_security_response_l181_18159

/-- Represents an email with potentially suspicious characteristics -/
structure Email :=
  (sender : String)
  (content : String)
  (links : List String)

/-- Represents a website with potentially suspicious characteristics -/
structure Website :=
  (url : String)
  (content : String)
  (requestedInfo : List String)

/-- Represents an offer that may be unrealistic -/
structure Offer :=
  (description : String)
  (price : Nat)
  (originalPrice : Nat)

/-- Represents security measures to be followed -/
inductive SecurityMeasure
  | UseSecureNetwork
  | UseAntivirus
  | UpdateApplications
  | CheckHTTPS
  | UseComplexPasswords
  | Use2FA
  | RecognizeBankProtocols

/-- Represents the correct security response -/
structure SecurityResponse :=
  (trustSource : Bool)
  (enterInformation : Bool)
  (measures : List SecurityMeasure)

/-- Main theorem: Given suspicious conditions, prove the correct security response -/
theorem correct_security_response 
  (email : Email) 
  (website : Website) 
  (offer : Offer) : 
  (email.sender ≠ "official@aliexpress.com" ∧ 
   website.url ≠ "https://www.aliexpress.com" ∧ 
   offer.price < offer.originalPrice / 10) → 
  ∃ (response : SecurityResponse), 
    response.trustSource = false ∧ 
    response.enterInformation = false ∧ 
    response.measures.length ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_correct_security_response_l181_18159


namespace NUMINAMATH_CALUDE_f_neither_odd_nor_even_l181_18194

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Statement: f is neither odd nor even
theorem f_neither_odd_nor_even :
  (∃ x : ℝ, f (-x) ≠ f x) ∧ (∃ x : ℝ, f (-x) ≠ -f x) := by
  sorry

end NUMINAMATH_CALUDE_f_neither_odd_nor_even_l181_18194


namespace NUMINAMATH_CALUDE_john_camera_rental_payment_l181_18107

def camera_rental_problem (camera_value : ℝ) (rental_rate : ℝ) (rental_weeks : ℕ) (friend_contribution_rate : ℝ) : Prop :=
  let weekly_rental := camera_value * rental_rate
  let total_rental := weekly_rental * rental_weeks
  let friend_contribution := total_rental * friend_contribution_rate
  let john_payment := total_rental - friend_contribution
  john_payment = 1200

theorem john_camera_rental_payment :
  camera_rental_problem 5000 0.10 4 0.40 := by
  sorry

end NUMINAMATH_CALUDE_john_camera_rental_payment_l181_18107


namespace NUMINAMATH_CALUDE_senior_class_size_l181_18140

theorem senior_class_size (total : ℕ) 
  (h1 : total / 5 = total / 5)  -- A fifth of the senior class is in the marching band
  (h2 : (total / 5) / 2 = (total / 5) / 2)  -- Half of the marching band plays brass instruments
  (h3 : ((total / 5) / 2) / 5 = ((total / 5) / 2) / 5)  -- A fifth of the brass instrument players play saxophone
  (h4 : (((total / 5) / 2) / 5) / 3 = (((total / 5) / 2) / 5) / 3)  -- A third of the saxophone players play alto saxophone
  (h5 : (((total / 5) / 2) / 5) / 3 = 4)  -- 4 students play alto saxophone
  : total = 600 := by
  sorry

end NUMINAMATH_CALUDE_senior_class_size_l181_18140


namespace NUMINAMATH_CALUDE_min_valid_subset_l181_18126

def isValid (S : Finset ℕ) : Prop :=
  ∀ n : ℕ, n ≤ 20 → (n ∈ S ∨ ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a + b = n)

theorem min_valid_subset :
  ∃ S : Finset ℕ,
    S ⊆ Finset.range 11 \ {0} ∧
    Finset.card S = 6 ∧
    isValid S ∧
    ∀ T : Finset ℕ, T ⊆ Finset.range 11 \ {0} → Finset.card T < 6 → ¬isValid T :=
  sorry

end NUMINAMATH_CALUDE_min_valid_subset_l181_18126


namespace NUMINAMATH_CALUDE_triangle_area_l181_18110

theorem triangle_area (a b c : ℝ) (B : ℝ) : 
  B = 2 * Real.pi / 3 →
  b = Real.sqrt 13 →
  a + c = 4 →
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l181_18110


namespace NUMINAMATH_CALUDE_basil_sage_ratio_l181_18184

def herb_ratio (basil sage verbena : ℕ) : Prop :=
  sage = verbena - 5 ∧
  basil = 12 ∧
  basil + sage + verbena = 29

theorem basil_sage_ratio :
  ∀ basil sage verbena : ℕ,
    herb_ratio basil sage verbena →
    (basil : ℚ) / sage = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_basil_sage_ratio_l181_18184


namespace NUMINAMATH_CALUDE_fifteenth_clap_theorem_l181_18143

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  circumference : ℝ
  a_lap_time : ℝ
  b_lap_time : ℝ
  a_reverse_laps : ℕ

/-- Calculates the time and distance for A and B to clap hands 15 times -/
def clap_hands_15_times (track : CircularTrack) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct time and distance for the 15th clap -/
theorem fifteenth_clap_theorem (track : CircularTrack) 
  (h1 : track.circumference = 400)
  (h2 : track.a_lap_time = 4)
  (h3 : track.b_lap_time = 7)
  (h4 : track.a_reverse_laps = 10) :
  let (time, distance) := clap_hands_15_times track
  time = 66 + 2/11 ∧ distance = 3781 + 9/11 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_clap_theorem_l181_18143


namespace NUMINAMATH_CALUDE_rhombus_area_l181_18186

/-- The area of a rhombus with sides of length 3 cm and one internal angle of 45 degrees is (9√2)/2 square centimeters. -/
theorem rhombus_area (s : ℝ) (angle : ℝ) (h1 : s = 3) (h2 : angle = 45 * π / 180) :
  let area := s * s * Real.sin angle
  area = 9 * Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l181_18186


namespace NUMINAMATH_CALUDE_missy_watch_time_l181_18163

/-- The total time Missy spends watching TV -/
def total_watch_time (num_reality_shows : ℕ) (reality_show_duration : ℕ) (num_cartoons : ℕ) (cartoon_duration : ℕ) : ℕ :=
  num_reality_shows * reality_show_duration + num_cartoons * cartoon_duration

/-- Theorem stating that Missy spends 150 minutes watching TV -/
theorem missy_watch_time :
  total_watch_time 5 28 1 10 = 150 := by
  sorry

end NUMINAMATH_CALUDE_missy_watch_time_l181_18163


namespace NUMINAMATH_CALUDE_cheaper_store_difference_l181_18125

/-- The list price of Book Y in dollars -/
def list_price : ℚ := 24.95

/-- The discount amount at Readers' Delight in dollars -/
def readers_delight_discount : ℚ := 5

/-- The discount percentage at Book Bargains -/
def book_bargains_discount_percent : ℚ := 20

/-- The sale price at Readers' Delight in dollars -/
def readers_delight_price : ℚ := list_price - readers_delight_discount

/-- The sale price at Book Bargains in dollars -/
def book_bargains_price : ℚ := list_price * (1 - book_bargains_discount_percent / 100)

/-- The price difference in cents -/
def price_difference_cents : ℤ := ⌊(book_bargains_price - readers_delight_price) * 100⌋

theorem cheaper_store_difference :
  price_difference_cents = 1 :=
sorry

end NUMINAMATH_CALUDE_cheaper_store_difference_l181_18125


namespace NUMINAMATH_CALUDE_largest_x_floor_fraction_l181_18190

open Real

theorem largest_x_floor_fraction : 
  (∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) / x = 11 / 12) ∧ 
  (∀ (y : ℝ), y > 0 → (⌊y⌋ : ℝ) / y = 11 / 12 → y ≤ 120 / 11) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_fraction_l181_18190


namespace NUMINAMATH_CALUDE_election_votes_l181_18108

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (difference_percent : ℚ) : 
  total_votes = 9720 →
  invalid_percent = 1/5 →
  difference_percent = 3/20 →
  ∃ (a_votes b_votes : ℕ),
    b_votes = 3159 ∧
    a_votes + b_votes = total_votes * (1 - invalid_percent) ∧
    a_votes = b_votes + total_votes * difference_percent :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l181_18108


namespace NUMINAMATH_CALUDE_xyz_value_l181_18154

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 15 - x * y * z) :
  x * y * z = 15 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l181_18154


namespace NUMINAMATH_CALUDE_trig_identity_l181_18129

theorem trig_identity (x y : ℝ) :
  Real.sin (x + y) * Real.cos (2 * y) + Real.cos (x + y) * Real.sin (2 * y) = Real.sin (x + 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l181_18129


namespace NUMINAMATH_CALUDE_only_sixteen_seventeen_not_divide_l181_18169

/-- A number satisfying the conditions of the problem -/
def special_number (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range 30, k + 2 ∣ n ∨ k + 3 ∣ n

/-- The theorem stating that 16 and 17 are the only consecutive numbers
    that don't divide the special number -/
theorem only_sixteen_seventeen_not_divide (n : ℕ) (h : special_number n) :
    ∃! (a : ℕ), a ∈ Finset.range 30 ∧ ¬(a + 2 ∣ n) ∧ ¬(a + 3 ∣ n) ∧ a = 14 := by
  sorry

#check only_sixteen_seventeen_not_divide

end NUMINAMATH_CALUDE_only_sixteen_seventeen_not_divide_l181_18169


namespace NUMINAMATH_CALUDE_equation_solvable_l181_18160

/-- For a given real number b, this function represents the equation x - b = ∑_{k=0}^∞ x^k -/
def equation (b x : ℝ) : Prop :=
  x - b = (∑' k, x^k)

/-- This theorem states the conditions on b for which the equation has solutions -/
theorem equation_solvable (b : ℝ) : 
  (∃ x : ℝ, equation b x) ↔ (b ≤ -1 ∨ (-3/2 < b ∧ b ≤ -1)) :=
sorry

end NUMINAMATH_CALUDE_equation_solvable_l181_18160


namespace NUMINAMATH_CALUDE_travel_options_l181_18124

theorem travel_options (train_services : ℕ) (ferry_services : ℕ) : 
  train_services = 3 → ferry_services = 2 → train_services * ferry_services = 6 := by
  sorry

#check travel_options

end NUMINAMATH_CALUDE_travel_options_l181_18124


namespace NUMINAMATH_CALUDE_ant_meeting_point_l181_18116

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point on the perimeter of the triangle -/
structure PerimeterPoint where
  distanceFromP : ℝ

/-- The theorem statement -/
theorem ant_meeting_point (t : Triangle) (s : PerimeterPoint) : 
  t.a = 7 ∧ t.b = 8 ∧ t.c = 9 →
  s.distanceFromP = (t.a + t.b + t.c) / 2 →
  s.distanceFromP - t.a = 5 := by
  sorry

end NUMINAMATH_CALUDE_ant_meeting_point_l181_18116


namespace NUMINAMATH_CALUDE_divisibility_by_five_l181_18151

theorem divisibility_by_five (a b : ℕ+) : 
  (5 ∣ (a * b)) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l181_18151


namespace NUMINAMATH_CALUDE_stadium_perimeter_stadium_breadth_l181_18138

/-- Represents a rectangular stadium -/
structure Stadium where
  perimeter : ℝ
  length : ℝ
  breadth : ℝ

/-- The perimeter of a rectangle is twice the sum of its length and breadth -/
theorem stadium_perimeter (s : Stadium) : s.perimeter = 2 * (s.length + s.breadth) := by sorry

/-- Given a stadium with perimeter 800 and length 100, its breadth is 300 -/
theorem stadium_breadth : 
  ∀ (s : Stadium), s.perimeter = 800 ∧ s.length = 100 → s.breadth = 300 := by sorry

end NUMINAMATH_CALUDE_stadium_perimeter_stadium_breadth_l181_18138


namespace NUMINAMATH_CALUDE_extracurricular_materials_selection_l181_18161

theorem extracurricular_materials_selection : 
  let total_materials : ℕ := 6
  let materials_per_student : ℕ := 2
  let common_materials : ℕ := 1
  
  (total_materials.choose common_materials) * 
  ((total_materials - common_materials).choose (materials_per_student - common_materials)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_extracurricular_materials_selection_l181_18161


namespace NUMINAMATH_CALUDE_menu_restriction_l181_18103

theorem menu_restriction (total_dishes : ℕ) (sugar_free_ratio : ℚ) (shellfish_free_ratio : ℚ)
  (h1 : sugar_free_ratio = 1 / 10)
  (h2 : shellfish_free_ratio = 3 / 4) :
  (sugar_free_ratio * shellfish_free_ratio : ℚ) = 3 / 40 := by
  sorry

end NUMINAMATH_CALUDE_menu_restriction_l181_18103


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l181_18195

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x * (x - 1) = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l181_18195


namespace NUMINAMATH_CALUDE_gum_cost_l181_18123

/-- Given that P packs of gum can be purchased for C coins,
    and 1 pack of gum costs 3 coins, prove that X packs of gum cost 3X coins. -/
theorem gum_cost (P C X : ℕ) (h1 : C = 3 * P) (h2 : X > 0) : 3 * X = C * X / P :=
sorry

end NUMINAMATH_CALUDE_gum_cost_l181_18123


namespace NUMINAMATH_CALUDE_divisor_problem_l181_18111

theorem divisor_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 165 →
  quotient = 9 →
  remainder = 3 →
  dividend = divisor * quotient + remainder →
  divisor = 18 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l181_18111


namespace NUMINAMATH_CALUDE_fish_lives_12_years_l181_18196

/-- The lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a dog in years -/
def dog_lifespan : ℝ := 4 * hamster_lifespan

/-- The lifespan of a well-cared fish in years -/
def fish_lifespan : ℝ := dog_lifespan + 2

/-- Theorem stating that the lifespan of a well-cared fish is 12 years -/
theorem fish_lives_12_years : fish_lifespan = 12 := by sorry

end NUMINAMATH_CALUDE_fish_lives_12_years_l181_18196


namespace NUMINAMATH_CALUDE_function_inequality_l181_18168

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_deriv : ∀ x : ℝ, x ≠ 1 → (x - 1) * deriv f x > 0) :
  f 0 + f 2 > 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l181_18168


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l181_18115

/-- The perimeter of a large rectangle composed of nine identical smaller rectangles -/
theorem large_rectangle_perimeter (small_length : ℝ) (h1 : small_length = 10) :
  let large_length := 2 * small_length
  let large_height := 4 * small_length / 5
  let perimeter := 2 * (large_length + large_height)
  perimeter = 76 := by sorry

end NUMINAMATH_CALUDE_large_rectangle_perimeter_l181_18115


namespace NUMINAMATH_CALUDE_system_solution_l181_18167

theorem system_solution : ∃ (x y z : ℤ),
  (5732 * x + 2134 * y + 2134 * z = 7866) ∧
  (2134 * x + 5732 * y + 2134 * z = 670) ∧
  (2134 * x + 2134 * y + 5732 * z = 11464) ∧
  x = 1 ∧ y = -1 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l181_18167


namespace NUMINAMATH_CALUDE_subset_implies_a_value_l181_18157

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 = 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- State the theorem
theorem subset_implies_a_value (a : ℝ) : B a ⊆ A → a ∈ ({0, 1, -1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_value_l181_18157


namespace NUMINAMATH_CALUDE_rectangle_area_lower_bound_l181_18133

theorem rectangle_area_lower_bound 
  (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a * x = 1) 
  (eq2 : c * x = 3) 
  (eq3 : b * y = 10) 
  (eq4 : a * z = 9) : 
  (a + b + c) * (x + y + z) ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_lower_bound_l181_18133


namespace NUMINAMATH_CALUDE_xiao_gao_score_l181_18130

/-- Represents a test score system with a standard score and a recorded score. -/
structure TestScore where
  standard : ℕ
  recorded : ℤ

/-- Calculates the actual score given a TestScore. -/
def actualScore (ts : TestScore) : ℕ :=
  ts.standard + ts.recorded.toNat

/-- Theorem stating that for a standard score of 80 and a recorded score of 12,
    the actual score is 92. -/
theorem xiao_gao_score :
  let ts : TestScore := { standard := 80, recorded := 12 }
  actualScore ts = 92 := by
  sorry

end NUMINAMATH_CALUDE_xiao_gao_score_l181_18130


namespace NUMINAMATH_CALUDE_negative_roots_condition_l181_18176

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - (a + 1) * x + a + 4

-- Define the condition for both roots being negative
def both_roots_negative (a : ℝ) : Prop :=
  ∀ x : ℝ, quadratic a x = 0 → x < 0

-- Theorem statement
theorem negative_roots_condition :
  ∀ a : ℝ, both_roots_negative a ↔ -4 < a ∧ a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_negative_roots_condition_l181_18176


namespace NUMINAMATH_CALUDE_sin_cos_inequality_l181_18101

theorem sin_cos_inequality (t : ℝ) (h : t > 0) : 3 * Real.sin t < 2 * t + t * Real.cos t := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_l181_18101


namespace NUMINAMATH_CALUDE_sequence_formula_l181_18158

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) - a n = r * (a n - a (n - 1))

theorem sequence_formula (a : ℕ → ℝ) :
  geometric_sequence (λ n => a (n + 1) - a n) ∧
  (a 2 - a 1 = 1) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) - a n = (1 / 3) * (a n - a (n - 1))) →
  ∀ n : ℕ, n > 0 → a n = (3 / 2) * (1 - (1 / 3) ^ n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l181_18158


namespace NUMINAMATH_CALUDE_student_selection_count_l181_18144

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_selected : ℕ := 4

def select_students (b g s : ℕ) : ℕ :=
  (Nat.choose b 3 * Nat.choose g 1) +
  (Nat.choose b 2 * Nat.choose g 2) +
  (Nat.choose b 1 * Nat.choose g 3)

theorem student_selection_count :
  select_students num_boys num_girls total_selected = 34 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_count_l181_18144


namespace NUMINAMATH_CALUDE_trigonometric_identity_l181_18114

theorem trigonometric_identity (α : Real) :
  (2 * (Real.cos (2 * α))^2 - 1) / 
  (2 * Real.tan (π/4 - 2*α) * (Real.sin (3*π/4 - 2*α))^2) - 
  Real.tan (2*α) + Real.cos (2*α) - Real.sin (2*α) = 
  (2 * Real.sqrt 2 * Real.sin (π/4 - 2*α) * (Real.cos α)^2) / 
  Real.cos (2*α) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l181_18114


namespace NUMINAMATH_CALUDE_exists_special_six_digit_number_l181_18185

/-- A six-digit number is between 100000 and 999999 -/
def SixDigitNumber (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- The last six digits of a number -/
def LastSixDigits (n : ℕ) : ℕ := n % 1000000

theorem exists_special_six_digit_number :
  ∃ A : ℕ, SixDigitNumber A ∧
    ∀ k m : ℕ, 1 ≤ k → k < m → m ≤ 500000 →
      LastSixDigits (k * A) ≠ LastSixDigits (m * A) := by
  sorry

end NUMINAMATH_CALUDE_exists_special_six_digit_number_l181_18185


namespace NUMINAMATH_CALUDE_fourth_power_nested_root_l181_18172

theorem fourth_power_nested_root : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_nested_root_l181_18172


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l181_18198

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 1/5) : 
  Real.sin (2 * α) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l181_18198


namespace NUMINAMATH_CALUDE_extra_lives_in_first_level_l181_18100

theorem extra_lives_in_first_level :
  let initial_lives : ℕ := 2
  let lives_gained_second_level : ℕ := 11
  let total_lives_after_second_level : ℕ := 19
  let extra_lives_first_level : ℕ := total_lives_after_second_level - lives_gained_second_level - initial_lives
  extra_lives_first_level = 6 :=
by sorry

end NUMINAMATH_CALUDE_extra_lives_in_first_level_l181_18100


namespace NUMINAMATH_CALUDE_f_max_value_f_min_value_f_touches_x_axis_l181_18165

/-- A cubic function that touches the x-axis at (1,0) -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x

/-- The maximum value of f(x) is 4/27 -/
theorem f_max_value : ∃ (x : ℝ), f x = 4/27 ∧ ∀ (y : ℝ), f y ≤ 4/27 :=
sorry

/-- The minimum value of f(x) is 0 -/
theorem f_min_value : ∃ (x : ℝ), f x = 0 ∧ ∀ (y : ℝ), f y ≥ 0 :=
sorry

/-- The function f(x) touches the x-axis at (1,0) -/
theorem f_touches_x_axis : f 1 = 0 ∧ ∀ (x : ℝ), x ≠ 1 → f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_f_max_value_f_min_value_f_touches_x_axis_l181_18165


namespace NUMINAMATH_CALUDE_only_one_solution_l181_18149

def sum_of_squares (K : ℕ) : ℕ := K * (K + 1) * (2 * K + 1) / 6

theorem only_one_solution (K : ℕ) (M : ℕ) :
  sum_of_squares K = M^3 →
  M < 50 →
  K = 1 :=
by sorry

end NUMINAMATH_CALUDE_only_one_solution_l181_18149


namespace NUMINAMATH_CALUDE_computer_from_syllables_l181_18127

/-- Represents a syllable in a word --/
structure Syllable where
  value : String

/-- Represents a word composed of syllables --/
def Word := List Syllable

/-- Function to combine syllables into a word --/
def combineWord (syllables : Word) : String :=
  String.join (syllables.map (λ s => s.value))

/-- Theorem: Given the specific syllables, the resulting word is "компьютер" --/
theorem computer_from_syllables (s1 s2 s3 : Syllable) 
  (h1 : s1.value = "ком")  -- First syllable: A big piece of a snowman
  (h2 : s2.value = "пьют") -- Second syllable: Something done by elephants at watering hole
  (h3 : s3.value = "ер")   -- Third syllable: Called as the hard sign used to be called
  : combineWord [s1, s2, s3] = "компьютер" := by
  sorry

/-- The word has exactly three syllables --/
axiom word_has_three_syllables (w : Word) : w.length = 3

#check computer_from_syllables
#check word_has_three_syllables

end NUMINAMATH_CALUDE_computer_from_syllables_l181_18127


namespace NUMINAMATH_CALUDE_winning_team_fourth_quarter_points_l181_18128

theorem winning_team_fourth_quarter_points :
  ∀ (first_quarter second_quarter third_quarter fourth_quarter : ℕ),
  let losing_team_first_quarter := 10
  let total_points := 80
  first_quarter = 2 * losing_team_first_quarter →
  second_quarter = first_quarter + 10 →
  third_quarter = second_quarter + 20 →
  fourth_quarter = total_points - (first_quarter + second_quarter + third_quarter) →
  fourth_quarter = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_winning_team_fourth_quarter_points_l181_18128


namespace NUMINAMATH_CALUDE_f_composition_nine_equals_one_eighth_l181_18171

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -Real.sqrt x else 2^x

theorem f_composition_nine_equals_one_eighth :
  f (f 9) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_nine_equals_one_eighth_l181_18171


namespace NUMINAMATH_CALUDE_max_value_quadratic_l181_18109

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 4) :
  ∃ (y : ℝ), y = x * (8 - 2 * x) ∧ ∀ (z : ℝ), z = x * (8 - 2 * x) → z ≤ y ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l181_18109


namespace NUMINAMATH_CALUDE_count_multiples_count_multiples_equals_1002_l181_18112

theorem count_multiples : ℕ :=
  let range_start := 1
  let range_end := 2005
  let count_multiples_of_3 := (range_end / 3 : ℕ)
  let count_multiples_of_4 := (range_end / 4 : ℕ)
  let count_multiples_of_12 := (range_end / 12 : ℕ)
  count_multiples_of_3 + count_multiples_of_4 - count_multiples_of_12

theorem count_multiples_equals_1002 : count_multiples = 1002 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_count_multiples_equals_1002_l181_18112


namespace NUMINAMATH_CALUDE_potato_bag_weight_l181_18105

theorem potato_bag_weight (bag_weight : ℝ) (h : bag_weight = 36) :
  bag_weight / (bag_weight / 2) = 2 ∧ bag_weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l181_18105


namespace NUMINAMATH_CALUDE_no_two_right_angles_l181_18164

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_angles : a + b + c = 180

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop := angle = 90

-- Theorem statement
theorem no_two_right_angles (t : Triangle) : 
  ¬(is_right_angle t.a ∧ is_right_angle t.b) ∧ 
  ¬(is_right_angle t.b ∧ is_right_angle t.c) ∧ 
  ¬(is_right_angle t.c ∧ is_right_angle t.a) :=
sorry

end NUMINAMATH_CALUDE_no_two_right_angles_l181_18164


namespace NUMINAMATH_CALUDE_complement_A_in_U_l181_18162

-- Define the set U
def U : Set ℝ := {y | ∃ x : ℝ, y = 2^x ∧ x ≥ -1}

-- Define the set A
def A : Set ℝ := {x | (x - 2) * (x - 1) < 0}

-- State the theorem
theorem complement_A_in_U : 
  (U \ A) = {x | x ∈ Set.Icc (1/2 : ℝ) 1 ∨ x ∈ Set.Ici 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l181_18162


namespace NUMINAMATH_CALUDE_max_roses_325_l181_18135

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual_price : ℚ
  dozen_price : ℚ
  two_dozen_price : ℚ

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def max_roses (budget : ℚ) (pricing : RosePricing) : ℕ :=
  sorry

/-- The theorem stating that given the specific pricing and budget, 325 roses is the maximum that can be purchased -/
theorem max_roses_325 :
  let pricing : RosePricing := {
    individual_price := 23/10,
    dozen_price := 36,
    two_dozen_price := 50
  }
  max_roses 680 pricing = 325 := by sorry

end NUMINAMATH_CALUDE_max_roses_325_l181_18135


namespace NUMINAMATH_CALUDE_lexie_picked_12_apples_l181_18178

/-- The number of apples Lexie and Tom picked together -/
def total_apples : ℕ := 36

/-- Lexie's apples -/
def lexie_apples : ℕ := 12

/-- Tom's apples -/
def tom_apples : ℕ := 2 * lexie_apples

/-- Theorem stating that Lexie picked 12 apples given the conditions -/
theorem lexie_picked_12_apples : 
  (tom_apples = 2 * lexie_apples) ∧ (lexie_apples + tom_apples = total_apples) → 
  lexie_apples = 12 := by
  sorry

#check lexie_picked_12_apples

end NUMINAMATH_CALUDE_lexie_picked_12_apples_l181_18178


namespace NUMINAMATH_CALUDE_max_polygon_length_8x8_grid_l181_18199

/-- Represents a square grid -/
structure SquareGrid where
  size : ℕ

/-- Represents a polygon on a grid -/
structure GridPolygon where
  grid : SquareGrid
  length : ℕ
  closed : Bool
  self_avoiding : Bool

/-- The maximum length of a closed self-avoiding polygon on an 8x8 grid is 80 -/
theorem max_polygon_length_8x8_grid :
  ∃ (p : GridPolygon), p.grid.size = 8 ∧ p.closed ∧ p.self_avoiding ∧
    p.length = 80 ∧
    ∀ (q : GridPolygon), q.grid.size = 8 → q.closed → q.self_avoiding →
      q.length ≤ p.length := by
  sorry

end NUMINAMATH_CALUDE_max_polygon_length_8x8_grid_l181_18199


namespace NUMINAMATH_CALUDE_rahims_average_book_price_l181_18156

/-- Calculates the average price per book given two separate book purchases -/
def average_price_per_book (books1 : ℕ) (price1 : ℕ) (books2 : ℕ) (price2 : ℕ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating that the average price per book for Rahim's purchases is 20 -/
theorem rahims_average_book_price :
  average_price_per_book 50 1000 40 800 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rahims_average_book_price_l181_18156


namespace NUMINAMATH_CALUDE_class_average_after_exclusion_l181_18152

theorem class_average_after_exclusion (total_students : ℕ) (initial_avg : ℚ) 
  (excluded_students : ℕ) (excluded_avg : ℚ) :
  total_students = 30 →
  initial_avg = 80 →
  excluded_students = 5 →
  excluded_avg = 30 →
  let remaining_students := total_students - excluded_students
  let total_marks := initial_avg * total_students
  let excluded_marks := excluded_avg * excluded_students
  let remaining_marks := total_marks - excluded_marks
  (remaining_marks / remaining_students) = 90 := by
  sorry

end NUMINAMATH_CALUDE_class_average_after_exclusion_l181_18152


namespace NUMINAMATH_CALUDE_min_stamps_for_35_cents_l181_18132

/-- Represents the number of ways to make a certain amount of cents using 5-cent and 7-cent stamps -/
def stamp_combinations (cents : ℕ) : Set (ℕ × ℕ) :=
  {(x, y) | 5 * x + 7 * y = cents}

/-- The total number of stamps used in a combination -/
def total_stamps (combo : ℕ × ℕ) : ℕ :=
  combo.1 + combo.2

theorem min_stamps_for_35_cents :
  ∃ (combo : ℕ × ℕ),
    combo ∈ stamp_combinations 35 ∧
    ∀ (other : ℕ × ℕ), other ∈ stamp_combinations 35 →
      total_stamps combo ≤ total_stamps other ∧
      total_stamps combo = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_stamps_for_35_cents_l181_18132


namespace NUMINAMATH_CALUDE_rotation_sum_l181_18155

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Represents a rotation transformation -/
structure Rotation where
  angle : ℝ
  center : Point

/-- Checks if a rotation transforms one triangle to another -/
def rotates (r : Rotation) (t1 t2 : Triangle) : Prop := sorry

theorem rotation_sum (t1 t2 : Triangle) (r : Rotation) :
  t1.p1 = Point.mk 2 2 ∧
  t1.p2 = Point.mk 2 14 ∧
  t1.p3 = Point.mk 18 2 ∧
  t2.p1 = Point.mk 32 26 ∧
  t2.p2 = Point.mk 44 26 ∧
  t2.p3 = Point.mk 32 10 ∧
  rotates r t1 t2 ∧
  0 < r.angle ∧ r.angle < 180 →
  r.angle + r.center.x + r.center.y = 124 := by
  sorry

end NUMINAMATH_CALUDE_rotation_sum_l181_18155


namespace NUMINAMATH_CALUDE_max_travel_distance_is_3_4_l181_18121

/-- Represents the taxi fare structure and travel constraints -/
structure TaxiRide where
  initialFare : ℝ
  initialDistance : ℝ
  additionalFarePerUnit : ℝ
  additionalDistanceUnit : ℝ
  tip : ℝ
  totalBudget : ℝ
  timeLimit : ℝ
  averageSpeed : ℝ

/-- Calculates the maximum distance that can be traveled given the taxi fare structure and constraints -/
def maxTravelDistance (ride : TaxiRide) : ℝ :=
  sorry

/-- Theorem stating that the maximum travel distance is approximately 3.4 miles -/
theorem max_travel_distance_is_3_4 (ride : TaxiRide) 
  (h1 : ride.initialFare = 4)
  (h2 : ride.initialDistance = 3/4)
  (h3 : ride.additionalFarePerUnit = 0.3)
  (h4 : ride.additionalDistanceUnit = 0.1)
  (h5 : ride.tip = 3)
  (h6 : ride.totalBudget = 15)
  (h7 : ride.timeLimit = 45/60)
  (h8 : ride.averageSpeed = 30) :
  ∃ ε > 0, abs (maxTravelDistance ride - 3.4) < ε :=
sorry

end NUMINAMATH_CALUDE_max_travel_distance_is_3_4_l181_18121


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l181_18117

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x^2 - x - 2 = 0 → -1 ≤ x ∧ x ≤ 2) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ x^2 - x - 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l181_18117


namespace NUMINAMATH_CALUDE_vector_perpendicular_to_line_l181_18145

/-- Given a vector a and a line l, prove that they are perpendicular -/
theorem vector_perpendicular_to_line (a : ℝ × ℝ) (l : ℝ → ℝ → Prop) : 
  a = (2, 3) → 
  (∀ x y, l x y ↔ 2 * x + 3 * y - 1 = 0) → 
  ∃ k, k * a.1 + a.2 = 0 ∧ k * 2 - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_vector_perpendicular_to_line_l181_18145


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l181_18182

theorem real_part_of_complex_fraction : 
  Complex.re (5 / (1 - Complex.I * 2)) = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l181_18182


namespace NUMINAMATH_CALUDE_zach_ticket_purchase_l181_18146

/-- The number of tickets Zach needs to buy for both rides -/
def tickets_needed (ferris_wheel_cost roller_coaster_cost multiple_ride_discount coupon : ℝ) : ℝ :=
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - coupon

/-- Theorem stating the number of tickets Zach needs to buy -/
theorem zach_ticket_purchase :
  tickets_needed 2.0 7.0 1.0 1.0 = 7.0 := by
  sorry

#eval tickets_needed 2.0 7.0 1.0 1.0

end NUMINAMATH_CALUDE_zach_ticket_purchase_l181_18146


namespace NUMINAMATH_CALUDE_symmetric_distribution_property_l181_18136

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  /-- The mean of the distribution -/
  mean : ℝ
  /-- The standard deviation of the distribution -/
  std_dev : ℝ
  /-- The cumulative distribution function -/
  cdf : ℝ → ℝ
  /-- The distribution is symmetric about the mean -/
  symmetric : ∀ x, cdf (mean - x) + cdf (mean + x) = 1
  /-- 68% of the distribution lies within one standard deviation of the mean -/
  std_dev_property : cdf (mean + std_dev) - cdf (mean - std_dev) = 0.68

/-- 
Theorem: In a symmetric distribution about the mean m, where 68% of the distribution 
lies within one standard deviation h of the mean, the percentage of the distribution 
less than m + h is 84%.
-/
theorem symmetric_distribution_property (d : SymmetricDistribution) : 
  d.cdf (d.mean + d.std_dev) = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_distribution_property_l181_18136


namespace NUMINAMATH_CALUDE_rabbit_catches_cat_l181_18104

/-- Proves that a rabbit catches up to a cat in 1 hour given their speeds and the cat's head start -/
theorem rabbit_catches_cat (rabbit_speed cat_speed : ℝ) (head_start : ℝ) : 
  rabbit_speed = 25 →
  cat_speed = 20 →
  head_start = 0.25 →
  (rabbit_speed - cat_speed) * 1 = cat_speed * head_start := by
  sorry

#check rabbit_catches_cat

end NUMINAMATH_CALUDE_rabbit_catches_cat_l181_18104


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l181_18120

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The theorem statement -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = 3)
  (h2 : seq.a 1 + seq.a 2 + seq.a 3 = 21) :
  seq.a 4 + seq.a 5 + seq.a 6 = 57 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l181_18120


namespace NUMINAMATH_CALUDE_inequality_proof_l181_18141

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^4 + y^2*z^2) / (x^(5/2)*(y+z)) + (y^4 + z^2*x^2) / (y^(5/2)*(z+x)) +
  (z^4 + x^2*y^2) / (z^(5/2)*(x+y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l181_18141


namespace NUMINAMATH_CALUDE_probability_two_green_marbles_l181_18153

/-- The probability of drawing two green marbles without replacement from a jar -/
theorem probability_two_green_marbles (red green white blue : ℕ) 
  (h_red : red = 3)
  (h_green : green = 4)
  (h_white : white = 10)
  (h_blue : blue = 5) :
  let total := red + green + white + blue
  let prob_first := green / total
  let prob_second := (green - 1) / (total - 1)
  prob_first * prob_second = 2 / 77 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_green_marbles_l181_18153


namespace NUMINAMATH_CALUDE_line_equation_theorem_l181_18118

-- Define the line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def has_slope (l : Line) (m : ℝ) : Prop :=
  l.a ≠ 0 ∧ -l.b / l.a = m

def triangle_area (l : Line) (area : ℝ) : Prop :=
  l.c ≠ 0 ∧ abs (l.c / l.a) * abs (l.c / l.b) / 2 = area

def passes_through (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

def equal_absolute_intercepts (l : Line) : Prop :=
  abs (l.c / l.a) = abs (l.c / l.b)

-- Define the theorem
theorem line_equation_theorem (l : Line) :
  has_slope l (3/4) ∧
  triangle_area l 6 ∧
  passes_through l 4 (-3) ∧
  equal_absolute_intercepts l →
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -1) ∨
  (l.a = 1 ∧ l.b = -1 ∧ l.c = 7) ∨
  (l.a = 3 ∧ l.b = 4 ∧ l.c = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_theorem_l181_18118


namespace NUMINAMATH_CALUDE_hyperbola_equation_l181_18191

/-- Given a parabola and a hyperbola with specific properties, 
    prove that the standard equation of the hyperbola is x² - y²/2 = 1 -/
theorem hyperbola_equation 
  (parabola : ℝ → ℝ → Prop) 
  (hyperbola : ℝ → ℝ → ℝ → ℝ → Prop)
  (a b : ℝ)
  (A B F : ℝ × ℝ)
  (h_parabola : ∀ x y, parabola x y ↔ y^2 = 4 * Real.sqrt 3 * x)
  (h_hyperbola : ∀ x y, hyperbola a b x y ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_intersect : parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
                 hyperbola a b A.1 A.2 ∧ hyperbola a b B.1 B.2)
  (h_A_above_B : A.2 > B.2)
  (h_asymptote : ∀ x, b * x / a = Real.sqrt 2 * x)
  (h_F_focus : F = (Real.sqrt 3, 0))
  (h_equilateral : 
    (A.1 - F.1)^2 + (A.2 - F.2)^2 = 
    (B.1 - F.1)^2 + (B.2 - F.2)^2 ∧
    (A.1 - F.1)^2 + (A.2 - F.2)^2 = 
    (A.1 - B.1)^2 + (A.2 - B.2)^2) :
  ∀ x y, hyperbola a b x y ↔ x^2 - y^2 / 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l181_18191


namespace NUMINAMATH_CALUDE_bottles_lasted_74_days_l181_18134

/-- The number of bottles Debby bought -/
def total_bottles : ℕ := 8066

/-- The number of bottles Debby drank per day -/
def bottles_per_day : ℕ := 109

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := total_bottles / bottles_per_day

theorem bottles_lasted_74_days : days_lasted = 74 := by
  sorry

end NUMINAMATH_CALUDE_bottles_lasted_74_days_l181_18134


namespace NUMINAMATH_CALUDE_triangle_median_altitude_equations_l181_18119

/-- Triangle ABC in the Cartesian coordinate system -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of the median from a vertex to the opposite side -/
def median (t : Triangle) (v : ℝ × ℝ) : Line := sorry

/-- Definition of the altitude from a vertex to the opposite side -/
def altitude (t : Triangle) (v : ℝ × ℝ) : Line := sorry

theorem triangle_median_altitude_equations :
  let t : Triangle := { A := (7, 8), B := (10, 4), C := (2, -4) }
  (median t t.B = { a := 8, b := -1, c := -48 }) ∧
  (altitude t t.B = { a := 1, b := 1, c := -15 }) := by sorry

end NUMINAMATH_CALUDE_triangle_median_altitude_equations_l181_18119


namespace NUMINAMATH_CALUDE_white_area_calculation_l181_18193

/-- Given a rectangular grid with the following properties:
  - The total area of the rectangle is 32 cm².
  - The sum of two grey areas is 31 cm².
  - The dark grey area counted twice in the sum is 4 cm².
  Prove that the white area is 5 cm². -/
theorem white_area_calculation (total_area : ℝ) (sum_grey_areas : ℝ) (dark_grey_area : ℝ) 
  (h1 : total_area = 32)
  (h2 : sum_grey_areas = 31)
  (h3 : dark_grey_area = 4) :
  total_area - (sum_grey_areas - dark_grey_area) = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_area_calculation_l181_18193


namespace NUMINAMATH_CALUDE_product_equality_l181_18175

theorem product_equality : 500 * 2019 * 0.02019 * 5 = 0.25 * 2019^2 := by sorry

end NUMINAMATH_CALUDE_product_equality_l181_18175


namespace NUMINAMATH_CALUDE_inequality_solution_l181_18187

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the inequality function
def inequality (a : ℝ) (x : ℝ) : Prop :=
  log a (x^2 - x - 2) > log a (x - 2/a) + 1

-- Theorem statement
theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → ∀ x, inequality a x ↔ x > 1 + a) ∧
  (0 < a ∧ a < 1 → ¬∃ x, inequality a x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l181_18187


namespace NUMINAMATH_CALUDE_paige_finished_problems_l181_18188

/-- Given that Paige had 43 math problems, 12 science problems, and 11 problems left to do for homework,
    prove that she finished 44 problems at school. -/
theorem paige_finished_problems (math_problems : ℕ) (science_problems : ℕ) (problems_left : ℕ)
  (h1 : math_problems = 43)
  (h2 : science_problems = 12)
  (h3 : problems_left = 11) :
  math_problems + science_problems - problems_left = 44 := by
  sorry

end NUMINAMATH_CALUDE_paige_finished_problems_l181_18188


namespace NUMINAMATH_CALUDE_stratified_sampling_class_c_l181_18181

theorem stratified_sampling_class_c (total_students : ℕ) (class_a class_b class_c class_d sample_size : ℕ) : 
  total_students = class_a + class_b + class_c + class_d →
  class_a = 75 →
  class_b = 75 →
  class_c = 200 →
  class_d = 150 →
  sample_size = 20 →
  (class_c * sample_size) / total_students = 8 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_class_c_l181_18181


namespace NUMINAMATH_CALUDE_cello_viola_pairs_l181_18180

/-- The number of cellos in stock -/
def num_cellos : ℕ := 800

/-- The number of violas in stock -/
def num_violas : ℕ := 600

/-- The probability of randomly choosing a cello and a viola made from the same tree -/
def same_tree_prob : ℚ := 1 / 4800

/-- The number of cello-viola pairs made with wood from the same tree -/
def num_pairs : ℕ := 100

theorem cello_viola_pairs :
  num_pairs = (same_tree_prob * (num_cellos * num_violas : ℚ)).num := by
  sorry

end NUMINAMATH_CALUDE_cello_viola_pairs_l181_18180


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l181_18183

/-- Represents the percentage of defective units shipped from each stage -/
structure DefectiveShipped :=
  (stage1 : ℝ)
  (stage2 : ℝ)
  (stage3 : ℝ)

/-- Represents the percentage of defective units in each stage -/
structure DefectivePercentage :=
  (stage1 : ℝ)
  (stage2 : ℝ)
  (stage3 : ℝ)

/-- Represents the percentage of defective units shipped from each stage -/
structure ShippedPercentage :=
  (stage1 : ℝ)
  (stage2 : ℝ)
  (stage3 : ℝ)

/-- Calculates the percentage of total units that are defective and shipped -/
def calculate_defective_shipped (dp : DefectivePercentage) (sp : ShippedPercentage) : ℝ :=
  let ds : DefectiveShipped := {
    stage1 := dp.stage1 * sp.stage1,
    stage2 := (1 - dp.stage1) * dp.stage2 * sp.stage2,
    stage3 := (1 - dp.stage1) * (1 - dp.stage2) * dp.stage3 * sp.stage3
  }
  ds.stage1 + ds.stage2 + ds.stage3

/-- Theorem: Given the production process conditions, 2% of total units are defective and shipped -/
theorem defective_shipped_percentage :
  let dp : DefectivePercentage := { stage1 := 0.06, stage2 := 0.08, stage3 := 0.10 }
  let sp : ShippedPercentage := { stage1 := 0.05, stage2 := 0.07, stage3 := 0.10 }
  calculate_defective_shipped dp sp = 0.02 := by
  sorry


end NUMINAMATH_CALUDE_defective_shipped_percentage_l181_18183
