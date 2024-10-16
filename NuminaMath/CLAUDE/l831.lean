import Mathlib

namespace NUMINAMATH_CALUDE_scientific_notation_4040000_l831_83159

theorem scientific_notation_4040000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 4040000 = a * (10 : ℝ) ^ n ∧ a = 4.04 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_4040000_l831_83159


namespace NUMINAMATH_CALUDE_fourth_row_sum_spiral_l831_83176

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the spiral filling of the grid -/
def spiralFill (n : ℕ) : List (Position × ℕ) := sorry

/-- The sum of the smallest and largest numbers in a given row -/
def sumMinMaxInRow (row : ℕ) (filled : List (Position × ℕ)) : ℕ := sorry

theorem fourth_row_sum_spiral (n : ℕ) (h : n = 21) :
  let filled := spiralFill n
  sumMinMaxInRow 4 filled = 742 := by sorry

end NUMINAMATH_CALUDE_fourth_row_sum_spiral_l831_83176


namespace NUMINAMATH_CALUDE_small_rhombus_area_l831_83157

theorem small_rhombus_area (r : ℝ) (h : r = 10) : 
  let large_rhombus_diagonal := 2 * r
  let small_rhombus_side := large_rhombus_diagonal / 2
  small_rhombus_side ^ 2 = 100 := by sorry

end NUMINAMATH_CALUDE_small_rhombus_area_l831_83157


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l831_83121

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube -/
def surfaceArea (c : CubeDimensions) : ℝ :=
  6 * c.length * c.width

/-- Represents the original cube -/
def originalCube : CubeDimensions :=
  { length := 4, width := 4, height := 4 }

/-- Represents the corner cube to be removed -/
def cornerCube : CubeDimensions :=
  { length := 2, width := 2, height := 2 }

/-- The number of corners in a cube -/
def numCorners : ℕ := 8

theorem surface_area_unchanged :
  surfaceArea originalCube = surfaceArea originalCube := by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l831_83121


namespace NUMINAMATH_CALUDE_committee_probability_l831_83192

/-- The number of members in the Grammar club -/
def total_members : ℕ := 20

/-- The number of boys in the Grammar club -/
def num_boys : ℕ := 10

/-- The number of girls in the Grammar club -/
def num_girls : ℕ := 10

/-- The size of the committee to be chosen -/
def committee_size : ℕ := 4

/-- The probability of selecting a committee with at least one boy and one girl -/
theorem committee_probability : 
  (Nat.choose total_members committee_size - 
   (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size)) / 
   Nat.choose total_members committee_size = 295 / 323 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l831_83192


namespace NUMINAMATH_CALUDE_ball_attendees_l831_83190

theorem ball_attendees :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendees_l831_83190


namespace NUMINAMATH_CALUDE_fraction_power_equality_l831_83144

theorem fraction_power_equality : (72000 ^ 4) / (24000 ^ 4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l831_83144


namespace NUMINAMATH_CALUDE_strategy_game_cost_l831_83185

def total_spent : ℝ := 35.52
def football_cost : ℝ := 14.02
def batman_cost : ℝ := 12.04

theorem strategy_game_cost :
  total_spent - football_cost - batman_cost = 9.46 := by
  sorry

end NUMINAMATH_CALUDE_strategy_game_cost_l831_83185


namespace NUMINAMATH_CALUDE_skipping_competition_probability_l831_83105

theorem skipping_competition_probability :
  let total_boys : ℕ := 4
  let total_girls : ℕ := 6
  let selected_boys : ℕ := 2
  let selected_girls : ℕ := 2
  let total_selections : ℕ := (Nat.choose total_boys selected_boys) * (Nat.choose total_girls selected_girls)
  let selections_without_A_and_B : ℕ := (Nat.choose (total_boys - 1) selected_boys) * (Nat.choose (total_girls - 1) selected_girls)
  (total_selections - selections_without_A_and_B) / total_selections = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_skipping_competition_probability_l831_83105


namespace NUMINAMATH_CALUDE_restaurant_at_park_office_l831_83106

/-- Represents the time in minutes for various parts of Dante's journey -/
structure JourneyTimes where
  toHiddenLake : ℕ
  fromHiddenLake : ℕ
  toRestaurant : ℕ

/-- The actual journey times given in the problem -/
def actualJourney : JourneyTimes where
  toHiddenLake := 15
  fromHiddenLake := 7
  toRestaurant := 0

/-- Calculates the total time for a journey without visiting the restaurant -/
def totalTimeWithoutRestaurant (j : JourneyTimes) : ℕ :=
  j.toHiddenLake + j.fromHiddenLake

/-- Calculates the total time for a journey with visiting the restaurant -/
def totalTimeWithRestaurant (j : JourneyTimes) : ℕ :=
  j.toRestaurant + j.toRestaurant + j.toHiddenLake + j.fromHiddenLake

/-- Theorem stating that the time to the restaurant is 0 given the journey times are equal -/
theorem restaurant_at_park_office (j : JourneyTimes) 
  (h : totalTimeWithoutRestaurant j = totalTimeWithRestaurant j) : 
  j.toRestaurant = 0 := by
  sorry

#check restaurant_at_park_office

end NUMINAMATH_CALUDE_restaurant_at_park_office_l831_83106


namespace NUMINAMATH_CALUDE_frustum_volume_l831_83128

/-- The volume of a frustum formed by cutting a square pyramid --/
theorem frustum_volume (base_edge : ℝ) (altitude : ℝ) (small_base_edge : ℝ) (small_altitude : ℝ)
  (h1 : base_edge = 10)
  (h2 : altitude = 10)
  (h3 : small_base_edge = 5)
  (h4 : small_altitude = 5) :
  (base_edge ^ 2 * altitude / 3) - (small_base_edge ^ 2 * small_altitude / 3) = 875 / 3 := by
  sorry

end NUMINAMATH_CALUDE_frustum_volume_l831_83128


namespace NUMINAMATH_CALUDE_a_upper_bound_l831_83199

def f (x : ℝ) := x + x^3

theorem a_upper_bound
  (h : ∀ θ : ℝ, 0 < θ → θ < π/2 → ∀ a : ℝ, f (a * Real.sin θ) + f (1 - a) > 0) :
  ∀ a : ℝ, a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_a_upper_bound_l831_83199


namespace NUMINAMATH_CALUDE_smallest_number_l831_83187

theorem smallest_number (jungkook yoongi yuna : ℕ) : 
  jungkook = 6 - 3 → yoongi = 4 → yuna = 5 → min jungkook (min yoongi yuna) = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_l831_83187


namespace NUMINAMATH_CALUDE_major_axis_length_is_8_l831_83120

/-- The length of the major axis of an ellipse formed by intersecting a plane with a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The length of the major axis is 8 when a plane intersects a right circular cylinder with radius 2, forming an ellipse where the major axis is double the minor axis -/
theorem major_axis_length_is_8 :
  major_axis_length 2 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_is_8_l831_83120


namespace NUMINAMATH_CALUDE_largest_angle_after_change_l831_83164

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ)

-- Define the initial conditions
def initial_triangle : Triangle :=
  { D := 60, E := 60, F := 60 }

-- Define the angle decrease
def angle_decrease : ℝ := 20

-- Theorem statement
theorem largest_angle_after_change (t : Triangle) :
  t = initial_triangle →
  ∃ (new_t : Triangle),
    new_t.D = t.D - angle_decrease ∧
    new_t.D + new_t.E + new_t.F = 180 ∧
    new_t.E = new_t.F ∧
    max new_t.D (max new_t.E new_t.F) = 70 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_after_change_l831_83164


namespace NUMINAMATH_CALUDE_sum_of_ones_and_twos_2020_l831_83165

theorem sum_of_ones_and_twos_2020 :
  (Finset.filter (fun p : ℕ × ℕ => 4 * p.1 + 5 * p.2 = 2020) (Finset.product (Finset.range 505) (Finset.range 404))).card = 102 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ones_and_twos_2020_l831_83165


namespace NUMINAMATH_CALUDE_janet_investment_l831_83126

/-- Calculates the total investment amount given the conditions of Janet's investment -/
theorem janet_investment
  (rate1 rate2 : ℚ)
  (interest_total : ℚ)
  (investment_at_rate1 : ℚ)
  (h1 : rate1 = 1/10)
  (h2 : rate2 = 1/100)
  (h3 : interest_total = 1390)
  (h4 : investment_at_rate1 = 12000)
  (h5 : investment_at_rate1 * rate1 + (total - investment_at_rate1) * rate2 = interest_total) :
  ∃ (total : ℚ), total = 31000 := by
  sorry

end NUMINAMATH_CALUDE_janet_investment_l831_83126


namespace NUMINAMATH_CALUDE_position_of_negative_three_l831_83169

theorem position_of_negative_three : 
  ∀ (x : ℝ), (x = 1 - 4) → (x = -3) :=
by
  sorry

end NUMINAMATH_CALUDE_position_of_negative_three_l831_83169


namespace NUMINAMATH_CALUDE_two_solution_range_l831_83140

/-- 
Given a system of equations:
  y = x^2
  y = x + m
The range of m for which the system has two distinct solutions is (-1/4, +∞).
-/
theorem two_solution_range (x y m : ℝ) : 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 = x₁ + m ∧ x₂^2 = x₂ + m) ↔ m > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_two_solution_range_l831_83140


namespace NUMINAMATH_CALUDE_f_composition_negative_three_l831_83156

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / (5 - x) else Real.log x / Real.log 4

theorem f_composition_negative_three (f : ℝ → ℝ) :
  (∀ x ≤ 0, f x = 1 / (5 - x)) →
  (∀ x > 0, f x = Real.log x / Real.log 4) →
  f (f (-3)) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_three_l831_83156


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l831_83173

theorem domain_of_composite_function 
  (f : ℝ → ℝ) 
  (h : ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (2 * k * Real.pi - Real.pi / 6) (2 * k * Real.pi + 2 * Real.pi / 3) → f (Real.cos x) ∈ Set.range f) :
  Set.range f = Set.Icc (-1/2) 1 := by
sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l831_83173


namespace NUMINAMATH_CALUDE_bucket_capacity_problem_l831_83116

theorem bucket_capacity_problem (capacity : ℝ) : 
  (24 * capacity = 36 * 9) → capacity = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_problem_l831_83116


namespace NUMINAMATH_CALUDE_volunteer_schedule_lcm_l831_83129

theorem volunteer_schedule_lcm : Nat.lcm 5 (Nat.lcm 3 (Nat.lcm 9 8)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_schedule_lcm_l831_83129


namespace NUMINAMATH_CALUDE_no_a_in_either_subject_l831_83170

theorem no_a_in_either_subject (total_students : ℕ) (a_in_chemistry : ℕ) (a_in_physics : ℕ) (a_in_both : ℕ) :
  total_students = 40 →
  a_in_chemistry = 10 →
  a_in_physics = 18 →
  a_in_both = 6 →
  total_students - (a_in_chemistry + a_in_physics - a_in_both) = 18 :=
by sorry

end NUMINAMATH_CALUDE_no_a_in_either_subject_l831_83170


namespace NUMINAMATH_CALUDE_giants_playoff_wins_l831_83134

theorem giants_playoff_wins (total_games : ℕ) (games_to_win : ℕ) (more_wins_needed : ℕ) : 
  total_games = 30 →
  games_to_win = (2 * total_games) / 3 →
  more_wins_needed = 8 →
  games_to_win - more_wins_needed = 12 :=
by sorry

end NUMINAMATH_CALUDE_giants_playoff_wins_l831_83134


namespace NUMINAMATH_CALUDE_certain_number_proof_l831_83152

theorem certain_number_proof (p q x : ℝ) 
  (h1 : 3 / p = x)
  (h2 : 3 / q = 18)
  (h3 : p - q = 0.20833333333333334) :
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l831_83152


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l831_83138

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l831_83138


namespace NUMINAMATH_CALUDE_equation_solution_l831_83198

theorem equation_solution (c d : ℝ) (h : d ≠ 0) :
  let x := (9 * d^2 - 4 * c^2) / (6 * d)
  x^2 + 4 * c^2 = (3 * d - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l831_83198


namespace NUMINAMATH_CALUDE_cube_preserves_order_l831_83145

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l831_83145


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l831_83196

theorem arithmetic_geometric_inequality (n : ℕ) (a b : ℕ → ℝ) 
  (h1 : a 1 = b 1) 
  (h2 : a 1 > 0)
  (h3 : a (2*n+1) = b (2*n+1))
  (h4 : ∀ k, a (k+1) - a k = a 2 - a 1)  -- arithmetic sequence condition
  (h5 : ∀ k, b (k+1) / b k = b 2 / b 1)  -- geometric sequence condition
  : a (n+1) ≥ b (n+1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l831_83196


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l831_83175

theorem unique_solution_inequality (x : ℝ) :
  (x > 0 ∧ x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16) ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l831_83175


namespace NUMINAMATH_CALUDE_dogs_combined_weight_l831_83102

/-- The combined weight of two dogs given the weight of one and its ratio to the other -/
theorem dogs_combined_weight (evans_dog_weight : ℕ) (weight_ratio : ℕ) : 
  evans_dog_weight = 63 → weight_ratio = 7 → 
  evans_dog_weight + evans_dog_weight / weight_ratio = 72 := by
  sorry

#check dogs_combined_weight

end NUMINAMATH_CALUDE_dogs_combined_weight_l831_83102


namespace NUMINAMATH_CALUDE_matrix_power_2018_l831_83184

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 1, 1]

theorem matrix_power_2018 :
  A ^ 2018 = !![2^2017, 2^2017; 2^2017, 2^2017] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2018_l831_83184


namespace NUMINAMATH_CALUDE_celine_book_days_l831_83117

/-- The number of days in May -/
def days_in_may : ℕ := 31

/-- The daily charge for borrowing a book (in dollars) -/
def daily_charge : ℚ := 1/2

/-- The total amount Celine paid (in dollars) -/
def total_paid : ℚ := 41

/-- The number of books Celine borrowed -/
def num_books : ℕ := 3

theorem celine_book_days :
  ∃ (x : ℕ), 
    daily_charge * x + daily_charge * (num_books - 1) * days_in_may = total_paid ∧
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_celine_book_days_l831_83117


namespace NUMINAMATH_CALUDE_min_value_implies_b_range_l831_83162

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 6*b*x + 3*b

-- Define the derivative of f
def f' (b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*b

-- State the theorem
theorem min_value_implies_b_range (b : ℝ) :
  (∃ x ∈ (Set.Ioo 0 1), ∀ y ∈ (Set.Ioo 0 1), f b x ≤ f b y) →
  b ∈ (Set.Ioo 0 (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_implies_b_range_l831_83162


namespace NUMINAMATH_CALUDE_q_of_q_of_q_2000_pow_2000_l831_83178

/-- Sum of digits of a natural number -/
def q (n : ℕ) : ℕ := sorry

/-- Theorem stating that q(q(q(2000^2000))) = 4 -/
theorem q_of_q_of_q_2000_pow_2000 : q (q (q (2000^2000))) = 4 := by sorry

end NUMINAMATH_CALUDE_q_of_q_of_q_2000_pow_2000_l831_83178


namespace NUMINAMATH_CALUDE_oil_distribution_l831_83123

theorem oil_distribution (a b c : ℝ) : 
  c = 48 →
  (2/3 * a = 4/5 * (b + 1/3 * a)) →
  (2/3 * a = 48 + 1/5 * (b + 1/3 * a)) →
  a = 96 ∧ b = 48 := by
sorry

end NUMINAMATH_CALUDE_oil_distribution_l831_83123


namespace NUMINAMATH_CALUDE_parking_lot_useable_percentage_l831_83110

/-- Proves that the percentage of a parking lot useable for parking is 80%, given specific conditions. -/
theorem parking_lot_useable_percentage :
  ∀ (length width : ℝ) (area_per_car : ℝ) (num_cars : ℕ),
    length = 400 →
    width = 500 →
    area_per_car = 10 →
    num_cars = 16000 →
    (((num_cars : ℝ) * area_per_car) / (length * width)) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_useable_percentage_l831_83110


namespace NUMINAMATH_CALUDE_arccos_cos_eleven_l831_83100

theorem arccos_cos_eleven : 
  Real.arccos (Real.cos 11) = 11 - 4 * Real.pi + 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eleven_l831_83100


namespace NUMINAMATH_CALUDE_movie_theater_ticket_sales_l831_83112

/-- Represents the type of ticket --/
inductive TicketType
  | Adult
  | Child
  | SeniorOrStudent

/-- Represents the showtime --/
inductive Showtime
  | Matinee
  | Evening

/-- Returns the price of a ticket based on its type and showtime --/
def ticketPrice (t : TicketType) (s : Showtime) : ℕ :=
  match s, t with
  | Showtime.Matinee, TicketType.Adult => 5
  | Showtime.Matinee, TicketType.Child => 3
  | Showtime.Matinee, TicketType.SeniorOrStudent => 4
  | Showtime.Evening, TicketType.Adult => 9
  | Showtime.Evening, TicketType.Child => 5
  | Showtime.Evening, TicketType.SeniorOrStudent => 6

theorem movie_theater_ticket_sales
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (adult_tickets : ℕ)
  (child_tickets : ℕ)
  (senior_student_tickets : ℕ)
  (h1 : total_tickets = 1500)
  (h2 : total_revenue = 10500)
  (h3 : child_tickets = adult_tickets + 300)
  (h4 : 2 * (adult_tickets + child_tickets) = senior_student_tickets)
  (h5 : total_tickets = adult_tickets + child_tickets + senior_student_tickets) :
  adult_tickets = 100 := by
  sorry

#check movie_theater_ticket_sales

end NUMINAMATH_CALUDE_movie_theater_ticket_sales_l831_83112


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_value_l831_83179

/-- Triangle DEF with given side lengths -/
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (FD : ℝ)

/-- Lines parallel to the sides of the triangle -/
structure ParallelLines :=
  (ℓD : ℝ)  -- Length of intersection with triangle interior
  (ℓE : ℝ)
  (ℓF : ℝ)

/-- The perimeter of the inner triangle formed by parallel lines -/
def inner_triangle_perimeter (t : Triangle) (p : ParallelLines) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the inner triangle -/
theorem inner_triangle_perimeter_value :
  let t : Triangle := { DE := 150, EF := 250, FD := 200 }
  let p : ParallelLines := { ℓD := 65, ℓE := 55, ℓF := 25 }
  inner_triangle_perimeter t p = 990 :=
sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_value_l831_83179


namespace NUMINAMATH_CALUDE_cube_root_negative_a_l831_83136

theorem cube_root_negative_a (a : ℝ) : 
  ((-a : ℝ) ^ (1/3 : ℝ) = Real.sqrt 2) → (a ^ (1/3 : ℝ) = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_negative_a_l831_83136


namespace NUMINAMATH_CALUDE_no_base6_digit_divisible_by_7_l831_83150

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (d : ℕ) : ℕ := 3 * 6^3 + d * 6^2 + d * 6 + 6

/-- Represents a base-6 digit --/
def isBase6Digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 5

theorem no_base6_digit_divisible_by_7 : 
  ∀ d : ℕ, isBase6Digit d → ¬(base6ToBase10 d % 7 = 0) := by
  sorry

#check no_base6_digit_divisible_by_7

end NUMINAMATH_CALUDE_no_base6_digit_divisible_by_7_l831_83150


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l831_83163

/-- The complex equation whose roots define the points on the ellipse -/
def complex_equation (z : ℂ) : Prop :=
  (z + 1) * (z^2 + 6*z + 10) * (z^2 + 8*z + 18) = 0

/-- The set of solutions to the complex equation -/
def solution_set : Set ℂ :=
  {z : ℂ | complex_equation z}

/-- The condition that the solutions are in the form x_k + y_k*i with x_k and y_k real -/
axiom solutions_form : ∀ z ∈ solution_set, ∃ (x y : ℝ), z = x + y * Complex.I

/-- The unique ellipse passing through the points defined by the solutions -/
axiom exists_unique_ellipse : ∃! E : Set (ℝ × ℝ), 
  ∀ z ∈ solution_set, (z.re, z.im) ∈ E

/-- The eccentricity of the ellipse -/
def eccentricity (E : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the eccentricity of the ellipse is √(3/4) -/
theorem ellipse_eccentricity : 
  ∀ E : Set (ℝ × ℝ), (∀ z ∈ solution_set, (z.re, z.im) ∈ E) → 
    eccentricity E = Real.sqrt (3/4) := 
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l831_83163


namespace NUMINAMATH_CALUDE_restore_original_example_l831_83131

def original_product : ℕ := 4 * 5 * 4 * 5 * 4

def changed_product : ℕ := 2247

def num_changed_digits : ℕ := 2

theorem restore_original_example :
  (original_product = 2240) ∧
  (∃ (a b : ℕ), a ≠ b ∧ a ≤ 9 ∧ b ≤ 9 ∧
    changed_product = original_product + a * 10 - b) :=
sorry

end NUMINAMATH_CALUDE_restore_original_example_l831_83131


namespace NUMINAMATH_CALUDE_simplify_fraction_l831_83167

theorem simplify_fraction (a : ℝ) (h : a ≠ -3) :
  (a^2 / (a + 3)) - (9 / (a + 3)) = a - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l831_83167


namespace NUMINAMATH_CALUDE_distribute_fraction_over_parentheses_l831_83133

theorem distribute_fraction_over_parentheses (x : ℝ) : (1 / 3) * (6 * x - 3) = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_distribute_fraction_over_parentheses_l831_83133


namespace NUMINAMATH_CALUDE_product_evaluation_l831_83132

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 6560 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l831_83132


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_condition_l831_83101

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_parallel_to_plane : Line → Plane → Prop)

-- Define the containment relation of a line in a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane_condition 
  (a : Line) (α : Plane) :
  (∃ β : Plane, line_in_plane a β ∧ plane_parallel α β) →
  line_parallel_to_plane a α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_condition_l831_83101


namespace NUMINAMATH_CALUDE_nells_baseball_cards_l831_83109

/-- Nell's baseball card collection problem -/
theorem nells_baseball_cards 
  (cards_given_to_jeff : ℕ) 
  (cards_left : ℕ) 
  (h1 : cards_given_to_jeff = 28) 
  (h2 : cards_left = 276) : 
  cards_given_to_jeff + cards_left = 304 := by
sorry

end NUMINAMATH_CALUDE_nells_baseball_cards_l831_83109


namespace NUMINAMATH_CALUDE_complex_sum_powers_l831_83195

theorem complex_sum_powers (z : ℂ) (hz : z^5 + z + 1 = 0) :
  z^103 + z^104 + z^105 + z^106 + z^107 + z^108 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l831_83195


namespace NUMINAMATH_CALUDE_vertices_count_l831_83174

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for convex polyhedra -/
axiom eulers_formula (p : ConvexPolyhedron) : p.vertices - p.edges + p.faces = 2

/-- A face of a polyhedron -/
inductive Face
| Triangle : Face

/-- Our specific polyhedron -/
def our_polyhedron : ConvexPolyhedron where
  vertices := 12  -- This is what we want to prove
  edges := 30
  faces := 20

/-- All faces of our polyhedron are triangles -/
axiom all_faces_triangular : ∀ f : Face, f = Face.Triangle

/-- The number of vertices in our polyhedron is correct -/
theorem vertices_count : our_polyhedron.vertices = 12 := by sorry

end NUMINAMATH_CALUDE_vertices_count_l831_83174


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l831_83103

theorem trigonometric_equation_solution :
  ∀ t : ℝ,
  (5.7 * Real.cos t * Real.sin (π / 2 + 6 * t) + Real.cos (π / 2 - t) * Real.sin (6 * t) = Real.cos (6 * t) + Real.cos (4 * t)) ↔
  (∃ k : ℤ, t = π / 10 * (2 * k + 1) ∨ t = π / 3 + 2 * π * k ∨ t = -π / 3 + 2 * π * k) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l831_83103


namespace NUMINAMATH_CALUDE_second_hand_movement_l831_83127

/-- Represents the number of seconds it takes for the second hand to move from one number to another on a clock face -/
def secondsBetweenNumbers (start finish : Nat) : Nat :=
  ((finish - start + 12) % 12) * 5

theorem second_hand_movement : secondsBetweenNumbers 5 9 ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_second_hand_movement_l831_83127


namespace NUMINAMATH_CALUDE_dog_walking_family_size_dog_walking_family_size_proof_l831_83183

/-- Calculates the number of family members contributing to a vacation based on dog-walking earnings --/
theorem dog_walking_family_size 
  (total_vacation_cost : ℝ)
  (start_fee : ℝ)
  (per_block_fee : ℝ)
  (num_dogs : ℕ)
  (total_blocks : ℕ)
  (h1 : total_vacation_cost = 1000)
  (h2 : start_fee = 2)
  (h3 : per_block_fee = 1.25)
  (h4 : num_dogs = 20)
  (h5 : total_blocks = 128)
  : ℕ :=
let total_earned := start_fee * num_dogs + per_block_fee * total_blocks
let family_size := total_vacation_cost / total_earned
5

theorem dog_walking_family_size_proof 
  (total_vacation_cost : ℝ)
  (start_fee : ℝ)
  (per_block_fee : ℝ)
  (num_dogs : ℕ)
  (total_blocks : ℕ)
  (h1 : total_vacation_cost = 1000)
  (h2 : start_fee = 2)
  (h3 : per_block_fee = 1.25)
  (h4 : num_dogs = 20)
  (h5 : total_blocks = 128)
  : dog_walking_family_size total_vacation_cost start_fee per_block_fee num_dogs total_blocks h1 h2 h3 h4 h5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dog_walking_family_size_dog_walking_family_size_proof_l831_83183


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l831_83124

/-- The equation of the tangent line to the circle x^2 + y^2 = 5 at the point (2, 1) is 2x + y - 5 = 0 -/
theorem tangent_line_to_circle (x y : ℝ) : 
  (2 : ℝ)^2 + 1^2 = 5 →  -- Point (2, 1) lies on the circle
  (∀ (a b : ℝ), a^2 + b^2 = 5 → (2*a + b = 5 → a = 2 ∧ b = 1)) →  -- (2, 1) is the only point of intersection
  2*x + y - 5 = 0 ↔ (x - 2)*(2) + (y - 1)*(1) = 0  -- Equation of tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l831_83124


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l831_83125

theorem power_mod_thirteen : 7^2000 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l831_83125


namespace NUMINAMATH_CALUDE_second_row_sum_is_528_l831_83172

/-- Represents a square grid -/
structure Grid (n : ℕ) :=
  (elements : Matrix (Fin n) (Fin n) ℕ)

/-- Fills the grid with numbers from 1 to n^2 in a clockwise spiral starting from the center -/
def fillGrid (n : ℕ) : Grid n :=
  sorry

/-- Returns the second row from the top of the grid -/
def secondRow (g : Grid 17) : Fin 17 → ℕ :=
  sorry

/-- The greatest number in the second row -/
def maxSecondRow (g : Grid 17) : ℕ :=
  sorry

/-- The least number in the second row -/
def minSecondRow (g : Grid 17) : ℕ :=
  sorry

/-- Theorem stating that the sum of the greatest and least numbers in the second row is 528 -/
theorem second_row_sum_is_528 :
  let g := fillGrid 17
  maxSecondRow g + minSecondRow g = 528 :=
sorry

end NUMINAMATH_CALUDE_second_row_sum_is_528_l831_83172


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l831_83155

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_equals_one (m : ℝ) : B m ⊆ A m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l831_83155


namespace NUMINAMATH_CALUDE_cosine_range_in_triangle_l831_83108

theorem cosine_range_in_triangle (A B C : Real) (h : 1 / Real.tan B + 1 / Real.tan C = 1 / Real.tan A) :
  2/3 ≤ Real.cos A ∧ Real.cos A < 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_range_in_triangle_l831_83108


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l831_83141

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (9 * a) % 35 = 1 ∧ 
    (7 * b) % 35 = 1 ∧ 
    (7 * a + 3 * b) % 35 = 8 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l831_83141


namespace NUMINAMATH_CALUDE_train_average_speed_l831_83193

/-- Given a train that travels two segments with known distances and times, 
    calculate its average speed. -/
theorem train_average_speed 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) 
  (h1 : distance1 = 325) (h2 : time1 = 3.5) 
  (h3 : distance2 = 470) (h4 : time2 = 4) : 
  (distance1 + distance2) / (time1 + time2) = 106 := by
  sorry

#eval (325 + 470) / (3.5 + 4)

end NUMINAMATH_CALUDE_train_average_speed_l831_83193


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l831_83186

theorem quadratic_root_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hroot : a * 2^2 - (a + b + c) * 2 + (b + c) = 0) :
  ∃ x : ℝ, x ≠ 2 ∧ a * x^2 - (a + b + c) * x + (b + c) = 0 ∧ x = (b + c - a) / a :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l831_83186


namespace NUMINAMATH_CALUDE_two_group_subcommittee_count_l831_83146

theorem two_group_subcommittee_count :
  let total_people : ℕ := 8
  let group_a_size : ℕ := 5
  let group_b_size : ℕ := 3
  let subcommittee_size : ℕ := 2
  group_a_size + group_b_size = total_people →
  group_a_size * group_b_size = 15
  := by sorry

end NUMINAMATH_CALUDE_two_group_subcommittee_count_l831_83146


namespace NUMINAMATH_CALUDE_fireworks_saved_l831_83122

/-- The number of fireworks Henry and his friend had saved from last year -/
def fireworks_problem (henry_new : ℕ) (friend_new : ℕ) (total : ℕ) : Prop :=
  henry_new + friend_new + (total - (henry_new + friend_new)) = total

theorem fireworks_saved (henry_new friend_new total : ℕ) 
  (h1 : henry_new = 2)
  (h2 : friend_new = 3)
  (h3 : total = 11) :
  fireworks_problem henry_new friend_new total ∧ 
  (total - (henry_new + friend_new) = 6) :=
by sorry

end NUMINAMATH_CALUDE_fireworks_saved_l831_83122


namespace NUMINAMATH_CALUDE_selene_sandwich_count_l831_83118

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℕ := 2

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℕ := 2

/-- The cost of a hotdog in dollars -/
def hotdog_cost : ℕ := 1

/-- The cost of a can of fruit juice in dollars -/
def juice_cost : ℕ := 2

/-- The number of hamburgers Tanya buys -/
def tanya_hamburgers : ℕ := 2

/-- The number of cans of fruit juice Tanya buys -/
def tanya_juice : ℕ := 2

/-- The total amount spent by Selene and Tanya in dollars -/
def total_spent : ℕ := 16

/-- The number of sandwiches Selene bought -/
def selene_sandwiches : ℕ := 3

theorem selene_sandwich_count :
  ∃ (x : ℕ), x * sandwich_cost + juice_cost + 
  tanya_hamburgers * hamburger_cost + tanya_juice * juice_cost = total_spent ∧
  x = selene_sandwiches :=
by sorry

end NUMINAMATH_CALUDE_selene_sandwich_count_l831_83118


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l831_83166

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 200 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l831_83166


namespace NUMINAMATH_CALUDE_age_difference_decade_difference_l831_83168

/-- Given that the sum of ages of x and y is 10 years greater than the sum of ages of y and z,
    prove that x is 1 decade older than z. -/
theorem age_difference (x y z : ℕ) (h : x + y = y + z + 10) : x = z + 10 := by
  sorry

/-- A decade is defined as 10 years. -/
def decade : ℕ := 10

/-- Given that x is 10 years older than z, prove that x is 1 decade older than z. -/
theorem decade_difference (x z : ℕ) (h : x = z + 10) : x = z + decade := by
  sorry

end NUMINAMATH_CALUDE_age_difference_decade_difference_l831_83168


namespace NUMINAMATH_CALUDE_trapezium_side_length_first_parallel_side_length_l831_83143

theorem trapezium_side_length : ℝ → Prop :=
  fun x =>
    let area : ℝ := 247
    let other_side : ℝ := 18
    let height : ℝ := 13
    area = (1 / 2) * (x + other_side) * height →
    x = 20

/-- The length of the first parallel side of the trapezium is 20 cm. -/
theorem first_parallel_side_length : trapezium_side_length 20 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_first_parallel_side_length_l831_83143


namespace NUMINAMATH_CALUDE_correct_average_l831_83119

/-- Given 10 numbers with an initial average of 14, where one number 36 was incorrectly read as 26, prove that the correct average is 15. -/
theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 14 →
  incorrect_num = 26 →
  correct_num = 36 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 15 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l831_83119


namespace NUMINAMATH_CALUDE_ferry_distance_ratio_l831_83191

/-- The ratio of distances covered by two ferries --/
theorem ferry_distance_ratio :
  let v_p : ℝ := 6  -- Speed of ferry P in km/h
  let t_p : ℝ := 3  -- Time taken by ferry P in hours
  let v_q : ℝ := v_p + 3  -- Speed of ferry Q in km/h
  let t_q : ℝ := t_p + 1  -- Time taken by ferry Q in hours
  let d_p : ℝ := v_p * t_p  -- Distance covered by ferry P
  let d_q : ℝ := v_q * t_q  -- Distance covered by ferry Q
  d_q / d_p = 2 :=
by sorry

end NUMINAMATH_CALUDE_ferry_distance_ratio_l831_83191


namespace NUMINAMATH_CALUDE_min_value_of_a_l831_83115

theorem min_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x / (x^2 + 3*x + 1) ≤ a) → a ≥ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l831_83115


namespace NUMINAMATH_CALUDE_window_width_calculation_l831_83158

/-- Represents the dimensions of a glass pane -/
structure Pane where
  height : ℝ
  width : ℝ

/-- Represents the dimensions of a window -/
structure Window where
  rows : ℕ
  columns : ℕ
  pane : Pane
  border_width : ℝ

/-- Calculates the width of a window given its specifications -/
def window_width (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- Theorem stating the width of the window with given specifications -/
theorem window_width_calculation (x : ℝ) :
  let w : Window := {
    rows := 3,
    columns := 4,
    pane := { height := 4 * x, width := 3 * x },
    border_width := 3
  }
  window_width w = 12 * x + 15 := by sorry

end NUMINAMATH_CALUDE_window_width_calculation_l831_83158


namespace NUMINAMATH_CALUDE_problem_statement_l831_83148

def f (m x : ℝ) : ℝ := m * x^2 + (1 - m) * x + m - 2

theorem problem_statement :
  (∀ m : ℝ, (∀ x : ℝ, f m x + 2 ≥ 0) ↔ m ≥ 1/3) ∧
  (∀ m : ℝ, m < 0 →
    (∀ x : ℝ, f m x < m - 1 ↔
      (m ≤ -1 ∧ (x < -1/m ∨ x > 1)) ∨
      (-1 < m ∧ m < 0 ∧ (x < 1 ∨ x > -1/m)))) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l831_83148


namespace NUMINAMATH_CALUDE_quadratic_factorization_l831_83188

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l831_83188


namespace NUMINAMATH_CALUDE_sqrt_13_parts_sum_l831_83139

theorem sqrt_13_parts_sum (a b : ℝ) : 
  (a = ⌊Real.sqrt 13⌋) → 
  (b = Real.sqrt 13 - ⌊Real.sqrt 13⌋) → 
  2 * a^2 + b - Real.sqrt 13 = 15 := by
sorry

end NUMINAMATH_CALUDE_sqrt_13_parts_sum_l831_83139


namespace NUMINAMATH_CALUDE_divisibility_by_thirteen_l831_83107

theorem divisibility_by_thirteen (n : ℕ) : (4 * 3^(2^n) + 3 * 4^(2^n)) % 13 = 0 ↔ n % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_thirteen_l831_83107


namespace NUMINAMATH_CALUDE_fraction_simplification_l831_83177

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l831_83177


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l831_83171

/-- Given a mixture of two types of candy, prove the cost of the second type. -/
theorem candy_mixture_cost
  (first_candy_weight : ℝ)
  (first_candy_cost : ℝ)
  (second_candy_weight : ℝ)
  (mixture_cost : ℝ)
  (h1 : first_candy_weight = 20)
  (h2 : first_candy_cost = 10)
  (h3 : second_candy_weight = 80)
  (h4 : mixture_cost = 6)
  : (((first_candy_weight + second_candy_weight) * mixture_cost
     - first_candy_weight * first_candy_cost) / second_candy_weight) = 5 := by
  sorry

#check candy_mixture_cost

end NUMINAMATH_CALUDE_candy_mixture_cost_l831_83171


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l831_83154

/-- The perpendicular bisector of a line segment with endpoints (x₁, y₁) and (x₂, y₂) -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - x₁)^2 + (p.2 - y₁)^2 = (p.1 - x₂)^2 + (p.2 - y₂)^2}

theorem perpendicular_bisector_equation :
  perpendicular_bisector 1 3 5 (-1) = {p : ℝ × ℝ | p.1 - p.2 - 2 = 0} := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l831_83154


namespace NUMINAMATH_CALUDE_prob_third_grade_parent_is_three_fifths_l831_83189

/-- Represents the number of parents in each grade's committee -/
structure ParentCommittee where
  grade1 : Nat
  grade2 : Nat
  grade3 : Nat

/-- Represents the number of parents sampled from each grade -/
structure SampledParents where
  grade1 : Nat
  grade2 : Nat
  grade3 : Nat

/-- Calculates the total number of parents in all committees -/
def totalParents (pc : ParentCommittee) : Nat :=
  pc.grade1 + pc.grade2 + pc.grade3

/-- Calculates the stratified sample for each grade -/
def calculateSample (pc : ParentCommittee) (totalSample : Nat) : SampledParents :=
  let ratio := totalSample / (totalParents pc)
  { grade1 := pc.grade1 * ratio
  , grade2 := pc.grade2 * ratio
  , grade3 := pc.grade3 * ratio }

/-- Calculates the probability of selecting at least one third-grade parent -/
def probThirdGradeParent (sp : SampledParents) : Rat :=
  let totalCombinations := (sp.grade1 + sp.grade2 + sp.grade3).choose 2
  let favorableCombinations := sp.grade3 * (sp.grade1 + sp.grade2) + sp.grade3.choose 2
  favorableCombinations / totalCombinations

theorem prob_third_grade_parent_is_three_fifths 
  (pc : ParentCommittee) 
  (h1 : pc.grade1 = 54)
  (h2 : pc.grade2 = 18)
  (h3 : pc.grade3 = 36)
  (totalSample : Nat)
  (h4 : totalSample = 6) :
  probThirdGradeParent (calculateSample pc totalSample) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_third_grade_parent_is_three_fifths_l831_83189


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_equals_5_l831_83111

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 9 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = (3/5) * x

-- Theorem statement
theorem hyperbola_asymptote_implies_a_equals_5 (a : ℝ) (h1 : a > 0) :
  (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) → a = 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_equals_5_l831_83111


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l831_83135

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the property of skew lines
variable (skew : Line → Line → Prop)

-- Theorem 1: If a line is perpendicular to two planes, then those planes are parallel
theorem perpendicular_implies_parallel
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : perpendicular m β) :
  plane_parallel α β :=
sorry

-- Theorem 2: If two skew lines are each perpendicular to one plane and parallel to the other, 
-- then the planes are perpendicular
theorem skew_perpendicular_parallel_implies_perpendicular
  (m n : Line) (α β : Plane)
  (h1 : skew m n)
  (h2 : perpendicular m α)
  (h3 : parallel m β)
  (h4 : perpendicular n β)
  (h5 : parallel n α) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l831_83135


namespace NUMINAMATH_CALUDE_stating_cardboard_ratio_l831_83181

/-- Represents the number of dogs on a Type 1 cardboard -/
def dogs_type1 : ℕ := 28

/-- Represents the number of cats on a Type 1 cardboard -/
def cats_type1 : ℕ := 28

/-- Represents the number of cats on a Type 2 cardboard -/
def cats_type2 : ℕ := 42

/-- Represents the required ratio of cats to dogs -/
def required_ratio : ℚ := 5 / 3

/-- 
Theorem stating that the ratio of Type 1 to Type 2 cardboard 
that satisfies the required cat to dog ratio is 9:4
-/
theorem cardboard_ratio : 
  ∀ (x y : ℚ), 
    x > 0 → y > 0 →
    (cats_type1 * x + cats_type2 * y) / (dogs_type1 * x) = required_ratio →
    x / y = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_stating_cardboard_ratio_l831_83181


namespace NUMINAMATH_CALUDE_spring_mows_count_l831_83161

def total_mows : ℕ := 11
def summer_mows : ℕ := 5

theorem spring_mows_count : total_mows - summer_mows = 6 := by
  sorry

end NUMINAMATH_CALUDE_spring_mows_count_l831_83161


namespace NUMINAMATH_CALUDE_append_digits_divisible_by_36_l831_83114

/-- A function that checks if a number is divisible by 36 -/
def isDivisibleBy36 (n : ℕ) : Prop := n % 36 = 0

/-- A function that appends two digits to 2020 -/
def appendTwoDigits (a b : ℕ) : ℕ := 202000 + 10 * a + b

theorem append_digits_divisible_by_36 :
  ∀ a b : ℕ, a < 10 → b < 10 →
    (isDivisibleBy36 (appendTwoDigits a b) ↔ (a = 3 ∧ b = 2) ∨ (a = 6 ∧ b = 8)) := by
  sorry

#check append_digits_divisible_by_36

end NUMINAMATH_CALUDE_append_digits_divisible_by_36_l831_83114


namespace NUMINAMATH_CALUDE_tangent_equation_solutions_l831_83182

open Real

theorem tangent_equation_solutions (t : ℝ) :
  cos t ≠ 0 →
  (tan t = (sin t ^ 2 + sin (2 * t) - 1) / (cos t ^ 2 - sin (2 * t) + 1)) ↔
  (∃ k : ℤ, t = π / 4 + π * k) ∨
  (∃ n : ℤ, t = arctan ((1 - Real.sqrt 5) / 2) + π * n) ∨
  (∃ l : ℤ, t = arctan ((1 + Real.sqrt 5) / 2) + π * l) :=
by sorry

end NUMINAMATH_CALUDE_tangent_equation_solutions_l831_83182


namespace NUMINAMATH_CALUDE_quadratic_to_linear_inequality_l831_83137

theorem quadratic_to_linear_inequality (a b : ℝ) :
  (∀ x, x^2 + a*x + b > 0 ↔ x < 3 ∨ x > 1) →
  (∀ x, a*x + b < 0 ↔ x > 3/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_to_linear_inequality_l831_83137


namespace NUMINAMATH_CALUDE_special_rhombus_sum_l831_83113

/-- A rhombus with specific vertex coordinates and area -/
structure SpecialRhombus where
  a : ℤ
  b : ℤ
  a_pos : 0 < a
  b_pos : 0 < b
  a_neq_b : a ≠ b
  area_eq : 2 * (a - b)^2 = 32

/-- The sum of a and b in a SpecialRhombus is 8 -/
theorem special_rhombus_sum (r : SpecialRhombus) : r.a + r.b = 8 := by
  sorry

end NUMINAMATH_CALUDE_special_rhombus_sum_l831_83113


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l831_83160

/-- The surface area of a cuboid with given dimensions. -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * breadth + breadth * height + length * height)

/-- Theorem: The surface area of a cuboid with length 15, breadth 10, and height 16 is 1100. -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 15 10 16 = 1100 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l831_83160


namespace NUMINAMATH_CALUDE_chord_intercept_theorem_l831_83149

def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - 20 = 0

def line_equation (x y c : ℝ) : Prop :=
  5*x - 12*y + c = 0

def chord_length (c : ℝ) : ℝ := 8

theorem chord_intercept_theorem (c : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation x₁ y₁ c ∧ line_equation x₂ y₂ c ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = (chord_length c)^2) ↔
  c = 10 ∨ c = -68 :=
sorry

end NUMINAMATH_CALUDE_chord_intercept_theorem_l831_83149


namespace NUMINAMATH_CALUDE_curve_and_intersection_l831_83194

/-- The polar equation of curve C -/
def polar_equation (ρ θ a : ℝ) : Prop :=
  ρ * Real.sqrt (a^2 * Real.sin θ^2 + 4 * Real.cos θ^2) = 2 * a

/-- The Cartesian equation of curve C -/
def cartesian_equation (x y a : ℝ) : Prop :=
  4 * x^2 + a^2 * y^2 = 4 * a^2

/-- The parametric equations of line l -/
def line_equation (x y t : ℝ) : Prop :=
  x = Real.sqrt 3 + t ∧ y = 7 + Real.sqrt 3 * t

/-- Point P -/
def point_P : ℝ × ℝ := (0, 4)

/-- The distance product condition -/
def distance_product (a : ℝ) : Prop :=
  ∃ (M N : ℝ × ℝ), line_equation M.1 M.2 (M.1 - Real.sqrt 3) ∧
                   line_equation N.1 N.2 (N.1 - Real.sqrt 3) ∧
                   cartesian_equation M.1 M.2 a ∧
                   cartesian_equation N.1 N.2 a ∧
                   (M.1 - point_P.1)^2 + (M.2 - point_P.2)^2 *
                   (N.1 - point_P.1)^2 + (N.2 - point_P.2)^2 = 14^2

theorem curve_and_intersection :
  (∀ (ρ θ : ℝ), polar_equation ρ θ a ↔ cartesian_equation (ρ * Real.cos θ) (ρ * Real.sin θ) a) ∧
  (distance_product a → a = 2 * Real.sqrt 21 / 3) := by
  sorry

end NUMINAMATH_CALUDE_curve_and_intersection_l831_83194


namespace NUMINAMATH_CALUDE_new_profit_percentage_after_doubling_price_l831_83180

-- Define the initial profit percentage
def initial_profit_percentage : ℝ := 30

-- Define the price multiplier for the new selling price
def price_multiplier : ℝ := 2

-- Theorem to prove
theorem new_profit_percentage_after_doubling_price :
  let original_selling_price := 100 + initial_profit_percentage
  let new_selling_price := price_multiplier * original_selling_price
  let new_profit := new_selling_price - 100
  let new_profit_percentage := (new_profit / 100) * 100
  new_profit_percentage = 160 := by sorry

end NUMINAMATH_CALUDE_new_profit_percentage_after_doubling_price_l831_83180


namespace NUMINAMATH_CALUDE_sarah_sock_purchase_l831_83153

/-- Represents the number of pairs of socks at each price point --/
structure SockCounts where
  two_dollar : ℕ
  four_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the given sock counts satisfy the problem conditions --/
def is_valid_solution (s : SockCounts) : Prop :=
  s.two_dollar + s.four_dollar + s.five_dollar = 15 ∧
  2 * s.two_dollar + 4 * s.four_dollar + 5 * s.five_dollar = 45 ∧
  s.two_dollar ≥ 1 ∧ s.four_dollar ≥ 1 ∧ s.five_dollar ≥ 1

theorem sarah_sock_purchase :
  ∃ (s : SockCounts), is_valid_solution s ∧ (s.two_dollar = 8 ∨ s.two_dollar = 9) :=
sorry

end NUMINAMATH_CALUDE_sarah_sock_purchase_l831_83153


namespace NUMINAMATH_CALUDE_product_of_3_6_and_0_25_l831_83197

theorem product_of_3_6_and_0_25 : (3.6 : ℝ) * 0.25 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_3_6_and_0_25_l831_83197


namespace NUMINAMATH_CALUDE_card_position_unique_card_position_valid_l831_83142

/-- Represents a position in a 6x6 grid -/
structure Position where
  row : Fin 6
  col : Fin 6

/-- Represents the magician's trick setup -/
structure MagicTrick where
  initialColumn : Fin 6
  finalColumn : Fin 6

/-- Given the initial and final column numbers, determines the unique position of the card in the final layout -/
def findCardPosition (trick : MagicTrick) : Position :=
  { row := trick.initialColumn
  , col := trick.finalColumn }

/-- Theorem stating that the card position can be uniquely determined -/
theorem card_position_unique (trick : MagicTrick) :
  ∃! pos : Position, pos = findCardPosition trick :=
sorry

/-- Theorem stating that the determined position is valid within the 6x6 grid -/
theorem card_position_valid (trick : MagicTrick) :
  let pos := findCardPosition trick
  pos.row < 6 ∧ pos.col < 6 :=
sorry

end NUMINAMATH_CALUDE_card_position_unique_card_position_valid_l831_83142


namespace NUMINAMATH_CALUDE_work_earnings_equation_l831_83151

theorem work_earnings_equation (t : ℝ) : 
  (t + 1) * (3 * t - 3) = (3 * t - 5) * (t + 2) + 2 → t = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equation_l831_83151


namespace NUMINAMATH_CALUDE_battery_purchase_l831_83130

/-- Given three types of batteries A, B, and C with different prices, prove that 48 type C batteries can be bought with a certain amount of money. -/
theorem battery_purchase (x y z : ℚ) (W : ℕ) : 
  (4 * x + 18 * y + 16 * z = W * z) →
  (2 * x + 15 * y + 24 * z = W * z) →
  (6 * x + 12 * y + 20 * z = W * z) →
  W = 48 := by sorry

end NUMINAMATH_CALUDE_battery_purchase_l831_83130


namespace NUMINAMATH_CALUDE_long_furred_brown_dogs_l831_83104

theorem long_furred_brown_dogs (total : ℕ) (long_furred : ℕ) (brown : ℕ) (neither : ℕ) :
  total = 45 →
  long_furred = 36 →
  brown = 27 →
  neither = 8 →
  long_furred + brown - (total - neither) = 26 :=
by sorry

end NUMINAMATH_CALUDE_long_furred_brown_dogs_l831_83104


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l831_83147

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (x + 1) / (x - 1) = 0 ∧ x ≠ 1 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l831_83147
