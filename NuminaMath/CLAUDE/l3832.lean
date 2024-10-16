import Mathlib

namespace NUMINAMATH_CALUDE_students_agreement_count_l3832_383245

theorem students_agreement_count :
  let third_grade_count : ℕ := 154
  let fourth_grade_count : ℕ := 237
  third_grade_count + fourth_grade_count = 391 :=
by
  sorry

end NUMINAMATH_CALUDE_students_agreement_count_l3832_383245


namespace NUMINAMATH_CALUDE_border_area_is_144_l3832_383286

/-- The area of the border of a framed rectangular photograph -/
def border_area (photo_height photo_width border_width : ℝ) : ℝ :=
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - photo_height * photo_width

/-- Theorem: The area of the border of a framed rectangular photograph is 144 square inches -/
theorem border_area_is_144 :
  border_area 8 10 3 = 144 := by
  sorry

end NUMINAMATH_CALUDE_border_area_is_144_l3832_383286


namespace NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_450_l3832_383232

theorem least_multiple_of_25_greater_than_450 :
  ∀ n : ℕ, n > 0 ∧ 25 ∣ n ∧ n > 450 → n ≥ 475 :=
by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_450_l3832_383232


namespace NUMINAMATH_CALUDE_adams_goats_l3832_383211

theorem adams_goats (adam andrew ahmed : ℕ) 
  (andrew_eq : andrew = 2 * adam + 5)
  (ahmed_eq : ahmed = andrew - 6)
  (ahmed_count : ahmed = 13) : 
  adam = 7 := by
  sorry

end NUMINAMATH_CALUDE_adams_goats_l3832_383211


namespace NUMINAMATH_CALUDE_broomstick_charge_theorem_l3832_383202

/-- Represents the state of the broomstick at a given time -/
structure BroomState where
  minutes : Nat  -- Minutes since midnight
  charge : Nat   -- Current charge (0-100)

/-- Calculates the charge of the broomstick given the number of minutes since midnight -/
def calculateCharge (minutes : Nat) : Nat :=
  100 - minutes / 6

/-- Checks if the given time (in minutes since midnight) is a solution -/
def isSolution (minutes : Nat) : Bool :=
  let charge := calculateCharge minutes
  let minutesPastHour := minutes % 60
  charge == minutesPastHour

/-- The set of solution times -/
def solutionTimes : List BroomState :=
  [
    { minutes := 292, charge := 52 },  -- 04:52
    { minutes := 343, charge := 43 },  -- 05:43
    { minutes := 395, charge := 35 },  -- 06:35
    { minutes := 446, charge := 26 },  -- 07:26
    { minutes := 549, charge := 9 }    -- 09:09
  ]

/-- Main theorem: The given solution times are correct and complete -/
theorem broomstick_charge_theorem :
  (∀ t ∈ solutionTimes, isSolution t.minutes) ∧
  (∀ m, 0 ≤ m ∧ m < 600 → isSolution m → (∃ t ∈ solutionTimes, t.minutes = m)) :=
sorry


end NUMINAMATH_CALUDE_broomstick_charge_theorem_l3832_383202


namespace NUMINAMATH_CALUDE_triangle_existence_and_perimeter_l3832_383235

/-- A triangle with sides a, b, and c is valid if it satisfies the triangle inequality theorem -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- The perimeter of a triangle with sides a, b, and c -/
def triangle_perimeter (a b c : ℝ) : ℝ :=
  a + b + c

/-- Theorem: The given lengths form a valid triangle with perimeter 44 -/
theorem triangle_existence_and_perimeter :
  let a := 15
  let b := 11
  let c := 18
  is_valid_triangle a b c ∧ triangle_perimeter a b c = 44 := by sorry

end NUMINAMATH_CALUDE_triangle_existence_and_perimeter_l3832_383235


namespace NUMINAMATH_CALUDE_probability_king_of_diamonds_l3832_383224

/-- Represents a standard playing card suit -/
inductive Suit
| Spades
| Hearts
| Diamonds
| Clubs

/-- Represents a standard playing card rank -/
inductive Rank
| Ace
| Two
| Three
| Four
| Five
| Six
| Seven
| Eight
| Nine
| Ten
| Jack
| Queen
| King

/-- Represents a standard playing card -/
structure Card where
  rank : Rank
  suit : Suit

def standardDeck : Finset Card := sorry

/-- The probability of drawing a specific card from a standard deck -/
def probabilityOfCard (c : Card) : ℚ :=
  1 / (Finset.card standardDeck)

theorem probability_king_of_diamonds :
  probabilityOfCard ⟨Rank.King, Suit.Diamonds⟩ = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_probability_king_of_diamonds_l3832_383224


namespace NUMINAMATH_CALUDE_range_of_a_l3832_383205

theorem range_of_a (x a : ℝ) : 
  (∀ x, (x - 1) / (x - 3) < 0 → |x - a| < 2) ∧ 
  (∃ x, |x - a| < 2 ∧ (x - 1) / (x - 3) ≥ 0) →
  a ∈ Set.Icc 1 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3832_383205


namespace NUMINAMATH_CALUDE_largest_prime_to_test_for_500_to_550_l3832_383291

theorem largest_prime_to_test_for_500_to_550 (n : ℕ) :
  500 ≤ n ∧ n ≤ 550 →
  (∀ p : ℕ, Prime p ∧ p ≤ Real.sqrt n → p ≤ 23) ∧
  Prime 23 ∧ 23 ≤ Real.sqrt n :=
sorry

end NUMINAMATH_CALUDE_largest_prime_to_test_for_500_to_550_l3832_383291


namespace NUMINAMATH_CALUDE_unequal_grandchildren_probability_l3832_383257

theorem unequal_grandchildren_probability (n : ℕ) (p_male : ℝ) (p_female : ℝ) : 
  n = 12 →
  p_male = 0.6 →
  p_female = 0.4 →
  p_male + p_female = 1 →
  let p_equal := (n.choose (n / 2)) * (p_male ^ (n / 2)) * (p_female ^ (n / 2))
  1 - p_equal = 0.823 := by
sorry

end NUMINAMATH_CALUDE_unequal_grandchildren_probability_l3832_383257


namespace NUMINAMATH_CALUDE_sum_of_roots_l3832_383280

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a = 1) 
  (hb : b^3 - 3*b^2 + 5*b = 5) : 
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3832_383280


namespace NUMINAMATH_CALUDE_w_squared_value_l3832_383218

theorem w_squared_value (w : ℝ) (h : (w + 10)^2 = (4*w + 6)*(w + 5)) : w^2 = 70/3 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l3832_383218


namespace NUMINAMATH_CALUDE_martha_juice_bottles_l3832_383250

theorem martha_juice_bottles (initial_bottles pantry_bottles fridge_bottles consumed_bottles final_bottles : ℕ) 
  (h1 : initial_bottles = pantry_bottles + fridge_bottles)
  (h2 : pantry_bottles = 4)
  (h3 : fridge_bottles = 4)
  (h4 : consumed_bottles = 3)
  (h5 : final_bottles = 10) : 
  final_bottles - (initial_bottles - consumed_bottles) = 5 := by
  sorry

end NUMINAMATH_CALUDE_martha_juice_bottles_l3832_383250


namespace NUMINAMATH_CALUDE_no_such_function_exists_l3832_383265

theorem no_such_function_exists : ¬∃ (f : ℤ → ℤ), ∀ (x y : ℤ), f (x + f y) = f x - y := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l3832_383265


namespace NUMINAMATH_CALUDE_factor_expression_l3832_383275

theorem factor_expression (x : ℝ) : 16 * x^4 - 4 * x^2 = 4 * x^2 * (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3832_383275


namespace NUMINAMATH_CALUDE_eva_patch_area_l3832_383262

/-- Represents a rectangular vegetable patch -/
structure VegetablePatch where
  short_side : ℕ  -- Number of posts on the shorter side
  long_side : ℕ   -- Number of posts on the longer side
  post_spacing : ℕ -- Distance between posts in yards

/-- Properties of Eva's vegetable patch -/
def eva_patch : VegetablePatch where
  short_side := 3
  long_side := 9
  post_spacing := 6

/-- Total number of posts -/
def total_posts (p : VegetablePatch) : ℕ :=
  2 * (p.short_side + p.long_side) - 4

/-- Relationship between short and long sides -/
def side_relationship (p : VegetablePatch) : Prop :=
  p.long_side = 3 * p.short_side

/-- Calculate the area of the vegetable patch -/
def patch_area (p : VegetablePatch) : ℕ :=
  (p.short_side - 1) * (p.long_side - 1) * p.post_spacing * p.post_spacing

/-- Theorem stating the area of Eva's vegetable patch -/
theorem eva_patch_area :
  total_posts eva_patch = 24 ∧
  side_relationship eva_patch ∧
  patch_area eva_patch = 576 := by
  sorry

#eval patch_area eva_patch

end NUMINAMATH_CALUDE_eva_patch_area_l3832_383262


namespace NUMINAMATH_CALUDE_negation_of_implication_l3832_383244

theorem negation_of_implication (a b : ℝ) :
  ¬(a = 0 ∧ b = 0 → a^2 + b^2 = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3832_383244


namespace NUMINAMATH_CALUDE_square_division_has_triangle_l3832_383297

/-- A convex polygon within a square --/
structure PolygonInSquare where
  sides : ℕ
  convex : Bool
  inSquare : Bool

/-- Represents a division of a square into polygons --/
def SquareDivision := List PolygonInSquare

/-- Checks if all polygons in the division are convex and within the square --/
def isValidDivision (d : SquareDivision) : Prop :=
  d.all (λ p => p.convex ∧ p.inSquare)

/-- Checks if all polygons have distinct number of sides --/
def hasDistinctSides (d : SquareDivision) : Prop :=
  d.map (λ p => p.sides) |>.Nodup

/-- Checks if there's a triangle in the division --/
def hasTriangle (d : SquareDivision) : Prop :=
  d.any (λ p => p.sides = 3)

theorem square_division_has_triangle (d : SquareDivision) :
  d.length > 1 → isValidDivision d → hasDistinctSides d → hasTriangle d := by
  sorry

end NUMINAMATH_CALUDE_square_division_has_triangle_l3832_383297


namespace NUMINAMATH_CALUDE_product_expansion_sum_l3832_383269

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + c + d = 19 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l3832_383269


namespace NUMINAMATH_CALUDE_range_of_power_function_l3832_383203

/-- The range of g(x) = x^m for m > 0 on the interval (0, 1) is (0, 1) -/
theorem range_of_power_function (m : ℝ) (hm : m > 0) :
  Set.range (fun x : ℝ => x ^ m) ∩ Set.Ioo 0 1 = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_power_function_l3832_383203


namespace NUMINAMATH_CALUDE_cow_hen_problem_l3832_383296

theorem cow_hen_problem (cows hens : ℕ) : 
  4 * cows + 2 * hens = 2 * (cows + hens) + 8 → cows = 4 := by
  sorry

end NUMINAMATH_CALUDE_cow_hen_problem_l3832_383296


namespace NUMINAMATH_CALUDE_intersection_equality_implies_m_equals_five_l3832_383253

def A (m : ℝ) : Set ℝ := {-1, 3, m}
def B : Set ℝ := {3, 5}

theorem intersection_equality_implies_m_equals_five (m : ℝ) :
  B ∩ A m = B → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_m_equals_five_l3832_383253


namespace NUMINAMATH_CALUDE_tan_15_30_product_equals_two_l3832_383209

theorem tan_15_30_product_equals_two :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 :=
by
  have tan_45_eq_1 : Real.tan (45 * π / 180) = 1 := by sorry
  have tan_sum_15_30 : Real.tan ((15 + 30) * π / 180) = 
    (Real.tan (15 * π / 180) + Real.tan (30 * π / 180)) / 
    (1 - Real.tan (15 * π / 180) * Real.tan (30 * π / 180)) := by sorry
  sorry

end NUMINAMATH_CALUDE_tan_15_30_product_equals_two_l3832_383209


namespace NUMINAMATH_CALUDE_common_tangent_implies_a_b_equal_three_l3832_383288

/-- Given two functions f and g with a common tangent at (1, c), prove a = b = 3 -/
theorem common_tangent_implies_a_b_equal_three
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (a b : ℝ)
  (h_f : ∀ x, f x = a * x^2 + 1)
  (h_a_pos : a > 0)
  (h_g : ∀ x, g x = x^3 + b * x)
  (h_intersection : f 1 = g 1)
  (h_common_tangent : (deriv f) 1 = (deriv g) 1) :
  a = 3 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_common_tangent_implies_a_b_equal_three_l3832_383288


namespace NUMINAMATH_CALUDE_monotone_xfx_l3832_383260

open Real

theorem monotone_xfx (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, HasDerivAt f (f' x) x) 
  (h_ineq : ∀ x, x * f' x > -f x) (x₁ x₂ : ℝ) (h_lt : x₁ < x₂) : 
  x₁ * f x₁ < x₂ * f x₂ := by
  sorry

end NUMINAMATH_CALUDE_monotone_xfx_l3832_383260


namespace NUMINAMATH_CALUDE_bus_passengers_second_stop_l3832_383246

/-- Given a bus with the following properties:
  * 23 rows of 4 seats each
  * 16 people board at the start
  * At the first stop, 15 people board and 3 get off
  * At the second stop, 17 people board
  * There are 57 empty seats after the second stop
  Prove that 10 people got off at the second stop. -/
theorem bus_passengers_second_stop 
  (total_seats : ℕ) 
  (initial_passengers : ℕ) 
  (first_stop_on : ℕ) 
  (first_stop_off : ℕ) 
  (second_stop_on : ℕ) 
  (empty_seats_after_second : ℕ) 
  (h1 : total_seats = 23 * 4)
  (h2 : initial_passengers = 16)
  (h3 : first_stop_on = 15)
  (h4 : first_stop_off = 3)
  (h5 : second_stop_on = 17)
  (h6 : empty_seats_after_second = 57) :
  ∃ (second_stop_off : ℕ), 
    second_stop_off = 10 ∧ 
    empty_seats_after_second = total_seats - (initial_passengers + first_stop_on - first_stop_off + second_stop_on - second_stop_off) :=
by sorry

end NUMINAMATH_CALUDE_bus_passengers_second_stop_l3832_383246


namespace NUMINAMATH_CALUDE_inequality_solution_count_l3832_383276

theorem inequality_solution_count : 
  ∃! (x : ℕ), x > 0 ∧ 15 < -2 * (x : ℤ) + 17 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l3832_383276


namespace NUMINAMATH_CALUDE_total_spots_l3832_383258

def cow_spots (left_spots : ℕ) (right_spots : ℕ) : Prop :=
  (left_spots = 16) ∧ 
  (right_spots = 3 * left_spots + 7)

theorem total_spots : ∀ left_spots right_spots : ℕ, 
  cow_spots left_spots right_spots → left_spots + right_spots = 71 := by
sorry

end NUMINAMATH_CALUDE_total_spots_l3832_383258


namespace NUMINAMATH_CALUDE_probability_B_given_A_l3832_383201

/-- Represents the number of people in the research study group -/
def group_size : ℕ := 6

/-- Represents the number of halls in the exhibition -/
def num_halls : ℕ := 3

/-- Represents the event A: In the first hour, each hall has exactly 2 people -/
def event_A : Prop := True

/-- Represents the event B: In the second hour, there are exactly 2 people in Hall A -/
def event_B : Prop := True

/-- Represents the number of ways event B can occur given event A has occurred -/
def ways_B_given_A : ℕ := 3

/-- Represents the total number of possible distributions in the second hour -/
def total_distributions : ℕ := 8

/-- The probability of event B given event A -/
def P_B_given_A : ℚ := ways_B_given_A / total_distributions

theorem probability_B_given_A : 
  P_B_given_A = 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_B_given_A_l3832_383201


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l3832_383273

/-- Given a two-digit number with digit sum 6, if the product of this number and
    the number formed by swapping its digits is 1008, then the original number
    is either 42 or 24. -/
theorem two_digit_number_puzzle (n : ℕ) : 
  (n ≥ 10 ∧ n < 100) →  -- n is a two-digit number
  (n / 10 + n % 10 = 6) →  -- digit sum is 6
  (n * (10 * (n % 10) + (n / 10)) = 1008) →  -- product condition
  (n = 42 ∨ n = 24) := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l3832_383273


namespace NUMINAMATH_CALUDE_train_speed_is_45_km_per_hour_l3832_383283

-- Define the given parameters
def train_length : ℝ := 140
def bridge_length : ℝ := 235
def crossing_time : ℝ := 30

-- Define the conversion factor
def meters_per_second_to_km_per_hour : ℝ := 3.6

-- Theorem statement
theorem train_speed_is_45_km_per_hour :
  let total_distance := train_length + bridge_length
  let speed_in_meters_per_second := total_distance / crossing_time
  let speed_in_km_per_hour := speed_in_meters_per_second * meters_per_second_to_km_per_hour
  speed_in_km_per_hour = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_is_45_km_per_hour_l3832_383283


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3832_383214

theorem inequality_solution_set :
  {x : ℝ | -1/3 * x + 1 ≤ -5} = {x : ℝ | x ≥ 18} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3832_383214


namespace NUMINAMATH_CALUDE_congruence_problem_l3832_383252

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % (2^3) = 3^2 % (2^3))
  (h2 : (6 + x) % (3^3) = 2^3 % (3^3))
  (h3 : (8 + x) % (5^3) = 7^2 % (5^3)) :
  x % 30 = 17 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l3832_383252


namespace NUMINAMATH_CALUDE_pi_irrational_among_given_numbers_l3832_383268

theorem pi_irrational_among_given_numbers :
  (∃ (a b : ℤ), (1 : ℝ) / 3 = a / b) ∧
  (∃ (c d : ℤ), (0.201 : ℝ) = c / d) ∧
  (∃ (e f : ℤ), Real.sqrt 9 = e / f) →
  ¬∃ (m n : ℤ), Real.pi = m / n :=
by sorry

end NUMINAMATH_CALUDE_pi_irrational_among_given_numbers_l3832_383268


namespace NUMINAMATH_CALUDE_number_count_l3832_383261

theorem number_count (average : ℝ) (avg1 avg2 avg3 : ℝ) : 
  average = 6.40 →
  avg1 = 6.2 →
  avg2 = 6.1 →
  avg3 = 6.9 →
  (2 * avg1 + 2 * avg2 + 2 * avg3) / 6 = average →
  6 = (2 * avg1 + 2 * avg2 + 2 * avg3) / average :=
by sorry

end NUMINAMATH_CALUDE_number_count_l3832_383261


namespace NUMINAMATH_CALUDE_shielas_neighbors_l3832_383221

theorem shielas_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) (h1 : total_drawings = 54) (h2 : drawings_per_neighbor = 9) (h3 : drawings_per_neighbor > 0) :
  total_drawings / drawings_per_neighbor = 6 := by
  sorry

end NUMINAMATH_CALUDE_shielas_neighbors_l3832_383221


namespace NUMINAMATH_CALUDE_team_average_score_l3832_383225

theorem team_average_score (lefty_score : ℕ) (righty_score : ℕ) (other_score : ℕ) :
  lefty_score = 20 →
  righty_score = lefty_score / 2 →
  other_score = righty_score * 6 →
  (lefty_score + righty_score + other_score) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_team_average_score_l3832_383225


namespace NUMINAMATH_CALUDE_special_ellipse_property_l3832_383219

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  center : ℝ × ℝ := (0, 0)
  focus_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ
  ecc_eq : eccentricity = Real.sqrt (6/3)
  point_eq : passes_through = (Real.sqrt 5, 0)

/-- Line intersecting the ellipse -/
structure IntersectingLine where
  fixed_point : ℝ × ℝ
  point_eq : fixed_point = (-1, 0)

/-- Intersection points of the line with the ellipse -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  midpoint_x : ℝ
  mid_eq : midpoint_x = -1/2

/-- The theorem statement -/
theorem special_ellipse_property
  (e : SpecialEllipse) (l : IntersectingLine) (p : IntersectionPoints) :
  ∃ (M : ℝ × ℝ), M.1 = -7/3 ∧ M.2 = 0 ∧
  (∀ (A B : ℝ × ℝ), 
    ((A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2)) = 4/9) :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_property_l3832_383219


namespace NUMINAMATH_CALUDE_exists_square_composition_function_l3832_383238

theorem exists_square_composition_function : ∃ F : ℕ → ℕ, ∀ n : ℕ, (F ∘ F) n = n^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_square_composition_function_l3832_383238


namespace NUMINAMATH_CALUDE_unique_divisible_by_eight_l3832_383259

theorem unique_divisible_by_eight : ∃! n : ℕ, 70 < n ∧ n < 80 ∧ n % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_eight_l3832_383259


namespace NUMINAMATH_CALUDE_no_rearranged_powers_of_two_l3832_383295

-- Define a function to check if a number is a power of 2
def isPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

-- Define a function to check if two numbers have the same digits
def haveSameDigits (m n : ℕ) : Prop :=
  ∃ (digits : List ℕ) (perm : List ℕ), 
    digits.length > 0 ∧
    perm.isPerm digits ∧
    m = digits.foldl (fun acc d => acc * 10 + d) 0 ∧
    n = perm.foldl (fun acc d => acc * 10 + d) 0 ∧
    perm.head? ≠ some 0

theorem no_rearranged_powers_of_two :
  ¬∃ (m n : ℕ), m ≠ n ∧ m > 0 ∧ n > 0 ∧ 
  isPowerOfTwo m ∧ isPowerOfTwo n ∧ 
  haveSameDigits m n :=
sorry

end NUMINAMATH_CALUDE_no_rearranged_powers_of_two_l3832_383295


namespace NUMINAMATH_CALUDE_cos_4050_degrees_l3832_383266

theorem cos_4050_degrees : Real.cos (4050 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_4050_degrees_l3832_383266


namespace NUMINAMATH_CALUDE_max_area_difference_l3832_383279

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem stating the maximum area difference between two rectangles -/
theorem max_area_difference :
  ∃ (r1 r2 : Rectangle),
    perimeter r1 = 200 ∧
    perimeter r2 = 200 ∧
    r2.width = 20 ∧
    ∀ (r3 r4 : Rectangle),
      perimeter r3 = 200 →
      perimeter r4 = 200 →
      r4.width = 20 →
      area r1 - area r2 ≥ area r3 - area r4 ∧
      area r1 - area r2 = 900 :=
sorry

end NUMINAMATH_CALUDE_max_area_difference_l3832_383279


namespace NUMINAMATH_CALUDE_walking_distance_approx_2_9_l3832_383237

/-- Represents a journey with cycling and walking portions -/
structure Journey where
  total_time : ℝ
  bike_speed : ℝ
  walk_speed : ℝ
  bike_fraction : ℝ
  walk_fraction : ℝ

/-- Calculates the walking distance for a given journey -/
def walking_distance (j : Journey) : ℝ :=
  let total_distance := (j.bike_speed * j.bike_fraction + j.walk_speed * j.walk_fraction) * j.total_time
  total_distance * j.walk_fraction

/-- Theorem stating that for the given journey parameters, the walking distance is approximately 2.9 km -/
theorem walking_distance_approx_2_9 :
  let j : Journey := {
    total_time := 1,
    bike_speed := 20,
    walk_speed := 4,
    bike_fraction := 2/3,
    walk_fraction := 1/3
  }
  ∃ ε > 0, |walking_distance j - 2.9| < ε :=
sorry

end NUMINAMATH_CALUDE_walking_distance_approx_2_9_l3832_383237


namespace NUMINAMATH_CALUDE_spinner_probability_l3832_383229

theorem spinner_probability (p_A p_B p_C p_D p_E : ℚ) : 
  p_A = 1/5 →
  p_B = 1/3 →
  p_C = p_D →
  p_E = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 7/60 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l3832_383229


namespace NUMINAMATH_CALUDE_price_difference_l3832_383249

/-- Given the total cost of a shirt and sweater, and the price of the shirt,
    calculate the difference in price between the sweater and the shirt. -/
theorem price_difference (total_cost shirt_price : ℚ) 
  (h1 : total_cost = 80.34)
  (h2 : shirt_price = 36.46)
  (h3 : shirt_price < total_cost - shirt_price) :
  total_cost - shirt_price - shirt_price = 7.42 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_l3832_383249


namespace NUMINAMATH_CALUDE_dark_integer_characterization_l3832_383287

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer is of the form a999...999 -/
def isA999Form (n : ℕ+) : Prop := sorry

/-- A positive integer is shiny if it can be written as the sum of two integers
    with the same sum of digits -/
def isShiny (n : ℕ+) : Prop :=
  ∃ a b : ℕ, n = a + b ∧ sumOfDigits ⟨a, sorry⟩ = sumOfDigits ⟨b, sorry⟩

theorem dark_integer_characterization (n : ℕ+) :
  ¬isShiny n ↔ isA999Form n ∧ Odd (sumOfDigits n) := by sorry

end NUMINAMATH_CALUDE_dark_integer_characterization_l3832_383287


namespace NUMINAMATH_CALUDE_triangle_problem_l3832_383227

theorem triangle_problem (AB BC : ℝ) (θ : ℝ) (h t : ℝ) 
  (hyp1 : AB = 7)
  (hyp2 : BC = 25)
  (hyp3 : 100 * Real.sin θ = t)
  (hyp4 : h = AB * Real.sin θ) :
  t = 96 ∧ h = 168 / 25 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3832_383227


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l3832_383264

theorem polar_to_cartesian (ρ θ x y : Real) :
  ρ * Real.cos θ = 1 ↔ x + y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l3832_383264


namespace NUMINAMATH_CALUDE_hawks_touchdowns_l3832_383263

theorem hawks_touchdowns (total_points : ℕ) (points_per_touchdown : ℕ) 
  (h1 : total_points = 21) 
  (h2 : points_per_touchdown = 7) : 
  total_points / points_per_touchdown = 3 := by
  sorry

end NUMINAMATH_CALUDE_hawks_touchdowns_l3832_383263


namespace NUMINAMATH_CALUDE_cookies_per_bag_l3832_383299

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 703) (h2 : num_bags = 37) :
  total_cookies / num_bags = 19 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l3832_383299


namespace NUMINAMATH_CALUDE_square_of_nilpotent_matrix_is_zero_l3832_383222

theorem square_of_nilpotent_matrix_is_zero (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : A ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_of_nilpotent_matrix_is_zero_l3832_383222


namespace NUMINAMATH_CALUDE_total_goals_is_16_l3832_383200

def bruce_goals : ℕ := 4

def michael_goals : ℕ := 3 * bruce_goals

def total_goals : ℕ := bruce_goals + michael_goals

theorem total_goals_is_16 : total_goals = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_goals_is_16_l3832_383200


namespace NUMINAMATH_CALUDE_reverse_digits_when_multiplied_by_nine_l3832_383215

theorem reverse_digits_when_multiplied_by_nine : ∃ n : ℕ, 
  (100000 ≤ n ∧ n < 1000000) ∧  -- six-digit number
  (n * 9 = 
    ((n % 10) * 100000 + 
     ((n / 10) % 10) * 10000 + 
     ((n / 100) % 10) * 1000 + 
     ((n / 1000) % 10) * 100 + 
     ((n / 10000) % 10) * 10 + 
     (n / 100000))) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_reverse_digits_when_multiplied_by_nine_l3832_383215


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l3832_383292

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m > 60 ∧ ¬(m ∣ (n * (n+1) * (n+2) * (n+3) * (n+4))) ∧
  ∀ k : ℤ, k ≤ 60 → k ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l3832_383292


namespace NUMINAMATH_CALUDE_inequality_for_increasing_function_l3832_383206

theorem inequality_for_increasing_function (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum : a + b ≤ 0) : 
  f a + f b ≤ f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_increasing_function_l3832_383206


namespace NUMINAMATH_CALUDE_rectangle_y_value_l3832_383282

/-- A rectangle with vertices at (1, y), (9, y), (1, 5), and (9, 5), where y is positive and the area is 64 square units, has y = 13. -/
theorem rectangle_y_value (y : ℝ) (h1 : y > 0) (h2 : (9 - 1) * (y - 5) = 64) : y = 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l3832_383282


namespace NUMINAMATH_CALUDE_lassis_from_ten_mangoes_l3832_383298

/-- A recipe for making lassis from mangoes -/
structure Recipe where
  mangoes : ℕ
  lassis : ℕ

/-- Given a recipe and a number of mangoes, calculate the number of lassis that can be made -/
def makeLassis (recipe : Recipe) (numMangoes : ℕ) : ℕ :=
  (recipe.lassis * numMangoes) / recipe.mangoes

theorem lassis_from_ten_mangoes (recipe : Recipe) 
  (h1 : recipe.mangoes = 3) 
  (h2 : recipe.lassis = 15) : 
  makeLassis recipe 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_lassis_from_ten_mangoes_l3832_383298


namespace NUMINAMATH_CALUDE_fuel_cost_per_liter_l3832_383272

/-- The cost per liter of fuel given the tank capacity, initial fuel amount, and money spent. -/
theorem fuel_cost_per_liter
  (tank_capacity : ℝ)
  (initial_fuel : ℝ)
  (money_spent : ℝ)
  (h1 : tank_capacity = 150)
  (h2 : initial_fuel = 38)
  (h3 : money_spent = 336)
  : (money_spent / (tank_capacity - initial_fuel)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_fuel_cost_per_liter_l3832_383272


namespace NUMINAMATH_CALUDE_wire_length_proof_l3832_383216

theorem wire_length_proof (total_wires : ℕ) (overall_avg : ℝ) (long_wires : ℕ) (long_avg : ℝ) :
  total_wires = 6 →
  overall_avg = 80 →
  long_wires = 4 →
  long_avg = 85 →
  let short_wires := total_wires - long_wires
  let short_avg := (total_wires * overall_avg - long_wires * long_avg) / short_wires
  short_avg = 70 := by sorry

end NUMINAMATH_CALUDE_wire_length_proof_l3832_383216


namespace NUMINAMATH_CALUDE_volunteer_hours_theorem_l3832_383230

/-- Calculates the total hours volunteered per year given the frequency per month and hours per session -/
def total_volunteer_hours_per_year (sessions_per_month : ℕ) (hours_per_session : ℕ) : ℕ :=
  sessions_per_month * 12 * hours_per_session

/-- Proves that volunteering twice a month for 3 hours each time results in 72 hours per year -/
theorem volunteer_hours_theorem :
  total_volunteer_hours_per_year 2 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_hours_theorem_l3832_383230


namespace NUMINAMATH_CALUDE_factors_of_539_l3832_383228

theorem factors_of_539 : 
  ∃ (p q : Nat), p.Prime ∧ q.Prime ∧ p * q = 539 ∧ p = 13 ∧ q = 41 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_539_l3832_383228


namespace NUMINAMATH_CALUDE_mrs_heine_items_l3832_383248

/-- The number of items Mrs. Heine will buy for her dogs -/
def total_items (num_dogs : ℕ) (biscuits_per_dog : ℕ) (boots_per_set : ℕ) : ℕ :=
  num_dogs * (biscuits_per_dog + boots_per_set)

/-- Proof that Mrs. Heine will buy 18 items -/
theorem mrs_heine_items : total_items 2 5 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_mrs_heine_items_l3832_383248


namespace NUMINAMATH_CALUDE_rancher_problem_l3832_383285

theorem rancher_problem :
  ∃! (b h : ℕ), b > 0 ∧ h > 0 ∧ 30 * b + 32 * h = 1200 ∧ b > h := by
  sorry

end NUMINAMATH_CALUDE_rancher_problem_l3832_383285


namespace NUMINAMATH_CALUDE_total_cranes_eq_262_l3832_383277

/-- The number of cranes Hyerin folds per day -/
def hyerin_cranes_per_day : ℕ := 16

/-- The number of days Hyerin folds cranes -/
def hyerin_days : ℕ := 7

/-- The number of cranes Taeyeong folds per day -/
def taeyeong_cranes_per_day : ℕ := 25

/-- The number of days Taeyeong folds cranes -/
def taeyeong_days : ℕ := 6

/-- The total number of cranes folded by Hyerin and Taeyeong -/
def total_cranes : ℕ := hyerin_cranes_per_day * hyerin_days + taeyeong_cranes_per_day * taeyeong_days

theorem total_cranes_eq_262 : total_cranes = 262 := by
  sorry

end NUMINAMATH_CALUDE_total_cranes_eq_262_l3832_383277


namespace NUMINAMATH_CALUDE_unique_not_in_range_is_30_l3832_383256

/-- Function f with the given properties -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- Theorem stating that 30 is the unique number not in the range of f -/
theorem unique_not_in_range_is_30
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : f a b c d 10 = 10)
  (h2 : f a b c d 50 = 50)
  (h3 : ∀ x ≠ -d/c, f a b c d (f a b c d x) = x) :
  ∃! y, ∀ x, f a b c d x ≠ y ∧ y = 30 :=
sorry

end NUMINAMATH_CALUDE_unique_not_in_range_is_30_l3832_383256


namespace NUMINAMATH_CALUDE_minuend_value_l3832_383267

theorem minuend_value (M S D : ℤ) : 
  M + S + D = 2016 → M - S = D → M = 1008 := by
  sorry

end NUMINAMATH_CALUDE_minuend_value_l3832_383267


namespace NUMINAMATH_CALUDE_equation_solution_l3832_383236

theorem equation_solution (x : ℝ) : 
  (Real.sqrt ((3 + Real.sqrt 5) ^ x)) ^ 2 + (Real.sqrt ((3 - Real.sqrt 5) ^ x)) ^ 2 = 18 ↔ 
  x = 2 ∨ x = -2 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3832_383236


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3832_383271

theorem inequality_system_solution (x : ℝ) :
  (5 * x - 1 > 3 * (x + 1) ∧ x - 1 ≤ 7 - x) → (2 < x ∧ x ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3832_383271


namespace NUMINAMATH_CALUDE_hoseok_number_division_l3832_383294

theorem hoseok_number_division (x : ℤ) (h : x + 8 = 88) : x / 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_number_division_l3832_383294


namespace NUMINAMATH_CALUDE_rectangle_width_equals_square_side_l3832_383226

/-- The width of a rectangle with length 4 cm and area equal to a square with sides 4 cm is 4 cm. -/
theorem rectangle_width_equals_square_side {width : ℝ} (h : width > 0) : 
  4 * width = 4 * 4 → width = 4 := by
  sorry

#check rectangle_width_equals_square_side

end NUMINAMATH_CALUDE_rectangle_width_equals_square_side_l3832_383226


namespace NUMINAMATH_CALUDE_vegetable_production_equation_l3832_383207

def vegetable_growth_rate (initial_production final_production : ℝ) (years : ℕ) (x : ℝ) : Prop :=
  initial_production * (1 + x) ^ years = final_production

theorem vegetable_production_equation :
  ∃ x : ℝ, vegetable_growth_rate 800 968 2 x :=
sorry

end NUMINAMATH_CALUDE_vegetable_production_equation_l3832_383207


namespace NUMINAMATH_CALUDE_chloe_boxes_l3832_383208

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 2

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 6

/-- The total number of pieces of winter clothing Chloe has -/
def total_pieces : ℕ := 32

/-- The number of boxes Chloe found -/
def boxes : ℕ := total_pieces / (scarves_per_box + mittens_per_box)

theorem chloe_boxes : boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_chloe_boxes_l3832_383208


namespace NUMINAMATH_CALUDE_ellipse_trajectory_and_minimum_l3832_383270

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 + y^2/4 = 1 ∧ x > 0 ∧ y > 0

-- Define the tangent line
def tangent_line (x₀ y₀ x y : ℝ) : Prop :=
  y = -4*x₀/y₀ * (x - x₀) + y₀

-- Define point M
def point_M (x y : ℝ) : Prop :=
  ∃ x₀ y₀, ellipse x₀ y₀ ∧
  ∃ xA yB, tangent_line x₀ y₀ xA 0 ∧ tangent_line x₀ y₀ 0 yB ∧
  x = xA ∧ y = yB

theorem ellipse_trajectory_and_minimum (x y : ℝ) :
  point_M x y →
  (1/x^2 + 4/y^2 = 1 ∧ x > 1 ∧ y > 2) ∧
  (∀ x' y', point_M x' y' → x'^2 + y'^2 ≥ 9) ∧
  (∃ x₀ y₀, point_M x₀ y₀ ∧ x₀^2 + y₀^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_trajectory_and_minimum_l3832_383270


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l3832_383290

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) :
  x^2 + y^2 ≥ 229 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l3832_383290


namespace NUMINAMATH_CALUDE_janet_return_time_l3832_383239

/-- Represents the number of blocks Janet walks in each direction --/
structure WalkingDistance where
  north : ℕ
  west : ℕ
  south : ℕ
  east : ℕ

/-- Calculates the time taken to walk a given distance at a given speed --/
def timeToWalk (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

/-- Janet's walking pattern and speed --/
def janet : WalkingDistance × ℕ :=
  ({ north := 3
   , west := 3 * 7
   , south := 3
   , east := 3 * 2
   }, 2)

/-- Theorem: Janet takes 9 minutes to return home --/
theorem janet_return_time : 
  let (walk, speed) := janet
  timeToWalk (walk.south + (walk.west - walk.east)) speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_janet_return_time_l3832_383239


namespace NUMINAMATH_CALUDE_tourist_contact_probability_l3832_383217

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 42

theorem tourist_contact_probability (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  contact_probability p =
    1 - (1 - p) ^ (6 * 7) :=
by sorry

end NUMINAMATH_CALUDE_tourist_contact_probability_l3832_383217


namespace NUMINAMATH_CALUDE_problem_solution_l3832_383210

theorem problem_solution (x : ℚ) : (1/2 * (12*x + 3) = 3*x + 2) → x = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3832_383210


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l3832_383293

theorem similar_triangles_leg_sum (a b c d : ℕ) : 
  a * b = 18 →  -- area of smaller triangle is 9
  a^2 + b^2 = 25 →  -- hypotenuse of smaller triangle is 5
  a ≠ 3 ∨ b ≠ 4 →  -- not a 3-4-5 triangle
  c * d = 450 →  -- area of larger triangle is 225
  (c : ℝ) / a = (d : ℝ) / b →  -- triangles are similar
  (c + d : ℝ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l3832_383293


namespace NUMINAMATH_CALUDE_sin_phi_value_l3832_383255

theorem sin_phi_value (φ : ℝ) : 
  (∀ x, 2 * Real.sin x + Real.cos x = 2 * Real.sin (x - φ) - Real.cos (x - φ)) →
  Real.sin φ = 4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_phi_value_l3832_383255


namespace NUMINAMATH_CALUDE_dress_designs_count_l3832_383213

/-- The number of color choices available for a dress design. -/
def num_colors : ℕ := 5

/-- The number of pattern choices available for a dress design. -/
def num_patterns : ℕ := 4

/-- The number of accessory choices available for a dress design. -/
def num_accessories : ℕ := 2

/-- The total number of possible dress designs. -/
def total_designs : ℕ := num_colors * num_patterns * num_accessories

/-- Theorem stating that the total number of possible dress designs is 40. -/
theorem dress_designs_count : total_designs = 40 := by
  sorry

end NUMINAMATH_CALUDE_dress_designs_count_l3832_383213


namespace NUMINAMATH_CALUDE_triangle_area_l3832_383223

theorem triangle_area (base height : ℝ) (h1 : base = 8.4) (h2 : height = 5.8) :
  (base * height) / 2 = 24.36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3832_383223


namespace NUMINAMATH_CALUDE_bookshop_inventory_problem_l3832_383281

/-- Represents the bookshop inventory problem -/
theorem bookshop_inventory_problem (S : ℕ) : 
  743 - (S + 128 + 2*S + (128 + 34)) + 160 = 502 → S = 37 := by
  sorry

end NUMINAMATH_CALUDE_bookshop_inventory_problem_l3832_383281


namespace NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l3832_383247

/-- Given two lines in the xy-plane, this theorem proves that if they are perpendicular,
    then the coefficient 'a' in the first line's equation must equal 2/3. -/
theorem perpendicular_lines_coefficient (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 2 = 0 → 2 * x + 3 * y + 1 = 0 → 
   ((-1 : ℝ) / a) * (-2 / 3) = -1) →
  a = 2 / 3 := by
sorry


end NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l3832_383247


namespace NUMINAMATH_CALUDE_math_only_count_l3832_383231

def brainiac_survey (total : ℕ) (rebus math logic : ℕ) 
  (rebus_math rebus_logic math_logic all_three neither : ℕ) : Prop :=
  total = 500 ∧
  rebus = 2 * math ∧
  logic = math ∧
  rebus_math = 72 ∧
  rebus_logic = 40 ∧
  math_logic = 36 ∧
  all_three = 10 ∧
  neither = 20

theorem math_only_count 
  (total rebus math logic rebus_math rebus_logic math_logic all_three neither : ℕ) :
  brainiac_survey total rebus math logic rebus_math rebus_logic math_logic all_three neither →
  math - rebus_math - math_logic + all_three = 54 :=
by sorry

end NUMINAMATH_CALUDE_math_only_count_l3832_383231


namespace NUMINAMATH_CALUDE_dvd_sales_proof_l3832_383241

theorem dvd_sales_proof (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * cd →
  dvd + cd = 273 →
  dvd = 168 := by
sorry

end NUMINAMATH_CALUDE_dvd_sales_proof_l3832_383241


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l3832_383234

theorem matching_shoes_probability (n : ℕ) (h : n = 12) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes.choose 2 : ℚ)
  let matching_pairs := n
  matching_pairs / total_combinations = 1 / 46 :=
by sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l3832_383234


namespace NUMINAMATH_CALUDE_constant_term_value_l3832_383212

theorem constant_term_value (x y : ℝ) (C : ℝ) : 
  5 * x + y = C →
  x + 3 * y = 1 →
  3 * x + 2 * y = 10 →
  C = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l3832_383212


namespace NUMINAMATH_CALUDE_square_of_sqrt_17_l3832_383220

theorem square_of_sqrt_17 : (Real.sqrt 17) ^ 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sqrt_17_l3832_383220


namespace NUMINAMATH_CALUDE_festival_attendance_theorem_l3832_383204

/-- Represents the attendance for each day of a four-day music festival --/
structure FestivalAttendance where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Calculates the total attendance for all four days --/
def totalAttendance (attendance : FestivalAttendance) : ℕ :=
  attendance.day1 + attendance.day2 + attendance.day3 + attendance.day4

/-- Theorem stating that the total attendance for the festival is 3600 --/
theorem festival_attendance_theorem (attendance : FestivalAttendance) :
  (attendance.day2 = attendance.day1 / 2) →
  (attendance.day3 = attendance.day1 * 3) →
  (attendance.day4 = attendance.day2 * 2) →
  (totalAttendance attendance = 3600) :=
by
  sorry

#check festival_attendance_theorem

end NUMINAMATH_CALUDE_festival_attendance_theorem_l3832_383204


namespace NUMINAMATH_CALUDE_rectangle_area_l3832_383243

theorem rectangle_area (b : ℝ) : 
  let square_area : ℝ := 1296
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := circle_radius / 6
  let rectangle_breadth : ℝ := b
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area = 6 * b := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3832_383243


namespace NUMINAMATH_CALUDE_percentage_of_women_in_study_group_l3832_383274

theorem percentage_of_women_in_study_group :
  let percentage_women_lawyers : ℝ := 0.4
  let prob_woman_lawyer : ℝ := 0.32
  let percentage_women : ℝ := prob_woman_lawyer / percentage_women_lawyers
  percentage_women = 0.8 := by sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_study_group_l3832_383274


namespace NUMINAMATH_CALUDE_one_third_complex_point_l3832_383254

theorem one_third_complex_point (z₁ z₂ z : ℂ) :
  z₁ = -5 + 6*I →
  z₂ = 7 - 4*I →
  z = (1 - 1/3) * z₁ + 1/3 * z₂ →
  z = -1 + 8/3 * I :=
by sorry

end NUMINAMATH_CALUDE_one_third_complex_point_l3832_383254


namespace NUMINAMATH_CALUDE_greatest_x_value_l3832_383251

theorem greatest_x_value (x : ℝ) : 
  x ≠ 9 → 
  (x^2 - x - 90) / (x - 9) = 2 / (x + 7) → 
  x ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3832_383251


namespace NUMINAMATH_CALUDE_unplanted_field_fraction_l3832_383289

theorem unplanted_field_fraction (a b c x : ℝ) : 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → x = 5/3 → 
  x^2 / (a * b / 2) = 5/54 := by sorry

end NUMINAMATH_CALUDE_unplanted_field_fraction_l3832_383289


namespace NUMINAMATH_CALUDE_count_numbers_with_digit_product_180_l3832_383284

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 9

def is_five_digit_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.prod

def count_valid_numbers : ℕ := sorry

theorem count_numbers_with_digit_product_180 :
  count_valid_numbers = 360 := by sorry

end NUMINAMATH_CALUDE_count_numbers_with_digit_product_180_l3832_383284


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3832_383240

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 80 → 
  b = 150 → 
  c^2 = a^2 + b^2 → 
  c = 170 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3832_383240


namespace NUMINAMATH_CALUDE_exam_max_marks_l3832_383242

theorem exam_max_marks (victor_score : ℝ) (victor_percentage : ℝ) (max_marks : ℝ) : 
  victor_score = 368 ∧ 
  victor_percentage = 0.92 ∧ 
  victor_score = victor_percentage * max_marks → 
  max_marks = 400 := by
sorry

end NUMINAMATH_CALUDE_exam_max_marks_l3832_383242


namespace NUMINAMATH_CALUDE_positive_combination_l3832_383233

theorem positive_combination (x y : ℝ) (h1 : x + y > 0) (h2 : 4 * x + y > 0) : 
  8 * x + 5 * y > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_combination_l3832_383233


namespace NUMINAMATH_CALUDE_cleaning_assignment_cases_l3832_383278

def number_of_people : ℕ := 6
def people_for_floor : ℕ := 2
def people_for_window : ℕ := 1

theorem cleaning_assignment_cases :
  (Nat.choose (number_of_people - 1) (people_for_floor - 1)) *
  (Nat.choose (number_of_people - people_for_floor) people_for_window) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_assignment_cases_l3832_383278
