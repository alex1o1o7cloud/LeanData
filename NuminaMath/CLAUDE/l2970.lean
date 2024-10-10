import Mathlib

namespace golden_hyperbola_eccentricity_l2970_297024

theorem golden_hyperbola_eccentricity :
  ∀ e : ℝ, e > 1 → e^2 - e = 1 → e = (Real.sqrt 5 + 1) / 2 := by
  sorry

end golden_hyperbola_eccentricity_l2970_297024


namespace book_distribution_ways_l2970_297048

theorem book_distribution_ways (n m : ℕ) (h1 : n = 3) (h2 : m = 2) : 
  n * (n - 1) = 6 := by
  sorry

end book_distribution_ways_l2970_297048


namespace books_sum_is_41_l2970_297047

/-- The number of books Keith has -/
def keith_books : ℕ := 20

/-- The number of books Jason has -/
def jason_books : ℕ := 21

/-- The total number of books Keith and Jason have together -/
def total_books : ℕ := keith_books + jason_books

theorem books_sum_is_41 : total_books = 41 := by
  sorry

end books_sum_is_41_l2970_297047


namespace quadratic_inequality_solution_l2970_297003

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 + b*x - a < 0 ↔ -2 < x ∧ x < 3) → a + b = 5 := by
  sorry

end quadratic_inequality_solution_l2970_297003


namespace equation_solution_l2970_297088

theorem equation_solution (x : ℝ) :
  (Real.sqrt (2 * x + 7)) / (Real.sqrt (8 * x + 10)) = 2 / Real.sqrt 5 →
  x = -5 / 22 := by
  sorry

end equation_solution_l2970_297088


namespace miles_driven_with_budget_l2970_297063

-- Define the given conditions
def miles_per_gallon : ℝ := 32
def cost_per_gallon : ℝ := 4
def budget : ℝ := 20

-- Define the theorem
theorem miles_driven_with_budget :
  (budget / cost_per_gallon) * miles_per_gallon = 160 := by
  sorry

end miles_driven_with_budget_l2970_297063


namespace total_cloud_count_l2970_297014

def cloud_count (carson_count : ℕ) (brother_multiplier : ℕ) (sister_divisor : ℕ) : ℕ :=
  carson_count + (carson_count * brother_multiplier) + (carson_count / sister_divisor)

theorem total_cloud_count :
  cloud_count 12 5 2 = 78 :=
by sorry

end total_cloud_count_l2970_297014


namespace regular_polygon_sides_l2970_297036

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (∀ angle : ℝ, angle = 150 → (n : ℝ) * angle = 180 * ((n : ℝ) - 2)) → n = 12 := by
  sorry

end regular_polygon_sides_l2970_297036


namespace total_fruits_eaten_l2970_297075

theorem total_fruits_eaten (sophie_oranges_per_day : ℕ) (hannah_grapes_per_day : ℕ) (days : ℕ) :
  sophie_oranges_per_day = 20 →
  hannah_grapes_per_day = 40 →
  days = 30 →
  sophie_oranges_per_day * days + hannah_grapes_per_day * days = 1800 :=
by sorry

end total_fruits_eaten_l2970_297075


namespace square_root_equals_seven_l2970_297052

theorem square_root_equals_seven (m : ℝ) : (∀ x : ℝ, x ^ 2 = m ↔ x = 7 ∨ x = -7) → m = 49 := by
  sorry

end square_root_equals_seven_l2970_297052


namespace rose_difference_l2970_297005

theorem rose_difference (total : ℕ) (red_fraction : ℚ) (yellow_fraction : ℚ)
  (h_total : total = 48)
  (h_red : red_fraction = 3/8)
  (h_yellow : yellow_fraction = 5/16) :
  ↑total * red_fraction - ↑total * yellow_fraction = 3 :=
by sorry

end rose_difference_l2970_297005


namespace optimal_inequality_l2970_297095

theorem optimal_inequality (a b c d : ℝ) 
  (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) (hd : d ≥ -1) :
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3/4) * (a + b + c + d) ∧ 
  ∀ k > 3/4, ∃ x y z w : ℝ, x ≥ -1 ∧ y ≥ -1 ∧ z ≥ -1 ∧ w ≥ -1 ∧ 
    x^3 + y^3 + z^3 + w^3 + 1 < k * (x + y + z + w) :=
by sorry

end optimal_inequality_l2970_297095


namespace inequalities_proof_l2970_297086

theorem inequalities_proof :
  (∀ x : ℝ, 3*x - 2*x^2 + 2 ≥ 0 ↔ 1/2 ≤ x ∧ x ≤ 2) ∧
  (∀ x : ℝ, 4 < |2*x - 3| ∧ |2*x - 3| ≤ 7 ↔ (5 ≥ x ∧ x > 7/2) ∨ (-2 ≤ x ∧ x < -1/2)) ∧
  (∀ x : ℝ, |x - 8| - |x - 4| > 2 ↔ x < 5) :=
by sorry

end inequalities_proof_l2970_297086


namespace sin_transform_l2970_297039

/-- Given a function f(x) = sin(x - π/3), prove that after stretching the x-coordinates
    to twice their original length and shifting the resulting graph to the left by π/3 units,
    the resulting function is g(x) = sin(1/2x - π/6) -/
theorem sin_transform (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.sin (x - π/3)
  let g : ℝ → ℝ := fun x => Real.sin (x/2 - π/6)
  let h : ℝ → ℝ := fun x => f (x/2 + π/3)
  h = g := by sorry

end sin_transform_l2970_297039


namespace photo_arrangement_count_l2970_297060

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items in n positions --/
def arrange (n k : ℕ) : ℕ := sorry

theorem photo_arrangement_count :
  let total_students : ℕ := 12
  let initial_front_row : ℕ := 4
  let initial_back_row : ℕ := 8
  let students_to_move : ℕ := 2
  let final_front_row : ℕ := initial_front_row + students_to_move
  choose initial_back_row students_to_move * arrange final_front_row students_to_move =
    choose 8 2 * arrange 6 2 := by sorry

end photo_arrangement_count_l2970_297060


namespace safe_plucking_percentage_is_correct_l2970_297050

/-- The number of tail feathers each flamingo has -/
def feathers_per_flamingo : ℕ := 20

/-- The number of boas Milly needs to make -/
def number_of_boas : ℕ := 12

/-- The number of feathers needed for each boa -/
def feathers_per_boa : ℕ := 200

/-- The number of flamingoes Milly needs to harvest -/
def flamingoes_to_harvest : ℕ := 480

/-- The percentage of tail feathers Milly can safely pluck from each flamingo -/
def safe_plucking_percentage : ℚ := 25 / 100

theorem safe_plucking_percentage_is_correct :
  safe_plucking_percentage = 
    (number_of_boas * feathers_per_boa) / 
    (flamingoes_to_harvest * feathers_per_flamingo) := by
  sorry

end safe_plucking_percentage_is_correct_l2970_297050


namespace complex_point_l2970_297056

theorem complex_point (i : ℂ) (h : i ^ 2 = -1) :
  let z : ℂ := i + 2 * i^2 + 3 * i^3
  (z.re = -2) ∧ (z.im = -2) := by sorry

end complex_point_l2970_297056


namespace bernardo_wins_smallest_number_l2970_297067

theorem bernardo_wins_smallest_number : ∃ M : ℕ, 
  M ≤ 799 ∧ 
  (∀ k : ℕ, k < M → 
    (2 * k ≤ 800 ∧ 
     2 * k + 70 ≤ 800 ∧ 
     4 * k + 140 ≤ 800 ∧ 
     4 * k + 210 ≤ 800 ∧ 
     8 * k + 420 ≤ 800) → 
    8 * k + 490 ≤ 800) ∧
  2 * M ≤ 800 ∧ 
  2 * M + 70 ≤ 800 ∧ 
  4 * M + 140 ≤ 800 ∧ 
  4 * M + 210 ≤ 800 ∧ 
  8 * M + 420 ≤ 800 ∧ 
  8 * M + 490 > 800 ∧
  M = 37 :=
by sorry

end bernardo_wins_smallest_number_l2970_297067


namespace pentagon_vertex_c_y_coordinate_l2970_297066

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The area of a pentagon -/
def area (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a vertical line of symmetry -/
def has_vertical_symmetry (p : Pentagon) : Prop := sorry

/-- The main theorem -/
theorem pentagon_vertex_c_y_coordinate
  (p : Pentagon)
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 5))
  (h3 : ∃ y, p.C = (2.5, y))
  (h4 : p.D = (5, 5))
  (h5 : p.E = (5, 0))
  (h6 : has_vertical_symmetry p)
  (h7 : area p = 50)
  : p.C.2 = 15 := by sorry


end pentagon_vertex_c_y_coordinate_l2970_297066


namespace school_survey_sample_size_l2970_297070

/-- Represents a survey conducted in a school -/
structure SchoolSurvey where
  total_students : ℕ
  selected_students : ℕ

/-- The sample size of a school survey is the number of selected students -/
def sample_size (survey : SchoolSurvey) : ℕ := survey.selected_students

/-- Theorem: For a school with 3600 students and 200 randomly selected for a survey,
    the sample size is 200 -/
theorem school_survey_sample_size :
  let survey := SchoolSurvey.mk 3600 200
  sample_size survey = 200 := by
  sorry

end school_survey_sample_size_l2970_297070


namespace marge_garden_weeds_l2970_297074

def garden_problem (total_seeds planted_seeds non_growing_seeds eaten_fraction 
                    strangled_fraction kept_weeds final_plants : ℕ) : Prop :=
  let grown_plants := planted_seeds - non_growing_seeds
  let eaten_plants := (grown_plants / 3 : ℕ)
  let uneaten_plants := grown_plants - eaten_plants
  let strangled_plants := (uneaten_plants / 3 : ℕ)
  let healthy_plants := uneaten_plants - strangled_plants
  healthy_plants + kept_weeds = final_plants

theorem marge_garden_weeds : 
  ∃ (pulled_weeds : ℕ), 
    garden_problem 23 23 5 (1/3) (1/3) 1 9 ∧ 
    pulled_weeds = 3 := by sorry

end marge_garden_weeds_l2970_297074


namespace dodecagon_diagonals_l2970_297080

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A regular dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l2970_297080


namespace max_candies_eaten_l2970_297090

/-- Represents the state of the board and the total candies eaten -/
structure BoardState where
  numbers : List Nat
  candies : Nat

/-- The process of combining two numbers on the board -/
def combineNumbers (state : BoardState) : BoardState :=
  match state.numbers with
  | x :: y :: rest => {
      numbers := (x + y) :: rest,
      candies := state.candies + x * y
    }
  | _ => state

/-- Theorem stating the maximum number of candies that can be eaten -/
theorem max_candies_eaten :
  ∃ (final : BoardState),
    (combineNumbers^[48] {numbers := List.replicate 49 1, candies := 0}) = final ∧
    final.candies = 1176 := by
  sorry

#check max_candies_eaten

end max_candies_eaten_l2970_297090


namespace least_tiles_required_l2970_297028

/-- The length of the room in centimeters -/
def room_length : ℕ := 1517

/-- The breadth of the room in centimeters -/
def room_breadth : ℕ := 902

/-- The greatest common divisor of the room length and breadth -/
def tile_side : ℕ := Nat.gcd room_length room_breadth

/-- The area of the room in square centimeters -/
def room_area : ℕ := room_length * room_breadth

/-- The area of a single tile in square centimeters -/
def tile_area : ℕ := tile_side * tile_side

/-- The number of tiles required to pave the room -/
def num_tiles : ℕ := (room_area + tile_area - 1) / tile_area

theorem least_tiles_required :
  num_tiles = 814 :=
sorry

end least_tiles_required_l2970_297028


namespace no_multiple_with_smaller_digit_sum_l2970_297010

/-- The number composed of m digits all being ones -/
def ones_number (m : ℕ) : ℕ :=
  (10^m - 1) / 9

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number composed of m ones has no multiple with digit sum less than m -/
theorem no_multiple_with_smaller_digit_sum (m : ℕ) :
  ∀ k : ℕ, k > 0 → digit_sum (k * ones_number m) ≥ m :=
sorry

end no_multiple_with_smaller_digit_sum_l2970_297010


namespace binary_to_decimal_11001001_l2970_297017

/-- Converts a list of binary digits to its decimal equivalent -/
def binaryToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 2^(digits.length - 1 - i)) 0

/-- The binary representation of the number -/
def binaryNumber : List Nat := [1, 1, 0, 0, 1, 0, 0, 1]

theorem binary_to_decimal_11001001 :
  binaryToDecimal binaryNumber = 201 := by
  sorry

end binary_to_decimal_11001001_l2970_297017


namespace largest_perfect_square_factor_of_9240_l2970_297081

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem largest_perfect_square_factor_of_9240 :
  ∃ (n : ℕ), is_perfect_square n ∧ 
             is_factor n 9240 ∧ 
             (∀ m : ℕ, is_perfect_square m → is_factor m 9240 → m ≤ n) ∧
             n = 36 := by sorry

end largest_perfect_square_factor_of_9240_l2970_297081


namespace candy_bowl_problem_l2970_297015

theorem candy_bowl_problem (talitha_pieces solomon_pieces remaining_pieces : ℕ) 
  (h1 : talitha_pieces = 108)
  (h2 : solomon_pieces = 153)
  (h3 : remaining_pieces = 88) :
  talitha_pieces + solomon_pieces + remaining_pieces = 349 := by
  sorry

end candy_bowl_problem_l2970_297015


namespace intersection_sum_l2970_297054

/-- Given two functions f and g where
    f(x) = -|x - a| + b
    g(x) = |x - c| + d
    If f and g intersect at points (2, 5) and (8, 3), then a + c = 10 -/
theorem intersection_sum (a b c d : ℝ) : 
  (∀ x, -|x - a| + b = |x - c| + d → x = 2 ∨ x = 8) →
  -|2 - a| + b = 5 →
  -|8 - a| + b = 3 →
  |2 - c| + d = 5 →
  |8 - c| + d = 3 →
  a + c = 10 := by
  sorry

end intersection_sum_l2970_297054


namespace cafeteria_optimal_location_l2970_297029

-- Define the offices and their employee counts
structure Office where
  location : ℝ × ℝ
  employees : ℕ

-- Define the triangle formed by the offices
def office_triangle (A B C : Office) : Prop :=
  A.location ≠ B.location ∧ B.location ≠ C.location ∧ C.location ≠ A.location

-- Define the total distance function
def total_distance (cafeteria : ℝ × ℝ) (A B C : Office) : ℝ :=
  A.employees * dist cafeteria A.location +
  B.employees * dist cafeteria B.location +
  C.employees * dist cafeteria C.location

-- State the theorem
theorem cafeteria_optimal_location (A B C : Office) 
  (h_triangle : office_triangle A B C)
  (h_employees : A.employees = 10 ∧ B.employees = 20 ∧ C.employees = 30) :
  ∀ cafeteria : ℝ × ℝ, total_distance C.location A B C ≤ total_distance cafeteria A B C :=
sorry

end cafeteria_optimal_location_l2970_297029


namespace perpendicular_to_parallel_implies_perpendicular_l2970_297033

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_parallel_implies_perpendicular
  (α β : Plane) (m n : Line)
  (h1 : α ≠ β)
  (h2 : m ≠ n)
  (h3 : perpendicular m β)
  (h4 : parallel n β) :
  perpendicular_lines m n :=
sorry

end perpendicular_to_parallel_implies_perpendicular_l2970_297033


namespace savings_in_cents_l2970_297041

-- Define the prices and quantities for each store
def store1_price : ℚ := 3
def store1_quantity : ℕ := 6
def store2_price : ℚ := 4
def store2_quantity : ℕ := 10

-- Define the price per apple for each store
def price_per_apple_store1 : ℚ := store1_price / store1_quantity
def price_per_apple_store2 : ℚ := store2_price / store2_quantity

-- Define the savings per apple in dollars
def savings_per_apple : ℚ := price_per_apple_store1 - price_per_apple_store2

-- Theorem to prove
theorem savings_in_cents : savings_per_apple * 100 = 10 := by
  sorry

end savings_in_cents_l2970_297041


namespace externally_tangent_circles_l2970_297018

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The equation of circle C₁ -/
def C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- The equation of circle C₂ -/
def C2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + m = 0

theorem externally_tangent_circles (m : ℝ) :
  (∃ c1 : ℝ × ℝ, ∃ r1 : ℝ, ∀ x y : ℝ, C1 x y ↔ (x - c1.1)^2 + (y - c1.2)^2 = r1^2) →
  (∃ c2 : ℝ × ℝ, ∃ r2 : ℝ, ∀ x y : ℝ, C2 x y m ↔ (x - c2.1)^2 + (y - c2.2)^2 = r2^2) →
  (∃ c1 c2 : ℝ × ℝ, ∃ r1 r2 : ℝ, externally_tangent c1 c2 r1 r2) →
  m = -3 := by
  sorry

end externally_tangent_circles_l2970_297018


namespace pizza_fraction_l2970_297092

theorem pizza_fraction (total_slices : ℕ) (whole_slice : ℚ) (shared_slice : ℚ) : 
  total_slices = 16 → whole_slice = 1 → shared_slice = 1/3 → 
  whole_slice / total_slices + (shared_slice / total_slices) = 1/12 := by
sorry

end pizza_fraction_l2970_297092


namespace sandwich_combinations_l2970_297096

def lunch_meat : ℕ := 12
def cheese : ℕ := 8

theorem sandwich_combinations : 
  (lunch_meat.choose 1) * (cheese.choose 2) = 336 := by
  sorry

end sandwich_combinations_l2970_297096


namespace polygon_sides_l2970_297059

theorem polygon_sides (sum_known_angles : ℕ) (angle_a angle_b angle_c : ℕ) :
  sum_known_angles = 3780 →
  angle_a = 3 * angle_c →
  angle_b = 3 * angle_c →
  ∃ (n : ℕ), n = 23 ∧ sum_known_angles = 180 * (n - 2) :=
by sorry

end polygon_sides_l2970_297059


namespace total_path_satisfies_conditions_l2970_297004

/-- The total length of Gyeongyeon's travel path --/
def total_path : ℝ := 2200

/-- Gyeongyeon's travel segments --/
structure TravelSegments where
  bicycle : ℝ
  first_walk : ℝ
  bus : ℝ
  final_walk : ℝ

/-- Conditions of Gyeongyeon's travel --/
def travel_conditions (d : ℝ) : Prop :=
  ∃ (segments : TravelSegments),
    segments.bicycle = d / 2 ∧
    segments.first_walk = 300 ∧
    segments.bus = (d / 2 - 300) / 2 ∧
    segments.final_walk = 400 ∧
    segments.bicycle + segments.first_walk + segments.bus + segments.final_walk = d

/-- Theorem stating that the total path length satisfies the travel conditions --/
theorem total_path_satisfies_conditions : travel_conditions total_path := by
  sorry


end total_path_satisfies_conditions_l2970_297004


namespace digit_1500_is_1_l2970_297046

/-- The fraction we're considering -/
def f : ℚ := 7/22

/-- The length of the repeating cycle in the decimal expansion of f -/
def cycle_length : ℕ := 6

/-- The position of the digit we're looking for -/
def target_position : ℕ := 1500

/-- The function that returns the nth digit after the decimal point
    in the decimal expansion of f -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_1500_is_1 : nth_digit target_position = 1 := by sorry

end digit_1500_is_1_l2970_297046


namespace ab_value_l2970_297000

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 := by
  sorry

end ab_value_l2970_297000


namespace star_equation_solution_l2970_297093

/-- Definition of the star operation -/
def star (a b : ℕ) : ℕ := a^b - a*b + 5

/-- Theorem stating that if a^b - ab + 5 = 13 for a ≥ 2 and b ≥ 3, then a + b = 6 -/
theorem star_equation_solution (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 3) (h_eq : star a b = 13) :
  a + b = 6 := by
  sorry

end star_equation_solution_l2970_297093


namespace max_angle_MPN_x_coordinate_l2970_297089

/-- The x-coordinate of point P when angle MPN is maximum -/
def max_angle_x_coordinate : ℝ := 1

/-- Point M with coordinates (-1, 2) -/
def M : ℝ × ℝ := (-1, 2)

/-- Point N with coordinates (1, 4) -/
def N : ℝ × ℝ := (1, 4)

/-- Point P moves on the positive half of the x-axis -/
def P (x : ℝ) : ℝ × ℝ := (x, 0)

/-- The angle MPN as a function of the x-coordinate of P -/
noncomputable def angle_MPN (x : ℝ) : ℝ := sorry

theorem max_angle_MPN_x_coordinate :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 → angle_MPN y ≤ angle_MPN x) ∧
  x = max_angle_x_coordinate := by sorry

end max_angle_MPN_x_coordinate_l2970_297089


namespace min_area_triangle_l2970_297058

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := (1 / a) * Real.log x

theorem min_area_triangle (a : ℝ) (h : a ≠ 0) :
  let P := (0, a)
  let Q := (Real.exp (a^2), a)
  let R := (0, a - 1/a)
  let area := (Real.exp (a^2)) / (2 * |a|)
  ∃ (min_area : ℝ), min_area = Real.exp (1/2) / Real.sqrt 2 ∧
    ∀ a' : ℝ, a' ≠ 0 → area ≥ min_area := by sorry

end min_area_triangle_l2970_297058


namespace real_y_condition_l2970_297021

theorem real_y_condition (x : ℝ) :
  (∃ y : ℝ, 4 * y^2 - 2 * x * y + 2 * x + 9 = 0) ↔ (x ≤ -3 ∨ x ≥ 12) := by
  sorry

end real_y_condition_l2970_297021


namespace multiply_and_add_equality_l2970_297019

theorem multiply_and_add_equality : 45 * 28 + 72 * 45 = 4500 := by
  sorry

end multiply_and_add_equality_l2970_297019


namespace ducks_in_marsh_l2970_297061

theorem ducks_in_marsh (total_birds geese : ℕ) (h1 : total_birds = 95) (h2 : geese = 58) :
  total_birds - geese = 37 := by
  sorry

end ducks_in_marsh_l2970_297061


namespace smallest_common_nondivisor_l2970_297055

theorem smallest_common_nondivisor : 
  ∃ (a : ℕ), a > 0 ∧ 
  (∀ (k : ℕ), 0 < k ∧ k < a → (Nat.gcd k 77 = 1 ∨ Nat.gcd k 66 = 1)) ∧ 
  Nat.gcd a 77 > 1 ∧ Nat.gcd a 66 > 1 ∧ 
  a = 11 :=
sorry

end smallest_common_nondivisor_l2970_297055


namespace cubic_roots_from_quadratic_l2970_297084

theorem cubic_roots_from_quadratic (b c : ℝ) :
  let x₁ := b + c
  let x₂ := b - c
  (∀ x, x^2 - 2*b*x + b^2 - c^2 = 0 ↔ x = x₁ ∨ x = x₂) →
  (∀ x, x^2 - 2*b*(b^2 + 3*c^2)*x + (b^2 - c^2)^3 = 0 ↔ x = x₁^3 ∨ x = x₂^3) :=
by sorry

end cubic_roots_from_quadratic_l2970_297084


namespace probability_at_2_3_after_5_moves_l2970_297078

/-- Represents the probability of a particle reaching a specific point after a number of moves -/
def particle_probability (x y n : ℕ) : ℚ :=
  if x + y = n then
    (n.choose y : ℚ) * (1/2)^n
  else
    0

/-- Theorem stating the probability of reaching (2,3) after 5 moves -/
theorem probability_at_2_3_after_5_moves :
  particle_probability 2 3 5 = 5/16 := by
  sorry

end probability_at_2_3_after_5_moves_l2970_297078


namespace arithmetic_sequence_a2_l2970_297020

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a2 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 3 + a 11 = 50) 
  (h_a4 : a 4 = 13) : 
  a 2 = 5 := by
sorry

end arithmetic_sequence_a2_l2970_297020


namespace current_speed_l2970_297034

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 25)
  (h2 : speed_against_current = 20) : 
  ∃ (mans_speed current_speed : ℝ),
    speed_with_current = mans_speed + current_speed ∧
    speed_against_current = mans_speed - current_speed ∧
    current_speed = 2.5 := by
sorry

end current_speed_l2970_297034


namespace polynomial_simplification_l2970_297006

theorem polynomial_simplification (x : ℝ) :
  (2 * x^2 + 5 * x - 7) - (x^2 + 9 * x - 3) = x^2 - 4 * x - 4 := by
  sorry

end polynomial_simplification_l2970_297006


namespace modular_inverse_of_3_mod_257_l2970_297022

theorem modular_inverse_of_3_mod_257 : ∃ x : ℕ, x < 257 ∧ (3 * x) % 257 = 1 :=
  by
    use 86
    sorry

end modular_inverse_of_3_mod_257_l2970_297022


namespace parabola_y_axis_intersection_l2970_297082

/-- The parabola defined by y = 2x^2 + 3 intersects the y-axis at the point (0, 3) -/
theorem parabola_y_axis_intersection :
  let f : ℝ → ℝ := fun x ↦ 2 * x^2 + 3
  (0, f 0) = (0, 3) := by sorry

end parabola_y_axis_intersection_l2970_297082


namespace function_property_implications_l2970_297032

def FunctionProperty (f : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, f x = f (x^2 + x + 1)

theorem function_property_implications
  (f : ℤ → ℤ) (h : FunctionProperty f) :
  ((∀ x : ℤ, f x = f (-x)) → (∃ c : ℤ, ∀ x : ℤ, f x = c)) ∧
  ((∀ x : ℤ, f (-x) = -f x) → (∀ x : ℤ, f x = 0)) := by
  sorry

end function_property_implications_l2970_297032


namespace fraction_equivalence_l2970_297042

theorem fraction_equivalence : (15 : ℝ) / (4 * 63) = 1.5 / (0.4 * 63) := by
  sorry

end fraction_equivalence_l2970_297042


namespace last_two_digits_product_l2970_297085

/-- Given an integer n, returns the tens digit of n. -/
def tens_digit (n : ℤ) : ℤ := (n / 10) % 10

/-- Given an integer n, returns the units digit of n. -/
def units_digit (n : ℤ) : ℤ := n % 10

/-- Theorem: For any integer n divisible by 6 with the sum of its last two digits being 15,
    the product of its last two digits is either 56 or 54. -/
theorem last_two_digits_product (n : ℤ) 
  (div_by_6 : n % 6 = 0)
  (sum_15 : tens_digit n + units_digit n = 15) :
  tens_digit n * units_digit n = 56 ∨ tens_digit n * units_digit n = 54 := by
  sorry

end last_two_digits_product_l2970_297085


namespace function_property_l2970_297091

/-- Iterated function application -/
def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (iterate f n x)

/-- The main theorem -/
theorem function_property (f : ℕ → ℕ) 
  (h : ∀ x y, iterate f (x + 1) y + iterate f (y + 1) x = 2 * f (x + y)) :
  ∀ n, f (f n) = f (n + 1) := by
  sorry

end function_property_l2970_297091


namespace tuesday_attendance_theorem_l2970_297026

/-- Represents the attendance status of students at Dunkley S.S. over two days -/
structure AttendanceData where
  total_students : ℕ
  monday_absent_rate : ℚ
  tuesday_return_rate : ℚ
  tuesday_absent_rate : ℚ

/-- Calculates the percentage of students present on Tuesday -/
def tuesday_present_percentage (data : AttendanceData) : ℚ :=
  let monday_present := 1 - data.monday_absent_rate
  let tuesday_present_from_monday := monday_present * (1 - data.tuesday_absent_rate)
  let tuesday_present_from_absent := data.monday_absent_rate * data.tuesday_return_rate
  (tuesday_present_from_monday + tuesday_present_from_absent) * 100

/-- Theorem stating that given the conditions, the percentage of students present on Tuesday is 82% -/
theorem tuesday_attendance_theorem (data : AttendanceData) 
  (h1 : data.total_students > 0)
  (h2 : data.monday_absent_rate = 1/10)
  (h3 : data.tuesday_return_rate = 1/10)
  (h4 : data.tuesday_absent_rate = 1/10) :
  tuesday_present_percentage data = 82 := by
  sorry


end tuesday_attendance_theorem_l2970_297026


namespace duck_snail_problem_l2970_297035

theorem duck_snail_problem :
  let total_ducklings : ℕ := 8
  let first_group_size : ℕ := 3
  let second_group_size : ℕ := 3
  let first_group_snails_per_duckling : ℕ := 5
  let second_group_snails_per_duckling : ℕ := 9
  let total_snails : ℕ := 294

  let first_group_snails := first_group_size * first_group_snails_per_duckling
  let second_group_snails := second_group_size * second_group_snails_per_duckling
  let first_two_groups_snails := first_group_snails + second_group_snails

  let remaining_ducklings := total_ducklings - first_group_size - second_group_size
  let mother_duck_snails := (total_snails - first_two_groups_snails) / 2

  mother_duck_snails = 3 * first_two_groups_snails := by
    sorry

end duck_snail_problem_l2970_297035


namespace logical_equivalence_l2970_297098

theorem logical_equivalence (P Q R : Prop) :
  ((P ∧ ¬R) → ¬Q) ↔ (Q → (¬P ∨ R)) := by sorry

end logical_equivalence_l2970_297098


namespace john_money_left_l2970_297071

/-- The amount of money John has left after shopping --/
def money_left (initial_amount : ℝ) (roast_cost : ℝ) (vegetable_cost : ℝ) (wine_cost : ℝ) (dessert_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  let total_cost := roast_cost + vegetable_cost + wine_cost + dessert_cost
  let discounted_cost := total_cost * (1 - discount_rate)
  initial_amount - discounted_cost

/-- Theorem stating that John has €56.8 left after shopping --/
theorem john_money_left :
  money_left 100 17 11 12 8 0.1 = 56.8 := by
  sorry

end john_money_left_l2970_297071


namespace imaginary_part_of_complex_expression_l2970_297073

theorem imaginary_part_of_complex_expression : 
  Complex.im ((2 * Complex.I) / (1 - Complex.I) + 2) = 1 := by
  sorry

end imaginary_part_of_complex_expression_l2970_297073


namespace number_wall_top_l2970_297068

/-- Represents a number wall with 5 base numbers -/
structure NumberWall (a b c d e : ℕ) where
  level1 : Fin 5 → ℕ
  level2 : Fin 4 → ℕ
  level3 : Fin 3 → ℕ
  level4 : Fin 2 → ℕ
  top : ℕ
  base_correct : level1 = ![a, b, c, d, e]
  level2_correct : ∀ i : Fin 4, level2 i = level1 i + level1 (i.succ)
  level3_correct : ∀ i : Fin 3, level3 i = level2 i + level2 (i.succ)
  level4_correct : ∀ i : Fin 2, level4 i = level3 i + level3 (i.succ)
  top_correct : top = level4 0 + level4 1

/-- The theorem stating that the top of the number wall is x + 103 -/
theorem number_wall_top (x : ℕ) : 
  ∀ (w : NumberWall x 4 8 7 11), w.top = x + 103 := by
  sorry

end number_wall_top_l2970_297068


namespace probability_shaded_isosceles_triangle_l2970_297031

/-- Represents a game board shaped like an isosceles triangle -/
structure GameBoard where
  regions : ℕ
  shaded_regions : ℕ

/-- Calculates the probability of landing in a shaded region -/
def probability_shaded (board : GameBoard) : ℚ :=
  board.shaded_regions / board.regions

theorem probability_shaded_isosceles_triangle :
  ∀ (board : GameBoard),
    board.regions = 7 →
    board.shaded_regions = 3 →
    probability_shaded board = 3 / 7 := by
  sorry

#eval probability_shaded { regions := 7, shaded_regions := 3 }

end probability_shaded_isosceles_triangle_l2970_297031


namespace min_overs_for_max_wickets_l2970_297023

/-- Represents the maximum number of wickets a bowler can take in one over -/
def max_wickets_per_over : ℕ := 3

/-- Represents the maximum number of wickets a bowler can take in the innings -/
def max_wickets_in_innings : ℕ := 10

/-- Theorem stating the minimum number of overs required to potentially take the maximum wickets -/
theorem min_overs_for_max_wickets :
  ∃ (overs : ℕ), overs * max_wickets_per_over ≥ max_wickets_in_innings ∧
  ∀ (n : ℕ), n < overs → n * max_wickets_per_over < max_wickets_in_innings :=
by sorry

end min_overs_for_max_wickets_l2970_297023


namespace gasoline_price_increase_l2970_297076

theorem gasoline_price_increase (original_price original_quantity : ℝ) 
  (h1 : original_price > 0) (h2 : original_quantity > 0) : 
  let new_quantity := original_quantity * (1 - 0.1)
  let new_total_cost := original_price * original_quantity * (1 + 0.08)
  let price_increase_factor := new_total_cost / (new_quantity * original_price)
  price_increase_factor = 1.2 := by sorry

end gasoline_price_increase_l2970_297076


namespace isosceles_triangle_l2970_297044

theorem isosceles_triangle (A B C : ℝ) (h1 : 0 < A ∧ A < π)
    (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π)
    (h5 : 2 * Real.cos B * Real.sin A = Real.sin C) : A = B := by
  sorry

end isosceles_triangle_l2970_297044


namespace lucy_max_notebooks_l2970_297011

/-- The amount of money Lucy has in cents -/
def lucys_money : ℕ := 2550

/-- The cost of each notebook in cents -/
def notebook_cost : ℕ := 240

/-- The maximum number of notebooks Lucy can buy -/
def max_notebooks : ℕ := lucys_money / notebook_cost

theorem lucy_max_notebooks :
  max_notebooks = 10 ∧
  max_notebooks * notebook_cost ≤ lucys_money ∧
  (max_notebooks + 1) * notebook_cost > lucys_money :=
by sorry

end lucy_max_notebooks_l2970_297011


namespace max_diff_divisible_sum_digits_l2970_297045

def sum_of_digits (n : ℕ) : ℕ := sorry

def has_divisible_sum_between (a b : ℕ) : Prop :=
  ∃ k, a < k ∧ k < b ∧ sum_of_digits k % 7 = 0

theorem max_diff_divisible_sum_digits :
  ∃ a b : ℕ, sum_of_digits a % 7 = 0 ∧
             sum_of_digits b % 7 = 0 ∧
             b - a = 13 ∧
             ¬ has_divisible_sum_between a b ∧
             ∀ c d : ℕ, sum_of_digits c % 7 = 0 →
                        sum_of_digits d % 7 = 0 →
                        ¬ has_divisible_sum_between c d →
                        d - c ≤ 13 := by sorry

end max_diff_divisible_sum_digits_l2970_297045


namespace polygon_sides_count_l2970_297087

theorem polygon_sides_count (n : ℕ) (exterior_angle : ℝ) : 
  (n ≥ 2) →
  (exterior_angle > 0) →
  (exterior_angle < 45) →
  (n * exterior_angle = 360) →
  (n ≥ 9) :=
sorry

end polygon_sides_count_l2970_297087


namespace rectangle_area_perimeter_product_l2970_297072

/-- Represents a point on a 2D grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a rectangle on a 2D grid --/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomRight : Point
  bottomLeft : Point

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℕ :=
  (r.topRight.x - r.topLeft.x) * (r.topLeft.y - r.bottomLeft.y)

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℕ :=
  2 * ((r.topRight.x - r.topLeft.x) + (r.topLeft.y - r.bottomLeft.y))

/-- The main theorem to prove --/
theorem rectangle_area_perimeter_product :
  let r := Rectangle.mk
    (Point.mk 1 5) (Point.mk 5 5)
    (Point.mk 5 2) (Point.mk 1 2)
  area r * perimeter r = 168 := by
  sorry

end rectangle_area_perimeter_product_l2970_297072


namespace circle_center_l2970_297079

/-- The center of a circle with diameter endpoints (3, 3) and (9, -3) is (6, 0) -/
theorem circle_center (K : Set (ℝ × ℝ)) (p₁ p₂ : ℝ × ℝ) : 
  p₁ = (3, 3) → p₂ = (9, -3) → 
  (∀ x ∈ K, ∃ y ∈ K, (x.1 - y.1)^2 + (x.2 - y.2)^2 = (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) →
  (∃ c : ℝ × ℝ, c = (6, 0) ∧ ∀ x ∈ K, (x.1 - c.1)^2 + (x.2 - c.2)^2 = ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) / 4) :=
by sorry

end circle_center_l2970_297079


namespace line_perp_plane_contained_in_plane_implies_planes_perp_l2970_297043

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_contained_in_plane_implies_planes_perp
  (a : Line) (M N : Plane) :
  perpendicular a M → contained_in a N → planes_perpendicular M N :=
sorry

end line_perp_plane_contained_in_plane_implies_planes_perp_l2970_297043


namespace courageous_iff_coprime_l2970_297064

/-- A function is courageous if it and its 100 shifts are all bijections -/
def IsCourageous (n : ℕ) (g : ZMod n → ZMod n) : Prop :=
  Function.Bijective g ∧
  ∀ k : Fin 101, Function.Bijective (λ x => g x + k * x)

/-- The main theorem: existence of a courageous function is equivalent to n being coprime to 101! -/
theorem courageous_iff_coprime (n : ℕ) :
  (∃ g : ZMod n → ZMod n, IsCourageous n g) ↔ Nat.Coprime n (Nat.factorial 101) := by
  sorry

end courageous_iff_coprime_l2970_297064


namespace sum_seven_consecutive_integers_l2970_297037

theorem sum_seven_consecutive_integers (n : ℤ) :
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 7 * n + 7 := by
  sorry

end sum_seven_consecutive_integers_l2970_297037


namespace ship_typhoon_probability_l2970_297057

/-- The probability of a ship being affected by a typhoon -/
theorem ship_typhoon_probability 
  (OA OB : ℝ) 
  (h_OA : OA = 100) 
  (h_OB : OB = 100) 
  (r_min r_max : ℝ) 
  (h_r_min : r_min = 50) 
  (h_r_max : r_max = 100) : 
  ∃ (P : ℝ), P = 1 - Real.sqrt 2 / 2 ∧ 
  P = (r_max - Real.sqrt (OA^2 + OB^2) / 2) / (r_max - r_min) := by
  sorry

#check ship_typhoon_probability

end ship_typhoon_probability_l2970_297057


namespace units_digit_of_sum_of_powers_divided_by_five_l2970_297016

theorem units_digit_of_sum_of_powers_divided_by_five :
  ∃ n : ℕ, (2^2023 + 3^2023) / 5 ≡ n [ZMOD 10] ∧ n = 1 := by
  sorry

end units_digit_of_sum_of_powers_divided_by_five_l2970_297016


namespace nathaniel_best_friends_l2970_297077

theorem nathaniel_best_friends (total_tickets : ℕ) (tickets_per_friend : ℕ) (tickets_left : ℕ) :
  total_tickets = 11 →
  tickets_per_friend = 2 →
  tickets_left = 3 →
  (total_tickets - tickets_left) / tickets_per_friend = 4 :=
by
  sorry

end nathaniel_best_friends_l2970_297077


namespace inequality_solution_set_l2970_297053

theorem inequality_solution_set (x : ℝ) : 
  (Set.Ioo (-3 : ℝ) 1 : Set ℝ) = {x | (1 - x) * (3 + x) > 0} := by
sorry

end inequality_solution_set_l2970_297053


namespace smallest_gcd_qr_l2970_297007

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) :
  ∃ (q' r' : ℕ+), Nat.gcd q'.val r'.val = 10 ∧
    ∀ (q'' r'' : ℕ+), Nat.gcd q''.val r''.val < 10 →
      ¬(Nat.gcd p q''.val = 210 ∧ Nat.gcd p r''.val = 770) :=
by sorry

end smallest_gcd_qr_l2970_297007


namespace right_triangle_median_geometric_mean_l2970_297002

theorem right_triangle_median_geometric_mean (c : ℝ) (h : c > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    c^2 = a^2 + b^2 ∧
    (c / 2)^2 = a * b ∧
    a + b = c * Real.sqrt (3 / 2) :=
by
  sorry

end right_triangle_median_geometric_mean_l2970_297002


namespace triangular_projections_imply_triangular_pyramid_l2970_297025

/-- Represents the shape of a projection in an orthographic view -/
inductive Projection
  | Triangular
  | Circular
  | Rectangular
  | Trapezoidal

/-- Represents a geometric solid -/
inductive GeometricSolid
  | Cone
  | TriangularPyramid
  | TriangularPrism
  | FrustumOfPyramid

/-- Represents the orthographic views of a solid -/
structure OrthographicViews where
  front : Projection
  top : Projection
  side : Projection

/-- Determines if a set of orthographic views corresponds to a triangular pyramid -/
def isTriangularPyramid (views : OrthographicViews) : Prop :=
  views.front = Projection.Triangular ∧
  views.top = Projection.Triangular ∧
  views.side = Projection.Triangular

theorem triangular_projections_imply_triangular_pyramid (views : OrthographicViews) :
  isTriangularPyramid views → GeometricSolid.TriangularPyramid = 
    match views with
    | ⟨Projection.Triangular, Projection.Triangular, Projection.Triangular⟩ => GeometricSolid.TriangularPyramid
    | _ => GeometricSolid.Cone  -- This is just a placeholder for other cases
    :=
  sorry

end triangular_projections_imply_triangular_pyramid_l2970_297025


namespace count_numbers_with_three_ones_l2970_297083

/-- Recursive function to count numbers without three consecutive 1's -/
def count_without_three_ones (n : ℕ) : ℕ :=
  if n ≤ 3 then
    match n with
    | 1 => 2
    | 2 => 4
    | 3 => 7
    | _ => 0
  else
    count_without_three_ones (n - 1) + count_without_three_ones (n - 2) + count_without_three_ones (n - 3)

/-- Theorem stating the count of 12-digit numbers with three consecutive 1's -/
theorem count_numbers_with_three_ones : 
  (2^12 : ℕ) - count_without_three_ones 12 = 3592 :=
sorry

end count_numbers_with_three_ones_l2970_297083


namespace milk_consumption_l2970_297030

theorem milk_consumption (total_monitors : ℕ) (monitors_per_group : ℕ) (students_per_group : ℕ)
  (girl_percentage : ℚ) (boy_milk : ℕ) (girl_milk : ℕ) :
  total_monitors = 8 →
  monitors_per_group = 2 →
  students_per_group = 15 →
  girl_percentage = 2/5 →
  boy_milk = 1 →
  girl_milk = 2 →
  (total_monitors / monitors_per_group * students_per_group *
    ((1 - girl_percentage) * boy_milk + girl_percentage * girl_milk) : ℚ) = 84 :=
by sorry

end milk_consumption_l2970_297030


namespace consecutive_product_not_perfect_power_l2970_297001

theorem consecutive_product_not_perfect_power :
  ∀ x y : ℤ, ∀ n : ℕ, n > 1 → x * (x + 1) ≠ y^n := by
  sorry

end consecutive_product_not_perfect_power_l2970_297001


namespace melanie_breadcrumbs_count_l2970_297065

/-- Represents the number of pieces a bread slice is divided into -/
structure BreadDivision where
  firstHalf : Nat
  secondHalf : Nat

/-- Calculates the total number of pieces for a bread slice -/
def totalPieces (division : BreadDivision) : Nat :=
  division.firstHalf + division.secondHalf

/-- Represents Melanie's bread slicing method -/
def melanieBreadSlicing : List BreadDivision :=
  [{ firstHalf := 3, secondHalf := 4 },  -- First slice
   { firstHalf := 2, secondHalf := 10 }] -- Second slice

/-- Theorem: Melanie's bread slicing method results in 19 total pieces -/
theorem melanie_breadcrumbs_count :
  (melanieBreadSlicing.map totalPieces).sum = 19 := by
  sorry

#eval (melanieBreadSlicing.map totalPieces).sum

end melanie_breadcrumbs_count_l2970_297065


namespace base_7_conversion_correct_l2970_297027

/-- Converts a list of digits in base 7 to its decimal (base 10) representation -/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The decimal number we want to convert -/
def decimalNumber : Nat := 1987

/-- The proposed base 7 representation -/
def base7Digits : List Nat := [6, 3, 5, 3, 5]

/-- Theorem stating that the conversion is correct -/
theorem base_7_conversion_correct :
  toDecimal base7Digits = decimalNumber := by sorry

end base_7_conversion_correct_l2970_297027


namespace range_of_m_l2970_297008

def has_two_distinct_negative_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

def satisfies_conditions (m : ℝ) : Prop :=
  (has_two_distinct_negative_roots m ∨ has_no_real_roots m) ∧
  ¬(has_two_distinct_negative_roots m ∧ has_no_real_roots m)

theorem range_of_m : 
  {m : ℝ | satisfies_conditions m} = {m : ℝ | 1 < m ∧ m ≤ 2 ∨ 3 ≤ m} :=
sorry

end range_of_m_l2970_297008


namespace weight_replacement_l2970_297099

theorem weight_replacement (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 6 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 80 :=
by
  sorry

end weight_replacement_l2970_297099


namespace inequality_solution_set_l2970_297062

theorem inequality_solution_set (x : ℝ) : 
  (x ≠ 0 ∧ (x - 1) / x ≤ 0) ↔ (0 < x ∧ x ≤ 1) := by sorry

end inequality_solution_set_l2970_297062


namespace y_one_gt_y_two_l2970_297051

/-- Two points on a line with negative slope -/
structure PointsOnLine where
  y₁ : ℝ
  y₂ : ℝ
  h₁ : y₁ = -1/2 * (-5)
  h₂ : y₂ = -1/2 * (-2)

/-- Theorem: For two points A(-5, y₁) and B(-2, y₂) on the line y = -1/2x, y₁ > y₂ -/
theorem y_one_gt_y_two (p : PointsOnLine) : p.y₁ > p.y₂ := by
  sorry

end y_one_gt_y_two_l2970_297051


namespace different_color_probability_l2970_297013

theorem different_color_probability (total : ℕ) (white : ℕ) (black : ℕ) (drawn : ℕ) :
  total = white + black →
  total = 5 →
  white = 3 →
  black = 2 →
  drawn = 2 →
  (Nat.choose white 1 * Nat.choose black 1 : ℚ) / Nat.choose total drawn = 3 / 5 := by
  sorry

end different_color_probability_l2970_297013


namespace smallest_radius_is_one_l2970_297069

/-- Triangle ABC with a circle inscribed on side AB -/
structure TriangleWithInscribedCircle where
  /-- Length of side AC -/
  ac : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The circle's center is on side AB -/
  center_on_ab : Bool
  /-- The circle is tangent to sides AC and BC -/
  tangent_to_ac_bc : Bool

/-- The smallest positive integer radius for the given triangle configuration -/
def smallest_integer_radius (t : TriangleWithInscribedCircle) : ℕ :=
  sorry

/-- Theorem stating that the smallest positive integer radius is 1 -/
theorem smallest_radius_is_one :
  ∀ t : TriangleWithInscribedCircle,
    t.ac = 5 ∧ t.bc = 3 ∧ t.center_on_ab ∧ t.tangent_to_ac_bc →
    smallest_integer_radius t = 1 :=
  sorry

end smallest_radius_is_one_l2970_297069


namespace prime_square_minus_one_remainder_l2970_297049

theorem prime_square_minus_one_remainder (p : ℕ) (hp : Nat.Prime p) :
  ∃ r ∈ ({0, 3, 8} : Set ℕ), (p^2 - 1) % 12 = r :=
sorry

end prime_square_minus_one_remainder_l2970_297049


namespace shifted_parabola_equation_l2970_297040

/-- Represents a vertical shift of a parabola -/
def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := 
  λ x => f x + shift

/-- The original parabola function -/
def original_parabola : ℝ → ℝ := 
  λ x => x^2 - 1

/-- The shifted parabola G -/
def G : ℝ → ℝ := vertical_shift original_parabola 3

/-- Theorem stating that G is equivalent to x^2 + 2 -/
theorem shifted_parabola_equation : G = λ x => x^2 + 2 := by
  sorry

end shifted_parabola_equation_l2970_297040


namespace jane_calculation_l2970_297094

structure CalculationData where
  a : ℝ
  b : ℝ
  c : ℝ
  incorrect_result : ℝ
  correct_result : ℝ

theorem jane_calculation (data : CalculationData) 
  (h1 : data.a * (data.b / data.c) = data.incorrect_result)
  (h2 : (data.a * data.b) / data.c = data.correct_result)
  (h3 : data.incorrect_result = 12)
  (h4 : data.correct_result = 4)
  (h5 : data.c ≠ 0) :
  data.a * data.b = 4 * data.c ∨ data.a * data.b = 12 * data.c :=
by sorry

end jane_calculation_l2970_297094


namespace janes_earnings_is_75_l2970_297012

/-- The amount of money Jane earned for planting flower bulbs -/
def janes_earnings : ℚ :=
  let price_per_bulb : ℚ := 1/2
  let tulip_bulbs : ℕ := 20
  let iris_bulbs : ℕ := tulip_bulbs / 2
  let daffodil_bulbs : ℕ := 30
  let crocus_bulbs : ℕ := daffodil_bulbs * 3
  let total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs
  (total_bulbs : ℚ) * price_per_bulb

/-- Theorem stating that Jane earned $75 for planting flower bulbs -/
theorem janes_earnings_is_75 : janes_earnings = 75 := by
  sorry

end janes_earnings_is_75_l2970_297012


namespace monthly_salary_calculation_l2970_297038

def monthly_salary (rent : ℚ) (savings : ℚ) : ℚ :=
  let food := (5 : ℚ) / 9 * rent
  let mortgage := 5 * food
  let utilities := (1 : ℚ) / 5 * mortgage
  let transportation := (1 : ℚ) / 3 * food
  let insurance := (2 : ℚ) / 3 * utilities
  let healthcare := (3 : ℚ) / 8 * food
  let car_maintenance := (1 : ℚ) / 4 * transportation
  let taxes := (4 : ℚ) / 9 * savings
  rent + food + mortgage + utilities + transportation + insurance + healthcare + car_maintenance + savings + taxes

theorem monthly_salary_calculation (rent savings : ℚ) :
  monthly_salary rent savings = rent + (5 : ℚ) / 9 * rent + 5 * ((5 : ℚ) / 9 * rent) +
    (1 : ℚ) / 5 * (5 * ((5 : ℚ) / 9 * rent)) + (1 : ℚ) / 3 * ((5 : ℚ) / 9 * rent) +
    (2 : ℚ) / 3 * ((1 : ℚ) / 5 * (5 * ((5 : ℚ) / 9 * rent))) +
    (3 : ℚ) / 8 * ((5 : ℚ) / 9 * rent) +
    (1 : ℚ) / 4 * ((1 : ℚ) / 3 * ((5 : ℚ) / 9 * rent)) +
    savings + (4 : ℚ) / 9 * savings :=
by sorry

example : monthly_salary 850 2200 = 8022 + (98 : ℚ) / 100 :=
by sorry

end monthly_salary_calculation_l2970_297038


namespace infinitely_many_primes_dividing_n_squared_plus_n_plus_one_l2970_297097

theorem infinitely_many_primes_dividing_n_squared_plus_n_plus_one :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ n^2 + n + 1} := by
  sorry

end infinitely_many_primes_dividing_n_squared_plus_n_plus_one_l2970_297097


namespace inequality_proof_l2970_297009

theorem inequality_proof (a b c : ℝ) (h : a * b + b * c + c * a = 1) :
  |a - b| / |1 + c^2| + |b - c| / |1 + a^2| ≥ |c - a| / |1 + b^2| := by
  sorry

end inequality_proof_l2970_297009
